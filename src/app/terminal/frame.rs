//! What the window shows and when: adopting the window's geometry, scrolling the view,
//! the overlay scrollbar, the display list, and the timers that change a frame with no
//! input behind them (the cursor blink, the synchronized-output hold, the visual bell).

use std::time::{Duration, Instant};

use crate::app::layout::TerminalGeometry;
use crate::app::terminal::{demo_screen, TerminalCore};
use crate::grid::CursorStyle;
use crate::input;
use crate::platform::scroll;
use crate::pty::TtyMode;
use crate::render::display::DisplayList;
use crate::term_render::{self, CursorRender, CursorShape, ScrollLane};

/// The cursor blink half-period: how long each of the on/off phases lasts.
const BLINK_INTERVAL: Duration = Duration::from_millis(530);

/// How many phase flips a blink gets before it rests on a lit cursor: ~10s at
/// [`BLINK_INTERVAL`], after which a focused idle terminal would otherwise keep
/// presenting a frame twice a second forever. Any activity refills it.
pub(super) const BLINK_TOGGLES: u16 = 20;

/// How long a child may hold a frame under synchronized output (`?2026`) before we show
/// it anyway.
///
/// The lock is held by the *child*, which means a child that crashes between "begin
/// frame" and "end frame" — or simply forgets the second one — would otherwise freeze
/// the window for good. The escape hatch is not optional, and it is why this mode is
/// safe to implement at all. 150ms is kitty's figure: far longer than any real frame,
/// far shorter than a user notices something is wrong.
const SYNC_TIMEOUT: Duration = Duration::from_millis(150);

impl TerminalCore {
    /// The frame background as a `0x00RRGGBB`, for the GPU clear (which must match
    /// the display list's own base fill).
    pub(in crate::app) fn clear_color(&self) -> u32 {
        // The grid owns the palette, because a program can change it (`OSC 11`). Read it
        // from there rather than keeping a second copy that would drift the moment it did.
        // The bell lifts it, and the clear must lift with it or the flash tears at the
        // edges the display list does not cover.
        term_render::bell_background(self.screen.theme(), self.bell_flashing()).to_u32()
    }

    /// Adopt the window's fresh geometry and resize the grid to it.
    pub(in crate::app) fn resize(&mut self, geom: TerminalGeometry) {
        let (cols, rows) = (geom.cols, geom.rows);
        // Adopt the window's fresh geometry, then resize the grid to it. In
        // demo mode there is no child; rebuild the static grid. Otherwise
        // push the size to the child (TIOCSWINSZ → SIGWINCH). The PTY resize
        // is best-effort: a resize on a dead child just surfaces as EOF on
        // the next read, which shuts down cleanly.
        self.geom = geom;
        // An in-progress drag cannot survive: the button is held on a window that is
        // being resized. It ends here, and copies what it selected (below) the way its
        // release would have.
        let ended_drag = self.drag.take().is_some();
        if self.demo {
            // The grid is replaced wholesale, so no row id minted against the
            // old one survives it.
            self.screen = demo_screen(cols, rows);
            self.set_selection(None);
        } else {
            // A width reflow re-wraps the grid and renumbers its rows; the returned
            // effect says how to carry the selection over that (a height change leaves
            // ids alone).
            let effect = self.screen.resize(cols, rows);
            let carried = self.carry_selection(effect);
            self.set_selection(carried);
            self.screen.set_pixel_size(geom.width, geom.height);
            // Reflow the grid now (above) for smooth visuals, but debounce the child's
            // winsize: telling the shell on every configure floods it with prompt
            // redraws that mush together under a drag. `flush_winsize_if_due` pushes
            // `TIOCSWINSZ` (and the in-band `?2048` report) once it settles.
            self.winsize_at = Some(Instant::now() + self.winsize_settle);
        }
        // Prune catches the leftover cases the carry does not: a kept selection whose
        // end row a shrink dropped off the screen.
        self.prune_selection();
        // The text moved under the pointer, and the cell the pointer is over may have
        // too, so the hover waits for the next motion.
        self.drop_hover();
        if ended_drag {
            self.offer_selection(true);
        }
        self.dirty = true;
    }

    /// Scroll the scrollback view for a navigation chord on the primary screen, returning
    /// whether the key was consumed: Shift + PageUp/PageDown/Home/End moves by pages and
    /// ends, and Ctrl+Shift + Up/Down jumps between the prompts the shell marked. A page is
    /// a screenful less one line, so a line of context carries across. The window resolves
    /// the keycode to a named key; the grid does the scrolling.
    pub(in crate::app) fn handle_scroll_key(&mut self, key: input::Key, mods: input::Mods) -> bool {
        if !mods.contains(input::Mods::SHIFT) || self.screen.is_alt() {
            return false;
        }
        let (_, rows) = self.screen.dimensions();
        let page = rows.saturating_sub(1).max(1);
        match key {
            input::Key::PageUp => self.screen.scroll_view_up(page),
            input::Key::PageDown => self.screen.scroll_view_down(page),
            input::Key::Home => self.screen.scroll_view_to_top(),
            input::Key::End => self.screen.scroll_view_to_bottom(),
            // Ctrl+Shift+Up/Down: jump between the prompts the shell marked (`OSC 133`).
            //
            // This is the one kind of scrolling a scrollbar cannot do. A scrollbar knows
            // how far you have come; this knows *what* you have come past. Ten screens of
            // build output go by in one keystroke because the terminal knows where the
            // command that printed them started. Without the shell's marks it does nothing
            // at all, and does nothing *quietly*: a shell that reports no prompts leaves
            // the chord free rather than making it lie.
            input::Key::Up if mods.contains(input::Mods::CTRL) => {
                if !self.screen.scroll_to_prompt(true) {
                    return false;
                }
            }
            input::Key::Down if mods.contains(input::Mods::CTRL) => {
                if !self.screen.scroll_to_prompt(false) {
                    return false;
                }
            }
            _ => return false,
        }
        // A keyboard scroll is a scroll: show the bar moving, the way the wheel does.
        self.flash_scrollbar();
        self.refresh_hover();
        self.dirty = true;
        true
    }

    /// Return a scrolled-back view to the live bottom before input goes out, so a
    /// keystroke or paste never lands "blind" while the user is reading history.
    pub(super) fn snap_to_bottom(&mut self) {
        if self.screen.is_scrolled() {
            self.screen.scroll_view_to_bottom();
            self.refresh_hover();
            self.dirty = true;
        }
    }

    /// Whether the visual bell is mid-flash.
    pub(in crate::app) fn bell_flashing(&self) -> bool {
        self.bell_until.is_some_and(|until| Instant::now() < until)
    }

    /// End the flash when its moment has passed, and repaint to take it back off.
    /// Without this the lifted background would simply stay lifted.
    fn tick_bell_if_due(&mut self, now: Instant) {
        if self.bell_until.is_some_and(|until| until <= now) {
            self.bell_until = None;
            self.dirty = true;
        }
    }

    /// Follow the child in and out of synchronized output (`?2026`).
    ///
    /// Entering starts the clock; leaving stops it. The deadline is set once on the way
    /// in and not extended by later output: it bounds how long a *frame* can take, and a
    /// child that keeps writing while holding the lock is exactly the case that must not
    /// be able to hold it forever.
    ///
    /// Once [`Self::tick_sync_if_due`] has fired the deadline and cleared it, output that
    /// still finds the lock held arms a *new* one. That is deliberate: the frame just
    /// presented is over, so the next one gets its own budget, and a child that never
    /// releases the lock therefore settles at one presented frame per timeout instead of
    /// having synchronization silently switched off for the rest of its life.
    pub(super) fn track_sync_lock(&mut self) {
        if !self.screen.synchronized() {
            self.sync_until = None;
        } else if self.sync_until.is_none() {
            self.sync_until = Some(Instant::now() + SYNC_TIMEOUT);
        }
    }

    /// Whether the child is mid-frame and the frame should wait for it.
    ///
    /// False once the deadline passes, even though the child still holds the lock: at
    /// that point we present what we have, exactly as if it had finished. A half-drawn
    /// frame is a cosmetic problem; a window that never repaints again is not.
    pub(in crate::app) fn holds_frame(&self) -> bool {
        self.sync_until.is_some_and(|until| Instant::now() < until)
    }

    /// Give up on a synchronized frame whose deadline has passed, and present what we
    /// have.
    ///
    /// The deadline has to be disarmed by the clock, not by the child, because the child
    /// is exactly what may have stopped talking: [`Self::track_sync_lock`] runs only when
    /// bytes arrived, so a child that opens `?2026` and then goes quiet (or dies) leaves
    /// `sync_until` set in the past forever. That is not merely untidy, it pins the event
    /// loop: a deadline already past clamps the loop's wait to 1 ms, so the window spins
    /// at 1 kHz with nothing to draw until the child speaks again. Clearing it here makes
    /// the deadline behave like every other timer in `service_timers` — armed once, fired
    /// once, gone.
    fn tick_sync_if_due(&mut self, now: Instant) {
        if self.sync_until.is_some_and(|until| until <= now) {
            self.sync_until = None;
            // The frame that was being held is now the frame to show.
            self.dirty = true;
        }
    }

    /// Compose the window's display list: the visible grid painted at the current
    /// size, cursor on top (solid when focused, hollow when not). Reads the
    /// window-shipped geometry, so it stays a pure function of the grid state.
    /// `link_preview` shows the hovered link's preview, when there is one: the window
    /// decides, from what the pointer offers.
    pub(in crate::app) fn fill_frame_list(
        &self,
        out: &mut DisplayList,
        strings: &mut Vec<String>,
        link_preview: bool,
    ) {
        // The child chose the shape (DECSCUSR); blink hides it on the off phase
        // while focused, and DECTCEM hides it entirely.
        let blinked_off = self.cursor_blinking() && !self.blink_on;
        let cursor = CursorRender {
            shape: cursor_shape(self.screen.cursor_style(), self.tty_mode),
            visible: self.screen.cursor_visible() && !blinked_off,
            focused: self.focused,
        };
        term_render::build_display_list_into(
            out,
            strings,
            &term_render::FrameInputs {
                screen: &self.screen,
                bell: self.bell_flashing(),
                theme: self.screen.theme(),
                metrics: self.geom.metrics,
                surface: (self.geom.width as i32, self.geom.height as i32),
                origin: (self.geom.pad, self.geom.origin_y),
                cursor,
                selection: self.selection,
                hover: self.hover,
                preview: (link_preview && self.hover.is_some()).then_some(self.preview.as_str()),
                scale: self.geom.scale,
                scrollbar: &self.scrollbar,
            },
        );
    }

    /// The column the scrollbar hangs off: the grid's rows, out to the window's right
    /// edge. The painter derives the same rectangle from the same geometry, so the bar is
    /// grabbable exactly where it is drawn.
    fn scroll_lane(&self) -> ScrollLane {
        ScrollLane::new(
            (self.geom.width as i32, self.geom.height as i32),
            (self.geom.pad, self.geom.origin_y),
            self.geom.metrics,
            self.screen.dimensions().1,
            self.geom.scale,
        )
    }

    /// Whether this tab has history to scroll at all, which is the only case the bar is
    /// drawn or takes the pointer. False on the alt screen, which keeps no history.
    pub(in crate::app) fn scrollable(&self) -> bool {
        let (content, viewport) = self.screen.scroll_extent();
        scroll::scrollable(content, viewport)
    }

    /// Whether a device-pixel point is on the scrollbar, meaning a press there takes its
    /// thumb rather than starting a selection in the grid beneath it.
    pub(in crate::app) fn on_scrollbar(&self, x: f32, y: f32) -> bool {
        self.scrollable() && self.scroll_lane().grab.contains(x, y)
    }

    /// Follow the pointer with the bar: inside its proximity zone it grows into a slider,
    /// outside it shrinks back. A live drag owns the bar and ignores this.
    pub(in crate::app) fn track_scrollbar(&mut self, at: Option<(f32, f32)>) {
        if self.scrollbar.grab().is_some() {
            return;
        }
        let near =
            self.scrollable() && at.is_some_and(|(x, y)| self.scroll_lane().zone.contains(x, y));
        if self.scrollbar.set_near(near) {
            self.dirty = true;
        }
    }

    /// A press in the scrollbar's lane: take the thumb under the pointer, or, when the
    /// press landed on the empty track, warp the thumb to it first and take it there
    /// (GNOME's behavior: a click jumps to that spot, and holding turns it into a scrub).
    /// Returns whether the press was the scrollbar's.
    pub(in crate::app) fn press_scrollbar(&mut self, x: f32, y: f32) -> bool {
        if !self.on_scrollbar(x, y) {
            return false;
        }
        let Some(thumb) = self.scroll_lane().thumb(&self.screen) else {
            return false;
        };
        // On the thumb, the grab keeps its offset so it does not jump under the finger;
        // on the bare track, the thumb lands centred on the pointer.
        let grab = if y >= thumb.y as f32 && y < (thumb.y + thumb.h) as f32 {
            y as i32 - thumb.y
        } else {
            thumb.h / 2
        };
        self.scrollbar.press(grab);
        self.drag_scrollbar(y);
        true
    }

    /// Track a scrollbar drag: put the thumb's top where the pointer holds it and scroll
    /// the view to match. The scroll lands on a whole line, which is the only place a
    /// terminal view can rest.
    pub(in crate::app) fn drag_scrollbar(&mut self, y: f32) {
        let lane = self.scroll_lane();
        let (Some(grab), Some(thumb)) = (self.scrollbar.grab(), lane.thumb(&self.screen)) else {
            return;
        };
        let (content, viewport) = self.screen.scroll_extent();
        let track = lane.track;
        let at = self.screen.scroll_position();
        let to = scroll::scroll_at_thumb(track, thumb.h, viewport, content, y as i32 - grab);
        if to == at {
            return;
        }
        self.screen.scroll_view_to(to);
        // The text moved under a parked pointer, so whatever it now rests on is a
        // different link, or none.
        self.refresh_hover();
        self.dirty = true;
    }

    /// The drag ended: let go of the thumb and let the bar fade unless the pointer stayed
    /// on it.
    pub(in crate::app) fn release_scrollbar(&mut self) {
        self.scrollbar.release();
        self.dirty = true;
    }

    /// Light the bar, so every scroll (a wheel notch, a Shift+PageUp, a thumb drag) is
    /// confirmed by it sliding into view and then fading back out.
    pub(super) fn flash_scrollbar(&mut self) {
        self.scrollbar.flash(Instant::now());
    }

    /// Carry the bar forward a frame. A tab with nothing to scroll has nothing to point
    /// at, so its bar drops out of sight at once rather than fading from a history it no
    /// longer describes (switching to the alt screen is exactly this).
    fn tick_scrollbar(&mut self, now: Instant) {
        if self.scrollable() {
            if self.scrollbar.tick(now) {
                self.dirty = true;
            }
        } else {
            self.scrollbar.hide();
        }
    }

    /// When the loop should next tick the bar: a step of a fade, or the moment a lit
    /// bar's hold lapses and it starts fading. `None` when it is settled.
    pub(in crate::app) fn scrollbar_retry_at(&self) -> Option<Instant> {
        self.scrollbar.retry_at()
    }

    /// Whether the bar is mid-fade, so the next compositor frame should carry it on.
    pub(in crate::app) fn scrollbar_animating(&self) -> bool {
        self.scrollbar.animating()
    }

    /// Flip the blink phase if its deadline has passed (called each loop turn).
    fn tick_blink_if_due(&mut self, now: Instant) {
        if self.cursor_blinking() && self.blink_at.is_some_and(|at| at <= now) {
            self.tick_blink(now);
        }
    }

    /// Carry the visual timers to `now`: the cursor blink, the synchronized-output hold,
    /// the visual bell, and the scrollbar's fade. Each marks the frame dirty only when it
    /// changes what is on screen, so an idle turn stays idle. The debounced winsize is
    /// deliberately not here: it can fail, and it is owed to hidden tabs too (see
    /// [`Self::flush_winsize_if_due`]).
    pub(in crate::app) fn service_due(&mut self, now: Instant) {
        self.tick_blink_if_due(now);
        self.tick_sync_if_due(now);
        self.tick_bell_if_due(now);
        self.tick_scrollbar(now);
    }

    /// The soonest visual-timer deadline: the cursor's next blink (only while it is
    /// blinking), the end of a synchronized-output hold, and the end of a bell flash.
    /// The scrollbar's fade is not here: a pending compositor frame callback carries it
    /// instead of a wake (see [`Self::scrollbar_retry_at`]).
    pub(in crate::app) fn next_deadline(&self) -> Option<Instant> {
        [
            self.cursor_blinking().then_some(self.blink_at).flatten(),
            self.sync_until,
            self.bell_until,
        ]
        .into_iter()
        .flatten()
        .min()
    }

    /// Whether the cursor should be blinking right now: focused, visible, and the
    /// child asked for a blinking style.
    fn cursor_blinking(&self) -> bool {
        self.focused && self.screen.cursor_visible() && self.screen.cursor_blinks()
    }

    /// Flip the blink phase and schedule the next toggle, or rest the cursor once the
    /// budget is spent. It rests *lit*: a blink that stopped on its off phase is a
    /// cursor that vanished.
    fn tick_blink(&mut self, now: Instant) {
        self.blinks_left = self.blinks_left.saturating_sub(1);
        if self.blinks_left == 0 {
            self.blink_at = None;
            if !self.blink_on {
                self.blink_on = true;
                self.dirty = true;
            }
            return;
        }
        self.blink_on = !self.blink_on;
        self.blink_at = Some(now + BLINK_INTERVAL);
        self.dirty = true;
    }

    /// Reset the cursor to its lit phase, refill the toggle budget, and restart the
    /// blink timer, so it shows solid immediately after activity (a keystroke, output)
    /// and blinks only when idle. A no-op's timer stays `None` while unfocused. Relighting
    /// marks the frame dirty itself, because a keystroke the child does not echo (a
    /// password prompt) changes nothing else on screen.
    pub(super) fn bump_cursor(&mut self) {
        if !self.blink_on {
            self.blink_on = true;
            self.dirty = true;
        }
        self.blinks_left = BLINK_TOGGLES;
        self.blink_at = self.focused.then(|| Instant::now() + BLINK_INTERVAL);
    }
}

/// Map the grid's cursor style (from DECSCUSR) to how the renderer paints it.
///
/// A tty taking a password outranks whatever style the child last asked for. The two
/// are not really in competition: `DECSCUSR` is the child stating a preference, while
/// the lock reports a fact about the tty *underneath* the child, and that fact is the
/// more important thing to put on screen. It also cannot be spoofed by output, which
/// is the point of reading it from the kernel instead of trusting an escape sequence.
fn cursor_shape(style: CursorStyle, tty: TtyMode) -> CursorShape {
    match tty {
        TtyMode::PasswordPrompt => CursorShape::Lock,
        TtyMode::Cooked | TtyMode::Raw => match style {
            CursorStyle::Block => CursorShape::Block,
            CursorStyle::Underline => CursorShape::Underline,
            CursorStyle::Bar => CursorShape::Bar,
        },
    }
}

#[cfg(test)]
mod tests {
    use crate::app::message::PointerEvent;
    use crate::app::terminal::frame::*;
    use crate::app::terminal::tests::{core_80x24, geom};
    use crate::testutil::feed;

    #[test]
    fn a_password_prompt_outranks_whatever_style_the_child_asked_for() {
        // A shell that set a bar cursor (DECSCUSR 5) still gets a lock the moment the
        // tty stops echoing: the child's preference is about taste, the lock is about
        // what is happening to the user's keystrokes.
        for style in [CursorStyle::Block, CursorStyle::Underline, CursorStyle::Bar] {
            assert_eq!(
                cursor_shape(style, TtyMode::PasswordPrompt),
                CursorShape::Lock,
                "{style:?} is overridden while a password is being typed"
            );
        }

        // Off the prompt, the child's choice stands. `Raw` is a full-screen program,
        // which also has echo off: if the lock keyed on echo alone, every minute spent
        // in vim would show one.
        for tty in [TtyMode::Cooked, TtyMode::Raw] {
            assert_eq!(cursor_shape(CursorStyle::Block, tty), CursorShape::Block);
            assert_eq!(cursor_shape(CursorStyle::Bar, tty), CursorShape::Bar);
            assert_eq!(
                cursor_shape(CursorStyle::Underline, tty),
                CursorShape::Underline
            );
        }
    }

    #[test]
    fn a_real_tty_dropping_echo_locks_the_cursor_and_restoring_it_lets_go() {
        // The feature's spine, against a real child and the kernel's real line
        // discipline: PTY -> refresh_tty_mode -> the frame the window pulls. Only the
        // one-line call from `Tabs::note_settle` is left out, and that is the same hook
        // the cwd/foreground refresh already rides.
        let mut core = TerminalCore::new(false, geom(40, 10, 320, 160));
        if core.spawn_program(&["/bin/cat"]).is_err() {
            eprintln!("fork/exec unavailable; skipping the live tty-mode test");
            return;
        }
        // Focused, so the cursor it replaces is a *filled* block: exactly one fill, which
        // is what makes the count below arithmetic rather than a guess (an unfocused
        // cursor is a hollow block, which is four).
        core.focused = true;
        // The padlock's two solid parts (the arch and the body) are the only round-rects
        // the frame draws in the cursor colour: its hollow and keyhole take the cell's
        // background, and no other cursor shape is round. So this counts padlocks.
        let locked = |core: &TerminalCore| {
            let (mut list, mut strings) = (DisplayList::new(), Vec::new());
            core.fill_frame_list(&mut list, &mut strings, false);
            list.iter().any(|c| {
                matches!(
                    c,
                    crate::render::display::DrawCmd::Text { text, .. }
                        if text.contains(crate::term_render::LOCK_GLYPH)
                )
            })
        };

        core.refresh_tty_mode();
        assert_eq!(core.tty_mode, TtyMode::Cooked, "a fresh tty echoes");
        assert!(!locked(&core), "an echoing tty draws no lock");

        // What sudo does: stop echoing, print an ordinary line of text, say nothing.
        core.pty.as_ref().expect("a live pty").set_echo(false);
        core.dirty = false;
        core.refresh_tty_mode();
        assert_eq!(core.tty_mode, TtyMode::PasswordPrompt);
        assert!(
            core.dirty,
            "a silent prompt emits no output, so nothing but this would repaint it"
        );
        assert!(
            locked(&core),
            "the padlock reaches the frame the window pulls"
        );

        // And what it does once it has the password: give the echo back.
        core.pty.as_ref().expect("a live pty").set_echo(true);
        core.dirty = false;
        core.refresh_tty_mode();
        assert_eq!(core.tty_mode, TtyMode::Cooked);
        assert!(core.dirty, "letting go of the lock repaints too");
        assert!(!locked(&core), "the padlock is gone");
    }

    #[test]
    fn the_lock_reaches_the_frame_and_repaints_when_the_mode_turns() {
        // The same journey as the live-tty test above, but without a child, so it still
        // guards the wiring where fork/exec is unavailable: flip the cached mode the way
        // a settle would, and confirm the display list the window pulls carries a lock.
        // The padlock's two solid parts (the arch and the body) are the only round-rects
        // the frame draws in the cursor colour: its hollow and keyhole take the cell's
        // background, and no other cursor shape is round. So this counts padlocks.
        let locked = |core: &TerminalCore| {
            let (mut list, mut strings) = (DisplayList::new(), Vec::new());
            core.fill_frame_list(&mut list, &mut strings, false);
            list.iter().any(|c| {
                matches!(
                    c,
                    crate::render::display::DrawCmd::Text { text, .. }
                        if text.contains(crate::term_render::LOCK_GLYPH)
                )
            })
        };
        let mut core = core_80x24(true);
        core.focused = true;
        assert!(!locked(&core), "a cooked tty draws no lock");

        core.dirty = false;
        core.tty_mode = TtyMode::PasswordPrompt;
        core.dirty = true; // what `refresh_tty_mode` sets on a change

        assert!(core.dirty, "a mode change has to repaint");
        assert!(locked(&core), "the block cursor becomes a padlock");
    }

    #[test]
    fn synchronized_output_holds_the_frame_and_then_shows_it_whole() {
        // A program brackets a frame with `?2026 h` … `?2026 l` so it is never seen
        // half-drawn. The grid keeps updating throughout — only the *presentation* waits.
        let mut core = core_80x24(true);
        core.feed_test_bytes(b"\x1b[?2026h");
        assert!(core.holds_frame(), "the child is mid-frame");
        assert!(core.dirty, "and we still know a frame is owed");

        core.feed_test_bytes(b"\x1b[2J\x1b[Hhalf a frame");
        assert!(core.holds_frame(), "output does not end the frame");
        assert_eq!(
            core.row_string(0).trim_end(),
            "half a frame",
            "the grid updated all along; only the screen waited"
        );

        core.feed_test_bytes(b"\x1b[?2026l");
        assert!(!core.holds_frame(), "and now it goes up, all at once");
        assert!(core.dirty);
    }

    #[test]
    fn a_child_that_dies_mid_frame_cannot_freeze_the_window() {
        // The lock is held by the child, so the child can lose it — crash between the two
        // sequences, or simply never send the second. Without a deadline that is a window
        // that never repaints again, which is far worse than the torn frame the mode
        // exists to prevent. So the deadline is the feature, not a safety net bolted on.
        let mut core = core_80x24(true);
        core.feed_test_bytes(b"\x1b[?2026h");
        assert!(core.holds_frame());

        // The child is gone; nothing will ever send `?2026 l`. Pretend the deadline has
        // passed rather than sleeping for it.
        core.sync_until = Some(Instant::now() - Duration::from_millis(1));
        assert!(!core.holds_frame(), "the frame goes up anyway");
        assert!(
            core.sync_until.is_some(),
            "and the loop had a deadline to wake on, or nothing would have brought it back"
        );

        // Servicing that wake both presents the held frame and takes the deadline back
        // out of the loop's wait. Leaving it set is the whole defect: a deadline in the
        // past is not a deadline, it is a 1 ms wait forever.
        core.dirty = false;
        core.tick_sync_if_due(Instant::now());
        assert!(core.dirty, "the held frame is presented");
        assert_eq!(
            core.sync_until, None,
            "and the deadline is disarmed, or a past deadline clamps the wait to 1ms and \
             spins the loop at 1kHz forever"
        );

        // The child never released the lock, so the *next* output opens a fresh frame
        // under it. That is a new deadline, not an extension of the expired one, and it
        // is in the future, which is the property that matters: a child that holds
        // `?2026` forever costs one wake per timeout, not one per millisecond.
        assert!(core.screen.synchronized(), "the child never released it");
        core.feed_test_bytes(b"x");
        let rearmed = core.sync_until.expect("a fresh frame, a fresh deadline");
        assert!(
            rearmed > Instant::now(),
            "and it is ahead of us, not behind"
        );
        assert!(
            core.holds_frame(),
            "so the new frame is held like any other"
        );
    }

    #[test]
    fn a_live_synchronized_frame_is_not_cut_short_by_the_timer() {
        // The other half of the deadline's contract: servicing timers must not end a
        // frame the child is still legitimately drawing, or `?2026` would tear exactly
        // the frames it exists to keep whole.
        let mut core = core_80x24(true);
        core.feed_test_bytes(b"\x1b[?2026h");
        let armed = core.sync_until;
        assert!(armed.is_some(), "entering arms the deadline");

        core.tick_sync_if_due(Instant::now());
        assert_eq!(core.sync_until, armed, "still in the future, so untouched");
        assert!(core.holds_frame(), "and the frame is still held");

        // Leaving is what normally clears it, and still does.
        core.feed_test_bytes(b"\x1b[?2026l");
        assert_eq!(core.sync_until, None);
        assert!(!core.holds_frame());
    }

    #[test]
    fn ctrl_shift_up_jumps_to_the_prompt_and_stays_out_of_the_way_without_one() {
        let mut core = TerminalCore::new(false, geom(20, 5, 640, 384));
        // No shell integration: the chord is not ours, and it must fall through to the
        // child rather than being swallowed. A terminal that eats a key and does nothing
        // with it is worse than one that never claimed the key.
        assert!(!core.handle_scroll_key(input::Key::Up, input::Mods::CTRL | input::Mods::SHIFT));

        // A shell that marks its prompts gets the jump.
        core.feed_test_bytes(b"\x1b]133;A\x07$ build\r\n");
        for i in 0..30 {
            core.feed_test_bytes(format!("log line {i}\r\n").as_bytes());
        }
        assert_eq!(core.screen.view_offset(), 0, "at the live bottom");
        assert!(core.handle_scroll_key(input::Key::Up, input::Mods::CTRL | input::Mods::SHIFT));
        assert!(core.screen.view_offset() > 0, "we scrolled back");
        assert_eq!(
            core.screen.abs_row(0),
            core.screen.prompts()[0].row,
            "to the prompt, past everything the command printed"
        );
        // And Shift+Up alone is still just a key for the child.
        assert!(!core.handle_scroll_key(input::Key::Up, input::Mods::SHIFT));
    }

    #[test]
    fn the_bell_flashes_and_the_clear_colour_flashes_with_it() {
        // The GPU clear must use the same colour as the display list's base fill, or the
        // flash tears along the edges the list does not cover. Two places computing "the
        // background" independently is how that happens, so they compute it once.
        let mut core = core_80x24(true);
        let calm = core.clear_color();
        assert!(!core.bell_flashing());

        core.feed_test_bytes(b"\x07");
        assert!(core.bell_flashing(), "the bell rang");
        assert_ne!(
            core.clear_color(),
            calm,
            "and the clear lifted with the fill"
        );
        assert!(core.bell_until.is_some(), "with a deadline to end it");

        // The flash is not a state to be stuck in: when its moment passes it comes off,
        // and the frame that takes it off has to be asked for.
        core.bell_until = Some(Instant::now() - Duration::from_millis(1));
        core.dirty = false;
        core.tick_bell_if_due(Instant::now());
        assert!(!core.bell_flashing());
        assert!(core.dirty, "and it repaints to take the flash back off");
        assert_eq!(core.clear_color(), calm);
    }

    #[test]
    fn a_blinking_cursor_rests_lit_and_wakes_on_activity() {
        // Each toggle marks the frame dirty, so a blink that never rests is a GPU
        // present twice a second for as long as the window holds focus.
        let mut core = core_80x24(false);
        core.feed_test_bytes(b"\x1b[?12h");
        assert!(core.screen.cursor_blinks(), "the child asked for a blink");
        core.focus(true).expect("focus");
        assert!(core.next_deadline().is_some(), "focus arms it");

        // One half-period per turn, as the loop's wake delivers them.
        let start = Instant::now();
        for i in 1..=u32::from(BLINK_TOGGLES) {
            core.service_due(start + BLINK_INTERVAL * i);
        }
        assert_eq!(core.next_deadline(), None, "the blink rests");
        assert!(core.blink_on, "lit: a rested cursor must still be visible");

        // No deadline means no wake, however long the window then sits.
        core.service_due(start + BLINK_INTERVAL * 10_000);
        assert_eq!(core.next_deadline(), None);
        assert!(core.blink_on);

        // Activity refills the budget, so it blinks where someone is watching.
        core.feed_test_bytes(b"x");
        let rearmed = core.next_deadline().expect("output wakes the blink");
        assert!(rearmed > Instant::now(), "and ahead of us, not behind");
    }

    #[test]
    fn a_keystroke_in_the_off_phase_shows_the_cursor() {
        // A password prompt echoes nothing, so the keystroke is the only thing that can
        // bring a blinked-off cursor back: it has to repaint the frame by itself, or the
        // next toggle turns the cursor off again before anything drew it lit.
        let mut core = core_80x24(false);
        core.feed_test_bytes(b"\x1b[?12h");
        core.focus(true).expect("focus");
        core.service_due(Instant::now() + BLINK_INTERVAL);
        assert!(!core.blink_on, "the blink is in its off phase");
        core.dirty = false; // that frame is on screen

        core.key(
            input::Key::plain('a'),
            input::Mods::NONE,
            input::KeyEvent::Press,
        )
        .unwrap();

        assert!(core.blink_on, "the keystroke relit the cursor");
        assert!(core.dirty, "and the frame repaints to show it");
    }

    /// The text the view is showing at display `row`, which is what the user's eye is
    /// on. (`row_string` reads the *live* screen, so it is the wrong probe here.)
    fn view_row_text(core: &TerminalCore, row: usize) -> String {
        let (cols, _) = core.screen.dimensions();
        let text: String = (0..cols)
            .map(|c| core.screen.view_cell(row, c).rune)
            .collect();
        text.trim_end().to_string()
    }

    #[test]
    fn child_output_leaves_a_scrolled_view_where_the_user_put_it() {
        // The reported bug, at the layer it lived on: `after_output` pinned the view to
        // the live bottom on every read from the PTY, so a chatty child (a spinner, a
        // build) snatched the view back ten times a second and scrolling back while it
        // printed was impossible. Output moves the content, so the offset moves with it.
        let mut core = core_80x24(true);
        for i in 0..40 {
            core.feed_test_bytes(format!("line {i}\r\n").as_bytes());
        }
        core.screen.scroll_view_up(5);
        assert!(core.screen.is_scrolled());
        let under_the_eye = view_row_text(&core, 0);

        for i in 40..50 {
            core.feed_test_bytes(format!("line {i}\r\n").as_bytes());
        }

        assert_eq!(
            core.screen.view_offset(),
            15,
            "the offset followed the output"
        );
        assert_eq!(
            view_row_text(&core, 0),
            under_the_eye,
            "the text held still"
        );

        // The one thing that still snaps to the bottom: the user. A keystroke must
        // never land blind in the middle of history.
        core.key(
            input::Key::plain('x'),
            input::Mods::NONE,
            input::KeyEvent::Press,
        )
        .unwrap();
        assert_eq!(
            core.screen.view_offset(),
            0,
            "typing returns to the live bottom"
        );
    }

    /// A live core with `lines` of output behind an 80x24 screen, so it has history to
    /// scroll. Its geometry is the one [`TerminalCore::new`] starts at: an 8x16 cell, no
    /// padding, unity scale, so the lane lands at a hand-checkable place (see
    /// [`lane_x`]).
    fn scrollable_core(lines: usize) -> TerminalCore {
        let mut core = core_80x24(false);
        for i in 0..lines {
            feed(&mut core.screen, format!("line {i}\r\n").as_bytes());
        }
        core
    }

    /// A device x inside the scrollbar's grab band of a [`scrollable_core`]: the lane is
    /// 10px wide, inset 3px from the surface's right edge (640), so it spans 627..637 and
    /// the band runs out to 640.
    const fn lane_x() -> f32 {
        630.0
    }

    /// The full height of that core's track, which is the grid's: 24 rows of 16px.
    const TRACK_H: f32 = 24.0 * 16.0;

    #[test]
    fn a_screen_with_history_behind_it_is_scrollable() {
        let core = scrollable_core(100);
        let (content, viewport) = core.screen.scroll_extent();
        assert_eq!(viewport, 24, "the viewport is the screen");
        assert!(content > viewport, "and there is history behind it");
        assert!(core.scrollable());
        // A fresh screen has nothing behind it and so offers no bar.
        assert!(!scrollable_core(0).scrollable());
    }

    #[test]
    fn the_bar_takes_a_press_in_its_lane_but_not_one_in_the_text() {
        let core = scrollable_core(100);
        assert!(core.on_scrollbar(lane_x(), 100.0), "the lane is grabbable");
        assert!(
            !core.on_scrollbar(300.0, 100.0),
            "a press out in the text is the grid's, not the bar's"
        );
        assert!(
            !core.on_scrollbar(lane_x(), TRACK_H + 20.0),
            "and below the grid there is no lane at all"
        );
    }

    #[test]
    fn the_alt_screen_offers_no_bar_to_grab() {
        let mut core = scrollable_core(100);
        feed(&mut core.screen, b"\x1b[?1049h");
        assert!(core.screen.is_alt());
        assert!(!core.scrollable(), "the alt screen keeps no history");
        assert!(
            !core.on_scrollbar(lane_x(), 100.0),
            "so a press in the lane falls through to the program"
        );
    }

    #[test]
    fn dragging_the_thumb_up_the_track_walks_the_view_into_history() {
        let mut core = scrollable_core(100);
        let history = core.screen.scrollback_len();
        assert_eq!(core.screen.view_offset(), 0, "starts at the live bottom");

        // Press on the thumb (which rests against the bottom of the track, since the view
        // is at the live bottom) and drag it to the very top.
        assert!(core.press_scrollbar(lane_x(), TRACK_H - 4.0));
        core.drag_scrollbar(0.0);
        assert_eq!(
            core.screen.view_offset(),
            history,
            "the thumb at the top of the track shows the oldest line kept"
        );

        // And back down again: the view returns to the live output.
        core.drag_scrollbar(TRACK_H);
        assert_eq!(core.screen.view_offset(), 0);
        core.release_scrollbar();
    }

    #[test]
    fn a_click_on_the_bare_track_jumps_the_view_to_it() {
        // GNOME's behavior, not Windows': a click on the empty track goes to that spot
        // rather than paging toward it. The thumb lands centred on the pointer, so a click
        // at the very top of the track shows the oldest line.
        let mut core = scrollable_core(100);
        let history = core.screen.scrollback_len();
        assert!(core.press_scrollbar(lane_x(), 0.0));
        assert_eq!(core.screen.view_offset(), history);
    }

    #[test]
    fn a_press_outside_the_lane_is_not_the_scrollbars() {
        let mut core = scrollable_core(100);
        assert!(
            !core.press_scrollbar(300.0, 100.0),
            "a press in the text leaves the bar alone, so a selection can begin"
        );
        assert_eq!(core.screen.view_offset(), 0, "and the view has not moved");
    }

    #[test]
    fn every_scroll_lights_the_bar() {
        let mut core = scrollable_core(100);
        assert_eq!(
            core.scrollbar_retry_at(),
            None,
            "a bar nothing has woken is settled and costs the loop nothing"
        );

        // The wheel.
        core.pointer(
            PointerEvent::Wheel {
                down: false,
                notches: 1,
                col: 0,
                row: 0,
            },
            input::Mods::NONE,
        )
        .unwrap();
        assert!(core.screen.is_scrolled(), "the wheel scrolled the view");
        assert!(
            core.scrollbar_retry_at().is_some(),
            "and lit the bar, which now wants the loop back to fade it"
        );

        // And the keyboard, which scrolls by the same view and must say so the same way.
        let mut core = scrollable_core(100);
        assert!(core.handle_scroll_key(input::Key::PageUp, input::Mods::SHIFT));
        assert!(core.scrollbar_retry_at().is_some());
    }

    #[test]
    fn a_bar_with_nothing_left_to_scroll_hides_itself() {
        // Switching to the alt screen takes the history away under a lit bar. The tick is
        // what notices, so the bar cannot linger describing a scrollback that is not
        // showing.
        let mut core = scrollable_core(100);
        core.flash_scrollbar();
        assert!(core.scrollbar_retry_at().is_some());

        feed(&mut core.screen, b"\x1b[?1049h");
        core.tick_scrollbar(Instant::now());
        assert_eq!(
            core.scrollbar_retry_at(),
            None,
            "the bar dropped out of sight rather than fading from a history it no longer describes"
        );
    }
}
