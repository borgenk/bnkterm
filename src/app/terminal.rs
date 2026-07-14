//! The per-tab terminal core behind the tabs/window seam. One [`TerminalCore`]
//! exists for each live tab and owns its PTY, VT parser, and grid, plus everything
//! that is a function of them: the text selection, the cursor blink phase, the
//! theme, and the frame geometry it needs to lay the grid out. It never touches
//! Wayland, xkb, or the GPU.
//!
//! ```text
//!   ToTerminal ─▶ TerminalCore::apply ─┬─▶ PTY write   (input → child)
//!                                      └─▶ grid mutate  (parser, selection, scroll)
//!   PTY master ─▶ gather thread ─▶ pump ─▶ parser ─▶ grid ─▶ outbox (Title / Closed)
//!   grid state ─▶ fill_frame_list ─▶ DisplayList (pulled by the window each frame)
//! ```
//!
//! [`super::tabs::Tabs`] drives each core on the main thread: it routes the
//! window's [`ToTerminal`] messages to the active core (or resize to all cores),
//! pulls only the active [`DisplayList`], and translates each per-core
//! [`ToWindow`] fact. The child's output is drained off-thread by [`crate::gather`] (see
//! off-thread); `pump` consumes the published batches. Only
//! reads move off the main thread; the parser, grid, and every write stay here.

use std::os::fd::RawFd;
use std::time::{Duration, Instant};

use super::message::{PointerEvent, ToTerminal, ToWindow};
use crate::color::Theme;
use crate::config::TabBarConfig;
use crate::error::Result;
use crate::gather::{GatherEnd, Gatherer};
use crate::grid::{AbsRow, ClipboardTarget, CursorStyle, LinkProbe, RowEpoch, Screen};
use crate::input;
use crate::mouse::{self, MouseButton, MouseKind};
use crate::platform::browser;
use crate::platform::geom::{Rect, Scale};
use crate::platform::scroll::{self, Scrollbar};
use crate::pty::{Pty, TtyMode, ZombieChild};
use crate::render::display::DisplayList;
use crate::term_render::{self, CellMetrics, CellSpan, CursorRender, CursorShape, Selection};
use crate::vt::Parser;

/// The cursor blink half-period: how long each of the on/off phases lasts.
const BLINK_INTERVAL: Duration = Duration::from_millis(530);

/// How long a child may hold a frame under synchronized output (`?2026`) before we show
/// it anyway.
///
/// The lock is held by the *child*, which means a child that crashes between "begin
/// frame" and "end frame" — or simply forgets the second one — would otherwise freeze
/// the window for good. The escape hatch is not optional, and it is why this mode is
/// safe to implement at all. 150ms is kitty's figure: far longer than any real frame,
/// far shorter than a user notices something is wrong.
const SYNC_TIMEOUT: Duration = Duration::from_millis(150);

/// How long the visual bell lasts. Long enough to catch the eye, short enough that a
/// program ringing the bell in a loop (a shell tab-completing against nothing does it)
/// reads as a flicker and not a strobe.
const BELL_FLASH: Duration = Duration::from_millis(120);

/// Lines the scrollback view moves per wheel notch, and arrows sent per notch
/// when the wheel falls back to arrow keys on the alt screen.
const WHEEL_LINES: usize = 3;

/// The granularity a selection drag extends by, chosen from the press's click
/// count: a single click selects by character, a double-click by word, a
/// triple-click by whole (soft-wrapped) line.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SelectMode {
    Char,
    Word,
    Line,
}

/// A left-drag in progress: everything the pointer needs to keep extending it.
///
/// One value rather than three fields, because they are only meaningful together. In
/// particular the `epoch`: a character drag deliberately paints *no* selection until it
/// leaves the cell it started in (so a plain click never flashes), yet it is already
/// pinned to an anchor. If the grid renumbers its rows in that window — a resize, an
/// alt-screen switch, a `clear`, a region scroll — the anchor names nothing, and the drag
/// has to be cancelled rather than resumed against whatever row inherited the id. Kept
/// beside the anchor it qualifies, there is no way to check one and forget the other.
#[derive(Clone, Copy)]
struct Drag {
    /// The granularity the drag extends by, from the press's click count.
    mode: SelectMode,
    /// The clicked unit's inclusive cell range (a character, word, or line), the pivot a
    /// drag extends around so the anchored unit always stays selected.
    anchor: ((AbsRow, usize), (AbsRow, usize)),
    /// The identity regime the anchor's rows were minted in.
    epoch: RowEpoch,
}

/// What one budgeted pass over a core's gather queue observed. The tab manager
/// spends one global budget across these outcomes and owns per-tab teardown.
pub(super) struct PumpOutcome {
    pub(super) bytes: usize,
    pub(super) more: bool,
    pub(super) end: Option<GatherEnd>,
}

/// The terminal half of the app: the PTY, parser, and grid, plus the state that
/// is a pure function of them. The window feeds it [`ToTerminal`] messages, pulls
/// a [`DisplayList`], and drains the [`ToWindow`] outbox. It holds its own copies
/// of the frame geometry (`metrics`/`width`/`height`/`pad`); the window computes
/// them (it owns the fonts and scale) and ships them over on a [`ToTerminal::Resize`].
pub(super) struct TerminalCore {
    /// The grid being shown, always sized to the current window's `(cols, rows)`.
    screen: Screen,
    /// The VT state machine driving `screen` from the child's output bytes.
    parser: Parser,
    /// The dedicated PTY reader: it drains the master on its own thread into a
    /// bounded pool, and `pump` consumes the published batches. `None` in demo
    /// mode and before the shell is spawned. Declared before `pty` so it stops and
    /// joins before the master fd closes.
    gatherer: Option<Gatherer>,
    /// The child on the far side of the PTY; `None` in demo mode (and before the
    /// first configure, since the PTY is sized to the granted window).
    pty: Option<Pty>,
    /// Demo mode: a static grid, no shell.
    demo: bool,
    /// Reused key-encoding buffer (allocated once, not per key press).
    key_buf: Vec<u8>,
    /// Whether the surface holds keyboard focus, so the cursor draws solid when
    /// focused and hollow when not.
    focused: bool,
    /// Cursor blink: the current on/off phase, and when it next toggles (`None`
    /// when not blinking, e.g. unfocused). Activity resets it to on.
    blink_on: bool,
    blink_at: Option<Instant>,
    /// When the current synchronized-output lock gives up and we present regardless.
    /// `None` when the child is not holding a frame.
    sync_until: Option<Instant>,
    /// When the visual bell stops flashing. `None` when it is not ringing.
    bell_until: Option<Instant>,
    /// What the child's tty is doing, as of the last time its output settled (see
    /// [`refresh_tty_mode`](Self::refresh_tty_mode)). Cached rather than probed per
    /// frame because it changes only when the child calls `tcsetattr`, and because a
    /// password prompt is *silent* by definition: the user types and nothing comes
    /// back, so there is no output to re-probe on while the lock must stay up.
    tty_mode: TtyMode,
    /// The button held for drag reporting under mouse mode (`None` when none is
    /// down).
    mouse_held: Option<MouseButton>,
    /// The active text selection (a left-drag), or `None`. In absolute rows, so child
    /// output scrolling the grid does not drag it off the text it was made over; see
    /// [`Self::prune_selection`] for the (short) list of things that do end it.
    selection: Option<Selection>,
    /// The left-drag in progress (the button is down), or `None`. Absolute, like the
    /// selection it pivots: output during a drag must not move the anchor.
    drag: Option<Drag>,
    /// The hyperlink under the pointer: its cells paint underlined and Ctrl+click
    /// follows it. `None` whenever there is nothing to follow — no URL under the
    /// pointer, a drag in progress (the gesture is a selection), a program grabbing
    /// the mouse (its clicks are its own), or the pointer outside the grid.
    hover: Option<CellSpan>,
    /// The cell the pointer sits in while hovering is live, or `None` when it is not.
    /// Two jobs: it makes a motion inside one cell free (the common case, since a
    /// pointer crosses many pixels per cell), and it is the position
    /// [`TerminalCore::refresh_hover`] re-probes from when the text scrolls out from
    /// under a parked pointer.
    hover_cell: Option<(usize, usize)>,
    /// Reused scratch for the hyperlink probe, so hovering allocates nothing once its
    /// buffers have grown (see [`LinkProbe`]).
    probe: LinkProbe,
    /// The content changed and a frame should be drawn. The window reads it to
    /// pace repaints, and sets it on a geometry change it drives.
    pub(super) dirty: bool,
    /// Frame geometry, all window-computed and shipped over on a resize: the fixed
    /// cell box, the device surface size the grid lays out in, the device padding inset
    /// on every side, and the display scale the chrome is sized in.
    metrics: CellMetrics,
    width: u32,
    height: u32,
    pad: i32,
    origin_y: i32,
    scale: Scale,
    /// This tab's overlay scrollbar: it fades in on a scroll and out after it, and grows
    /// into a grabbable slider under the pointer. Per-tab because the scrollback it
    /// describes is: each tab scrolls its own history, and only the visible one animates.
    scrollbar: Scrollbar,
    /// Outbound messages for the window to act on after the next drain (a title
    /// change, a fresh selection to own, or the child exiting).
    outbox: Vec<ToWindow>,
    /// The child's last-seen window title, so a `Title` is emitted only when it
    /// changes (the sender dedupes, so Stage 2 never spams the channel).
    last_title: String,
    /// The child's working directory, re-read when the tab's output settles. It
    /// labels the tab when no program has set a title; `None` before the first
    /// read (or on a kernel without `/proc`).
    cwd: Option<std::path::PathBuf>,
    /// The foreground program's name (`comm`), re-read alongside `cwd`. When it
    /// matches the tab bar's configured list, the label is prefixed with `cwd` even
    /// though a title is set, so a program that names its own tab (e.g. `claude`)
    /// still reveals its directory. `None` before the first read.
    foreground: Option<String>,
}

impl TerminalCore {
    /// Build the core at the window's initial geometry. The window owns the fonts
    /// and scale, so it computes `metrics`/`width`/`height`/`pad` and hands over
    /// copies; a later [`ToTerminal::Resize`] keeps them current. Demo mode starts
    /// on a static grid; live mode starts blank until the shell fills it.
    pub(super) fn new(
        demo: bool,
        cols: usize,
        rows: usize,
        metrics: CellMetrics,
        width: u32,
        height: u32,
        pad: i32,
    ) -> Self {
        let screen = if demo {
            demo_screen(cols, rows)
        } else {
            Screen::new(cols, rows)
        };
        Self {
            screen,
            parser: Parser::new(),
            gatherer: None,
            pty: None,
            demo,
            key_buf: Vec::new(),
            focused: false,
            blink_on: true,
            blink_at: None,
            sync_until: None,
            bell_until: None,
            tty_mode: TtyMode::Cooked,
            mouse_held: None,
            selection: None,
            drag: None,
            hover: None,
            hover_cell: None,
            probe: LinkProbe::default(),
            dirty: true,
            metrics,
            width,
            height,
            pad,
            origin_y: pad,
            scale: Scale::ONE,
            scrollbar: Scrollbar::default(),
            outbox: Vec::new(),
            last_title: String::new(),
            cwd: None,
            foreground: None,
        }
    }

    /// Spawn the shell on a PTY sized to the current grid, returning the grid it
    /// was sized to (for the startup log). Only the live path calls this; demo mode
    /// never spawns a child. A gather thread is started on a duplicate of the master
    /// fd to drain the child's output; if it cannot start (no eventfd or thread,
    /// which on Linux means the process is already out of descriptors or threads),
    /// shell startup fails cleanly rather than limping on.
    pub(super) fn spawn_shell(&mut self) -> Result<(usize, usize)> {
        let (cols, rows) = self.screen.dimensions();
        let pty = Pty::spawn(cols, rows)?;
        let gatherer = match Gatherer::start(pty.fd()) {
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
        // Seed the directory label so a fresh tab shows its cwd before the shell
        // has printed a thing.
        self.cwd = self.pty.as_ref().and_then(Pty::cwd);
        Ok((cols, rows))
    }

    /// Stop and join the gather thread, close the PTY master, and hand the child
    /// pid to the tabs layer for repeated nonblocking reaping. Demo and not-yet-
    /// spawned cores have no child.
    pub(super) fn into_child(mut self) -> Option<ZombieChild> {
        drop(self.gatherer.take());
        self.pty.take().map(Pty::into_zombie)
    }

    /// Whether this core is the static demo (no PTY, no shell).
    pub(super) fn is_demo(&self) -> bool {
        self.demo
    }

    /// The fd for the event-loop `poll`: the gather thread's ready eventfd, which
    /// signals when the child produced output. `None` before the shell is spawned
    /// (and in demo mode).
    pub(super) fn poll_fd(&self) -> Option<RawFd> {
        self.gatherer.as_ref().map(Gatherer::ready_fd)
    }

    /// Whether this core still has published gather batches waiting to be parsed.
    pub(super) fn has_pending(&self) -> bool {
        self.gatherer.as_ref().is_some_and(Gatherer::has_pending)
    }

    /// The grid dimensions currently owned by this core.
    pub(super) fn dimensions(&self) -> (usize, usize) {
        self.screen.dimensions()
    }

    #[cfg(test)]
    pub(super) fn origin_y(&self) -> i32 {
        self.origin_y
    }

    /// The title shown for this core, with the empty/default title mapped to the
    /// application name just like outbound title messages. This is the *window*
    /// title (the OS caption); the tab bar uses [`tab_label`](Self::tab_label).
    pub(super) fn title(&self) -> &str {
        if self.last_title.is_empty() {
            "bnkterm"
        } else {
            &self.last_title
        }
    }

    /// The label for this core's tab: the child-set window title when there is one,
    /// otherwise its working directory (`~`-abbreviated), falling back to `shell`.
    /// Owned because the directory string is derived, not stored ready to lend.
    /// This mirrors the sibling `wezterm.lua`, whose tab label prefers a program's
    /// own title and otherwise shows the `~`-abbreviated cwd.
    ///
    /// One exception keeps directory context for programs that name their own tab:
    /// when a title is set *and* the foreground program is in
    /// [`TabBarConfig::path_prefix_programs`], the label is `"<dir> - <title>"`,
    /// where `<dir>` is the cwd's final component. `claude`, for instance, sets the
    /// session name as the title; the prefix restores which directory it runs in.
    /// The directory leads so the end-truncating [`tab_bar::fit_end`](crate::tab_bar)
    /// clips the volatile title first.
    pub(super) fn tab_label(&self, cfg: &TabBarConfig) -> String {
        if !self.last_title.is_empty() {
            if let Some(dir) = self.prefix_dir(cfg) {
                return format!("{dir} - {}", self.last_title);
            }
            return self.last_title.clone();
        }
        match &self.cwd {
            Some(path) => abbreviate_home(path),
            None => "shell".to_string(),
        }
    }

    /// The cwd's final component when the foreground program opts into the path
    /// prefix, else `None`. A rootless cwd (`/`, with no final component) yields
    /// `None`, so the label simply shows the bare title.
    fn prefix_dir(&self, cfg: &TabBarConfig) -> Option<String> {
        let program = self.foreground.as_deref()?;
        if !cfg.path_prefix_programs.iter().any(|p| p == program) {
            return None;
        }
        let name = self.cwd.as_ref()?.file_name()?;
        Some(name.to_string_lossy().into_owned())
    }

    /// Re-read the working directory and foreground program from the child and
    /// report whether either changed, so the manager rebuilds the bar only when the
    /// label might move. Both feed the label (the cwd directly, the program through
    /// [`TabBarConfig::path_prefix_programs`]), so a change in either can shift what
    /// is shown; the manager applies the cheap rebuild without re-deriving here.
    pub(super) fn refresh_process(&mut self) -> bool {
        let cwd = self.pty.as_ref().and_then(Pty::cwd);
        let foreground = self.pty.as_ref().and_then(Pty::foreground_program);
        if cwd == self.cwd && foreground == self.foreground {
            return false;
        }
        self.cwd = cwd;
        self.foreground = foreground;
        true
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
    pub(super) fn refresh_tty_mode(&mut self) {
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
    pub(super) fn feed_test_bytes(&mut self, bytes: &[u8]) {
        self.parser.advance_bytes(&mut self.screen, bytes);
        self.after_output();
    }

    /// Test-only visible row text for cross-tab PTY routing assertions.
    #[cfg(test)]
    pub(super) fn row_string(&self, row: usize) -> String {
        self.screen.row_string(row)
    }

    /// The frame background as a `0x00RRGGBB`, for the GPU clear (which must match
    /// the display list's own base fill).
    /// The palette the grid is currently rendering with.
    pub(super) fn theme(&self) -> &Theme {
        self.screen.theme()
    }

    pub(super) fn clear_color(&self) -> u32 {
        // The grid owns the palette, because a program can change it (`OSC 11`). Read it
        // from there rather than keeping a second copy that would drift the moment it did.
        // The bell lifts it, and the clear must lift with it or the flash tears at the
        // edges the display list does not cover.
        term_render::bell_background(self.screen.theme(), self.bell_flashing()).to_u32()
    }

    /// Take the outbound messages for the window to act on, leaving the outbox
    /// empty (and its capacity for reuse). Empty in steady state, so this is
    /// allocation-free on the hot path.
    pub(super) fn take_outbox(&mut self) -> Vec<ToWindow> {
        std::mem::take(&mut self.outbox)
    }

    /// Apply a [`ToTerminal`] message: the terminal half of the seam, turning
    /// window intent into PTY bytes and grid mutations. Returns whether the child
    /// received bytes, which the key path uses to gate auto-repeat.
    pub(super) fn apply(&mut self, msg: ToTerminal) -> Result<bool> {
        match msg {
            ToTerminal::Key { key, mods, event } => {
                let modes = input::Modes::from_screen(&self.screen);
                self.key_buf.clear();
                input::encode_event(key, mods, event, modes, &mut self.key_buf);
                if self.key_buf.is_empty() {
                    return Ok(false);
                }
                // Typing snaps the view back to the live bottom before the bytes go
                // out, so a keystroke never lands "blind" while reading history.
                if self.screen.is_scrolled() {
                    self.screen.scroll_view_to_bottom();
                    self.dirty = true;
                }
                self.bump_cursor(); // keep the cursor solid while typing
                if let Some(pty) = &self.pty {
                    pty.write_all(&self.key_buf)?;
                }
                Ok(true)
            }
            ToTerminal::Pointer { event, mods } => {
                self.apply_pointer(event, mods)?;
                Ok(false)
            }
            ToTerminal::Resize {
                cols,
                rows,
                width,
                height,
                metrics,
                pad,
                origin_y,
                scale,
            } => {
                // Adopt the window's fresh geometry, then resize the grid to it. In
                // demo mode there is no child; rebuild the static grid. Otherwise
                // push the size to the child (TIOCSWINSZ → SIGWINCH). The PTY resize
                // is best-effort: a resize on a dead child just surfaces as EOF on
                // the next read, which shuts down cleanly.
                self.width = width;
                self.height = height;
                self.metrics = metrics;
                self.pad = pad;
                self.origin_y = origin_y;
                self.scale = scale;
                if self.demo {
                    // The grid is replaced wholesale, so no row id minted against the
                    // old one survives it.
                    self.screen = demo_screen(cols, rows);
                    self.selection = None;
                    self.drag = None;
                } else {
                    // A resize re-indexes rows and ends their epoch, which is what the
                    // prune below reads to drop a selection that no longer means
                    // anything. It is checked here and not only after output because a
                    // resize need not be followed by any.
                    self.screen.resize(cols, rows);
                    if let Some(pty) = &self.pty {
                        let _ = pty.resize(cols, rows);
                    }
                    // `?2048`: the same news in band, for the child that asked. SIGWINCH
                    // reaches only this side of the pty, so a program behind ssh or tmux
                    // hears nothing from it. Silent unless it was asked for.
                    self.screen.set_pixel_size(width, height);
                    self.screen.report_size();
                    self.flush_responses()?;
                }
                self.prune_selection();
                self.dirty = true;
                Ok(false)
            }
            ToTerminal::Focus(focused) => {
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
                Ok(false)
            }
            ToTerminal::Paste(text) => {
                let text = String::from_utf8_lossy(&text);
                if text.is_empty() {
                    return Ok(false);
                }
                // A pasted newline is delivered as CR; collapse CRLF so it is not
                // doubled. Wrap in bracketed-paste markers when the program enabled
                // them (`?2004`). Snaps the view to the bottom, like any input.
                let normalized = text.replace("\r\n", "\r").replace('\n', "\r");
                let bracketed = self.screen.bracketed_paste();
                let mut buf = Vec::with_capacity(normalized.len() + 12);
                if bracketed {
                    buf.extend_from_slice(b"\x1b[200~");
                }
                buf.extend_from_slice(normalized.as_bytes());
                if bracketed {
                    buf.extend_from_slice(b"\x1b[201~");
                }
                if self.screen.is_scrolled() {
                    self.screen.scroll_view_to_bottom();
                    self.dirty = true;
                }
                if let Some(pty) = &self.pty {
                    pty.write_all(&buf)?;
                }
                Ok(true)
            }
        }
    }

    /// The terminal half of a pointer event: report it to a program grabbing the
    /// mouse, or drive local selection / scrollback scroll / hyperlinks. The window
    /// already mapped the event to a cell and supplied the modifier chord (Shift
    /// forces local use even while a program is reporting the mouse).
    fn apply_pointer(&mut self, event: PointerEvent, mods: input::Mods) -> Result<()> {
        let reporting = self.screen.mouse_mode().reports() && !mods.contains(input::Mods::SHIFT);
        match event {
            PointerEvent::Button {
                button,
                pressed,
                col,
                row,
                count,
            } => {
                if reporting {
                    self.mouse_held = pressed.then_some(button);
                    let kind = if pressed {
                        MouseKind::Press
                    } else {
                        MouseKind::Release
                    };
                    self.write_mouse(button, kind, col, row, mods)?;
                } else if button == MouseButton::Left {
                    // Ctrl+click follows a hyperlink instead of starting a selection.
                    // It sits inside the local branch on purpose: while a program is
                    // grabbing the mouse, its clicks are its own, and Shift (which
                    // already means "this one is mine") is what frees a link there.
                    if pressed && mods.contains(input::Mods::CTRL) && self.open_link(row, col) {
                        return Ok(());
                    }
                    // Local selection: a press begins one at the click's granularity
                    // (character/word/line), a release ends the drag and offers the
                    // text to the clipboard and primary selection.
                    if pressed {
                        self.begin_selection(row, col, count);
                    } else {
                        self.drag = None;
                        self.finish_selection();
                    }
                    self.dirty = true;
                } else if button == MouseButton::Middle && pressed {
                    // Middle-click pastes the primary selection (the Linux
                    // convention). The window owns the data device, so it does the
                    // receive; the core only asks.
                    self.outbox.push(ToWindow::PastePrimary);
                }
            }
            PointerEvent::Motion { col, row } => {
                if self.drag.is_some() {
                    self.extend_selection(row, col);
                } else if reporting {
                    // Report motion to a program that asked for it (drag under ?1002,
                    // any move under ?1003).
                    let button = self.mouse_held.unwrap_or(MouseButton::None);
                    self.write_mouse(button, MouseKind::Motion, col, row, mods)?;
                }
                // A hover is live only when the gesture is not already spoken for: a
                // drag is a selection, and a grabbed mouse belongs to the program.
                self.track_hover(row, col, !reporting && self.drag.is_none());
            }
            PointerEvent::Left => self.track_hover(0, 0, false),
            PointerEvent::Wheel {
                down,
                notches,
                col,
                row,
            } => {
                if reporting {
                    let button = if down {
                        MouseButton::WheelDown
                    } else {
                        MouseButton::WheelUp
                    };
                    for _ in 0..notches {
                        self.write_mouse(button, MouseKind::Press, col, row, mods)?;
                    }
                } else if !self.screen.is_alt() {
                    let lines = notches as usize * WHEEL_LINES;
                    if down {
                        self.screen.scroll_view_down(lines);
                    } else {
                        self.screen.scroll_view_up(lines);
                    }
                    self.flash_scrollbar();
                    // The text moved but the pointer did not: whatever it now rests on
                    // is a different link, or none.
                    self.refresh_hover();
                    self.dirty = true;
                } else {
                    // Alt screen without mouse reporting: wheel becomes arrow keys, the
                    // conventional fallback so wheel-scrolling `less`/`man` works.
                    let key = if down {
                        input::Key::Down
                    } else {
                        input::Key::Up
                    };
                    let modes = input::Modes::from_screen(&self.screen);
                    for _ in 0..(notches as usize * WHEEL_LINES) {
                        self.key_buf.clear();
                        input::encode(key, input::Mods::NONE, modes, &mut self.key_buf);
                        if let Some(pty) = &self.pty {
                            pty.write_all(&self.key_buf)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Track the hyperlink under the pointer as it moves to display `(row, col)`.
    /// `live` is false when the gesture belongs to something else (a drag, a program
    /// grabbing the mouse) or the pointer has left the grid, which drops the hover.
    ///
    /// The cell check is what keeps this cheap: a pointer crosses many pixels per
    /// cell, so all but the first motion event within a cell returns here.
    fn track_hover(&mut self, row: usize, col: usize, live: bool) {
        if !live {
            self.hover_cell = None;
            self.set_hover(None);
            return;
        }
        if self.hover_cell == Some((row, col)) {
            return;
        }
        self.hover_cell = Some((row, col));
        self.refresh_hover();
    }

    /// Re-probe the link under the pointer where it currently rests. Called both when
    /// the pointer moves to a new cell and when the grid changes under a *parked*
    /// pointer: output scrolling the screen moves the text out from under it, so a
    /// hover left alone would underline whatever slid into its place.
    ///
    /// Cheap enough to run on any turn that changed the screen: one logical line
    /// scanned, and no allocation once [`LinkProbe`]'s buffers have grown.
    fn refresh_hover(&mut self) {
        let Some((row, col)) = self.hover_cell else {
            return;
        };
        let found = self.screen.link_at(row, col, &mut self.probe);
        self.set_hover(found.map(|(start, end)| CellSpan { start, end }));
    }

    /// Adopt a new hovered span, repainting only when it actually changed (a pointer
    /// crossing cells *within* one link must not redraw the frame).
    fn set_hover(&mut self, hover: Option<CellSpan>) {
        if self.hover != hover {
            self.hover = hover;
            self.dirty = true;
        }
    }

    /// Follow a Ctrl+clicked hyperlink at display `(row, col)`, returning whether one
    /// was there (in which case the click is consumed and starts no selection).
    ///
    /// The URL is re-probed at the clicked cell rather than read off the hovered span,
    /// so what opens is what was clicked, with no chance of acting on a hover that
    /// output has since invalidated. It is vetted before it is queued: the text came
    /// from the child, which can print anything, so only a scheme
    /// [`browser::open_url`] will actually launch ever reaches the window.
    fn open_link(&mut self, row: usize, col: usize) -> bool {
        if self.screen.link_at(row, col, &mut self.probe).is_none() {
            return false;
        }
        let url = self.probe.url();
        if !browser::can_open(url) {
            return false;
        }
        self.outbox.push(ToWindow::OpenUrl(url.to_string()));
        true
    }

    /// Encode one mouse event under the current mouse mode and write it to the child
    /// (nothing when the mode produces no bytes for it).
    fn write_mouse(
        &mut self,
        button: MouseButton,
        kind: MouseKind,
        col: usize,
        row: usize,
        mods: input::Mods,
    ) -> Result<()> {
        let mode = self.screen.mouse_mode();
        self.key_buf.clear();
        if mouse::encode(mode, button, kind, col, row, mods, &mut self.key_buf) {
            if let Some(pty) = &self.pty {
                pty.write_all(&self.key_buf)?;
            }
        }
        Ok(())
    }

    /// Scroll the scrollback view for a navigation chord (Shift + PageUp/PageDown/
    /// Home/End) on the primary screen, returning whether the key was consumed. A
    /// page is a screenful less one line, so a line of context carries across. The
    /// window resolves the keycode to a named key; the grid does the scrolling.
    pub(super) fn handle_scroll_key(&mut self, key: input::Key, mods: input::Mods) -> bool {
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
            _ => return false,
        }
        // A keyboard scroll is a scroll: show the bar moving, the way the wheel does.
        self.flash_scrollbar();
        self.refresh_hover();
        self.dirty = true;
        true
    }

    /// Copy the current selection's text into the outbox for the window to own on
    /// the clipboard (no-op without a selection or with empty text). The grid holds
    /// the text; the window makes the data-device request.
    pub(super) fn copy_selection(&mut self) {
        let Some(sel) = self.selection else {
            return;
        };
        let text = self.screen.selection_text(sel.anchor, sel.head);
        if !text.is_empty() {
            self.outbox
                .push(ToWindow::OfferSelection(text.into_bytes()));
        }
    }

    /// Begin a selection at display `(row, col)` with the granularity the click
    /// `count` picks: one click a character, two the word, three the whole
    /// (soft-wrapped) line. The clicked unit is the anchor a drag pivots around. A
    /// plain click paints nothing yet (its selection appears only once the drag
    /// leaves the cell, see [`Self::extend_selection`]), so a single click never
    /// flashes the cell under it; a word/line click is a real selection at once.
    fn begin_selection(&mut self, row: usize, col: usize, count: usize) {
        let abs = self.screen.abs_row(row);
        let (mode, anchor) = match count {
            2 => (SelectMode::Word, self.screen.word_at(abs, col)),
            n if n >= 3 => (SelectMode::Line, self.screen.line_at(abs)),
            _ => (SelectMode::Char, ((abs, col), (abs, col))),
        };
        let epoch = self.screen.row_epoch();
        self.drag = Some(Drag {
            mode,
            anchor,
            epoch,
        });
        self.selection = (mode != SelectMode::Char).then_some(Selection {
            anchor: anchor.0,
            head: anchor.1,
            epoch,
        });
    }

    /// Extend the in-progress drag to display `(row, col)`, snapped to its
    /// granularity: the span runs from the anchored unit to the unit under the
    /// pointer, so a word/line drag never splits a word or line. A character drag
    /// that has not yet left the anchor cell stays empty, so it reads as a click.
    fn extend_selection(&mut self, row: usize, col: usize) {
        let Some(drag) = self.drag else {
            return;
        };
        let abs = self.screen.abs_row(row);
        let unit = match drag.mode {
            SelectMode::Char => ((abs, col), (abs, col)),
            SelectMode::Word => self.screen.word_at(abs, col),
            SelectMode::Line => self.screen.line_at(abs),
        };
        let start = drag.anchor.0.min(unit.0);
        let end = drag.anchor.1.max(unit.1);
        let sel = (drag.mode != SelectMode::Char || start != end).then_some(Selection {
            anchor: start,
            head: end,
            epoch: drag.epoch,
        });
        if self.selection != sel {
            self.selection = sel;
            self.dirty = true;
        }
    }

    /// End a left-drag: a selection with text is offered to both the clipboard and
    /// the primary selection (copy-on-select, matching the `copy-on-select =
    /// clipboard` convention), so Ctrl+V and middle-click both paste it. A plain
    /// click leaves no selection at all. A drag over only blank cells keeps its
    /// highlight (so selecting the empty space below the prompt sticks, the way
    /// wezterm/ghostty leave it until the next click or output clears it) but offers
    /// nothing, since there is no text to copy.
    fn finish_selection(&mut self) {
        let Some(sel) = self.selection else {
            return;
        };
        let text = self.screen.selection_text(sel.anchor, sel.head);
        if text.is_empty() {
            return;
        }
        let bytes = text.into_bytes();
        self.outbox.push(ToWindow::OfferPrimary(bytes.clone()));
        self.outbox.push(ToWindow::OfferSelection(bytes));
    }

    /// Drain published child output through the parser until `max_bytes` or
    /// `deadline` is reached. The tab manager shares those limits across every
    /// core in a turn. Query responses are flushed after each batch, and the end
    /// marker is reported only after the gather queue is empty, so teardown never
    /// drops bytes that were already read. A no-op before a shell is spawned.
    pub(super) fn pump(&mut self, max_bytes: usize, deadline: Instant) -> Result<PumpOutcome> {
        if self.gatherer.is_none() {
            return Ok(PumpOutcome {
                bytes: 0,
                more: false,
                end: None,
            });
        }
        // Drain the wake eventfd once per pump; a publish that races this still
        // re-arms it (empty→nonempty), so the next poll returns and we catch it.
        if let Some(g) = &self.gatherer {
            g.clear_wakeup();
        }

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
        if consumed_any {
            self.after_output();
        }

        // The end marker is withheld until the ready queue drains, so this only
        // fires once every buffered byte has reached the grid.
        let (end, more) = match &self.gatherer {
            Some(g) => (g.completion(), g.has_pending()),
            None => (None, false),
        };
        Ok(PumpOutcome {
            bytes: consumed_bytes,
            more,
            end,
        })
    }

    /// Write back any query replies (DA/DSR) the grid queued while parsing, through
    /// the main PTY handle (writes stay on this thread).
    fn flush_responses(&mut self) -> Result<()> {
        let responses = self.screen.take_responses();
        if !responses.is_empty() {
            if let Some(pty) = &self.pty {
                pty.write_all(&responses)?;
            }
        }
        Ok(())
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
    /// typing, pasting, or pressing End), and only a genuine loss of row identity ends a
    /// selection (see [`Self::prune_selection`]).
    fn after_output(&mut self) {
        self.prune_selection();
        self.refresh_hover();
        self.bump_cursor();
        self.dirty = true;
        self.refresh_title();
        self.track_sync_lock();
        self.flush_clipboard_writes();
        if self.screen.take_bell() {
            self.bell_until = Some(Instant::now() + BELL_FLASH);
        }
    }

    /// Whether the visual bell is mid-flash.
    pub(super) fn bell_flashing(&self) -> bool {
        self.bell_until.is_some_and(|until| Instant::now() < until)
    }

    /// End the flash when its moment has passed, and repaint to take it back off.
    /// Without this the lifted background would simply stay lifted.
    pub(super) fn tick_bell_if_due(&mut self) {
        if self.bell_until.is_some_and(|until| until <= Instant::now()) {
            self.bell_until = None;
            self.dirty = true;
        }
    }

    /// When the flash ends, for the event-loop wait.
    pub(super) fn bell_deadline(&self) -> Option<Instant> {
        self.bell_until
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

    /// Follow the child in and out of synchronized output (`?2026`).
    ///
    /// Entering starts the clock; leaving stops it. The deadline is set once on the way
    /// in and not extended by later output, because the point is to bound how long a
    /// *frame* can take, and a child that keeps writing while holding the lock is exactly
    /// the case that must not be able to hold it forever.
    fn track_sync_lock(&mut self) {
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
    pub(super) fn holds_frame(&self) -> bool {
        self.sync_until.is_some_and(|until| Instant::now() < until)
    }

    /// When to wake and present anyway, for the event loop's wait.
    pub(super) fn sync_deadline(&self) -> Option<Instant> {
        self.sync_until
    }

    /// Drop the selection, and any drag pinned to it, when the rows they name have
    /// stopped meaning what they meant.
    ///
    /// Two ways that happens, and only two. The grid can end the identity regime those
    /// rows were minted in — a reset, a resize, an alt-screen switch, a full-display
    /// erase, or a scroll that renumbered the stream (see
    /// [`RowEpoch`](crate::grid::RowEpoch)). Or the ring can simply outrun them: once the
    /// *last* row of a selection has aged off the front of history, the whole thing is
    /// behind the oldest line the terminal still holds, and there is nothing left to copy.
    /// A selection that has merely scrolled out of *sight* is not pruned — it is still
    /// there, and scrolling back reveals it, as in xterm.
    ///
    /// The drag is checked separately from the selection, and must be: a character drag
    /// that has not yet left the cell it started in has an anchor but no selection to
    /// speak for it, and resuming it against a renumbered grid would drag out a span from
    /// a row the user never touched.
    ///
    /// Everything else — a printing child, a scroll, a program overwriting the cells
    /// underneath — leaves both alone. A selection is a region, not a snapshot: it copies
    /// whatever those cells hold when you ask for them.
    fn prune_selection(&mut self) {
        let epoch = self.screen.row_epoch();
        let live_rows = |cell: (AbsRow, usize)| self.screen.row_exists(cell.0);

        if self
            .drag
            .is_some_and(|d| d.epoch != epoch || !live_rows(d.anchor.1))
        {
            self.drag = None;
        }
        let stale = self
            .selection
            .is_some_and(|sel| sel.epoch != epoch || !live_rows(sel.ordered().1));
        if stale {
            self.selection = None;
            self.drag = None;
            self.dirty = true;
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
        let shown = if self.last_title.is_empty() {
            "bnkterm"
        } else {
            &self.last_title
        };
        self.outbox.push(ToWindow::Title(shown.to_string()));
    }

    /// Compose the window's display list: the visible grid painted at the current
    /// size, cursor on top (solid when focused, hollow when not). Reads the
    /// window-shipped geometry, so it stays a pure function of the grid state.
    pub(super) fn fill_frame_list(&self, out: &mut DisplayList, strings: &mut Vec<String>) {
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
                metrics: self.metrics,
                surface: (self.width as i32, self.height as i32),
                origin: (self.pad, self.origin_y),
                cursor,
                selection: self.selection,
                hover: self.hover,
                scale: self.scale,
                scrollbar: &self.scrollbar,
            },
        );
    }

    /// The column the scrollbar hangs off: the grid's rows, out to the window's right
    /// edge. The painter derives the same rectangle from the same geometry, so the bar is
    /// grabbable exactly where it is drawn.
    fn scroll_column(&self) -> Rect {
        term_render::scroll_column(
            (self.width as i32, self.height as i32),
            (self.pad, self.origin_y),
            self.metrics,
            self.screen.dimensions().1,
        )
    }

    /// The thumb as it is drawn right now, or `None` when there is nothing to scroll.
    fn scroll_thumb(&self) -> Option<Rect> {
        let (content, viewport) = self.screen.scroll_extent();
        let track = term_render::scroll_lane(self.scroll_column(), self.scale).track;
        scroll::thumb(
            track,
            viewport,
            content,
            self.screen.scroll_position(),
            term_render::scroll_min_thumb(self.scale),
        )
    }

    /// Whether this tab has history to scroll at all, which is the only case the bar is
    /// drawn or takes the pointer. False on the alt screen, which keeps no history.
    pub(super) fn scrollable(&self) -> bool {
        let (content, viewport) = self.screen.scroll_extent();
        scroll::scrollable(content, viewport)
    }

    /// Whether a device-pixel point is on the scrollbar, meaning a press there takes its
    /// thumb rather than starting a selection in the grid beneath it.
    pub(super) fn on_scrollbar(&self, x: f32, y: f32) -> bool {
        self.scrollable()
            && term_render::scroll_lane(self.scroll_column(), self.scale)
                .grab
                .contains(x, y)
    }

    /// Follow the pointer with the bar: inside its proximity zone it grows into a slider,
    /// outside it shrinks back. A live drag owns the bar and ignores this.
    pub(super) fn track_scrollbar(&mut self, at: Option<(f32, f32)>) {
        if self.scrollbar.grab().is_some() {
            return;
        }
        let near = self.scrollable()
            && at.is_some_and(|(x, y)| {
                term_render::scroll_lane(self.scroll_column(), self.scale)
                    .zone
                    .contains(x, y)
            });
        if self.scrollbar.set_near(near) {
            self.dirty = true;
        }
    }

    /// A press in the scrollbar's lane: take the thumb under the pointer, or, when the
    /// press landed on the empty track, warp the thumb to it first and take it there
    /// (GNOME's behavior: a click jumps to that spot, and holding turns it into a scrub).
    /// Returns whether the press was the scrollbar's.
    pub(super) fn press_scrollbar(&mut self, x: f32, y: f32) -> bool {
        if !self.on_scrollbar(x, y) {
            return false;
        }
        let Some(thumb) = self.scroll_thumb() else {
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
    pub(super) fn drag_scrollbar(&mut self, y: f32) {
        let (Some(grab), Some(thumb)) = (self.scrollbar.grab(), self.scroll_thumb()) else {
            return;
        };
        let (content, viewport) = self.screen.scroll_extent();
        let track = term_render::scroll_lane(self.scroll_column(), self.scale).track;
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
    pub(super) fn release_scrollbar(&mut self) {
        self.scrollbar.release();
        self.dirty = true;
    }

    /// Light the bar, so every scroll (a wheel notch, a Shift+PageUp, a thumb drag) is
    /// confirmed by it sliding into view and then fading back out.
    fn flash_scrollbar(&mut self) {
        self.scrollbar.flash(Instant::now());
    }

    /// Carry the bar forward a frame. A tab with nothing to scroll has nothing to point
    /// at, so its bar drops out of sight at once rather than fading from a history it no
    /// longer describes (switching to the alt screen is exactly this).
    pub(super) fn tick_scrollbar(&mut self) {
        if self.scrollable() {
            if self.scrollbar.tick(Instant::now()) {
                self.dirty = true;
            }
        } else {
            self.scrollbar.hide();
        }
    }

    /// When the loop should next tick the bar: a step of a fade, or the moment a lit
    /// bar's hold lapses and it starts fading. `None` when it is settled.
    pub(super) fn scrollbar_retry_at(&self) -> Option<Instant> {
        self.scrollbar.retry_at()
    }

    /// Whether the bar is mid-fade, so the next compositor frame should carry it on.
    pub(super) fn scrollbar_animating(&self) -> bool {
        self.scrollbar.animating()
    }

    /// Whether the pointer is resting on a hyperlink. The window reads it to offer the
    /// hand cursor, so the pointer only promises a click will do something when
    /// Ctrl+clicking actually would.
    pub(super) fn hovering_link(&self) -> bool {
        self.hover.is_some()
    }

    /// Flip the blink phase if its deadline has passed (called each loop turn).
    pub(super) fn tick_blink_if_due(&mut self) {
        if self.cursor_blinking() && self.blink_at.is_some_and(|at| at <= Instant::now()) {
            self.tick_blink();
        }
    }

    /// The next cursor-blink deadline, or `None` when the cursor is not blinking;
    /// the window folds it into the event-loop wait.
    pub(super) fn blink_deadline(&self) -> Option<Instant> {
        self.cursor_blinking().then_some(self.blink_at).flatten()
    }

    /// Whether the cursor should be blinking right now: focused, visible, and the
    /// child asked for a blinking style.
    fn cursor_blinking(&self) -> bool {
        self.focused && self.screen.cursor_visible() && self.screen.cursor_blinks()
    }

    /// Flip the blink phase and schedule the next toggle.
    fn tick_blink(&mut self) {
        self.blink_on = !self.blink_on;
        self.blink_at = Some(Instant::now() + BLINK_INTERVAL);
        self.dirty = true;
    }

    /// Reset the cursor to its lit phase and restart the blink timer, so it shows
    /// solid immediately after activity (a keystroke, output) and blinks only when
    /// idle. A no-op's timer stays `None` while unfocused.
    fn bump_cursor(&mut self) {
        self.blink_on = true;
        self.blink_at = self.focused.then(|| Instant::now() + BLINK_INTERVAL);
    }
}

/// Render an absolute path with the user's home directory collapsed to `~`
/// (`~`, then `~/src`, ...), matching the sibling `wezterm.lua` cwd label. Falls
/// back to the plain display when `$HOME` is unset or the path is elsewhere.
fn abbreviate_home(path: &std::path::Path) -> String {
    match std::env::var_os("HOME") {
        Some(home) => abbreviate_under(path, std::path::Path::new(&home)),
        None => path.display().to_string(),
    }
}

/// The `$HOME`-free core of [`abbreviate_home`], taking the home directory
/// explicitly so it is testable without touching the process environment.
fn abbreviate_under(path: &std::path::Path, home: &std::path::Path) -> String {
    match path.strip_prefix(home) {
        Ok(rest) if rest.as_os_str().is_empty() => "~".to_string(),
        Ok(rest) => format!("~/{}", rest.display()),
        Err(_) => path.display().to_string(),
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

/// Fill a fresh grid with a static demo that exercises the phase-2 acceptance
/// list: the 16 ANSI colors, the text styles, DEC box drawing, a CJK wide char,
/// an emoji cluster, a combining mark, and truecolor. Driven through the real
/// `Parser` -> `Screen`, so what the window shows is exactly what the VT pipeline
/// produces, not a bespoke fixture.
fn demo_screen(cols: usize, rows: usize) -> Screen {
    let mut s = Screen::new(cols.max(1), rows.max(1));
    let mut p = Parser::new();
    let mut out: Vec<u8> = Vec::new();

    // Move the cursor to 1-based (row, col) and reset the pen.
    let at = |out: &mut Vec<u8>, row: usize, col: usize| {
        out.extend_from_slice(format!("\x1b[{row};{col}H\x1b[0m").as_bytes());
    };

    at(&mut out, 1, 1);
    out.extend_from_slice(
        b"\x1b[1;36mbnkterm\x1b[0m \x1b[2m-- Wayland + Vulkan terminal, phase 2 static demo\x1b[0m",
    );

    at(&mut out, 3, 1);
    out.extend_from_slice(b"ANSI: ");
    for c in 0..8 {
        out.extend_from_slice(format!("\x1b[4{c}m  ").as_bytes());
    }
    out.extend_from_slice(b"\x1b[0m ");
    for c in 0..8 {
        out.extend_from_slice(format!("\x1b[10{c}m  ").as_bytes());
    }

    at(&mut out, 5, 1);
    out.extend_from_slice(
        b"styles: normal \x1b[1mbold\x1b[0m \x1b[3mitalic\x1b[0m \x1b[4munderline\x1b[0m \
          \x1b[9mstrike\x1b[0m \x1b[7mreverse\x1b[0m \x1b[2mdim\x1b[0m",
    );

    // A box via DEC Special Graphics (ESC ( 0 designates G0), then back to ASCII.
    let box_w = 22.min(cols.saturating_sub(2)).max(2);
    let mid = "q".repeat(box_w.saturating_sub(2));
    at(&mut out, 7, 1);
    out.extend_from_slice(format!("\x1b(0l{mid}k\x1b(B").as_bytes());
    at(&mut out, 8, 1);
    out.extend_from_slice(b"\x1b(0x\x1b(B");
    out.extend_from_slice(b" box drawing (DEC) ");
    at(&mut out, 8, box_w);
    out.extend_from_slice(b"\x1b(0x\x1b(B");
    at(&mut out, 9, 1);
    out.extend_from_slice(format!("\x1b(0m{mid}j\x1b(B").as_bytes());

    at(&mut out, 11, 1);
    out.extend_from_slice(
        "unicode: CJK \u{6f22}\u{5b57}  emoji \u{1F600}\u{1F389}  accent cafe\u{0301}".as_bytes(),
    );

    at(&mut out, 13, 1);
    out.extend_from_slice(b"256-color: ");
    for i in (16..=231).step_by(18) {
        out.extend_from_slice(format!("\x1b[48;5;{i}m  ").as_bytes());
    }
    out.extend_from_slice(b"\x1b[0m");

    at(&mut out, 15, 1);
    out.extend_from_slice(b"truecolor: ");
    for step in 0..24 {
        let r = 255 - step * 10;
        let g = step * 10;
        let b = 128;
        out.extend_from_slice(format!("\x1b[48;2;{r};{g};{b}m ").as_bytes());
    }
    out.extend_from_slice(b"\x1b[0m");

    // Park the cursor somewhere visible for the block-cursor demo.
    at(&mut out, 17, 1);
    out.extend_from_slice(b"prompt$ ");

    p.advance_bytes(&mut s, &out);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn home_is_collapsed_to_tilde_and_other_paths_are_left_alone() {
        let home = Path::new("/home/ada");
        assert_eq!(abbreviate_under(Path::new("/home/ada"), home), "~");
        assert_eq!(
            abbreviate_under(Path::new("/home/ada/projects/bnkterm"), home),
            "~/projects/bnkterm"
        );
        // A path outside home keeps its absolute form; a home-prefixed *name* that
        // is not a path component (…/adam) is not mistaken for it.
        assert_eq!(
            abbreviate_under(Path::new("/etc/hosts"), home),
            "/etc/hosts"
        );
        assert_eq!(
            abbreviate_under(Path::new("/home/adam/x"), home),
            "/home/adam/x"
        );
    }

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
        std::env::set_var("SHELL", "/bin/cat");
        let mut core = TerminalCore::new(false, 40, 10, METRICS, 320, 160, 0);
        if core.spawn_shell().is_err() {
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
            core.fill_frame_list(&mut list, &mut strings);
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
            core.fill_frame_list(&mut list, &mut strings);
            list.iter().any(|c| {
                matches!(
                    c,
                    crate::render::display::DrawCmd::Text { text, .. }
                        if text.contains(crate::term_render::LOCK_GLYPH)
                )
            })
        };
        let mut core = TerminalCore::new(true, 80, 24, METRICS, 640, 384, 0);
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
        let mut core = TerminalCore::new(true, 80, 24, METRICS, 640, 384, 0);
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
    fn the_bell_flashes_and_the_clear_colour_flashes_with_it() {
        // The GPU clear must use the same colour as the display list's base fill, or the
        // flash tears along the edges the list does not cover. Two places computing "the
        // background" independently is how that happens, so they compute it once.
        let mut core = TerminalCore::new(true, 80, 24, METRICS, 640, 384, 0);
        let calm = core.clear_color();
        assert!(!core.bell_flashing());

        core.feed_test_bytes(b"\x07");
        assert!(core.bell_flashing(), "the bell rang");
        assert_ne!(
            core.clear_color(),
            calm,
            "and the clear lifted with the fill"
        );
        assert!(core.bell_deadline().is_some(), "with a deadline to end it");

        // The flash is not a state to be stuck in: when its moment passes it comes off,
        // and the frame that takes it off has to be asked for.
        core.bell_until = Some(Instant::now() - Duration::from_millis(1));
        core.dirty = false;
        core.tick_bell_if_due();
        assert!(!core.bell_flashing());
        assert!(core.dirty, "and it repaints to take the flash back off");
        assert_eq!(core.clear_color(), calm);
    }

    #[test]
    fn a_child_that_dies_mid_frame_cannot_freeze_the_window() {
        // The lock is held by the child, so the child can lose it — crash between the two
        // sequences, or simply never send the second. Without a deadline that is a window
        // that never repaints again, which is far worse than the torn frame the mode
        // exists to prevent. So the deadline is the feature, not a safety net bolted on.
        let mut core = TerminalCore::new(true, 80, 24, METRICS, 640, 384, 0);
        core.feed_test_bytes(b"\x1b[?2026h");
        assert!(core.holds_frame());

        // The child is gone; nothing will ever send `?2026 l`. Pretend the deadline has
        // passed rather than sleeping for it.
        core.sync_until = Some(Instant::now() - Duration::from_millis(1));
        assert!(!core.holds_frame(), "the frame goes up anyway");
        assert!(
            core.sync_deadline().is_some(),
            "and the loop had a deadline to wake on, or nothing would have brought it back"
        );
    }

    #[test]
    fn tab_label_prefers_a_program_title_then_the_cwd_then_shell() {
        let cfg = TabBarConfig::default();
        let mut core = TerminalCore::new(true, 80, 24, METRICS, 640, 384, 0);
        // No title and no known cwd: the bare fallback.
        assert_eq!(core.tab_label(&cfg), "shell");
        // A known cwd (as /proc reports it) shows in place of the fallback. A path
        // outside home stays absolute regardless of the ambient $HOME, so this
        // stays deterministic without touching the process environment (the
        // home-abbreviation itself is covered by `abbreviate_under` above).
        core.cwd = Some(std::path::PathBuf::from("/opt/service"));
        assert_eq!(core.tab_label(&cfg), "/opt/service");
        // A program-set window title wins over the directory.
        core.last_title = "vim README".to_string();
        assert_eq!(core.tab_label(&cfg), "vim README");
    }

    #[test]
    fn a_configured_program_prefixes_the_title_with_its_directory() {
        // The `claude` case: a program that names its own tab still reveals its
        // directory, because it is in the default `path_prefix_programs` list. The
        // prefix is the cwd's final component, and the title follows the separator.
        let cfg = TabBarConfig::default(); // path_prefix_programs = ["claude"]
        let mut core = TerminalCore::new(true, 80, 24, METRICS, 640, 384, 0);
        core.cwd = Some(std::path::PathBuf::from("/home/ada/projects/bnkterm"));
        core.last_title = "fixing the parser".to_string();

        // Foreground is the shell: no prefix, the bare title shows.
        core.foreground = Some("zsh".to_string());
        assert_eq!(core.tab_label(&cfg), "fixing the parser");

        // Foreground is a listed program: the directory leads, then the title.
        core.foreground = Some("claude".to_string());
        assert_eq!(core.tab_label(&cfg), "bnkterm - fixing the parser");

        // Without a title there is nothing to prefix; the cwd shows on its own.
        core.last_title.clear();
        assert_eq!(core.tab_label(&cfg), "/home/ada/projects/bnkterm");
    }

    const METRICS: CellMetrics = CellMetrics {
        size: 16,
        w: 8,
        h: 16,
        baseline: 12,
        ascent: 12,
        descent: 4,
        lock_glyph: true,
    };

    #[test]
    fn demo_screen_fills_a_grid_without_panicking() {
        // The demo drives the real parser into the grid; it must stay in bounds
        // for a range of sizes (including tiny ones the box math must clamp for).
        // On a small grid the content wraps and scrolls, so only its dimensions
        // are asserted; a full grid keeps its title (checked below).
        for &(cols, rows) in &[(80, 24), (40, 12), (10, 4), (2, 2), (1, 1)] {
            let s = demo_screen(cols, rows);
            assert_eq!(s.dimensions(), (cols, rows));
        }
        // At a comfortable size the title sits untouched on the top-left.
        assert_eq!(demo_screen(80, 24).cell(0, 0).rune, 'b');
    }

    #[test]
    fn demo_screen_renders_to_a_nonempty_display_list() {
        // The bring-up seam: a demo grid produces real draw commands (a base
        // fill plus glyph runs), so the window would show content.
        let s = demo_screen(80, 24);
        let metrics = CellMetrics {
            size: 16,
            w: 8,
            h: 16,
            baseline: 12,
            ascent: 12,
            descent: 4,
            lock_glyph: true,
        };
        let bar = Scrollbar::hidden();
        let list = term_render::build_display_list(&term_render::FrameInputs {
            screen: &s,
            bell: false,
            theme: &Theme::default(),
            metrics,
            surface: (80 * 8, 24 * 16),
            origin: (0, 0),
            cursor: CursorRender::default(),
            selection: None,
            hover: None,
            scale: Scale::ONE,
            scrollbar: &bar,
        });
        assert!(list.len() > 1, "more than just the background fill");
    }

    /// A demo core (static grid with content in the top-left) for driving pointer
    /// events through the real selection path.
    fn pointer_core() -> TerminalCore {
        let metrics = CellMetrics {
            size: 16,
            w: 8,
            h: 16,
            baseline: 12,
            ascent: 12,
            descent: 4,
            lock_glyph: true,
        };
        TerminalCore::new(true, 80, 24, metrics, 80 * 8, 24 * 16, 0)
    }

    fn press(core: &mut TerminalCore, button: MouseButton, pressed: bool, col: usize, row: usize) {
        click(core, button, pressed, col, row, 1);
    }

    fn click(
        core: &mut TerminalCore,
        button: MouseButton,
        pressed: bool,
        col: usize,
        row: usize,
        count: usize,
    ) {
        core.apply(ToTerminal::Pointer {
            event: PointerEvent::Button {
                button,
                pressed,
                col,
                row,
                count,
            },
            mods: input::Mods::NONE,
        })
        .unwrap();
    }

    fn drag_to(core: &mut TerminalCore, col: usize, row: usize) {
        core.apply(ToTerminal::Pointer {
            event: PointerEvent::Motion { col, row },
            mods: input::Mods::NONE,
        })
        .unwrap();
    }

    /// The text of the sole primary offer in `core`'s outbox, if any.
    fn primary_offer(core: &mut TerminalCore) -> Option<String> {
        core.take_outbox().into_iter().find_map(|m| match m {
            ToWindow::OfferPrimary(bytes) => String::from_utf8(bytes).ok(),
            _ => None,
        })
    }

    /// A blank (non-demo) core with `text` printed at the top-left, for driving the
    /// hyperlink gestures over content a case chooses.
    fn core_showing(text: &str) -> TerminalCore {
        let metrics = CellMetrics {
            size: 16,
            w: 8,
            h: 16,
            baseline: 12,
            ascent: 12,
            descent: 4,
            lock_glyph: true,
        };
        let mut core = TerminalCore::new(false, 80, 24, metrics, 80 * 8, 24 * 16, 0);
        let mut parser = Parser::new();
        parser.advance_bytes(&mut core.screen, text.as_bytes());
        core
    }

    /// Move the pointer to a cell with a modifier chord held.
    fn hover_at(core: &mut TerminalCore, col: usize, row: usize, mods: input::Mods) {
        core.apply(ToTerminal::Pointer {
            event: PointerEvent::Motion { col, row },
            mods,
        })
        .unwrap();
    }

    /// Press or release a button at a cell with a modifier chord held.
    fn click_mods(
        core: &mut TerminalCore,
        pressed: bool,
        col: usize,
        row: usize,
        mods: input::Mods,
    ) {
        core.apply(ToTerminal::Pointer {
            event: PointerEvent::Button {
                button: MouseButton::Left,
                pressed,
                col,
                row,
                count: 1,
            },
            mods,
        })
        .unwrap();
    }

    /// The URL of the sole open request in `core`'s outbox, if any.
    fn opened_url(core: &mut TerminalCore) -> Option<String> {
        core.take_outbox().into_iter().find_map(|m| match m {
            ToWindow::OpenUrl(url) => Some(url),
            _ => None,
        })
    }

    #[test]
    fn hovering_a_url_marks_it_and_hovering_off_clears_it() {
        // "see https://example.com/a here": the URL sits at cols 4..=24.
        let mut core = core_showing("see https://example.com/a here");
        hover_at(&mut core, 10, 0, input::Mods::NONE);
        assert!(core.hovering_link(), "the pointer rests on the link");
        assert_eq!(
            core.hover,
            Some(CellSpan {
                start: (0, 4),
                end: (0, 24)
            })
        );
        // Off the link, onto the prose after it.
        hover_at(&mut core, 27, 0, input::Mods::NONE);
        assert!(!core.hovering_link(), "the prose is not a link");
        // And off the grid entirely: an underline must not outlive the pointer.
        hover_at(&mut core, 10, 0, input::Mods::NONE);
        assert!(core.hovering_link());
        core.apply(ToTerminal::Pointer {
            event: PointerEvent::Left,
            mods: input::Mods::NONE,
        })
        .unwrap();
        assert!(!core.hovering_link(), "the pointer left the grid");
    }

    #[test]
    fn moving_within_one_link_does_not_redraw() {
        // The hover is repainted only when it *changes*: crossing cells inside one
        // link must not dirty a frame, or a moving pointer would repaint constantly.
        let mut core = core_showing("see https://example.com/a here");
        hover_at(&mut core, 5, 0, input::Mods::NONE);
        assert!(core.dirty, "the link appearing under the pointer redraws");
        core.dirty = false;
        for col in 6..=24 {
            hover_at(&mut core, col, 0, input::Mods::NONE);
        }
        assert!(!core.dirty, "moving along the same link redraws nothing");
        hover_at(&mut core, 25, 0, input::Mods::NONE);
        assert!(core.dirty, "leaving it does");
    }

    #[test]
    fn ctrl_click_opens_a_link_and_starts_no_selection() {
        let mut core = core_showing("see https://example.com/a here");
        click_mods(&mut core, true, 10, 0, input::Mods::CTRL);
        assert_eq!(
            opened_url(&mut core).as_deref(),
            Some("https://example.com/a")
        );
        assert!(core.selection.is_none(), "the click began no selection");
        assert!(core.drag.is_none(), "and no drag is in progress");
    }

    #[test]
    fn a_plain_click_on_a_link_selects_instead_of_opening() {
        // The link only follows on Ctrl+click, so an ordinary click still just places
        // a selection: a stray click can never launch a browser.
        let mut core = core_showing("see https://example.com/a here");
        press(&mut core, MouseButton::Left, true, 10, 0);
        drag_to(&mut core, 12, 0);
        press(&mut core, MouseButton::Left, false, 12, 0);
        let out = core.take_outbox();
        assert!(
            !out.iter().any(|m| matches!(m, ToWindow::OpenUrl(_))),
            "a plain click opens nothing"
        );
        assert!(
            out.iter().any(|m| matches!(m, ToWindow::OfferPrimary(_))),
            "it selects, as it always did"
        );
    }

    #[test]
    fn ctrl_click_off_a_link_still_selects() {
        // Ctrl only means "follow" when there is something to follow; elsewhere the
        // click keeps its ordinary meaning rather than being swallowed.
        let mut core = core_showing("see https://example.com/a here");
        click_mods(&mut core, true, 1, 0, input::Mods::CTRL);
        assert!(core.drag.is_some(), "a selection drag began");
        assert!(opened_url(&mut core).is_none());
    }

    #[test]
    fn a_hostile_scheme_is_never_opened() {
        // The grid's text comes from the child, which can print anything. Neither the
        // hover nor the click may reach a scheme the opener refuses.
        let mut core = core_showing("javascript:alert(1) data:text/html,x");
        hover_at(&mut core, 4, 0, input::Mods::NONE);
        assert!(!core.hovering_link(), "not even underlined");
        click_mods(&mut core, true, 4, 0, input::Mods::CTRL);
        assert!(opened_url(&mut core).is_none(), "and never opened");
    }

    #[test]
    fn a_program_grabbing_the_mouse_keeps_its_clicks() {
        // Under mouse reporting the pointer belongs to the program (vim, tmux): links
        // neither underline nor open, so its own Ctrl+click still reaches it. Shift is
        // the existing escape hatch that hands the gesture back to the terminal.
        let mut core = core_showing("\x1b[?1000hsee https://example.com/a here");
        assert!(core.screen.mouse_mode().reports());

        hover_at(&mut core, 10, 0, input::Mods::NONE);
        assert!(!core.hovering_link(), "the program owns the pointer");
        click_mods(&mut core, true, 10, 0, input::Mods::CTRL);
        assert!(
            opened_url(&mut core).is_none(),
            "the click went to the child"
        );

        // Shift frees the gesture: the link underlines, and Ctrl+Shift+click opens it.
        hover_at(&mut core, 10, 0, input::Mods::SHIFT);
        assert!(core.hovering_link(), "Shift takes the pointer back");
        click_mods(
            &mut core,
            true,
            10,
            0,
            input::Mods::CTRL | input::Mods::SHIFT,
        );
        assert_eq!(
            opened_url(&mut core).as_deref(),
            Some("https://example.com/a")
        );
    }

    #[test]
    fn output_under_a_parked_pointer_re_probes_the_hover() {
        // The pointer sits on a link and the child prints a line, scrolling it up. The
        // underline must follow the text, not stay where the pointer happens to be.
        let mut core = core_showing("https://example.com/a\r\n");
        hover_at(&mut core, 5, 1, input::Mods::NONE);
        assert!(!core.hovering_link(), "row 1 is blank");
        // Print a link onto row 1, where the pointer already rests.
        let mut parser = Parser::new();
        parser.advance_bytes(&mut core.screen, b"https://example.com/b");
        core.after_output();
        assert!(
            core.hovering_link(),
            "the freshly printed link is under the pointer"
        );
        assert_eq!(
            core.hover,
            Some(CellSpan {
                start: (1, 0),
                end: (1, 20)
            })
        );
    }

    #[test]
    fn a_drag_offers_the_selection_to_clipboard_and_primary() {
        // Dragging across the demo title (row 0 begins "bnkterm") offers that text on
        // release to *both* the clipboard and the primary selection, so Ctrl+V and
        // middle-click both paste it.
        let mut core = pointer_core();
        press(&mut core, MouseButton::Left, true, 0, 0);
        drag_to(&mut core, 3, 0);
        press(&mut core, MouseButton::Left, false, 3, 0);
        let out = core.take_outbox();
        let primary = out.iter().find_map(|m| match m {
            ToWindow::OfferPrimary(b) => Some(b.as_slice()),
            _ => None,
        });
        let clipboard = out.iter().find_map(|m| match m {
            ToWindow::OfferSelection(b) => Some(b.as_slice()),
            _ => None,
        });
        assert_eq!(primary, Some(b"bnkt".as_slice()), "primary gets the text");
        assert_eq!(clipboard, Some(b"bnkt".as_slice()), "clipboard too");
    }

    #[test]
    fn a_double_click_selects_the_word() {
        // A double-click inside "bnkterm" selects the whole word, bounded by the
        // trailing space, with no drag needed.
        let mut core = pointer_core();
        click(&mut core, MouseButton::Left, true, 2, 0, 2);
        click(&mut core, MouseButton::Left, false, 2, 0, 1);
        assert_eq!(primary_offer(&mut core).as_deref(), Some("bnkterm"));
    }

    #[test]
    fn a_triple_click_selects_the_line() {
        // A triple-click selects the whole logical line (row 0's title), trailing
        // blanks trimmed.
        let mut core = pointer_core();
        let row = core.screen.abs_row(0);
        let expected = core.screen.selection_text((row, 0), (row, 79));
        click(&mut core, MouseButton::Left, true, 5, 0, 3);
        click(&mut core, MouseButton::Left, false, 5, 0, 1);
        let offered = primary_offer(&mut core);
        assert_eq!(offered.as_deref(), Some(expected.as_str()));
        assert!(
            offered.as_deref().is_some_and(|t| t.starts_with("bnkterm")),
            "the line begins with the title"
        );
    }

    /// The text `core` would copy right now, or `None` with nothing selected.
    fn copied(core: &mut TerminalCore) -> Option<String> {
        core.copy_selection();
        core.take_outbox().into_iter().find_map(|m| match m {
            ToWindow::OfferSelection(bytes) => String::from_utf8(bytes).ok(),
            _ => None,
        })
    }

    /// Double-click the word at display `(row, col)` and leave it selected.
    fn select_word(core: &mut TerminalCore, col: usize, row: usize) {
        click(core, MouseButton::Left, true, col, row, 2);
        click(core, MouseButton::Left, false, col, row, 1);
        core.take_outbox();
    }

    #[test]
    fn a_selection_survives_the_child_printing_under_it() {
        // The other half of the reported bug: selecting text while a program prints used
        // to be impossible, because every burst of output dropped the selection. It now
        // holds, and keeps copying the words it was made over even after they have
        // scrolled off the top of the screen.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        assert_eq!(copied(&mut core).as_deref(), Some("world"));

        // A screenful and a half of output: "hello world" scrolls into history.
        core.feed_test_bytes(b"\r\n");
        for i in 0..30 {
            core.feed_test_bytes(format!("line {i}\r\n").as_bytes());
        }

        assert!(core.selection.is_some(), "output did not evaporate it");
        assert_eq!(
            copied(&mut core).as_deref(),
            Some("world"),
            "and it still names the line it was made over"
        );
    }

    #[test]
    fn a_drag_in_progress_stays_anchored_while_the_child_prints() {
        // The anchor a drag pivots around is absolute too. Here the child scrolls the
        // grid by three rows *between* the press and the drag, so the anchored line moves
        // from display row 5 to display row 2 while the pointer follows it down there. A
        // display-row anchor would still be pinned at row 5 and would drag out three rows
        // of the wrong text; an absolute one is still on the line the user grabbed.
        let mut core = core_showing("\x1b[6;1Halpha beta"); // display row 5
        click(&mut core, MouseButton::Left, true, 0, 5, 1); // press on "alpha"

        core.feed_test_bytes(b"\x1b[24;1H\r\n\r\n\r\n"); // three rows into history

        drag_to(&mut core, 9, 2); // the same line, now three rows higher
        click(&mut core, MouseButton::Left, false, 9, 2, 1);
        assert_eq!(copied(&mut core).as_deref(), Some("alpha beta"));
    }

    #[test]
    fn the_alt_screen_takes_the_selection_with_it() {
        // Rows on the alt screen are a different buffer with no history: an id minted on
        // the primary names nothing there, so the switch ends the epoch and the prune
        // drops the selection rather than highlighting whatever now sits at that row.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        assert!(core.selection.is_some());

        core.feed_test_bytes(b"\x1b[?1049h");

        assert!(core.selection.is_none(), "the selection did not follow");
    }

    #[test]
    fn a_full_screen_erase_takes_the_selection_with_it() {
        // `clear` (ED 2 + ED 3): the text the user picked is gone, so the highlight goes
        // with it rather than sitting over the blank cells that replaced it.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        core.feed_test_bytes(b"\x1b[2J\x1b[3J");
        assert!(core.selection.is_none());
    }

    #[test]
    fn a_resize_takes_the_selection_with_it() {
        // Rows re-index on a resize and we do not re-wrap, so an id minted before it can
        // name a different line (or none). A resize sends no output, so this is the path
        // that proves the prune does not depend on the child saying something.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        core.apply(ToTerminal::Resize {
            cols: 40,
            rows: 12,
            width: 40 * 8,
            height: 12 * 16,
            metrics: METRICS,
            pad: 0,
            origin_y: 0,
            scale: Scale::ONE,
        })
        .unwrap();
        assert!(core.selection.is_none());
    }

    #[test]
    fn a_selection_is_dropped_once_the_ring_outruns_it() {
        // Absolute ids are stable, but not immortal: history is finite, and once the last
        // of a selection's rows has aged off the front of the ring there is nothing left
        // to copy. (A selection that has merely scrolled out of *sight* is kept — see
        // `a_selection_survives_the_child_printing_under_it`.)
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        assert!(core.selection.is_some());

        // More line feeds than the ring is deep, in one burst: row 0 is evicted.
        core.feed_test_bytes(&b"\r\n".repeat(11_000));

        assert!(
            core.selection.is_none(),
            "the row it named has aged out of history"
        );
    }

    #[test]
    fn a_pending_character_drag_is_cancelled_when_its_anchor_dies() {
        // A single click pins an anchor but paints no selection until the drag leaves the
        // cell (so a plain click never flashes). That leaves a window in which the drag
        // holds row ids that nothing else speaks for: if the grid renumbers in it, the
        // drag has to die too, or the next motion resumes from an anchor that now names a
        // row the user never touched and drags out a span of it.
        let mut core = core_showing("alpha beta");
        click(&mut core, MouseButton::Left, true, 0, 0, 1); // press: anchor, no selection
        assert!(core.selection.is_none());
        assert!(core.drag.is_some());

        core.feed_test_bytes(b"\x1b[?1049h"); // alt screen: every id minted is void

        drag_to(&mut core, 5, 3);
        click(&mut core, MouseButton::Left, false, 5, 3, 1);
        assert!(
            core.selection.is_none(),
            "the drag did not resume from a dead anchor"
        );
        assert_eq!(copied(&mut core), None, "and nothing was offered to copy");
    }

    #[test]
    fn a_bare_click_selects_nothing() {
        // Press and release on the same cell (no drag) selects nothing: no primary
        // offer, and the selection is cleared so no stray cell stays highlighted.
        let mut core = pointer_core();
        press(&mut core, MouseButton::Left, true, 0, 0);
        press(&mut core, MouseButton::Left, false, 0, 0);
        assert!(
            !core
                .take_outbox()
                .iter()
                .any(|m| matches!(m, ToWindow::OfferPrimary(_))),
            "a bare click offers nothing"
        );
        assert!(core.selection.is_none(), "and leaves no selection");
    }

    #[test]
    fn a_click_never_highlights_the_clicked_cell() {
        // The single-click "blink": a plain click, even with sub-cell motion that
        // stays in the same cell, must never create a selection, so the clicked cell
        // does not flash the selection colour.
        let mut core = pointer_core();
        press(&mut core, MouseButton::Left, true, 2, 0);
        assert!(core.selection.is_none(), "the press alone shows nothing");
        drag_to(&mut core, 2, 0); // motion that does not leave the cell
        assert!(core.selection.is_none(), "same-cell motion shows nothing");
        press(&mut core, MouseButton::Left, false, 2, 0);
        assert!(core.selection.is_none(), "and the release leaves nothing");
    }

    #[test]
    fn a_drag_over_empty_space_sticks_but_copies_nothing() {
        // Selecting the empty area below the prompt: the drag leaves a live selection
        // (so the highlight sticks after release, matching wezterm/ghostty) yet offers
        // nothing to the clipboard, since blank cells have no text to copy. Row 20 is
        // below the demo's last content row (16), so it is all blank cells.
        let mut core = pointer_core();
        press(&mut core, MouseButton::Left, true, 5, 20);
        drag_to(&mut core, 15, 20);
        press(&mut core, MouseButton::Left, false, 15, 20);
        assert!(
            core.selection.is_some(),
            "the empty-space selection stays highlighted"
        );
        assert!(
            !core
                .take_outbox()
                .iter()
                .any(|m| matches!(m, ToWindow::OfferPrimary(_) | ToWindow::OfferSelection(_))),
            "but nothing is offered to copy"
        );
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
        let mut core = pointer_core();
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
        core.apply(ToTerminal::Key {
            key: input::Key::plain('x'),
            mods: input::Mods::NONE,
            event: input::KeyEvent::Press,
        })
        .unwrap();
        assert_eq!(
            core.screen.view_offset(),
            0,
            "typing returns to the live bottom"
        );
    }

    #[test]
    fn middle_click_requests_a_primary_paste() {
        // With no program grabbing the mouse, a middle-click asks the window to paste
        // the primary selection (the window owns the data device).
        let mut core = pointer_core();
        press(&mut core, MouseButton::Middle, true, 5, 5);
        assert!(
            core.take_outbox()
                .iter()
                .any(|m| matches!(m, ToWindow::PastePrimary)),
            "middle-click requests a primary paste"
        );
    }

    /// A live core with `lines` of output behind an 80x24 screen, so it has history to
    /// scroll. Its geometry is the one [`TerminalCore::new`] starts at: an 8x16 cell, no
    /// padding, unity scale, so the lane lands at a hand-checkable place (see
    /// [`lane_x`]).
    fn scrollable_core(lines: usize) -> TerminalCore {
        let mut core = TerminalCore::new(false, 80, 24, METRICS, 80 * 8, 24 * 16, 0);
        let mut parser = Parser::new();
        for i in 0..lines {
            parser.advance_bytes(&mut core.screen, format!("line {i}\r\n").as_bytes());
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
        let mut parser = Parser::new();
        parser.advance_bytes(&mut core.screen, b"\x1b[?1049h");
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
        core.apply(ToTerminal::Pointer {
            event: PointerEvent::Wheel {
                down: false,
                notches: 1,
                col: 0,
                row: 0,
            },
            mods: input::Mods::NONE,
        })
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

        let mut parser = Parser::new();
        parser.advance_bytes(&mut core.screen, b"\x1b[?1049h");
        core.tick_scrollbar();
        assert_eq!(
            core.scrollbar_retry_at(),
            None,
            "the bar dropped out of sight rather than fading from a history it no longer describes"
        );
    }
}
