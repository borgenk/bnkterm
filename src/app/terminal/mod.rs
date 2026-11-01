//! The per-tab terminal core behind the tabs/window seam. One [`TerminalCore`]
//! exists for each live tab and owns its PTY, VT parser, and grid, plus everything
//! that is a function of them: the text selection, the cursor blink phase, the
//! theme, and the frame geometry it needs to lay the grid out. It never touches
//! Wayland, xkb, or the GPU.
//!
//! ```text
//!   key / paste / pointer ─┬─▶ PTY write    (input → child)
//!   resize / focus         └─▶ grid mutate  (parser, selection, scroll)
//!   PTY master ─▶ gather thread ─▶ pump ─▶ parser ─▶ grid ─▶ outbox (Title / Closed)
//!   grid state ─▶ fill_frame_list ─▶ DisplayList (pulled by the window each frame)
//! ```
//!
//! [`crate::app::tabs::Tabs`] drives each core on the main thread: it routes input to
//! the active core (and a resize to all of them), pulls only the active
//! [`DisplayList`](crate::render::display::DisplayList), and translates each per-core
//! [`ToWindow`] fact. The child's output is drained off-thread by [`crate::gather`], and
//! `pump` consumes the published batches. Only reads move off the main thread; the
//! parser, grid, and every write stay here.
//!
//! This file holds the core's state, its constructor, the labels the tab bar reads, and
//! the demo grid. The operations are split by what they do:
//!
//! ```text
//!   io.rs       the child: spawn and reap, key, focus and paste, the pump and what
//!               its output sets off, the write queue
//!   pointer.rs  the grid under the pointer: selection, mouse reports, hovered links
//!   frame.rs    what is shown: resize, the view's scroll, the scrollbar, the display
//!               list, and the timers that change a frame (blink, sync hold, bell)
//! ```

mod frame;
mod io;
mod pointer;

use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::app::layout::TerminalGeometry;
use crate::app::message::ToWindow;
use crate::app::terminal::frame::BLINK_TOGGLES;
use crate::app::terminal::pointer::Drag;
use crate::color::Theme;
use crate::config::TabBarConfig;
use crate::gather::{GatherEnd, Gatherer};
use crate::grid::{LinkProbe, Screen};
use crate::mouse::MouseButton;
use crate::platform::host::Hostname;
use crate::platform::scroll::Scrollbar;
use crate::program_status::State;
use crate::pty::{Pty, Target, TtyMode};
use crate::term_render::{LinkHover, Selection};
use crate::vt::Parser;

/// Default settle time before a resize is delivered to the child (`TIOCSWINSZ` →
/// `SIGWINCH`). The grid reflows on every configure for smooth visual feedback, but the
/// child is told only once the drag settles: telling it mid-drag makes it redraw and
/// re-emit its OSC 133 prompt marks faster than the reflow can consume them, and the freeze
/// then protects stale rows — the mush. So this is a *debounce*, not a throttle; lowering it
/// makes the prompt snap sooner after the drag, but it must stay above the compositor's
/// configure interval (~16 ms at 60 Hz) or a continuous drag would leak an update through.
/// Override with `BNKTERM_RESIZE_SETTLE_MS` to tune it live.
const WINSIZE_SETTLE_DEFAULT: Duration = Duration::from_millis(30);

/// The settle time in force, from `BNKTERM_RESIZE_SETTLE_MS` or [`WINSIZE_SETTLE_DEFAULT`],
/// clamped to a range that still coalesces a drag at the low end and cannot strand the
/// child at the high end. Read once when a core is built.
fn resize_settle() -> Duration {
    std::env::var("BNKTERM_RESIZE_SETTLE_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|ms| Duration::from_millis(ms.clamp(10, 1000)))
        .unwrap_or(WINSIZE_SETTLE_DEFAULT)
}

/// What one budgeted pass over a core's gather queue observed. The tab manager
/// spends one global budget across these outcomes and owns per-tab teardown.
pub(super) struct PumpOutcome {
    pub(super) bytes: usize,
    pub(super) more: bool,
    pub(super) end: Option<GatherEnd>,
}

/// The terminal half of the app: the PTY, parser, and grid, plus the state that
/// is a pure function of them. The window drives it through [`Self::key`],
/// [`Self::paste`], [`Self::pointer`], [`Self::focus`] and [`Self::resize`], pulls a
/// [`DisplayList`](crate::render::display::DisplayList), and drains the [`ToWindow`]
/// outbox. It holds its own copy of the frame geometry; the window computes it (it owns
/// the fonts and scale) and ships it over on a [`resize`](Self::resize).
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
    /// Where this tab's shell runs, and so where its `file:` links name files: set when
    /// the shell spawns, and kept if the PTY goes, so a link is never judged against the
    /// wrong filesystem.
    shell_target: Target,
    /// Demo mode: a static grid, no shell.
    demo: bool,
    /// Reused key-encoding buffer (allocated once, not per key press).
    key_buf: Vec<u8>,
    /// Bytes owed to the child that the kernel would not take yet, and how far into them
    /// it has got. Everything bound for the PTY goes through here (see
    /// [`Self::pump_writes`]); in the steady state it is written and emptied in the
    /// same call and never holds anything.
    out_buf: Vec<u8>,
    out_head: usize,
    /// Whether the surface holds keyboard focus, so the cursor draws solid when
    /// focused and hollow when not.
    focused: bool,
    /// Cursor blink: the current on/off phase, and when it next toggles (`None`
    /// when not blinking, e.g. unfocused). Activity resets it to on.
    blink_on: bool,
    blink_at: Option<Instant>,
    /// Phase flips left before the blink rests; refilled by any activity.
    blinks_left: u16,
    /// When a settled resize should be pushed to the child (`TIOCSWINSZ`), or `None` when
    /// none is pending. The grid is already reflowed; this debounces only the child's
    /// `SIGWINCH` so a drag does not flood it. See [`resize_settle`].
    winsize_at: Option<Instant>,
    /// The settle interval this core debounces resizes by, resolved once at construction.
    winsize_settle: Duration,
    /// When the current synchronized-output lock gives up and we present regardless.
    /// `None` when the child is not holding a frame.
    sync_until: Option<Instant>,
    /// When the visual bell stops flashing. `None` when it is not ringing.
    bell_until: Option<Instant>,
    /// When the shell was forked, held until its first prompt mark answers and then
    /// dropped — a stopwatch, not a history. The interval it yields is the shell's own
    /// startup, which is the one part of a slow tab the terminal can attribute but not
    /// explain (see [`crate::config::ShellStartupConfig`]).
    ///
    /// A shell that emits no `OSC 133;A` never answers, so this simply stays armed and
    /// nothing is ever reported. That is the intended shape: without the integration the
    /// terminal cannot know when a prompt appeared, and guessing from output bytes would
    /// mean warning about programs that are not prompts at all.
    spawned_at: Option<Instant>,
    /// What the child's tty is doing, as of the last time its output settled (see
    /// [`refresh_tty_mode`](Self::refresh_tty_mode)). Cached rather than probed per
    /// frame because it changes only when the child calls `tcsetattr`, and because a
    /// password prompt is *silent* by definition: the user types and nothing comes
    /// back, so there is no output to re-probe on while the lock must stay up.
    tty_mode: TtyMode,
    /// The button held for drag reporting under mouse mode (`None` when none is
    /// down).
    mouse_held: Option<MouseButton>,
    /// The cell the last motion report named, so an identical one is not sent again.
    ///
    /// A mouse reports position in pixels and the protocol reports it in *cells*, so
    /// under `?1003` (or `?1002` with a button down) a slow crossing of one 8px column
    /// with a 1000 Hz mouse produces dozens of byte-identical reports. Downstream, a TUI
    /// that repaints per motion report repaints tens of times per cell, and a
    /// tmux-over-ssh session pays a round trip for each duplicate. xterm and foot both
    /// track the last reported cell and stay silent until it changes.
    ///
    /// Cleared on a button press or release and when the pointer leaves the grid, so a
    /// program always hears a fresh position at the start of a gesture rather than
    /// inheriting one from the last.
    reported_cell: Option<(usize, usize)>,
    /// The active text selection (a left-drag), or `None`. In absolute rows, so child
    /// output scrolling the grid does not drag it off the text it was made over; see
    /// [`Self::prune_selection`] for the (short) list of things that do end it.
    ///
    /// The region alone, because that is all the painter needs. What a copy hands over
    /// lives in [`Self::selection_text`], and the two are only ever written together by
    /// [`Self::set_selection`].
    selection: Option<Selection>,
    /// The text the grid held under [`Self::selection`] at the moment the *user* last
    /// set that region, and what any later copy hands over. Empty with no selection, or
    /// with one that covers only blanks.
    ///
    /// A selection is a region on a grid the child keeps writing to, so the cells under
    /// it are not the user's to keep: a prompt redraw, a progress bar rewinding with
    /// `\r`, any in-place repaint leaves the highlight where it was and puts something
    /// else beneath it. Reading the grid at copy time meant Ctrl+Shift+C could hand over
    /// text that was never highlighted — the clipboard changing under the user with no
    /// action of theirs. Capturing at selection time makes the highlight and the
    /// clipboard the same promise.
    ///
    /// One buffer for the life of the tab, cleared and refilled rather than replaced, so
    /// a drag across a screenful of cells allocates only while it grows.
    selection_text: String,
    /// The left-drag in progress (the button is down), or `None`. Absolute, like the
    /// selection it pivots: output during a drag must not move the anchor.
    drag: Option<Drag>,
    /// The hyperlink under the pointer: its cells paint underlined, and Ctrl+click
    /// follows it when its verdict lets it open. `None` whenever there is no link to
    /// show — no URL under the pointer, a drag in progress (the gesture is a
    /// selection), a program grabbing the mouse (its clicks are its own), or the
    /// pointer outside the grid.
    hover: Option<LinkHover>,
    /// The cell the pointer sits in while hovering is live, or `None` when it is not.
    /// Two jobs: it makes a motion inside one cell free (the common case, since a
    /// pointer crosses many pixels per cell), and it is the position
    /// [`TerminalCore::refresh_hover`] re-probes from when the text scrolls out from
    /// under a parked pointer.
    hover_cell: Option<(usize, usize)>,
    /// The hovered link's target, and the line previewing it ([`crate::link_preview`])
    /// that shows while Ctrl is held. Both empty off a link, and rebuilt only when the
    /// hovered link or its target changes.
    hover_url: String,
    preview: String,
    /// Reused scratch for the hyperlink probe, so hovering allocates nothing once its
    /// buffers have grown (see [`LinkProbe`]).
    probe: LinkProbe,
    /// The content changed and a frame should be drawn. The window reads it to
    /// pace repaints, and sets it on a geometry change it drives.
    pub(super) dirty: bool,
    /// Frame geometry, window-computed and shipped over on a resize.
    geom: TerminalGeometry,
    /// This tab's overlay scrollbar: it fades in on a scroll and out after it, and grows
    /// into a grabbable slider under the pointer. Per-tab because the scrollback it
    /// describes is: each tab scrolls its own history, and only the visible one animates.
    scrollbar: Scrollbar,
    /// Outbound messages for the window to act on after the next drain (a title
    /// change, a fresh selection to own, or the child exiting).
    outbox: Vec<ToWindow>,
    /// The child's last-seen window title, so a `Title` is emitted only when it
    /// changes.
    last_title: String,
    /// The child's working directory, re-read when the tab's output settles. It
    /// labels the tab when no title or foreground job is known; `None` before the
    /// first read (or on a kernel without `/proc` and no shell report).
    cwd: Option<std::path::PathBuf>,
    /// A foreground job's name (`comm`), re-read alongside `cwd`. Labels a tab with
    /// no explicit title, and selects configured directory prefixes when a title is
    /// set. `None` at the original shell, before the first read, or on a Flatpak host.
    foreground: Option<String>,
}

impl TerminalCore {
    /// Build the core at the window's current geometry; a later [`Self::resize`]
    /// keeps it current. Demo mode starts on a static grid; live mode starts blank
    /// until the shell fills it.
    pub(in crate::app) fn new(demo: bool, geom: TerminalGeometry) -> Self {
        let (cols, rows) = (geom.cols, geom.rows);
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
            shell_target: Target::Local,
            demo,
            key_buf: Vec::new(),
            out_buf: Vec::new(),
            out_head: 0,
            focused: false,
            blink_on: true,
            blink_at: None,
            blinks_left: BLINK_TOGGLES,
            winsize_at: None,
            winsize_settle: resize_settle(),
            sync_until: None,
            bell_until: None,
            spawned_at: None,
            tty_mode: TtyMode::Cooked,
            mouse_held: None,
            reported_cell: None,
            selection: None,
            selection_text: String::new(),
            drag: None,
            hover: None,
            hover_cell: None,
            hover_url: String::new(),
            preview: String::new(),
            probe: LinkProbe::default(),
            dirty: true,
            geom,
            scrollbar: Scrollbar::default(),
            outbox: Vec::new(),
            last_title: String::new(),
            cwd: None,
            foreground: None,
        }
    }

    /// Adopt the configured colours (see [`Screen::set_theme`]).
    pub(in crate::app) fn set_theme(&mut self, theme: Rc<Theme>) {
        self.screen.set_theme(theme);
    }

    /// This machine's name, which decides whether a `file:` link names it.
    pub(in crate::app) fn set_hostname(&mut self, hostname: Hostname) {
        self.screen.set_hostname(hostname);
    }

    /// Whether this core is the static demo (no PTY, no shell).
    pub(in crate::app) fn is_demo(&self) -> bool {
        self.demo
    }

    /// The grid dimensions currently owned by this core.
    pub(in crate::app) fn dimensions(&self) -> (usize, usize) {
        self.screen.dimensions()
    }

    #[cfg(test)]
    pub(in crate::app) fn origin_y(&self) -> i32 {
        self.geom.origin_y
    }

    /// The title shown for this core, with the empty/default title mapped to the
    /// application name just like outbound title messages. This is the *window*
    /// title (the OS caption); the tab bar uses [`tab_label`](Self::tab_label).
    pub(in crate::app) fn title(&self) -> &str {
        if self.last_title.is_empty() {
            "bnkterm"
        } else {
            &self.last_title
        }
    }

    /// The label for this core's tab: the child-set window title when there is one,
    /// otherwise its foreground job, then its working directory (`~`-abbreviated),
    /// falling back to `shell`.
    /// Owned because the directory string is derived, not stored ready to lend.
    ///
    /// One exception keeps directory context for programs that name their own tab:
    /// when a title is set *and* the foreground program is in
    /// [`TabBarConfig::path_prefix_programs`], the label is `"<dir> - <title>"`,
    /// where `<dir>` is the cwd's final component. The prefix keeps the directory
    /// visible when an application's title describes its session.
    /// The directory leads so the end-truncating [`tab_bar::fit_end`](crate::tab_bar)
    /// clips the volatile title first.
    pub(in crate::app) fn tab_label(&self, cfg: &TabBarConfig) -> String {
        if !self.last_title.is_empty() {
            if let Some(dir) = self.prefix_dir(cfg) {
                return format!("{dir} - {}", self.last_title);
            }
            return self.last_title.clone();
        }
        if let Some(program) = &self.foreground {
            return program.clone();
        }
        match &self.cwd {
            Some(path) => abbreviate_home(path),
            None => "shell".to_string(),
        }
    }

    /// The state this tab's status marker shows (`OSC 7501`).
    pub(in crate::app) fn program_status(&self) -> Option<State> {
        self.screen.program_status()
    }

    /// The user has seen this tab: drop the results it was holding (`OSC 7501`).
    pub(in crate::app) fn acknowledge_program_status(&mut self) {
        self.screen.acknowledge_program_status();
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
    /// label might move. Both feed the label and [`TabBarConfig::path_prefix_programs`],
    /// so a change in either can shift what is shown; the manager applies the cheap
    /// rebuild without re-deriving here.
    pub(super) fn refresh_process(&mut self) -> bool {
        // The shell's own word (`OSC 7`) beats /proc, and it is not a matter of taste:
        // /proc tells us the working directory of the process on *this* machine, which is
        // the ssh client when the shell you are actually looking at is three thousand miles
        // away. The one case where knowing the directory is worth the most is the one case
        // /proc gets wrong.
        let cwd = self
            .screen
            .reported_cwd()
            .map(std::path::PathBuf::from)
            .or_else(|| self.pty.as_ref().and_then(Pty::cwd));
        let foreground = self.pty.as_ref().and_then(Pty::foreground_program);
        if cwd == self.cwd && foreground == self.foreground {
            return false;
        }
        self.cwd = cwd;
        self.foreground = foreground;
        true
    }

    /// Test-only visible row text for cross-tab PTY routing assertions.
    #[cfg(test)]
    pub(in crate::app) fn row_string(&self, row: usize) -> String {
        self.screen.row_string(row)
    }

    /// Test-only: how far down the history the view rests, which a scroll moves.
    #[cfg(test)]
    pub(in crate::app) fn scroll_position(&self) -> i32 {
        self.screen.scroll_position()
    }

    /// The palette the grid is currently rendering with.
    pub(in crate::app) fn theme(&self) -> &Theme {
        self.screen.theme()
    }

    /// Drain the outbound messages for the routing layer to act on, leaving the
    /// outbox empty but keeping its capacity for the next batch. Empty in steady
    /// state, so this is allocation-free on the hot path.
    pub(in crate::app) fn drain_outbox(&mut self) -> std::vec::Drain<'_, ToWindow> {
        self.outbox.drain(..)
    }

    /// Take the outbound messages, leaving the outbox empty. A read-and-clear
    /// convenience for tests; production drains in place via [`drain_outbox`](Self::drain_outbox)
    /// to keep the capacity.
    #[cfg(test)]
    pub(in crate::app) fn take_outbox(&mut self) -> Vec<ToWindow> {
        std::mem::take(&mut self.outbox)
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

/// The rows [`demo_screen`] writes into. A capture opens the window this tall plus a
/// blank row under the prompt; `demo_screen_fits_its_declared_extent` holds it here.
#[cfg(test)]
pub(super) const DEMO_ROWS: usize = 23;

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
    out.extend_from_slice(b"\x1b[1;36mbnkterm\x1b[0m \x1b[2m-- Wayland + Vulkan terminal\x1b[0m");

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

    // A two-line prompt, with the cursor parked after the caret for the block demo.
    // Set off from the samples above by more than the one row that separates them.
    at(&mut out, 22, 1);
    out.extend_from_slice(b"\x1b[1;38;5;212mborgenk@bnk-desktop\x1b[0m ~/projects/bnkterm");
    out.extend_from_slice("\x1b[38;5;114m \u{f418} main\x1b[0m".as_bytes());
    at(&mut out, 23, 1);
    out.extend_from_slice(b"\x1b[1;38;5;159m>\x1b[0m ");

    p.advance_bytes(&mut s, &out);
    s
}

#[cfg(test)]
mod tests {
    use crate::app::terminal::*;
    use crate::platform::geom::Scale;
    use crate::term_render::{self, CellMetrics, CursorRender};
    use crate::testutil::METRICS;
    use std::path::Path;

    /// A geometry on [`METRICS`] cells, with no padding at unity scale.
    pub(super) fn geom(cols: usize, rows: usize, width: u32, height: u32) -> TerminalGeometry {
        TerminalGeometry {
            cols,
            rows,
            width,
            height,
            metrics: METRICS,
            pad: 0,
            origin_y: 0,
            scale: Scale::ONE,
        }
    }

    /// An 80x24 core on [`METRICS`] cells with no padding: the demo grid when `demo`,
    /// else a blank one with no child.
    pub(super) fn core_80x24(demo: bool) -> TerminalCore {
        TerminalCore::new(demo, geom(80, 24, 80 * 8, 24 * 16))
    }

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
    fn tab_label_prefers_a_title_then_a_foreground_job_then_the_cwd() {
        let cfg = TabBarConfig::default();
        let mut core = core_80x24(true);
        // No title and no known cwd: the bare fallback.
        assert_eq!(core.tab_label(&cfg), "shell");
        // A known cwd (as /proc reports it) shows in place of the fallback. A path
        // outside home stays absolute regardless of the ambient $HOME, so this
        // stays deterministic without touching the process environment (the
        // home-abbreviation itself is covered by `abbreviate_under` above).
        core.cwd = Some(std::path::PathBuf::from("/opt/service"));
        assert_eq!(core.tab_label(&cfg), "/opt/service");
        // A foreground job supplies its name without sending any title sequences.
        core.foreground = Some("htop".to_string());
        assert_eq!(core.tab_label(&cfg), "htop");
        assert_eq!(core.title(), "bnkterm", "the OS caption stays independent");
        // An explicit title wins over the process name.
        core.last_title = "vim README".to_string();
        assert_eq!(core.tab_label(&cfg), "vim README");
        // Returning to the shell restores the directory after the title is cleared.
        core.last_title.clear();
        core.foreground = None;
        assert_eq!(core.tab_label(&cfg), "/opt/service");
        // Process inspection can succeed even when the directory cannot be read.
        core.cwd = None;
        core.foreground = Some("htop".to_string());
        assert_eq!(core.tab_label(&cfg), "htop");
    }

    #[test]
    fn a_prompt_report_retires_what_the_last_command_left_behind() {
        // The `ssh` case, end to end. The remote shell names the tab after its host and
        // reports its own directory, and both are right while the connection is up: a
        // directory three thousand miles away is exactly what `refresh_process` prefers
        // over /proc. Neither survives `exit`, and nothing in the protocol says so — the
        // local prompt has to say what is true again, which is what the injected
        // `_bnkterm_report` hook emits (see crate::shell_integration).
        let cfg = TabBarConfig::default();
        let mut core = core_80x24(true);
        core.cwd = Some(std::path::PathBuf::from("/opt/service"));

        core.feed_test_bytes(b"\x1b]0;ada@remote: ~/srv\x07\x1b]7;file://remote/srv\x07");
        core.refresh_process();
        core.foreground = Some("ssh".to_string());
        assert_eq!(core.tab_label(&cfg), "ada@remote: ~/srv");
        assert_eq!(
            core.cwd.as_deref(),
            Some(std::path::Path::new("/srv")),
            "the remote's directory is adopted while it is the one you are in"
        );

        // `exit`, then the first local prompt: an empty title and this machine's directory.
        core.feed_test_bytes(b"\x1b]0;\x07\x1b]7;file://local/opt/service\x07");
        core.refresh_process();
        assert_eq!(core.tab_label(&cfg), "/opt/service");
    }

    #[test]
    fn a_configured_program_prefixes_the_title_with_its_directory() {
        // A program in the default prefix list keeps its directory visible beside
        // its own title. The prefix is the cwd's final component, followed by the
        // separator and title.
        let cfg = TabBarConfig::default();
        let mut core = core_80x24(true);
        core.cwd = Some(std::path::PathBuf::from("/home/ada/projects/bnkterm"));
        core.last_title = "fixing the parser".to_string();

        // Foreground is the shell: no prefix, the bare title shows.
        core.foreground = None;
        assert_eq!(core.tab_label(&cfg), "fixing the parser");

        // Foreground is a listed program: the directory leads, then the title.
        core.foreground = Some("claude".to_string());
        assert_eq!(core.tab_label(&cfg), "bnkterm - fixing the parser");

        // Without a title there is nothing to prefix; the program name stands alone.
        core.last_title.clear();
        assert_eq!(core.tab_label(&cfg), "claude");
    }

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

    /// A line past either bound is cropped out of the asset by the capture window.
    #[test]
    fn demo_screen_fits_its_declared_extent() {
        let (cols, rows) = (crate::app::CAPTURE_COLS, DEMO_ROWS + 6);
        let s = demo_screen(cols + 20, rows);
        let blank = |row: usize, col: usize| s.cell(row, col) == crate::grid::Cell::BLANK;

        assert!(
            (0..cols).any(|col| !blank(DEMO_ROWS - 1, col)),
            "the prompt is the last row the demo writes"
        );
        for row in DEMO_ROWS..rows {
            assert!(
                (0..cols).all(|col| blank(row, col)),
                "row {row} is past DEMO_ROWS and must be empty"
            );
        }
        for col in cols..cols + 20 {
            assert!(
                (0..rows).all(|row| blank(row, col)),
                "column {col} is past the capture width and must be empty"
            );
        }
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
            preview: None,
            scale: Scale::ONE,
            scrollbar: &bar,
        });
        assert!(list.len() > 1, "more than just the background fill");
    }
}
