//! The terminal core: the half of the app behind the terminal/window seam.
//! [`TerminalCore`] owns the PTY, the VT
//! parser, and the grid, plus everything that is a function of them: the text
//! selection, the cursor blink phase, the theme, and the frame geometry it needs
//! to lay the grid out. It never touches Wayland, xkb, or the GPU.
//!
//! ```text
//!   ToTerminal ─▶ TerminalCore::apply ─┬─▶ PTY write   (input → child)
//!                                      └─▶ grid mutate  (parser, selection, scroll)
//!   PTY read   ─▶ pump_pty ─▶ parser ─▶ grid ─▶ outbox (Title / Closed)
//!   grid state ─▶ fill_frame_list ─▶ DisplayList (pulled by the window each frame)
//! ```
//!
//! The window drives it: it resolves compositor events into [`ToTerminal`]
//! messages (which turn into PTY bytes and grid mutations here), pulls a
//! [`DisplayList`] each frame, and drains the [`ToWindow`] outbox for the few
//! actions only the window can take (set the title, own the clipboard, shut
//! down). Right now the window calls straight into the core on one thread; Stage 2
//! moves the core onto its own thread and swaps these calls for the two channels,
//! with this seam unchanged.

use std::os::fd::RawFd;
use std::time::{Duration, Instant};

use super::message::{PointerEvent, ToTerminal, ToWindow};
use crate::color::Theme;
use crate::error::Result;
use crate::grid::{CursorStyle, Screen};
use crate::input;
use crate::mouse::{self, MouseButton, MouseKind};
use crate::pty::{Pty, ReadOutcome};
use crate::render::display::DisplayList;
use crate::term_render::{self, CellMetrics, CursorRender, CursorShape, Selection};
use crate::vt::Parser;

/// The cursor blink half-period: how long each of the on/off phases lasts.
const BLINK_INTERVAL: Duration = Duration::from_millis(530);

/// The PTY read chunk: large so a burst of output drains in few syscalls.
const PTY_READ_CHUNK: usize = 64 * 1024;

/// Lines the scrollback view moves per wheel notch, and arrows sent per notch
/// when the wheel falls back to arrow keys on the alt screen.
const WHEEL_LINES: usize = 3;

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
    /// The child on the far side of the PTY; `None` in demo mode (and before the
    /// first configure, since the PTY is sized to the granted window).
    pty: Option<Pty>,
    /// Demo mode: a static grid, no shell.
    demo: bool,
    /// Reused PTY read buffer (allocated once, not per drain).
    pty_read_buf: Vec<u8>,
    /// Reused key-encoding buffer (allocated once, not per key press).
    key_buf: Vec<u8>,
    /// The color palette the grid renders with, and the clear color the window
    /// reads for the frame background.
    pub(super) theme: Theme,
    /// Whether the surface holds keyboard focus, so the cursor draws solid when
    /// focused and hollow when not.
    focused: bool,
    /// Cursor blink: the current on/off phase, and when it next toggles (`None`
    /// when not blinking, e.g. unfocused). Activity resets it to on.
    blink_on: bool,
    blink_at: Option<Instant>,
    /// The button held for drag reporting under mouse mode (`None` when none is
    /// down).
    mouse_held: Option<MouseButton>,
    /// The active text selection (a left-drag), or `None`. In display coords.
    selection: Option<Selection>,
    /// Whether a selection drag is in progress (the button is down).
    selecting: bool,
    /// The content changed and a frame should be drawn. The window reads it to
    /// pace repaints, and sets it on a geometry change it drives.
    pub(super) dirty: bool,
    /// Frame geometry, all window-computed and shipped over on a resize: the fixed
    /// cell box, the device surface size the grid lays out in, and the device
    /// padding inset on every side.
    metrics: CellMetrics,
    width: u32,
    height: u32,
    pad: i32,
    /// Outbound messages for the window to act on after the next drain (a title
    /// change, a fresh selection to own, or the child exiting).
    outbox: Vec<ToWindow>,
    /// The child's last-seen window title, so a `Title` is emitted only when it
    /// changes (the sender dedupes, so Stage 2 never spams the channel).
    last_title: String,
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
            pty: None,
            demo,
            pty_read_buf: vec![0u8; PTY_READ_CHUNK],
            key_buf: Vec::new(),
            theme: Theme::default(),
            focused: false,
            blink_on: true,
            blink_at: None,
            mouse_held: None,
            selection: None,
            selecting: false,
            dirty: true,
            metrics,
            width,
            height,
            pad,
            outbox: Vec::new(),
            last_title: String::new(),
        }
    }

    /// Spawn the shell on a PTY sized to the current grid, returning the grid it
    /// was sized to (for the startup log). Only the live path calls this; demo
    /// mode never spawns a child.
    pub(super) fn spawn_shell(&mut self) -> Result<(usize, usize)> {
        let (cols, rows) = self.screen.dimensions();
        self.pty = Some(Pty::spawn(cols, rows)?);
        Ok((cols, rows))
    }

    /// Whether this core is the static demo (no PTY, no shell).
    pub(super) fn is_demo(&self) -> bool {
        self.demo
    }

    /// The PTY master fd for the event-loop `poll`, or `None` before the shell is
    /// spawned (and in demo mode).
    pub(super) fn pty_fd(&self) -> Option<RawFd> {
        self.pty.as_ref().map(Pty::fd)
    }

    /// The frame background as a `0x00RRGGBB`, for the GPU clear (which must match
    /// the display list's own base fill).
    pub(super) fn clear_color(&self) -> u32 {
        self.theme.bg.to_u32()
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
            ToTerminal::Key { key, mods } => {
                let modes = input::Modes::from_screen(&self.screen);
                self.key_buf.clear();
                input::encode(key, mods, modes, &mut self.key_buf);
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
                if self.demo {
                    self.screen = demo_screen(cols, rows);
                } else {
                    self.screen.resize(cols, rows);
                    if let Some(pty) = &self.pty {
                        let _ = pty.resize(cols, rows);
                    }
                }
                self.dirty = true;
                Ok(false)
            }
            ToTerminal::Focus(focused) => {
                self.focused = focused;
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
    /// mouse, or drive local selection / scrollback scroll. The window already
    /// mapped the event to a cell and supplied the modifier chord (Shift forces
    /// local use even while a program is reporting the mouse).
    fn apply_pointer(&mut self, event: PointerEvent, mods: input::Mods) -> Result<()> {
        let reporting = self.screen.mouse_mode().reports() && !mods.contains(input::Mods::SHIFT);
        match event {
            PointerEvent::Button {
                button,
                pressed,
                col,
                row,
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
                    // Local selection: press begins a fresh one, release ends the drag.
                    if pressed {
                        self.selection = Some(Selection {
                            anchor: (row, col),
                            head: (row, col),
                        });
                        self.selecting = true;
                    } else {
                        self.selecting = false;
                    }
                    self.dirty = true;
                }
            }
            PointerEvent::Motion { col, row } => {
                if self.selecting {
                    // Extend the in-progress selection to the pointer's current cell.
                    if let Some(sel) = self.selection.as_mut() {
                        if sel.head != (row, col) {
                            sel.head = (row, col);
                            self.dirty = true;
                        }
                    }
                } else if reporting {
                    // Report motion to a program that asked for it (drag under ?1002,
                    // any move under ?1003).
                    let button = self.mouse_held.unwrap_or(MouseButton::None);
                    self.write_mouse(button, MouseKind::Motion, col, row, mods)?;
                }
            }
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

    /// Read one chunk of the child's output through the parser into the grid. A
    /// closed PTY (the shell exited) queues `Closed`; new output may change the
    /// title, which queues `Title`.
    pub(super) fn pump_pty(&mut self) -> Result<()> {
        // Borrow the PTY and its buffer as distinct fields, so the read does not
        // conflict with the parser/screen borrows below.
        let outcome = match &self.pty {
            Some(pty) => pty.read(&mut self.pty_read_buf)?,
            None => return Ok(()),
        };
        match outcome {
            ReadOutcome::Data(n) => {
                self.parser
                    .advance_bytes(&mut self.screen, &self.pty_read_buf[..n]);
                // Answer any query the child made (DA/DSR): the grid queued the
                // reply bytes; write them back through the PTY.
                let responses = self.screen.take_responses();
                if !responses.is_empty() {
                    if let Some(pty) = &self.pty {
                        pty.write_all(&responses)?;
                    }
                }
                // New output snaps the view to the live bottom (xterm behavior),
                // so a stream of output always shows its latest line.
                self.screen.scroll_view_to_bottom();
                // The cells under any selection just changed meaning; drop it
                // rather than leave a highlight over stale content.
                self.selection = None;
                self.selecting = false;
                self.bump_cursor(); // output shows the cursor solid, then blinks
                self.dirty = true;
                // The child may have set its title via OSC 0/2.
                self.refresh_title();
            }
            ReadOutcome::WouldBlock => {}
            ReadOutcome::Eof => self.outbox.push(ToWindow::Closed),
        }
        Ok(())
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
            shape: cursor_shape(self.screen.cursor_style()),
            visible: self.screen.cursor_visible() && !blinked_off,
            focused: self.focused,
        };
        term_render::build_display_list_into(
            out,
            strings,
            &term_render::FrameInputs {
                screen: &self.screen,
                theme: &self.theme,
                metrics: self.metrics,
                surface: (self.width as i32, self.height as i32),
                origin: (self.pad, self.pad),
                cursor,
                selection: self.selection,
            },
        );
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

/// Map the grid's cursor style (from DECSCUSR) to how the renderer paints it.
fn cursor_shape(style: CursorStyle) -> CursorShape {
    match style {
        CursorStyle::Block => CursorShape::Block,
        CursorStyle::Underline => CursorShape::Underline,
        CursorStyle::Bar => CursorShape::Bar,
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
            ascent: 12,
            descent: 4,
        };
        let list = term_render::build_display_list(
            &s,
            &Theme::default(),
            metrics,
            (80 * 8, 24 * 16),
            (0, 0),
            CursorRender::default(),
            None,
        );
        assert!(list.len() > 1, "more than just the background fill");
    }
}
