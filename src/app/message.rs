//! The typed messages crossing the terminal/window seam.
//!
//! The window side owns Wayland, xkb, and the GPU; each terminal core owns one PTY,
//! parser, and grid. [`super::tabs::Tabs`] sits between them: active-only input,
//! focus, title, blink, and frames route to one core; resize and pumping fan out to
//! all cores. `Title` and `Closed` are per-tab facts until `Tabs` translates them
//! into window actions. Everything except PTY reads stays on `app::State`'s main
//! thread; gather threads publish only byte batches.

use crate::input::{Key, Mods};
use crate::mouse::MouseButton;
use crate::platform::geom::Scale;
use crate::term_render::CellMetrics;

/// A pointer event, already mapped to a grid cell by the window (it holds the scale,
/// padding, and metrics to hit-test). The terminal interprets it: a mouse report
/// (using its mouse mode), a local text selection, or a scrollback scroll.
pub enum PointerEvent {
    /// The pointer moved to a cell (a selection drag, or a motion report).
    Motion { col: usize, row: usize },
    /// A mapped button pressed or released at a cell. `count` is the multi-click
    /// count on a left press (1 character, 2 word, 3 line); it is 1 otherwise.
    Button {
        button: MouseButton,
        pressed: bool,
        col: usize,
        row: usize,
        count: usize,
    },
    /// Whole vertical wheel notches at a cell (the window coalesced the fractional
    /// axis deltas). `down` is the scroll direction.
    Wheel {
        down: bool,
        notches: u32,
        col: usize,
        row: usize,
    },
    /// The pointer is no longer over the grid: it left the surface, or crossed into
    /// the tab strip. There is no cell, because there is no longer a cell under it.
    /// The terminal drops the hovered hyperlink; nothing else is pointer-positional.
    Left,
}

/// Window → terminal: input and geometry the terminal turns into PTY bytes, grid
/// mutations, and frames.
pub enum ToTerminal {
    /// A resolved key press. The window did the keycode → keysym mapping (xkb lives
    /// with the Wayland keyboard); the terminal encodes it under the current terminal
    /// modes (which it owns) and writes the bytes to the child.
    Key { key: Key, mods: Mods },
    /// A pointer event mapped to a cell, plus the modifier chord (Shift forces local
    /// use even while a program is grabbing the mouse).
    Pointer { event: PointerEvent, mods: Mods },
    /// A new grid size in cells, plus the frame geometry that produced it. The
    /// window owns the fonts and scale, so it computes the cell `metrics`, the
    /// device surface `width`/`height`, and the device `pad`; the terminal resizes
    /// the grid, pushes the size to the child (`TIOCSWINSZ`), and keeps the geometry
    /// copies it lays frames out with.
    Resize {
        cols: usize,
        rows: usize,
        width: u32,
        height: u32,
        metrics: CellMetrics,
        pad: i32,
        /// Device-pixel y coordinate of the grid's first row. This is `pad` with
        /// one tab and `pad + metrics.h` while the tab bar is visible.
        origin_y: i32,
        /// The display scale, so the core can size the chrome it owns (the scrollbar)
        /// in the same device pixels the rest of this geometry is in.
        scale: Scale,
    },
    /// Keyboard focus gained or lost. The window observes it (Wayland); the terminal
    /// needs it because the cursor draws solid when focused, hollow when not.
    Focus(bool),
    /// Clipboard text to paste. The window fetched it (data device); the terminal
    /// normalizes newlines, wraps it in bracketed-paste markers if the program asked
    /// (`?2004`), and writes it to the child.
    Paste(Vec<u8>),
}

/// Terminal → tabs/window: per-core facts produced while parsing output or making
/// a copy. `Tabs` translates title/close facts according to active-tab state, and
/// the window turns the routed actions into Wayland requests. Frames remain pulled
/// directly from the active core because all state except PTY reads shares the main
/// thread; no frame message or channel is needed.
pub enum ToWindow {
    /// This child's title changed (OSC 0/2), already mapped to the shown string.
    /// `Tabs` forwards it only when this is the active child.
    Title(String),
    /// A fresh selection's text to own on the clipboard (Ctrl+Shift+C). The window
    /// becomes the data-device selection owner serving these bytes.
    OfferSelection(Vec<u8>),
    /// A fresh selection's text to own on the primary selection (copy-on-select).
    /// The window becomes the primary-selection owner; a middle-click elsewhere then
    /// pastes it, the Linux convention.
    OfferPrimary(Vec<u8>),
    /// A middle-click asked to paste the primary selection. The window owns the data
    /// device, so it does the receive and feeds the bytes back as a [`ToTerminal::Paste`].
    PastePrimary,
    /// A Ctrl+click landed on a hyperlink: hand it to the user's default handler. The
    /// core found it in the grid and vetted its scheme (see
    /// [`crate::platform::browser::can_open`]); the window spawns the opener, since
    /// launching a process is a window-side concern like every other action here.
    OpenUrl(String),
    /// The window should shut down: every tab is gone. A core signals its own
    /// child's exit to [`super::tabs::Tabs`] through the pump's stream-end, not
    /// this message; `Tabs` removes that tab and synthesizes `Closed` only once
    /// the last one is gone.
    Closed,
}
