//! The typed messages crossing the terminal/window seam.
//!
//! The window thread owns Wayland, xkb, and the GPU; the terminal thread owns the
//! PTY, parser, and grid. The window resolves compositor events into [`ToTerminal`]
//! messages, and the terminal turns them into PTY bytes, grid mutations, and frames
//! (which come back the other way as `ToWindow`). Right now everything runs on one
//! thread (`app::State`) and these are applied inline; Stage 2 sends them across the
//! two threads unchanged. The vocabulary grows one path at a time as each concern
//! moves behind the seam — see the staged plan in the design doc. This starts with
//! the keyboard path.

use crate::input::{Key, Mods};
use crate::mouse::MouseButton;
use crate::term_render::CellMetrics;

/// A pointer event, already mapped to a grid cell by the window (it holds the scale,
/// padding, and metrics to hit-test). The terminal interprets it: a mouse report
/// (using its mouse mode), a local text selection, or a scrollback scroll.
pub enum PointerEvent {
    /// The pointer moved to a cell (a selection drag, or a motion report).
    Motion { col: usize, row: usize },
    /// A mapped button pressed or released at a cell.
    Button {
        button: MouseButton,
        pressed: bool,
        col: usize,
        row: usize,
    },
    /// Whole vertical wheel notches at a cell (the window coalesced the fractional
    /// axis deltas). `down` is the scroll direction.
    Wheel {
        down: bool,
        notches: u32,
        col: usize,
        row: usize,
    },
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
    },
    /// Keyboard focus gained or lost. The window observes it (Wayland); the terminal
    /// needs it because the cursor draws solid when focused, hollow when not.
    Focus(bool),
    /// Clipboard text to paste. The window fetched it (data device); the terminal
    /// normalizes newlines, wraps it in bracketed-paste markers if the program asked
    /// (`?2004`), and writes it to the child.
    Paste(Vec<u8>),
}

/// Terminal → window: the actions only the window can take, produced as the
/// terminal parses output or a copy is made. The window drains these after each
/// PTY pump and turns each into a Wayland request. The vocabulary grows with the
/// seam: the frame itself is still pulled directly in Stage 1 (the window calls
/// [`build_frame_list`](super::terminal::TerminalCore::build_frame_list)), so it
/// is not a message yet; Stage 2 adds `Frame` when the terminal becomes a separate
/// producer.
pub enum ToWindow {
    /// The child's window title changed (OSC 0/2), already mapped to the shown
    /// string (the app name when empty). The window sets the toplevel title.
    Title(String),
    /// A fresh selection's text to own on the clipboard. The window becomes the
    /// data-device selection owner serving these bytes.
    OfferSelection(Vec<u8>),
    /// The child exited (PTY EOF). The window begins shutdown.
    Closed,
}
