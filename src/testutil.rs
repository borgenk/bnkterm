//! Fixtures the unit tests share: the cell metrics most of them lay text out on, and the
//! parser run that turns bytes into a screen.

use crate::grid::Screen;
use crate::term_render::CellMetrics;
use crate::vt::Parser;

/// An 8x16 cell with the baseline at 12: the metrics most tests lay text out on.
pub(crate) const METRICS: CellMetrics = CellMetrics {
    size: 16,
    w: 8,
    h: 16,
    baseline: 12,
    ascent: 12,
    descent: 4,
    lock_glyph: true,
};

/// Run `bytes` into `screen` through a fresh parser.
pub(crate) fn feed(screen: &mut Screen, bytes: &[u8]) {
    Parser::new().advance_bytes(screen, bytes);
}

/// A `cols` x `rows` screen after `bytes`.
pub(crate) fn screen_after(cols: usize, rows: usize, bytes: &[u8]) -> Screen {
    let mut screen = Screen::new(cols, rows);
    feed(&mut screen, bytes);
    screen
}
