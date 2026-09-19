//! The frame geometry a grid is drawn in, as one value.
//!
//! The window owns the fonts and the display scale, so it computes this and every
//! terminal core keeps a copy. Passing it whole is what keeps a core from pairing the
//! cell count of one layout pass with the pixel metrics of another.

use crate::platform::geom::Scale;
use crate::term_render::CellMetrics;

/// One grid's size and the device-pixel box it lays out in.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) struct TerminalGeometry {
    pub(super) cols: usize,
    pub(super) rows: usize,
    /// The device surface the grid lays out in.
    pub(super) width: u32,
    pub(super) height: u32,
    /// The fixed cell box every row is placed on.
    pub(super) metrics: CellMetrics,
    /// The device padding inset on every side.
    pub(super) pad: i32,
    /// Device-pixel y of the grid's first row: `pad` with one tab, `pad + metrics.h`
    /// while the tab bar is visible.
    pub(super) origin_y: i32,
    /// The display scale, so the chrome a core owns (the scrollbar) is sized in the
    /// same device pixels as the grid.
    pub(super) scale: Scale,
}

/// One layout pass over the whole window: the grid's geometry and the strip above or
/// below it. The window computes it; `Tabs` keeps the strip fields and hands the
/// terminal half to every core.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) struct WindowLayout {
    pub(super) terminal: TerminalGeometry,
    /// The cell box tab labels are measured and drawn on.
    pub(super) label_metrics: CellMetrics,
    /// The strip's device-pixel top edge and height; the height is 0 while it is hidden.
    pub(super) bar_y: i32,
    pub(super) bar_h: i32,
}
