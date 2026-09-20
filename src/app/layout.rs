//! Where the grid and the tab strip sit in the window.
//!
//! One pass of device-pixel arithmetic: a surface size, a scale, and the cell boxes in;
//! the geometry a frame is laid out with out. The window owns the fonts and the scale,
//! so it runs this and every terminal core keeps a copy of the grid's half. Passing the
//! result whole is what keeps a core from pairing the cell count of one pass with the
//! pixel metrics of another.
//!
//! Nothing here touches the compositor or a font, so the placement rules are tested
//! directly.

use crate::config::{TabBarConfig, TabBarPosition};
use crate::platform::geom::Scale;
use crate::term_render::CellMetrics;

/// Blank margin, in logical pixels, between the window edge and the grid on every side
/// (a common terminal default of `padding = 5`). The surface background fills behind the
/// grid, so the inset reads as a border of background.
pub(super) const WINDOW_PADDING: i32 = 5;

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

/// Fit a grid and the tab strip into a device `surface`. `bar` is `None` while the strip
/// is hidden, which leaves the grid everything inside the padding.
pub(super) fn window(
    surface: (u32, u32),
    scale: Scale,
    metrics: CellMetrics,
    label_metrics: CellMetrics,
    bar: Option<&TabBarConfig>,
) -> WindowLayout {
    let (width, height) = surface;
    // Reserve the padding on all sides, so the grid fits inside the margins.
    let pad = scale.px(WINDOW_PADDING);
    let (bar_h, gap) = strip(bar, scale, metrics);
    let usable_w = (width as i32 - 2 * pad).max(0);
    let usable_h = (height as i32 - 2 * pad - bar_h - gap).max(0);
    let (cols, rows) = metrics.columns_rows(usable_w, usable_h);
    // The strip steals its height (and the gap) from the side it sits on: a top bar
    // tucks under the padding and pushes the grid down past the gap; a bottom bar sits
    // flush above the bottom padding, the gap reserved above it.
    let (origin_y, bar_y) = match bar.map(|cfg| cfg.position).filter(|_| bar_h > 0) {
        None => (pad, pad),
        Some(TabBarPosition::Top) => (pad + bar_h + gap, pad),
        Some(TabBarPosition::Bottom) => (pad, height as i32 - pad - bar_h),
    };
    WindowLayout {
        terminal: TerminalGeometry {
            cols,
            rows,
            width,
            height,
            metrics,
            pad,
            origin_y,
            scale,
        },
        label_metrics,
        bar_y,
        bar_h,
    }
}

/// The strip's device-pixel height and the gap it keeps from the grid, or `(0, 0)` when
/// it is hidden. The height floors at one text row, so a label cannot clip on a small
/// configured height or a large font.
fn strip(bar: Option<&TabBarConfig>, scale: Scale, metrics: CellMetrics) -> (i32, i32) {
    let Some(cfg) = bar else {
        return (0, 0);
    };
    (
        scale.px(cfg.height_px as i32).max(metrics.h),
        scale.px(cfg.gap_px as i32),
    )
}

/// The device surface that holds exactly `cols` x `rows` cells with the padding and the
/// strip around them: [`window`] run backwards, for a capture pinned to a cell count.
#[cfg(test)]
pub(super) fn surface_for_cells(
    cols: usize,
    rows: usize,
    scale: Scale,
    metrics: CellMetrics,
    bar: Option<&TabBarConfig>,
) -> (u32, u32) {
    let pad = scale.px(WINDOW_PADDING);
    let (bar_h, gap) = strip(bar, scale, metrics);
    let w = cols as i32 * metrics.w + 2 * pad;
    let h = rows as i32 * metrics.h + 2 * pad + bar_h + gap;
    (w.max(1) as u32, h.max(1) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const METRICS: CellMetrics = CellMetrics {
        size: 16,
        w: 8,
        h: 16,
        baseline: 12,
        ascent: 12,
        descent: 4,
        lock_glyph: true,
    };

    /// A 20px strip with a 6px gap, at `position`.
    fn bar(position: TabBarPosition) -> TabBarConfig {
        TabBarConfig {
            position,
            height_px: 20,
            gap_px: 6,
            ..TabBarConfig::default()
        }
    }

    fn laid_out(surface: (u32, u32), scale: Scale, bar: Option<&TabBarConfig>) -> WindowLayout {
        window(surface, scale, METRICS, METRICS, bar)
    }

    #[test]
    fn a_hidden_strip_leaves_the_grid_everything_inside_the_padding() {
        let layout = laid_out((800, 600), Scale::ONE, None);
        assert_eq!(layout.bar_h, 0);
        assert_eq!(layout.terminal.pad, WINDOW_PADDING);
        assert_eq!(layout.terminal.origin_y, WINDOW_PADDING);
        // 790 x 590 usable, on an 8x16 cell.
        assert_eq!((layout.terminal.cols, layout.terminal.rows), (98, 36));
    }

    #[test]
    fn a_top_strip_pushes_the_grid_down_past_the_gap() {
        let cfg = bar(TabBarPosition::Top);
        let layout = laid_out((800, 600), Scale::ONE, Some(&cfg));
        assert_eq!((layout.bar_y, layout.bar_h), (WINDOW_PADDING, 20));
        assert_eq!(layout.terminal.origin_y, WINDOW_PADDING + 20 + 6);
        // The strip and its gap come out of the rows, not out of the padding.
        assert_eq!(layout.terminal.rows, (600 - 10 - 26) / 16);
    }

    #[test]
    fn a_bottom_strip_sits_flush_above_the_bottom_padding() {
        let cfg = bar(TabBarPosition::Bottom);
        let layout = laid_out((800, 600), Scale::ONE, Some(&cfg));
        assert_eq!(layout.bar_y, 600 - WINDOW_PADDING - 20);
        // The grid still starts under the top padding; only the rows shrink.
        assert_eq!(layout.terminal.origin_y, WINDOW_PADDING);
        assert_eq!(layout.terminal.rows, (600 - 10 - 26) / 16);
    }

    #[test]
    fn the_strip_never_shrinks_below_one_text_row() {
        let cfg = TabBarConfig {
            height_px: 4,
            ..bar(TabBarPosition::Top)
        };
        let layout = laid_out((800, 600), Scale::ONE, Some(&cfg));
        assert_eq!(layout.bar_h, METRICS.h, "a 4px strip would clip its label");
    }

    #[test]
    fn every_device_length_follows_the_scale() {
        let cfg = bar(TabBarPosition::Top);
        let layout = laid_out((1600, 1200), Scale::from_120(240), Some(&cfg));
        assert_eq!(layout.terminal.pad, 2 * WINDOW_PADDING);
        assert_eq!(layout.bar_h, 40);
        assert_eq!(layout.terminal.origin_y, 10 + 40 + 12);
        assert_eq!(layout.terminal.scale, Scale::from_120(240));
    }

    #[test]
    fn a_surface_smaller_than_its_own_padding_still_reports_a_cell() {
        // The subtraction floors at zero rather than wrapping, and a grid is never
        // zero-sized: the window is simply too small to show what it holds.
        let layout = laid_out((4, 4), Scale::ONE, None);
        assert_eq!((layout.terminal.cols, layout.terminal.rows), (1, 1));
    }

    #[test]
    fn a_surface_sized_for_cells_lays_out_as_those_cells() {
        for position in [TabBarPosition::Top, TabBarPosition::Bottom] {
            let cfg = bar(position);
            for strip in [None, Some(&cfg)] {
                let scale = Scale::from_120(180);
                let surface = surface_for_cells(80, 24, scale, METRICS, strip);
                let layout = laid_out(surface, scale, strip);
                assert_eq!(
                    (layout.terminal.cols, layout.terminal.rows),
                    (80, 24),
                    "{position:?} strip: {strip:?}"
                );
            }
        }
    }
}
