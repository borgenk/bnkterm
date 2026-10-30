//! The plate that previews a hovered link's target while Ctrl is held, drawn over the grid
//! and its scrollbar on the shared [`Panel`].

use crate::panel::Panel;
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::term_render::{preview_room, Painter};
use crate::width;

impl Painter<'_> {
    /// The hovered link's preview `text` on a plate just below the link's last row, or
    /// just above its first when the grid has no room below, cut with `…` to the grid's
    /// width. It starts under the link when the link sits on one row, and at the grid's
    /// left edge for a link that wraps. A grid too small to hold the plate shows none.
    ///
    /// ```text
    ///   see https://github.com/x/y for details
    ///       ┌────────────────────────────────────┐
    ///       │ github.com  https://github.com/x/y │
    ///       └────────────────────────────────────┘
    /// ```
    pub(super) fn push_link_preview(&mut self, text: &str, cols: usize, rows: usize) {
        let Some(hover) = self.hover else {
            return;
        };
        let m = self.metrics;
        if m.w <= 0 || m.h <= 0 {
            return;
        }
        let panel = Panel::snug(m);
        let grid_w = cols as i32 * m.w;
        let room = preview_room(cols, m);
        let (top, bottom) = (self.cell_y(0), self.cell_y(rows));
        let panel_h = panel.height(1);
        if room == 0 || panel_h > bottom - top {
            return;
        }
        let (shown, cut) = fit_cells(text, room);
        let shown_cols = width::text_cells(shown) as i32;
        let panel_w = panel.width(shown_cols + i32::from(cut));

        let below = self.cell_y(hover.span.end.0 + 1);
        let above = self.cell_y(hover.span.start.0) - panel_h;
        let y0 = if below + panel_h <= bottom || above < top {
            below.min(bottom - panel_h).max(top)
        } else {
            above
        };
        let left = if hover.span.start.0 == hover.span.end.0 {
            self.cell_x(hover.span.start.1)
        } else {
            self.cell_x(0)
        };
        let x0 = left
            .min(self.cell_x(0) + grid_w - panel_w)
            .max(self.cell_x(0));

        let rect = Rect {
            x: x0,
            y: y0,
            w: panel_w,
            h: panel_h,
        };
        let mut pen = panel.open(
            self.list,
            self.strings,
            rect,
            self.theme.fg,
            Panel::plate(self.theme),
        );
        pen.write(0, 0, shown);
        if cut {
            pen.write(0, shown_cols, "…");
        }
    }
}

/// The longest start of `text` that fits `cols` cells with room left for a `…`, and
/// whether it had to be cut; all of it when it fits whole. Measured the way
/// [`width::text_cells`] measures, so the cut lands where the drawn text would overflow,
/// and read only until it overflows, however long the text.
fn fit_cells(text: &str, cols: usize) -> (&str, bool) {
    let mut used = 0;
    let mut cut = 0;
    for (at, cluster) in grapheme::graphemes(text) {
        if used < cols {
            cut = at;
        }
        used += usize::from(width::cluster_width(cluster)).max(1);
        if used > cols {
            return (text.get(..cut).unwrap_or_default(), true);
        }
    }
    (text, false)
}

#[cfg(test)]
mod tests {

    use crate::term_render::tests::*;
    use crate::term_render::*;
    use crate::testutil::feed;

    /// [`list_of`] with a hovered link that opens over it, previewed as `text`, the way a
    /// frame draws while Ctrl is held.
    fn list_previewing(s: &Screen, span: CellSpan, text: &str) -> DisplayList {
        build_display_list(&FrameInputs {
            hover: Some(LinkHover {
                span,
                verdict: Verdict::Open,
            }),
            preview: Some(text),
            ..inputs(s, &Theme::default())
        })
    }

    /// The preview's plate, and the text drawn after it.
    fn preview_plate(list: &[DrawCmd]) -> Option<(Rect, String)> {
        let at = list
            .iter()
            .rposition(|c| matches!(c, DrawCmd::RoundRect { .. }))?;
        let DrawCmd::RoundRect { rect, .. } = list[at] else {
            return None;
        };
        let text = cells_runs(&list[at..])
            .into_iter()
            .map(|(_, _, t)| t)
            .collect();
        Some((rect, text))
    }

    #[test]
    fn the_preview_sits_under_the_hovered_link_or_above_it_on_the_last_row() {
        let mut s = Screen::new(40, 4);
        feed(&mut s, b"see https://github.com/x");
        let line = "github.com  https://github.com/x";
        let span = CellSpan {
            start: (0, 4),
            end: (0, 23),
        };
        let (rect, text) = preview_plate(&list_previewing(&s, span, line)).unwrap();
        assert_eq!(text, line);
        assert_eq!(
            (rect.x, rect.y),
            (4 * M.w, M.h),
            "on the row below, under it"
        );
        assert!(
            preview_plate(&list_hovering(&s, span)).is_none(),
            "nothing while Ctrl is up"
        );

        let span = CellSpan {
            start: (3, 4),
            end: (3, 23),
        };
        let (rect, _) = preview_plate(&list_previewing(&s, span, line)).unwrap();
        assert_eq!(
            rect.y + rect.h,
            3 * M.h,
            "above the link's row, with none below"
        );
    }

    #[test]
    fn a_preview_is_cut_to_the_grid_and_kept_inside_it() {
        let s = Screen::new(20, 3);
        let span = CellSpan {
            start: (0, 15),
            end: (0, 19),
        };
        let long = "refused: it hides a character  https://example.com/a";
        let (rect, text) = preview_plate(&list_previewing(&s, span, long)).unwrap();
        let kept = text.strip_suffix('…').expect("cut with an ellipsis");
        assert!(long.starts_with(kept), "{text}");
        assert_eq!(
            width::text_cells(&text),
            19,
            "every cell inside the padding"
        );
        assert!(rect.x >= 0 && rect.x + rect.w <= 20 * M.w, "{rect:?}");

        let (rect, text) = preview_plate(&list_previewing(&s, span, "github.com")).unwrap();
        assert_eq!(text, "github.com");
        assert_eq!(rect.x + rect.w, 20 * M.w, "moved left to fit, not cut");
    }

    #[test]
    fn a_grid_too_small_for_the_plate_shows_no_preview() {
        let span = CellSpan {
            start: (0, 0),
            end: (0, 0),
        };
        for (cols, rows) in [(1, 4), (40, 1)] {
            let s = Screen::new(cols, rows);
            assert!(
                preview_plate(&list_previewing(&s, span, "github.com")).is_none(),
                "{cols}x{rows}"
            );
        }
    }
}
