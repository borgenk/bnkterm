//! The grid's rows: background runs, fixed-pitch glyph runs and the wide glyphs broken
//! out of them, underline and strike rules, the selection band, and the hovered link's
//! rule. Each is drawn per row, in the stacking order [`crate::term_render`] describes.

use crate::color::Rgb;
use crate::grid::{Attrs, Cell, UnderlineStyle};
use crate::platform::freetype::FaceKey;
use crate::platform::geom::Rect;
use crate::platform::verdict::Verdict;
use crate::render::display::DrawCmd;
use crate::term_render::{
    bell_background, cells_safe, dim, push_owned_cells, style_of, take_string,
    take_string_with_capacity, CellSpan, Painter,
};

/// The selection highlight background: a muted steel blue that stays legible
/// under the default fg.
const SELECTION_BG: Rgb = Rgb::new(0x41, 0x57, 0x76);

/// What a fixed-pitch run is uniform in: its resolved foreground, the face it draws
/// in, and the two rules the painter lays over it. Built by [`Painter::run_style`],
/// which says why both the run's extent and its decorations come from this one value.
///
/// Two fields, and both are cheap on purpose: the painter compares this once per cell in
/// the run scan, so it is a colour and one masked `u16` rather than a set of unpacked
/// flags. Building the unpacked form here cost the frame 14% on the gate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct RunStyle {
    /// The resolved foreground the run's glyphs and rules are drawn in (reverse and dim
    /// already applied, which is why neither bit appears in `key`).
    fg: Rgb,
    /// The face and rule bits, masked and normalised by [`Attrs::run_key`]. Kept packed
    /// rather than unpacked because this is the per-cell comparison; the rules are read
    /// back out of it once per run, in [`Painter::push_decorations`].
    key: Attrs,
}

impl Painter<'_> {
    /// The base background: one fill of the theme background covering the whole
    /// surface (including any partial cell at the right/bottom edge). Every
    /// default-background cell is then just this fill showing through, so the
    /// per-cell background pass emits nothing for them.
    pub(super) fn background(&mut self, surface: (i32, i32)) {
        self.list.push(DrawCmd::Fill {
            rect: Rect {
                x: 0,
                y: 0,
                w: surface.0.max(0),
                h: surface.1.max(0),
            },
            color: bell_background(self.theme, self.bell).to_u32(),
        });
    }

    /// Per-cell backgrounds for one row, coalesced into runs of equal colour so a
    /// solid band is one fill. A run equal to the theme background is skipped (the
    /// base fill already covers it). A wide glyph's spacer carries the leader's
    /// colours, so it coalesces with the leader with no special case here.
    pub(super) fn background_row(&mut self, row: usize, cols: usize) {
        let m = self.metrics;
        // The selected span is one inclusive column range per row, so the per-cell
        // test below stays O(1) and the whole band coalesces into one fill.
        let sel = self.selection_cols(row);
        let selected = |col: usize| sel.is_some_and(|(a, b)| a <= col && col <= b);
        let mut col = 0;
        while col < cols {
            let bg = self.cell_bg(row, col, selected(col));
            let start = col;
            col += 1;
            while col < cols && self.cell_bg(row, col, selected(col)) == bg {
                col += 1;
            }
            if bg != self.theme.bg {
                let x = self.cell_x(start);
                let y = self.cell_y(row);
                self.list.push(DrawCmd::Fill {
                    rect: Rect {
                        x,
                        y,
                        w: (col - start) as i32 * m.w,
                        h: m.h,
                    },
                    color: bg.to_u32(),
                });
            }
        }
    }

    /// The foreground glyphs and text decorations for one row. Consecutive
    /// single-width cells that share a foreground colour, face style, and
    /// underline/strike state batch into one fixed-pitch [`DrawCmd::Cells`] run;
    /// a wide glyph (its leader) breaks out into its own [`DrawCmd::Text`], and a
    /// cell [`cells_safe`] excludes is a run of one, so the run's one-cluster-per-cell
    /// contract holds. See [`crate::term_render`]'s header for why the pitch, not the
    /// advance, drives placement.
    pub(super) fn foreground_row(&mut self, row: usize, cols: usize) {
        let baseline = self.baseline(row);
        let mut col = 0;
        while col < cols {
            let cell = self.cell(row, col);
            if cell.is_wide_spacer() {
                // An orphaned spacer (its leader scrolled off the left) draws
                // nothing; the leader owns the glyph.
                col += 1;
                continue;
            }
            if cell.is_wide_leader() {
                // A wide glyph stands alone at its exact column, so its advance can
                // never shift a following cell.
                self.push_wide_glyph(row, col, cell, baseline);
                col += 2;
                continue;
            }
            col = self.push_run(row, col, cols, cell, baseline);
        }
        // Over the glyphs, like a run's own rules.
        self.push_hover_rule(row);
    }

    /// Emit a fixed-pitch run starting at `col`: the maximal span of single-width,
    /// same-style cells, or just the first when [`cells_safe`] excludes its rune (the
    /// next cell could join its cluster). The [`DrawCmd::Cells`] text is trimmed to the
    /// last inked cell (trailing blanks in the run draw nothing, so carrying them would
    /// only cost allocation and coarsen the damage diff), while underline/strike rules
    /// span the full styled run, because a styled trailing space still shows its
    /// rule (xterm draws it).
    ///
    /// Returns the run's end column (exclusive), which is where the caller's scan
    /// resumes.
    ///
    /// The run's extent is discovered *while* its text is built, in one walk. Finding
    /// the end first and then filling the string reads more clearly, but it is two
    /// passes over the same cells — the extent scan resolving a colour and a style per
    /// cell, the fill re-reading every one of them — to emit a single run. Scanning
    /// once and copying once is the same rule the grid's own hot paths follow.
    fn push_run(
        &mut self,
        row: usize,
        col: usize,
        cols: usize,
        first: Cell,
        baseline: i32,
    ) -> usize {
        let m = self.metrics;
        // The run breaks on foreground, not background, so the first cell's background
        // stands in for the whole run when weighting the glyph anti-aliasing.
        //
        // **A selection band is the case where that is wrong**, and the claim that used to
        // sit here — "reverse video and selection tint uniformly" — was only true of
        // reverse video. A selection covers *part* of a row, and `cell_bg_of`
        // deliberately reports the cell's own background rather than the band's, so glyphs
        // painted over `SELECTION_BG` are weighted against the ground they would have had
        // without it. On a light theme (dark ink on a light page) the selected text comes
        // out visibly heavier than the same text beside it.
        //
        // Left alone on purpose. Fixing it means ending the run at the selection's edges,
        // which is a per-cell selection test on the painter's hottest loop — the exact
        // shape that cost the hovered-link underline 12.9% of the frame before it was
        // moved to a per-row span. It is stable, so
        // the damage diff stays honest, and it is invisible on the default dark theme.
        // Recorded as an open question on `fixed-pitch-run-painting` rather than traded
        // for a measured regression.
        let bg = self.cell_bg_of(first);
        let run = self.run_style(first);
        // A run cannot outlast its row, so ask the pool for that much and let every
        // run settle on one capacity: the pool probes from the back for the first
        // buffer that fits, and uniform sizes are what make that probe hit first try.
        // It is a hint either way — combining marks push more chars than there are
        // cells, so no bound here is exact.
        let mut text = take_string_with_capacity(self.strings, cols - col);
        // Bytes and cell count up to and including the last cell with ink, so a
        // trailing blank never lands in the emitted text.
        let mut inked_bytes = 0;
        let mut inked_cells = 0;
        let limit = if cells_safe(first.rune) {
            cols
        } else {
            col + 1
        };
        let mut end = col;
        while end < limit {
            // The caller already read the first cell to decide this was a run at all.
            let cell = if end == col {
                first
            } else {
                self.cell(row, end)
            };
            if end > col {
                // A wide glyph stands alone and an excluded rune starts a run of its
                // own: past here is a different command.
                if cell.is_wide_leader() || cell.is_wide_spacer() || !cells_safe(cell.rune) {
                    break;
                }
                if self.run_style(cell) != run {
                    break;
                }
            }
            // The one glyph path that does **not** go through [`Shown`], and the reason
            // is measured: building it per cell in this scan cost `frame_tabbar`
            // 11.8 → 12.2-12.6 µs (+3-6%), stable across runs and unmoved by
            // `#[inline(always)]`. This is the same loop the `RunStyle` tombstone is
            // about — per-cell struct construction here does not survive contact with
            // the run scan's code layout. So the rule is spelled out flat, and it is
            // `Shown`'s rule to the letter: a concealed cell contributes a space and
            // none of its marks. `a_concealed_cell_is_still_concealed_under_the_cursor`
            // compares whole display lists, so it pins this copy as well as that one.
            let hidden = cell.attrs.contains(Attrs::HIDDEN);
            text.push(if hidden { ' ' } else { cell.rune });
            let has_marks = !hidden && !self.marks_empty(row, end);
            if has_marks {
                text.extend(self.marks(row, end));
            }
            if !hidden && (cell.rune != ' ' || has_marks) {
                inked_bytes = text.len();
                inked_cells = end - col + 1;
            }
            end += 1;
        }
        let x = self.cell_x(col);
        if inked_cells > 0 {
            text.truncate(inked_bytes);
            push_owned_cells(
                self.list,
                text,
                x,
                baseline,
                inked_cells,
                m,
                FaceKey::Prose {
                    size: m.size,
                    style: style_of(run.key),
                },
                run.fg.to_u32(),
                bg.to_u32(),
            );
        } else {
            // A blank run emits no command, so explicitly return its scratch
            // buffer; otherwise it is dropped and reallocated next frame. Clear it
            // first: the pool's invariant is that its buffers are empty (every
            // buffer salvaged from a retired command enters via `into_text_buf`,
            // which clears it), and the next taker appends without clearing.
            text.clear();
            self.strings.push(text);
        }
        self.push_decorations(run, x, (end - col) as i32 * m.w, baseline);
        end
    }

    /// Everything about a cell that the painter's per-run output depends on, and so
    /// everything a run may not batch across.
    ///
    /// One value serves both questions — where the run ends, and what its single set of
    /// decorations is drawn from — because deriving them separately is how a run comes to
    /// batch across a difference it cannot then draw. What is deliberately *not* here is
    /// what a run can absorb: a cell's background (a run is one span of glyphs over
    /// whatever ground the background pass laid down) and its `HIDDEN` bit (a concealed
    /// cell goes into the text as a space and keeps the run going).
    fn run_style(&self, cell: Cell) -> RunStyle {
        RunStyle {
            fg: self.cell_fg(cell),
            key: cell.attrs.run_key(),
        }
    }

    /// One underline, in the shape the cell asked for (`SGR 4:n`).
    ///
    /// Every style is built from the one primitive the display list already has, a filled
    /// rectangle, rather than by adding a shader or a new `DrawCmd`. A dash is a rect; a
    /// double is two; a curly is a staircase of them.
    ///
    /// The staircase is the only one that costs anything, so it is stepped in whole
    /// *pixels of period*, not per pixel: a wave with a period of roughly a third of a
    /// cell reads as a squiggle at any size a terminal is legible at, and a run of 20
    /// underlined cells costs a few dozen rects rather than several hundred. Underlines
    /// are drawn per *run* of same-attribute cells, not per cell, so an editor squiggling
    /// one diagnostic is a single call with one width.
    fn push_underline(&mut self, style: UnderlineStyle, x: i32, w: i32, baseline: i32, fg: Rgb) {
        let m = self.metrics;
        let color = fg.to_u32();
        let thickness = self.rule_thickness();
        let base = self.underline_rect(x, w, baseline);

        let floor = self.cell_floor(baseline);

        match style {
            UnderlineStyle::Single => self.list.push(DrawCmd::Fill { rect: base, color }),
            UnderlineStyle::Double => {
                // Two rules and a gap as tall as one, raised as far as it takes to end
                // above the cell floor. A descent too shallow for that at full weight
                // (keeping a pixel clear under the text) thins the rules, down to one.
                let t = thickness.min(((m.descent - 1) / 3).max(1));
                let top = base.y.min(floor - 3 * t);
                for y in [top, top + 2 * t] {
                    self.list.push(DrawCmd::Fill {
                        rect: Rect { y, h: t, ..base },
                        color,
                    });
                }
            }
            UnderlineStyle::Dotted | UnderlineStyle::Dashed => {
                // A dash and its gap, tiled across the run. Dotted is a short dash; the
                // difference is only the duty cycle.
                let (dash, gap) = if style == UnderlineStyle::Dotted {
                    (thickness, thickness)
                } else {
                    ((m.w / 3).max(2), (m.w / 4).max(2))
                };
                let mut at = base.x;
                let end = base.x + base.w;
                while at < end {
                    self.list.push(DrawCmd::Fill {
                        rect: Rect {
                            x: at,
                            w: dash.min(end - at),
                            ..base
                        },
                        color,
                    });
                    at += dash + gap;
                }
            }
            UnderlineStyle::Curly => {
                // A triangle wave, drawn as a staircase of short rects whose amplitude is
                // the rule's own thickness. Like the double, the wave is raised to end
                // above the cell floor and thinned where the descent cannot hold it.
                let step = (m.w / 3).max(2);
                let t = thickness.min(((m.descent - 1) / 2).max(1));
                let top = base.y.min(floor - 2 * t);
                let mut at = base.x;
                let end = base.x + base.w;
                let mut up = true;
                while at < end {
                    let y = if up { top } else { top + t };
                    self.list.push(DrawCmd::Fill {
                        rect: Rect {
                            x: at,
                            y,
                            w: step.min(end - at),
                            h: t,
                        },
                        color,
                    });
                    at += step;
                    up = !up;
                }
            }
        }
    }

    /// The box an underline rule occupies: `w` pixels from `x`, just below `baseline`
    /// and never past the cell floor. Shared by the SGR underline and the hovered link's
    /// rule, so the two can never end up at different heights.
    fn underline_rect(&self, x: i32, w: i32, baseline: i32) -> Rect {
        let h = self.rule_thickness();
        let y = baseline + (self.metrics.descent / 2).max(1);
        Rect {
            x,
            y: y.min(self.cell_floor(baseline) - h),
            w,
            h,
        }
    }

    /// The first pixel row of the cell below the one whose text sits on `baseline`.
    /// Nothing drawn for a cell may reach it: the next row paints over it, and a damage
    /// diff of this row alone would leave it behind.
    fn cell_floor(&self, baseline: i32) -> i32 {
        baseline - self.metrics.baseline + self.metrics.h
    }

    /// The underline and strike rules for a run, drawn as solid fills spanning the
    /// run's width in the run's foreground colour.
    fn push_decorations(&mut self, run: RunStyle, x: i32, run_w: i32, baseline: i32) {
        if run.key.contains(Attrs::UNDERLINE) {
            self.push_underline(run.key.underline_style(), x, run_w, baseline, run.fg);
        }
        if run.key.contains(Attrs::STRIKE) {
            self.list.push(DrawCmd::Fill {
                rect: Rect {
                    x,
                    y: baseline - self.metrics.ascent / 3,
                    w: run_w,
                    h: self.rule_thickness(),
                },
                color: run.fg.to_u32(),
            });
        }
    }

    /// How thick a decoration rule is drawn, from the font size. Every rule the painter
    /// draws — underline, strike, the hovered link's — asks here, so they cannot come out
    /// at different weights on the same text.
    fn rule_thickness(&self) -> i32 {
        (self.metrics.size as i32 / 12).max(1)
    }

    /// A wide cell's glyph (base rune plus any combining marks) as a standalone
    /// [`DrawCmd::Text`] at its exact column. A hidden or blank cell draws no glyph, but
    /// still takes its decorations: an underline runs under the cells it covers whether
    /// or not they have ink, the same rule [`Self::push_run`] follows for a styled
    /// trailing space.
    fn push_wide_glyph(&mut self, row: usize, col: usize, cell: Cell, baseline: i32) {
        let m = self.metrics;
        let x = self.cell_x(col);
        let width_cells = 2;
        let (fg, bg) = (self.cell_fg(cell), self.cell_bg_of(cell));
        let shown = self.shown(row, col, cell);
        if shown.inked {
            let mut text = take_string(self.strings);
            text.push(shown.rune);
            if shown.marks {
                text.extend(self.marks(row, col));
            }
            let style = style_of(cell.attrs);
            self.push_glyph_text(x, baseline, width_cells, style, (fg, bg), text);
        }
        // A wide glyph is drawn standalone, so it never rides a run's rule; without
        // this it would be the one gap in an underlined span (an SGR 4 CJK character,
        // or a hovered link with one in its path).
        let run = self.run_style(cell);
        self.push_decorations(run, x, width_cells * m.w, baseline);
    }

    /// The resolved background colour of cell `(row, col)`; `selected` (precomputed
    /// by the caller from the content-trimmed row span) swaps in the selection tint.
    ///
    /// A selected cell takes the tint whatever it holds, so that case resolves nothing
    /// and does not even read the cell. Dim never reaches a background: `resolve`
    /// applies it to the foreground after reverse has already swapped the two.
    fn cell_bg(&self, row: usize, col: usize, selected: bool) -> Rgb {
        if selected {
            return SELECTION_BG;
        }
        self.cell_bg_of(self.cell(row, col))
    }

    /// The inclusive column span of `row` covered by a reading-order [`CellSpan`]: the
    /// first row from its start column to the right edge, whole rows in between, and
    /// the last row from the left edge to its end column. `None` when `row` lies
    /// outside it. One inclusive range per row is what keeps the per-cell test that
    /// consumes it O(1), and both things the painter overlays — the selection band and
    /// the hovered link's rule — are this same shape.
    fn span_cols(&self, span: CellSpan, row: usize) -> Option<(usize, usize)> {
        self.row_in_span(
            row >= span.start.0 && row <= span.end.0,
            row == span.start.0,
            row == span.end.0,
            span.start.1,
            span.end.1,
        )
    }

    /// One row's inclusive column range inside a reading-order span: from the start
    /// column on the row the span opens on, to the end column on the row it closes on,
    /// edge to edge on every row between. `None` when the row is outside it.
    ///
    /// Takes the three row comparisons already made rather than the span itself, because
    /// the painter's two overlays index rows differently (the hovered link by display row,
    /// the selection by [`AbsRow`](crate::grid::AbsRow), since it must not drift when the
    /// child prints) and only the comparisons differ. The clamping rule is the part that
    /// must not, so it lives here once.
    fn row_in_span(
        &self,
        inside: bool,
        at_start: bool,
        at_end: bool,
        start_col: usize,
        end_col: usize,
    ) -> Option<(usize, usize)> {
        if !inside {
            return None;
        }
        let last_col = self.screen.dimensions().0.saturating_sub(1);
        let first = if at_start { start_col } else { 0 };
        let last = if at_end { end_col } else { last_col }.min(last_col);
        Some((first, last))
    }

    /// The inclusive column span of `row` the selection highlights: the whole
    /// geometric span the drag covers, blank cells included, so dragging across the
    /// empty area below the prompt highlights it, the way xterm/wezterm/ghostty do.
    /// The paint is deliberately wider than
    /// [`Screen::selection_text_into`](crate::grid::Screen::selection_text_into) copies:
    /// it shows the drag for feedback while the copy trims trailing blanks.
    ///
    /// The selection is held in absolute rows, so it is resolved against the display
    /// here, at paint time. That is also where the clipping comes from for free: a
    /// selection whose top has scrolled above the window simply has no display row
    /// inside it until the part you can see, so the visible part still highlights.
    ///
    /// Each end is widened to a whole wide character as it is painted. The selection was
    /// stored that way, but output since can have put a wide character across one of its
    /// ends, and half a highlighted glyph reads as a rendering fault.
    fn selection_cols(&self, row: usize) -> Option<(usize, usize)> {
        let (start, end) = self.selection?.ordered();
        let abs = self.screen.abs_row(row);
        let (first, last) = self.row_in_span(
            abs >= start.0 && abs <= end.0,
            abs == start.0,
            abs == end.0,
            start.1,
            end.1,
        )?;
        let last_col = self.screen.dimensions().0.saturating_sub(1);
        let first = match first.checked_sub(1) {
            Some(left) if self.screen.view_cell(row, first).is_wide_spacer() => left,
            _ => first,
        };
        let last = if last < last_col && self.screen.view_cell(row, last).is_wide_leader() {
            last + 1
        } else {
            last
        };
        Some((first, last))
    }

    /// The underline under the hovered hyperlink where it crosses `row`, drawn over its
    /// columns: one solid rule for a link that opens, a dashed one dimmed for a refused
    /// link. Nothing on a row the link does not reach, so an ordinary frame pays a
    /// single `None` check per row for the whole feature.
    ///
    /// It takes the colour of the link's first cell on the row, so the rule reads as
    /// part of the text it underlines rather than as chrome laid over it.
    fn push_hover_rule(&mut self, row: usize) {
        let Some(hover) = self.hover else {
            return;
        };
        let Some((first, last)) = self.span_cols(hover.span, row) else {
            return;
        };
        let fg = self.cell_fg(self.cell(row, first));
        let x = self.cell_x(first);
        let w = (last + 1 - first) as i32 * self.metrics.w;
        let baseline = self.baseline(row);
        match hover.verdict {
            Verdict::Open => {
                let rect = self.underline_rect(x, w, baseline);
                self.list.push(DrawCmd::Fill {
                    rect,
                    color: fg.to_u32(),
                });
            }
            Verdict::Refused(_) => {
                self.push_underline(UnderlineStyle::Dashed, x, w, baseline, dim(fg));
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::color::Color;
    use crate::platform::verdict::Refusal;
    use crate::term_render::rows::*;
    use crate::term_render::tests::*;
    use crate::term_render::*;
    use crate::testutil::{feed, screen_after};

    #[test]
    fn blank_screen_is_just_the_background() {
        let s = Screen::new(4, 2);
        let list = list_of(&s);
        // One base fill covering the whole surface, nothing else.
        assert_eq!(
            fills(&list),
            vec![(
                Rect {
                    x: 0,
                    y: 0,
                    w: 40,
                    h: 40
                },
                Theme::default().bg.to_u32()
            )]
        );
        assert!(cells_runs(&list).is_empty(), "no glyphs on a blank screen");
    }

    #[test]
    fn plain_text_is_one_fixed_pitch_run_per_line() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"hello");
        let runs = cells_runs(&list_of(&s));
        assert_eq!(
            runs,
            vec![(0, M.baseline, "hello".to_string())],
            "the whole line batches into one run at column 0, seated on the baseline"
        );
    }

    #[test]
    fn a_colour_change_splits_the_run() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"ab\x1b[31mcd");
        let runs = cells_runs(&list_of(&s));
        assert_eq!(
            runs,
            vec![
                (0, M.baseline, "ab".to_string()),
                (2 * M.w, M.baseline, "cd".to_string()),
            ],
            "the run breaks where the colour changes; the second starts at its column"
        );
    }

    #[test]
    fn trailing_blanks_emit_no_run() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"hi");
        let runs = cells_runs(&list_of(&s));
        // "hi" then eight blanks: the blanks are default cells, so only "hi" is a
        // run and there are no stray blank runs.
        assert_eq!(runs, vec![(0, M.baseline, "hi".to_string())]);
    }

    #[test]
    fn a_coloured_background_is_a_fill_band() {
        let mut s = Screen::new(6, 1);
        feed(&mut s, b"\x1b[41mXX"); // red background, two cells
        let f = fills(&list_of(&s));
        let red = Color::Ansi(1).resolve(&Theme::default(), Ground::Background);
        // Base fill first, then the two-cell red band.
        assert_eq!(f[0].1, Theme::default().bg.to_u32());
        assert_eq!(
            f[1],
            (
                Rect {
                    x: 0,
                    y: 0,
                    w: 2 * M.w,
                    h: M.h
                },
                red.to_u32()
            ),
            "adjacent same-bg cells coalesce into one band"
        );
    }

    #[test]
    fn reverse_video_swaps_foreground_and_background() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"\x1b[7mA");
        let list = list_of(&s);
        let t = Theme::default();
        // The cell background becomes the default fg (a fill), and the glyph is
        // drawn in the default bg.
        let band = fills(&list)
            .into_iter()
            .find(|(r, _)| r.x == 0 && r.w == M.w)
            .expect("a reverse cell paints its background");
        assert_eq!(band.1, t.fg.to_u32());
        let run = cells_runs(&list).into_iter().next().expect("glyph run");
        // Foreground colour lives on the Cells command, not returned by helper;
        // re-read it here.
        if let DrawCmd::Cells { color, .. } = list
            .iter()
            .find(|c| matches!(c, DrawCmd::Cells { .. }))
            .unwrap()
        {
            assert_eq!(*color, t.bg.to_u32(), "reversed glyph uses the default bg");
        }
        assert_eq!(run.2, "A");
    }

    #[test]
    fn wide_char_is_broken_out_and_the_spacer_is_skipped() {
        let mut s = Screen::new(6, 1);
        feed(&mut s, "a漢b".as_bytes());
        let list = list_of(&s);
        // 'a' at col 0 (run), 漢 standalone Text at col 1 (two cells wide), 'b' at
        // col 3 (the spacer at col 2 drew nothing).
        let texts: Vec<(i32, String)> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { x, text, .. } => Some((*x, text.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec![(M.w, "漢".to_string())]);
        assert_eq!(
            cells_runs(&list),
            vec![
                (0, M.baseline, "a".to_string()),
                (3 * M.w, M.baseline, "b".to_string()),
            ],
            "'b' lands at column 3, past the wide glyph's spacer"
        );
    }

    #[test]
    fn combining_mark_rides_with_its_base_in_the_run() {
        let mut s = Screen::new(6, 1);
        feed(&mut s, "e\u{0301}x".as_bytes()); // é (e + combining acute), then x
        let runs = cells_runs(&list_of(&s));
        assert_eq!(
            runs,
            vec![(0, M.baseline, "e\u{0301}x".to_string())],
            "the mark stays glued to its base; both cells share one run"
        );
    }

    #[test]
    fn underline_adds_a_rule_under_the_run() {
        let mut s = Screen::new(6, 1);
        feed(&mut s, b"\x1b[4mok");
        let list = list_of(&s);
        let t = Theme::default();
        let rule = fills(&list)
            .into_iter()
            .find(|(r, _)| r.y > M.baseline) // below the baseline
            .expect("an underline rule");
        assert_eq!(rule.0.x, 0);
        assert_eq!(rule.0.w, 2 * M.w, "the rule spans the whole run");
        assert_eq!(rule.1, t.fg.to_u32());
    }

    #[test]
    fn every_underline_style_stays_inside_its_row_at_common_sizes() {
        // An underline hangs in the descent, which ends at the cell floor. A rule past the
        // floor paints over the top of the next row and is wiped when that row repaints.
        // Measured on the installed monospace face at the sizes people use, then on
        // shallower descents than it has, where the curly bled too.
        let sizes = [10, 12, 13, 14, 16, 18, 20, 24, 28, 32];
        let fonts = crate::platform::freetype::Fonts::new(&sizes).expect("a monospace face");
        let shallow = [(24, 6), (24, 5), (16, 3), (12, 2)].map(|(size, descent)| {
            let h = size as i32 * 7 / 6;
            CellMetrics {
                size,
                w: size as i32 * 3 / 5,
                h,
                baseline: h - descent,
                ascent: h - descent,
                descent,
                lock_glyph: false,
            }
        });
        let metrics = sizes
            .iter()
            .map(|&size| CellMetrics::from_fonts(&fonts, size))
            .chain(shallow);
        let theme = Theme::default();
        let styles: [(&str, &[u8]); 5] = [
            ("single", b"\x1b[4m"),
            ("double", b"\x1b[4:2m"),
            ("curly", b"\x1b[4:3m"),
            ("dotted", b"\x1b[4:4m"),
            ("dashed", b"\x1b[4:5m"),
        ];
        let mut strays = Vec::new();
        for m in metrics {
            let size = m.size;
            for (style, sgr) in styles {
                // Row 1, so the row below exists to be bled into.
                let mut s = Screen::new(6, 3);
                feed(&mut s, b"\r\n");
                feed(&mut s, sgr);
                feed(&mut s, b"ok");
                let list = build_display_list(&FrameInputs {
                    metrics: m,
                    surface: (6 * m.w, 3 * m.h),
                    ..inputs(&s, &theme)
                });
                let (top, floor) = (m.h, 2 * m.h);
                let rules: Vec<Rect> = fills(&list)
                    .into_iter()
                    .filter(|(_, color)| *color == theme.fg.to_u32())
                    .map(|(rect, _)| rect)
                    .collect();
                assert!(!rules.is_empty(), "{style} at {size}px draws something");
                for r in rules {
                    if r.y < top || r.y + r.h > floor {
                        strays.push(format!(
                            "{style} at {size}px: rows {}..{} outside {top}..{floor} (descent {})",
                            r.y,
                            r.y + r.h,
                            m.descent
                        ));
                    }
                }
            }
        }
        assert!(strays.is_empty(), "{}", strays.join("\n"));
    }

    #[test]
    fn each_underline_style_draws_a_different_shape() {
        // The point of the whole feature: an editor squiggling an error must not come out
        // looking like an editor underlining a link. Every style is built from the one
        // primitive the display list has (a filled rect), so the difference is in how many
        // and where.
        let under = |seq: &[u8]| {
            let mut s = Screen::new(6, 1);
            let mut p = crate::vt::Parser::new();
            p.advance_bytes(&mut s, seq);
            p.advance_bytes(&mut s, b"ok"); // two cells, so a run is 2 * M.w wide
            let t = Theme::default();
            let list = list_of(&s);
            fills(&list)
                .into_iter()
                .filter(|(r, c)| r.y > M.baseline && *c == t.fg.to_u32())
                .map(|(r, _)| r)
                .collect::<Vec<_>>()
        };

        // Single: one rule spanning the run.
        let single = under(b"\x1b[4m");
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].w, 2 * M.w);

        // Double: two rules, one below the other, both spanning the run.
        let double = under(b"\x1b[4:2m");
        assert_eq!(double.len(), 2);
        assert_eq!(double[0].w, 2 * M.w);
        assert_eq!(double[1].w, 2 * M.w);
        assert!(double[1].y > double[0].y, "the second sits below the first");

        // Curly: a staircase, alternating between two heights.
        let curly = under(b"\x1b[4:3m");
        assert!(curly.len() > 2, "a squiggle is more than a line");
        let heights: Vec<i32> = curly.iter().map(|r| r.y).collect();
        assert!(
            heights.windows(2).all(|w| w[0] != w[1]),
            "it goes up and down: {heights:?}"
        );
        assert_eq!(
            heights
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            2,
            "between exactly two heights, so it stays inside the descent"
        );

        // Dotted and dashed: a run of marks with gaps, at one height.
        for seq in [&b"\x1b[4:4m"[..], &b"\x1b[4:5m"[..]] {
            let dashes = under(seq);
            assert!(dashes.len() > 1, "{seq:?} is more than one mark");
            assert!(dashes.iter().all(|r| r.y == dashes[0].y), "{seq:?} is flat");
            assert!(
                dashes.iter().all(|r| r.w < 2 * M.w),
                "{seq:?} has gaps in it"
            );
        }

        // And the cost stays bounded: a squiggle under two cells is a handful of rects,
        // not one per pixel. The damage diff walks this list every painted frame.
        assert!(curly.len() <= 8, "{} rects for two cells", curly.len());
    }

    #[test]
    fn a_change_of_underline_shape_ends_the_run() {
        // Two diagnostics of different kinds side by side: an error squiggled curly, a hint
        // dotted. A run carries *one* set of decorations, drawn from its first cell, so a
        // run that batches across a change of shape draws the whole span in the first
        // shape and the hint comes out squiggled. Same fg, same face, both underlined —
        // every other reason to break the run is absent, which is exactly why the shape has
        // to be one of them.
        let mut s = Screen::new(6, 1);
        feed(&mut s, b"\x1b[4:3maa\x1b[4:4mbb");
        let list = list_of(&s);
        assert_eq!(
            cells_runs(&list)
                .iter()
                .map(|(x, _, text)| (*x, text.clone()))
                .collect::<Vec<_>>(),
            vec![(0, "aa".to_string()), (2 * M.w, "bb".to_string())],
            "the run ends where the shape changes"
        );

        // And the two halves really are drawn as different shapes.
        let t = Theme::default();
        let rules: Vec<Rect> = fills(&list)
            .into_iter()
            .filter(|(r, c)| r.y > M.baseline && *c == t.fg.to_u32())
            .map(|(r, _)| r)
            .collect();
        let curly: Vec<i32> = rules
            .iter()
            .filter(|r| r.x < 2 * M.w)
            .map(|r| r.y)
            .collect();
        let dotted: Vec<i32> = rules
            .iter()
            .filter(|r| r.x >= 2 * M.w)
            .map(|r| r.y)
            .collect();
        assert!(
            curly.len() > 2 && curly.windows(2).all(|w| w[0] != w[1]),
            "the error stays a squiggle: {curly:?}"
        );
        assert!(
            dotted.len() > 1 && dotted.iter().all(|y| *y == dotted[0]),
            "and the hint stays flat: {dotted:?}"
        );
    }

    #[test]
    fn a_nonzero_origin_insets_the_grid_but_not_the_background() {
        let mut s = Screen::new(4, 1);
        feed(&mut s, b"X");
        s.move_to(0, 0); // park the cursor on the glyph
        let t = Theme::default();
        let (ox, oy) = (5, 5);
        let list = build_display_list(&FrameInputs {
            origin: (ox, oy),
            cursor: CursorRender::default(),
            ..inputs(&s, &t)
        });
        // The base fill still covers the whole surface: the inset is a margin of
        // background around the grid, not a smaller canvas.
        assert_eq!(
            fills(&list)[0],
            (
                Rect {
                    x: 0,
                    y: 0,
                    w: 40,
                    h: 20
                },
                t.bg.to_u32()
            )
        );
        // The run, its inverted stamp, and the cursor block all ride the origin: the
        // glyphs at (ox, oy + ascent), the block at (ox, oy).
        assert_eq!(
            cells_runs(&list),
            vec![(ox, oy + M.baseline, "X".to_string()); 2],
            "the run and its inverted stamp are shifted by the content origin"
        );
        let cursor_fill = fills(&list)
            .into_iter()
            .find(|(_, c)| *c == t.cursor.to_u32())
            .expect("a filled cursor block");
        assert_eq!(
            cursor_fill.0,
            Rect {
                x: ox,
                y: oy,
                w: M.w,
                h: M.h
            }
        );
    }

    /// The underline rules in `list`: the fills that sit below the baseline.
    fn rules(list: &DisplayList) -> Vec<Rect> {
        fills(list)
            .into_iter()
            .filter(|(r, _)| r.y > M.baseline)
            .map(|(r, _)| r)
            .collect()
    }

    #[test]
    fn a_hovered_link_is_underlined_across_exactly_its_cells() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"a http://x");
        // The link occupies cols 2..=9; the painter is told so and rules those cells.
        let list = list_hovering(
            &s,
            CellSpan {
                start: (0, 2),
                end: (0, 9),
            },
        );
        assert_eq!(
            rules(&list),
            vec![Rect {
                x: 2 * M.w,
                y: M.baseline + (M.descent / 2).max(1),
                w: 8 * M.w,
                h: (M.size as i32 / 12).max(1),
            }],
            "one rule, starting at the link and spanning exactly its cells"
        );
        // The row's text is untouched: the rule is drawn over the glyphs, so hovering
        // never disturbs how they batch.
        assert_eq!(
            cells_runs(&list),
            cells_runs(&list_of(&s)),
            "the glyph runs are the same hovered or not"
        );
    }

    #[test]
    fn a_hovered_link_is_ruled_over_its_wide_glyphs_too() {
        // One rule over the whole span, so a wide glyph inside a link is covered like
        // any other cell rather than leaving a gap where the run breaks around it.
        let mut s = Screen::new(6, 1);
        feed(&mut s, "ab漢c".as_bytes()); // 漢 is wide: cols 2..=3
        let list = list_hovering(
            &s,
            CellSpan {
                start: (0, 0),
                end: (0, 4),
            },
        );
        assert_eq!(rules(&list).len(), 1, "one rule, not one per run");
        assert_eq!(rules(&list)[0].x, 0);
        assert_eq!(
            rules(&list)[0].w,
            5 * M.w,
            "including the wide glyph's cells"
        );
    }

    #[test]
    fn a_hovered_link_wraps_onto_the_next_row() {
        // A link that runs off the right margin is ruled on both rows: to the edge on
        // the first, from the edge on the second.
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"ab https://x"); // wraps after col 5
        let list = list_hovering(
            &s,
            CellSpan {
                start: (0, 3),
                end: (1, 5),
            },
        );
        let rules = rules(&list);
        assert_eq!(rules.len(), 2, "one rule per row: {rules:?}");
        assert_eq!(rules[0].x, 3 * M.w, "the first row starts at the link");
        assert_eq!(rules[0].w, 3 * M.w, "and runs to the right edge");
        assert_eq!(rules[1].x, 0, "the second row starts at the left edge");
        assert_eq!(rules[1].w, 6 * M.w, "and runs to where the link ends");
    }

    #[test]
    fn a_refused_link_is_ruled_dashed_and_dimmed() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"a http://x");
        let span = CellSpan {
            start: (0, 2),
            end: (0, 9),
        };
        let solid = rules(&list_hovering(&s, span))[0];
        let refused = list_hovering_verdict(&s, span, Verdict::Refused(Refusal::Scheme));
        let dashes: Vec<(Rect, u32)> = fills(&refused)
            .into_iter()
            .filter(|(r, _)| r.y > M.baseline)
            .collect();
        let fg = Color::Default.resolve(&Theme::default(), Ground::Foreground);
        assert!(dashes.len() > 1, "dashed, not one rule: {dashes:?}");
        assert_eq!(dashes[0].0.x, solid.x, "it starts where the link does");
        for (rect, color) in &dashes {
            assert_eq!(*color, dim(fg).to_u32(), "dimmed: {rect:?}");
            assert_eq!((rect.y, rect.h), (solid.y, solid.h), "at the rule's height");
            assert!(
                rect.x >= solid.x && rect.x + rect.w <= solid.x + solid.w,
                "inside the link's cells: {rect:?}"
            );
        }
        let inked: i32 = dashes.iter().map(|(r, _)| r.w).sum();
        assert!(inked < solid.w, "with gaps between the dashes");
        assert_eq!(cells_runs(&refused), cells_runs(&list_of(&s)));
    }

    #[test]
    fn no_hover_means_no_underline() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, b"a http://x");
        assert!(
            rules(&list_of(&s)).is_empty(),
            "an unhovered link is plain text"
        );
    }

    #[test]
    fn a_wide_glyph_takes_its_sgr_underline() {
        // A wide glyph is drawn standalone, outside any run, so it draws its own rules
        // or none at all. Nothing to do with links: `\x1b[4m漢` must underline.
        let mut s = Screen::new(4, 1);
        feed(&mut s, "\x1b[4m漢".as_bytes());
        let rules = rules(&list_of(&s));
        assert_eq!(rules.len(), 1, "the wide glyph is underlined: {rules:?}");
        assert_eq!(rules[0].x, 0);
        assert_eq!(rules[0].w, 2 * M.w, "the rule spans both its cells");
    }

    /// A selection over the display cells `a`..`b`, resolved to absolute rows the way a
    /// pointer drag resolves one.
    fn sel(s: &Screen, a: (usize, usize), b: (usize, usize)) -> Selection {
        Selection {
            anchor: (s.abs_row(a.0), a.1),
            head: (s.abs_row(b.0), b.1),
            epoch: s.row_epoch(),
        }
    }

    /// [`list_of`] with a selection over it.
    fn list_selecting(s: &Screen, selection: Selection) -> DisplayList {
        build_display_list(&FrameInputs {
            selection: Some(selection),
            ..inputs(s, &Theme::default())
        })
    }

    #[test]
    fn selection_paints_the_selected_cells() {
        let mut s = Screen::new(6, 1);
        feed(&mut s, b"abcdef");
        let list = list_selecting(&s, sel(&s, (0, 1), (0, 3)));
        // Columns 1..=3 get the selection background as one band.
        let band = fills(&list)
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("a selection band");
        assert_eq!(
            band.0,
            Rect {
                x: M.w,
                y: 0,
                w: 3 * M.w,
                h: M.h
            }
        );
    }

    #[test]
    fn a_selection_never_paints_half_a_wide_character() {
        // `漢` fills columns 1-2 and `字` columns 4-5. A selection from the right half of
        // the first to the left half of the second paints both whole.
        let mut s = Screen::new(8, 1);
        feed(&mut s, "a漢b字c".as_bytes());
        let list = list_selecting(&s, sel(&s, (0, 2), (0, 4)));
        let band = fills(&list)
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("a selection band");
        assert_eq!(
            band.0,
            Rect {
                x: M.w,
                y: 0,
                w: 5 * M.w,
                h: M.h
            },
            "columns 1 to 5"
        );
    }

    #[test]
    fn selection_highlights_the_full_span_including_trailing_blanks() {
        // "abc" in a 6-wide row, selected edge to edge. The highlight spans all six
        // cells, the trailing blanks (cols 3..=5) included, matching wezterm/ghostty;
        // the copy still trims them (see grid::selection_text_into), so paint is wider.
        let mut s = Screen::new(6, 1);
        feed(&mut s, b"abc");
        let list = list_selecting(&s, sel(&s, (0, 0), (0, 5)));
        let band = fills(&list)
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("a selection band");
        assert_eq!(
            band.0,
            Rect {
                x: 0,
                y: 0,
                w: 6 * M.w,
                h: M.h
            }
        );
    }

    #[test]
    fn selection_over_blank_cells_paints_the_full_band() {
        // Dragging across an empty region highlights the whole geometric span (the
        // "select the empty space below the prompt" behaviour): one full-width band,
        // even though the row holds no glyphs and would copy nothing.
        let s = Screen::new(6, 1);
        let list = list_selecting(&s, sel(&s, (0, 0), (0, 5)));
        let band = fills(&list)
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("a selection band over the blank cells");
        assert_eq!(
            band.0,
            Rect {
                x: 0,
                y: 0,
                w: 6 * M.w,
                h: M.h
            }
        );
    }

    #[test]
    fn a_selection_scrolled_off_the_top_still_paints_the_part_that_shows() {
        // The selection is held in absolute rows and resolved against the display at
        // paint time, so clipping is not a special case: rows above the window simply
        // have no display row inside the span, and the visible tail still highlights.
        let mut s = Screen::new(4, 2);
        feed(&mut s, b"aa\r\nbb"); // both rows live
        let selection = sel(&s, (0, 0), (1, 3)); // both rows selected
        feed(&mut s, b"\r\ncc"); // "aa" scrolls into history, out of the window

        let bands: Vec<Rect> = fills(&list_selecting(&s, selection))
            .into_iter()
            .filter(|(_, c)| *c == SELECTION_BG.to_u32())
            .map(|(r, _)| r)
            .collect();
        assert_eq!(
            bands.len(),
            1,
            "one band, for the one visible row: {bands:?}"
        );
        assert_eq!(
            bands[0],
            Rect {
                x: 0,
                y: 0,
                w: 4 * M.w,
                h: M.h
            },
            "the surviving row of the selection, now at the top of the window"
        );
    }

    #[test]
    fn a_selection_follows_its_rows_as_output_scrolls_the_grid() {
        // The painted band tracks the content: the same selection, after one row scrolls
        // into history, highlights the row where that text now is.
        let mut s = Screen::new(4, 3);
        feed(&mut s, b"aa\r\nbb\r\ncc");
        let selection = sel(&s, (1, 0), (1, 3)); // the middle row, "bb"
        let before = fills(&list_selecting(&s, selection))
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("a band");
        assert_eq!(before.0.y, M.h, "row 1 to start with");

        feed(&mut s, b"\r\ndd"); // everything shifts up one row

        let after = fills(&list_selecting(&s, selection))
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("still a band");
        assert_eq!(after.0.y, 0, "and the highlight moved up with the text");
    }

    #[test]
    fn dropping_a_selection_damages_the_band_it_leaves_behind() {
        // Typing clears the selection, and the rows it covered are usually nowhere near
        // the row the keystroke echoes on. If the diff did not see the band disappear,
        // nothing would damage those rows and the highlight would sit there until
        // something else happened to repaint them.
        let mut s = Screen::new(20, 5);
        feed(&mut s, b"hello world\r\nsecond line");
        let selection = sel(&s, (0, 0), (0, 4)); // "hello", well above any prompt
        let with = list_selecting(&s, selection);
        let without = list_of(&s);
        let (w, h) = (20 * M.w, 5 * M.h);

        let d = crate::render::display::damage(&with, &without, w, h);

        assert!(!d.is_empty(), "the band vanishing has to damage something");
        let band = fills(&with)
            .into_iter()
            .find(|(_, c)| *c == SELECTION_BG.to_u32())
            .expect("a band was painted");
        assert!(
            d.iter()
                .any(|r| r.y <= band.0.y && r.y + r.h >= band.0.y + band.0.h),
            "the damage has to cover the row the band was on: band {:?}, damage {:?}",
            band.0,
            d
        );
    }

    #[test]
    fn a_run_is_seated_on_the_baseline_and_bounded_by_its_cells() {
        // The bug this pins: seating a run on the ascent (14) instead of the baseline
        // (16) rides every glyph 2px up in its cell, so capitals crowd the cell's top
        // edge while descenders float clear of its bottom. Row 1 catches an off-by-one
        // that row 0 would hide, since its cell top is not the surface's.
        let mut s = Screen::new(4, 2);
        feed(&mut s, b"Ty\r\nTy");
        let list = list_of(&s);
        assert_eq!(
            cells_runs(&list),
            vec![
                (0, M.baseline, "Ty".to_string()),
                (0, M.h + M.baseline, "Ty".to_string()),
            ],
            "each run rides its own cell's baseline"
        );
        // A run's bounds is its cell band, padded: a box-drawing glyph fills its cell
        // to the edge, so bounds measured from the ink box would shear its top row.
        let pad_y = (M.size as i32 / 4).max(1);
        let bounds: Vec<Rect> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Cells { bounds, .. } => Some(*bounds),
                _ => None,
            })
            .collect();
        for (row, b) in bounds.iter().enumerate() {
            let cell_top = row as i32 * M.h;
            assert!(
                b.y <= cell_top && b.y + b.h >= cell_top + M.h,
                "row {row}: bounds {b:?} must cover the whole cell band, padded by {pad_y}"
            );
        }
    }

    /// A single-width cell a run cannot batch is a run of one, so an emoji in it is sized
    /// to the cell rather than the em.
    #[test]
    fn a_single_width_cell_a_run_cannot_batch_is_a_run_of_one() {
        let mut s = Screen::new(8, 2);
        // The ZWJ composes onto 🌡's cell and ❤ takes the next one; in one run the two
        // would segment as one cluster.
        feed(
            &mut s,
            "a\u{1F321}\u{200D}\u{2764}b\u{1D400}\r\n".as_bytes(),
        );
        // Under ?2027 a ZWJ sequence of two narrow pictographs is one narrow cell.
        feed(&mut s, "\x1b[?2027h\u{1F321}\u{200D}\u{1F321}x".as_bytes());
        let list = list_of(&s);
        assert!(
            !list.iter().any(|c| matches!(c, DrawCmd::Text { .. })),
            "nothing single-width draws at the em: {list:?}"
        );
        let (top, bottom) = (M.baseline, M.h + M.baseline);
        let runs = cells_runs(&list);
        let runs: Vec<(i32, i32, &str)> = runs
            .iter()
            .map(|(x, baseline, text)| (*x, *baseline, text.as_str()))
            .collect();
        assert_eq!(
            runs,
            [
                (0, top, "a"),
                (M.w, top, "\u{1F321}\u{200D}"),
                (2 * M.w, top, "\u{2764}b"),
                (4 * M.w, top, "\u{1D400}"),
                (0, bottom, "\u{1F321}\u{200D}\u{1F321}"),
                (M.w, bottom, "x"),
            ]
        );
    }

    #[test]
    fn a_scrolled_view_paints_scrollback_not_the_live_screen() {
        // Two rows visible, older lines in history; scrolled up, the painter must
        // draw the history rows, not the live bottom.
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"one\r\ntwo\r\nthree\r\nfour");
        s.scroll_view_up(2); // to the top: "one" over "two"
        let runs = cells_runs(&list_of(&s));
        let texts: Vec<&str> = runs.iter().map(|(_, _, t)| t.as_str()).collect();
        assert!(texts.contains(&"one"), "history row is painted: {texts:?}");
        assert!(
            !texts.contains(&"four"),
            "the live bottom is not: {texts:?}"
        );
    }

    /// The column of the first cell holding anything, for anchoring a placement check.
    fn first_inked_col(screen: &Screen) -> i32 {
        (0..12)
            .find(|&c| screen.cell(0, c).rune != ' ')
            .unwrap_or(0) as i32
    }

    /// Every cell in a painted frame lands on its own column, for the scripts a
    /// fixed-pitch run cannot batch.
    ///
    /// A `Cells` run is drawn by segmenting its text and multiplying the cluster *index*
    /// by the cell width, so a run holding a scalar that a segmenter merges leftward
    /// silently paints every following cell one column short. That is not a subtle
    /// artefact: for Devanagari it was every line of text, with the background fill, the
    /// selection band and the cursor all still on the true grid.
    ///
    /// Checked by walking the display list and asking where each cluster actually lands,
    /// rather than by asserting the internal predicate, so the property survives a change
    /// in how the exclusion is implemented.
    #[test]
    fn every_cell_paints_on_its_own_column_in_every_script() {
        let theme = Theme::default();
        // One line per script, each ending in an ASCII letter: the letter is what visibly
        // walks left when a cluster merge eats a column.
        for (script, text) in [
            ("Devanagari (SpacingMark)", "कीx"),
            ("Devanagari (conjunct)", "क्षx"),
            ("Bengali", "কীx"),
            ("Gurmukhi", "ਕੀx"),
            ("Gujarati", "કીx"),
            ("Oriya", "କୀx"),
            ("Tamil", "கீx"),
            ("Telugu", "కీx"),
            ("Kannada", "ಕೀx"),
            ("Malayalam", "കീx"),
        ] {
            let screen = screen_after(12, 2, text.as_bytes());
            let list = build_display_list(&inputs(&screen, &theme));

            // Where the painter puts each cluster: one entry per column it draws at.
            let mut pens: Vec<i32> = Vec::new();
            for cmd in list.iter() {
                match cmd {
                    DrawCmd::Cells {
                        x, cell_w, text, ..
                    } => {
                        for (i, _) in grapheme::graphemes(text.as_str()).enumerate() {
                            pens.push(x + i as i32 * cell_w);
                        }
                    }
                    DrawCmd::Text { x, .. } => pens.push(*x),
                    _ => {}
                }
            }
            pens.sort_unstable();
            pens.dedup();

            // The grid's own answer, from the cells it filled.
            let cell_w = M.w;
            let expected: Vec<i32> = (0..12)
                .filter(|&c| screen.cell(0, c).rune != ' ')
                .map(|c| pens[0] + (c as i32 - first_inked_col(&screen)) * cell_w)
                .collect();
            assert_eq!(
                pens,
                expected,
                "{script}: {} clusters painted at {pens:?}, grid wants {expected:?}",
                pens.len()
            );
        }
    }
}
