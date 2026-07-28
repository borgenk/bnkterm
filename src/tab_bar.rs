//! Pure tab-bar layout, sanitization, painting, and hit testing.
//!
//! Titles originate in OSC 0/2 and are therefore child-controlled. Layout works
//! in terminal cells, segments titles by grapheme cluster, replaces controls and
//! zero-width clusters, and truncates only at cluster boundaries. Painting then
//! emits ordinary display-list commands into the same pooled strings as the grid.
//!
//! The look mirrors the sibling `wezterm.lua`: every tab is one *equal* width
//! (clamped into the config's `[min_width, max_width]`), left-aligned so a few
//! tabs read as neat fixed blocks rather than stretching edge to edge, its label
//! centered with the numeric index dropped. A label too long for its tab is cut
//! from the *end* (the start stays put) and its tail dissolves into the tab
//! instead of ending on an abrupt ellipsis. A one-pixel divider sits between two
//! inactive tabs so equal blocks sharing the strip background stay separable.
//!
//! ```text
//!   pad │  tab 0   │  tab 1  │ tab 2 │            (strip background)          │
//!       │ centered │ cente…  │       │  <- fade    <- slack past the last tab
//!       └ divider between two inactive tabs; hidden beside the active block
//! ```
//!
//! The dissolve is a ramp on the label's *ink* ([`Fade`], applied per pixel by the
//! renderer), not a mix of its colour toward the tab background, and the difference
//! is visible. A colour mix never arrives at the background, so it leaves a stain
//! where the label was cut, and on a light tab (the active block) a dark stain is
//! read as dirt rather than as a fade. Worse, the glyph anti-aliasing is weighted by
//! each run's contrast with its background, so mixing a tail *toward* that
//! background made it report a contrast it did not have: the faded glyphs kept
//! roughly twice the anti-aliased ink of the solid ones and came out fat as well as
//! muddy. Ramping ink leaves both the colour and the stroke weight alone and lands
//! on exactly the tab background, whatever colour that is.

use std::ops::Range;

use crate::config::{TabBarConfig, TabColors};
use crate::platform::freetype::{FaceKey, FontStyle, Fonts};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::render::display::{DisplayList, DrawCmd, Fade};
use crate::term_render::{self, CellMetrics};

/// One breathing-room column each side of a label, so text never butts against a
/// tab edge or a divider. Dropped on tabs too narrow to spare it.
const SIDE_PAD: usize = 1;

/// The tab width at or above which the [`SIDE_PAD`] gutters are worth keeping; a
/// narrower tab spends every column on the label.
const PAD_MIN_WIDTH: usize = 6;

/// How much of a truncated label's tail dissolves into the tab, in label cells. The
/// fade signals "there is more" in place of an ellipsis, so it is kept short: about
/// the last two glyphs go, not a long soft gradient across the whole tail.
///
/// A width, not a cluster count, because the ramp is per-pixel (the backend applies
/// it in the fragment shader; see [`Fade`]) and so has no reason to quantize to
/// glyph boundaries. Scaling with the cell width keeps it proportional to the label
/// size rather than pinned to one font's idea of a pixel.
const FADE_SPAN: f32 = 2.0;

/// The device-pixel rectangle the strip occupies, plus the cell metrics and left
/// inset its contents lay out against. Computed window-side (it depends on the
/// surface size and the top/bottom config) and handed to the tabs layer on every
/// resize, so painting stays a pure function of it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BarGeom {
    /// The terminal cell metrics. Tab blocks, dividers, and pointer hit testing all
    /// ride this grid, so a click maps through the same `metrics.w` the window uses.
    pub metrics: CellMetrics,
    /// The (smaller) label font metrics: the labels are drawn at this size and pack
    /// at this cell width, centered within each block's pixel span. Its ascent and
    /// descent set the baseline. Equal to `metrics` when the label scale is 100%.
    pub label: CellMetrics,
    /// Full surface width in device pixels: the strip background spans it.
    pub surface_width: i32,
    /// Left inset (the window padding) tab column zero starts at.
    pub pad: i32,
    /// Strip top in device pixels.
    pub y: i32,
    /// Strip height in device pixels (>= one text row).
    pub h: i32,
}

/// One resolved title supplied by the tabs manager.
pub(crate) struct TabLabel<'a> {
    pub title: &'a str,
    pub active: bool,
}

/// One tab's block geometry and its sanitized (but not yet fitted) label. The label
/// is shaped and truncated at paint time, where the interface font is available to
/// measure proportional advances; only the block geometry here is font-free, so it
/// serves pointer hit testing without loading a font.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Slot {
    /// The whole tab block in terminal cells, for the background fill and pointer
    /// hit testing (both ride the terminal grid the window maps clicks against).
    pub cells: Range<usize>,
    /// The child-safe label (controls and zero-width clusters already replaced),
    /// with the empty title mapped to `shell`. Fitted to the block at paint time.
    pub title: String,
    /// Left edge of the block's content area in device pixels from the strip's
    /// content origin (add [`BarGeom::pad`]); the label centers within it.
    pub content_left: i32,
    /// Width of the block's content area in device pixels: the label fits into this
    /// span and is truncated (with a trailing fade) when it overflows.
    pub content_px: i32,
    pub active: bool,
}

/// Give every tab one equal width, clamped into the config bounds, and lay them
/// out left-aligned; the space past the last tab is left to the strip background.
/// Each title is sanitized (child-controlled bytes), stripped of its old numeric
/// prefix, and either centered whole or cut from the end to fit.
///
/// Blocks are sized in terminal cells (`cell_w`), so they align to the grid and hit
/// test against the same pitch the window maps clicks with. Each slot records its
/// content span in device pixels and the sanitized title; the proportional label is
/// fitted, centered, and (when it overflows) truncated with a trailing fade at paint
/// time by [`fill_bar`], which has the interface font to measure advances. `cell_w`
/// is the one-past-config terminal cell width (`>= 1`).
pub(crate) fn layout(
    cols: usize,
    labels: &[TabLabel<'_>],
    cfg: &TabBarConfig,
    cell_w: i32,
) -> Vec<Slot> {
    if labels.is_empty() {
        return Vec::new();
    }
    let n = labels.len();
    // The widest equal width that still fills the bar, then clamped to taste.
    //
    // Not `base.clamp(min, max)`: `usize::clamp` **panics** when `min > max`, and the two
    // are configuration. They are 10 and 24 today, so the call is safe today — and that
    // is exactly the shape of a trap, since the first user-supplied pair makes a
    // no-panic-outside-tests violation out of a line nobody edited. An inverted pair is
    // nonsense either way; taking the floor as the answer is the reading that keeps every
    // tab visible.
    let base = cols / n;
    let mut width = base
        .max(cfg.min_width)
        .min(cfg.max_width.max(cfg.min_width));
    // Clamping up to `min_width` can overflow the bar when there are more tabs
    // than `cols / min_width`; there is no width that both fits and honors the
    // floor, so fall back to an equal fill. Tabs then all stay visible and equal,
    // just below the configured minimum.
    if width.saturating_mul(n) > cols {
        width = base;
    }
    let width = width.max(1);
    let cell_w = cell_w.max(1);

    labels
        .iter()
        .enumerate()
        .map(|(index, label)| {
            let start = index * width;
            // The final tabs clip to the surface edge when many tabs overflow a
            // narrow window; an empty range is simply not painted or hit.
            let end = (start + width).min(cols);
            let cells = start..end;
            let avail = end.saturating_sub(start);
            let pad = if avail >= PAD_MIN_WIDTH { SIDE_PAD } else { 0 };
            let content_cells = avail.saturating_sub(2 * pad);
            // The content area in device pixels, and its left edge from the strip's
            // content origin. Blocks ride the terminal grid, so both use `cell_w`.
            let content_px = content_cells as i32 * cell_w;
            let content_left = (start + pad) as i32 * cell_w;

            let clean = sanitize_title(label.title);
            let title = if clean.is_empty() {
                "shell".to_string()
            } else {
                clean
            };
            Slot {
                cells,
                title,
                content_left,
                content_px,
                active: label.active,
            }
        })
        .collect()
}

/// Paint the strip: a full-width background, each active tab's block, every label,
/// then the dividers between adjacent inactive tabs. The label is the proportional
/// interface font ([`FaceKey::Ui`]), so `fonts` is needed to measure advances and
/// fit/center/truncate each title; the fitting runs only on painted frames and emits
/// pooled strings, so a steady repaint allocates nothing.
///
/// When a tab is being dragged, `lift` names it: its home block is left as a gap (the
/// `continue` in the slot loop) and the block is repainted last, at the floated `left`,
/// so it rides over its neighbours and any divider it currently covers. The dragged
/// tab is always the active tab, so the gap is already strip-coloured and the hairlines
/// beside it are already suppressed (see the divider loop's `active` guard); the lift
/// therefore needs no divider or gap-colour handling of its own, and adds exactly one
/// fill and one text run per frame.
pub(crate) fn fill_bar(
    out: &mut DisplayList,
    strings: &mut Vec<String>,
    slots: &[Slot],
    bar: &BarGeom,
    cfg: &TabBarConfig,
    fonts: &Fonts,
    lift: Option<Lift>,
) {
    let m = bar.metrics;
    let lm = bar.label;
    let strip_bg = cfg.inactive.bg;
    // The strip background spans the whole width, so it reads seamless with the
    // grid and covers the slack past the last tab in one fill.
    out.push(DrawCmd::Fill {
        rect: Rect {
            x: 0,
            y: bar.y,
            w: bar.surface_width.max(0),
            h: bar.h,
        },
        color: strip_bg.to_u32(),
    });
    // Center the label's ink box (ascent + descent) in the strip, not the full line
    // box (`lm.h`): the line height carries the font's line gap, which sits below
    // the descent, so centering `lm.h` would bias the text upward.
    let baseline = bar.y + (bar.h - lm.ascent - lm.descent) / 2 + lm.ascent;
    // Every label is the interface sans at one medium weight (heavier than regular,
    // lighter than bold); the active tab is marked by its colored block and
    // foreground, not by a heavier stroke. Medium falls back to regular for a family
    // that ships no medium file.
    let face = FaceKey::Ui {
        size: lm.size,
        style: FontStyle::Medium,
    };

    for (index, slot) in slots.iter().enumerate() {
        // The lifted tab leaves a gap here; it is repainted at its floated x below.
        if slot.cells.is_empty() || lift.is_some_and(|l| l.slot == index) {
            continue;
        }
        let colors = if slot.active {
            cfg.active
        } else {
            cfg.inactive
        };
        let x = bar.pad + slot.cells.start as i32 * m.w;
        // An inactive tab already sits on the strip background; only a differing
        // background (the active block, or a customized inactive bg) needs a fill,
        // so a steady multi-tab frame paints just the one active block.
        if colors.bg != strip_bg {
            out.push(DrawCmd::Fill {
                rect: Rect {
                    x,
                    y: bar.y,
                    w: (slot.cells.end - slot.cells.start) as i32 * m.w,
                    h: bar.h,
                },
                color: colors.bg.to_u32(),
            });
        }
        paint_label(
            out,
            strings,
            fonts,
            &slot.title,
            bar.pad + slot.content_left,
            slot.content_px,
            baseline,
            face,
            colors,
            lm,
        );
    }

    // A divider between two adjacent inactive tabs; never beside the active block,
    // so the active tab reads as a clean solid block on both sides.
    let inset = bar.h / 5;
    for pair in slots.windows(2) {
        let (prev, cur) = (&pair[0], &pair[1]);
        if cur.cells.is_empty() || prev.active || cur.active {
            continue;
        }
        out.push(DrawCmd::Fill {
            rect: Rect {
                x: bar.pad + cur.cells.start as i32 * m.w,
                y: bar.y + inset,
                w: 1,
                h: (bar.h - 2 * inset).max(1),
            },
            color: cfg.divider.to_u32(),
        });
    }

    // The dragged tab, painted over everything so it floats above its neighbours and
    // any divider it now covers. It is the active tab, so it wears the active colours
    // (no new theme entry): the tab you are holding looks like the tab you selected.
    // The block floats at its full nominal pitch (slot zero's width) even if its own
    // home slot was clipped, and its label rides along by the same shift.
    if let Some(lift) = lift {
        if let Some(slot) = slots.get(lift.slot).filter(|slot| !slot.cells.is_empty()) {
            let pitch = slots.first().map_or(0, |s| s.cells.len() as i32 * m.w);
            out.push(DrawCmd::Fill {
                rect: Rect {
                    x: lift.left,
                    y: bar.y,
                    w: pitch,
                    h: bar.h,
                },
                color: cfg.active.bg.to_u32(),
            });
            let dx = lift.left - (bar.pad + slot.cells.start as i32 * m.w);
            paint_label(
                out,
                strings,
                fonts,
                &slot.title,
                bar.pad + slot.content_left + dx,
                slot.content_px,
                baseline,
                face,
                cfg.active,
                lm,
            );
        }
    }
}

/// Tab index under `col`, if any. Empty ranges (more tabs than columns) are not
/// hittable, and the slack past the last tab maps to no tab.
pub(crate) fn hit_test(slots: &[Slot], col: usize) -> Option<usize> {
    slots.iter().position(|slot| slot.cells.contains(&col))
}

/// A tab lifted out of the run by a drag: which slot, and the device-pixel left
/// edge its block floats at (already clamped into the run by the caller). Consumed
/// by [`fill_bar`], which then leaves a gap at `slot`'s home block and paints the
/// block last at `left` so it rides over its neighbours.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Lift {
    pub slot: usize,
    pub left: i32,
}

/// The device-pixel left edge and width of slot `i`'s block, measured from the
/// surface origin ([`BarGeom::pad`] included, the same space the pointer scales
/// into). The press reads slot `i`'s left edge to measure the grab offset; the drag
/// reads slot zero's width for the pitch (slot zero is never clipped, so its block
/// is the true pitch even when a narrow window cuts the last one). `None` when `i`
/// is out of range.
pub(crate) fn block_px(slots: &[Slot], bar: &BarGeom, i: usize) -> Option<(i32, i32)> {
    let slot = slots.get(i)?;
    let left = bar.pad + slot.cells.start as i32 * bar.metrics.w;
    let width = (slot.cells.end - slot.cells.start) as i32 * bar.metrics.w;
    Some((left, width))
}

/// Which slot a block whose left edge sits at device-x `left` wants to land in: the
/// slot its own midpoint falls in, clamped to the run. Blocks are one equal width,
/// so this is a divide against the pitch, not a scan; and because the pointer is not
/// consulted, where within the tab it was grabbed does not shift the landing. The
/// pitch comes from slot zero (never clipped), so a narrow window cutting the last
/// block does not skew the divide. `None` only when there are no tabs.
pub(crate) fn drop_index(slots: &[Slot], bar: &BarGeom, left: i32) -> Option<usize> {
    let pitch = (slots.first()?.cells.len() as i32 * bar.metrics.w).max(1);
    let mid = (left + pitch / 2 - bar.pad).max(0);
    Some(((mid / pitch) as usize).min(slots.len() - 1))
}

/// Fit `title` into `content_px` device pixels of the proportional interface face,
/// center it (or, when it overflows, keep it flush-left and fade its tail), and emit
/// the run. `content_left` is the block content's device x from the strip origin.
///
/// Either way this is *one* [`DrawCmd::Text`]; an overflowing label differs only by
/// carrying a [`Fade`], which ramps the run's ink to nothing across its last
/// [`FADE_SPAN`] so the tail dissolves in place of an ellipsis. The label keeps its
/// own foreground throughout: the ink lands
/// on exactly the tab background (no stain on a light block), and the coverage gamma
/// still reads the label's true contrast, so the tail's strokes weigh the same as
/// the head's. Fitting measures glyph advances but allocates nothing beyond the one
/// pooled run string, so a steady repaint stays allocation-free.
#[allow(clippy::too_many_arguments)]
fn paint_label(
    out: &mut DisplayList,
    strings: &mut Vec<String>,
    fonts: &Fonts,
    title: &str,
    content_left: i32,
    content_px: i32,
    baseline: i32,
    face: FaceKey,
    colors: TabColors,
    lm: CellMetrics,
) {
    if content_px <= 0 || title.is_empty() {
        return;
    }
    let budget = content_px as f32;
    // How much of the title fits, and its measured width. The first cluster always
    // goes in (even overflowing a tiny tab), so a block never paints empty.
    let mut fitted_bytes = 0usize;
    let mut fitted_w = 0.0f32;
    let mut truncated = false;
    for (offset, cluster) in grapheme::graphemes(title) {
        let advance = cluster_advance(fonts, face, cluster);
        if fitted_bytes > 0 && fitted_w + advance > budget {
            truncated = true;
            break;
        }
        fitted_w += advance;
        fitted_bytes = offset + cluster.len();
    }
    // Center a label that fits; a truncated one has filled the span, so it sits flush
    // to the content's left edge.
    let start_x = if truncated {
        content_left
    } else {
        content_left + ((budget - fitted_w) / 2.0).round() as i32
    };
    let ink_w = fitted_w.round() as i32;
    // The ramp ends on the last glyph's own right edge, not the block's: a title cut
    // mid-cluster stops short of `content_px`, and a ramp reaching zero out in the
    // empty gutter past it would leave the tail solid. Its start is clamped to the
    // label's left edge, so a title too short to hold a full span dissolves across
    // itself rather than beginning already half-gone.
    let fade = truncated.then(|| {
        let end = start_x + ink_w;
        Fade {
            from: (end - fade_span(lm)).max(start_x),
            to: end,
        }
    });
    term_render::push_text_run(
        out,
        strings,
        &title[..fitted_bytes],
        start_x,
        ink_w,
        baseline,
        face,
        colors.fg.to_u32(),
        colors.bg.to_u32(),
        fade,
        lm,
    );
}

/// The width of a truncated label's dissolving tail, in device pixels: [`FADE_SPAN`]
/// label cells, floored at one pixel so the span is never empty (which the backend
/// reads as "no fade", leaving a hard cut).
fn fade_span(lm: CellMetrics) -> i32 {
    ((FADE_SPAN * lm.w as f32).round() as i32).max(1)
}

/// The total advance of a grapheme cluster in `face`, summed over its characters
/// (combining marks advance ~zero and stack on the base). Each character resolves
/// through the same fallback chain the renderer uses, so a measured width matches
/// what is painted.
fn cluster_advance(fonts: &Fonts, face: FaceKey, cluster: &str) -> f32 {
    cluster
        .chars()
        .map(|ch| fonts.glyph_face(face, ch).advance(ch))
        .sum()
}

fn sanitize_title(title: &str) -> String {
    let mut clean = String::with_capacity(title.len());
    for (_, cluster) in grapheme::graphemes(title) {
        let width = term_render::display_cluster_width(cluster);
        if cluster.chars().any(char::is_control) || width == 0 || width > 2 {
            clean.push('\u{fffd}');
        } else {
            clean.push_str(cluster);
        }
    }
    clean
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

    fn label(title: &str, active: bool) -> TabLabel<'_> {
        TabLabel { title, active }
    }

    /// Lay out at the test cell width. Blocks are font-free, so this needs no fonts;
    /// only painting (which fits the proportional label) does.
    fn lay(cols: usize, labels: &[TabLabel<'_>], cfg: &TabBarConfig) -> Vec<Slot> {
        layout(cols, labels, cfg, METRICS.w)
    }

    /// The real interface fonts at the test size. Painting the bar measures glyph
    /// advances, so the tests drive the true font layer (no mock) exactly as the
    /// window does.
    fn fonts() -> Fonts {
        Fonts::new(&[METRICS.size]).expect("open fonts")
    }

    fn geom(fonts: &Fonts, width_cols: usize, pad: i32) -> BarGeom {
        BarGeom {
            metrics: METRICS,
            label: CellMetrics::from_ui(fonts, METRICS.size),
            surface_width: pad + width_cols as i32 * METRICS.w + pad,
            pad,
            y: 0,
            h: METRICS.h,
        }
    }

    /// Paint `slots` and return the display list, so tests inspect the emitted
    /// commands rather than internal fitting state.
    fn paint(
        fonts: &Fonts,
        slots: &[Slot],
        cfg: &TabBarConfig,
        width_cols: usize,
        pad: i32,
    ) -> Vec<DrawCmd> {
        paint_lift(fonts, slots, cfg, width_cols, pad, None)
    }

    /// Paint `slots` with a tab lifted out by a drag, returning the display list.
    fn paint_lift(
        fonts: &Fonts,
        slots: &[Slot],
        cfg: &TabBarConfig,
        width_cols: usize,
        pad: i32,
        lift: Option<Lift>,
    ) -> Vec<DrawCmd> {
        let mut out = Vec::new();
        let mut strings = Vec::new();
        fill_bar(
            &mut out,
            &mut strings,
            slots,
            &geom(fonts, width_cols, pad),
            cfg,
            fonts,
            lift,
        );
        out
    }

    /// Each emitted label run's (text, color, weight), in list order. Only the
    /// interface (`Ui`) runs are labels; a run in any other face is skipped.
    fn text_runs(out: &[DrawCmd]) -> Vec<(String, u32, FontStyle)> {
        out.iter()
            .filter_map(|c| match c {
                DrawCmd::Text {
                    text,
                    color,
                    face: FaceKey::Ui { style, .. },
                    ..
                } => Some((text.clone(), *color, *style)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn tabs_are_equal_width_and_left_aligned() {
        let cfg = TabBarConfig::default();
        let labels = [
            label("one", true),
            label("two", false),
            label("three", false),
        ];
        // 80 cols / 3 = 26, clamped to max_width 24; three equal 24-wide blocks,
        // left-aligned, with the remaining 8 cols left to the strip background.
        let slots = lay(80, &labels, &cfg);
        assert_eq!(slots[0].cells, 0..24);
        assert_eq!(slots[1].cells, 24..48);
        assert_eq!(slots[2].cells, 48..72);
    }

    #[test]
    fn width_is_capped_at_max_and_floored_at_min() {
        let cfg = TabBarConfig::default(); // min 10, max 24
                                           // Two tabs in a wide bar cap at max_width, not half the window each.
        let two = lay(200, &[label("a", true), label("b", false)], &cfg);
        assert_eq!(two[0].cells.len(), 24);
        assert_eq!(two[1].cells.len(), 24);
        // Eight tabs in 80 cols land exactly at the floor (80 / 8 == 10).
        let eight: Vec<_> = (0..8).map(|_| label("t", false)).collect();
        let slots = lay(80, &eight, &cfg);
        assert!(slots.iter().all(|s| s.cells.len() == 10));
    }

    #[test]
    fn too_many_tabs_fall_back_to_an_equal_fill_below_min() {
        let cfg = TabBarConfig::default(); // min 10
                                           // Twenty tabs cannot each be 10 wide in 80 cols; they fill equally at 4.
        let many: Vec<_> = (0..20).map(|_| label("t", false)).collect();
        let slots = lay(80, &many, &cfg);
        assert!(slots.iter().all(|s| s.cells.len() == 4));
    }

    #[test]
    fn empty_title_falls_back_to_shell() {
        let cfg = TabBarConfig::default();
        assert_eq!(lay(80, &[label("", true)], &cfg)[0].title, "shell");
    }

    #[test]
    fn controls_and_zero_width_clusters_are_replaced() {
        let cfg = TabBarConfig::default();
        let slot = lay(80, &[label("bad\n\u{200b}title", false)], &cfg).remove(0);
        assert_eq!(slot.title, "bad\u{fffd}\u{fffd}title");
        assert_eq!(slot.title.matches('\u{fffd}').count(), 2);
    }

    #[test]
    fn hit_testing_uses_half_open_ranges() {
        let cfg = TabBarConfig {
            min_width: 5,
            max_width: 5,
            ..TabBarConfig::default()
        };
        let slots = lay(10, &[label("a", true), label("b", false)], &cfg);
        assert_eq!(hit_test(&slots, 0), Some(0));
        assert_eq!(hit_test(&slots, 4), Some(0));
        assert_eq!(hit_test(&slots, 5), Some(1));
        assert_eq!(hit_test(&slots, 10), None);
    }

    #[test]
    fn an_inverted_width_range_lays_out_instead_of_panicking() {
        // `usize::clamp` panics when `min > max`, and both bounds are configuration. The
        // defaults are 10 and 24, so the old `base.clamp(min, max)` was safe *today* — and
        // that is the shape of a trap rather than a reason to leave it: the first
        // user-supplied pair turns a line nobody edited into a no-panic-outside-tests
        // violation, and the panic lands on window paint.
        let cfg = TabBarConfig {
            min_width: 30,
            max_width: 4,
            ..TabBarConfig::default()
        };
        let slots = lay(80, &[label("a", true), label("b", false)], &cfg);
        assert_eq!(slots.len(), 2, "both tabs still laid out");
        assert!(slots.iter().all(|s| !s.cells.is_empty()));
        assert!(
            slots.iter().map(|s| s.cells.len()).sum::<usize>() <= 80,
            "and they still fit the bar"
        );

        // Degenerate bounds are not a special case either.
        for (min_width, max_width) in [(0, 0), (usize::MAX, 0), (0, usize::MAX)] {
            let cfg = TabBarConfig {
                min_width,
                max_width,
                ..TabBarConfig::default()
            };
            let slots = lay(80, &[label("a", true)], &cfg);
            assert_eq!(slots.len(), 1);
            assert!(!slots[0].cells.is_empty(), "{min_width}/{max_width}");
        }
    }

    #[test]
    fn a_fitting_label_paints_one_centered_run() {
        let cfg = TabBarConfig::default();
        let fonts = fonts();
        // A short title in a wide (24-cell) tab fits whole: one run carrying the full
        // title, no truncation, no fade.
        let slots = lay(80, &[label("hi", true)], &cfg);
        let out = paint(&fonts, &slots, &cfg, 80, 0);
        let runs = text_runs(&out);
        assert_eq!(runs.len(), 1, "a fitting label is one run");
        assert_eq!(runs[0].0, "hi");
        // Centered within the block content (starts past the left gutter, ends before
        // the right edge), and painted in the active foreground.
        let (x, run_w) = out
            .iter()
            .find_map(|c| match c {
                DrawCmd::Text { x, bounds, .. } => Some((*x, bounds.w)),
                _ => None,
            })
            .expect("a run");
        assert!(
            x > slots[0].content_left,
            "left of a centered run clears the gutter"
        );
        assert!(
            x + run_w < slots[0].content_left + slots[0].content_px + METRICS.w,
            "and it stays within the block"
        );
        assert_eq!(runs[0].1, cfg.active.fg.to_u32());
    }

    /// The `(x, bounds, fade)` of every emitted label run, in list order.
    fn fades(out: &[DrawCmd]) -> Vec<(i32, Rect, Option<Fade>)> {
        out.iter()
            .filter_map(|c| match c {
                DrawCmd::Text {
                    x,
                    bounds,
                    fade,
                    face: FaceKey::Ui { .. },
                    ..
                } => Some((*x, *bounds, *fade)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_long_title_truncates_from_the_end_and_its_tail_fades_to_nothing() {
        let cfg = TabBarConfig::default();
        let fonts = fonts();
        let title = "a-very-long-window-title-that-will-not-fit";
        let slots = lay(80, &[label(title, true)], &cfg);
        let out = paint(&fonts, &slots, &cfg, 80, 0);
        let runs = text_runs(&out);

        // One run, whatever the fade: the tail dissolves per pixel, so it needs no
        // per-cluster runs to re-colour.
        assert_eq!(runs.len(), 1, "a faded label is still one run");
        let painted = &runs[0].0;
        assert!(
            title.starts_with(painted) && painted.len() < title.len(),
            "the start is kept and the end dropped: {painted:?}"
        );
        assert!(!painted.contains('…'), "no ellipsis, a fade instead");

        // The label keeps its own foreground end to end. This is the load-bearing
        // one: mixing the tail toward the background would both leave a stain on a
        // light tab (the mix never arrives) and misreport the run's contrast to the
        // coverage gamma, so the tail's strokes would weigh differently than the
        // head's. The fade is ink, not colour.
        assert_eq!(
            runs[0].1,
            cfg.active.fg.to_u32(),
            "the whole run stays the label's foreground"
        );

        let (x, _, fade) = fades(&out)[0];
        let fade = fade.expect("a truncated label fades");
        assert!(fade.to > fade.from, "a non-empty span, or the backend cuts");
        assert!(fade.from > x, "a long label keeps a solid head");
        assert_eq!(
            fade.to - fade.from,
            fade_span(CellMetrics::from_ui(&fonts, METRICS.size)),
            "the tail dissolves over FADE_SPAN label cells"
        );
    }

    #[test]
    fn a_fitting_label_does_not_fade() {
        let cfg = TabBarConfig::default();
        let fonts = fonts();
        let out = paint(&fonts, &lay(80, &[label("hi", true)], &cfg), &cfg, 80, 0);
        assert_eq!(fades(&out)[0].2, None, "an untruncated label is solid");
    }

    #[test]
    fn the_ramp_stays_inside_the_label_however_narrow_the_tab() {
        // The ramp must land on the label's own ink at both ends. If it began out in
        // the left gutter the head would already be half-gone; if it reached zero out
        // in the right gutter (a title cut mid-cluster stops short of the block edge)
        // the tail would stay solid and the cut would be hard. Neither may happen at
        // any tab width, down to a single cell, whatever the label font measures: a
        // label too short to hold a full span dissolves across the whole of itself.
        let fonts = fonts();
        let span = fade_span(CellMetrics::from_ui(&fonts, METRICS.size));
        for width in 1..=12usize {
            let cfg = TabBarConfig {
                min_width: width,
                max_width: width,
                ..TabBarConfig::default()
            };
            let slots = lay(width, &[label("a-title-far-too-long-to-fit", true)], &cfg);
            let (x, bounds, fade) = fades(&paint(&fonts, &slots, &cfg, width, 0))[0];
            let fade = fade.expect("the title truncates at every one of these widths");
            assert!(fade.to > fade.from, "{width}: a non-empty span");
            assert!(fade.from >= x, "{width}: the ramp starts inside the label");
            assert!(
                fade.to > x && fade.to <= bounds.x + bounds.w,
                "{width}: the ramp reaches zero on the ink, not past it"
            );
            let expected = if fade.to - x <= span {
                x
            } else {
                fade.to - span
            };
            assert_eq!(fade.from, expected, "{width}: clamped to the left edge");
        }
    }

    #[test]
    fn painting_is_a_strip_fill_then_the_active_block_and_labels() {
        let cfg = TabBarConfig::default();
        let fonts = fonts();
        let slots = lay(80, &[label("one", true), label("two", false)], &cfg);
        let out = paint(&fonts, &slots, &cfg, 80, 5);

        // Strip background first.
        assert!(matches!(out[0], DrawCmd::Fill { .. }));
        // Exactly one tab-background fill besides the strip: the active block
        // (inactive tabs share the strip background and are not re-filled).
        let tab_bgs = out
            .iter()
            .filter(|c| matches!(c, DrawCmd::Fill { rect, .. } if rect.h == METRICS.h))
            .count();
        assert_eq!(tab_bgs, 2, "strip background + one active block");
        // Both labels fit, so each is one run: the active in its foreground, the
        // inactive in its own. Both are the same medium weight (the active tab is
        // marked by its block and colour, not a heavier stroke).
        let runs = text_runs(&out);
        assert_eq!(runs.len(), 2);
        assert_eq!(
            runs[0],
            ("one".to_string(), cfg.active.fg.to_u32(), FontStyle::Medium)
        );
        assert_eq!(
            runs[1],
            (
                "two".to_string(),
                cfg.inactive.fg.to_u32(),
                FontStyle::Medium
            )
        );
        // Each run carries its own tab background for the anti-aliasing weight.
        let bgs: Vec<u32> = out
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { bg, .. } => Some(*bg),
                _ => None,
            })
            .collect();
        assert_eq!(bgs, vec![cfg.active.bg.to_u32(), cfg.inactive.bg.to_u32()]);
    }

    #[test]
    fn a_divider_separates_two_inactive_tabs_but_not_the_active_one() {
        let cfg = TabBarConfig {
            min_width: 8,
            max_width: 8,
            ..TabBarConfig::default()
        };
        let fonts = fonts();
        // Three tabs, the middle active: divider between 0|1 is hidden (1 active),
        // 1|2 hidden (1 active); with the active in the middle, no divider shows.
        let mid = lay(
            24,
            &[label("a", false), label("b", true), label("c", false)],
            &cfg,
        );
        let out = paint(&fonts, &mid, &cfg, 24, 0);
        let dividers = out
            .iter()
            .filter(|c| matches!(c, DrawCmd::Fill { rect, .. } if rect.w == 1))
            .count();
        assert_eq!(dividers, 0, "both boundaries touch the active middle tab");

        // First tab active: only the 1|2 boundary (both inactive) shows a divider.
        let first = lay(
            24,
            &[label("a", true), label("b", false), label("c", false)],
            &cfg,
        );
        let out = paint(&fonts, &first, &cfg, 24, 0);
        let dividers = out
            .iter()
            .filter(|c| matches!(c, DrawCmd::Fill { rect, .. } if rect.w == 1))
            .count();
        assert_eq!(dividers, 1, "one divider between the two inactive tabs");
    }

    #[test]
    fn labels_center_the_ink_box_not_the_line_box() {
        let cfg = TabBarConfig::default();
        let fonts = fonts();
        let slots = lay(80, &[label("a", true)], &cfg);
        let mut out = Vec::new();
        let mut strings = Vec::new();
        // A label metric with a line gap (h = 20 > ascent + descent = 16) in a tall
        // strip: the baseline centers the ink box, so the gap never biases it up.
        let gapped = CellMetrics {
            size: METRICS.size,
            w: 8,
            h: 20,
            baseline: 16,
            ascent: 12,
            descent: 4,
            lock_glyph: true,
        };
        let geom = BarGeom {
            metrics: METRICS,
            label: gapped,
            surface_width: 80 * METRICS.w,
            pad: 0,
            y: 0,
            h: 40,
        };
        fill_bar(&mut out, &mut strings, &slots, &geom, &cfg, &fonts, None);
        let baseline = out.iter().find_map(|c| match c {
            DrawCmd::Text { baseline, .. } => Some(*baseline),
            _ => None,
        });
        // Ink-centered: (40 - 12 - 4)/2 + 12 = 24. Line-box centering would give
        // (40 - 20)/2 + 12 = 22, biasing the text two pixels high.
        assert_eq!(
            baseline,
            Some((40 - gapped.ascent - gapped.descent) / 2 + gapped.ascent)
        );
        assert_ne!(
            baseline,
            Some((40 - gapped.h) / 2 + gapped.ascent),
            "must not center the line box"
        );
    }

    /// Four equal 10-cell blocks in 40 cols, pad 0: pitch is exactly 80 device px.
    fn equal_blocks(cfg: &TabBarConfig, n: usize) -> Vec<Slot> {
        let labels: Vec<_> = (0..n).map(|_| label("t", false)).collect();
        lay(n * 10, &labels, cfg)
    }

    fn equal_cfg() -> TabBarConfig {
        TabBarConfig {
            min_width: 10,
            max_width: 10,
            ..TabBarConfig::default()
        }
    }

    #[test]
    fn drop_index_follows_the_dragged_blocks_own_centre() {
        let fonts = fonts();
        let cfg = equal_cfg();
        let slots = equal_blocks(&cfg, 4);
        let bar = geom(&fonts, 40, 0);
        let pitch = slots[0].cells.len() as i32 * METRICS.w;
        assert_eq!(pitch, 80);

        // A block whose left edge sits at k*pitch centres in block k and lands at k.
        for k in 0..4 {
            assert_eq!(drop_index(&slots, &bar, k as i32 * pitch), Some(k));
        }
        // The boundary is exactly half a block of travel: at left = pitch/2 the
        // centre reaches the 0|1 seam and tips into slot 1, not one pixel before.
        assert_eq!(drop_index(&slots, &bar, pitch / 2 - 1), Some(0));
        assert_eq!(drop_index(&slots, &bar, pitch / 2), Some(1));

        // Grab-invariance, pinned explicitly (this is the whole reason the drop reads
        // the block's own left edge, not the pointer): any (pointer, grab) pair that
        // places the block's left at the same px lands in the same slot, so where in
        // the tab it was grabbed never shifts the landing.
        let left = pitch + 3;
        for grab in 0..pitch {
            let pointer = left + grab; // left == pointer - grab
            assert_eq!(
                drop_index(&slots, &bar, pointer - grab),
                drop_index(&slots, &bar, left),
                "grab offset {grab} must not change the landing"
            );
        }
    }

    #[test]
    fn drop_index_clamps_to_the_run() {
        let fonts = fonts();
        let cfg = equal_cfg();
        let slots = equal_blocks(&cfg, 3);
        let bar = geom(&fonts, 30, 0);
        let pitch = slots[0].cells.len() as i32 * METRICS.w;

        // Dragged left of the strip pins at slot 0; past the last block pins at the
        // final slot. Neither slides into dead space.
        assert_eq!(drop_index(&slots, &bar, -1000), Some(0));
        assert_eq!(drop_index(&slots, &bar, 100 * pitch), Some(2));

        // A single tab always lands at 0, wherever it is dragged.
        let one = lay(10, &[label("t", true)], &cfg);
        let bar1 = geom(&fonts, 10, 0);
        assert_eq!(drop_index(&one, &bar1, 5 * pitch), Some(0));
        assert_eq!(drop_index(&one, &bar1, -pitch), Some(0));

        // No tabs, no slot.
        assert_eq!(drop_index(&[], &bar, 0), None);
    }

    #[test]
    fn drop_index_uses_the_nominal_pitch_when_the_last_block_is_clipped() {
        let fonts = fonts();
        let cfg = equal_cfg();
        let mut slots = equal_blocks(&cfg, 3); // 0..10, 10..20, 20..30
                                               // A narrow window cut the final block to half its width.
        slots[2].cells = 20..25;
        let bar = geom(&fonts, 30, 0);
        let pitch = slots[0].cells.len() as i32 * METRICS.w; // 80, from the full slot 0

        // The divide still uses slot zero's pitch, so the clipped last block does not
        // skew the landing: the boundary into slot 2 is at 1.5 pitches of travel.
        assert_eq!(drop_index(&slots, &bar, pitch + pitch / 2 - 1), Some(1));
        assert_eq!(drop_index(&slots, &bar, pitch + pitch / 2), Some(2));
        assert_eq!(drop_index(&slots, &bar, 2 * pitch), Some(2));
    }

    #[test]
    fn a_lifted_tab_leaves_a_gap_and_paints_over_its_neighbours() {
        let fonts = fonts();
        let cfg = equal_cfg();
        // Three tabs, the middle one active: the dragged tab is always the active one.
        let slots = lay(
            30,
            &[label("aa", false), label("bb", true), label("cc", false)],
            &cfg,
        );
        let pitch = slots[0].cells.len() as i32 * METRICS.w;
        // The middle tab's home block left edge (pad is 0 here).
        let home1 = slots[1].cells.start as i32 * METRICS.w;
        // Float the middle tab half a block toward tab 0.
        let floated = home1 - pitch / 2;
        let out = paint_lift(
            &fonts,
            &slots,
            &cfg,
            30,
            0,
            Some(Lift {
                slot: 1,
                left: floated,
            }),
        );

        // All three labels still paint (none is dropped): the two home neighbours plus
        // the floating one.
        assert_eq!(text_runs(&out).len(), 3);

        // The floating active block is the last full-height, full-pitch fill, at the
        // drag x and in the active colour, so it rides over its neighbours.
        let last_block = out
            .iter()
            .rev()
            .find_map(|c| match c {
                DrawCmd::Fill { rect, color } if rect.h == METRICS.h && rect.w == pitch => {
                    Some((*rect, *color))
                }
                _ => None,
            })
            .expect("a floating active block");
        assert_eq!(last_block.0.x, floated, "it floats at the drag x");
        assert_eq!(last_block.1, cfg.active.bg.to_u32(), "in the active colour");

        // The lifted tab's home block is a bare gap: no active-coloured fill sits
        // there.
        let home_filled = out.iter().any(|c| {
            matches!(c, DrawCmd::Fill { rect, color }
                if rect.h == METRICS.h && rect.x == home1 && *color == cfg.active.bg.to_u32())
        });
        assert!(
            !home_filled,
            "the home block is a gap, not a filled active block"
        );
    }

    #[test]
    fn a_lifted_gap_needs_no_divider_repair() {
        let fonts = fonts();
        let cfg = TabBarConfig {
            min_width: 8,
            max_width: 8,
            ..TabBarConfig::default()
        };
        // The middle tab active and lifted hard left over tab 0. Its two edges must
        // stay divider-free: the gap it leaves is bounded by the active tab, whose
        // neighbours never draw a hairline against it. If a future change ever let a
        // *non-active* tab be dragged, this is the test that goes red.
        let slots = lay(
            24,
            &[label("a", false), label("b", true), label("c", false)],
            &cfg,
        );
        let out = paint_lift(&fonts, &slots, &cfg, 24, 0, Some(Lift { slot: 1, left: 0 }));
        let dividers = out
            .iter()
            .filter(|c| matches!(c, DrawCmd::Fill { rect, .. } if rect.w == 1))
            .count();
        assert_eq!(
            dividers, 0,
            "no hairline borders the lifted active tab's gap"
        );
    }
}
