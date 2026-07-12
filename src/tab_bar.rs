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
//! from the *end* (the start stays put) and its trailing cells fade into the tab
//! background instead of ending on an abrupt ellipsis. A one-pixel divider sits
//! between two inactive tabs so equal blocks sharing the strip background stay
//! separable.
//!
//! ```text
//!   pad │  tab 0   │  tab 1  │ tab 2 │            (strip background)          │
//!       │ centered │ cente…  │       │  <- fade    <- slack past the last tab
//!       └ divider between two inactive tabs; hidden beside the active block
//! ```

use std::ops::Range;

use crate::config::{TabBarConfig, TabColors};
use crate::platform::freetype::{FaceKey, FontStyle, Fonts};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::render::display::{DisplayList, DrawCmd};
use crate::term_render::{self, CellMetrics};

/// One breathing-room column each side of a label, so text never butts against a
/// tab edge or a divider. Dropped on tabs too narrow to spare it.
const SIDE_PAD: usize = 1;

/// The tab width at or above which the [`SIDE_PAD`] gutters are worth keeping; a
/// narrower tab spends every column on the label.
const PAD_MIN_WIDTH: usize = 6;

/// How many trailing grapheme clusters of a truncated label fade toward the tab
/// background. The fade signals "there is more" in place of an ellipsis. Kept short
/// and end-weighted (see the quadratic ramp in [`paint_label`]) so only the last
/// glyph or so dissolves, rather than a long soft gradient across the whole tail.
const FADE_CLUSTERS: usize = 2;

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
    let base = cols / n;
    let mut width = base.clamp(cfg.min_width, cfg.max_width);
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
pub(crate) fn fill_bar(
    out: &mut DisplayList,
    strings: &mut Vec<String>,
    slots: &[Slot],
    bar: &BarGeom,
    cfg: &TabBarConfig,
    fonts: &Fonts,
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

    for slot in slots.iter().filter(|slot| !slot.cells.is_empty()) {
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
}

/// Tab index under `col`, if any. Empty ranges (more tabs than columns) are not
/// hittable, and the slack past the last tab maps to no tab.
pub(crate) fn hit_test(slots: &[Slot], col: usize) -> Option<usize> {
    slots.iter().position(|slot| slot.cells.contains(&col))
}

/// Fit `title` into `content_px` device pixels of the proportional interface face,
/// center it (or, when it overflows, keep it flush-left and fade its tail), and emit
/// the run(s). `content_left` is the block content's device x from the strip origin.
///
/// A label that fits is one [`DrawCmd::Text`] run. A label that overflows is the
/// solid prefix as one run plus one run per faded trailing cluster ([`FADE_CLUSTERS`]
/// of them), each re-colored a quadratic step deeper toward the background so the
/// last glyph all but dissolves in place of an ellipsis. Fitting measures glyph
/// advances but allocates nothing beyond the pooled run strings, so a steady repaint
/// stays allocation-free.
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
    let mut count = 0usize;
    let mut truncated = false;
    for (offset, cluster) in grapheme::graphemes(title) {
        let advance = cluster_advance(fonts, face, cluster);
        if count > 0 && fitted_w + advance > budget {
            truncated = true;
            break;
        }
        fitted_w += advance;
        fitted_bytes = offset + cluster.len();
        count += 1;
    }
    let fitted = &title[..fitted_bytes];
    let fg = colors.fg.to_u32();
    let bg = colors.bg.to_u32();
    // Center a label that fits; a truncated one has filled the span, so it sits flush
    // to the content's left edge.
    let start_x = if truncated {
        content_left
    } else {
        content_left + ((budget - fitted_w) / 2.0).round() as i32
    };
    if !truncated {
        term_render::push_text_run(
            out,
            strings,
            fitted,
            start_x,
            fitted_w.round() as i32,
            baseline,
            face,
            fg,
            bg,
            lm,
        );
        return;
    }

    // The tail fades over the last `fade_n` clusters. Find where that tail begins
    // (its byte offset and the x it starts at), so the solid prefix is one run.
    let fade_n = FADE_CLUSTERS.min(count);
    let prefix_clusters = count - fade_n;
    let mut prefix_bytes = 0usize;
    let mut prefix_w = 0.0f32;
    for (index, (offset, cluster)) in grapheme::graphemes(fitted).enumerate() {
        if index == prefix_clusters {
            prefix_bytes = offset;
            break;
        }
        prefix_w += cluster_advance(fonts, face, cluster);
    }
    if prefix_clusters > 0 {
        term_render::push_text_run(
            out,
            strings,
            &fitted[..prefix_bytes],
            start_x,
            prefix_w.round() as i32,
            baseline,
            face,
            fg,
            bg,
            lm,
        );
    }
    // Each trailing cluster in its own run, a quadratic step deeper toward the
    // background so the fade stays near full strength until the very end and then
    // drops off fast: with two clusters the weights are 1/5 and 4/5 of the way to
    // the background. `depth` counts from the end, so the last glyph is always the
    // most faded even when only one cluster fades.
    let denom = (FADE_CLUSTERS * FADE_CLUSTERS) as u16 + 1;
    let mut x = prefix_w;
    for (step, (offset, cluster)) in grapheme::graphemes(&fitted[prefix_bytes..]).enumerate() {
        let advance = cluster_advance(fonts, face, cluster);
        let depth = (FADE_CLUSTERS - (fade_n - 1 - step)) as u16;
        let faded = colors.fg.mix(colors.bg, depth * depth, denom);
        let start = prefix_bytes + offset;
        term_render::push_text_run(
            out,
            strings,
            &fitted[start..start + cluster.len()],
            start_x + x.round() as i32,
            advance.round() as i32,
            baseline,
            face,
            faded.to_u32(),
            bg,
            lm,
        );
        x += advance;
    }
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
        ascent: 12,
        descent: 4,
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
        let mut out = Vec::new();
        let mut strings = Vec::new();
        fill_bar(
            &mut out,
            &mut strings,
            slots,
            &geom(fonts, width_cols, pad),
            cfg,
            fonts,
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

    #[test]
    fn a_long_title_truncates_from_the_end_with_a_faded_tail() {
        let cfg = TabBarConfig::default();
        let fonts = fonts();
        let title = "a-very-long-window-title-that-will-not-fit";
        let slots = lay(80, &[label(title, true)], &cfg);
        let out = paint(&fonts, &slots, &cfg, 80, 0);
        let runs = text_runs(&out);

        // A solid prefix plus one run per faded trailing cluster (FADE_CLUSTERS of
        // them), so more than one run and the tail glyphs are singletons.
        assert!(runs.len() >= 2, "prefix run plus per-cluster fade runs");
        let painted: String = runs.iter().map(|(t, _, _)| t.as_str()).collect();
        assert!(
            title.starts_with(&painted) && painted.len() < title.len(),
            "the start is kept and the end dropped: {painted:?}"
        );
        assert!(!painted.contains('…'), "no ellipsis, a fade instead");

        // The prefix keeps the full foreground; every trailing run is mixed toward
        // the background, and the very last is the most faded (nearest the bg).
        let fg = cfg.active.fg.to_u32();
        let bg = cfg.active.bg.to_u32();
        assert_eq!(runs[0].1, fg, "the prefix stays full strength");
        assert!(
            runs[1..].iter().all(|(_, c, _)| *c != fg),
            "faded runs are re-colored toward the background"
        );
        let dist = |c: u32| {
            let ch =
                |s: u32, m: u32| ((c >> s) & 0xff).abs_diff((bg >> s) & 0xff) as i32 * m as i32;
            ch(16, 1) + ch(8, 1) + ch(0, 1)
        };
        assert!(
            dist(runs.last().unwrap().1) < dist(runs[1].1),
            "the last glyph fades closest to the background"
        );
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
            ascent: 12,
            descent: 4,
        };
        let geom = BarGeom {
            metrics: METRICS,
            label: gapped,
            surface_width: 80 * METRICS.w,
            pad: 0,
            y: 0,
            h: 40,
        };
        fill_bar(&mut out, &mut strings, &slots, &geom, &cfg, &fonts);
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
}
