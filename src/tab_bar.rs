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
use crate::platform::freetype::{FaceKey, FontStyle};
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

/// How many trailing columns of a truncated label fade toward the tab background.
/// The fade signals "there is more" in place of an ellipsis.
const FADE_CELLS: usize = 4;

/// The device-pixel rectangle the strip occupies, plus the cell metrics and left
/// inset its contents lay out against. Computed window-side (it depends on the
/// surface size and the top/bottom config) and handed to the tabs layer on every
/// resize, so painting stays a pure function of it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BarGeom {
    pub metrics: CellMetrics,
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

/// One tab's half-open cell range and safe, fitted label.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Slot {
    /// The whole tab block, for the background fill and pointer hit testing.
    pub cells: Range<usize>,
    /// The fitted label, cut from the end (never past a cluster boundary).
    pub text: String,
    /// Absolute column the label begins at: centered when it fits, one gutter in
    /// when it was truncated (and therefore fills the content area).
    pub text_start: usize,
    /// Label width in columns.
    pub text_cells: usize,
    /// The label was cut to fit, so its trailing cells fade.
    pub truncated: bool,
    pub active: bool,
}

/// Give every tab one equal width, clamped into the config bounds, and lay them
/// out left-aligned; the space past the last tab is left to the strip background.
/// Each title is sanitized (child-controlled bytes), stripped of its old numeric
/// prefix, and either centered whole or cut from the end to fit.
pub(crate) fn layout(cols: usize, labels: &[TabLabel<'_>], cfg: &TabBarConfig) -> Vec<Slot> {
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
            let content = avail.saturating_sub(2 * pad);

            let title = sanitize_title(label.title);
            let source = if title.is_empty() { "shell" } else { &title };
            let (text, truncated) = fit_end(source, content);
            let text_cells = text_width(&text);
            // Center a label that fits; a truncated one fills the content area, so
            // it just sits one gutter in from the left edge.
            let left = if truncated {
                pad
            } else {
                pad + avail.saturating_sub(2 * pad).saturating_sub(text_cells) / 2
            };
            Slot {
                cells,
                text,
                text_start: start + left,
                text_cells,
                truncated,
                active: label.active,
            }
        })
        .collect()
}

/// Paint the strip: a full-width background, each active tab's block, every
/// label, then the dividers between adjacent inactive tabs.
pub(crate) fn fill_bar(
    out: &mut DisplayList,
    strings: &mut Vec<String>,
    slots: &[Slot],
    bar: &BarGeom,
    cfg: &TabBarConfig,
) {
    let m = bar.metrics;
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
    let face = FaceKey::Prose {
        size: m.size,
        style: FontStyle::Regular,
    };
    // Center the glyph ink box (ascent + descent) in the strip, not the full line
    // box (`m.h`): the line height carries the font's line gap, which sits below
    // the descent, so centering `m.h` would bias the text upward.
    let baseline = bar.y + (bar.h - m.ascent - m.descent) / 2 + m.ascent;

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
        let text_x = bar.pad + slot.text_start as i32 * m.w;
        paint_label(
            out,
            strings,
            &slot.text,
            slot.truncated,
            text_x,
            baseline,
            m,
            face,
            colors,
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

/// Emit a label, fading its trailing [`FADE_CELLS`] columns toward the tab
/// background when it was truncated. An untruncated label is one batched run; a
/// truncated one is the solid prefix plus one small run per faded cell, each
/// re-colored a step further toward the background so the text dissolves rather
/// than ending abruptly. The fade only runs on truncated tabs, off the steady
/// path.
#[allow(clippy::too_many_arguments)]
fn paint_label(
    out: &mut DisplayList,
    strings: &mut Vec<String>,
    text: &str,
    truncated: bool,
    x: i32,
    baseline: i32,
    m: CellMetrics,
    face: FaceKey,
    colors: TabColors,
) {
    let fg = colors.fg.to_u32();
    let bg = colors.bg.to_u32();
    if !truncated {
        term_render::push_cell_text(out, strings, text, x, baseline, m, face, fg, bg);
        return;
    }

    let total = text_width(text);
    let fade_start = total.saturating_sub(FADE_CELLS);
    // The byte offset and column of the first cluster at or past the fade edge.
    let mut prefix_bytes = text.len();
    let mut prefix_col = total;
    let mut col = 0usize;
    for (offset, cluster) in grapheme::graphemes(text) {
        if col >= fade_start {
            prefix_bytes = offset;
            prefix_col = col;
            break;
        }
        col += term_render::display_cluster_width(cluster).max(1);
    }
    // The solid prefix as one batched run.
    if prefix_bytes > 0 {
        term_render::push_cell_text(
            out,
            strings,
            &text[..prefix_bytes],
            x,
            baseline,
            m,
            face,
            fg,
            bg,
        );
    }
    // Each faded cluster at its own column, a step deeper toward the background.
    let denom = FADE_CELLS as u16 + 1;
    let mut col = prefix_col;
    for (offset, cluster) in grapheme::graphemes(&text[prefix_bytes..]) {
        let depth = (col - fade_start).min(FADE_CELLS - 1) as u16;
        let faded = colors.fg.mix(colors.bg, depth + 1, denom);
        let cell_x = x + col as i32 * m.w;
        let start = prefix_bytes + offset;
        term_render::push_cell_text(
            out,
            strings,
            &text[start..start + cluster.len()],
            cell_x,
            baseline,
            m,
            face,
            faded.to_u32(),
            bg,
        );
        col += term_render::display_cluster_width(cluster).max(1);
    }
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

/// Truncate `text` to at most `max_cells` columns, cutting from the *end* (the
/// start is kept) at a cluster boundary, with no ellipsis. Returns the fitted
/// text and whether anything was cut.
fn fit_end(text: &str, max_cells: usize) -> (String, bool) {
    if text_width(text) <= max_cells {
        return (text.to_string(), false);
    }
    let mut fitted = String::new();
    let mut used = 0usize;
    for (_, cluster) in grapheme::graphemes(text) {
        let width = term_render::display_cluster_width(cluster).max(1);
        if used + width > max_cells {
            break;
        }
        fitted.push_str(cluster);
        used += width;
    }
    (fitted, true)
}

fn text_width(text: &str) -> usize {
    grapheme::graphemes(text)
        .map(|(_, cluster)| term_render::display_cluster_width(cluster).max(1))
        .sum()
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

    fn bar(width_cols: usize, pad: i32) -> BarGeom {
        BarGeom {
            metrics: METRICS,
            surface_width: pad + width_cols as i32 * METRICS.w + pad,
            pad,
            y: 0,
            h: METRICS.h,
        }
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
        let slots = layout(80, &labels, &cfg);
        assert_eq!(slots[0].cells, 0..24);
        assert_eq!(slots[1].cells, 24..48);
        assert_eq!(slots[2].cells, 48..72);
    }

    #[test]
    fn width_is_capped_at_max_and_floored_at_min() {
        let cfg = TabBarConfig::default(); // min 10, max 24
                                           // Two tabs in a wide bar cap at max_width, not half the window each.
        let two = layout(200, &[label("a", true), label("b", false)], &cfg);
        assert_eq!(two[0].cells.len(), 24);
        assert_eq!(two[1].cells.len(), 24);
        // Eight tabs in 80 cols land exactly at the floor (80 / 8 == 10).
        let eight: Vec<_> = (0..8).map(|_| label("t", false)).collect();
        let slots = layout(80, &eight, &cfg);
        assert!(slots.iter().all(|s| s.cells.len() == 10));
    }

    #[test]
    fn too_many_tabs_fall_back_to_an_equal_fill_below_min() {
        let cfg = TabBarConfig::default(); // min 10
                                           // Twenty tabs cannot each be 10 wide in 80 cols; they fill equally at 4.
        let many: Vec<_> = (0..20).map(|_| label("t", false)).collect();
        let slots = layout(80, &many, &cfg);
        assert!(slots.iter().all(|s| s.cells.len() == 4));
    }

    #[test]
    fn labels_carry_no_numeric_prefix_and_center_when_they_fit() {
        let cfg = TabBarConfig::default();
        let slots = layout(80, &[label("hi", true), label("there", false)], &cfg);
        assert_eq!(slots[0].text, "hi");
        assert_eq!(slots[1].text, "there");
        // "hi" (2 cells) centered in a 24-wide tab (22 content cols): 1 gutter +
        // (22 - 2) / 2 == 11 in from the left edge (cell 0).
        assert_eq!(slots[0].text_start, 1 + 10);
    }

    #[test]
    fn empty_title_falls_back_to_shell() {
        let cfg = TabBarConfig::default();
        assert_eq!(layout(80, &[label("", true)], &cfg)[0].text, "shell");
    }

    #[test]
    fn long_titles_are_cut_from_the_end_without_an_ellipsis() {
        let cfg = TabBarConfig::default();
        let slots = layout(80, &[label("a-very-long-window-title", true)], &cfg);
        // 24-wide tab, 22 content cols: the start is kept, the end dropped, no "…".
        assert_eq!(slots[0].text, "a-very-long-window-tit");
        assert!(slots[0].truncated);
        assert!(!slots[0].text.contains('…'));
    }

    #[test]
    fn titles_are_cluster_safe_and_truncated_by_cell_width() {
        let cfg = TabBarConfig::default();
        // A four-cell content area (tab of 6, one gutter each side).
        let cfg = TabBarConfig {
            max_width: 6,
            min_width: 6,
            ..cfg
        };
        let slot = layout(6, &[label("世😀e\u{301}tail", true)], &cfg).remove(0);
        // 世(2) 😀(2) fills the four content cells; the rest is dropped.
        assert_eq!(slot.text, "世😀");
        assert_eq!(slot.text_cells, 4);
        assert!(slot.truncated);
    }

    #[test]
    fn controls_and_zero_width_clusters_are_replaced() {
        let cfg = TabBarConfig::default();
        let slot = layout(80, &[label("bad\n\u{200b}title", false)], &cfg).remove(0);
        assert_eq!(slot.text, "bad\u{fffd}\u{fffd}title");
        assert_eq!(slot.text.matches('\u{fffd}').count(), 2);
    }

    #[test]
    fn hit_testing_uses_half_open_ranges() {
        let cfg = TabBarConfig {
            min_width: 5,
            max_width: 5,
            ..TabBarConfig::default()
        };
        let slots = layout(10, &[label("a", true), label("b", false)], &cfg);
        assert_eq!(hit_test(&slots, 0), Some(0));
        assert_eq!(hit_test(&slots, 4), Some(0));
        assert_eq!(hit_test(&slots, 5), Some(1));
        assert_eq!(hit_test(&slots, 10), None);
    }

    #[test]
    fn painting_is_a_strip_fill_then_the_active_block_and_labels() {
        let cfg = TabBarConfig::default();
        let slots = layout(80, &[label("one", true), label("two", false)], &cfg);
        let mut out = Vec::new();
        let mut strings = Vec::new();
        fill_bar(&mut out, &mut strings, &slots, &bar(80, 5), &cfg);

        // Strip background first.
        assert!(matches!(out[0], DrawCmd::Fill { .. }));
        // Exactly one tab-background fill: the active block (inactive tabs share
        // the strip background and are not re-filled).
        let tab_bgs = out
            .iter()
            .filter(|c| matches!(c, DrawCmd::Fill { rect, .. } if rect.h == METRICS.h))
            .count();
        assert_eq!(tab_bgs, 2, "strip background + one active block");
        // Both labels reach the list as text runs, in distinct foreground colors,
        // each weighted against its own tab background (not the strip background):
        // the glyph anti-aliasing thickens the active tab's dark-on-light text, so
        // the run must carry the active block's own background, not the dark strip.
        let runs: Vec<(u32, u32)> = out
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Cells { color, bg, .. } => Some((*color, *bg)),
                _ => None,
            })
            .collect();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0], (cfg.active.fg.to_u32(), cfg.active.bg.to_u32()));
        assert_eq!(
            runs[1],
            (cfg.inactive.fg.to_u32(), cfg.inactive.bg.to_u32())
        );
    }

    #[test]
    fn a_truncated_label_fades_its_trailing_cells() {
        let cfg = TabBarConfig::default();
        let slots = layout(80, &[label("a-very-long-window-title", true)], &cfg);
        let mut out = Vec::new();
        let mut strings = Vec::new();
        fill_bar(&mut out, &mut strings, &slots, &bar(80, 0), &cfg);

        // The trailing cells carry progressively background-mixed colors, all
        // distinct from the solid foreground; the very last is the most faded.
        let text_colors: Vec<u32> = out
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Cells { color, x, .. } => Some((*x, *color)),
                _ => None,
            })
            .map(|(_, color)| color)
            .collect();
        assert!(text_colors.len() > 1, "prefix run plus per-cell fade runs");
        let fg = cfg.active.fg.to_u32();
        assert_eq!(text_colors[0], fg, "the prefix stays full strength");
        assert!(
            text_colors.iter().skip(1).all(|c| *c != fg),
            "faded cells are re-colored toward the background"
        );
    }

    #[test]
    fn a_divider_separates_two_inactive_tabs_but_not_the_active_one() {
        let cfg = TabBarConfig {
            min_width: 8,
            max_width: 8,
            ..TabBarConfig::default()
        };
        // Three tabs, the middle active: divider between 0|1 is hidden (1 active),
        // 1|2 hidden (1 active); with the active in the middle, no divider shows.
        let mid = layout(
            24,
            &[label("a", false), label("b", true), label("c", false)],
            &cfg,
        );
        let mut out = Vec::new();
        let mut strings = Vec::new();
        fill_bar(&mut out, &mut strings, &mid, &bar(24, 0), &cfg);
        let dividers = out
            .iter()
            .filter(|c| matches!(c, DrawCmd::Fill { rect, .. } if rect.w == 1))
            .count();
        assert_eq!(dividers, 0, "both boundaries touch the active middle tab");

        // First tab active: only the 1|2 boundary (both inactive) shows a divider.
        let first = layout(
            24,
            &[label("a", true), label("b", false), label("c", false)],
            &cfg,
        );
        let mut out = Vec::new();
        let mut strings = Vec::new();
        fill_bar(&mut out, &mut strings, &first, &bar(24, 0), &cfg);
        let dividers = out
            .iter()
            .filter(|c| matches!(c, DrawCmd::Fill { rect, .. } if rect.w == 1))
            .count();
        assert_eq!(dividers, 1, "one divider between the two inactive tabs");
    }

    #[test]
    fn labels_center_the_ink_box_not_the_line_box() {
        let cfg = TabBarConfig::default();
        let slots = layout(80, &[label("a", true), label("b", false)], &cfg);
        let mut out = Vec::new();
        let mut strings = Vec::new();
        // A metrics with a line gap (h = 20 > ascent + descent = 16) in a tall
        // strip: the baseline centers the ink box, so the gap never biases it up.
        let gapped = CellMetrics {
            size: 16,
            w: 8,
            h: 20,
            ascent: 12,
            descent: 4,
        };
        let geom = BarGeom {
            metrics: gapped,
            surface_width: 80 * gapped.w,
            pad: 0,
            y: 0,
            h: 40,
        };
        fill_bar(&mut out, &mut strings, &slots, &geom, &cfg);
        let baseline = out.iter().find_map(|c| match c {
            DrawCmd::Cells { baseline, .. } => Some(*baseline),
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

    #[test]
    fn wide_clusters_never_enter_cells_runs() {
        let cfg = TabBarConfig::default();
        let slots = layout(80, &[label("世😀", true)], &cfg);
        let mut out = Vec::new();
        let mut strings = Vec::new();
        fill_bar(&mut out, &mut strings, &slots, &bar(80, 0), &cfg);
        assert!(out.iter().any(|cmd| matches!(cmd, DrawCmd::Text { .. })));
        for cmd in &out {
            if let DrawCmd::Cells { text, .. } = cmd {
                assert!(grapheme::graphemes(text)
                    .all(|(_, cluster)| term_render::display_cluster_width(cluster) == 1));
            }
        }
    }
}
