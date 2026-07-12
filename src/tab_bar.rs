//! Pure tab-bar layout, sanitization, painting, and hit testing.
//!
//! Titles originate in OSC 0/2 and are therefore child-controlled. Layout works
//! in terminal cells, segments titles by grapheme cluster, replaces controls and
//! zero-width clusters, and truncates only at cluster boundaries. Painting then
//! emits ordinary display-list commands into the same pooled strings as the grid.

use std::ops::Range;

use crate::color::{Rgb, Theme};
use crate::platform::freetype::{FaceKey, FontStyle};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::render::display::{DisplayList, DrawCmd};
use crate::term_render::{self, CellMetrics};

/// Width below which a slot collapses to its numeric index rather than attempting
/// to show a title.
const MIN_SLOT: usize = 4;

/// One resolved title supplied by the tabs manager.
pub(crate) struct TabLabel<'a> {
    pub title: &'a str,
    pub active: bool,
}

/// One tab's half-open cell range and safe, fitted label.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Slot {
    pub cells: Range<usize>,
    pub text: String,
    pub text_cells: usize,
    pub active: bool,
}

/// Divide `cols` equally among the labels (leftmost slots receive leftovers),
/// sanitize each title, and fit it to its slot with a trailing ellipsis.
pub(crate) fn layout(cols: usize, labels: &[TabLabel<'_>]) -> Vec<Slot> {
    if labels.is_empty() {
        return Vec::new();
    }
    let base = cols / labels.len();
    let extra = cols % labels.len();
    let mut cursor = 0usize;
    labels
        .iter()
        .enumerate()
        .map(|(index, label)| {
            let width = base + usize::from(index < extra);
            let cells = cursor..cursor + width;
            cursor += width;
            let source = if width < MIN_SLOT {
                (index + 1).to_string()
            } else {
                let title = sanitize_title(label.title);
                format!(
                    "{}:{}",
                    index + 1,
                    if title.is_empty() { "shell" } else { &title }
                )
            };
            let text = fit(&source, width);
            let text_cells = text_width(&text);
            Slot {
                cells,
                text,
                text_cells,
                active: label.active,
            }
        })
        .collect()
}

/// Append a full-width bar fill, then each visible slot and its label.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_bar(
    out: &mut DisplayList,
    strings: &mut Vec<String>,
    slots: &[Slot],
    metrics: CellMetrics,
    theme: &Theme,
    surface_width: i32,
    pad: i32,
) {
    let inactive_bg = mix(theme.bg, theme.fg, 1, 10);
    let inactive_fg = dim(theme.fg);
    out.push(DrawCmd::Fill {
        rect: Rect {
            x: 0,
            y: pad,
            w: surface_width.max(0),
            h: metrics.h,
        },
        color: inactive_bg.to_u32(),
    });
    let face = FaceKey::Prose {
        size: metrics.size,
        style: FontStyle::Regular,
    };
    let baseline = pad + metrics.ascent;
    for slot in slots.iter().filter(|slot| !slot.cells.is_empty()) {
        let (fg, bg) = if slot.active {
            (theme.fg, theme.bg)
        } else {
            (inactive_fg, inactive_bg)
        };
        let x = pad + slot.cells.start as i32 * metrics.w;
        out.push(DrawCmd::Fill {
            rect: Rect {
                x,
                y: pad,
                w: (slot.cells.end - slot.cells.start) as i32 * metrics.w,
                h: metrics.h,
            },
            color: bg.to_u32(),
        });
        term_render::push_cell_text(
            out,
            strings,
            &slot.text,
            x,
            baseline,
            metrics,
            face,
            fg.to_u32(),
            bg.to_u32(),
        );
    }
}

/// Tab index under `col`, if any. Empty ranges (more tabs than columns) are not
/// hittable.
pub(crate) fn hit_test(slots: &[Slot], col: usize) -> Option<usize> {
    slots.iter().position(|slot| slot.cells.contains(&col))
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

fn fit(text: &str, max_cells: usize) -> String {
    if max_cells == 0 {
        return String::new();
    }
    if text_width(text) <= max_cells {
        return text.to_string();
    }
    if max_cells == 1 {
        return "…".to_string();
    }
    let target = max_cells - 1;
    let mut fitted = String::new();
    let mut used = 0usize;
    for (_, cluster) in grapheme::graphemes(text) {
        let width = term_render::display_cluster_width(cluster).max(1);
        if used + width > target {
            break;
        }
        fitted.push_str(cluster);
        used += width;
    }
    fitted.push('…');
    fitted
}

fn text_width(text: &str) -> usize {
    grapheme::graphemes(text)
        .map(|(_, cluster)| term_render::display_cluster_width(cluster).max(1))
        .sum()
}

fn dim(color: Rgb) -> Rgb {
    let channel = |value: u8| ((u16::from(value) * 2) / 3) as u8;
    Rgb::new(channel(color.r), channel(color.g), channel(color.b))
}

fn mix(a: Rgb, b: Rgb, b_parts: u16, total: u16) -> Rgb {
    let channel = |av: u8, bv: u8| {
        ((u16::from(av) * (total - b_parts) + u16::from(bv) * b_parts) / total) as u8
    };
    Rgb::new(channel(a.r, b.r), channel(a.g, b.g), channel(a.b, b.b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_divides_cells_and_gives_leftovers_to_the_left() {
        let labels = [
            TabLabel {
                title: "one",
                active: true,
            },
            TabLabel {
                title: "two",
                active: false,
            },
            TabLabel {
                title: "three",
                active: false,
            },
        ];
        let slots = layout(20, &labels);
        assert_eq!(slots[0].cells, 0..7);
        assert_eq!(slots[1].cells, 7..14);
        assert_eq!(slots[2].cells, 14..20);
    }

    #[test]
    fn narrow_slots_collapse_to_indices() {
        let labels: Vec<_> = (0..17)
            .map(|_| TabLabel {
                title: "long title",
                active: false,
            })
            .collect();
        let slots = layout(17, &labels);
        assert!(slots.iter().all(|slot| slot.cells.len() == 1));
        assert!(slots.iter().all(|slot| slot.text_cells <= 1));
    }

    #[test]
    fn titles_are_cluster_safe_and_truncated_by_cell_width() {
        let labels = [TabLabel {
            title: "世😀e\u{301}tail",
            active: true,
        }];
        let slot = layout(9, &labels).remove(0);
        assert_eq!(slot.text, "1:世😀e\u{301}t…");
        assert_eq!(slot.text_cells, 9);
    }

    #[test]
    fn controls_and_zero_width_clusters_are_replaced() {
        let labels = [TabLabel {
            title: "bad\n\u{200b}title",
            active: false,
        }];
        let slot = layout(30, &labels).remove(0);
        assert_eq!(slot.text, "1:bad��title");
        assert_eq!(slot.text.matches('\u{fffd}').count(), 2);
    }

    #[test]
    fn empty_title_falls_back_to_shell() {
        let labels = [TabLabel {
            title: "",
            active: true,
        }];
        assert_eq!(layout(20, &labels)[0].text, "1:shell");
    }

    #[test]
    fn hit_testing_uses_half_open_ranges() {
        let labels = [
            TabLabel {
                title: "a",
                active: true,
            },
            TabLabel {
                title: "b",
                active: false,
            },
        ];
        let slots = layout(10, &labels);
        assert_eq!(hit_test(&slots, 0), Some(0));
        assert_eq!(hit_test(&slots, 4), Some(0));
        assert_eq!(hit_test(&slots, 5), Some(1));
        assert_eq!(hit_test(&slots, 10), None);
    }

    #[test]
    fn painting_is_fill_then_one_fill_and_text_run_per_ascii_slot() {
        let labels = [
            TabLabel {
                title: "one",
                active: true,
            },
            TabLabel {
                title: "two",
                active: false,
            },
        ];
        let slots = layout(20, &labels);
        let metrics = CellMetrics {
            size: 16,
            w: 8,
            h: 16,
            ascent: 12,
            descent: 4,
        };
        let mut out = Vec::new();
        let mut strings = Vec::new();
        fill_bar(
            &mut out,
            &mut strings,
            &slots,
            metrics,
            &Theme::default(),
            170,
            5,
        );

        assert_eq!(out.len(), 5);
        assert!(matches!(out[0], DrawCmd::Fill { .. }));
        assert!(matches!(out[1], DrawCmd::Fill { .. }));
        assert!(matches!(out[2], DrawCmd::Cells { .. }));
        assert!(matches!(out[3], DrawCmd::Fill { .. }));
        assert!(matches!(out[4], DrawCmd::Cells { .. }));
        let active_color = match &out[2] {
            DrawCmd::Cells { color, .. } => *color,
            _ => unreachable!(),
        };
        let inactive_color = match &out[4] {
            DrawCmd::Cells { color, .. } => *color,
            _ => unreachable!(),
        };
        assert_ne!(active_color, inactive_color);
    }

    #[test]
    fn wide_clusters_never_enter_cells_runs() {
        let slots = layout(
            20,
            &[TabLabel {
                title: "世😀",
                active: true,
            }],
        );
        let metrics = CellMetrics {
            size: 16,
            w: 8,
            h: 16,
            ascent: 12,
            descent: 4,
        };
        let mut out = Vec::new();
        let mut strings = Vec::new();
        fill_bar(
            &mut out,
            &mut strings,
            &slots,
            metrics,
            &Theme::default(),
            160,
            0,
        );
        assert!(out.iter().any(|cmd| matches!(cmd, DrawCmd::Text { .. })));
        for cmd in &out {
            if let DrawCmd::Cells { text, .. } = cmd {
                assert!(grapheme::graphemes(text)
                    .all(|(_, cluster)| { term_render::display_cluster_width(cluster) == 1 }));
            }
        }
    }
}
