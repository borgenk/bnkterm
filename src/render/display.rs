//! The frame as data: a flat list of drawing primitives the painter emits and a
//! backend consumes. This is the seam the architecture notes call the "display
//! list" (section 1a). Two things hang off it:
//!
//! - **Damage tracking** (section 1b): because a frame is now a value, two frames
//!   can be compared. [`damage`] diffs the previous frame against the new one and
//!   returns just the screen rectangles that changed, so an idle caret blink
//!   presents a caret-sized damaged region instead of the whole surface, and a
//!   keystroke reports the edited line rather than the screen.
//! - **A backend-agnostic frame** (section 5): the list is turned into pixels by
//!   the GPU batcher ([`crate::render::gpu::build_frame`]), which consumes exactly these
//!   commands. Nothing here is Vulkan-specific.
//!
//! Commands are in *screen* space (already shifted by the scroll), so scrolling,
//! which moves every command, naturally diffs as "everything changed" and falls
//! back to a full repaint, exactly as the notes predict.
//!
//! ```text
//!   layout + params ──build_display_list──▶ DisplayList ──┬─ gpu::build_frame ─▶ vertices
//!                                                         └─ damage(old,new) ─▶ [Rect]
//! ```

use std::collections::HashMap;

use crate::platform::freetype::FaceKey;
use crate::platform::geom::Rect;

/// A frame, as the ordered primitives that draw it. Stacking is the list order:
/// background first, then panels, selection, glyphs, and the caret last.
pub type DisplayList = Vec<DrawCmd>;

/// Beyond this many distinct changed regions, repainting the whole surface is
/// cheaper than tracking and clipping each one.
const MAX_DAMAGE_RECTS: usize = 16;

/// Which corners a [`DrawCmd::RoundRect`] rounds. A fenced code block's panel is
/// drawn as one rounded rectangle: the GPU shader rounds whichever corners this
/// selects, so a single-line block rounds both ends and a multi-line block rounds
/// only its first and last rows.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RoundedCorners {
    None,
    Top,
    Bottom,
    Both,
}

/// One drawing primitive in screen (post-scroll) pixels. The set is deliberately
/// small and low level so a future GPU backend can execute the same list; the
/// painter decomposes higher-level chrome (an image's selection border, a button)
/// into these. Every variant is `Eq + Hash` so [`damage`] can diff two frames by
/// value, which is why text runs carry an owned [`String`] and a borrow-free
/// [`FaceKey`] rather than a `&str`/`&Face` tied to the rope and fonts.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum DrawCmd {
    /// A solid, opaque rectangle (background, selection, caret, quote bar, rule).
    Fill { rect: Rect, color: u32 },
    /// A solid rectangle with anti-aliased rounded corners (a code panel).
    RoundRect {
        rect: Rect,
        color: u32,
        radius: i32,
        corners: RoundedCorners,
    },
    /// A run of text drawn from `x` along `baseline`. `bounds` is the run's screen
    /// rectangle padded for glyph ink that overhangs the advance box, so the
    /// damage diff and the clip test never clip a glyph's edge.
    Text {
        bounds: Rect,
        x: i32,
        baseline: i32,
        face: FaceKey,
        color: u32,
        text: String,
    },
    /// A run of monospace cells drawn at a fixed pitch: the `i`th grapheme
    /// cluster in `text` is placed with its pen at `x + i * cell_w`, ignoring the
    /// glyph's own advance. This is the terminal's core primitive and the reason
    /// it exists separately from [`DrawCmd::Text`]: a terminal grid is fixed
    /// pitch, but a font's advance is fractional, so accumulating advances the way
    /// `Text` does would drift a whole-line run away from the integer cell grid
    /// (the cursor, drawn on the grid, would part company with the glyphs). Anchor
    /// each cell to its own column instead and the drift is gone by construction.
    /// Every cluster must be single-width (one column); the painter breaks a wide
    /// glyph out into its own [`DrawCmd::Text`] so the pitch stays uniform here.
    Cells {
        bounds: Rect,
        x: i32,
        baseline: i32,
        /// The fixed per-cell advance in pixels; cluster `i` draws at `x + i*cell_w`.
        cell_w: i32,
        face: FaceKey,
        color: u32,
        text: String,
    },
}

impl DrawCmd {
    /// The screen rectangle this command can touch: a region intersecting it must
    /// be repainted, and the command itself is skipped when it misses the clip.
    fn bounds(&self) -> Rect {
        match self {
            DrawCmd::Fill { rect, .. } | DrawCmd::RoundRect { rect, .. } => *rect,
            DrawCmd::Text { bounds, .. } | DrawCmd::Cells { bounds, .. } => *bounds,
        }
    }
}

/// The screen regions that differ between the previous frame `old` and the new
/// frame `new`, clamped to the `width` x `height` surface and free of overlaps.
///
/// A multiset difference does the work: a command appearing an unequal number of
/// times in the two frames contributes its bounds, so an identical command (same
/// geometry, face, colour, text) cancels and costs nothing, while a moved or
/// restyled run shows up as a removal plus an addition and damages both spots.
/// A frame that changed most of the surface (a scroll, a resize, a large edit)
/// returns a single full-surface rectangle, since repainting whole then beats
/// clipping many regions. An empty result means the two frames are identical.
pub fn damage(old: &[DrawCmd], new: &[DrawCmd], width: i32, height: i32) -> Vec<Rect> {
    // Dropping an equal pair (one command from each frame) never changes the
    // multiset difference, so strip the common prefix and suffix first. Lists are
    // built in document order, so a blink or a small edit differs only in a
    // handful of commands and the hash work below shrinks to just those;
    // equality bails on the first differing field, cheaper than hashing.
    let prefix = old.iter().zip(new).take_while(|(a, b)| a == b).count();
    let (old, new) = (&old[prefix..], &new[prefix..]);
    let suffix = old
        .iter()
        .rev()
        .zip(new.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let (old, new) = (&old[..old.len() - suffix], &new[..new.len() - suffix]);

    let mut counts: HashMap<&DrawCmd, i32> = HashMap::new();
    for c in old {
        *counts.entry(c).or_insert(0) += 1;
    }
    for c in new {
        *counts.entry(c).or_insert(0) -= 1;
    }
    let surface = Rect {
        x: 0,
        y: 0,
        w: width.max(0),
        h: height.max(0),
    };
    let mut rects: Vec<Rect> = counts
        .into_iter()
        .filter(|&(_, n)| n != 0)
        .filter_map(|(c, _)| intersection(c.bounds(), surface))
        .collect();
    coalesce(&mut rects);
    let area: i64 = rects.iter().map(|r| r.w as i64 * r.h as i64).sum();
    let surface_area = surface.w as i64 * surface.h as i64;
    if rects.len() > MAX_DAMAGE_RECTS || area * 2 > surface_area {
        return if surface_area > 0 {
            vec![surface]
        } else {
            Vec::new()
        };
    }
    rects
}

/// Merge any two overlapping rectangles into their bounding box, repeated until
/// none overlap, so a region is never cleared and repainted twice. The damage set
/// is tiny in practice, so the quadratic scan never matters.
fn coalesce(rects: &mut Vec<Rect>) {
    let mut merged = true;
    while merged {
        merged = false;
        'scan: for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                if intersects(rects[i], rects[j]) {
                    rects[i] = union(rects[i], rects[j]);
                    rects.remove(j);
                    merged = true;
                    break 'scan;
                }
            }
        }
    }
}

/// Whether two rectangles share any interior pixel (touching edges do not count):
/// exactly when their [`intersection`] is non-empty.
fn intersects(a: Rect, b: Rect) -> bool {
    intersection(a, b).is_some()
}

/// The smallest rectangle covering both `a` and `b`.
fn union(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 = (a.x + a.w).max(b.x + b.w);
    let y1 = (a.y + a.h).max(b.y + b.h);
    Rect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    }
}

/// The overlap of `a` and `b`, or `None` when they are disjoint.
fn intersection(a: Rect, b: Rect) -> Option<Rect> {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w).min(b.x + b.w);
    let y1 = (a.y + a.h).min(b.y + b.h);
    (x0 < x1 && y0 < y1).then_some(Rect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(x: i32, y: i32, w: i32, h: i32, color: u32) -> DrawCmd {
        DrawCmd::Fill {
            rect: Rect { x, y, w, h },
            color,
        }
    }

    const W: i32 = 800;
    const H: i32 = 600;

    #[test]
    fn identical_frames_have_no_damage() {
        let frame = vec![fill(0, 0, W, H, 0x010101), fill(10, 10, 5, 5, 0xffffff)];
        assert!(damage(&frame, &frame, W, H).is_empty());
    }

    #[test]
    fn a_caret_toggle_damages_only_the_caret() {
        // The blink case: the frames differ by one small Fill (the caret).
        let bg = fill(0, 0, W, H, 0x010101);
        let caret = fill(100, 50, 2, 18, 0xffffff);
        let off = vec![bg.clone()];
        let on = vec![bg, caret];
        assert_eq!(
            damage(&off, &on, W, H),
            vec![Rect {
                x: 100,
                y: 50,
                w: 2,
                h: 18
            }],
            "only the caret rectangle is damaged"
        );
    }

    #[test]
    fn a_moved_command_damages_both_positions() {
        // A run that shifts (an edit before it) appears as a remove plus an add;
        // the two non-overlapping spots both need repainting.
        let bg = fill(0, 0, W, H, 0x010101);
        let before = vec![bg.clone(), fill(10, 10, 4, 4, 0xaa)];
        let after = vec![bg, fill(500, 400, 4, 4, 0xaa)];
        let mut d = damage(&before, &after, W, H);
        d.sort_by_key(|r| r.x);
        assert_eq!(
            d,
            vec![
                Rect {
                    x: 10,
                    y: 10,
                    w: 4,
                    h: 4
                },
                Rect {
                    x: 500,
                    y: 400,
                    w: 4,
                    h: 4
                },
            ]
        );
    }

    #[test]
    fn overlapping_changes_coalesce_into_one_region() {
        let bg = fill(0, 0, W, H, 0x010101);
        let old = vec![bg.clone()];
        let new = vec![bg, fill(10, 10, 20, 20, 0x01), fill(20, 20, 20, 20, 0x02)];
        // The two added fills overlap, so the damage is their bounding box.
        assert_eq!(
            damage(&old, &new, W, H),
            vec![Rect {
                x: 10,
                y: 10,
                w: 30,
                h: 30
            }]
        );
    }

    #[test]
    fn a_large_change_falls_back_to_a_full_repaint() {
        // Two big disjoint fills cover more than half the surface, so the diff
        // collapses to one whole-surface rectangle rather than tracking regions.
        let old: Vec<DrawCmd> = Vec::new();
        let new = vec![fill(0, 0, W, H / 2 + 10, 0x01), fill(0, H / 2, W, H, 0x02)];
        assert_eq!(
            damage(&old, &new, W, H),
            vec![Rect {
                x: 0,
                y: 0,
                w: W,
                h: H
            }]
        );
    }

    #[test]
    fn a_changed_cells_run_damages_its_bounds() {
        // The terminal's line primitive diffs like any other command: an
        // identical frame is free, and editing the run's text damages exactly the
        // run's padded bounds rectangle.
        let bounds = Rect {
            x: 0,
            y: 0,
            w: 200,
            h: 20,
        };
        let run = |text: &str| DrawCmd::Cells {
            bounds,
            x: 0,
            baseline: 16,
            cell_w: 10,
            face: FaceKey::Code { size: 16 },
            color: 0x00ff_ffff,
            text: text.to_string(),
        };
        let old = vec![run("hello")];
        assert!(damage(&old, &old, W, H).is_empty(), "identical run is free");
        assert_eq!(
            damage(&old, &[run("hEllo")], W, H),
            vec![bounds],
            "an edited run damages its bounds"
        );
    }

    #[test]
    fn damage_is_clamped_to_the_surface() {
        // A command partly off-screen damages only its on-surface part; one fully
        // off-screen contributes nothing.
        let on = fill(-5, -5, 20, 20, 0x01);
        let off = fill(W + 100, 0, 10, 10, 0x02);
        let new = vec![on, off];
        assert_eq!(
            damage(&[], &new, W, H),
            vec![Rect {
                x: 0,
                y: 0,
                w: 15,
                h: 15
            }]
        );
    }
}
