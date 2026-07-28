//! The GPU backend's CPU half: turning a display list into GPU-consumable
//! data. This module is deliberately free of Vulkan: it packs glyph rasters
//! into atlas mirrors and emits vertices and draw batches as plain values, so
//! the geometry and packing logic is unit-testable without a device.
//! `vulkan.rs` consumes the output ([`FrameData`]) verbatim.
//!
//! ```text
//!   &[DrawCmd] ─┬─ Fill/RoundRect ──────────────▶ quads (solid / SDF corners)
//!               └─ Text ─▶ FreeType rasters ─▶ atlas slots ─▶ textured quads
//! ```
//!
//! Text traversal matches how layout measured each run (the ASCII fast path,
//! then grapheme clusters routed to the emoji or per-character path), so the
//! pixels the GPU composites land exactly where the caret and wrapping expect.

use std::collections::HashMap;

use crate::platform::bytes;
use crate::platform::freetype::{Face, FaceKey, Fonts};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::render::boxdraw;
use crate::render::display::DrawCmd;
use crate::render::display::Fade;
use crate::render::display::RoundedCorners;

/// Vertex modes, matching shaders/quad.frag. (Mode 3, the decoded-image path,
/// is excised: a terminal draws no images.)
pub const MODE_SOLID: u32 = 0;
pub const MODE_GLYPH: u32 = 1;
pub const MODE_EMOJI: u32 = 2;
pub const MODE_ROUND: u32 = 4;

/// How hard to correct the two opposite ways linear-light compositing misweights
/// anti-aliased text. Two dials, not one, because they pull in opposite directions:
/// see [`coverage_exponent`], which turns them into a run's exponent, and
/// [`crate::app::TEXT_GAMMA`], which holds the shipping values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextGamma {
    /// Thins light-on-dark text. `> 1` thins, `1.0` is no correction.
    pub light_on_dark: f32,
    /// Thickens dark-on-light text, in proportion to the run's contrast. `> 1`
    /// thickens (it is raised to a negative power), `1.0` is no correction.
    pub dark_on_light: f32,
}

/// A glyph run's paint, threaded through the glyph emitters as one value: the
/// foreground `color`, the coverage `exponent` derived once from its contrast with
/// the run background (see [`coverage_exponent`]), and the optional ink ramp that
/// dissolves its tail (see [`Fade`]).
///
/// `exponent` is derived from the run's *own* colour against its background, and the
/// fade deliberately leaves that colour alone: were the fade a colour mix toward the
/// background instead, it would collapse the very contrast this exponent reads, and a
/// label's stroke weight would drift along its own tail.
#[derive(Clone, Copy)]
struct Paint {
    color: u32,
    exponent: f32,
    fade: Option<Fade>,
}

impl Paint {
    fn new(color: u32, bg: u32, fade: Option<Fade>, gamma: TextGamma) -> Self {
        Paint {
            color,
            exponent: coverage_exponent(color, bg, gamma),
            fade,
        }
    }
}

/// Atlases start here and double when full, up to [`ATLAS_MAX`]; growth wipes
/// the atlas (sources are cached in `Fonts`, so re-inserting is cheap) and the
/// frame build restarts so no stale coordinates survive.
const ATLAS_START: u32 = 1024;
const ATLAS_MAX: u32 = 8192;

/// The most slot records a glyph cache keeps before it is dropped whole. The atlas
/// is bounded by [`ATLAS_MAX`], but the slot maps are not: a `None` record is kept
/// for every distinct non-emoji cluster (the negative cache) and for every glyph
/// that would not fit a full atlas, so a hostile stream of distinct clusters or
/// glyphs — attacker-controlled bytes, per the threat model — would grow them
/// without limit. Far above any real workload's distinct-glyph count, so ordinary
/// use never trips it; when it does, dropping the map bounds memory and the glyphs
/// simply re-cache (a bounded burst) rather than accumulating forever.
const MAX_GLYPH_CACHE: usize = 1 << 16;

/// One vertex, laid out exactly as the pipeline's vertex input expects
/// (`repr(C)`: pos, uv, color, extra, mode). Six of these make a quad.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub extra: [f32; 4],
    pub mode: u32,
}

/// A contiguous vertex range drawn with one descriptor binding. Every quad binds
/// the same atlases (set 0), so a frame is a single batch; the type stays a range
/// so the renderer's draw loop and the (future) clipped-repaint path are unchanged.
#[derive(Debug, PartialEq, Eq)]
pub struct Batch {
    pub start: u32,
    pub count: u32,
}

/// A rectangle of an atlas that changed and needs a texture upload, with its
/// bytes extracted row-contiguous from the mirror.
pub struct AtlasUpload {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub bytes: Vec<u8>,
}

/// Everything the Vulkan renderer needs to draw one frame. Reused across frames by
/// [`build_frame_into`]: its `vertices`/`batches` are cleared and refilled, so a
/// steady frame keeps their capacity instead of reallocating.
#[derive(Default)]
pub struct FrameData {
    pub vertices: Vec<Vertex>,
    pub batches: Vec<Batch>,
    /// Atlas geometry and generation, so the renderer can (re)create textures.
    pub glyph_atlas: AtlasInfo,
    pub emoji_atlas: AtlasInfo,
    /// Dirty-region uploads for the two atlases, if any.
    pub glyph_upload: Option<AtlasUpload>,
    pub emoji_upload: Option<AtlasUpload>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AtlasInfo {
    pub width: u32,
    pub height: u32,
    /// Bumped whenever the atlas is wiped and regrown; the renderer recreates
    /// the texture (and re-uploads in full) when it changes.
    pub generation: u32,
}

/// Where a raster landed in an atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// A glyph's atlas placement plus its pen metrics, cached beside the slot so a
/// steady-state frame emits the quad without re-rasterising the glyph. `slot` is
/// `None` for an inkless glyph (a space, or an emoji that shaped to no ink) or
/// one the atlas could not fit; the pen still advances by `advance`, so the
/// caret and the glyphs after it land where layout measured.
#[derive(Clone, Copy)]
struct PackedGlyph {
    slot: Option<Slot>,
    left: i32,
    top: i32,
    advance: f32,
}

/// A shelf-packing texture atlas with a CPU mirror. Rasters are packed onto
/// horizontal shelves (rows share a height class); the mirror holds the texel
/// bytes so the GPU texture can be (re)filled at any time, and a dirty
/// rectangle accumulates what changed since the last upload.
pub struct Atlas {
    width: u32,
    height: u32,
    /// Bytes per texel: 1 (R8 coverage) or 4 (BGRA emoji).
    bpp: u32,
    pixels: Vec<u8>,
    /// Shelves as (y, height, x_used).
    shelves: Vec<(u32, u32, u32)>,
    next_y: u32,
    generation: u32,
    /// Union of slots written since the last [`Atlas::take_upload`].
    dirty: Option<(u32, u32, u32, u32)>,
}

impl Atlas {
    pub fn new(bpp: u32) -> Self {
        Self {
            width: ATLAS_START,
            height: ATLAS_START,
            bpp,
            pixels: vec![0; (ATLAS_START * ATLAS_START * bpp) as usize],
            shelves: Vec::new(),
            next_y: 0,
            generation: 0,
            dirty: None,
        }
    }

    pub fn info(&self) -> AtlasInfo {
        AtlasInfo {
            width: self.width,
            height: self.height,
            generation: self.generation,
        }
    }

    /// Reserve a `w` x `h` slot. `None` means the atlas is full even after
    /// growing to its cap (the caller skips the raster; a pathological case).
    /// Growing wipes all previous slots and bumps the generation, which the
    /// frame builder answers by restarting the build.
    fn reserve(&mut self, w: u32, h: u32) -> Option<Slot> {
        if w == 0 || h == 0 || w > ATLAS_MAX || h > ATLAS_MAX {
            return None;
        }
        loop {
            // A shelf fits if the raster is no taller than the shelf and the
            // remaining width holds it. Shelves are created at the raster's
            // height, so height classes cluster and waste stays bounded.
            if let Some(shelf) = self
                .shelves
                .iter_mut()
                .find(|(_, sh, used)| h <= *sh && used + w <= self.width)
            {
                let slot = Slot {
                    x: shelf.2,
                    y: shelf.0,
                    w,
                    h,
                };
                shelf.2 += w;
                return Some(slot);
            }
            if self.next_y + h <= self.height && w <= self.width {
                self.shelves.push((self.next_y, h, w));
                let slot = Slot {
                    x: 0,
                    y: self.next_y,
                    w,
                    h,
                };
                self.next_y += h;
                return Some(slot);
            }
            if !self.grow() {
                return None;
            }
        }
    }

    /// Double the atlas (square), wiping every slot. Returns false at the cap.
    fn grow(&mut self) -> bool {
        let next = (self.width * 2).min(ATLAS_MAX);
        if next == self.width {
            return false;
        }
        self.width = next;
        self.height = next;
        self.pixels = vec![0; (next * next * self.bpp) as usize];
        self.shelves.clear();
        self.next_y = 0;
        self.generation += 1;
        self.dirty = None;
        Ok::<(), ()>(()).is_ok()
    }

    /// Extend the dirty rectangle to cover `slot`, so the next upload carries it.
    fn mark_dirty(&mut self, slot: Slot) {
        let rect = (slot.x, slot.y, slot.x + slot.w, slot.y + slot.h);
        self.dirty = Some(match self.dirty {
            None => rect,
            Some((x0, y0, x1, y1)) => (
                x0.min(rect.0),
                y0.min(rect.1),
                x1.max(rect.2),
                y1.max(rect.3),
            ),
        });
    }

    /// Write a raster's rows into the mirror at `slot` and extend the dirty
    /// rectangle over it. `src` is `w * h * bpp` bytes, row-major, tight.
    fn write(&mut self, slot: Slot, src: &[u8]) {
        let row_bytes = (slot.w * self.bpp) as usize;
        for row in 0..slot.h {
            let src_start = row as usize * row_bytes;
            let dst_start = (((slot.y + row) * self.width + slot.x) * self.bpp) as usize;
            self.pixels[dst_start..dst_start + row_bytes]
                .copy_from_slice(&src[src_start..src_start + row_bytes]);
        }
        self.mark_dirty(slot);
    }

    /// Write a `w`-word-per-row ARGB raster (`w * h` words, row-major, tight) into
    /// the mirror at `slot`, one pixel's `to_ne_bytes` at a time. Lets the color
    /// atlas (`bpp == 4`) pack a glyph straight from its words without first
    /// collecting them into a byte vector.
    fn write_words(&mut self, slot: Slot, src: &[u32]) {
        let w = slot.w as usize;
        for row in 0..slot.h {
            let src_row = row as usize * w;
            let dst_row = (((slot.y + row) * self.width + slot.x) * self.bpp) as usize;
            for x in 0..w {
                let dst = dst_row + x * self.bpp as usize;
                self.pixels[dst..dst + 4].copy_from_slice(&src[src_row + x].to_ne_bytes());
            }
        }
        self.mark_dirty(slot);
    }

    /// Reserve a slot for a `w` x `h` raster and write `src` into it, returning
    /// the slot; `None` for a zero-area raster or a full atlas (the caller
    /// records the empty placement so the pen still advances). `src` is
    /// `w * h * bpp` bytes, row-major, tight.
    fn pack(&mut self, w: u32, h: u32, src: &[u8]) -> Option<Slot> {
        if w == 0 || h == 0 || src.len() < self.raster_len(w, h) {
            return None;
        }
        let slot = self.reserve(w, h)?;
        self.write(slot, src);
        Some(slot)
    }

    /// Like [`pack`](Self::pack) but from ARGB words (the color-glyph path), writing
    /// them into the 4-bpp mirror directly. `src` is `w * h` words, row-major, tight.
    fn pack_words(&mut self, w: u32, h: u32, src: &[u32]) -> Option<Slot> {
        if w == 0 || h == 0 || src.len() < (w as usize).saturating_mul(h as usize) {
            return None;
        }
        let slot = self.reserve(w, h)?;
        self.write_words(slot, src);
        Some(slot)
    }

    /// How many bytes a tight `w`-by-`h` raster occupies at this atlas's depth: what
    /// [`Self::write`] will read, and therefore what [`Self::pack`] insists on having.
    ///
    /// The check exists because the dimensions and the buffer come from different places
    /// and are not guaranteed to agree. `copy_coverage` returns an empty `Vec` when
    /// FreeType hands back a null `bitmap.buffer`, while `Glyph::width`/`rows` keep the
    /// nonzero dimensions the metrics reported — and `packed_scalar` then packs
    /// `(g.width, g.rows, &g.coverage)`, so the slice range in `write` traps. Release
    /// builds are `panic = "abort"`, so that is the whole process for a font or driver
    /// anomaly. `pack_words` already had this shape; `pack` did not.
    fn raster_len(&self, w: u32, h: u32) -> usize {
        (w as usize)
            .saturating_mul(h as usize)
            .saturating_mul(self.bpp as usize)
    }

    /// The bytes of the dirty region (row-contiguous), clearing the flag.
    /// A texture recreated after a grow needs no separate full upload: the
    /// grow wiped the mirror, so everything in it is inside the dirty region.
    pub fn take_upload(&mut self) -> Option<AtlasUpload> {
        let (x0, y0, x1, y1) = self.dirty.take()?;
        let (w, h) = (x1 - x0, y1 - y0);
        let row_bytes = (w * self.bpp) as usize;
        let mut bytes = Vec::with_capacity(row_bytes * h as usize);
        for row in y0..y1 {
            let start = ((row * self.width + x0) * self.bpp) as usize;
            bytes.extend_from_slice(&self.pixels[start..start + row_bytes]);
        }
        Some(AtlasUpload {
            x: x0,
            y: y0,
            w,
            h,
            bytes,
        })
    }
}

/// The persistent glyph caches the batcher packs into: slots by scalar and by
/// cluster, over the two atlases. Owned by the app next to the display list;
/// survives across frames so steady-state frames insert nothing.
/// The first printable ASCII scalar, and how many there are (`0x20..=0x7E`): the
/// span [`GlyphCache`] direct-maps.
const ASCII_FIRST: u32 = 0x20;
const ASCII_SLOTS: usize = 95;

/// The most faces that hold a direct-mapped ASCII row at once. Four styles across a
/// couple of sizes is the real ceiling (a `FaceKey`'s size changes only when the
/// *user* zooms, never at a child's request), so this is slack, not a budget; past
/// it, ASCII simply falls back to the map and stays correct.
const MAX_ASCII_FACES: usize = 16;

/// One face's placements for printable ASCII, indexed by `ch - 0x20`.
struct AsciiRow {
    face: FaceKey,
    slots: Box<[Option<PackedGlyph>; ASCII_SLOTS]>,
}

pub struct GlyphCache {
    pub glyphs: Atlas,
    pub emoji: Atlas,
    /// Printable ASCII, direct-mapped per face — an index, not a hash.
    ///
    /// This is the terminal's whole hot path: the scalar cache is probed once per
    /// glyph, thousands of times a frame, and hashing `(FaceKey, char)` with the
    /// standard library's SipHash cost half of `build_frame` on its own. ASCII is 95
    /// fixed codepoints and a run already knows its face, so the placement is an
    /// array index off a face found by a compare over a handful of rows.
    ///
    /// The split is deliberate rather than incidental: a cheaper *hash* would serve
    /// the same frame, but `scalar_slots` is keyed on characters a hostile child
    /// chooses, which is precisely the case the standard library's SipHash default
    /// exists to defend (`MAX_GLYPH_CACHE` already bounds that map's growth; nothing
    /// bounds its probe lengths). A direct-mapped row cannot be steered — the 95
    /// slots are fixed, an index cannot collide, and there is no probe to lengthen —
    /// so the hot path gives up the hash entirely while every exotic character keeps
    /// the defended map underneath.
    ascii_slots: Vec<AsciiRow>,
    scalar_slots: HashMap<(FaceKey, char), PackedGlyph>,
    /// Procedurally-drawn box/block rasters, held apart from the font ones.
    ///
    /// A separate map rather than a shared key, because the same character genuinely has
    /// two different rasters and both are correct in their place: on the grid `─` is drawn
    /// by [`boxdraw`] to fill the cell edge to edge and tile with its neighbours, while in
    /// a tab title it is prose and comes from the font at the font's own bearing and
    /// advance. Keying them together made whichever arrived first win for the rest of the
    /// process's life — a `DrawCmd::Text` carrying one box char was enough to leave every
    /// box glyph on the grid drawn at font metrics, tofu or hairline seams where the
    /// drawing should tile. It was a comment asserting "a given char is always a box glyph
    /// or never one"; now it is two maps, and no path can reach the other's entry.
    ///
    /// Bounded by construction: [`boxdraw::is_glyph`] is 160 codepoints, so this holds at
    /// most `160 * faces` and needs no cap of its own.
    box_slots: HashMap<(FaceKey, char), PackedGlyph>,
    /// Nested so lookups borrow the cluster as `&str` (no per-frame `String`).
    /// A `None` value records a cluster that is not color emoji, so a
    /// steady-state frame takes the per-character path without re-consulting
    /// the `Face`.
    cluster_slots: HashMap<FaceKey, HashMap<String, Option<PackedGlyph>>>,
}

impl GlyphCache {
    pub fn new() -> Self {
        Self {
            glyphs: Atlas::new(1),
            emoji: Atlas::new(4),
            ascii_slots: Vec::new(),
            scalar_slots: HashMap::new(),
            box_slots: HashMap::new(),
            cluster_slots: HashMap::new(),
        }
    }

    /// The direct-mapped index for a printable-ASCII scalar, or `None` for anything
    /// else (which belongs in [`Self::scalar_slots`]). The partition is total: a
    /// character is direct-mapped or hashed, never both, so the two can never
    /// disagree about a placement.
    fn ascii_index(ch: char) -> Option<usize> {
        let i = (ch as u32).wrapping_sub(ASCII_FIRST);
        (i < ASCII_SLOTS as u32).then_some(i as usize)
    }

    /// A printable-ASCII glyph's cached placement: find the face, index the row.
    fn ascii_get(&self, face: FaceKey, ch: char) -> Option<PackedGlyph> {
        let i = Self::ascii_index(ch)?;
        self.ascii_slots.iter().find(|r| r.face == face)?.slots[i]
    }

    /// Drop every slot record (after an atlas grew and wiped its content).
    fn reset_for(&mut self, glyphs_gen: u32, emoji_gen: u32) {
        if self.glyphs.generation != glyphs_gen {
            self.scalar_slots.clear();
            // Box rasters live in the same atlas, so a wipe takes them with it.
            self.box_slots.clear();
            // The direct-mapped rows hold atlas coordinates too; a wipe invalidates
            // them exactly as it does the map, and forgetting them here would leave
            // every ASCII glyph pointing into a dead slot.
            self.ascii_slots.clear();
        }
        if self.emoji.generation != emoji_gen {
            self.cluster_slots.clear();
        }
    }

    /// Record a scalar glyph's placement, dropping the whole scalar cache first if it
    /// has reached [`MAX_GLYPH_CACHE`], so distinct-glyph spam cannot grow it without
    /// bound. Clearing is safe: the already-emitted quads keep their valid atlas
    /// slots (the atlas is untouched), and the glyphs simply re-cache on next sight.
    fn cache_scalar(&mut self, key: (FaceKey, char), packed: PackedGlyph) {
        let (face, ch) = key;
        if let Some(i) = Self::ascii_index(ch) {
            if let Some(row) = self.ascii_slots.iter_mut().find(|r| r.face == face) {
                row.slots[i] = Some(packed);
                return;
            }
            if self.ascii_slots.len() < MAX_ASCII_FACES {
                let mut slots = Box::new([None; ASCII_SLOTS]);
                slots[i] = Some(packed);
                self.ascii_slots.push(AsciiRow { face, slots });
                return;
            }
            // Past the face ceiling ASCII keeps working, just through the map, which
            // is why `ascii_get` missing is never taken as "no such glyph".
        }
        if self.scalar_slots.len() >= MAX_GLYPH_CACHE {
            self.scalar_slots.clear();
        }
        self.scalar_slots.insert(key, packed);
    }

    /// Record a cluster's placement (or `None` for a non-emoji cluster: the negative
    /// cache that keeps a steady frame off the shaper), capping the per-face map the
    /// same way so a stream of distinct clusters cannot grow it without bound.
    fn cache_cluster(&mut self, face_key: FaceKey, cluster: &str, packed: Option<PackedGlyph>) {
        let by = self.cluster_slots.entry(face_key).or_default();
        if by.len() >= MAX_GLYPH_CACHE {
            by.clear();
        }
        by.insert(cluster.to_string(), packed);
    }
}

/// Build one frame's GPU data from the display list into `out` (whose vertex and
/// batch buffers are cleared and reused, so a steady frame allocates nothing).
/// Restarts internally if an atlas grows mid-build, so the vertices always
/// reference current atlas coordinates.
pub fn build_frame_into(
    fonts: &Fonts,
    list: &[DrawCmd],
    cache: &mut GlyphCache,
    gamma: TextGamma,
    out: &mut FrameData,
) {
    loop {
        let glyphs_gen = cache.glyphs.generation;
        let emoji_gen = cache.emoji.generation;
        out.vertices.clear();
        out.batches.clear();
        {
            let mut b = Batcher {
                fonts,
                cache: &mut *cache,
                gamma,
                vertices: &mut out.vertices,
                batches: &mut out.batches,
            };
            for cmd in list {
                b.command(cmd);
            }
        }
        // An atlas grew (and wiped) during the build: earlier quads reference
        // dead coordinates. Clear the stale slot records and rebuild; the
        // second pass fits by construction (or skips at the cap).
        if cache.glyphs.generation != glyphs_gen || cache.emoji.generation != emoji_gen {
            cache.reset_for(glyphs_gen, emoji_gen);
            continue;
        }
        out.glyph_atlas = cache.glyphs.info();
        out.emoji_atlas = cache.emoji.info();
        out.glyph_upload = cache.glyphs.take_upload();
        out.emoji_upload = cache.emoji.take_upload();
        return;
    }
}

/// Build a fresh frame, allocating its vertex and batch vectors. The one-shot path
/// for tests; the render loop calls [`build_frame_into`] with a reused [`FrameData`].
pub fn build_frame(
    fonts: &Fonts,
    list: &[DrawCmd],
    cache: &mut GlyphCache,
    gamma: TextGamma,
) -> FrameData {
    let mut out = FrameData::default();
    build_frame_into(fonts, list, cache, gamma, &mut out);
    out
}

struct Batcher<'a> {
    fonts: &'a Fonts,
    cache: &'a mut GlyphCache,
    gamma: TextGamma,
    vertices: &'a mut Vec<Vertex>,
    batches: &'a mut Vec<Batch>,
}

impl Batcher<'_> {
    fn command(&mut self, cmd: &DrawCmd) {
        match cmd {
            DrawCmd::Fill { rect, color } => {
                self.quad(*rect, MODE_SOLID, *color, [0.0; 2], [0.0; 4]);
            }
            DrawCmd::RoundRect {
                rect,
                color,
                radius,
                corners,
            } => self.round_rect(*rect, *color, *radius, *corners),
            DrawCmd::Text {
                x,
                baseline,
                face,
                color,
                bg,
                fade,
                text,
                ..
            } => self.text(
                *face,
                *x,
                *baseline,
                text,
                Paint::new(*color, *bg, *fade, self.gamma),
            ),
            DrawCmd::Cells {
                x,
                baseline,
                cell_w,
                face,
                color,
                bg,
                text,
                ..
            } => self.cells(
                *face,
                *x,
                *baseline,
                *cell_w,
                text,
                Paint::new(*color, *bg, None, self.gamma),
            ),
        }
    }

    /// Emit one axis-aligned quad (two triangles), with uv running from
    /// `[u0, v0]` at the top-left to `[u1, v1]` at the bottom-right (`uv` is
    /// `[u0, v0, u1, v1]`). The atlases and fills pass texel coordinates (uv
    /// advancing 1:1 with pixels).
    fn quad_uv(&mut self, rect: Rect, mode: u32, color: u32, uv: [f32; 4], extra: [f32; 4]) {
        if rect.w <= 0 || rect.h <= 0 {
            return;
        }
        let (x0, y0) = (rect.x as f32, rect.y as f32);
        let (x1, y1) = ((rect.x + rect.w) as f32, (rect.y + rect.h) as f32);
        let c = color_f32(color);
        let v = |x: f32, y: f32, u: f32, vv: f32| Vertex {
            pos: [x, y],
            uv: [u, vv],
            color: c,
            extra,
            mode,
        };
        let [u0, v0, u1, v1] = uv;
        self.vertices.extend_from_slice(&[
            v(x0, y0, u0, v0),
            v(x1, y0, u1, v0),
            v(x0, y1, u0, v1),
            v(x1, y0, u1, v0),
            v(x1, y1, u1, v1),
            v(x0, y1, u0, v1),
        ]);
        self.extend_batch(6);
    }

    /// Emit one axis-aligned quad with texel UVs: `uv0` is the coordinate at the
    /// quad's top-left and uv advances 1:1 with pixels. The atlas-bound path (no
    /// per-image texture), for fills, glyphs, emoji, and rounded corners.
    fn quad(&mut self, rect: Rect, mode: u32, color: u32, uv0: [f32; 2], extra: [f32; 4]) {
        let uv = [
            uv0[0],
            uv0[1],
            uv0[0] + rect.w as f32,
            uv0[1] + rect.h as f32,
        ];
        self.quad_uv(rect, mode, color, uv, extra);
    }

    /// Append `count` vertices to the frame's single batch (every quad binds the
    /// same atlases), opening it on the first quad.
    fn extend_batch(&mut self, count: u32) {
        let start = self.vertices.len() as u32 - count;
        if let Some(last) = self.batches.last_mut() {
            last.count += count;
            return;
        }
        self.batches.push(Batch { start, count });
    }

    fn round_rect(&mut self, rect: Rect, color: u32, radius: i32, corners: RoundedCorners) {
        // Clamp the radius to half the shorter side so opposite corners never
        // overlap; the shader's coverage function assumes this bound holds.
        let r = radius.clamp(0, (rect.w / 2).min(rect.h / 2).max(0));
        let mask = match corners {
            RoundedCorners::None => 0u32,
            RoundedCorners::Top => 1,
            RoundedCorners::Bottom => 2,
            RoundedCorners::Both => 3,
        };
        if r == 0 || mask == 0 {
            self.quad(rect, MODE_SOLID, color, [0.0; 2], [0.0; 4]);
            return;
        }
        // uv carries the position within the rect (pixels); extra carries the
        // geometry the shader's coverage function needs.
        let extra = [rect.w as f32, rect.h as f32, r as f32, mask as f32];
        self.quad(rect, MODE_ROUND, color, [0.0; 2], extra);
    }

    /// Emit a run's glyph quads: the ASCII fast path, then grapheme clusters
    /// routed to the emoji glyph or per-character drawing. The routing mirrors
    /// `shape::text_advance`, so the quads land where layout measured.
    fn text(&mut self, face_key: FaceKey, x: i32, baseline: i32, text: &str, paint: Paint) {
        let mut pen = x as f32;
        if text.is_ascii() {
            for ch in text.chars() {
                pen = self.scalar(face_key, ch, pen, baseline, paint);
            }
            return;
        }
        // Only the emoji cluster path needs the keyed face directly; the
        // per-scalar path resolves its own face (primary or fallback) per glyph.
        let face = self.fonts.face_for(face_key);
        // A wide cluster spans two cells, so its emoji is drawn at the em (its
        // natural, ~square size fills both cells and its advance carries the pen).
        let target = face_key.size();
        for (_, cluster) in grapheme::graphemes(text) {
            match self.packed_cluster(face, face_key, cluster, target) {
                Some(packed) => {
                    self.emit_glyph(&packed, pen, baseline, MODE_EMOJI, paint);
                    pen += packed.advance;
                }
                None => {
                    for ch in cluster.chars() {
                        pen = self.scalar(face_key, ch, pen, baseline, paint);
                    }
                }
            }
        }
    }

    /// A fixed-pitch run of monospace cells: cluster `i` in `text` draws with its
    /// pen anchored at `x + i*cell_w`, so the run rides the integer cell grid and
    /// never drifts the way an advance-accumulating [`Self::text`] run would (see
    /// [`DrawCmd::Cells`]). Each cluster occupies exactly one cell here; combining
    /// marks inside a cluster still stack on their base (they advance ~zero), and
    /// a color-emoji cluster keeps the anchored pen. The painter never feeds a
    /// wide cluster in, so one cell per cluster holds.
    fn cells(
        &mut self,
        face_key: FaceKey,
        x: i32,
        baseline: i32,
        cell_w: i32,
        text: &str,
        paint: Paint,
    ) {
        if is_plain_ascii(text) {
            self.cells_ascii(face_key, x, baseline, cell_w, text, paint)
        } else {
            self.cells_segmented(face_key, x, baseline, cell_w, text, paint)
        }
    }

    /// [`Self::cells`] for a printable-ASCII run: one byte is one cluster in one
    /// cell, so the pen is the column and the character is the cluster. Skipping the
    /// segmenter and the emoji probe is not an approximation — [`is_plain_ascii`]
    /// states the three facts that make this produce the very quads
    /// [`Self::cells_segmented`] would, and `ascii_runs_batch_exactly_like_the_segmented_path`
    /// holds the two outputs to being identical.
    fn cells_ascii(
        &mut self,
        face_key: FaceKey,
        x: i32,
        baseline: i32,
        cell_w: i32,
        text: &str,
        paint: Paint,
    ) {
        for (i, ch) in text.chars().enumerate() {
            // Fixed pitch: anchor the pen to the column and drop the advance the
            // scalar path returns, exactly as the segmented path does.
            let pen = (x + i as i32 * cell_w) as f32;
            self.scalar(face_key, ch, pen, baseline, paint);
        }
    }

    /// [`Self::cells`] for a run that needs real segmentation: anything carrying
    /// combining marks, box/block glyphs, or emoji.
    fn cells_segmented(
        &mut self,
        face_key: FaceKey,
        x: i32,
        baseline: i32,
        cell_w: i32,
        text: &str,
        paint: Paint,
    ) {
        // The cell box a procedurally-drawn box/block glyph fills: its height and
        // the baseline offset that seats it, both from the same metrics the grid
        // laid out on (read once, not per cluster).
        let m = self.fonts.metrics(face_key.size());
        let (cell_h, baseline_offset) = (m.line_height.max(1), m.baseline);
        let face = self.fonts.face_for(face_key);
        for (i, (_, cluster)) in grapheme::graphemes(text).enumerate() {
            let pen = (x + i as i32 * cell_w) as f32;
            // Box Drawing and Block Elements are rasterized here, not by the font,
            // so they cover the cell exactly and tile (see `render::boxdraw`).
            // They are single-width BMP scalars, so they only ever arrive as a
            // lone-char cluster on this fixed-pitch path, never through `text`.
            if let Some(ch) = box_glyph(cluster) {
                let packed = self.packed_box(face_key, ch, cell_w, cell_h, baseline_offset);
                self.emit_glyph(&packed, pen, baseline, MODE_GLYPH, paint);
                continue;
            }
            // A single-width color emoji (e.g. a bare ⚠) is scaled to one cell so
            // it does not balloon to the em (~two cells) and spill past its
            // column; the emoji is roughly square, so fitting the cell width caps
            // it at one cell.
            match self.packed_cluster(face, face_key, cluster, cell_w.max(1) as u32) {
                Some(packed) => self.emit_glyph(
                    &center_in_cell(packed, cell_w),
                    pen,
                    baseline,
                    MODE_EMOJI,
                    paint,
                ),
                None => {
                    // Draw the cluster's characters from the anchored pen; a base
                    // glyph advances and any combining marks land back over it.
                    let mut p = pen;
                    for ch in cluster.chars() {
                        p = self.scalar(face_key, ch, p, baseline, paint);
                    }
                }
            }
        }
    }

    /// One character's glyph quad (or nothing for empty rasters like spaces),
    /// returning the advanced pen. Its placement is cached beside the atlas
    /// slot, so a steady-state frame emits the quad without re-rasterising the
    /// glyph.
    fn scalar(
        &mut self,
        face_key: FaceKey,
        ch: char,
        pen: f32,
        baseline: i32,
        paint: Paint,
    ) -> f32 {
        let packed = self.packed_scalar(face_key, ch);
        self.emit_glyph(&packed, pen, baseline, MODE_GLYPH, paint);
        pen + packed.advance
    }

    /// The cached placement for a scalar glyph. On the first sight of this
    /// (face, character) the glyph is rasterized once and its coverage written
    /// straight into the atlas; later frames read the cached placement. The
    /// glyph is rasterized from [`Fonts::glyph_face`], so a scalar the keyed face
    /// lacks (a Nerd Font icon, a stray symbol) inks from a fallback face instead
    /// of the `.notdef` box; the `(key, ch)` cache key stays unique because that
    /// resolution is deterministic. A cache hit records against the keyed face
    /// and never re-resolves, so steady-state frames pay no fallback cost. The
    /// atlas cache is the only glyph raster cache, so its hit and miss are
    /// reported to the face for the frame stats.
    fn packed_scalar(&mut self, face_key: FaceKey, ch: char) -> PackedGlyph {
        let primary = self.fonts.face_for(face_key);
        if let Some(packed) = self.cache.ascii_get(face_key, ch) {
            primary.record_glyph_hit();
            return packed;
        }
        if let Some(&packed) = self.cache.scalar_slots.get(&(face_key, ch)) {
            primary.record_glyph_hit();
            return packed;
        }
        primary.record_glyph_miss();
        let g = self.fonts.glyph_face(face_key, ch).rasterize(ch);
        let atlas = &mut self.cache.glyphs;
        let packed = PackedGlyph {
            slot: atlas.pack(g.width as u32, g.rows as u32, &g.coverage),
            left: g.left,
            top: g.top,
            advance: g.advance,
        };
        self.cache.cache_scalar((face_key, ch), packed);
        packed
    }

    /// The cached placement for a procedurally-drawn box/block glyph, sized to the
    /// `w`-by-`h` cell. Its coverage comes from [`boxdraw::coverage`] instead of a
    /// FreeType raster, and it is anchored at `left = 0, top = baseline_offset` (the
    /// metrics' baseline, i.e. the distance from the cell's top edge down to the
    /// baseline) so the bitmap fills the cell box `[pen, pen + w) x [baseline -
    /// baseline_offset, + h)` exactly, tiling with its neighbours. Cached in
    /// [`GlyphCache::box_slots`], which is a *different* map from the font rasters: the
    /// same character has both a box raster and a font raster, so one key for both meant
    /// whichever path saw it first decided how it drew everywhere. The cell size is fixed
    /// by the size the key carries, so `(face_key, ch)` is a complete key within this map.
    fn packed_box(
        &mut self,
        face_key: FaceKey,
        ch: char,
        w: i32,
        h: i32,
        baseline_offset: i32,
    ) -> PackedGlyph {
        let primary = self.fonts.face_for(face_key);
        if let Some(&packed) = self.cache.box_slots.get(&(face_key, ch)) {
            primary.record_glyph_hit();
            return packed;
        }
        primary.record_glyph_miss();
        let cov = boxdraw::coverage(ch, w.max(0) as usize, h.max(0) as usize);
        let packed = PackedGlyph {
            slot: self
                .cache
                .glyphs
                .pack(w.max(0) as u32, h.max(0) as u32, &cov),
            left: 0,
            top: baseline_offset,
            advance: w as f32,
        };
        // Not `cache_scalar`: that routes ASCII into the direct-mapped rows and caps the
        // font map, neither of which applies here (box glyphs are U+2500..=U+259F, a
        // closed set of 160).
        self.cache.box_slots.insert((face_key, ch), packed);
        packed
    }

    /// The cached placement for an emoji cluster, or `None` when the cluster is
    /// not color emoji and the caller should draw its characters instead. The
    /// raster is packed into the color atlas on first sight and the decision
    /// cached, so a steady-state frame neither re-shapes nor touches the `Face`
    /// (beyond reporting the atlas hit for the frame stats). Non-emoji clusters
    /// are not tallied, matching the pre-atlas emoji-cache accounting.
    ///
    /// `target` is the pixel size to rasterize the color glyph at, chosen by the
    /// caller to fit its path: the em for a wide (two-cell) cluster on the
    /// [`Self::text`] path, one cell width for a single-width emoji on the
    /// [`Self::cells`] path so it does not overflow its column. A cluster only
    /// ever travels one path (its grid width is fixed), so the `(face_key,
    /// cluster)` cache key stays unique despite the per-path `target`.
    fn packed_cluster(
        &mut self,
        face: &Face,
        face_key: FaceKey,
        cluster: &str,
        target: u32,
    ) -> Option<PackedGlyph> {
        if let Some(&packed) = self
            .cache
            .cluster_slots
            .get(&face_key)
            .and_then(|by| by.get(cluster))
        {
            if packed.is_some() {
                face.record_cluster_hit();
            }
            return packed;
        }
        let atlas = &mut self.cache.emoji;
        let packed = face.with_cluster_glyph(cluster, target, |g| PackedGlyph {
            slot: pack_argb(atlas, g.width as u32, g.rows as u32, &g.argb),
            left: g.left,
            top: g.top,
            advance: g.advance,
        });
        if packed.is_some() {
            face.record_cluster_miss();
        }
        self.cache.cache_cluster(face_key, cluster, packed);
        packed
    }

    /// Emit a cached glyph's textured quad. An inkless or unplaced glyph (slot
    /// `None`) draws nothing; the caller advances the pen regardless.
    fn emit_glyph(
        &mut self,
        packed: &PackedGlyph,
        pen: f32,
        baseline: i32,
        mode: u32,
        paint: Paint,
    ) {
        let Some(slot) = packed.slot else {
            return;
        };
        let rect = Rect {
            x: pen.round() as i32 + packed.left,
            y: baseline - packed.top,
            w: slot.w as i32,
            h: slot.h as i32,
        };
        // A coverage glyph (mode 1) carries its per-run coverage exponent in extra.x;
        // emoji (mode 2) are pre-rendered colour and ignore it. Both carry the ink
        // ramp in extra.yz, so an emoji in a fading tail dissolves with the text
        // around it rather than standing solid where the letters have gone. An empty
        // span (`to <= from`, the 0.0 default) is the shader's "no fade".
        let exponent = if mode == MODE_GLYPH {
            paint.exponent
        } else {
            0.0
        };
        let (from, to) = match paint.fade {
            Some(fade) => (fade.from as f32, fade.to as f32),
            None => (0.0, 0.0),
        };
        self.quad(
            rect,
            mode,
            paint.color,
            [slot.x as f32, slot.y as f32],
            [exponent, from, to, 0.0],
        );
    }
}

/// Pack an emoji cluster's straight-`0xAARRGGBB` raster into the color atlas.
/// `None` for a zero-area or malformed glyph or a full atlas; the placement is
/// recorded as `None` so the pen advances without drawing.
fn pack_argb(atlas: &mut Atlas, w: u32, h: u32, argb: &[u32]) -> Option<Slot> {
    // ARGB words are B,G,R,A bytes in memory: B8G8R8A8 verbatim, written straight
    // into the mirror rather than first collected into a byte-conversion vector.
    // `pack_words` owns the dimension-versus-buffer check, so no caller can skip it.
    atlas.pack_words(w, h, argb)
}

/// Re-place a single-width color-emoji glyph so it sits on the text line rather
/// than on its natural pen bearing (which seats it near the cell's vertical
/// middle and reads *low* beside baseline-aligned text). The glyph is
/// bottom-aligned to the baseline and centered horizontally in the column, so it
/// rests on the line like the caps and digits around it and rises a little above
/// the x-height, exactly how a warning sign reads in other terminals.
///
/// The bottom dips `h / 12` below the baseline: a color glyph carries
/// a thin band of transparent padding beneath its ink, so dropping the raster
/// edge a hair below the line seats the *visible* bottom on it. An inkless glyph
/// (`slot` `None`) is returned unchanged. Only the fixed-pitch [`Batcher::cells`]
/// path uses this; a wide emoji keeps its advance-based place.
fn center_in_cell(packed: PackedGlyph, cell_w: i32) -> PackedGlyph {
    match packed.slot {
        Some(slot) => {
            let h = slot.h as i32;
            PackedGlyph {
                left: (cell_w - slot.w as i32) / 2,
                top: h - h / 12,
                ..packed
            }
        }
        None => packed,
    }
}

/// Whether every byte of a fixed-pitch run is printable ASCII (`0x20..=0x7E`),
/// which is what lets [`Batcher::cells`] skip segmentation for it.
///
/// The exclusions are each load-bearing, and all three must hold for one byte to
/// mean exactly one cluster in one cell:
///
/// - **Its own grapheme cluster.** No printable ASCII is `Extend`, `Prepend`, or
///   `SpacingMark`, so nothing merges with a neighbour. `CR`/`LF` are the one ASCII
///   pair a segmenter *does* join (UAX #29, GB3), and excluding the controls keeps
///   them out rather than betting they never reach a run.
/// - **Never a box/block glyph.** Those are `U+2500`-`U+259F`, well outside ASCII.
/// - **Never emoji.** `wants_emoji` routes a lone char on `Extended_Pictographic`,
///   which no ASCII scalar is, so the cluster probe can only ever answer `None`.
///
/// So the segmenter and the emoji probe cost a per-cell string hash to confirm what
/// this scan already knows.
///
/// The scan itself is [`bytes::all_printable`], eight bytes per word: a 120x80 frame
/// asks it about ~13 KB of run text, which a byte-at-a-time `all` walked at roughly
/// seven times the cost.
///
/// It is outlined deliberately, and that is worth as much as the wider scan. Let the
/// word loop inline and it lands inside [`build_frame_into`], already one of the
/// largest functions here, where it *lost* 5% of the frame — more than the scan saves.
/// Outlined it wins 3.5%, and outlining the old byte loop was itself worth 2%, so the
/// cost was never the scan alone but what a second loop nest does to this function's
/// layout.
#[inline(never)]
fn is_plain_ascii(text: &str) -> bool {
    bytes::all_printable(text.as_bytes())
}

/// The lone box/block scalar in `cluster`, or `None` if the cluster is not
/// exactly one such character. A box glyph never carries combining marks, so a
/// multi-char cluster is disqualified outright.
fn box_glyph(cluster: &str) -> Option<char> {
    let mut chars = cluster.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) if boxdraw::is_glyph(c) => Some(c),
        _ => None,
    }
}

/// A 0x00RRGGBB display-list color as straight-alpha RGBA floats.
fn color_f32(color: u32) -> [f32; 4] {
    [
        ((color >> 16) & 0xff) as f32 / 255.0,
        ((color >> 8) & 0xff) as f32 / 255.0,
        (color & 0xff) as f32 / 255.0,
        1.0,
    ]
}

/// The power the fragment shader raises a run's glyph coverage to, from the run's
/// foreground and background colours. Above 1 thins the anti-aliased edges, below 1
/// thickens them, and exactly 1 leaves the raster alone.
///
/// Linear-light compositing (the colour attachment is an sRGB view, so blending is
/// gamma-correct) misweights anti-aliased text in *opposite* directions depending on
/// which side is lighter, so the two cases get their own dial (see
/// [`crate::app::TEXT_GAMMA`]):
///
/// - **Light on dark**, the usual terminal text: rendered heavier than the
///   gamma-space stacks most terminals use, so thin it by
///   [`TextGamma::light_on_dark`] flat. The magnitude of the contrast does not enter:
///   the overweighting is there at any contrast.
/// - **Dark on light** (a reverse-video paste highlight, a light theme, the active
///   tab's dark label on its light block): washed out instead, so thicken it by
///   raising [`TextGamma::dark_on_light`] to a negative power, reaching its inverse
///   at maximum contrast. This one *does* scale with contrast, because the washout
///   does. Dark-on-light is never thinned, however faint the contrast: an earlier
///   `1 + 2d` ramp only crossed into thickening past half-contrast, so a dark label
///   on a mid-light block (this tab bar's turquoise) came out thin.
///
/// Computing this per run rather than per pixel keeps the policy in Rust, where it is
/// testable, and leaves the shader one `pow` instead of two.
///
/// Only the sign of `d` (which side is lighter) and a soft magnitude matter, so a
/// perceptual (gamma-space) luma is enough.
fn coverage_exponent(fg: u32, bg: u32, gamma: TextGamma) -> f32 {
    let d = luma(fg) - luma(bg);
    if d >= 0.0 {
        gamma.light_on_dark
    } else {
        // `d` is already in `[-1, 0)`; `max` is a defensive clamp, so the exponent
        // can never fall below the dial's inverse however the colours are chosen.
        gamma.dark_on_light.powf(d.max(-1.0))
    }
}

/// Rec. 709 relative luminance of a `0x00RRGGBB` colour in `[0, 1]`, taken in the
/// gamma-encoded byte space as a perceptual stand-in (enough to pick the contrast
/// direction and a soft strength; see [`coverage_exponent`]).
fn luma(color: u32) -> f32 {
    let r = ((color >> 16) & 0xff) as f32;
    let g = ((color >> 8) & 0xff) as f32;
    let b = (color & 0xff) as f32;
    (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both dials at 2.0 in tests, so an assertion reads as a plain power and a
    /// change to the shipping values in `app` cannot quietly move these numbers.
    const GAMMA: TextGamma = TextGamma {
        light_on_dark: 2.0,
        dark_on_light: 2.0,
    };

    #[test]
    fn atlas_packs_onto_shelves_and_tracks_dirty() {
        let mut a = Atlas::new(1);
        let s1 = a.reserve(10, 8).expect("fits");
        let s2 = a.reserve(6, 8).expect("fits on the same shelf");
        assert_eq!((s1.x, s1.y), (0, 0));
        assert_eq!((s2.x, s2.y), (10, 0), "same shelf, packed rightward");
        let s3 = a.reserve(10, 20).expect("taller: a new shelf");
        assert_eq!(s3.y, 8, "new shelf below the first");
        a.write(s1, &[7u8; 80]);
        let up = a.take_upload().expect("dirty after write");
        assert_eq!((up.x, up.y, up.w, up.h), (0, 0, 10, 8));
        assert!(up.bytes.iter().all(|&b| b == 7));
        assert!(a.take_upload().is_none(), "dirty cleared");
    }

    #[test]
    fn packing_a_raster_shorter_than_its_dimensions_refuses_instead_of_trapping() {
        // The dimensions and the buffer come from different places and are not guaranteed
        // to agree: `copy_coverage` returns an empty `Vec` when FreeType hands back a null
        // `bitmap.buffer`, while `Glyph::width`/`rows` keep the nonzero dimensions the
        // metrics reported. `packed_scalar` then packs `(width, rows, &coverage)` and the
        // slice range inside `write` traps — and release is `panic = "abort"`, so a font
        // or driver anomaly takes the whole process.
        let mut a = Atlas::new(1);
        assert_eq!(a.pack(4, 4, &[]), None, "the empty-coverage case exactly");
        assert_eq!(a.pack(4, 4, &[0u8; 15]), None, "one byte short is short");
        assert!(
            a.pack(4, 4, &[0u8; 16]).is_some(),
            "and exactly enough fits"
        );

        // The colour atlas counts four bytes to the pixel, so the same raster needs four
        // times the buffer. A check that forgot `bpp` would pass this and then trap.
        let mut a = Atlas::new(4);
        assert_eq!(a.pack(4, 4, &[0u8; 16]), None);
        assert!(a.pack(4, 4, &[0u8; 64]).is_some());

        // And the word path, which already refused, still does.
        let mut a = Atlas::new(4);
        assert_eq!(a.pack_words(2, 2, &[0u32; 3]), None);
        assert!(a.pack_words(2, 2, &[0u32; 4]).is_some());
    }

    #[test]
    fn pack_words_lays_down_the_same_bytes_as_a_conversion_would() {
        // The color-glyph path packs ARGB words straight into the 4-bpp mirror; the
        // result must match converting each word to its native bytes first, so
        // dropping that conversion vector changed nothing on screen.
        let mut a = Atlas::new(4);
        let argb: [u32; 4] = [0x1122_3344, 0x5566_7788, 0x99AA_BBCC, 0xDDEE_FF00];
        a.pack_words(2, 2, &argb).expect("fits");
        let up = a.take_upload().expect("dirty");
        assert_eq!((up.w, up.h), (2, 2));
        let expect: Vec<u8> = argb.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(up.bytes, expect);
    }

    #[test]
    fn glyph_caches_stay_bounded_under_distinct_input() {
        // Past the cap the maps drop and re-fill rather than growing forever, so a
        // hostile stream of distinct glyphs or clusters cannot exhaust memory.
        let mut cache = GlyphCache::new();
        let face = FaceKey::Code { size: 16 };
        let none = PackedGlyph {
            slot: None,
            left: 0,
            top: 0,
            advance: 0.0,
        };
        for i in 0..MAX_GLYPH_CACHE + 5 {
            let ch = char::from_u32(0x1_0000 + i as u32).expect("valid scalar");
            cache.cache_scalar((face, ch), none);
            cache.cache_cluster(face, &format!("c{i}"), None);
        }
        assert!(cache.scalar_slots.len() <= MAX_GLYPH_CACHE);
        let clusters: usize = cache.cluster_slots.values().map(|m| m.len()).sum();
        assert!(clusters <= MAX_GLYPH_CACHE);
    }

    #[test]
    fn atlas_grows_by_wiping_and_bumping_generation() {
        let mut a = Atlas::new(1);
        let g0 = a.generation;
        // A raster wider than the atlas forces growth.
        let s = a.reserve(ATLAS_START + 1, 4).expect("grows to fit");
        assert!(a.generation > g0);
        assert_eq!(a.width, ATLAS_START * 2);
        assert_eq!((s.x, s.y), (0, 0), "the grown atlas starts empty");
        // Beyond the cap is a clean None.
        assert!(a.reserve(ATLAS_MAX + 1, 1).is_none());
    }

    /// A placed glyph to cache; the coordinates are irrelevant to these tests, only
    /// whether the record survives or is dropped.
    fn placed() -> PackedGlyph {
        PackedGlyph {
            slot: Some(Slot {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            }),
            left: 0,
            top: 0,
            advance: 4.0,
        }
    }

    /// The direct-mapped rows hold atlas coordinates exactly as the map does, so an
    /// atlas wipe has to drop both. The map has always been dropped on a generation
    /// bump; if the rows are not, every ASCII glyph silently renders from a dead slot
    /// — the one corruption splitting the cache in two could introduce, and one no
    /// rendering test would notice until it was on screen.
    #[test]
    fn an_atlas_wipe_invalidates_the_direct_mapped_ascii_rows() {
        let mut cache = GlyphCache::new();
        let face = FaceKey::Code { size: 16 };
        cache.cache_scalar((face, 'A'), placed());
        cache.cache_scalar((face, '\u{2500}'), placed());
        assert!(cache.ascii_get(face, 'A').is_some());
        assert!(cache.scalar_slots.contains_key(&(face, '\u{2500}')));

        // A raster wider than the atlas forces it to grow, wiping every slot.
        let g0 = cache.glyphs.generation;
        let emoji_gen = cache.emoji.generation;
        cache
            .glyphs
            .reserve(ATLAS_START + 1, 4)
            .expect("grows to fit");
        assert!(
            cache.glyphs.generation > g0,
            "the wipe bumps the generation"
        );
        cache.reset_for(g0, emoji_gen);

        assert!(
            cache.ascii_get(face, 'A').is_none(),
            "a direct-mapped ASCII row survived an atlas wipe, so it now points at a dead slot"
        );
        assert!(
            cache.scalar_slots.is_empty(),
            "the map is dropped as before"
        );
    }

    /// The two stores must partition the characters between them: direct-mapped or
    /// hashed, never both, or the two could disagree about one glyph's placement.
    #[test]
    fn ascii_is_direct_mapped_and_every_other_scalar_is_hashed() {
        let mut cache = GlyphCache::new();
        let face = FaceKey::Code { size: 16 };
        // The span and both its edges.
        for ch in [' ', 'A', '~'] {
            cache.cache_scalar((face, ch), placed());
            assert!(
                cache.ascii_get(face, ch).is_some(),
                "{ch:?} should be mapped"
            );
            assert!(
                !cache.scalar_slots.contains_key(&(face, ch)),
                "{ch:?} should not also be hashed"
            );
        }
        // Just outside it at either end, plus the exotica an attacker actually picks.
        for ch in ['\u{1f}', '\u{7f}', 'é', '\u{2500}', '\u{1F600}'] {
            cache.cache_scalar((face, ch), placed());
            assert!(
                cache.ascii_get(face, ch).is_none(),
                "{ch:?} is not printable ASCII and must not be direct-mapped"
            );
            assert!(
                cache.scalar_slots.contains_key(&(face, ch)),
                "{ch:?} should be hashed"
            );
        }
    }

    /// Past the face ceiling the rows stop being handed out, and ASCII has to keep
    /// working through the map — `ascii_get` missing a glyph means "not here", never
    /// "no such glyph".
    #[test]
    fn ascii_past_the_face_ceiling_falls_back_to_the_map() {
        let mut cache = GlyphCache::new();
        for i in 0..MAX_ASCII_FACES {
            cache.cache_scalar((FaceKey::Code { size: i as u32 }, 'A'), placed());
        }
        assert_eq!(cache.ascii_slots.len(), MAX_ASCII_FACES);
        let extra = FaceKey::Code { size: 9999 };
        cache.cache_scalar((extra, 'A'), placed());
        assert!(
            cache.ascii_get(extra, 'A').is_none(),
            "no row is left for this face"
        );
        assert!(
            cache.scalar_slots.contains_key(&(extra, 'A')),
            "so the glyph must still be reachable through the map"
        );
    }

    /// The fixed-pitch fast path skips the grapheme segmenter and the emoji probe
    /// for a printable-ASCII run, which is only sound if it is not a *shortcut* but
    /// the same answer arrived at cheaply. So hold it to exactly that: the quads
    /// `cells_ascii` emits must be identical, field for field, to the ones
    /// `cells_segmented` (which does the full work) emits for the same run. Both
    /// start from a fresh cache, so the atlas slots they hand out line up too.
    #[test]
    fn ascii_runs_batch_exactly_like_the_segmented_path() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let face_key = FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        let paint = Paint::new(0x00ff_ffff, 0x0000_0000, None, GAMMA);
        let quads = |text: &str, segmented: bool| -> Vec<Vertex> {
            let mut cache = GlyphCache::new();
            let (mut vertices, mut batches) = (Vec::new(), Vec::new());
            let mut b = Batcher {
                fonts: &fonts,
                cache: &mut cache,
                gamma: GAMMA,
                vertices: &mut vertices,
                batches: &mut batches,
            };
            // A non-zero origin and a cell pitch unequal to any glyph's own advance,
            // so a path that accumulated advances instead of anchoring to the column
            // would drift visibly rather than coincidentally agreeing.
            if segmented {
                b.cells_segmented(face_key, 7, 16, 9, text, paint);
            } else {
                b.cells_ascii(face_key, 7, 16, 9, text, paint);
            }
            vertices
        };

        let alphabet: String = (0x20u8..=0x7e).map(|b| b as char).collect();
        for text in [
            alphabet.as_str(),
            "hello world",
            "   ", // blanks are inkless: no quads from either path
            "$ ls -la | grep '*' # 07",
        ] {
            assert!(is_plain_ascii(text), "{text:?} should take the fast path");
            assert_eq!(
                quads(text, false),
                quads(text, true),
                "the ASCII fast path diverged from the segmented path on {text:?}"
            );
        }
    }

    /// The three facts `is_plain_ascii` rests on, asserted over every byte it
    /// admits rather than argued in a comment. If a future change makes an ASCII
    /// scalar a box glyph or gives it an emoji presentation, the fast path becomes
    /// wrong and this goes red before the pixels do.
    #[test]
    fn every_printable_ascii_byte_is_one_plain_cluster() {
        for b in 0x20u8..=0x7e {
            let s = (b as char).to_string();
            assert!(is_plain_ascii(&s), "{s:?} should be plain ASCII");
            assert_eq!(
                grapheme::graphemes(&s).count(),
                1,
                "{s:?} must be exactly one cluster, or the pitch misaligns"
            );
            assert!(box_glyph(&s).is_none(), "{s:?} must not be a box glyph");
            assert!(
                !crate::platform::emoji::wants_emoji(&s, |_| false),
                "{s:?} must never route to the emoji font"
            );
        }
        // The controls are excluded, and CR LF is exactly why: it is the one ASCII
        // pair a segmenter joins, so a byte-per-cell walk would misalign the run.
        // The guard keeps it off the fast path rather than betting it never occurs.
        assert!(!is_plain_ascii("\r\n"));
        assert_eq!(
            grapheme::graphemes("\r\n").count(),
            1,
            "CR LF is one cluster"
        );
    }

    #[test]
    fn fill_becomes_one_solid_quad_batch() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::new();
        let list = vec![DrawCmd::Fill {
            rect: Rect {
                x: 1,
                y: 2,
                w: 10,
                h: 20,
            },
            color: 0x00ff_0000,
        }];
        let f = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert_eq!(f.vertices.len(), 6);
        assert_eq!(f.batches.len(), 1);
        assert_eq!(f.batches[0], Batch { start: 0, count: 6 });
        assert!(f.vertices.iter().all(|v| v.mode == MODE_SOLID));
        assert_eq!(f.vertices[0].pos, [1.0, 2.0]);
        assert_eq!(f.vertices[4].pos, [11.0, 22.0]);
        assert_eq!(f.vertices[0].color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn text_reuses_atlas_slots_across_frames() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::new();
        let list = vec![DrawCmd::Text {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 100,
                h: 20,
            },
            x: 0,
            baseline: 16,
            face: FaceKey::Prose {
                size: 16,
                style: crate::platform::freetype::FontStyle::Regular,
            },
            color: 0x00ff_ffff,
            bg: 0,
            fade: None,
            text: "abcabc".to_string(),
        }];
        let f1 = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert!(f1.glyph_upload.is_some(), "first frame uploads new glyphs");
        assert!(
            !f1.vertices.is_empty(),
            "glyph quads were emitted ({} vertices)",
            f1.vertices.len()
        );
        let f2 = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert!(f2.glyph_upload.is_none(), "steady state uploads nothing");
        assert_eq!(f1.vertices, f2.vertices, "same list, same geometry");
    }

    #[test]
    fn glyph_quads_share_one_batch_with_fills() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::new();
        let list = vec![
            DrawCmd::Fill {
                rect: Rect {
                    x: 0,
                    y: 0,
                    w: 5,
                    h: 5,
                },
                color: 0,
            },
            DrawCmd::Text {
                bounds: Rect {
                    x: 0,
                    y: 0,
                    w: 100,
                    h: 20,
                },
                x: 0,
                baseline: 16,
                face: FaceKey::Prose {
                    size: 16,
                    style: crate::platform::freetype::FontStyle::Regular,
                },
                color: 0x00ff_ffff,
                bg: 0,
                fade: None,
                text: "hi".to_string(),
            },
        ];
        let f = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert_eq!(
            f.batches.len(),
            1,
            "no image, so one batch: {:?}",
            f.batches
        );
        assert_eq!(f.batches[0].count, f.vertices.len() as u32);
    }

    #[test]
    fn round_rect_emits_sdf_quad_with_geometry_in_extra() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::new();
        let rect = Rect {
            x: 4,
            y: 6,
            w: 20,
            h: 10,
        };
        let list = vec![DrawCmd::RoundRect {
            rect,
            color: 0x0011_2233,
            radius: 4,
            corners: RoundedCorners::Top,
        }];
        let f = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert_eq!(f.vertices.len(), 6);
        let v = &f.vertices[0];
        assert_eq!(v.mode, MODE_ROUND);
        assert_eq!(v.extra, [20.0, 10.0, 4.0, 1.0], "w, h, r, corner mask");
        assert_eq!(v.uv, [0.0, 0.0], "uv is rect-local pixels");
        assert_eq!(f.vertices[4].uv, [20.0, 10.0]);
    }

    #[test]
    fn zero_radius_round_rect_degrades_to_solid() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::new();
        let list = vec![DrawCmd::RoundRect {
            rect: Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
            color: 0,
            radius: 0,
            corners: RoundedCorners::Both,
        }];
        let f = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert!(f.vertices.iter().all(|v| v.mode == MODE_SOLID));
    }

    #[test]
    fn cells_place_glyphs_at_a_fixed_pitch_not_the_advance() {
        // The whole point of DrawCmd::Cells: each cell's glyph is anchored to
        // `x + i*cell_w`, so the run tracks the integer cell grid instead of the
        // font's (fractional, and here deliberately different) advance. Repeating
        // one glyph keeps the left bearing constant, so equal spacing between the
        // quads' top-left corners is exactly the cell pitch.
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::new();
        let cell_w = 20; // wider than a 16px glyph's natural advance
        let list = vec![DrawCmd::Cells {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 200,
                h: 20,
            },
            x: 5,
            baseline: 16,
            cell_w,
            face: FaceKey::Code { size: 16 },
            color: 0x00ff_ffff,
            bg: 0,
            text: "MMMM".to_string(),
        }];
        let f = build_frame(&fonts, &list, &mut cache, GAMMA);
        // 'M' is an inked glyph, so each of the four cells emits one 6-vertex quad.
        assert_eq!(f.vertices.len(), 24, "four inked cells, one quad each");
        let top_left = |cell: usize| f.vertices[cell * 6].pos;
        for cell in 0..4 {
            assert_eq!(
                top_left(cell)[0] - top_left(0)[0],
                (cell as i32 * cell_w) as f32,
                "cell {cell} sits exactly {cell_w}px from the first"
            );
            // Identical glyphs share a top bearing, so the same baseline puts
            // every quad's top edge at the same y.
            assert_eq!(top_left(cell)[1], top_left(0)[1], "shared baseline");
        }
    }

    #[test]
    fn cells_skip_blank_columns_but_keep_the_pitch() {
        // A space rasterizes to nothing, so it emits no quad; the cell after it
        // must still land two pitches along, because placement is by column index,
        // not by walking a pen through the run.
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::new();
        let cell_w = 12;
        let list = vec![DrawCmd::Cells {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 100,
                h: 20,
            },
            x: 0,
            baseline: 16,
            cell_w,
            face: FaceKey::Code { size: 16 },
            color: 0x00ff_ffff,
            bg: 0,
            text: "M M".to_string(),
        }];
        let f = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert_eq!(f.vertices.len(), 12, "two inked cells (the space is blank)");
        // The two 'M' quads are two cell pitches apart (columns 0 and 2).
        assert_eq!(
            f.vertices[6].pos[0] - f.vertices[0].pos[0],
            (2 * cell_w) as f32
        );
    }

    #[test]
    fn box_glyph_fills_its_cell_from_boxdraw_not_the_font() {
        // `▛` (U+259B) is a quadrant block many monospace fonts lack, so through
        // the font it would be `.notdef` tofu. It must instead be rasterized by
        // `render::boxdraw` into a quad that covers the whole cell box exactly, so
        // it tiles: top-left at (pen, baseline - m.baseline), size (cell_w, cell_h).
        // Seating it on the ascent instead would float the quad above its cell by
        // the font's line gap, leaving a seam between stacked block rows.
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::new();
        let m = fonts.metrics(16);
        let (baseline_offset, cell_h) = (m.baseline, m.line_height.max(1));
        let cell_w = 11;
        let (x, baseline) = (7, 16);
        let list = vec![DrawCmd::Cells {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 100,
                h: 40,
            },
            x,
            baseline,
            cell_w,
            face: FaceKey::Code { size: 16 },
            color: 0x00ff_ffff,
            bg: 0,
            text: "▛".to_string(),
        }];
        let f = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert_eq!(
            f.vertices.len(),
            6,
            "one full-cell quad for the block glyph"
        );
        assert!(
            f.vertices.iter().all(|v| v.mode == MODE_GLYPH),
            "drawn as a coverage glyph, not tofu or a solid fill"
        );
        // The quad is the cell box: top-left corner and bottom-right corner.
        assert_eq!(
            f.vertices[0].pos,
            [x as f32, (baseline - baseline_offset) as f32],
            "top-left seats the glyph at the cell origin"
        );
        assert_eq!(
            f.vertices[4].pos,
            [
                (x + cell_w) as f32,
                (baseline - baseline_offset + cell_h) as f32
            ],
            "bottom-right fills the whole cell, so the glyph tiles"
        );
    }

    #[test]
    fn a_box_char_drawn_as_prose_does_not_poison_the_grid_raster() {
        // `─` has two right answers, and they are not interchangeable: on the grid it is
        // drawn by `render::boxdraw` to fill the cell edge to edge and tile with its
        // neighbours, and as prose it comes from the font at the font's own bearing and
        // advance. Cached under one key, whichever arrived first decided how `─` drew
        // everywhere for the rest of the process's life. Reaching the prose path with one
        // was never hypothetical: the cursor's inverted stamp emits `DrawCmd::Text`, so a
        // single box char under the cursor left every box glyph on the grid at font
        // metrics — tofu, or hairline seams where the drawing should tile.
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::new();
        let face = FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        let m = fonts.metrics(16);
        let (baseline_offset, cell_h) = (m.baseline, m.line_height.max(1));
        let cell_w = 11;
        let (x, baseline) = (0, 16);
        let bounds = Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 40,
        };

        // Prose first, so the font's raster is the one already in the cache.
        build_frame(
            &fonts,
            &[DrawCmd::Text {
                bounds,
                x,
                baseline,
                face,
                color: 0x00ff_ffff,
                bg: 0,
                fade: None,
                text: "\u{2500}".to_string(),
            }],
            &mut cache,
            GAMMA,
        );

        // Then the same character on the grid, under the same face key.
        let f = build_frame(
            &fonts,
            &[DrawCmd::Cells {
                bounds,
                x,
                baseline,
                cell_w,
                face,
                color: 0x00ff_ffff,
                bg: 0,
                text: "\u{2500}".to_string(),
            }],
            &mut cache,
            GAMMA,
        );

        assert_eq!(f.vertices.len(), 6, "one quad for the cell");
        assert_eq!(
            f.vertices[0].pos,
            [x as f32, (baseline - baseline_offset) as f32],
            "the grid still seats the box raster at the cell origin"
        );
        assert_eq!(
            f.vertices[4].pos,
            [
                (x + cell_w) as f32,
                (baseline - baseline_offset + cell_h) as f32
            ],
            "and it still fills the whole cell, so it still tiles"
        );

        // Both rasters exist, in their own maps. One key holding one of them is the bug.
        assert!(
            cache.scalar_slots.contains_key(&(face, '\u{2500}')),
            "the prose raster was cached"
        );
        assert!(
            cache.box_slots.contains_key(&(face, '\u{2500}')),
            "and the box raster separately"
        );
    }

    #[test]
    fn emoji_lands_in_the_color_atlas() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::new();
        let list = vec![DrawCmd::Text {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 100,
                h: 20,
            },
            x: 0,
            baseline: 16,
            face: FaceKey::Prose {
                size: 16,
                style: crate::platform::freetype::FontStyle::Regular,
            },
            color: 0x00ff_ffff,
            bg: 0,
            fade: None,
            text: "😀".to_string(),
        }];
        let f = build_frame(&fonts, &list, &mut cache, GAMMA);
        // With an emoji font installed the cluster takes MODE_EMOJI; without
        // one it renders as per-character glyphs. Either way it must not
        // crash, and with color output there must be an emoji upload.
        if f.vertices.iter().any(|v| v.mode == MODE_EMOJI) {
            assert!(f.emoji_upload.is_some(), "the color raster was uploaded");
        }
    }

    #[test]
    fn single_width_emoji_is_scaled_to_its_cell_not_the_em() {
        // A single-width color emoji (a bare ⚠) on the fixed-pitch `cells` path
        // is sized to one cell, so it does not balloon to the em (~two cells) and
        // spill past its column the way it does on the advance-based `text` path.
        let fonts = Fonts::new(&[16]).expect("default font");
        let key = FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        let warn = "\u{26A0}".to_string();
        let cell_w = 10;

        let mut cache = GlyphCache::new();
        let cells = build_frame(
            &fonts,
            &[DrawCmd::Cells {
                bounds: Rect {
                    x: 0,
                    y: 0,
                    w: 100,
                    h: 20,
                },
                x: 0,
                baseline: 16,
                cell_w,
                face: key,
                color: 0x00ff_ffff,
                bg: 0,
                text: warn.clone(),
            }],
            &mut cache,
            GAMMA,
        );
        let mut cache2 = GlyphCache::new();
        let text = build_frame(
            &fonts,
            &[DrawCmd::Text {
                bounds: Rect {
                    x: 0,
                    y: 0,
                    w: 100,
                    h: 20,
                },
                x: 0,
                baseline: 16,
                face: key,
                color: 0x00ff_ffff,
                bg: 0,
                fade: None,
                text: warn,
            }],
            &mut cache2,
            GAMMA,
        );

        // The width of the (single) emoji quad, if the cluster took the color
        // path at all; `None` when no emoji font is installed or it cannot form
        // the glyph, in which case there is nothing to compare.
        let emoji_width = |f: &FrameData| {
            f.vertices
                .iter()
                .any(|v| v.mode == MODE_EMOJI)
                .then(|| f.vertices[4].pos[0] - f.vertices[0].pos[0])
        };
        if let (Some(cell_px), Some(em_px)) = (emoji_width(&cells), emoji_width(&text)) {
            assert!(
                cell_px < em_px,
                "the fixed-pitch emoji ({cell_px}) is smaller than the em-sized one ({em_px})"
            );
            assert!(
                cell_px <= 1.5 * cell_w as f32,
                "and it fits about one cell (cell_w = {cell_w}, got {cell_px})"
            );
        }
    }

    #[test]
    fn single_width_emoji_sits_on_the_baseline_like_text() {
        // A single-width color emoji is bottom-aligned to the baseline (dipping a
        // little below it to seat the visible ink on the line) and rises above
        // the x-height, so it lines up with the text beside it rather than
        // floating above the line on its natural bearing.
        let fonts = Fonts::new(&[16]).expect("default font");
        let key = FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        let (baseline, cell_w) = (16, 10);
        let mut cache = GlyphCache::new();
        let f = build_frame(
            &fonts,
            &[DrawCmd::Cells {
                bounds: Rect {
                    x: 0,
                    y: 0,
                    w: 100,
                    h: 30,
                },
                x: 0,
                baseline,
                cell_w,
                face: key,
                color: 0x00ff_ffff,
                bg: 0,
                text: "\u{26A0}".to_string(),
            }],
            &mut cache,
            GAMMA,
        );
        if f.vertices.iter().any(|v| v.mode == MODE_EMOJI) {
            let (top, bottom) = (f.vertices[0].pos[1], f.vertices[4].pos[1]);
            let h = bottom - top;
            // The raster bottom rests on the baseline, dipping at most ~a quarter
            // of its height below it (never floating above the line).
            assert!(
                bottom >= baseline as f32 && bottom <= baseline as f32 + h / 4.0,
                "emoji bottom {bottom} rests on the baseline {baseline} (h = {h})"
            );
            // And it rises above the baseline into the text height.
            assert!(
                top < baseline as f32,
                "emoji {top} rises above the baseline"
            );
        }
    }

    /// Build one `Text` run and return its glyph quads' `extra` vectors.
    fn run_extras(
        fonts: &Fonts,
        text: &str,
        color: u32,
        bg: u32,
        fade: Option<Fade>,
    ) -> Vec<[f32; 4]> {
        let list = vec![DrawCmd::Text {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 200,
                h: 20,
            },
            x: 0,
            baseline: 16,
            face: FaceKey::Prose {
                size: 16,
                style: crate::platform::freetype::FontStyle::Regular,
            },
            color,
            bg,
            fade,
            text: text.to_string(),
        }];
        let mut cache = GlyphCache::new();
        build_frame(fonts, &list, &mut cache, GAMMA)
            .vertices
            .iter()
            .filter(|v| v.mode == MODE_GLYPH)
            .map(|v| v.extra)
            .collect()
    }

    #[test]
    fn a_faded_run_carries_its_ink_ramp_without_disturbing_its_contrast() {
        let fonts = Fonts::new(&[16]).expect("default font");
        // The tab bar's dark-on-light case: the label's own colours, plus a ramp.
        let (fg, bg) = (0x0012_3028, 0x008a_beb7);
        let fade = Fade { from: 40, to: 72 };
        let faded = run_extras(&fonts, "hello", fg, bg, Some(fade));
        let solid = run_extras(&fonts, "hello", fg, bg, None);
        assert!(!faded.is_empty(), "the run emitted glyphs");
        assert_eq!(faded.len(), solid.len());

        // Every glyph in the run carries the same span, so the ramp is continuous
        // across glyph boundaries rather than quantised to them.
        assert!(
            faded
                .iter()
                .all(|e| e[1] == fade.from as f32 && e[2] == fade.to as f32),
            "the fade span reaches every glyph quad"
        );
        // An unfaded run leaves an empty span, which the shader reads as "no fade".
        assert!(
            solid.iter().all(|e| e[2] <= e[1]),
            "no fade means an empty span, not a hard cut at zero"
        );
        // The load-bearing invariant: fading changes the ink, never the contrast the
        // coverage exponent is derived from. Were the fade a colour mix toward the
        // background (as it once was), this exponent would drift toward 1.0 exactly as
        // the tail faded, and the tail would carry ~2x the anti-aliased ink of the
        // head: a smudge on a light tab rather than a dissolve.
        let want = coverage_exponent(fg, bg, GAMMA);
        assert!(
            faded.iter().chain(&solid).all(|e| e[0] == want),
            "the run's coverage exponent is untouched by the fade"
        );
    }

    #[test]
    fn coverage_exponent_thins_light_on_dark_and_thickens_dark_on_light() {
        let white = 0x00ff_ffff;
        let black = 0x0000_0000;
        // Light on dark is thinned by the light_on_dark dial, flat: linear-light
        // compositing overweights it at *any* contrast, so the correction does not
        // soften as the contrast falls.
        assert_eq!(coverage_exponent(white, black, GAMMA), 2.0);
        assert_eq!(coverage_exponent(white, 0x0080_8080, GAMMA), 2.0);
        // Dark on white (a reverse-video paste highlight) is thickened instead: the
        // exponent drops below 1, bottoming at the dial's inverse at full contrast.
        assert_eq!(coverage_exponent(black, white, GAMMA), 0.5);
        assert!(
            coverage_exponent(black, white, GAMMA) < coverage_exponent(0x0060_6060, white, GAMMA)
        );
        // Dark text on a mid-light block (the tab bar's #123028 on #8abeb7) is
        // thickened, not left near-untouched: dark-on-light thickens in proportion
        // to contrast, so a moderately lighter background still gets real weight.
        let tab = coverage_exponent(0x0012_3028, 0x008a_beb7, GAMMA);
        assert!(tab < 0.76, "moderate dark-on-light thickens, got {tab}");
        // Equal luminance takes the thinning branch rather than a discontinuity.
        assert_eq!(coverage_exponent(0x0044_4444, 0x0044_4444, GAMMA), 2.0);
        // A run is thinned at most to the dial and thickened at most to its inverse,
        // so the shader's pow never sees an exponent that could erase or blot text.
        for &(fg, bg) in &[(white, black), (black, white), (0x0012_3456, 0x00fe_dcba)] {
            let e = coverage_exponent(fg, bg, GAMMA);
            assert!((0.5..=2.0).contains(&e), "exponent {e} in range");
        }
    }

    #[test]
    fn each_gamma_dial_moves_only_its_own_contrast_direction() {
        // The bug the split exists to prevent. With one dial governing both
        // corrections, softening it slid *both* toward the no-correction exponent of
        // 1.0 at once, which means opposite visual directions: light-on-dark got
        // bolder while dark-on-light got thinner. So lowering "the gamma" to embolden
        // the grid silently thinned the active tab's dark label. Each dial must now
        // move its own direction and leave the other alone.
        let grid = (0x00d8_dee9, 0x0028_2c34); // light on dark
        let label = (0x0012_3028, 0x008a_beb7); // the active tab's dark label
        let base = GAMMA;

        // Softening the light-on-dark dial emboldens light-on-dark text (a lower
        // exponent keeps more coverage) and leaves dark-on-light text alone.
        let softer_light = TextGamma {
            light_on_dark: 1.5,
            ..base
        };
        assert!(
            coverage_exponent(grid.0, grid.1, softer_light)
                < coverage_exponent(grid.0, grid.1, base),
            "the light-on-dark dial emboldens light-on-dark text"
        );
        assert_eq!(
            coverage_exponent(label.0, label.1, softer_light),
            coverage_exponent(label.0, label.1, base),
            "and must not touch the dark-on-light label"
        );

        // Symmetrically, the dark-on-light dial moves only dark-on-light text.
        let softer_dark = TextGamma {
            dark_on_light: 1.5,
            ..base
        };
        assert!(
            coverage_exponent(label.0, label.1, softer_dark)
                > coverage_exponent(label.0, label.1, base),
            "softening the dark-on-light dial thins the dark-on-light label"
        );
        assert_eq!(
            coverage_exponent(grid.0, grid.1, softer_dark),
            coverage_exponent(grid.0, grid.1, base),
            "and must not touch light-on-dark grid text"
        );
    }
}
