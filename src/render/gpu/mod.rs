//! The GPU backend's CPU half: turning a display list into GPU-consumable
//! data. This module is deliberately free of Vulkan: it packs glyph rasters
//! into atlases and emits vertices and draw batches as plain values, so
//! the geometry and packing logic is unit-testable without a device.
//! `render::vulkan` consumes the output ([`FrameData`]) verbatim.
//!
//! ```text
//!   &[DrawCmd] ─┬─ Fill/RoundRect ──────────────▶ quads (solid / SDF corners)
//!               └─ Text ─▶ FreeType rasters ─▶ atlas slots ─▶ textured quads
//! ```
//!
//! Text traversal matches how layout measured each run (the ASCII fast path,
//! then grapheme clusters routed to the emoji or per-character path), so the
//! pixels the GPU composites land exactly where the caret and wrapping expect.
//!
//! This file holds the frame's types, [`build_frame_into`] and the batcher that walks the
//! list. What the batcher draws from is split out:
//!
//! ```text
//!   atlas.rs  the shelf-packed atlases and the rasters queued for upload
//!   cache.rs  the slot maps that remember where each glyph was packed
//!   gamma.rs  a run's paint, and the coverage exponent that weights its edges
//! ```

mod atlas;
mod cache;
mod gamma;

use std::collections::HashMap;

use crate::platform::bytes;
use crate::platform::emoji::wants_emoji;
use crate::platform::freetype::{Face, FaceKey, Fonts};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::render::boxdraw;
use crate::render::display::DrawCmd;
use crate::render::gpu::atlas::{pack_argb, Atlas};
use crate::render::gpu::cache::{AsciiRow, ClusterSlots, EmojiFit, PackedGlyph};
use crate::render::gpu::gamma::Paint;

/// Vertex modes, matching shaders/quad.frag. (Mode 3, the decoded-image path,
/// is excised: a terminal draws no images.)
pub const MODE_SOLID: u32 = 0;

pub const MODE_GLYPH: u32 = 1;

pub const MODE_EMOJI: u32 = 2;

pub const MODE_ROUND: u32 = 4;

/// How hard to correct the two opposite ways linear-light compositing misweights
/// anti-aliased text. Two dials, not one, because they pull in opposite directions,
/// and nothing says the two corrections need the same magnitude; folding them into one
/// number makes "lighter correction" mean *bolder* in one direction and *thinner* in
/// the other. Both reach `1.0` for no correction at all. See `coverage_exponent`,
/// which turns them into a run's exponent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextGamma {
    /// Thins light-on-dark text. `> 1` thins, `1.0` is no correction.
    pub light_on_dark: f32,
    /// Thickens dark-on-light text, in proportion to the run's contrast. `> 1`
    /// thickens (it is raised to a negative power), `1.0` is no correction.
    pub dark_on_light: f32,
}

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

/// The atlas texels written since the last frame took them: each new raster's rectangle,
/// with its bytes tight and row-major in `bytes` from the region's `offset`. Only the
/// rasters are uploaded, never the box around them, which for two glyphs on shelves far
/// apart is most of the atlas.
#[derive(Default)]
pub struct AtlasUpload {
    pub regions: Vec<UploadRegion>,
    pub bytes: Vec<u8>,
}

/// One raster's rectangle in an [`AtlasUpload`], and where its bytes start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UploadRegion {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub offset: usize,
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
    /// The rasters each atlas gained this frame, empty when it gained none.
    pub glyph_upload: AtlasUpload,
    pub emoji_upload: AtlasUpload,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AtlasInfo {
    pub width: u32,
    pub height: u32,
    /// Bumped whenever the atlas is wiped, to grow or to reclaim its slots; the
    /// renderer recreates the texture at the new size (and re-uploads in full)
    /// when it changes.
    pub generation: u32,
}

/// The glyph cache's hit and miss tallies, for `--stats`. Once the working set is
/// packed into the atlas, steady-state frames add hits and almost no misses.
#[derive(Clone, Copy, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

/// The persistent glyph caches the batcher packs into: slots by scalar and by
/// cluster, over the two atlases. Owned by the app next to the display list;
/// survives across frames so steady-state frames insert nothing.
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
    /// [`boxdraw`] rasters, held apart from the font ones.
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
    /// Bounded by construction: [`boxdraw::is_glyph`] is 416 codepoints, so this holds at
    /// most `416 * faces` and needs no cap of its own.
    box_slots: HashMap<(FaceKey, char), PackedGlyph>,
    /// Emoji rasters at the em ([`EmojiFit::Em`]).
    cluster_slots: ClusterSlots,
    /// Emoji rasters at one cell's width ([`EmojiFit::Cell`]), held apart for the
    /// reason `box_slots` is: one cluster can reach both paths. ⚠️ and ❤️ are one
    /// column or two depending on mode 2027, and one that turns wide at the right
    /// margin stays narrow, so a shared key let whichever size drew first stand in
    /// for the other for the life of the process.
    cell_cluster_slots: ClusterSlots,
    /// The most records `scalar_slots`, or one face's map of clusters, holds before
    /// it is dropped: `MAX_GLYPH_CACHE`, or less in tests that need to overflow one.
    max_records: usize,
    /// Hits and misses across the scalar, box and emoji lookups since the cache was made.
    stats: CacheStats,
}

/// Build one frame's GPU data from the display list into `out` (whose vertex and
/// batch buffers are cleared and reused, so a steady frame allocates nothing).
/// Restarts internally if an atlas is wiped mid-build, by growing or because the
/// build left it [`spent`](Atlas::spent), so the vertices always reference current
/// atlas coordinates.
pub fn build_frame_into(
    fonts: &Fonts,
    list: &[DrawCmd],
    cache: &mut GlyphCache,
    gamma: TextGamma,
    out: &mut FrameData,
) {
    let mut wiped = false;
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
        // A spent atlas is wiped and the frame rebuilt into it, once per call. A
        // frame that alone overfills an atlas then draws what fit, and its other
        // glyphs stay recorded as unplaced until new glyphs fill the atlas again,
        // rather than wiping it every frame or rebuilding forever.
        let glyphs_spent = std::mem::take(&mut cache.glyphs.spent);
        let emoji_spent = std::mem::take(&mut cache.emoji.spent);
        if !wiped {
            if glyphs_spent {
                cache.glyphs.wipe();
            }
            if emoji_spent {
                cache.emoji.wipe();
            }
            wiped = glyphs_spent || emoji_spent;
        }
        // An atlas was wiped during the build: earlier quads reference dead
        // coordinates. Clear the stale slot records and rebuild.
        if cache.glyphs.generation != glyphs_gen || cache.emoji.generation != emoji_gen {
            cache.reset_for(glyphs_gen, emoji_gen);
            continue;
        }
        out.glyph_atlas = cache.glyphs.info();
        out.emoji_atlas = cache.emoji.info();
        cache.glyphs.take_upload(&mut out.glyph_upload);
        cache.emoji.take_upload(&mut out.emoji_upload);
        return;
    }
}

/// Build a fresh frame, allocating its vertex and batch vectors. The one-shot path
/// for tests; the render loop calls [`build_frame_into`] with a reused [`FrameData`].
#[cfg(test)]
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
            } => self.round_rect(*rect, *color, *radius),
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

    /// Emit one axis-aligned quad (two triangles) with texel UVs: `uv0` is the
    /// coordinate at the quad's top-left and uv advances 1:1 with pixels. Fills,
    /// glyphs, emoji and rounded corners all go through here.
    fn quad(&mut self, rect: Rect, mode: u32, color: u32, uv0: [f32; 2], extra: [f32; 4]) {
        if rect.w <= 0 || rect.h <= 0 {
            return;
        }
        let (x0, y0) = (rect.x as f32, rect.y as f32);
        let (x1, y1) = ((rect.x + rect.w) as f32, (rect.y + rect.h) as f32);
        let [u0, v0] = uv0;
        let (u1, v1) = (u0 + rect.w as f32, v0 + rect.h as f32);
        let c = color_f32(color);
        let v = |x: f32, y: f32, u: f32, vv: f32| Vertex {
            pos: [x, y],
            uv: [u, vv],
            color: c,
            extra,
            mode,
        };
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

    fn round_rect(&mut self, rect: Rect, color: u32, radius: i32) {
        // Clamp the radius to half the shorter side so opposite corners never
        // overlap; the shader's coverage function assumes this bound holds.
        let r = radius.clamp(0, (rect.w / 2).min(rect.h / 2).max(0));
        if r == 0 {
            self.quad(rect, MODE_SOLID, color, [0.0; 2], [0.0; 4]);
            return;
        }
        // uv carries the position within the rect (pixels); extra carries the
        // geometry the shader's coverage function needs. The shader's corner mask
        // is top 1, bottom 2; every rounded rect rounds all four, so it is 3.
        let extra = [rect.w as f32, rect.h as f32, r as f32, 3.0];
        self.quad(rect, MODE_ROUND, color, [0.0; 2], extra);
    }

    /// Emit a run's glyph quads: the ASCII fast path, then grapheme clusters
    /// routed to the emoji glyph or per-character drawing.
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
        for (_, cluster) in grapheme::graphemes(text) {
            match self.packed_cluster(face, face_key, cluster, EmojiFit::Em) {
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
    /// combining marks, [`boxdraw`] glyphs, or emoji.
    fn cells_segmented(
        &mut self,
        face_key: FaceKey,
        x: i32,
        baseline: i32,
        cell_w: i32,
        text: &str,
        paint: Paint,
    ) {
        // The cell box a procedurally-drawn glyph fills: its height and
        // the baseline offset that seats it, both from the same metrics the grid
        // laid out on (read once, not per cluster).
        let m = self.fonts.metrics(face_key.size());
        let (cell_h, baseline_offset) = (m.line_height.max(1), m.baseline);
        let face = self.fonts.face_for(face_key);
        for (i, (_, cluster)) in grapheme::graphemes(text).enumerate() {
            let pen = (x + i as i32 * cell_w) as f32;
            // `render::boxdraw` glyphs cover the cell exactly, so they tile. They are
            // single-width BMP scalars, so they only ever arrive as a lone-char cluster
            // on this fixed-pitch path, never through `text`.
            if let Some(ch) = box_glyph(cluster) {
                let packed = self.packed_box(face_key, ch, cell_w, cell_h, baseline_offset);
                self.emit_glyph(&packed, pen, baseline, MODE_GLYPH, paint);
                continue;
            }
            let fit = EmojiFit::Cell(cell_w.max(1) as u32);
            match self.packed_cluster(face, face_key, cluster, fit) {
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
    /// resolution is deterministic. A cache hit never re-resolves, so
    /// steady-state frames pay no fallback cost.
    fn packed_scalar(&mut self, face_key: FaceKey, ch: char) -> PackedGlyph {
        if let Some(packed) = self.cache.ascii_get(face_key, ch) {
            self.cache.stats.hits += 1;
            return packed;
        }
        if let Some(&packed) = self.cache.scalar_slots.get(&(face_key, ch)) {
            self.cache.stats.hits += 1;
            return packed;
        }
        self.cache.stats.misses += 1;
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

    /// The cached placement for a [`boxdraw`] glyph, sized to the
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
        if let Some(&packed) = self.cache.box_slots.get(&(face_key, ch)) {
            self.cache.stats.hits += 1;
            return packed;
        }
        self.cache.stats.misses += 1;
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
        // font map, neither of which applies here (`boxdraw::is_glyph` is a closed set
        // of 416 codepoints).
        self.cache.box_slots.insert((face_key, ch), packed);
        packed
    }

    /// The cached placement for an emoji cluster, or `None` when the cluster is
    /// not color emoji and the caller should draw its characters instead. The
    /// raster is packed into the color atlas on first sight and the decision
    /// cached, so a steady-state frame neither re-shapes nor touches the `Face`.
    /// Only emoji count toward the stats.
    ///
    /// `fit` is the size the caller's path draws at, and picks the map: the same
    /// cluster can arrive on both paths, and each keeps its own raster.
    ///
    /// A cluster that would not be emoji even where the text face lacks its glyph,
    /// such as a letter, an accented letter or a box-drawing character, returns
    /// `None` before the lookup, so plain text never enters the map.
    fn packed_cluster(
        &mut self,
        face: &Face,
        face_key: FaceKey,
        cluster: &str,
        fit: EmojiFit,
    ) -> Option<PackedGlyph> {
        if !wants_emoji(cluster, |_| false) {
            return None;
        }
        if let Some(&packed) = self
            .cache
            .clusters(fit)
            .get(&face_key)
            .and_then(|by| by.get(cluster))
        {
            if packed.is_some() {
                self.cache.stats.hits += 1;
            }
            return packed;
        }
        let atlas = &mut self.cache.emoji;
        let packed = face
            .cluster_glyph(cluster, fit.target(face_key))
            .map(|g| PackedGlyph {
                slot: pack_argb(atlas, g.width as u32, g.rows as u32, &g.argb),
                left: g.left,
                top: g.top,
                advance: g.advance,
            });
        if packed.is_some() {
            self.cache.stats.misses += 1;
        }
        self.cache.cache_cluster(fit, face_key, cluster, packed);
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
/// - **Never a box glyph.** Those start at `U+2500`, well outside ASCII.
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

/// The lone [`boxdraw`] scalar in `cluster`, or `None` if the cluster is not
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

#[cfg(test)]
mod tests {
    use crate::render::gpu::atlas::Slot;
    use crate::render::gpu::*;

    /// Both dials at 2.0 in tests, so an assertion reads as a plain power and a
    /// change to the shipping values in `app` cannot quietly move these numbers.
    pub(super) const GAMMA: TextGamma = TextGamma {
        light_on_dark: 2.0,
        dark_on_light: 2.0,
    };

    /// A placed glyph to cache; the coordinates are irrelevant to these tests, only
    /// whether the record survives or is dropped.
    pub(super) fn placed() -> PackedGlyph {
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

    /// One run of `text` in the 16 px prose face.
    pub(super) fn prose(text: &str) -> Vec<DrawCmd> {
        vec![DrawCmd::Text {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 1000,
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
            text: text.to_string(),
        }]
    }

    pub(super) fn glyph_quads(frame: &FrameData) -> usize {
        frame
            .vertices
            .iter()
            .filter(|v| v.mode == MODE_GLYPH)
            .count()
            / 6
    }

    /// A glyph first seen after the atlas filled at its cap used to be cached
    /// unplaced and stay invisible for the life of the process. The full atlas is
    /// wiped instead, and the frame packs into the reclaimed space.
    #[test]
    fn a_glyph_first_seen_after_the_atlas_fills_still_draws() {
        let fonts = Fonts::new(&[16]).expect("default font");
        // 64 texels square holds a few dozen 16 px glyphs, so the alphabet fills it.
        let mut cache = GlyphCache {
            glyphs: Atlas::with_max(1, 64),
            ..GlyphCache::default()
        };
        for ch in ('A'..='Z').chain('a'..='z').chain('0'..='9') {
            let frame = build_frame(&fonts, &prose(&ch.to_string()), &mut cache, GAMMA);
            assert_eq!(glyph_quads(&frame), 1, "{ch:?} drew nothing");
            assert!(!frame.glyph_upload.is_empty(), "{ch:?} was never uploaded");
        }
        assert!(cache.glyphs.generation > 0, "the atlas never filled");
    }

    /// Wiping cannot help a frame whose own glyphs overfill the atlas. It wipes
    /// once and draws what fits, and drawing it again must not wipe again: that
    /// would rebuild the frame forever, or throw the atlas away every frame.
    #[test]
    fn a_frame_larger_than_the_atlas_wipes_once_and_draws_what_fits() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache {
            glyphs: Atlas::with_max(1, 32),
            ..GlyphCache::default()
        };
        let alphabet: String = ('A'..='Z').chain('a'..='z').collect();
        let list = prose(&alphabet);

        let first = build_frame(&fonts, &list, &mut cache, GAMMA);
        let drawn = glyph_quads(&first);
        assert!(0 < drawn && drawn < 52, "{drawn} of 52 glyphs drawn");
        assert_eq!(cache.glyphs.generation, 1, "wiped once, then drew what fit");

        let again = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert_eq!(
            cache.glyphs.generation, 1,
            "the same frame wiped the atlas again"
        );
        assert_eq!(again.vertices, first.vertices);

        // A glyph the atlas has not seen still gets a wipe and draws.
        let fresh = build_frame(&fonts, &prose("0"), &mut cache, GAMMA);
        assert_eq!(glyph_quads(&fresh), 1, "the overfilled state stuck");
        assert_eq!(cache.glyphs.generation, 2);
    }

    /// A screen holding more distinct glyphs than a slot map drops the map on every
    /// frame. The first drop wipes the atlas; the ones after it must not, or every
    /// frame would rebuild and recreate both textures behind a device wait.
    #[test]
    fn a_screen_that_overflows_a_slot_map_wipes_the_atlases_once() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache {
            max_records: 16,
            ..GlyphCache::default()
        };
        // 25 distinct letters and 25 distinct emoji, past each map's cap of 16.
        let text: String = ('α'..='ω').chain('😀'..='😘').collect();
        let list = prose(&text);
        let emoji_quads = |frame: &FrameData| {
            frame
                .vertices
                .iter()
                .filter(|v| v.mode == MODE_EMOJI)
                .count()
                / 6
        };

        let first = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert_eq!(emoji_quads(&first), 25, "every emoji drew in colour");
        let generations = (cache.glyphs.generation, cache.emoji.generation);
        assert_eq!(generations, (1, 1), "the first drops wipe both atlases");
        for _ in 0..3 {
            let again = build_frame(&fonts, &list, &mut cache, GAMMA);
            assert_eq!(glyph_quads(&again), glyph_quads(&first));
            assert_eq!(emoji_quads(&again), emoji_quads(&first));
        }
        assert_eq!(
            (cache.glyphs.generation, cache.emoji.generation),
            generations,
            "a drop the screen forces on every frame wiped an atlas again"
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
            let mut cache = GlyphCache::default();
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
        let mut cache = GlyphCache::default();
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
        let mut cache = GlyphCache::default();
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
        assert!(
            !f1.glyph_upload.is_empty(),
            "first frame uploads new glyphs"
        );
        assert!(
            !f1.vertices.is_empty(),
            "glyph quads were emitted ({} vertices)",
            f1.vertices.len()
        );
        let f2 = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert!(f2.glyph_upload.is_empty(), "steady state uploads nothing");
        assert_eq!(f1.vertices, f2.vertices, "same list, same geometry");
    }

    #[test]
    fn glyph_quads_share_one_batch_with_fills() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::default();
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
        let mut cache = GlyphCache::default();
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
        }];
        let f = build_frame(&fonts, &list, &mut cache, GAMMA);
        assert_eq!(f.vertices.len(), 6);
        let v = &f.vertices[0];
        assert_eq!(v.mode, MODE_ROUND);
        assert_eq!(v.extra, [20.0, 10.0, 4.0, 3.0], "w, h, r, corner mask");
        assert_eq!(v.uv, [0.0, 0.0], "uv is rect-local pixels");
        assert_eq!(f.vertices[4].uv, [20.0, 10.0]);
    }

    #[test]
    fn zero_radius_round_rect_degrades_to_solid() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::default();
        let list = vec![DrawCmd::RoundRect {
            rect: Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
            color: 0,
            radius: 0,
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
        let mut cache = GlyphCache::default();
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
            face: FaceKey::Prose {
                size: 16,
                style: crate::platform::freetype::FontStyle::Regular,
            },
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
        let mut cache = GlyphCache::default();
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
            face: FaceKey::Prose {
                size: 16,
                style: crate::platform::freetype::FontStyle::Regular,
            },
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
        // Through the font, `▛` is tofu in many monospace fonts and `⠂` can come with
        // rings for its unraised dots. Both must be rasterized by `render::boxdraw` into
        // a quad covering exactly the cell box: top-left at (pen, baseline - m.baseline),
        // size (cell_w, cell_h). Seated on the ascent instead, it would float above its
        // cell by the line gap and leave a seam between rows.
        let fonts = Fonts::new(&[16]).expect("default font");
        let m = fonts.metrics(16);
        let (baseline_offset, cell_h) = (m.baseline, m.line_height.max(1));
        let cell_w = 11;
        let (x, baseline) = (7, 16);
        for glyph in ["▛", "⠂"] {
            let mut cache = GlyphCache::default();
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
                face: FaceKey::Prose {
                    size: 16,
                    style: crate::platform::freetype::FontStyle::Regular,
                },
                color: 0x00ff_ffff,
                bg: 0,
                text: glyph.to_string(),
            }];
            let f = build_frame(&fonts, &list, &mut cache, GAMMA);
            assert_eq!(f.vertices.len(), 6, "one full-cell quad for {glyph}");
            assert!(
                f.vertices.iter().all(|v| v.mode == MODE_GLYPH),
                "{glyph} drawn as a coverage glyph, not tofu or a solid fill"
            );
            // The quad is the cell box: top-left corner and bottom-right corner.
            assert_eq!(
                f.vertices[0].pos,
                [x as f32, (baseline - baseline_offset) as f32],
                "top-left seats {glyph} at the cell origin"
            );
            assert_eq!(
                f.vertices[4].pos,
                [
                    (x + cell_w) as f32,
                    (baseline - baseline_offset + cell_h) as f32
                ],
                "bottom-right fills the whole cell, so {glyph} tiles"
            );
        }
    }

    #[test]
    fn a_box_char_drawn_as_prose_does_not_poison_the_grid_raster() {
        // `─` has two right answers, and they are not interchangeable: on the grid it is
        // drawn by `render::boxdraw` to fill the cell edge to edge and tile with its
        // neighbours, and as prose it comes from the font at the font's own bearing and
        // advance. Cached under one key, whichever arrived first decided how `─` drew
        // everywhere for the rest of the process's life: tofu, or hairline seams where the
        // drawing should tile. Any `DrawCmd::Text` under a grid face can carry one.
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::default();
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
        let mut cache = GlyphCache::default();
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
            assert!(!f.emoji_upload.is_empty(), "the color raster was uploaded");
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

        let mut cache = GlyphCache::default();
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
        let mut cache2 = GlyphCache::default();
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

    /// One cluster can reach both paths, so each must keep its own raster in a
    /// shared cache, whichever path draws it first.
    #[test]
    fn an_emoji_drawn_by_both_paths_keeps_a_raster_per_path() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let key = FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        let warn = "\u{26A0}\u{FE0F}".to_string();
        let bounds = Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 20,
        };
        let cells = DrawCmd::Cells {
            bounds,
            x: 0,
            baseline: 16,
            cell_w: 10,
            face: key,
            color: 0x00ff_ffff,
            bg: 0,
            text: warn.clone(),
        };
        let text = DrawCmd::Text {
            bounds,
            x: 0,
            baseline: 16,
            face: key,
            color: 0x00ff_ffff,
            bg: 0,
            fade: None,
            text: warn,
        };
        // The width of each emoji quad, in the order the commands drew them.
        let widths = |list: &[DrawCmd], cache: &mut GlyphCache| -> Vec<f32> {
            let frame = build_frame(&fonts, list, cache, GAMMA);
            frame
                .vertices
                .chunks(6)
                .filter(|quad| quad[0].mode == MODE_EMOJI)
                .map(|quad| quad[4].pos[0] - quad[0].pos[0])
                .collect()
        };
        let cell = widths(std::slice::from_ref(&cells), &mut GlyphCache::default());
        let em = widths(std::slice::from_ref(&text), &mut GlyphCache::default());
        let ([cell], [em]) = (cell.as_slice(), em.as_slice()) else {
            eprintln!("no colour emoji font here; nothing to compare");
            return;
        };
        assert!(cell < em, "the cell raster ({cell}) is the em one ({em})");

        let shared = &mut GlyphCache::default();
        let both = widths(&[cells.clone(), text.clone()], shared);
        assert_eq!(both, [*cell, *em], "the cell path drew first");
        let shared = &mut GlyphCache::default();
        let both = widths(&[text, cells], shared);
        assert_eq!(both, [*em, *cell], "the text path drew first");
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
        let mut cache = GlyphCache::default();
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
}
