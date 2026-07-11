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

use crate::platform::freetype::{Face, FaceKey, Fonts};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::render::boxdraw;
use crate::render::display::DrawCmd;
use crate::render::display::RoundedCorners;

/// Vertex modes, matching shaders/quad.frag. (Mode 3, the decoded-image path,
/// is excised: a terminal draws no images.)
pub const MODE_SOLID: u32 = 0;
pub const MODE_GLYPH: u32 = 1;
pub const MODE_EMOJI: u32 = 2;
pub const MODE_ROUND: u32 = 4;

/// A glyph run's paint, threaded through the glyph emitters as one value: the
/// foreground `color` and the coverage-gamma `factor` derived once from its
/// contrast with the run background (see [`contrast_factor`]).
#[derive(Clone, Copy)]
struct Paint {
    color: u32,
    factor: f32,
}

impl Paint {
    fn new(color: u32, bg: u32) -> Self {
        Paint {
            color,
            factor: contrast_factor(color, bg),
        }
    }
}

/// Atlases start here and double when full, up to [`ATLAS_MAX`]; growth wipes
/// the atlas (sources are cached in `Fonts`, so re-inserting is cheap) and the
/// frame build restarts so no stale coordinates survive.
const ATLAS_START: u32 = 1024;
const ATLAS_MAX: u32 = 8192;

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

    /// Reserve a slot for a `w` x `h` raster and write `src` into it, returning
    /// the slot; `None` for a zero-area raster or a full atlas (the caller
    /// records the empty placement so the pen still advances). `src` is
    /// `w * h * bpp` bytes, row-major, tight.
    fn pack(&mut self, w: u32, h: u32, src: &[u8]) -> Option<Slot> {
        if w == 0 || h == 0 {
            return None;
        }
        let slot = self.reserve(w, h)?;
        self.write(slot, src);
        Some(slot)
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
pub struct GlyphCache {
    pub glyphs: Atlas,
    pub emoji: Atlas,
    scalar_slots: HashMap<(FaceKey, char), PackedGlyph>,
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
            scalar_slots: HashMap::new(),
            cluster_slots: HashMap::new(),
        }
    }

    /// Drop every slot record (after an atlas grew and wiped its content).
    fn reset_for(&mut self, glyphs_gen: u32, emoji_gen: u32) {
        if self.glyphs.generation != glyphs_gen {
            self.scalar_slots.clear();
        }
        if self.emoji.generation != emoji_gen {
            self.cluster_slots.clear();
        }
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
pub fn build_frame(fonts: &Fonts, list: &[DrawCmd], cache: &mut GlyphCache) -> FrameData {
    let mut out = FrameData::default();
    build_frame_into(fonts, list, cache, &mut out);
    out
}

struct Batcher<'a> {
    fonts: &'a Fonts,
    cache: &'a mut GlyphCache,
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
                text,
                ..
            } => self.text(*face, *x, *baseline, text, Paint::new(*color, *bg)),
            DrawCmd::Cells {
                x,
                baseline,
                cell_w,
                face,
                color,
                bg,
                text,
                ..
            } => self.cells(*face, *x, *baseline, *cell_w, text, Paint::new(*color, *bg)),
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
        // The cell box a procedurally-drawn box/block glyph fills: its height and
        // the baseline offset that seats it, both from the same metrics the grid
        // laid out on (read once, not per cluster).
        let m = self.fonts.metrics(face_key.size());
        let (cell_h, ascent) = (m.line_height.max(1), m.ascent);
        let face = self.fonts.face_for(face_key);
        for (i, (_, cluster)) in grapheme::graphemes(text).enumerate() {
            let pen = (x + i as i32 * cell_w) as f32;
            // Box Drawing and Block Elements are rasterized here, not by the font,
            // so they cover the cell exactly and tile (see `render::boxdraw`).
            // They are single-width BMP scalars, so they only ever arrive as a
            // lone-char cluster on this fixed-pitch path, never through `text`.
            if let Some(ch) = box_glyph(cluster) {
                let packed = self.packed_box(face_key, ch, cell_w, cell_h, ascent);
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
        self.cache.scalar_slots.insert((face_key, ch), packed);
        packed
    }

    /// The cached placement for a procedurally-drawn box/block glyph, sized to the
    /// `w`-by-`h` cell. Its coverage comes from [`boxdraw::coverage`] instead of a
    /// FreeType raster, and it is anchored at `left = 0, top = ascent` so the
    /// bitmap fills the cell box `[pen, pen + w) x [baseline - ascent, + h)`
    /// exactly, tiling with its neighbours. Cached in the same `scalar_slots` map
    /// as a font glyph: the `(face_key, ch)` key stays unique because a given char
    /// is always a box glyph or never one, and the cell size is fixed by the size
    /// the key carries.
    fn packed_box(
        &mut self,
        face_key: FaceKey,
        ch: char,
        w: i32,
        h: i32,
        ascent: i32,
    ) -> PackedGlyph {
        let primary = self.fonts.face_for(face_key);
        if let Some(&packed) = self.cache.scalar_slots.get(&(face_key, ch)) {
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
            top: ascent,
            advance: w as f32,
        };
        self.cache.scalar_slots.insert((face_key, ch), packed);
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
        self.cache
            .cluster_slots
            .entry(face_key)
            .or_default()
            .insert(cluster.to_string(), packed);
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
        // A coverage glyph (mode 1) carries its per-run contrast factor in extra.x,
        // steering the shader's coverage gamma; emoji (mode 2) ignore it.
        let extra = if mode == MODE_GLYPH {
            [paint.factor, 0.0, 0.0, 0.0]
        } else {
            [0.0; 4]
        };
        self.quad(
            rect,
            mode,
            paint.color,
            [slot.x as f32, slot.y as f32],
            extra,
        );
    }
}

/// Pack an emoji cluster's straight-`0xAARRGGBB` raster into the color atlas.
/// `None` for a zero-area or malformed glyph or a full atlas; the placement is
/// recorded as `None` so the pen advances without drawing.
fn pack_argb(atlas: &mut Atlas, w: u32, h: u32, argb: &[u32]) -> Option<Slot> {
    if w == 0 || h == 0 || argb.len() < (w * h) as usize {
        return None;
    }
    // ARGB words are B,G,R,A bytes in memory: B8G8R8A8 verbatim.
    let bytes: Vec<u8> = argb.iter().flat_map(|px| px.to_ne_bytes()).collect();
    atlas.pack(w, h, &bytes)
}

/// Re-place a single-width color-emoji glyph so it sits on the text line rather
/// than on its natural pen bearing (which seats it near the cell's vertical
/// middle and reads *low* beside baseline-aligned text). The glyph is
/// bottom-aligned to the baseline and centered horizontally in the column, so it
/// rests on the line like the caps and digits around it and rises a little above
/// the x-height, exactly how a warning sign reads in other terminals.
///
/// The bottom dips `h / 12` below the baseline on purpose: a color glyph carries
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

/// The per-run coverage-gamma exponent driver in `[-1, 1]`, from the run's
/// foreground and background colours. The fragment shader raises glyph coverage to
/// `G^factor` (`G` the base [`crate::app`] gamma):
///
/// - `+1` — foreground lighter than background, the usual light-on-dark terminal
///   text: reproduces the tuned thinning that offsets linear-light compositing,
///   byte-for-byte with the old single-gamma behaviour.
/// - `< 1` — dark text on a lighter background (a reverse-video paste highlight, a
///   light theme): thickens the anti-aliased edges that linear-light compositing
///   would otherwise wash out, reaching `-1` (`G^-1`, the inverse) at maximum
///   contrast. `0` leaves coverage untouched.
///
/// Only the sign-crossing near equal luminance matters; the magnitude is a soft
/// weight, so a perceptual (gamma-space) luma is enough.
fn contrast_factor(fg: u32, bg: u32) -> f32 {
    let d = luma(fg) - luma(bg);
    if d >= 0.0 {
        1.0
    } else {
        (1.0 + 2.0 * d).clamp(-1.0, 1.0)
    }
}

/// Rec. 709 relative luminance of a `0x00RRGGBB` colour in `[0, 1]`, taken in the
/// gamma-encoded byte space as a perceptual stand-in (enough to pick the contrast
/// direction and a soft strength; see [`contrast_factor`]).
fn luma(color: u32) -> f32 {
    let r = ((color >> 16) & 0xff) as f32;
    let g = ((color >> 8) & 0xff) as f32;
    let b = (color & 0xff) as f32;
    (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let f = build_frame(&fonts, &list, &mut cache);
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
            text: "abcabc".to_string(),
        }];
        let f1 = build_frame(&fonts, &list, &mut cache);
        assert!(f1.glyph_upload.is_some(), "first frame uploads new glyphs");
        assert!(
            !f1.vertices.is_empty(),
            "glyph quads were emitted ({} vertices)",
            f1.vertices.len()
        );
        let f2 = build_frame(&fonts, &list, &mut cache);
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
                text: "hi".to_string(),
            },
        ];
        let f = build_frame(&fonts, &list, &mut cache);
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
        let f = build_frame(&fonts, &list, &mut cache);
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
        let f = build_frame(&fonts, &list, &mut cache);
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
        let f = build_frame(&fonts, &list, &mut cache);
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
        let f = build_frame(&fonts, &list, &mut cache);
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
        // it tiles: top-left at (pen, baseline - ascent), size (cell_w, cell_h).
        let fonts = Fonts::new(&[16]).expect("default font");
        let mut cache = GlyphCache::new();
        let m = fonts.metrics(16);
        let (ascent, cell_h) = (m.ascent, m.line_height.max(1));
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
        let f = build_frame(&fonts, &list, &mut cache);
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
            [x as f32, (baseline - ascent) as f32],
            "top-left seats the glyph at the cell origin"
        );
        assert_eq!(
            f.vertices[4].pos,
            [(x + cell_w) as f32, (baseline - ascent + cell_h) as f32],
            "bottom-right fills the whole cell, so the glyph tiles"
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
            text: "😀".to_string(),
        }];
        let f = build_frame(&fonts, &list, &mut cache);
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
                text: warn,
            }],
            &mut cache2,
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

    #[test]
    fn contrast_factor_preserves_light_on_dark_and_thickens_dark_on_light() {
        let white = 0x00ff_ffff;
        let black = 0x0000_0000;
        // Light text on a dark background is the tuned default: factor 1 reproduces
        // the old single-gamma thinning exactly (exponent G^1 = G).
        assert_eq!(contrast_factor(white, black), 1.0);
        // A mid-grey background is still darker than white text: unchanged.
        assert_eq!(contrast_factor(white, 0x0080_8080), 1.0);
        // Dark text on white (a reverse-video paste highlight) is thickened: the
        // factor drops below 1, bottoming at -1 for maximum contrast (exponent G^-1).
        assert_eq!(contrast_factor(black, white), -1.0);
        assert!(contrast_factor(black, white) < contrast_factor(0x0060_6060, white));
        // Equal luminance takes the no-thinning branch rather than a discontinuity.
        assert_eq!(contrast_factor(0x0044_4444, 0x0044_4444), 1.0);
        // The driver never escapes the range the shader's pow expects.
        for &(fg, bg) in &[(white, black), (black, white), (0x0012_3456, 0x00fe_dcba)] {
            let f = contrast_factor(fg, bg);
            assert!((-1.0..=1.0).contains(&f), "factor {f} in range");
        }
    }
}
