//! The color emoji font: a bitmap-strike (CBDT) FreeType face plus a HarfBuzz
//! shaper over it. It turns a grapheme cluster (a ZWJ family, a flag, a skin
//! tone, a keycap) into a single ligature glyph, decodes that glyph's color
//! strike, and scales it to the surrounding text size, caching the result per
//! (cluster, size). Sits above the FFI layer in `freetype.rs` (whose faces and
//! glyph-slot mirror it reads) and the pure pixel math in `pixel.rs`.

use core::ffi::{c_uint, c_void};
use core::ptr::NonNull;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::platform::freetype::{
    CacheStats, FT_Load_Glyph, Face, FtBitmap, FT_LOAD_COLOR, FT_LOAD_RENDER, FT_PIXEL_MODE_BGRA,
};
use crate::platform::grapheme;
use crate::platform::pixel::{premultiplied_over, resample_argb, straight_from_premultiplied};
use crate::platform::shape::Shaper;

/// Whether a grapheme cluster should render as color emoji. `mono_has` reports
/// whether the text face can draw a character itself, so symbols the text font
/// covers (dagger, copyright, box drawing) stay text.
///
/// The rules, in order:
/// - a text-presentation selector (U+FE0E) anywhere forces the text path;
/// - a single scalar goes color only when it is `Extended_Pictographic` *and*
///   the text face has no glyph for it (a smiley, not a copyright sign);
/// - a multi-scalar cluster goes color when it carries an emoji-presentation
///   selector (U+FE0F) or keycap (U+20E3), starts pictographic (ZWJ sequences,
///   skin tones), or starts with a regional indicator (flags). Anything else
///   (plain combining marks) is ordinary text.
pub(crate) fn wants_emoji(cluster: &str, mono_has: impl Fn(char) -> bool) -> bool {
    let mut chars = cluster.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if cluster.contains('\u{FE0E}') {
        return false;
    }
    if chars.next().is_none() {
        return grapheme::is_extended_pictographic(first) && !mono_has(first);
    }
    cluster.contains('\u{FE0F}')
        || cluster.contains('\u{20E3}')
        || grapheme::is_extended_pictographic(first)
        || matches!(first, '\u{1F1E6}'..='\u{1F1FF}')
}

/// A rasterized emoji cluster at display size: straight (unpremultiplied)
/// `0xAARRGGBB` pixels, row-major, plus the same pen metrics [`Glyph`] carries.
/// Owned and cached by [`EmojiFont`]; the blitter draws straight out of it.
pub struct ColorGlyph {
    /// Horizontal offset from the pen to the bitmap's left edge.
    pub left: i32,
    /// Vertical offset from the baseline up to the bitmap's top edge.
    pub top: i32,
    pub width: usize,
    pub rows: usize,
    /// Pen advance in pixels.
    pub advance: f32,
    pub argb: Vec<u32>,
}

/// Where distros install the Noto color emoji font. Same hardcoded-path policy
/// as [`FONT_FAMILIES`]; a machine with none of these simply has no emoji.
const EMOJI_FONTS: &[&str] = &[
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/google-noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
];

/// A cache of cluster results, keyed by target pixel size then by cluster bytes.
/// Splitting on the size (a cheap `u32` key) lets the inner lookup borrow the
/// cluster `&str` directly (`Box<str>: Borrow<str>`), so a cache *hit* allocates
/// nothing; only a miss boxes the cluster to insert it. `None` records that the
/// font cannot form the cluster, so the miss is not re-shaped every frame.
type ClusterCache<T> = RefCell<HashMap<u32, HashMap<Box<str>, Option<T>>>>;

/// The color emoji font: a bitmap-strike (CBDT) face plus a HarfBuzz shaper
/// over it. The shaper is what turns a multi-scalar cluster (ZWJ family, flag,
/// skin tone, keycap) into the single ligature glyph the font's GSUB table
/// defines; FreeType then decodes that glyph's color strike, and the result is
/// scaled to the text size and cached per (cluster, size).
///
/// Everything degrades gracefully: no installed font, an outline-only emoji
/// font, or a cluster the font cannot form all yield `None`, and the caller
/// falls back to per-character rendering.
pub struct EmojiFont {
    /// Declared before `face` deliberately: fields drop in declaration order,
    /// and the shaper's HarfBuzz font holds a reference to the FreeType face
    /// that it releases with `FT_Done_Face` on destroy. The face's own `Drop`
    /// tears down its whole FreeType library, so HarfBuzz must let go first.
    shaper: Shaper,
    face: Face,
    /// The strike's nominal pixel size; glyph bitmaps and shaped positions are
    /// scaled by `target / strike` to land on the text's em square.
    strike: f32,
    /// The measure-only cache, filled by shaping alone (no strike decode), so
    /// layout can measure off-screen emoji without rasterizing them. The GPU
    /// atlas holds the rasterized clusters, so there is no color-glyph cache here.
    advances: ClusterCache<f32>,
    /// Cluster-cache hit and miss tallies, reported by the GPU batcher (which
    /// owns the atlas-placement cache) through [`Face::record_cluster_hit`] and
    /// [`Face::record_cluster_miss`]. Pure instrumentation.
    hits: Cell<u64>,
    misses: Cell<u64>,
}

impl EmojiFont {
    /// Open the first installed emoji font, or `None` when there is none, the
    /// font has no bitmap strikes (an outline-only build), or the shaper cannot
    /// bind to it.
    pub fn open() -> Option<Self> {
        let path = EMOJI_FONTS
            .iter()
            .find(|p| std::path::Path::new(p).exists())?;
        let face = Face::from_path(path).ok()?;
        let strike = face.select_first_strike()?;
        // SAFETY: the face handle is valid and has a size selected; the shaper
        // takes its own FreeType reference, so drop order does not matter.
        let shaper = unsafe { Shaper::from_ft_face(face.ft_face_ptr() as *mut c_void) }?;
        Some(Self {
            face,
            shaper,
            strike,
            advances: RefCell::new(HashMap::new()),
            hits: Cell::new(0),
            misses: Cell::new(0),
        })
    }

    /// Pen advance of `cluster` scaled to `target` pixels, or `None` when the
    /// font cannot form the cluster. Cached; filled by shaping only, no bitmap
    /// is decoded.
    pub(crate) fn advance(&self, cluster: &str, target: u32) -> Option<f32> {
        let mut cache = self.advances.borrow_mut();
        let by_cluster = cache.entry(target).or_default();
        // Borrow the cluster to probe: a hit returns without boxing it.
        if let Some(&hit) = by_cluster.get(cluster) {
            return hit;
        }
        let advance = self
            .shaped(cluster)
            .map(|glyphs| glyphs.iter().map(|g| g.x_advance).sum::<f32>() * self.scale(target));
        by_cluster.insert(cluster.into(), advance);
        advance
    }

    /// Run `f` over the color glyph for `cluster` at `target` pixels, rasterizing
    /// it (shape, decode strikes, composite, resample) on the spot. `None` means
    /// the font cannot form this cluster, matching what [`Self::advance`]
    /// reported, since both derive from the same shaping. The GPU batcher caches
    /// the atlas placement, so it calls this only on the first sight of a cluster.
    pub(crate) fn with_glyph<R>(
        &self,
        cluster: &str,
        target: u32,
        f: impl FnOnce(&ColorGlyph) -> R,
    ) -> Option<R> {
        Some(f(&self.raster(cluster, target)?))
    }

    /// Record that the GPU batcher served a color cluster from its atlas cache.
    /// Pure instrumentation.
    pub(crate) fn record_hit(&self) {
        self.hits.set(self.hits.get() + 1);
    }

    /// Record that the GPU batcher had to rasterize and pack a color cluster.
    /// Pure instrumentation.
    pub(crate) fn record_miss(&self) {
        self.misses.set(self.misses.get() + 1);
    }

    /// This font's cluster-cache hit and miss counts, folded into the frame
    /// stats alongside the per-character caches.
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.get(),
            misses: self.misses.get(),
        }
    }

    /// Shape `cluster` and keep the result only when the font can really form
    /// it: at least one glyph and no `.notdef`. The single success predicate
    /// behind both [`Self::advance`] and [`Self::with_glyph`], which is what
    /// keeps measuring and drawing in agreement.
    fn shaped(&self, cluster: &str) -> Option<Vec<crate::platform::shape::ShapedGlyph>> {
        let glyphs = self.shaper.shape(cluster);
        if glyphs.is_empty() || glyphs.iter().any(|g| g.id == 0) {
            return None;
        }
        Some(glyphs)
    }

    /// How much a strike-sized measurement shrinks to land on `target` pixels.
    fn scale(&self, target: u32) -> f32 {
        target as f32 / self.strike
    }

    /// Rasterize `cluster` at `target` pixels: decode each shaped glyph's color
    /// strike, composite them along the pen into one strike-sized image, then
    /// resample once to the target size. Emoji clusters almost always shape to
    /// a single glyph, so the composite loop usually runs once.
    ///
    /// The pipeline stays *premultiplied* (as FreeType decodes it) through
    /// compositing and resampling, and un-premultiplies only the final
    /// display-size pixels. Filtering straight alpha would mix the black of
    /// transparent texels into edge colors and fringe the glyph outline dark.
    fn raster(&self, cluster: &str, target: u32) -> Option<ColorGlyph> {
        let shaped = self.shaped(cluster)?;
        let scale = self.scale(target);
        let (placed, pen) = self.place_strikes(&shaped);
        let advance = pen * scale;
        if placed.is_empty() {
            // Shaped but inkless; keep the advance so layout stays consistent.
            return Some(ColorGlyph {
                left: 0,
                top: 0,
                width: 0,
                rows: 0,
                advance,
                argb: Vec::new(),
            });
        }
        let Composited {
            x0,
            top,
            w,
            h,
            argb,
        } = Self::composite(&placed);
        let dst_w = ((w as f32) * scale).round().max(1.0) as i32;
        let dst_h = ((h as f32) * scale).round().max(1.0) as i32;
        let argb: Vec<u32> = resample_argb(&argb, w as u32, h as u32, dst_w, dst_h)
            .into_iter()
            .map(straight_from_premultiplied)
            .collect();
        Some(ColorGlyph {
            left: (x0 as f32 * scale).round() as i32,
            top: (top as f32 * scale).round() as i32,
            width: dst_w as usize,
            rows: dst_h as usize,
            advance,
            argb,
        })
    }

    /// Decode and place each shaped glyph's color strike in strike space,
    /// returning the placed strikes and the total pen advance (in strike units,
    /// before the display-size scale). A glyph the font cannot decode is skipped
    /// but still advances the pen, so an inkless cluster keeps its width.
    fn place_strikes(&self, shaped: &[crate::platform::shape::ShapedGlyph]) -> (Vec<Placed>, f32) {
        let mut placed = Vec::new();
        let mut pen = 0.0f32;
        for g in shaped {
            if let Some(bitmap) = self.load_strike(g.id) {
                // x/y are the bitmap's left and top edges relative to the pen
                // origin and baseline.
                let x = (pen + g.x_offset).round() as i32 + bitmap.left;
                let y = bitmap.top + g.y_offset.round() as i32;
                placed.push(Placed { x, y, bitmap });
            }
            pen += g.x_advance;
        }
        (placed, pen)
    }

    /// Composite the placed strikes into one premultiplied, strike-sized canvas.
    /// A single pass over the strikes finds the union box (top-left `x0`/`top`,
    /// size `w` x `h`); a second blends each strike onto the canvas with
    /// premultiplied source-over. Only called for a non-empty placement.
    fn composite(placed: &[Placed]) -> Composited {
        // The union box of every placed bitmap, still in strike space, folded in
        // one pass: x0/x1 are the horizontal extent, top/bottom the vertical.
        let (mut x0, mut x1, mut top, mut bottom) = (i32::MAX, i32::MIN, i32::MIN, i32::MAX);
        for p in placed {
            x0 = x0.min(p.x);
            x1 = x1.max(p.x + p.bitmap.width as i32);
            top = top.max(p.y);
            bottom = bottom.min(p.y - p.bitmap.rows as i32);
        }
        let (w, h) = ((x1 - x0).max(0) as usize, (top - bottom).max(0) as usize);
        let mut argb = vec![0u32; w * h];
        for p in placed {
            for row in 0..p.bitmap.rows {
                let dst_row = (top - p.y) as usize + row;
                for col in 0..p.bitmap.width {
                    let dst_col = (p.x - x0) as usize + col;
                    let src = p.bitmap.argb[row * p.bitmap.width + col];
                    let dst = &mut argb[dst_row * w + dst_col];
                    *dst = premultiplied_over(src, *dst);
                }
            }
        }
        Composited {
            x0,
            top,
            w,
            h,
            argb,
        }
    }

    /// Decode one glyph's color strike into premultiplied ARGB. `None` for a
    /// load failure or a non-BGRA result (an inkless or malformed glyph).
    fn load_strike(&self, glyph_id: u32) -> Option<StrikeBitmap> {
        // SAFETY: face is valid with a strike selected; FT_LOAD_COLOR decodes
        // the CBDT strike into the face's shared glyph slot, read out below
        // before any later FreeType call.
        let err = unsafe {
            FT_Load_Glyph(
                self.face.ft_face_ptr(),
                glyph_id as c_uint,
                FT_LOAD_RENDER | FT_LOAD_COLOR,
            )
        };
        if err != 0 {
            return None;
        }
        // SAFETY: the load succeeded, so face->glyph points at the populated slot.
        let slot = unsafe { (*self.face.ft_face_ptr()).glyph };
        let slot = NonNull::new(slot)?;
        let slot = unsafe { slot.as_ref() };
        if slot.bitmap.pixel_mode != FT_PIXEL_MODE_BGRA {
            return None;
        }
        Some(StrikeBitmap {
            left: slot.bitmap_left,
            top: slot.bitmap_top,
            width: slot.bitmap.width as usize,
            rows: slot.bitmap.rows as usize,
            argb: copy_bgra(&slot.bitmap),
        })
    }
}

/// One glyph's decoded color strike in strike space: premultiplied ARGB plus
/// its placement relative to the pen and baseline.
struct StrikeBitmap {
    left: i32,
    top: i32,
    width: usize,
    rows: usize,
    argb: Vec<u32>,
}

/// A decoded strike positioned in strike space: `x`/`y` are the bitmap's left
/// and top edges relative to the pen origin and baseline.
struct Placed {
    x: i32,
    y: i32,
    bitmap: StrikeBitmap,
}

/// A composited, premultiplied, strike-sized canvas: its top-left corner in
/// strike space (`x0`, `top`) and the `w` x `h` `argb` pixels.
struct Composited {
    x0: i32,
    top: i32,
    w: usize,
    h: usize,
    argb: Vec<u32>,
}

/// Copy a BGRA color strike into tightly packed `0xAARRGGBB` pixels, keeping
/// FreeType's premultiplied alpha as-is (a pure byte reorder). Compositing and
/// resampling stay premultiplied; [`straight_from_premultiplied`] converts the
/// final display-size pixels for the straight-alpha blitter. The caller must
/// confirm FT_PIXEL_MODE_BGRA first.
fn copy_bgra(bitmap: &FtBitmap) -> Vec<u32> {
    let width = bitmap.width as usize;
    let rows = bitmap.rows as usize;
    if width == 0 || rows == 0 || bitmap.buffer.is_null() {
        return Vec::new();
    }
    let mut out = vec![0u32; width * rows];
    let pitch = bitmap.pitch as isize;
    for y in 0..rows {
        for x in 0..width {
            // SAFETY: FreeType guarantees |pitch| >= width*4 bytes per BGRA row,
            // so the 4-byte read at buffer + y*pitch + x*4 stays within the row;
            // out is a distinct buffer.
            let [b, g, r, a] = unsafe {
                let p = bitmap.buffer.offset(y as isize * pitch + (x as isize) * 4);
                [*p, *p.add(1), *p.add(2), *p.add(3)]
            };
            out[y * width + x] = u32::from_be_bytes([a, r, g, b]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emoji() -> EmojiFont {
        EmojiFont::open().expect("a color emoji font should be installed")
    }

    #[test]
    fn wants_emoji_routes_clusters_correctly() {
        let mono_lacks = |_: char| false;
        let mono_has = |_: char| true;
        // Plain text and combining marks never route to emoji.
        assert!(!wants_emoji("a", mono_lacks));
        assert!(
            !wants_emoji("e\u{301}", mono_lacks),
            "combining mark is text"
        );
        // A lone pictographic scalar goes color only when the text face lacks it.
        assert!(wants_emoji("😀", mono_lacks));
        assert!(!wants_emoji("😀", mono_has), "the text face covers it");
        // Presentation selectors win in both directions.
        assert!(wants_emoji("☺\u{fe0f}", mono_has), "emoji presentation");
        assert!(!wants_emoji("☺\u{fe0e}", mono_lacks), "text presentation");
        // Flags, skin tones, ZWJ sequences, and keycaps are emoji outright.
        assert!(wants_emoji("🇳🇴", mono_has));
        assert!(wants_emoji("👍🏽", mono_has));
        assert!(wants_emoji("👨\u{200d}👩\u{200d}👧", mono_has));
        assert!(wants_emoji("1\u{fe0f}\u{20e3}", mono_has));
    }

    #[test]
    fn emoji_font_measures_and_rasters_a_smiley() {
        let font = emoji();
        let advance = font.advance("😀", 32).expect("the smiley should shape");
        assert!(advance > 0.0);
        let (width, rows, has_ink, glyph_advance) = font
            .with_glyph("😀", 32, |g| {
                let ink = g.argb.iter().any(|&p| p >> 24 != 0);
                (g.width, g.rows, ink, g.advance)
            })
            .expect("the smiley should raster");
        assert!(width > 0 && rows > 0, "the smiley has pixels");
        assert!(has_ink, "the smiley has visible ink");
        assert!(
            rows <= 40,
            "scaled to the 32px em square, not the strike ({rows} rows)"
        );
        assert_eq!(glyph_advance, advance, "measure and raster agree");
    }

    #[test]
    fn zwj_and_flag_clusters_ligate_to_one_glyph() {
        let font = emoji();
        let single = font.advance("👨", 32).expect("the man emoji shapes");
        let family = font
            .advance("👨\u{200d}👩\u{200d}👧", 32)
            .expect("the family should ligate");
        assert!(
            family < single * 2.0,
            "a family is one ligature, not three glyphs ({family} vs {single})"
        );
        let flag = font.advance("🇳🇴", 32).expect("the flag should ligate");
        assert!(
            flag < single * 2.0,
            "a flag is one glyph, not two letter symbols"
        );
    }

    #[test]
    fn cluster_stats_count_reported_hits_and_misses() {
        let font = emoji();
        // with_glyph rasterizes on demand (the GPU atlas holds the result); the
        // hit and miss tallies are reported by the batcher, and both roll up
        // through cache_stats.
        font.with_glyph("🎉", 32, |_| ()).expect("party popper");
        font.record_miss();
        font.record_hit();
        let stats = font.cache_stats();
        assert_eq!((stats.hits, stats.misses), (1, 1));
    }

    #[test]
    fn an_unformable_cluster_answers_none() {
        let font = emoji();
        // A letter with a combining mark is no emoji ligature; the emoji font
        // has no glyphs for it and must say so rather than render garbage.
        // (Routing filters this out anyway; the font still answers safely.)
        assert!(font.advance("e\u{301}", 32).is_none());
        assert!(font.with_glyph("e\u{301}", 32, |_| ()).is_none());
    }
}
