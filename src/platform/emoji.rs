//! The color emoji font: a bitmap-strike (CBDT) FreeType face plus a HarfBuzz
//! shaper over it. It turns a grapheme cluster (a ZWJ family, a flag, a skin
//! tone, a keycap) into a single ligature glyph, decodes that glyph's color
//! strike, and scales it to the surrounding text size. Sits above the FFI layer in `freetype.rs` (whose faces and
//! glyph-slot mirror it reads) and the pure pixel math in `pixel.rs`.

use core::ffi::c_uint;
use core::ptr::NonNull;
use std::cell::Cell;
use std::rc::Rc;

use crate::platform::fontconfig::FontFile;
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
/// `0xAARRGGBB` pixels, row-major, plus the same pen metrics [`Glyph`](crate::platform::freetype::Glyph) carries.
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

/// The color emoji font: a bitmap-strike (CBDT) face plus a HarfBuzz shaper
/// over it. The shaper is what turns a multi-scalar cluster (ZWJ family, flag,
/// skin tone, keycap) into the single ligature glyph the font's GSUB table
/// defines; FreeType then decodes that glyph's color strike, and the result is
/// scaled to the text size.
///
/// Everything degrades gracefully: no installed font, an outline-only emoji
/// font, or a cluster the font cannot form all yield `None`, and the caller
/// falls back to per-character rendering.
pub struct EmojiFont {
    /// Shared with `shaper`, which keeps it alive for as long as HarfBuzz uses it.
    face: Rc<Face>,
    shaper: Shaper,
    /// The strike's nominal pixel size; glyph bitmaps and shaped positions are
    /// scaled by `target / strike` to land on the text's em square.
    strike: f32,
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
    pub fn open(file: &FontFile) -> Option<Self> {
        let face = Rc::new(Face::from_file(file).ok()?);
        // Reject a text face returned for an emoji alias.
        let strike = face.select_first_strike()?;
        let shaper = Shaper::new(Rc::clone(&face))?;
        Some(Self {
            face,
            shaper,
            strike,
            hits: Cell::new(0),
            misses: Cell::new(0),
        })
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
    /// it: at least one glyph and no `.notdef`.
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

    /// Rasterize `cluster` at `target` pixels, or `None` when the font cannot form
    /// it: decode each shaped glyph's color strike, composite them along the pen
    /// into one strike-sized image, then resample once to the target size. Emoji
    /// clusters almost always shape to a single glyph, so the composite loop usually
    /// runs once. The GPU batcher caches the atlas placement, so it asks only on the
    /// first sight of a cluster.
    ///
    /// The pipeline stays *premultiplied* (as FreeType decodes it) through
    /// compositing and resampling, and un-premultiplies only the final
    /// display-size pixels. Filtering straight alpha would mix the black of
    /// transparent texels into edge colors and fringe the glyph outline dark.
    pub(crate) fn raster(&self, cluster: &str, target: u32) -> Option<ColorGlyph> {
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
        let (width, rows, argb) = copy_bgra(&slot.bitmap);
        Some(StrikeBitmap {
            left: slot.bitmap_left,
            top: slot.bitmap_top,
            width,
            rows,
            argb,
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
///
/// Returns the width and rows it copied along with the pixels, all zero when
/// FreeType gave no buffer, so the size never claims pixels the copy lacks.
fn copy_bgra(bitmap: &FtBitmap) -> (usize, usize, Vec<u32>) {
    let width = bitmap.width as usize;
    let rows = width
        .checked_mul(4)
        .and_then(|row_bytes| bitmap.rows_top_down(row_bytes));
    let Some(rows) = rows else {
        return (0, 0, Vec::new());
    };
    let mut argb = Vec::with_capacity(width * bitmap.rows as usize);
    for row in rows {
        let pixels = row.as_chunks::<4>().0.iter();
        argb.extend(pixels.map(|&[b, g, r, a]| u32::from_be_bytes([a, r, g, b])));
    }
    (width, bitmap.rows as usize, argb)
}

#[cfg(test)]
mod tests {
    use crate::platform::emoji::*;

    /// `composite` indexes a strike's pixels by the size the copy reports, so the two
    /// must agree: no buffer is no pixels and no size. A negative pitch stores the rows
    /// bottom up, `buffer` at the last, and the copy must not read the memory before it.
    #[test]
    fn a_strike_copy_reports_the_size_it_copied() {
        // Two rows of two BGRA pixels, twelve bytes apart.
        let top = [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0];
        let bottom = [9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0];
        let down = [top, bottom].concat();
        // Bottom up, after a row of something else.
        let up = [[0xEE; 12], bottom, top].concat();
        let pixels = vec![0x0403_0201, 0x0807_0605, 0x0C0B_0A09, 0x100F_0E0D];
        assert_eq!(
            copy_bgra(&FtBitmap::over(2, 2, 12, std::ptr::null())),
            (0, 0, Vec::new())
        );
        assert_eq!(
            copy_bgra(&FtBitmap::over(2, 2, 12, down.as_ptr())),
            (2, 2, pixels.clone())
        );
        assert_eq!(
            copy_bgra(&FtBitmap::over(2, 2, -12, up[12..].as_ptr())),
            (2, 2, pixels)
        );
    }

    fn emoji() -> EmojiFont {
        let file = crate::platform::fontconfig::font_for_family(
            "emoji",
            crate::platform::freetype::FontStyle::Regular,
        )
        .expect("the emoji alias resolves");
        EmojiFont::open(&file).expect("a color emoji font should be installed")
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
    fn emoji_font_rasters_a_smiley() {
        let font = emoji();
        let g = font.raster("😀", 32).expect("the smiley should raster");
        assert!(g.width > 0 && g.rows > 0, "the smiley has pixels");
        assert!(
            g.argb.iter().any(|&p| p >> 24 != 0),
            "the smiley has visible ink"
        );
        assert!(
            g.rows <= 40,
            "scaled to the 32px em square, not the strike ({} rows)",
            g.rows
        );
        assert!(g.advance > 0.0);
    }

    #[test]
    fn zwj_and_flag_clusters_ligate_to_one_glyph() {
        let font = emoji();
        let advance = |cluster| font.raster(cluster, 32).map(|g| g.advance);
        let single = advance("👨").expect("the man emoji shapes");
        let family = advance("👨\u{200d}👩\u{200d}👧").expect("the family should ligate");
        assert!(
            family < single * 2.0,
            "a family is one ligature, not three glyphs ({family} vs {single})"
        );
        let flag = advance("🇳🇴").expect("the flag should ligate");
        assert!(
            flag < single * 2.0,
            "a flag is one glyph, not two letter symbols"
        );
    }

    #[test]
    fn cluster_stats_count_reported_hits_and_misses() {
        let font = emoji();
        // raster works on demand (the GPU atlas holds the result); the hit and
        // miss tallies are reported by the batcher, and both roll up through
        // cache_stats.
        font.raster("🎉", 32).expect("party popper");
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
        assert!(font.raster("e\u{301}", 32).is_none());
    }
}
