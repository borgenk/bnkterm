//! Glyph rasterization backed by the system FreeType (libfreetype.so).
//!
//! FreeType turns a TrueType/OpenType outline into an anti-aliased coverage
//! bitmap, handling hinting, scan conversion, and per-size metrics. This is the
//! thin FFI layer over it: a small extern block, one struct owning the C-side
//! handles, a Drop impl, and methods returning owned data. The font file is read
//! into a Vec the Face borrows, rather than mmapped, and coverage is a Vec<u8>.
//!
//! FreeType exposes glyph data as struct fields reached from the Face pointer
//! (face->glyph->bitmap, face->size->metrics), not accessors. The repr(C)
//! structs below mirror those layouts up to the last field this module reads;
//! FreeType allocates the full structs, we only read a prefix, so declaring it
//! with matching primitive types plus repr(C) padding is enough. The touched
//! fields are old and load-bearing for every FreeType consumer, so their
//! offsets have been ABI-stable for years.

use core::ffi::{c_char, c_int, c_long, c_short, c_uint, c_ulong, c_ushort, c_void};
use core::ptr::{self, NonNull};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::platform::emoji::{wants_emoji, ColorGlyph, EmojiFont};
use crate::platform::error::{Error, Result};

/// A font family and the four style files the editor draws emphasis with. A
/// missing bold, italic, or bold-italic variant falls back to the regular face
/// at render time, so a family that ships only some weights still renders (just
/// without the slant or weight). Paths are owned so a config, the built-in
/// default now and a loaded file later, can supply them.
#[derive(Clone, Debug)]
pub struct FontFamily {
    pub regular: String,
    pub bold: String,
    pub italic: String,
    pub bold_italic: String,
}

impl FontFamily {
    /// A family from its four style paths.
    pub fn new(regular: &str, bold: &str, italic: &str, bold_italic: &str) -> Self {
        Self {
            regular: regular.into(),
            bold: bold.into(),
            italic: italic.into(),
            bold_italic: bold_italic.into(),
        }
    }
}

/// Which fonts the editor opens: an ordered list of prose family candidates, an
/// ordered list of code face candidates, and an ordered fallback chain for
/// scalars the chosen family cannot draw. The first prose family whose regular
/// file exists is used; the first installed code face distinct from that regular
/// renders code, else code shares the prose face; the fallback chain is consulted
/// glyph by glyph (see [`Fonts::glyph_face`]) for a character the prose or code
/// face lacks. The concrete selection is the app's to supply: its `Default` impl
/// lives with the app config, keeping this portable layer free of per-project
/// font paths. A loader can later replace it.
#[derive(Clone, Debug)]
pub struct FontConfig {
    pub families: Vec<FontFamily>,
    /// Proportional interface families (first installed wins) for chrome text such
    /// as the tab bar, which is not grid content and reads better in a real UI sans
    /// than in the monospace body family. Only `regular` and `bold` are used. Empty
    /// (or none installed) falls back to the prose family, so UI text always
    /// resolves.
    pub ui: Vec<FontFamily>,
    /// Medium-weight interface face candidates (first installed wins), one file each,
    /// for the tab-bar label: a touch heavier than regular without bold's thickness.
    /// Kept separate from [`ui`](Self::ui) because it is a lone weight file, not a
    /// four-style family. Empty or none installed leaves the label at `ui` regular.
    pub ui_medium: Vec<String>,
    pub code: Vec<String>,
    /// Regular-weight faces consulted, in order, for a scalar the prose or code
    /// face has no glyph for: a symbols/icon font (a terminal's Nerd Font and
    /// Powerline glyphs live in the private-use area, which text families do not
    /// carry) or a wider-coverage Unicode fallback. Only installed files open;
    /// an empty list means no fallback (a missing glyph renders as the `.notdef`
    /// tofu box, the pre-fallback behavior). An app with no need for it (a
    /// text-only editor) leaves this empty.
    pub fallback: Vec<String>,
}

impl FontConfig {
    /// The first prose family whose regular face is installed on this machine.
    fn default_family(&self) -> Result<&FontFamily> {
        self.families
            .iter()
            .find(|f| std::path::Path::new(&f.regular).exists())
            .ok_or_else(|| Error::msg("no usable font family found in the candidate list"))
    }

    /// The first interface family whose regular face is installed, or the prose
    /// `fallback` family when none is (so UI text always resolves to something). The
    /// caller passes the already-resolved prose family as that fallback.
    fn default_ui_family<'a>(&'a self, prose: &'a FontFamily) -> &'a FontFamily {
        self.ui
            .iter()
            .find(|f| std::path::Path::new(&f.regular).exists())
            .unwrap_or(prose)
    }

    /// The first installed medium-weight interface face, or `None` (the label then
    /// stays at the UI regular weight).
    fn default_ui_medium(&self) -> Option<&str> {
        self.ui_medium
            .iter()
            .map(String::as_str)
            .find(|p| std::path::Path::new(p).exists())
    }

    /// The first installed code face whose path differs from the prose family's
    /// regular face, or `None` if none is available. A `None` leaves code sharing
    /// the prose face: still readable, just without the visual distinction.
    fn code_regular_path(&self, prose: &FontFamily) -> Option<&str> {
        self.code
            .iter()
            .map(String::as_str)
            .find(|&p| p != prose.regular.as_str() && std::path::Path::new(p).exists())
    }
}

/// Which variant of the font family a run renders in. Body text and syntax
/// markers use [`FontStyle::Regular`]; emphasis selects a heavier or slanted
/// file. Kept distinct from [`crate::markdown::Emphasis`] so this font layer
/// stays independent of the markdown vocabulary: an inline code span is its own
/// emphasis but still renders in the regular monospace face.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FontStyle {
    Regular,
    /// A mid weight between regular and bold, used by the interface face for chrome
    /// text (the tab bar) so labels read a touch heavier than body text without the
    /// thickness of bold. The prose family carries no medium file and falls back to
    /// its regular face, exactly as it does for a missing italic.
    Medium,
    Bold,
    Italic,
    BoldItalic,
}

/// A self-contained description of which [`Face`] a run draws in, with no borrow
/// of [`Fonts`]. The display list stores this (rather than a `&Face`) so a built
/// frame outlives the borrow and can be diffed across frames; [`Fonts::face_for`]
/// resolves it back to a face at paint time. The two arms mirror the two families
/// [`Fonts`] holds: the prose family at a style, and the distinct code family.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FaceKey {
    Prose {
        size: u32,
        style: FontStyle,
    },
    Code {
        size: u32,
    },
    /// The proportional interface family, for chrome (the tab bar). Only regular
    /// and bold are opened; italic keys resolve to their upright weight.
    Ui {
        size: u32,
        style: FontStyle,
    },
}

impl FaceKey {
    /// The pixel size this face is drawn at, carried by both arms.
    pub fn size(self) -> u32 {
        match self {
            FaceKey::Prose { size, .. } | FaceKey::Code { size } | FaceKey::Ui { size, .. } => size,
        }
    }
}

// ---------------------------------------------------------------------------
// FFI surface.
// ---------------------------------------------------------------------------

type FtError = c_int;
type FtLibrary = *mut c_void;
type FtFace = *mut FtFaceRec;

/// Load the glyph and metrics only, no bitmap (FT_LOAD_DEFAULT).
const FT_LOAD_DEFAULT: i32 = 0x0;
/// Also rasterize to an 8-bit gray coverage bitmap (FT_LOAD_RENDER).
pub(crate) const FT_LOAD_RENDER: i32 = 1 << 2;
/// Ignore embedded bitmap strikes so an outlined glyph renders to 8-bit gray
/// rather than a mono/color strike (FT_LOAD_NO_BITMAP).
const FT_LOAD_NO_BITMAP: i32 = 1 << 3;
/// Load embedded color bitmaps (the CBDT strikes a color emoji font ships)
/// rather than skipping them (FT_LOAD_COLOR).
pub(crate) const FT_LOAD_COLOR: i32 = 1 << 20;
/// One coverage byte per pixel: the only mode whose row width in bytes equals
/// its pixel width (FT_PIXEL_MODE_GRAY).
const FT_PIXEL_MODE_GRAY: u8 = 2;
/// Four bytes per pixel, blue-green-red-alpha, premultiplied: what a CBDT color
/// strike decodes to (FT_PIXEL_MODE_BGRA).
pub(crate) const FT_PIXEL_MODE_BGRA: u8 = 7;

#[link(name = "freetype")]
#[allow(non_snake_case)]
extern "C" {
    fn FT_Init_FreeType(alibrary: *mut FtLibrary) -> FtError;
    fn FT_Done_FreeType(library: FtLibrary) -> FtError;
    fn FT_New_Memory_Face(
        library: FtLibrary,
        file_base: *const u8,
        file_size: c_long,
        face_index: c_long,
        aface: *mut FtFace,
    ) -> FtError;
    fn FT_Done_Face(face: FtFace) -> FtError;
    fn FT_Set_Pixel_Sizes(face: FtFace, pixel_width: c_uint, pixel_height: c_uint) -> FtError;
    fn FT_Select_Size(face: FtFace, strike_index: c_int) -> FtError;
    fn FT_Load_Char(face: FtFace, char_code: c_ulong, load_flags: i32) -> FtError;
    pub(crate) fn FT_Load_Glyph(face: FtFace, glyph_index: c_uint, load_flags: i32) -> FtError;
    fn FT_Get_Char_Index(face: FtFace, charcode: c_ulong) -> c_uint;
}

// ---------------------------------------------------------------------------
// C struct layouts (read-only mirrors of the FreeType headers). Most fields
// exist only to place the ones we read at the right offset.
// ---------------------------------------------------------------------------

#[repr(C)]
#[allow(dead_code)]
struct FtGeneric {
    data: *mut c_void,
    finalizer: *mut c_void,
}

#[repr(C)]
#[allow(dead_code)]
struct FtVector {
    x: c_long,
    y: c_long,
}

#[repr(C)]
#[allow(dead_code)]
pub(crate) struct FtBitmap {
    pub(crate) rows: c_uint,
    pub(crate) width: c_uint,
    pub(crate) pitch: c_int,
    pub(crate) buffer: *const u8,
    num_grays: c_ushort,
    pub(crate) pixel_mode: u8,
    palette_mode: u8,
    palette: *mut c_void,
}

#[repr(C)]
#[allow(dead_code)]
struct FtSizeMetrics {
    x_ppem: c_ushort,
    y_ppem: c_ushort,
    x_scale: c_long,
    y_scale: c_long,
    ascender: c_long,
    descender: c_long,
    height: c_long,
    max_advance: c_long,
}

#[repr(C)]
#[allow(dead_code)]
pub(crate) struct FtGlyphSlotRec {
    library: *mut c_void,
    face: *mut c_void,
    next: *mut c_void,
    glyph_index: c_uint,
    generic: FtGeneric,
    metrics: [c_long; 8],
    linear_hori_advance: c_long,
    linear_vert_advance: c_long,
    advance: FtVector,
    format: c_int,
    pub(crate) bitmap: FtBitmap,
    pub(crate) bitmap_left: c_int,
    pub(crate) bitmap_top: c_int,
}

#[repr(C)]
#[allow(dead_code)]
struct FtSizeRec {
    face: *mut c_void,
    generic: FtGeneric,
    metrics: FtSizeMetrics,
}

#[repr(C)]
#[allow(dead_code)]
pub(crate) struct FtFaceRec {
    num_faces: c_long,
    face_index: c_long,
    face_flags: c_long,
    style_flags: c_long,
    num_glyphs: c_long,
    family_name: *mut c_char,
    style_name: *mut c_char,
    num_fixed_sizes: c_int,
    available_sizes: *mut c_void,
    num_charmaps: c_int,
    charmaps: *mut c_void,
    generic: FtGeneric,
    bbox: [c_long; 4],
    units_per_em: c_ushort,
    ascender: c_short,
    descender: c_short,
    height: c_short,
    max_advance_width: c_short,
    max_advance_height: c_short,
    underline_position: c_short,
    underline_thickness: c_short,
    pub(crate) glyph: *mut FtGlyphSlotRec,
    size: *mut FtSizeRec,
}

// ---------------------------------------------------------------------------
// Owning wrapper.
// ---------------------------------------------------------------------------

/// A rasterized glyph, owned so no reference into FreeType memory escapes.
/// `coverage` is tightly packed top-down, `width` bytes per row, `rows` rows.
pub struct Glyph {
    /// Horizontal offset from the pen to the bitmap's left edge.
    pub left: i32,
    /// Vertical offset from the baseline up to the bitmap's top edge.
    pub top: i32,
    pub width: usize,
    pub rows: usize,
    /// Pen advance in pixels.
    pub advance: f32,
    pub coverage: Vec<u8>,
}

impl Glyph {
    fn empty() -> Self {
        Self {
            left: 0,
            top: 0,
            width: 0,
            rows: 0,
            advance: 0.0,
            coverage: Vec::new(),
        }
    }
}

/// Render-ready line metrics for a face at its current pixel size, rounded to
/// whole pixels. Ascent and descent are both positive glyph metrics (pixels
/// above and below the baseline); `line_height` is the baseline-to-baseline
/// distance and may exceed ascent + descent.
///
/// # Where the baseline sits in the line box
///
/// `baseline` is the one number a renderer needs: pixels from the *top of the
/// line box* down to the baseline. It is not the ascent, and reaching for the
/// ascent instead is the classic way to misalign text.
///
/// The surplus (`line_height - ascent - descent`) is the font's line gap, and
/// where it goes decides where the ink sits in the cell:
///
/// ```text
///   line box              ascent-anchored (wrong)      baseline = h - descent
///   ┌───────────────┐     ┌───────────────┐            ┌───────────────┐
///   │               │     │ ┬─┬ ┬  ┬      │ ← ink      │               │ ← gap
///   │               │     │  │  └┬─┘      │            │ ┬─┬ ┬  ┬      │
///   │               │     │  │   ┌┘  base │            │  │  └┬─┘      │
///   │               │     ├──┴───┴────────┤ ← line     │  │   ┌┘  base │
///   │               │     │               │   gap      ├──┴───┴────────┤
///   └───────────────┘     └───────────────┘   dumped   └───────────────┘
///                                             below
/// ```
///
/// A font's line gap is *external leading*: space between one line's descent and
/// the next line's ascent. In a terminal, adjacent rows share an edge, so all of
/// it belongs above the ascent, seating the descent on the cell bottom. Do the
/// naive thing (baseline = ascent) and the gap piles up under the text instead:
/// a `T` hugs the cell top while a `y` floats well clear of the bottom, and every
/// accented capital pokes out of the top of its cell. Consolas asks for a 17%-of-em
/// gap, so this is a visible 2-3px, not a rounding curiosity. wezterm and Alacritty
/// seat the baseline the same way.
///
/// Leading a *caller* adds on top of the font's own line height (see
/// [`Fonts::with_config`]) is a different thing: that one is split evenly, so text
/// stays centered in the taller line box rather than sinking to its floor.
#[derive(Clone, Copy, Debug)]
pub struct Metrics {
    pub ascent: i32,
    pub descent: i32,
    pub line_height: i32,
    /// Pixels from the top of the line box down to the baseline.
    pub baseline: i32,
}

impl Metrics {
    /// The same metrics in a line box grown to at least `target` pixels, the extra
    /// split evenly above and below so the text stays centered in it. A `target`
    /// the line box already meets changes nothing.
    fn grown_to(self, target: i32) -> Self {
        let extra = target - self.line_height;
        if extra <= 0 {
            return self;
        }
        Metrics {
            line_height: target,
            baseline: self.baseline + extra / 2,
            ..self
        }
    }
}

/// Owns the FreeType library, the face, and the font-file bytes the face
/// borrows. FT_New_Memory_Face keeps a pointer into the bytes for the face's
/// life; the Vec's heap buffer has a stable address across moves of this
/// struct, so holding both together is sound as long as the bytes are never
/// mutated or reallocated (they are not).
pub struct Face {
    library: NonNull<c_void>,
    face: NonNull<FtFaceRec>,
    _data: Vec<u8>,
    /// Pen advances cached by character. Layout measures every character to wrap
    /// and size runs but the GPU atlas rasterizes only the ones on screen, and the
    /// advance is constant for a fixed-size face, so caching it turns a
    /// per-character `FT_Load_Char` (hinting plus metrics) into a map hit. Without
    /// this a full re-layout reloads every glyph's metrics through FreeType, which
    /// dominates the rebuild a selection drag triggers each frame. Interior
    /// mutability so measuring through a shared `&Face` can fill it.
    advances: RefCell<HashMap<char, f32>>,
    /// Glyph-cache hit and miss tallies, reported by the GPU batcher (which owns
    /// the atlas-placement cache) via [`Self::record_glyph_hit`] and
    /// [`Self::record_glyph_miss`]. Pure instrumentation (see [`CacheStats`]): a
    /// near-zero steady-state miss rate is the evidence that the cache is doing
    /// its job rather than thrashing.
    hits: Cell<u64>,
    misses: Cell<u64>,
    /// The pixel size this face was opened at, recorded by `set_pixel_size`.
    /// The emoji face scales its color glyphs to this so an emoji sits on the
    /// same em square as the surrounding text.
    pixel_size: Cell<u32>,
    /// The shared color emoji font, if one is installed; [`Fonts::new`] attaches
    /// it to every face it opens. `None` on the emoji face itself and in
    /// face-only tests, in which case emoji clusters take the per-character
    /// path (tofu, exactly the pre-emoji behavior).
    emoji: Option<Rc<EmojiFont>>,
}

/// A snapshot of a glyph cache's hit and miss tallies. Summing these across every
/// face turns "scrolling feels smooth" into a number: once the working set is
/// packed into the GPU atlas, steady-state frames add hits and almost no misses.
#[derive(Clone, Copy, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

impl CacheStats {
    fn merge(self, other: Self) -> Self {
        Self {
            hits: self.hits + other.hits,
            misses: self.misses + other.misses,
        }
    }

    /// Total lookups (hits + misses).
    pub fn lookups(self) -> u64 {
        self.hits + self.misses
    }

    /// Fraction of lookups served from the cache, 0.0 when nothing was looked up.
    pub fn hit_rate(self) -> f32 {
        match self.lookups() {
            0 => 0.0,
            total => self.hits as f32 / total as f32,
        }
    }
}

impl Face {
    /// Open the regular face of the first available font family at `pixel_size`.
    /// A test-only convenience: production goes through [`Fonts`], which opens the
    /// family's variants per size itself.
    #[cfg(test)]
    pub fn open_default(pixel_size: u32) -> Result<Self> {
        let config = FontConfig::default();
        let face = Self::from_path(&config.default_family()?.regular)?;
        face.set_pixel_size(pixel_size)?;
        Ok(face)
    }

    /// Build a face from a font file on disk.
    pub fn from_path(path: &str) -> Result<Self> {
        let data = std::fs::read(path).map_err(|e| Error::msg(format!("read font {path}: {e}")))?;

        let mut library: FtLibrary = ptr::null_mut();
        // SAFETY: alibrary points at a live local; FT writes the handle through
        // it and returns nonzero on failure.
        let err = unsafe { FT_Init_FreeType(&mut library) };
        let library = NonNull::new(library)
            .filter(|_| err == 0)
            .ok_or_else(|| Error::msg("FT_Init_FreeType failed"))?;

        let mut face: FtFace = ptr::null_mut();
        // SAFETY: library is valid; data outlives the face (it is moved into the
        // returned struct and dropped only after FT_Done_Face); aface is a live
        // local. FreeType keeps a pointer into data.
        let err = unsafe {
            FT_New_Memory_Face(
                library.as_ptr(),
                data.as_ptr(),
                data.len() as c_long,
                0,
                &mut face,
            )
        };
        let face = match NonNull::new(face).filter(|_| err == 0) {
            Some(face) => face,
            None => {
                // SAFETY: library came from FT_Init_FreeType and is freed once.
                unsafe { FT_Done_FreeType(library.as_ptr()) };
                return Err(Error::msg(format!("FT_New_Memory_Face failed for {path}")));
            }
        };

        Ok(Self {
            library,
            face,
            _data: data,
            advances: RefCell::new(HashMap::new()),
            hits: Cell::new(0),
            misses: Cell::new(0),
            pixel_size: Cell::new(0),
            emoji: None,
        })
    }

    /// Select the pixel size FreeType renders and reports metrics at; width 0
    /// matches it to the height. Construction-only (private): the glyph and
    /// advance caches key on `char` alone, so changing a face's size after it has
    /// cached anything would serve stale-size glyphs. Every face is sized once,
    /// at open time, and never resized.
    fn set_pixel_size(&self, size: u32) -> Result<()> {
        self.pixel_size.set(size);
        // SAFETY: face is a valid FT_Face; the call only reads size arguments.
        let err = unsafe { FT_Set_Pixel_Sizes(self.face.as_ptr(), 0, size) };
        if err != 0 {
            return Err(Error::msg("FT_Set_Pixel_Sizes failed"));
        }
        Ok(())
    }

    /// Record that the GPU batcher served a scalar glyph from its atlas cache
    /// (the glyph was already packed). Pure instrumentation.
    pub fn record_glyph_hit(&self) {
        self.hits.set(self.hits.get() + 1);
    }

    /// Record that the GPU batcher had to rasterize and pack a scalar glyph (its
    /// first sight of the character). Pure instrumentation.
    pub fn record_glyph_miss(&self) {
        self.misses.set(self.misses.get() + 1);
    }

    /// Forward a color-cluster cache hit to the shared emoji font, so its tally
    /// rolls up through [`Fonts::cache_stats`] alongside the scalar counts. A
    /// no-op when no emoji font is installed.
    pub fn record_cluster_hit(&self) {
        if let Some(emoji) = &self.emoji {
            emoji.record_hit();
        }
    }

    /// Forward a color-cluster cache miss (a first-time raster) to the shared
    /// emoji font. A no-op when no emoji font is installed.
    pub fn record_cluster_miss(&self) {
        if let Some(emoji) = &self.emoji {
            emoji.record_miss();
        }
    }

    /// This face's cumulative glyph-cache hit and miss counts.
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.get(),
            misses: self.misses.get(),
        }
    }

    /// Rasterize a character at the current pixel size, copying the coverage
    /// out. A load failure or glyphless character yields an empty glyph. The
    /// render path goes through [`Self::with_glyph`], which caches the result;
    /// this is the uncached primitive behind it.
    pub fn rasterize(&self, ch: char) -> Glyph {
        // SAFETY: face is valid; FT_Load_Char renders into the face's shared
        // glyph slot. No reference into that slot is held across this call.
        let err = unsafe {
            FT_Load_Char(
                self.face.as_ptr(),
                ch as u32 as c_ulong,
                FT_LOAD_RENDER | FT_LOAD_NO_BITMAP,
            )
        };
        if err != 0 {
            return Glyph::empty();
        }
        // SAFETY: the load succeeded, so face->glyph points at the populated
        // slot; the borrow ends before any later FreeType call.
        let slot = unsafe { (*self.face.as_ptr()).glyph };
        let Some(slot) = NonNull::new(slot) else {
            return Glyph::empty();
        };
        let slot = unsafe { slot.as_ref() };
        let advance = slot.advance.x as f32 / 64.0;
        // The blit treats coverage as one byte per pixel, true only for the gray
        // render mode. Skip non-gray ink but keep the advance so layout holds.
        if slot.bitmap.pixel_mode != FT_PIXEL_MODE_GRAY {
            return Glyph {
                advance,
                ..Glyph::empty()
            };
        }
        Glyph {
            left: slot.bitmap_left,
            top: slot.bitmap_top,
            width: slot.bitmap.width as usize,
            rows: slot.bitmap.rows as usize,
            advance,
            coverage: copy_coverage(&slot.bitmap),
        }
    }

    /// Pen advance of a character at the current pixel size, in pixels, cached on
    /// the face. The first measurement loads the glyph's metrics through FreeType;
    /// every later one is a map hit. Zero on a load failure.
    pub fn advance(&self, ch: char) -> f32 {
        // `measure_advance` reads only FreeType state, never this cache, so filling
        // a miss while the cache is borrowed cannot reenter this borrow.
        *self
            .advances
            .borrow_mut()
            .entry(ch)
            .or_insert_with(|| self.measure_advance(ch))
    }

    /// Load a character's pen advance from FreeType, metrics only (no
    /// rasterization), so measuring a run never pays for scan conversion. The
    /// uncached primitive behind [`Self::advance`]. Zero on a load failure.
    fn measure_advance(&self, ch: char) -> f32 {
        // SAFETY: face is valid; FT_LOAD_DEFAULT loads metrics into the glyph
        // slot without rendering. The advance is read out immediately.
        let err =
            unsafe { FT_Load_Char(self.face.as_ptr(), ch as u32 as c_ulong, FT_LOAD_DEFAULT) };
        if err != 0 {
            return 0.0;
        }
        // SAFETY: the load succeeded, so face->glyph points at the slot.
        let slot = unsafe { (*self.face.as_ptr()).glyph };
        let Some(slot) = NonNull::new(slot) else {
            return 0.0;
        };
        // SAFETY: slot is valid after a successful load.
        unsafe { slot.as_ref().advance.x as f32 / 64.0 }
    }

    /// Pen advance of one grapheme cluster, in pixels: the emoji face's glyph
    /// advance when the cluster routes to color emoji, else the sum of the
    /// per-character advances (identical to measuring the characters one by
    /// one, so ASCII fast paths and this stay in exact agreement). This is the
    /// measuring twin of [`Self::with_cluster_glyph`]; the two must route the
    /// same way or the caret drifts from the pixels.
    pub fn cluster_advance(&self, cluster: &str) -> f32 {
        if let Some((emoji, px)) = self.emoji_cluster(cluster) {
            if let Some(advance) = emoji.advance(cluster, px) {
                return advance;
            }
        }
        cluster.chars().map(|c| self.advance(c)).sum()
    }

    /// Run `f` over the color glyph for an emoji cluster rasterized at `target`
    /// pixels, or return `None` when the cluster renders through the ordinary
    /// per-character path (not an emoji, no emoji font installed, or the emoji
    /// font cannot form it). `None` tells the caller to draw the cluster's
    /// characters instead, matching what [`Self::cluster_advance`] measured.
    ///
    /// The caller passes `target` because the fit differs by path: a wide
    /// cluster (two grid cells) is drawn at the em (its natural ~square size
    /// filling both cells), but a single-width emoji is scaled down to one cell
    /// so it does not spill into the next column.
    pub fn with_cluster_glyph<R>(
        &self,
        cluster: &str,
        target: u32,
        f: impl FnOnce(&ColorGlyph) -> R,
    ) -> Option<R> {
        self.emoji_font(cluster)?.with_glyph(cluster, target, f)
    }

    /// The emoji font for `cluster`, or `None` when the cluster should take the
    /// per-character path. The single routing decision the measure and draw
    /// paths share, so they never disagree on what is an emoji.
    fn emoji_font(&self, cluster: &str) -> Option<&Rc<EmojiFont>> {
        let emoji = self.emoji.as_ref()?;
        wants_emoji(cluster, |c| self.has_scalar(c)).then_some(emoji)
    }

    /// The emoji font and its natural (em) target size for `cluster`. The
    /// measure path uses this so a run's width matches an emoji drawn at the em.
    fn emoji_cluster(&self, cluster: &str) -> Option<(&Rc<EmojiFont>, u32)> {
        Some((self.emoji_font(cluster)?, self.pixel_size.get()))
    }

    /// Whether this face's character map has a real glyph for `ch` (a missing
    /// character maps to glyph 0, `.notdef`, the tofu box).
    fn has_scalar(&self, ch: char) -> bool {
        // SAFETY: face is valid; the call is a read-only cmap lookup.
        unsafe { FT_Get_Char_Index(self.face.as_ptr(), ch as u32 as c_ulong) != 0 }
    }

    /// The raw FreeType face handle, for the emoji module to shape and decode
    /// color strikes through the same face (it reads the glyph slot after a
    /// load). Confined to `pub(crate)`; the pointer is valid for the face's life.
    pub(crate) fn ft_face_ptr(&self) -> *mut FtFaceRec {
        self.face.as_ptr()
    }

    /// Select the face's first fixed bitmap strike and return its nominal pixel
    /// size, or `None` when the face has none (an outline font). Only the emoji
    /// font uses strikes; outline faces size with [`Self::set_pixel_size`].
    pub(crate) fn select_first_strike(&self) -> Option<f32> {
        // SAFETY: face is valid; num_fixed_sizes is a plain field read.
        let strikes = unsafe { (*self.face.as_ptr()).num_fixed_sizes };
        if strikes <= 0 {
            return None;
        }
        // SAFETY: strike 0 exists (checked above); the call selects it.
        let err = unsafe { FT_Select_Size(self.face.as_ptr(), 0) };
        if err != 0 {
            return None;
        }
        // SAFETY: a successful select populates face->size.
        let size = NonNull::new(unsafe { (*self.face.as_ptr()).size })?;
        let ppem = unsafe { size.as_ref().metrics.y_ppem };
        (ppem > 0).then(|| f32::from(ppem))
    }

    /// Ascent and descent at the current pixel size, in pixels. Descent is
    /// negative, matching FreeType.
    pub fn line_metrics(&self) -> (f32, f32) {
        // SAFETY: face is valid; face->size is non-null once a size is set,
        // which open_default always does. The metrics are copied out.
        let size = unsafe { (*self.face.as_ptr()).size };
        let Some(size) = NonNull::new(size) else {
            return (0.0, 0.0);
        };
        let metrics = unsafe { &size.as_ref().metrics };
        (
            metrics.ascender as f32 / 64.0,
            metrics.descender as f32 / 64.0,
        )
    }

    /// The recommended baseline-to-baseline distance at the current pixel size,
    /// in pixels: FreeType's line height, which includes any line gap the font
    /// asks for. Zero if no size is set.
    pub fn line_height(&self) -> f32 {
        // SAFETY: face is valid; face->size is non-null once a size is set.
        let size = unsafe { (*self.face.as_ptr()).size };
        let Some(size) = NonNull::new(size) else {
            return 0.0;
        };
        unsafe { size.as_ref().metrics.height as f32 / 64.0 }
    }

    /// Render-ready [`Metrics`] at the current pixel size: ascent and descent
    /// rounded up to whole pixels, and the baseline-to-baseline line height,
    /// held to at least the ink height so stacked lines never overlap.
    ///
    /// FreeType rounds the ascender up and the descender down independently of
    /// the height it reports, so a font can hand back a baseline-to-baseline
    /// height a pixel below the rounded ascent + descent (Noto Sans does at some
    /// sizes). Taking the max keeps the line box honest for any font.
    ///
    /// The baseline seats the descent on the line box's bottom edge, which puts
    /// the font's line gap above the ascent where it belongs; see [`Metrics`].
    pub fn metrics(&self) -> Metrics {
        let (ascent, descent) = self.line_metrics();
        let ascent = ascent.ceil() as i32;
        let descent = (-descent).ceil() as i32;
        let line_height = (self.line_height().ceil() as i32).max(ascent + descent);
        Metrics {
            ascent,
            descent,
            line_height,
            baseline: line_height - descent,
        }
    }
}

impl Drop for Face {
    fn drop(&mut self) {
        // The face must be freed before the library it came from; the data Vec
        // is freed afterwards by its own Drop. SAFETY: each handle came from its
        // matching FreeType new call and is freed exactly once.
        unsafe {
            FT_Done_Face(self.face.as_ptr());
            FT_Done_FreeType(self.library.as_ptr());
        }
    }
}

/// The faces for one pixel size: the regular face (always present) plus any
/// bold/italic/bold-italic variants the family ships, the distinct code face (if
/// a separate code family is installed), and the cached line metrics. Metrics
/// come from the regular face only, so emphasis or code selects a different glyph
/// set without ever changing line height.
struct SizedFaces {
    size: u32,
    regular: Face,
    bold: Option<Face>,
    italic: Option<Face>,
    bold_italic: Option<Face>,
    /// The code family's regular face at this size, or `None` when no distinct
    /// code family is installed (code then falls back to `regular`).
    code: Option<Face>,
    /// The interface family's proportional faces at this size (for chrome text; see
    /// [`FaceKey::Ui`]). `ui_regular` is always present (it falls back to the prose
    /// family when no UI family is installed); `ui_medium` and `ui_bold` are `None`
    /// when no such weight file is installed, and then that weight uses `ui_regular`.
    /// `ui_metrics` are the interface face's own vertical metrics, so a UI baseline
    /// centers on the sans ink box rather than the monospace one.
    ui_regular: Face,
    ui_medium: Option<Face>,
    ui_bold: Option<Face>,
    ui_metrics: Metrics,
    /// The fallback chain at this size ([`FontConfig::fallback`]), in priority
    /// order, holding only the faces whose files exist. Consulted by
    /// [`Fonts::glyph_face`] for a scalar the keyed face cannot draw; empty when
    /// the config lists none.
    fallback: Vec<Face>,
    metrics: Metrics,
}

impl SizedFaces {
    /// Every face this size actually opened: the regular face plus whichever
    /// variants, code face, and fallback faces exist. Used to roll up cache stats
    /// across faces.
    fn faces(&self) -> impl Iterator<Item = &Face> {
        std::iter::once(&self.regular)
            .chain(self.bold.as_ref())
            .chain(self.italic.as_ref())
            .chain(self.bold_italic.as_ref())
            .chain(self.code.as_ref())
            .chain(std::iter::once(&self.ui_regular))
            .chain(self.ui_medium.as_ref())
            .chain(self.ui_bold.as_ref())
            .chain(self.fallback.iter())
    }
}

/// A set of [`Face`]s at different pixel sizes and styles, so a document can mix
/// body text, larger headings, and inline emphasis. Each distinct size opens the
/// family's regular face plus whatever variant files exist (the same files,
/// re-read per size); the first size opened is the fallback for any size that was
/// not requested, and a missing variant falls back to the regular face, so a
/// lookup can never fail and never panics.
pub struct Fonts {
    sized: Vec<SizedFaces>,
    /// The color emoji font, opened once and shared by every face, or `None`
    /// when no emoji font is installed (emoji then render as tofu, as before).
    emoji: Option<Rc<EmojiFont>>,
}

impl Fonts {
    /// Open the default font selection at each of `sizes`, with the font's own
    /// line height (a scale of 1.0, no extra leading). A convenience over
    /// [`Self::with_config`] using [`FontConfig::default`]; tests and the perf
    /// harness use it, while the editor threads its own config through
    /// [`Self::with_config`].
    pub fn new(sizes: &[u32]) -> Result<Self> {
        Self::with_config(&FontConfig::default(), sizes, 1.0)
    }

    /// Open `config`'s prose family and code face at each of `sizes` (zeros and
    /// duplicates skipped), loading every available style per size. Each size's
    /// line box is grown to at least `line_height_scale` times the size (the CSS
    /// line-height model), never below the font's own line height, so lines can
    /// be spaced out without overlapping. At least one nonzero size is required;
    /// the first opened becomes the fallback.
    pub fn with_config(config: &FontConfig, sizes: &[u32], line_height_scale: f32) -> Result<Self> {
        let family = config.default_family()?;
        let code_path = config.code_regular_path(family);
        let ui_family = config.default_ui_family(family);
        let ui_medium_path = config.default_ui_medium();
        let emoji = EmojiFont::open().map(Rc::new);
        let mut sized: Vec<SizedFaces> = Vec::new();
        for &size in sizes {
            if size == 0 || sized.iter().any(|s| s.size == size) {
                continue;
            }
            let regular = open_face(&family.regular, size, emoji.as_ref())?;
            // Grow the line box to the requested multiple of the size, keeping
            // the font's own line height as the floor so lines never overlap.
            // Ascent and descent stay the glyph metrics; the extra is leading the
            // caller asked for, so it splits evenly and the baseline moves down
            // half of it (see `Metrics`).
            let target = (line_height_scale * size as f32).round() as i32;
            let metrics = regular.metrics().grown_to(target);
            // Fallback faces draw only scalars the primary lacks and never route
            // emoji clusters themselves, so they carry no emoji font.
            //
            // They open at the primary `size` and their glyphs rasterize at the
            // font's own natural proportions: we deliberately do NOT scale a
            // fallback face down to the prose cell. Nerd Font's non-"Mono" cut draws
            // icons larger than one cell, so an icon glyph (a git-branch mark, a
            // Powerline separator) comes out big and crisp, as terminals that map
            // the same cut do. The cost of natural size is that a wide icon can
            // spill into the next cell; a trailing space beside it absorbs the
            // overflow. (The "Mono" cut, when installed, is pre-fit to one cell and
            // leads the chain in config.rs, so this only bites when it is absent.)
            //
            // ASSUMPTION, not verified: this is thought to be why some other
            // terminals render the same glyph smaller and softer — they are believed
            // to normalize a fallback face's metrics to the primary's, shrinking (and
            // thus re-rasterizing smaller) a natively larger icon. That is an
            // inference from observed output, not confirmed against their source;
            // treat it as a lead if this policy is ever revisited.
            let fallback = config
                .fallback
                .iter()
                .filter_map(|p| open_variant(p, size, None))
                .collect();
            // The interface face draws proportional chrome text, so it carries no
            // emoji font (chrome is text) and keeps its own vertical metrics. When
            // no distinct UI family is installed, `ui_family` is the prose family, so
            // `ui_regular` simply mirrors `regular`.
            let ui_regular = open_face(&ui_family.regular, size, None)?;
            let ui_metrics = ui_regular.metrics();
            sized.push(SizedFaces {
                size,
                regular,
                bold: open_variant(&family.bold, size, emoji.as_ref()),
                italic: open_variant(&family.italic, size, emoji.as_ref()),
                bold_italic: open_variant(&family.bold_italic, size, emoji.as_ref()),
                code: code_path.and_then(|p| open_variant(p, size, emoji.as_ref())),
                ui_regular,
                ui_medium: ui_medium_path.and_then(|p| open_variant(p, size, None)),
                ui_bold: open_variant(&ui_family.bold, size, None),
                ui_metrics,
                fallback,
                metrics,
            });
        }
        if sized.is_empty() {
            return Err(Error::msg("Fonts::new needs at least one nonzero size"));
        }
        Ok(Self { sized, emoji })
    }

    /// The entry for `size`, or the fallback (first opened) if `size` is unknown.
    fn entry(&self, size: u32) -> &SizedFaces {
        // `sized` is never empty (Fonts::new errors otherwise), so the `[0]`
        // fallback for an unknown size cannot panic.
        debug_assert!(!self.sized.is_empty());
        self.sized
            .iter()
            .find(|s| s.size == size)
            .unwrap_or(&self.sized[0])
    }

    /// The face for `size` in `style`, falling back to the regular face when the
    /// family lacks that variant and to the first opened size when `size` was not
    /// opened. Never fails, never panics.
    pub fn face(&self, size: u32, style: FontStyle) -> &Face {
        let entry = self.entry(size);
        match style {
            // The prose family ships no medium weight, so it reads as regular.
            FontStyle::Regular | FontStyle::Medium => &entry.regular,
            FontStyle::Bold => entry.bold.as_ref().unwrap_or(&entry.regular),
            FontStyle::Italic => entry.italic.as_ref().unwrap_or(&entry.regular),
            FontStyle::BoldItalic => entry.bold_italic.as_ref().unwrap_or(&entry.regular),
        }
    }

    /// The code face for `size`: the distinct code family's regular face, or the
    /// prose regular face when no code family is installed (or `size` was not
    /// opened). Code never renders bold or italic, so there is only the one face.
    /// Never fails, never panics.
    pub fn code_face(&self, size: u32) -> &Face {
        let entry = self.entry(size);
        entry.code.as_ref().unwrap_or(&entry.regular)
    }

    /// The interface face for `size` and weight: the proportional UI family's
    /// medium or bold when asked and available, else its regular (italic styles
    /// resolve to their upright weight, since UI text never slants). A missing
    /// medium or bold falls back to `ui_regular`, as does an unopened `size`. Never
    /// panics.
    pub fn ui_face(&self, size: u32, style: FontStyle) -> &Face {
        let entry = self.entry(size);
        match style {
            FontStyle::Medium => entry.ui_medium.as_ref().unwrap_or(&entry.ui_regular),
            FontStyle::Bold | FontStyle::BoldItalic => {
                entry.ui_bold.as_ref().unwrap_or(&entry.ui_regular)
            }
            FontStyle::Regular | FontStyle::Italic => &entry.ui_regular,
        }
    }

    /// The interface family's vertical metrics for `size` (its own ascent/descent),
    /// so a UI baseline centers on the sans ink box, not the monospace one.
    pub fn ui_metrics(&self, size: u32) -> Metrics {
        self.entry(size).ui_metrics
    }

    /// Resolve a [`FaceKey`] to its face, the inverse of the key a run records.
    /// Routes to [`Self::face`] or [`Self::code_face`], so it inherits their
    /// fallbacks and never panics.
    pub fn face_for(&self, key: FaceKey) -> &Face {
        match key {
            FaceKey::Prose { size, style } => self.face(size, style),
            FaceKey::Code { size } => self.code_face(size),
            FaceKey::Ui { size, style } => self.ui_face(size, style),
        }
    }

    /// The physical face to rasterize `ch` in for `key`: the keyed face when its
    /// character map covers `ch`, otherwise the first fallback face at that size
    /// that does, otherwise the keyed face again (so a scalar absent everywhere
    /// still draws its `.notdef` box, the pre-fallback behavior). This is how a
    /// Nerd Font / Powerline icon a prompt emits reaches its glyph in a symbols
    /// font even though the prose or code family has no cell for it.
    ///
    /// Emoji never arrive here: a color cluster is resolved through the emoji
    /// cluster path before any scalar is drawn. The keyed face's own cmap is
    /// consulted first and answers every ASCII and Latin glyph without touching a
    /// fallback face, so the common case pays a single [`FT_Get_Char_Index`], and
    /// only a genuine miss walks the (short) chain. The GPU batcher caches the
    /// resolved raster by `(key, ch)`, so this runs once per new glyph, never per
    /// frame. With an empty [`FontConfig::fallback`] the chain is empty and this
    /// always returns the keyed face.
    pub fn glyph_face(&self, key: FaceKey, ch: char) -> &Face {
        let primary = self.face_for(key);
        if primary.has_scalar(ch) {
            return primary;
        }
        self.entry(key.size())
            .fallback
            .iter()
            .find(|f| f.has_scalar(ch))
            .unwrap_or(primary)
    }

    /// The metrics for `size`, or the fallback's metrics if `size` was not opened.
    pub fn metrics(&self, size: u32) -> Metrics {
        self.entry(size).metrics
    }

    /// Glyph-cache hits and misses summed over every open face (the emoji
    /// font's cluster cache included), for frame-loop instrumentation.
    pub fn cache_stats(&self) -> CacheStats {
        let emoji = self
            .emoji
            .as_ref()
            .map(|e| e.cache_stats())
            .unwrap_or_default();
        self.sized
            .iter()
            .flat_map(SizedFaces::faces)
            .map(Face::cache_stats)
            .fold(emoji, CacheStats::merge)
    }
}

/// Open `path` at `size` with the shared `emoji` font attached: the three-step
/// face open (read the file, fix the pixel size, share the emoji font) that
/// both the required regular face and the optional variants go through.
fn open_face(path: &str, size: u32, emoji: Option<&Rc<EmojiFont>>) -> Result<Face> {
    let mut face = Face::from_path(path)?;
    face.set_pixel_size(size)?;
    face.emoji = emoji.cloned();
    Ok(face)
}

/// Open a style variant at `size`, or `None` if the file is absent or fails to
/// load, leaving the caller to fall back to the regular face. Keeps a family
/// that ships only some weights from being a hard error.
fn open_variant(path: &str, size: u32, emoji: Option<&Rc<EmojiFont>>) -> Option<Face> {
    if !std::path::Path::new(path).exists() {
        return None;
    }
    open_face(path, size, emoji).ok()
}

/// Copy a FreeType bitmap into a tightly packed top-down coverage buffer of
/// `width * rows` bytes, dropping inter-row pitch padding. Row y starts at
/// `buffer + y*pitch` for either pitch sign. Assumes one coverage byte per
/// pixel, so the caller must confirm FT_PIXEL_MODE_GRAY first.
fn copy_coverage(bitmap: &FtBitmap) -> Vec<u8> {
    let width = bitmap.width as usize;
    let rows = bitmap.rows as usize;
    let total = rows * width;
    if width == 0 || rows == 0 || bitmap.buffer.is_null() {
        return Vec::new();
    }
    let mut out = vec![0u8; total];
    let pitch = bitmap.pitch as isize;
    for y in 0..rows {
        // SAFETY: FreeType guarantees |pitch| >= width bytes per row, so the
        // width-byte read at buffer + y*pitch stays within that row; out is a
        // distinct buffer.
        unsafe {
            let src = bitmap.buffer.offset(y as isize * pitch);
            ptr::copy_nonoverlapping(src, out[y * width..].as_mut_ptr(), width);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open() -> Face {
        Face::open_default(16).expect("a default font should be available")
    }

    #[test]
    fn the_baseline_seats_the_descent_on_the_line_box_floor() {
        for size in [12u32, 13, 16, 20, 26] {
            let face = Face::open_default(size).expect("a default font");
            let m = face.metrics();
            assert_eq!(
                m.baseline,
                m.line_height - m.descent,
                "the font's line gap belongs above the ascent, not under the text"
            );
            assert!(
                m.baseline >= m.ascent,
                "size {size}: the baseline never sits above the ascent"
            );
        }
    }

    #[test]
    fn ascii_ink_lands_inside_its_line_box_at_every_size() {
        // What the rule buys, and the regression that would take it away. Seated on
        // the baseline, the printable ASCII a terminal actually draws fits its line
        // box top and bottom, at every size, on all four candidate monospace faces.
        // Seat it on the ascent instead (the old rule) and the tall thin glyphs go
        // negative: Consolas' `|` reaches 13px over the baseline at 16px against an
        // ascent of 12, so it bleeds a pixel into the row above.
        //
        // Ink *balance* is deliberately not asserted: how much air a face leaves over
        // its capitals is the face's business (Noto Sans Mono reserves 10px more than
        // Consolas does), and pinning it here would test the font, not the rule.
        for size in 10u32..=40 {
            let face = Face::open_default(size).expect("a default font");
            let m = face.metrics();
            for ch in '!'..='~' {
                let g = face.rasterize(ch);
                if g.rows == 0 {
                    continue;
                }
                let (top, bottom) = (m.baseline - g.top, m.baseline - g.top + g.rows as i32);
                assert!(
                    top >= 0 && bottom <= m.line_height,
                    "size {size}: '{ch}' ink spans {top}..{bottom}, outside the 0..{} line box",
                    m.line_height
                );
            }
        }
    }

    #[test]
    fn caller_leading_splits_evenly_around_the_text() {
        let face = Face::open_default(16).expect("a default font");
        let m = face.metrics();
        let taller = m.grown_to(m.line_height + 4);
        assert_eq!(taller.line_height, m.line_height + 4);
        assert_eq!(
            taller.baseline,
            m.baseline + 2,
            "leading a caller asks for is split above and below, not stacked on one side"
        );
        assert_eq!(
            m.grown_to(m.line_height - 1).line_height,
            m.line_height,
            "the font's own line height is the floor"
        );
    }

    #[test]
    fn rasterizes_a_visible_glyph() {
        let face = open();
        let g = face.rasterize('A');
        assert!(g.rows > 0 && g.width > 0, "A should have ink");
        assert_eq!(g.coverage.len(), g.rows * g.width);
        assert!(
            g.coverage.iter().any(|&c| c > 0),
            "A should have inked pixels"
        );
        assert!(g.advance > 0.0);
    }

    #[test]
    fn rasterize_is_deterministic() {
        let face = open();
        // A fixed-size face renders a glyph identically every time, which is what
        // lets the GPU atlas rasterize each glyph exactly once and reuse it.
        let a = face.rasterize('g');
        let b = face.rasterize('g');
        assert_eq!(a.left, b.left);
        assert_eq!(a.top, b.top);
        assert_eq!(a.width, b.width);
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.advance, b.advance);
        assert_eq!(a.coverage, b.coverage);
    }

    #[test]
    fn cache_stats_count_misses_then_hits() {
        let face = open();
        // A fresh face has been reported no lookups.
        assert_eq!(face.cache_stats().lookups(), 0);
        // The GPU batcher reports a miss the first time it packs a glyph into the
        // atlas and a hit on every later frame that reuses the slot.
        face.record_glyph_miss();
        face.record_glyph_miss();
        face.record_glyph_hit();
        face.record_glyph_hit();
        let stats = face.cache_stats();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 2);
        assert!((stats.hit_rate() - 0.5).abs() < 1e-6, "2 hits of 4 lookups");
    }

    #[test]
    fn fonts_cache_stats_sum_across_faces() {
        let fonts = Fonts::new(&[16, 24]).expect("a default font should be available");
        assert_eq!(fonts.cache_stats().lookups(), 0);
        // Report through two different sizes (hence different faces); the rolled-up
        // tally must see both faces' counts.
        fonts.face(16, FontStyle::Regular).record_glyph_miss();
        fonts.face(24, FontStyle::Regular).record_glyph_miss();
        fonts.face(16, FontStyle::Regular).record_glyph_hit();
        let stats = fonts.cache_stats();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn advance_is_cached_and_matches_the_uncached_measure() {
        let face = open();
        // A repeat lookup must return the same width and populate exactly one entry.
        let direct = face.measure_advance('M');
        let first = face.advance('M');
        let second = face.advance('M');
        assert_eq!(first, direct, "cached advance matches the raw measure");
        assert_eq!(first, second);
        assert!(first > 0.0);
        assert_eq!(face.advances.borrow().len(), 1, "one cached character");
    }

    #[test]
    fn space_is_blank_but_advances() {
        let face = open();
        let g = face.rasterize(' ');
        assert_eq!(g.rows, 0);
        assert!(g.coverage.is_empty());
        assert!(g.advance > 0.0, "space still advances the pen");
    }

    #[test]
    fn line_metrics_have_expected_signs() {
        let face = open();
        let (ascent, descent) = face.line_metrics();
        assert!(ascent > 0.0);
        assert!(descent < 0.0);
    }

    #[test]
    fn render_metrics_never_overlap_lines() {
        // FreeType can report a baseline-to-baseline height a pixel under the
        // rounded ink height (it rounds ascender up and descender down apart from
        // the height). The render metrics clamp so a line box always covers its
        // own ink, whatever the raw font reports.
        let m = open().metrics();
        assert!(m.line_height >= m.ascent + m.descent);
    }

    #[test]
    fn metrics_are_positive_and_consistent() {
        let m = open().metrics();
        assert!(m.ascent > 0 && m.descent > 0);
        assert!(m.line_height >= m.ascent + m.descent);
    }

    #[test]
    fn fonts_scale_metrics_with_size() {
        let fonts = Fonts::new(&[16, 32]).expect("a default font");
        let small = fonts.metrics(16);
        let large = fonts.metrics(32);
        assert!(large.line_height > small.line_height, "32px is taller");
    }

    #[test]
    fn line_height_scale_grows_the_box_and_keeps_glyph_metrics() {
        let base = Fonts::new(&[14]).expect("a default font").metrics(14);
        // A scale well above any 14px font's own line height forces the box to
        // the scaled target; 3.0 gives 42px, past ascent + descent.
        let big = Fonts::with_config(&FontConfig::default(), &[14], 3.0)
            .expect("a default font")
            .metrics(14);
        assert_eq!(big.ascent, base.ascent, "ascent stays the glyph metric");
        assert_eq!(big.descent, base.descent, "descent stays the glyph metric");
        assert_eq!(big.line_height, 42, "line box is round(3.0 * 14)");
        assert!(
            big.line_height > big.ascent + big.descent,
            "the surplus is leading below the line"
        );
    }

    #[test]
    fn line_height_scale_floors_at_the_font_line_height() {
        // A scale too small to matter leaves the font's own line height, so
        // lines never overlap.
        let base = Fonts::new(&[14]).expect("a default font").metrics(14);
        let tiny = Fonts::with_config(&FontConfig::default(), &[14], 0.1)
            .expect("a default font")
            .metrics(14);
        assert_eq!(tiny.line_height, base.line_height);
    }

    #[test]
    fn fonts_fall_back_for_an_unopened_size() {
        let fonts = Fonts::new(&[16]).expect("a default font");
        // An unrequested size resolves to the only opened face, not a panic.
        assert_eq!(fonts.metrics(99).line_height, fonts.metrics(16).line_height);
    }

    #[test]
    fn face_resolves_every_style_without_panicking() {
        let fonts = Fonts::new(&[16, 32]).expect("a default font");
        // Every (size, style) pair resolves to a usable face: a present variant,
        // or the regular face when the family lacks it, or the fallback size.
        for &size in &[16, 32, 99] {
            for style in [
                FontStyle::Regular,
                FontStyle::Bold,
                FontStyle::Italic,
                FontStyle::BoldItalic,
            ] {
                assert!(
                    fonts.face(size, style).advance('m') > 0.0,
                    "{style:?} at {size}px advances"
                );
            }
        }
    }

    #[test]
    fn fonts_reject_an_empty_size_set() {
        assert!(Fonts::new(&[]).is_err());
        assert!(Fonts::new(&[0]).is_err());
    }

    #[test]
    fn code_face_resolves_for_every_size_without_panicking() {
        let fonts = Fonts::new(&[16, 32]).expect("a default font");
        // The code face is usable at opened and unopened sizes: a distinct code
        // family if installed, else the prose regular face. Never panics.
        for &size in &[16, 32, 99] {
            assert!(
                fonts.code_face(size).advance('m') > 0.0,
                "code face at {size}px advances"
            );
        }
    }

    #[test]
    fn ui_faces_resolve_and_route_without_panicking() {
        let fonts = Fonts::new(&[16, 32]).expect("a default font");
        for &size in &[16, 32, 99] {
            // The interface face resolves for every style at opened and unopened
            // sizes (italic styles fall to their upright weight; a missing bold
            // falls to regular), always advancing, never panicking.
            for style in [
                FontStyle::Regular,
                FontStyle::Medium,
                FontStyle::Bold,
                FontStyle::Italic,
                FontStyle::BoldItalic,
            ] {
                assert!(
                    fonts.ui_face(size, style).advance('n') > 0.0,
                    "ui face {style:?} at {size}px advances"
                );
            }
            // A `FaceKey::Ui` routes through `face_for` to the same face `ui_face`
            // returns, so a display-list run and a direct lookup agree.
            let key = FaceKey::Ui {
                size,
                style: FontStyle::Bold,
            };
            assert!(std::ptr::eq(
                fonts.face_for(key),
                fonts.ui_face(size, FontStyle::Bold)
            ));
            // Its vertical metrics are a sane, positive box (used for the tab-bar
            // baseline). Bold is never lighter than regular (equal only when the
            // installed family ships no distinct bold).
            let m = fonts.ui_metrics(size);
            assert!(m.ascent > 0 && m.descent > 0 && m.line_height >= m.ascent + m.descent);
            let ink = |style| {
                fonts
                    .ui_face(size, style)
                    .rasterize('B')
                    .coverage
                    .iter()
                    .map(|&c| c as u64)
                    .sum::<u64>()
            };
            assert!(
                ink(FontStyle::Bold) >= ink(FontStyle::Regular),
                "ui bold is never lighter than ui regular"
            );
        }
    }

    #[test]
    fn faces_route_emoji_clusters_and_keep_ascii_on_the_char_path() {
        let fonts = Fonts::new(&[32]).expect("a default font");
        let face = fonts.face(32, FontStyle::Regular);
        let advance = face.cluster_advance("😀");
        assert!(advance > 0.0);
        // Drawn at the em (32px), the measure twin: `cluster_advance` measures at
        // the em too, so the two agree for a wide cluster.
        assert_eq!(
            face.with_cluster_glyph("😀", 32, |g| g.advance),
            Some(advance),
            "draw and measure route identically"
        );
        assert!(
            face.with_cluster_glyph("a", 32, |_| ()).is_none(),
            "ASCII never routes to the emoji path"
        );
        assert_eq!(
            face.cluster_advance("a"),
            face.advance('a'),
            "a simple cluster measures exactly like its character"
        );
    }

    #[test]
    fn glyph_face_falls_back_for_a_nerd_font_icon() {
        // U+F418 is the Nerd Font octicon "git-branch" (nf-oct-git_branch), the
        // icon a git prompt prints. It lives in the private-use area, so a plain
        // text or monospace family never carries it: the config's fallback chain
        // (a symbols font) must supply it or the cell renders as a blank / tofu
        // box. This holds for either app: with an empty fallback list (a text
        // editor) the resolver simply returns the primary and the ink assertion
        // is skipped; with a symbols font configured (a terminal) it inks.
        const BRANCH: char = '\u{F418}';
        let fonts = Fonts::new(&[16]).expect("a default font");
        let key = FaceKey::Prose {
            size: 16,
            style: FontStyle::Regular,
        };

        // A glyph the primary owns resolves to the primary itself, untouched, so
        // the common path never walks the fallback chain.
        let primary = fonts.face_for(key);
        assert!(primary.has_scalar('A'));
        assert!(std::ptr::eq(fonts.glyph_face(key, 'A'), primary));

        // Resolving the icon never panics whether or not a fallback is present.
        // When the primary lacks it but a fallback covers it, the resolved face
        // must ink it: that inked raster is exactly the fix for the blank cell.
        let resolved = fonts.glyph_face(key, BRANCH);
        if !primary.has_scalar(BRANCH) && resolved.has_scalar(BRANCH) {
            let g = resolved.rasterize(BRANCH);
            assert!(
                g.coverage.iter().any(|&c| c > 0),
                "the branch icon should ink from a fallback face"
            );
        }
    }
}
