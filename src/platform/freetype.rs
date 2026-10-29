//! Glyph rasterization backed by the system FreeType (libfreetype.so).
//!
//! FreeType turns a TrueType/OpenType outline into an anti-aliased coverage
//! bitmap, handling hinting, scan conversion, and per-size metrics. This is the
//! thin FFI layer over it: a small extern block, one struct owning the C-side
//! handles, a Drop impl, and methods returning owned data. The font file is read
//! into a Vec the Face borrows, rather than mmapped, and coverage is a `Vec<u8>`.
//!
//! FreeType exposes glyph data as struct fields reached from the Face pointer
//! (face->glyph->bitmap, face->size->metrics), not accessors. The repr(C)
//! structs below mirror those layouts up to the last field this module reads;
//! FreeType allocates the full structs, we only read a prefix. The offset of
//! every field read is pinned in the tests against the C headers.

use core::ffi::{c_char, c_int, c_long, c_short, c_uint, c_ulong, c_ushort, c_void};
use core::ptr::{self, NonNull};
use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashMap;
use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::rc::Rc;

use crate::platform::emoji::{wants_emoji, ColorGlyph, EmojiFont};
use crate::platform::error::{Error, Result};
use crate::platform::fontconfig::{self, FontFile, Fontconfig};

/// A font family resolved to one face per terminal style.
///
/// Missing cuts and cuts substituted from another family use the regular face. Each
/// [`FontFile`] retains collection and variable instance indices selected by fontconfig.
#[derive(Clone, Debug)]
pub struct FontFamily {
    pub regular: FontFile,
    pub bold: FontFile,
    pub italic: FontFile,
    pub bold_italic: FontFile,
    matched_families: Vec<String>,
}

impl FontFamily {
    /// Resolve a real or generic family name, querying each style independently.
    ///
    /// Missing cuts use the regular face. An uninstalled named family returns `None`.
    pub fn resolve(name: &str) -> Option<Self> {
        let regular = fontconfig::match_family(name, FontStyle::Regular)?;
        let mut family = Self {
            bold: regular.file.clone(),
            italic: regular.file.clone(),
            bold_italic: regular.file.clone(),
            regular: regular.file,
            matched_families: regular.families,
        };
        family.bold = family
            .resolve_style(name, FontStyle::Bold)
            .unwrap_or_else(|| family.regular.clone());
        family.italic = family
            .resolve_style(name, FontStyle::Italic)
            .unwrap_or_else(|| family.regular.clone());
        family.bold_italic = family
            .resolve_style(name, FontStyle::BoldItalic)
            .unwrap_or_else(|| family.regular.clone());
        Some(family)
    }

    /// Resolve a style only when fontconfig keeps it in this family.
    fn resolve_style(&self, name: &str, style: FontStyle) -> Option<FontFile> {
        let matched = fontconfig::match_family(name, style)?;
        style_file_in_family(matched, &self.matched_families)
    }
}

/// Keep a style match only when it shares a family name with the regular face.
fn style_file_in_family(
    matched: fontconfig::FamilyMatch,
    selected_families: &[String],
) -> Option<FontFile> {
    matched
        .families
        .iter()
        .any(|family| {
            selected_families
                .iter()
                .any(|selected| selected.eq_ignore_ascii_case(family))
        })
        .then_some(matched.file)
}

/// Font-family preferences for grid text, chrome, emoji, and explicit fallbacks.
///
/// Fontconfig resolves names to files on the current machine. Candidate lists use the
/// first installed family.
#[derive(Clone, Debug)]
pub struct FontConfig {
    /// Prose family candidates for the grid, first installed wins. A list ending in a
    /// generic alias (`monospace`) resolves on any machine with a font at all.
    pub families: Vec<String>,
    /// Proportional families for chrome, first installed wins.
    /// The prose family is used when none resolve.
    pub ui: Vec<String>,
    /// The colour emoji family. [`EmojiFont::open`] verifies its bitmap strikes.
    pub emoji: String,
    /// Ordered overrides for characters absent from the selected face. Remaining
    /// characters use system discovery through
    /// [`Fontconfig::font_for_char`](crate::platform::fontconfig::Fontconfig::font_for_char)).
    ///
    /// Private-use symbols need explicit families because their codepoints have no
    /// universal meaning.
    pub fallback: Vec<String>,
}

/// A [`FontConfig`] resolved to font files independently of pixel size.
///
/// Size changes reopen these files without repeating family discovery.
#[derive(Clone, Debug)]
pub struct FontSelection {
    /// The grid's family.
    pub prose: FontFamily,
    /// The chrome family; the prose family when no interface candidate is installed.
    pub ui: FontFamily,
    /// The medium cut of the interface family, for the tab label. `None` when the family
    /// has no such cut, which leaves the label at the UI regular weight.
    pub ui_medium: Option<FontFile>,
    /// The pinned fallback faces, in the order they are consulted.
    pub fallback: Vec<FontFile>,
    /// The colour emoji family, when one is installed.
    pub emoji: Option<FontFile>,
}

impl FontSelection {
    /// Resolve every candidate list in `config` against the installed fonts.
    ///
    /// Errors only when no prose candidate resolves, which for a list ending in a generic
    /// alias means a machine with no fonts at all.
    pub fn resolve(config: &FontConfig) -> Result<Self> {
        let (_, prose) = first_installed(&config.families).ok_or_else(|| {
            Error::msg(format!(
                "no usable font family: fontconfig resolves none of {:?}",
                config.families
            ))
        })?;

        // A medium query returning the regular face means the family has no medium cut.
        let ui = first_installed(&config.ui);
        let ui_medium = ui.as_ref().and_then(|(name, family)| {
            family
                .resolve_style(name, FontStyle::Medium)
                .filter(|medium| *medium != family.regular)
        });
        let ui = ui.map_or_else(|| prose.clone(), |(_, family)| family);

        let fallback = config
            .fallback
            .iter()
            .filter_map(|name| fontconfig::font_for_family(name, FontStyle::Regular))
            .collect();

        let emoji = fontconfig::font_for_family(&config.emoji, FontStyle::Regular);

        Ok(Self {
            prose,
            ui,
            ui_medium,
            fallback,
            emoji,
        })
    }
}

/// The first candidate in `names` that is installed, with the name that found it.
fn first_installed(names: &[String]) -> Option<(&str, FontFamily)> {
    names
        .iter()
        .find_map(|name| FontFamily::resolve(name).map(|family| (name.as_str(), family)))
}

/// Which variant of the font family a run renders in: one open font file each.
/// Ordinary output is [`FontStyle::Regular`]; a cell's `SGR 1`/`SGR 3` bits select a
/// heavier or slanted file (`term_render::style_of` does the mapping).
///
/// Kept distinct from [`crate::grid::Attrs`] so this font layer stays independent of
/// the terminal's vocabulary. It knows four files, not what made a run bold — which is
/// what lets the tab bar's chrome text ask for the same four without pretending to
/// carry a rendition.
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
/// [`Fonts`] holds: the prose (grid) family and the interface family, each at a style.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FaceKey {
    Prose {
        size: u32,
        style: FontStyle,
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
            FaceKey::Prose { size, .. } | FaceKey::Ui { size, .. } => size,
        }
    }
}

// ---------------------------------------------------------------------------
// FFI surface.
// ---------------------------------------------------------------------------

type FtError = c_int;
type FtLibrary = *mut c_void;
type FtFace = *mut FtFaceRec;

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
    fn FT_New_Face(
        library: FtLibrary,
        filepathname: *const c_char,
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

impl FtBitmap {
    /// The rows top down, the first `row_bytes` of each, or `None` when FreeType gave no
    /// buffer or a row would not fit in the pitch. A negative pitch stores the rows bottom
    /// up, with `buffer` at the last.
    pub(crate) fn rows_top_down(&self, row_bytes: usize) -> Option<impl Iterator<Item = &[u8]>> {
        let stride = self.pitch.unsigned_abs() as usize;
        if self.buffer.is_null() || row_bytes > stride {
            return None;
        }
        let rows = self.rows as usize;
        let bottom_up = self.pitch < 0;
        Some((0..rows).map(move |y| {
            let line = if bottom_up { rows - 1 - y } else { y };
            // SAFETY: FreeType's buffer holds `rows` lines of `|pitch|` bytes in memory
            // order, so line `line` starts `line * |pitch|` bytes in and its first
            // `row_bytes <= |pitch|` bytes lie inside it. The slice borrows the bitmap,
            // whose buffer FreeType keeps until the glyph slot's next load.
            unsafe { std::slice::from_raw_parts(self.buffer.add(line * stride), row_bytes) }
        }))
    }
}

#[cfg(test)]
impl FtBitmap {
    /// A bitmap over `buffer` as FreeType describes one, for the copy tests.
    pub(crate) fn over(rows: c_uint, width: c_uint, pitch: c_int, buffer: *const u8) -> Self {
        Self {
            rows,
            width,
            pitch,
            buffer,
            num_grays: 256,
            pixel_mode: 0,
            palette_mode: 0,
            palette: ptr::null_mut(),
        }
    }
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
/// Leading added on top of the font's own line height (see [`Fonts::with_selection`])
/// is a different thing: that one is split evenly, so text stays centered in the
/// taller line box rather than sinking to its floor.
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

/// Owns the FreeType library and one face opened from a font file.
pub struct Face {
    library: NonNull<c_void>,
    face: NonNull<FtFaceRec>,
    /// Pen advances cached by character. Layout measures every character to wrap
    /// and size runs but the GPU atlas rasterizes only the ones on screen, and the
    /// advance is constant for a fixed-size face, so caching it turns a
    /// per-character `FT_Load_Char` (hinting plus metrics) into a map hit. Without
    /// this a full re-layout reloads every glyph's metrics through FreeType, which
    /// dominates the rebuild a selection drag triggers each frame. Interior
    /// mutability so measuring through a shared `&Face` can fill it.
    advances: RefCell<HashMap<char, f32>>,
    /// The pixel size this face was opened at, recorded by `set_pixel_size`.
    /// The emoji face scales its color glyphs to this so an emoji sits on the
    /// same em square as the surrounding text.
    pixel_size: Cell<u32>,
    /// The shared color emoji font, if one is installed; [`Fonts::with_selection`] attaches
    /// it to every face it opens. `None` on the emoji face itself and in
    /// face-only tests, in which case emoji clusters take the per-character
    /// path (tofu, exactly the pre-emoji behavior).
    emoji: Option<Rc<EmojiFont>>,
}

/// How a glyph loads, whether to measure it or to draw it, so the pen moves by the
/// advance of the glyph whose ink is drawn. Drawing adds only [`FT_LOAD_RENDER`].
const GLYPH_LOAD: i32 = FT_LOAD_NO_BITMAP;

impl Face {
    /// Open the regular face of the first available font family at `pixel_size`.
    /// A test-only convenience: production goes through [`Fonts`], which opens the
    /// family's variants per size itself.
    #[cfg(test)]
    pub fn open_default(pixel_size: u32) -> Result<Self> {
        let selection = FontSelection::resolve(&FontConfig::default())?;
        let face = Self::from_file(&selection.prose.regular)?;
        face.set_pixel_size(pixel_size)?;
        Ok(face)
    }

    /// Open the path and face index selected by fontconfig.
    pub fn from_file(file: &FontFile) -> Result<Self> {
        Self::from_path_index(&file.path, file.index)
    }

    /// Build face `index` of a font file on disk. The index is 0 for an ordinary font
    /// and only meaningful for a TrueType *collection*, where several faces share one
    /// file: Noto's CJK fonts ship that way, and taking face 0 of a collection quietly
    /// gives the wrong language. Font discovery ([`Fontconfig`](crate::platform::fontconfig))
    /// reports the index alongside the path, so it is carried rather than assumed.
    ///
    /// FreeType maps the file itself. The pages belong to the page cache, shared with
    /// every other process using the font, and only the tables a glyph touches become
    /// resident. The cost is a mapping's: truncating the file in place under a running
    /// terminal faults it, while replacing it, as a package manager does, leaves the old
    /// file mapped until the face closes.
    pub fn from_path_index(path: &Path, index: i32) -> Result<Self> {
        let c_path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| Error::msg(format!("font path {} holds a NUL", path.display())))?;

        let mut library: FtLibrary = ptr::null_mut();
        // SAFETY: alibrary points at a live local; FT writes the handle through
        // it and returns nonzero on failure.
        let err = unsafe { FT_Init_FreeType(&mut library) };
        let library = NonNull::new(library)
            .filter(|_| err == 0)
            .ok_or_else(|| Error::msg("FT_Init_FreeType failed"))?;

        let mut face: FtFace = ptr::null_mut();
        // SAFETY: library is valid; c_path is a NUL-terminated string FreeType reads only
        // during the call; aface is a live local.
        let err = unsafe {
            FT_New_Face(
                library.as_ptr(),
                c_path.as_ptr(),
                index as c_long,
                &mut face,
            )
        };
        let face = match NonNull::new(face).filter(|_| err == 0) {
            Some(face) => face,
            None => {
                // SAFETY: library came from FT_Init_FreeType and is freed once.
                unsafe { FT_Done_FreeType(library.as_ptr()) };
                return Err(Error::msg(format!(
                    "FT_New_Face failed for {}",
                    path.display()
                )));
            }
        };

        Ok(Self {
            library,
            face,
            advances: RefCell::new(HashMap::new()),
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

    /// Rasterize a character at the current pixel size, copying the coverage
    /// out. A load failure or glyphless character yields an empty glyph. Nothing is
    /// cached here; the GPU batcher keeps the packed result per `(FaceKey, char)`.
    pub fn rasterize(&self, ch: char) -> Glyph {
        // SAFETY: face is valid; FT_Load_Char renders into the face's shared
        // glyph slot. No reference into that slot is held across this call.
        let err = unsafe {
            FT_Load_Char(
                self.face.as_ptr(),
                ch as u32 as c_ulong,
                GLYPH_LOAD | FT_LOAD_RENDER,
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
        // SAFETY: face is valid; without FT_LOAD_RENDER the load fills the glyph
        // slot's metrics and renders nothing. The advance is read out immediately.
        let err = unsafe { FT_Load_Char(self.face.as_ptr(), ch as u32 as c_ulong, GLYPH_LOAD) };
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

    /// The color glyph for an emoji cluster rasterized at `target` pixels, or `None`
    /// when the cluster renders through the ordinary per-character path (not an
    /// emoji, no emoji font installed, or the emoji font cannot form it). `None`
    /// tells the caller to draw the cluster's characters instead.
    ///
    /// The caller passes `target` because the fit differs by path: a wide
    /// cluster (two grid cells) is drawn at the em (its natural ~square size
    /// filling both cells), but a single-width emoji is scaled down to one cell
    /// so it does not spill into the next column.
    pub fn cluster_glyph(&self, cluster: &str, target: u32) -> Option<ColorGlyph> {
        let emoji = self.emoji.as_ref()?;
        if !wants_emoji(cluster, |c| self.has_scalar(c)) {
            return None;
        }
        emoji.raster(cluster, target)
    }

    /// Whether this face's character map has a real glyph for `ch` (a missing
    /// character maps to glyph 0, `.notdef`, the tofu box).
    fn has_scalar(&self, ch: char) -> bool {
        // SAFETY: face is valid; the call is a read-only cmap lookup.
        unsafe { FT_Get_Char_Index(self.face.as_ptr(), ch as u32 as c_ulong) != 0 }
    }

    /// The raw FreeType face handle, for binding a HarfBuzz font to this face and
    /// for the emoji module to decode color strikes (it reads the glyph slot after a
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
        // SAFETY: face is valid; face->size is non-null once a size is set. The three
        // 26.6 values are copied out, and read as zero when no size is set.
        let size = NonNull::new(unsafe { (*self.face.as_ptr()).size });
        let (ascender, descender, height) = size.map_or((0, 0, 0), |size| {
            let m = unsafe { &size.as_ref().metrics };
            (m.ascender, m.descender, m.height)
        });
        let ascent = (ascender as f32 / 64.0).ceil() as i32;
        let descent = (-(descender as f32 / 64.0)).ceil() as i32;
        let line_height = ((height as f32 / 64.0).ceil() as i32).max(ascent + descent);
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
        // The face must be freed before the library it came from. SAFETY: each handle
        // came from its matching FreeType new call and is freed exactly once.
        unsafe {
            FT_Done_Face(self.face.as_ptr());
            FT_Done_FreeType(self.library.as_ptr());
        }
    }
}

/// How many characters [`SizedFaces::discovered`] memoizes before it is dropped whole.
///
/// It is keyed on scalars the child chooses, and nothing ever evicted: `cat`ing a file
/// that sweeps the unassigned code points walks it toward ~1.1 M entries (~35 MB) and
/// leaves them there for the life of the process. Cheap per miss, so it is growth rather
/// than a stall — which is exactly the kind that goes unnoticed.
///
/// Generous, because the real working set is small. `discover` is only reached by a
/// scalar the keyed face *and* the pinned fallback list both fail to draw, so what
/// accumulates is scripts rather than characters: a session mixing Latin, CJK, Cyrillic
/// and a symbol font stays in the low hundreds.
///
/// Dropped whole rather than evicted entry by entry, the same policy as the glyph cache's
/// `MAX_GLYPH_CACHE` and for the same reason: a miss costs one fontconfig query, so
/// rebuilding is a bounded burst and an LRU's bookkeeping would cost more than it saves.
const MAX_DISCOVERED: usize = 4096;

/// The faces for one pixel size: the regular face (always present) plus any
/// bold/italic/bold-italic variants the family ships, and the cached line metrics.
/// Metrics come from the regular face only, so emphasis selects a different glyph
/// set without ever changing line height.
struct SizedFaces {
    size: u32,
    regular: Rc<Face>,
    bold: Option<Rc<Face>>,
    italic: Option<Rc<Face>>,
    bold_italic: Option<Rc<Face>>,
    /// The interface family's proportional faces at this size (for chrome text; see
    /// [`FaceKey::Ui`]). `ui_regular` is always present (it falls back to the prose
    /// family when no UI family is installed); `ui_medium` and `ui_bold` are `None`
    /// when no such weight file is installed, and then that weight uses `ui_regular`.
    /// `ui_metrics` are the interface face's own vertical metrics, so a UI baseline
    /// centers on the sans ink box rather than the monospace one.
    ui_regular: Rc<Face>,
    ui_medium: Option<Rc<Face>>,
    ui_bold: Option<Rc<Face>>,
    ui_metrics: Metrics,
    /// The pinned fallback chain at this size ([`FontConfig::fallback`]), in priority
    /// order, holding only the faces whose files exist. Consulted by
    /// [`Fonts::glyph_face`] for a scalar the keyed face cannot draw; empty when
    /// the config lists none.
    ///
    /// This is an *override* list, not the whole story: it is what we insist on
    /// (the private-use icon range comes from a Nerd Font, whatever the system
    /// thinks), and anything it does not answer falls through to `discovered`.
    fallback: Vec<Rc<Face>>,
    /// Faces the *system* offered for scalars nothing above could draw, memoized by
    /// character. `None` records a character fontconfig could not place either, so a
    /// glyph nothing on the machine has is asked about once and then remembered —
    /// otherwise a screenful of an unavailable character would re-query per cell.
    ///
    /// Capped at [`MAX_DISCOVERED`]: the keys come from the child, so this is
    /// attacker-growable like every other child-keyed map here.
    discovered: RefCell<HashMap<char, Option<Rc<Face>>>>,
    /// The faces behind `discovered`, keyed by the file they came from, so a font that
    /// answers for twenty characters is opened once and shared. This is why the faces
    /// above are `Rc`: they are reached by two maps and owned by neither.
    opened: RefCell<HashMap<FontFile, Rc<Face>>>,
    metrics: Metrics,
}

impl SizedFaces {
    /// The face the system offered for `ch`, opened at this size, or `None` when nothing
    /// installed can draw it. Memoized both ways: once per character, and once per font
    /// file behind it.
    ///
    /// `fc` is passed in rather than held because discovery is lazy — a session that
    /// never leaves Latin never builds the system font cache at all (see
    /// [`Fontconfig::new`]).
    fn discover(&self, fc: Option<&Fontconfig>, ch: char) -> Option<Rc<Face>> {
        if let Some(memo) = self.discovered.borrow().get(&ch) {
            return memo.clone();
        }
        let found = fc
            .and_then(|fc| fc.font_for_char(ch))
            .and_then(|file| self.open_discovered(file));
        let mut memo = self.discovered.borrow_mut();
        if memo.len() >= MAX_DISCOVERED {
            memo.clear();
        }
        memo.insert(ch, found.clone());
        found
    }

    /// Open (or reuse) the face for a font file the system named. Discovered faces carry
    /// no emoji font: they draw single scalars, never colour clusters.
    fn open_discovered(&self, file: FontFile) -> Option<Rc<Face>> {
        if let Some(face) = self.opened.borrow().get(&file) {
            return Some(Rc::clone(face));
        }
        let face = Face::from_file(&file).ok()?;
        face.set_pixel_size(self.size).ok()?;
        let face = Rc::new(face);
        self.opened.borrow_mut().insert(file, Rc::clone(&face));
        Some(face)
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
    /// The lazily ranked system fallback set. The inner `None` means fontconfig was
    /// unavailable; selected and explicit fallback faces remain usable.
    fc: OnceCell<Option<Fontconfig>>,
}

impl Fonts {
    /// Open the default font selection at each of `sizes`. Tests and the perf
    /// harness use this; the app resolves its configured selection once and opens it
    /// through [`Self::with_selection`].
    pub fn new(sizes: &[u32]) -> Result<Self> {
        Self::with_selection(&FontSelection::resolve(&FontConfig::default())?, sizes)
    }

    /// Open an already-resolved [`FontSelection`] at each of `sizes` (zeros and
    /// duplicates skipped), loading every available style per size. Each size's
    /// line box is the font's own line height, or the pixel size when that is
    /// taller. At least one nonzero size is required; the first opened becomes the
    /// fallback.
    pub fn with_selection(selection: &FontSelection, sizes: &[u32]) -> Result<Self> {
        let family = &selection.prose;
        let ui_family = &selection.ui;
        let emoji = selection
            .emoji
            .as_ref()
            .and_then(EmojiFont::open)
            .map(Rc::new);
        let mut sized: Vec<SizedFaces> = Vec::new();
        for &size in sizes {
            if size == 0 || sized.iter().any(|s| s.size == size) {
                continue;
            }
            let regular = open_face(&family.regular, size, emoji.as_ref())?;
            // A font spaced tighter than its pixel size has its line box grown to
            // the size, so a row is never shorter than an em. Ascent and descent
            // stay the glyph metrics; the extra is leading, so it splits evenly and
            // the baseline moves down half of it (see `Metrics`).
            let em = i32::try_from(size).map_err(|_| Error::msg("font size out of range"))?;
            let metrics = regular.metrics().grown_to(em);
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
            let fallback = selection
                .fallback
                .iter()
                .filter_map(|file| open_variant(file, size, None))
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
                ui_regular,
                ui_medium: selection
                    .ui_medium
                    .as_ref()
                    .and_then(|file| open_variant(file, size, None)),
                ui_bold: open_variant(&ui_family.bold, size, None),
                ui_metrics,
                fallback,
                discovered: RefCell::new(HashMap::new()),
                opened: RefCell::new(HashMap::new()),
                metrics,
            });
        }
        if sized.is_empty() {
            return Err(Error::msg("fonts need at least one nonzero size"));
        }
        Ok(Self {
            sized,
            fc: OnceCell::new(),
        })
    }

    /// The entry for `size`, or the fallback (first opened) if `size` is unknown.
    fn entry(&self, size: u32) -> &SizedFaces {
        // `sized` is never empty (`with_selection` errors otherwise), so the `[0]`
        // fallback for an unknown size cannot panic.
        debug_assert!(!self.sized.is_empty());
        self.sized
            .iter()
            .find(|s| s.size == size)
            .unwrap_or(&self.sized[0])
    }

    /// The interface family's vertical metrics for `size` (its own ascent/descent),
    /// so a UI baseline centers on the sans ink box, not the monospace one.
    pub fn ui_metrics(&self, size: u32) -> Metrics {
        self.entry(size).ui_metrics
    }

    /// Resolve a [`FaceKey`] to its face, the inverse of the key a run records. Falls
    /// back as [`face_rc`](Self::face_rc) describes, so it never fails and never panics.
    pub fn face_for(&self, key: FaceKey) -> &Face {
        self.face_rc(key)
    }

    /// The physical face to rasterize `ch` in for `key`, resolved in three steps:
    ///
    /// 1. **The keyed face**, when its character map covers `ch`. Every ASCII and Latin
    ///    glyph stops here, on one [`FT_Get_Char_Index`], so the common case pays nothing
    ///    for the machinery below it.
    /// 2. **The pinned chain** ([`FontConfig::fallback`]) — the fonts we insist on. A
    ///    terminal's Nerd Font and Powerline icons live in the private-use area, which no
    ///    text family carries and which no system rule would resolve the way we want, so
    ///    that range is nailed to a symbols font here rather than left to chance.
    /// 3. **The system** ([`Fontconfig`]) — whatever is installed. This is what makes
    ///    bnkterm draw the same `✓` as every other application on the machine, and the
    ///    reason a program can print a character nobody anticipated and still see it.
    ///
    /// Failing all three, the keyed face is returned again, so a scalar that genuinely
    /// exists nowhere still draws its `.notdef` box, as it always did.
    ///
    /// A face with the emoji font resolves a colour cluster through the emoji cluster
    /// path before any scalar is drawn. An interface face has none, so a tab title's
    /// emoji arrives here like any other scalar.
    ///
    /// Every step is memoized. Discovery caches per character and per font file
    /// ([`SizedFaces::discover`]), and the GPU batcher caches the raster by `(key, ch)`
    /// on top of that, so this runs once per glyph the terminal has never seen — never
    /// per frame, and never per cell.
    pub fn glyph_face(&self, key: FaceKey, ch: char) -> Rc<Face> {
        let primary = self.face_rc(key);
        if primary.has_scalar(ch) {
            return Rc::clone(primary);
        }
        let entry = self.entry(key.size());
        if let Some(face) = entry.fallback.iter().find(|f| f.has_scalar(ch)) {
            return Rc::clone(face);
        }
        entry
            .discover(self.fontconfig(), ch)
            .unwrap_or_else(|| Rc::clone(primary))
    }

    /// Rank system fallback fonts when the selected faces first miss a character.
    fn fontconfig(&self) -> Option<&Fontconfig> {
        self.fc.get_or_init(Fontconfig::new).as_ref()
    }

    /// The keyed face as the shared handle the caches hold, so [`glyph_face`](Self::glyph_face)
    /// can hand back either it or a discovered face without the two having different types.
    ///
    /// An unopened size resolves to the first opened one. A prose style the family lacks
    /// falls back to its regular face. The interface family gives medium or bold when
    /// asked and available, else its regular; italic styles resolve to their upright
    /// weight, since UI text never slants.
    fn face_rc(&self, key: FaceKey) -> &Rc<Face> {
        let entry = self.entry(key.size());
        match key {
            FaceKey::Prose { style, .. } => match style {
                // The prose family ships no medium weight, so it reads as regular.
                FontStyle::Regular | FontStyle::Medium => &entry.regular,
                FontStyle::Bold => entry.bold.as_ref().unwrap_or(&entry.regular),
                FontStyle::Italic => entry.italic.as_ref().unwrap_or(&entry.regular),
                FontStyle::BoldItalic => entry.bold_italic.as_ref().unwrap_or(&entry.regular),
            },
            FaceKey::Ui { style, .. } => match style {
                FontStyle::Medium => entry.ui_medium.as_ref().unwrap_or(&entry.ui_regular),
                FontStyle::Bold | FontStyle::BoldItalic => {
                    entry.ui_bold.as_ref().unwrap_or(&entry.ui_regular)
                }
                FontStyle::Regular | FontStyle::Italic => &entry.ui_regular,
            },
        }
    }

    /// Whether *any* face reachable from `key` has a real glyph for `ch`: the keyed face
    /// or, failing that, its fallback chain. The question [`glyph_face`](Self::glyph_face)
    /// cannot answer, because it returns the keyed face on a total miss so the scalar
    /// still draws its `.notdef` box.
    ///
    /// A caller that would rather draw *something else* than a tofu box has to ask first,
    /// and the password cursor is exactly that caller: the padlock lives in the Nerd Fonts
    /// private-use area, so on a machine without a symbols font it would render as an
    /// empty rectangle — the one glyph whose whole job is to be understood at a glance.
    pub fn covers(&self, key: FaceKey, ch: char) -> bool {
        let entry = self.entry(key.size());
        self.face_for(key).has_scalar(ch)
            || entry.fallback.iter().any(|f| f.has_scalar(ch))
            || entry.discover(self.fontconfig(), ch).is_some()
    }

    /// The metrics for `size`, or the fallback's metrics if `size` was not opened.
    pub fn metrics(&self, size: u32) -> Metrics {
        self.entry(size).metrics
    }
}

/// Open `path` at `size` with the shared `emoji` font attached: the three-step
/// face open (read the file, fix the pixel size, share the emoji font) that
/// both the required regular face and the optional variants go through.
fn open_face(file: &FontFile, size: u32, emoji: Option<&Rc<EmojiFont>>) -> Result<Rc<Face>> {
    let mut face = Face::from_file(file)?;
    face.set_pixel_size(size)?;
    face.emoji = emoji.cloned();
    Ok(Rc::new(face))
}

/// Open a style variant at `size`, or `None` if the file is absent or fails to
/// load, leaving the caller to fall back to the regular face. Keeps a family
/// that ships only some weights from being a hard error.
fn open_variant(file: &FontFile, size: u32, emoji: Option<&Rc<EmojiFont>>) -> Option<Rc<Face>> {
    if !file.path.exists() {
        return None;
    }
    open_face(file, size, emoji).ok()
}

/// Copy a FreeType bitmap into a tightly packed top-down coverage buffer of
/// `width * rows` bytes, dropping inter-row pitch padding, or an empty one when
/// FreeType gave no buffer. Assumes one coverage byte per pixel, so the caller
/// must confirm FT_PIXEL_MODE_GRAY first.
fn copy_coverage(bitmap: &FtBitmap) -> Vec<u8> {
    let width = bitmap.width as usize;
    let Some(rows) = bitmap.rows_top_down(width) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(width * bitmap.rows as usize);
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::platform::freetype::*;
    use core::mem::offset_of;

    /// Every field this module reads through a FreeType pointer, and the size of
    /// each embedded struct. Values are `offsetof`/`sizeof` over `freetype.h`, from
    /// a C compiler on x86_64 (LP64).
    #[test]
    fn struct_layouts_match_the_c_abi() {
        assert_eq!(offset_of!(FtFaceRec, num_fixed_sizes), 56);
        assert_eq!(offset_of!(FtFaceRec, glyph), 152);
        assert_eq!(offset_of!(FtFaceRec, size), 160);
        assert_eq!(offset_of!(FtGlyphSlotRec, advance), 128);
        assert_eq!(offset_of!(FtGlyphSlotRec, bitmap), 152);
        assert_eq!(offset_of!(FtGlyphSlotRec, bitmap_left), 192);
        assert_eq!(offset_of!(FtGlyphSlotRec, bitmap_top), 196);
        assert_eq!(size_of::<FtBitmap>(), 40);
        assert_eq!(offset_of!(FtBitmap, rows), 0);
        assert_eq!(offset_of!(FtBitmap, width), 4);
        assert_eq!(offset_of!(FtBitmap, pitch), 8);
        assert_eq!(offset_of!(FtBitmap, buffer), 16);
        assert_eq!(offset_of!(FtBitmap, pixel_mode), 26);
        assert_eq!(offset_of!(FtSizeRec, metrics), 24);
        assert_eq!(size_of::<FtSizeMetrics>(), 56);
        assert_eq!(offset_of!(FtSizeMetrics, y_ppem), 2);
        assert_eq!(offset_of!(FtSizeMetrics, ascender), 24);
        assert_eq!(offset_of!(FtSizeMetrics, descender), 32);
        assert_eq!(offset_of!(FtSizeMetrics, height), 40);
    }

    fn open() -> Face {
        Face::open_default(16).expect("a default font should be available")
    }

    /// A test config with a portable fallback.
    fn config_of(names: &[&str]) -> FontConfig {
        FontConfig {
            families: names.iter().map(|n| (*n).to_string()).collect(),
            ..FontConfig::default()
        }
    }

    #[test]
    fn a_candidate_nobody_has_installed_falls_through_to_the_next() {
        if fontconfig::font_for_family("monospace", FontStyle::Regular).is_none() {
            eprintln!("no usable fontconfig in this environment; skipping");
            return;
        }
        let selection = FontSelection::resolve(&config_of(&["NoSuchFamilyAtAll", "monospace"]))
            .expect("the alias answers where the named family could not");
        assert!(selection.prose.regular.path.exists());
        Face::from_file(&selection.prose.regular).expect("the face fontconfig named opens");
    }

    /// A face leaves its file to FreeType's mapping. Copied to the heap, the fonts a window
    /// opens at startup held about 27 MiB.
    #[test]
    fn a_face_maps_its_file_rather_than_copying_it() {
        let Some(source) = fontconfig::font_for_family("monospace", FontStyle::Regular) else {
            return;
        };
        let path = std::env::temp_dir().join(format!("bnkterm-mapped-{}.font", std::process::id()));
        std::fs::copy(&source.path, &path).expect("copy a font to the test path");

        let opened = Face::from_file(&FontFile {
            path: path.clone(),
            index: source.index,
        });
        let maps = std::fs::read_to_string("/proc/self/maps").expect("read the process maps");
        let _ = std::fs::remove_file(&path);
        let _face = opened.expect("the copied font opens");
        assert!(
            maps.contains(&*path.to_string_lossy()),
            "the font file is not mapped"
        );
    }

    #[test]
    fn a_face_opens_from_a_non_utf8_path() {
        use std::os::unix::ffi::OsStringExt;

        let Some(source) = fontconfig::font_for_family("monospace", FontStyle::Regular) else {
            return;
        };
        let mut name = format!("bnkterm-font-{}-", std::process::id()).into_bytes();
        name.push(0xff);
        name.extend_from_slice(b".font");
        let path = std::env::temp_dir().join(std::ffi::OsString::from_vec(name));
        std::fs::copy(&source.path, &path).expect("copy a font to the test path");

        let opened = Face::from_file(&FontFile {
            path: path.clone(),
            index: source.index,
        });
        let _ = std::fs::remove_file(path);
        opened.expect("open the copied face without converting its path to UTF-8");
    }

    #[test]
    fn the_first_installed_candidate_wins() {
        let (Some(sans), Some(mono)) = (
            fontconfig::font_for_family("sans-serif", FontStyle::Regular),
            fontconfig::font_for_family("monospace", FontStyle::Regular),
        ) else {
            return;
        };
        if sans == mono {
            return;
        }
        let first = FontSelection::resolve(&config_of(&["sans-serif", "monospace"]))
            .expect("resolves")
            .prose
            .regular;
        assert_eq!(first, sans, "the leading candidate is the one used");
        let second = FontSelection::resolve(&config_of(&["monospace", "sans-serif"]))
            .expect("resolves")
            .prose
            .regular;
        assert_eq!(second, mono, "and order is the only thing that decides it");
    }

    #[test]
    fn a_config_naming_nothing_installed_is_an_error_not_a_panic() {
        let err = FontSelection::resolve(&config_of(&["NoSuchFamilyAtAll", "NorThisOne"]))
            .expect_err("nothing resolves");
        assert!(
            format!("{err}").contains("NoSuchFamilyAtAll"),
            "the error names what was asked for: {err}"
        );
    }

    #[test]
    fn fonts_build_from_the_generic_alias_alone() {
        if fontconfig::font_for_family("monospace", FontStyle::Regular).is_none() {
            return;
        }
        let selection = FontSelection::resolve(&config_of(&["NoSuchFamilyAtAll", "monospace"]))
            .expect("the alias resolves");
        let fonts = Fonts::with_selection(&selection, &[16]).expect("fonts open");
        let face = fonts.face_for(FaceKey::Prose {
            size: 16,
            style: FontStyle::Regular,
        });
        assert!(
            face.metrics().line_height > 0,
            "a face opened from the alias has real metrics"
        );
    }

    #[test]
    fn the_ui_medium_cut_is_none_when_the_family_has_none() {
        if fontconfig::font_for_family("monospace", FontStyle::Regular).is_none() {
            return;
        }
        let selection = FontSelection::resolve(&FontConfig::default()).expect("resolves");
        if let Some(medium) = &selection.ui_medium {
            assert_ne!(
                *medium, selection.ui.regular,
                "a medium cut that is the regular file must be reported as absent"
            );
        }
    }

    #[test]
    fn a_style_substituted_from_another_family_is_refused() {
        let selected = vec!["Grid Face".to_string(), "Localized Grid Face".to_string()];
        let expected = FontFile {
            path: "/same-family.ttf".into(),
            index: 0,
        };
        let same_family = fontconfig::FamilyMatch {
            file: expected.clone(),
            families: vec!["localized grid face".to_string()],
        };
        assert_eq!(style_file_in_family(same_family, &selected), Some(expected));

        let foreign = fontconfig::FamilyMatch {
            file: FontFile {
                path: "/foreign-family.ttf".into(),
                index: 0,
            },
            families: vec!["Foreign Face".to_string()],
        };
        assert_eq!(style_file_in_family(foreign, &selected), None);
    }

    #[test]
    fn a_resolved_family_carries_a_cut_for_every_style() {
        let Some(family) = FontFamily::resolve("monospace") else {
            return;
        };
        for file in [
            &family.regular,
            &family.bold,
            &family.italic,
            &family.bold_italic,
        ] {
            assert!(file.path.exists(), "{} is missing", file.path.display());
        }
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
    fn added_leading_splits_evenly_and_keeps_the_glyph_metrics() {
        let face = Face::open_default(16).expect("a default font");
        let m = face.metrics();
        let taller = m.grown_to(m.line_height + 4);
        assert_eq!(taller.line_height, m.line_height + 4);
        assert_eq!(
            taller.baseline,
            m.baseline + 2,
            "added leading is split above and below, not stacked on one side"
        );
        assert_eq!(taller.ascent, m.ascent, "ascent stays the glyph metric");
        assert_eq!(taller.descent, m.descent, "descent stays the glyph metric");
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

    /// A TrueType font with an outline 'A' 500 units wide on a 1000-unit em, so 8 px at
    /// 16 px, and a 1-bit bitmap strike at 16 px whose 'A' advances 11 px. No installed
    /// font pairs outlines with a strike, so the test builds one.
    fn font_with_a_bitmap_strike() -> Vec<u8> {
        let u16s = |v: &[u16]| v.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>();
        let i16s = |v: &[i16]| v.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>();
        let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>();
        let head = [
            &u32s(&[0x0001_0000, 0x0001_0000, 0, 0x5F0F_3CF5])[..], // version, revision, checksum, magic
            &u16s(&[0, 1000]),                                      // flags, units per em
            &[0; 16],                                               // created, modified
            &i16s(&[0, 0, 500, 700]),                               // bounding box
            &u16s(&[0, 8]),    // mac style, smallest readable size
            &i16s(&[2, 0, 0]), // direction hint, short loca, glyph data format
        ]
        .concat();
        let hhea = [
            &u32s(&[0x0001_0000])[..],
            &i16s(&[800, -200, 0]), // ascender, descender, line gap
            &u16s(&[500]),          // widest advance
            &i16s(&[0, 0, 500, 1, 0, 0, 0, 0, 0, 0, 0]), // bearings, extent, caret, reserved, format
            &u16s(&[2]),                                 // horizontal metrics for both glyphs
        ]
        .concat();
        // Two glyphs, the outline one with 4 points in 1 contour, 2 zones.
        let maxp = [
            &u32s(&[0x0001_0000])[..],
            &u16s(&[2, 4, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
        ]
        .concat();
        let hmtx = u16s(&[500, 0, 500, 0]);
        let cmap = [
            &u16s(&[0, 1, 3, 1])[..],       // one encoding, Windows Unicode BMP
            &u32s(&[12]),                   // its subtable
            &u16s(&[4, 32, 0, 4, 4, 1, 0]), // format 4, length, language, 2 segments
            &u16s(&[0x41, 0xFFFF, 0]),      // segment ends, pad
            &u16s(&[0x41, 0xFFFF]),         // segment starts
            &i16s(&[1 - 0x41, 1]),          // 'A' to glyph 1
            &u16s(&[0, 0]),
        ]
        .concat();
        // A rectangle, clockwise from the bottom left, coordinates as 16-bit deltas.
        let glyf = [
            &i16s(&[1, 50, 0, 450, 700])[..], // one contour, bounding box
            &u16s(&[3, 0]),                   // last point, no instructions
            &[1, 1, 1, 1],                    // all on the curve
            &i16s(&[50, 0, 400, 0]),
            &i16s(&[0, 700, 0, -700]),
            &[0, 0],
        ]
        .concat();
        let loca = u16s(&[0, 0, 18]); // glyph 0 is empty, glyph 1 is 36 bytes
                                      // Ascender 13, descender -3, widest 11, caret upright, max before 13, min after -3.
        let line_metrics = [13, 0xFD, 11, 1, 0, 0, 0, 0, 13, 0xFD, 0, 0];
        let eblc = [
            &u32s(&[0x0002_0000, 1])[..], // version, one strike
            &u32s(&[56, 24, 1, 0]),       // index array offset and size, one subtable, colour
            &line_metrics,
            &line_metrics,
            &u16s(&[1, 1]),  // glyphs 1 to 1
            &[16, 16, 1, 1], // 16 px, 1 bit, horizontal
            &u16s(&[1, 1]),  // the index array: glyph 1, its subtable 8 bytes on
            &u32s(&[8]),
            &u16s(&[1, 1]),    // index format 1, image format 1
            &u32s(&[4, 0, 7]), // images from EBDT offset 4, glyph 1 spans 0..7
        ]
        .concat();
        // Small metrics (2 rows of 2 pixels, bearing 0 and 2, advance 11), a byte per row.
        let ebdt = [&u32s(&[0x0002_0000])[..], &[2, 2, 0, 2, 11, 0xC0, 0xC0]].concat();

        let tables = [
            (b"EBDT", ebdt),
            (b"EBLC", eblc),
            (b"cmap", cmap),
            (b"glyf", glyf),
            (b"head", head),
            (b"hhea", hhea),
            (b"hmtx", hmtx),
            (b"loca", loca),
            (b"maxp", maxp),
        ];
        let directory = 12 + 16 * tables.len();
        let mut font = [&u32s(&[0x0001_0000])[..], &u16s(&[9, 128, 3, 16])].concat();
        let mut data = Vec::new();
        for (tag, table) in &tables {
            let offset = u32::try_from(directory + data.len()).unwrap();
            let length = u32::try_from(table.len()).unwrap();
            font.extend_from_slice(*tag);
            font.extend(u32s(&[0, offset, length])); // FreeType does not check the checksum
            data.extend_from_slice(table);
            data.resize(data.len().next_multiple_of(4), 0);
        }
        font.extend(data);
        font
    }

    /// The draw takes a glyph's outline even where the face has a bitmap strike, so the
    /// measure must too. Measured from the strike, the pen moved by the strike's width
    /// while the ink drawn was the outline's.
    #[test]
    fn a_bitmap_strike_does_not_change_the_measured_advance() {
        let path = std::env::temp_dir().join(format!("bnk-strike-font-{}.ttf", std::process::id()));
        std::fs::write(&path, font_with_a_bitmap_strike()).unwrap();
        let face = Face::from_path_index(&path, 0);
        std::fs::remove_file(&path).unwrap();
        let face = face.expect("the font opens");
        assert_eq!(
            face.select_first_strike(),
            Some(16.0),
            "the font has a 16 px strike"
        );
        face.set_pixel_size(16).unwrap();
        let drawn = face.rasterize('A');
        assert!(drawn.rows > 0, "'A' draws from its outline");
        assert_eq!(face.advance('A'), drawn.advance);
    }

    /// On the installed fonts, every printable ASCII glyph advances as far measured as
    /// drawn, in each prose style and the interface face, at a spread of sizes.
    #[test]
    fn the_measured_advance_is_the_drawn_advance() {
        let sizes = [10, 13, 16, 21, 32];
        let fonts = Fonts::new(&sizes).expect("a default font");
        for size in sizes {
            let prose = [
                FontStyle::Regular,
                FontStyle::Bold,
                FontStyle::Italic,
                FontStyle::BoldItalic,
            ]
            .map(|style| FaceKey::Prose { size, style });
            let ui = FaceKey::Ui {
                size,
                style: FontStyle::Regular,
            };
            for key in prose.into_iter().chain([ui]) {
                let face = fonts.face_for(key);
                for ch in '!'..='~' {
                    assert_eq!(
                        face.advance(ch),
                        face.rasterize(ch).advance,
                        "{key:?} '{ch}'"
                    );
                }
            }
        }
    }

    /// FreeType stores a bitmap with a negative pitch bottom up, `buffer` at its last row.
    /// Read top down from `buffer`, its rows came from the memory before it.
    #[test]
    fn coverage_copies_top_down_for_either_pitch_sign() {
        // Three rows of three bytes, four bytes apart, and a spare row of slack.
        let down = [1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0, 0, 0, 0, 0];
        // The same rows bottom up, after two rows of something else.
        let up = [9, 9, 9, 9, 9, 9, 9, 9, 3, 3, 3, 0, 2, 2, 2, 0, 1, 1, 1, 0];
        let expected = [1, 1, 1, 2, 2, 2, 3, 3, 3];
        assert_eq!(
            copy_coverage(&FtBitmap::over(3, 3, 4, down.as_ptr())),
            expected
        );
        assert_eq!(
            copy_coverage(&FtBitmap::over(3, 3, -4, up[8..].as_ptr())),
            expected
        );
        assert!(copy_coverage(&FtBitmap::over(3, 3, 4, ptr::null())).is_empty());
        assert!(
            copy_coverage(&FtBitmap::over(3, 5, 4, down.as_ptr())).is_empty(),
            "a row wider than the pitch is no bitmap FreeType makes"
        );
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
    fn metrics_read_the_descender_as_a_positive_depth() {
        let m = open().metrics();
        assert!(m.ascent > 0);
        // FreeType's descender is negative; the line box measures it downward.
        assert!(m.descent > 0);
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
                    fonts.face_for(FaceKey::Prose { size, style }).advance('m') > 0.0,
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
                    fonts.face_for(FaceKey::Ui { size, style }).advance('n') > 0.0,
                    "ui face {style:?} at {size}px advances"
                );
            }
            // Its vertical metrics are a sane, positive box (used for the tab-bar
            // baseline). Bold is never lighter than regular (equal only when the
            // installed family ships no distinct bold).
            let m = fonts.ui_metrics(size);
            assert!(m.ascent > 0 && m.descent > 0 && m.line_height >= m.ascent + m.descent);
            let ink = |style| {
                fonts
                    .face_for(FaceKey::Ui { size, style })
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
        let face = fonts.face_for(FaceKey::Prose {
            size: 32,
            style: FontStyle::Regular,
        });
        assert!(
            face.cluster_glyph("😀", 32)
                .is_some_and(|g| g.advance > 0.0),
            "a smiley routes to the emoji path"
        );
        assert!(
            face.cluster_glyph("a", 32).is_none(),
            "ASCII never routes to the emoji path"
        );
    }

    /// Characters the Claude CLI prints that monospace fonts often have no cell for.
    /// Which of them the grid's family lacks depends on what `monospace` is here.
    const HOMELESS: [char; 6] = [
        '\u{23FA}', '\u{23BF}', '\u{2713}', '\u{2717}', '\u{273B}', '\u{2801}',
    ];

    #[test]
    fn the_system_draws_what_neither_the_family_nor_the_pinned_fonts_can() {
        let fonts = Fonts::new(&[16]).expect("a default font");
        let key = FaceKey::Prose {
            size: 16,
            style: FontStyle::Regular,
        };
        let primary = fonts.face_for(key);

        let missing: Vec<char> = HOMELESS
            .into_iter()
            .filter(|&ch| !primary.has_scalar(ch))
            .collect();
        assert!(
            !missing.is_empty(),
            "the grid family draws every sample, so fallback goes untested; add one it lacks"
        );

        for ch in missing {
            let face = fonts.glyph_face(key, ch);
            assert!(
                face.has_scalar(ch),
                "no font on this system draws {ch:?} (U+{:04X})",
                ch as u32
            );
            // Resolving the glyph is not enough; it has to rasterize to actual ink. A
            // face that reports the glyph but rasterizes empty is a disguised tofu box.
            let raster = face.rasterize(ch);
            assert!(
                raster.coverage.iter().any(|&c| c > 0),
                "{ch:?} resolved to a face that inks nothing"
            );
        }
    }

    #[test]
    fn fallback_discovery_is_memoized_and_stays_idle_for_latin() {
        let fonts = Fonts::new(&[16]).expect("a default font");
        let key = FaceKey::Prose {
            size: 16,
            style: FontStyle::Regular,
        };

        // Latin resolves before the ranked fallback set is needed.
        let _ = fonts.glyph_face(key, 'A');
        assert!(
            fonts.fc.get().is_none(),
            "an ASCII glyph built the fallback set"
        );

        // The same character twice hands back the same face, from the memo rather than a
        // second match: a screenful of one symbol must not re-query per cell.
        let first = fonts.glyph_face(key, HOMELESS[0]);
        let second = fonts.glyph_face(key, HOMELESS[0]);
        assert!(
            Rc::ptr_eq(&first, &second),
            "the same character resolved twice to two different faces"
        );
        assert!(fonts.fc.get().is_some(), "discovery never woke the system");
    }

    #[test]
    fn the_discovered_memo_stays_bounded_when_the_child_sweeps_unknown_characters() {
        // `discovered` is keyed on scalars the child chooses and nothing ever evicted, so
        // `cat`ing a file that sweeps the unassigned code points walked it toward ~1.1 M
        // entries (~35 MB) and left them there for the life of the process. Cheap per
        // miss, which is why it is growth rather than a stall — and why nobody notices.
        //
        // Plane 4 is entirely unassigned, so nothing installed draws any of these: every
        // character is a fresh miss that memoizes a `None`, which is the worst case.
        let fonts = Fonts::new(&[16]).expect("a default font");
        let key = FaceKey::Prose {
            size: 16,
            style: FontStyle::Regular,
        };
        let sweep = MAX_DISCOVERED as u32 + MAX_DISCOVERED as u32 / 2;
        for cp in 0x4_0000..0x4_0000 + sweep {
            if let Some(ch) = char::from_u32(cp) {
                let _ = fonts.glyph_face(key, ch);
            }
        }
        let held = fonts.entry(16).discovered.borrow().len();
        assert!(
            held <= MAX_DISCOVERED,
            "the memo held {held} entries, past its {MAX_DISCOVERED} cap"
        );
        assert!(held > 0, "and it is still a cache, not a disabled one");
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
        // the common path never walks the fallback chain and never wakes fontconfig.
        let primary = fonts.face_for(key);
        assert!(primary.has_scalar('A'));
        assert!(std::ptr::eq(
            Rc::as_ptr(&fonts.glyph_face(key, 'A')),
            primary as *const Face
        ));

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

    /// Discovery opens what the system offers at the cell's pixel size, so it must only
    /// be offered fonts that open at one. A colour emoji font does not: its glyphs are
    /// fixed-size bitmaps, though fontconfig calls it scalable. An interface face carries
    /// no emoji font, so a tab title's emoji asks discovery for one.
    #[test]
    fn every_font_offered_for_a_character_opens_at_a_pixel_size() {
        let fc = Fontconfig::new().expect("fontconfig");
        for ch in ['😀', '🎉', '❤', '✔', '中', 'ж', '\u{F418}'] {
            if let Some(file) = fc.font_for_char(ch) {
                let face = Face::from_file(&file).expect("the offered font reads");
                assert!(
                    face.set_pixel_size(16).is_ok(),
                    "{ch} offered {}, which does not open at 16 px",
                    file.path.display()
                );
            }
        }
    }
}
