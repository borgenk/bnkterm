//! Text measurement and cluster shaping.
//!
//! [`text_advance`] measures a run's pen advance from cached per-glyph metrics
//! (no rasterization). It routes ASCII and simple clusters straight to the
//! per-character caches; only a multi-scalar emoji cluster reaches the shaper
//! below.
//!
//! The shaper is backed by the system HarfBuzz (libharfbuzz.so). A terminal
//! renders almost all text one glyph per Unicode scalar and never touches it;
//! shaping is reserved for the one case that genuinely needs it: resolving a
//! multi-scalar emoji cluster (a ZWJ family, a flag's regional-indicator pair, a
//! skin-tone or keycap sequence) into the single ligature glyph the emoji font's
//! GSUB table defines, which FreeType alone cannot do.
//!
//! The HarfBuzz font is bound to the already-open FreeType face with
//! `hb_ft_font_create_referenced`, so glyph ids, metrics, and font tables all
//! come from the same font at the same selected size, and positions come back
//! in 26.6 fixed-point pixels at that size. HarfBuzz never returns null from
//! its constructors (failure yields an inert empty object that shapes to
//! nothing), so every error path here degrades to "no glyphs", which callers
//! treat as "fall back to per-character rendering".

use core::ffi::{c_char, c_int, c_uint, c_void};
use core::ptr::{self, NonNull};

use crate::platform::freetype::Face;
use crate::platform::grapheme;

/// Total pen advance of `text` at the face's current pixel size, in pixels.
/// Measures with metrics only (no rasterization), matching how the renderer
/// advances the pen, so a measured width and the drawn glyphs agree.
///
/// This is the *proportional* measure, for text placed by its own advances rather
/// than on the grid's fixed pitch — a chrome label, never terminal text, which is
/// measured by [`crate::width`] and placed a cell at a time. `render::gpu`'s glyph
/// routing deliberately mirrors this one, so anything measured here draws where it
/// was measured.
///
/// ASCII skips cluster segmentation entirely; other text measures per grapheme
/// cluster so an emoji cluster gets its color glyph's advance, routed identically
/// to how the renderer draws it.
pub fn text_advance(face: &Face, text: &str) -> f32 {
    if text.is_ascii() {
        return text.chars().map(|c| face.advance(c)).sum();
    }
    grapheme::graphemes(text)
        .map(|(_, cluster)| face.cluster_advance(cluster))
        .sum()
}

/// One glyph of a shaped cluster: the font glyph id and its pen placement in
/// pixels at the face's selected size (already divided down from 26.6).
pub struct ShapedGlyph {
    pub id: u32,
    pub x_advance: f32,
    pub x_offset: f32,
    pub y_offset: f32,
}

// ---------------------------------------------------------------------------
// FFI surface. Mirrors of hb_glyph_info_t / hb_glyph_position_t; both structs
// have been ABI-stable since HarfBuzz 1.0.
// ---------------------------------------------------------------------------

#[repr(C)]
#[allow(dead_code)]
struct HbGlyphInfo {
    codepoint: c_uint,
    mask: c_uint,
    cluster: c_uint,
    var1: c_uint,
    var2: c_uint,
}

#[repr(C)]
#[allow(dead_code)]
struct HbGlyphPosition {
    x_advance: c_int,
    y_advance: c_int,
    x_offset: c_int,
    y_offset: c_int,
    var: c_uint,
}

#[link(name = "harfbuzz")]
extern "C" {
    fn hb_ft_font_create_referenced(ft_face: *mut c_void) -> *mut c_void;
    fn hb_font_destroy(font: *mut c_void);
    fn hb_buffer_create() -> *mut c_void;
    fn hb_buffer_destroy(buffer: *mut c_void);
    fn hb_buffer_add_utf8(
        buffer: *mut c_void,
        text: *const c_char,
        text_length: c_int,
        item_offset: c_uint,
        item_length: c_int,
    );
    fn hb_buffer_guess_segment_properties(buffer: *mut c_void);
    fn hb_shape(
        font: *mut c_void,
        buffer: *mut c_void,
        features: *const c_void,
        num_features: c_uint,
    );
    fn hb_buffer_get_glyph_infos(buffer: *mut c_void, length: *mut c_uint) -> *const HbGlyphInfo;
    fn hb_buffer_get_glyph_positions(
        buffer: *mut c_void,
        length: *mut c_uint,
    ) -> *const HbGlyphPosition;
}

/// A HarfBuzz font handle over a FreeType face, used to shape one cluster at a
/// time. Dropping it releases only HarfBuzz's own reference; the FreeType face
/// is independently referenced (`_referenced`), so drop order with the owning
/// face does not matter.
pub struct Shaper {
    font: NonNull<c_void>,
}

impl Shaper {
    /// Bind a shaper to a FreeType face handle.
    ///
    /// # Safety
    /// `ft_face` must be a valid `FT_Face` with a size selected. HarfBuzz takes
    /// its own reference, so the face may be freed before or after the shaper.
    pub unsafe fn from_ft_face(ft_face: *mut c_void) -> Option<Self> {
        let font = hb_ft_font_create_referenced(ft_face);
        Some(Self {
            font: NonNull::new(font)?,
        })
    }

    /// Shape `text` (one grapheme cluster) into positioned glyphs. An empty
    /// result means the buffer could not be built; a `.notdef` (id 0) entry
    /// means the font cannot form this cluster. Callers treat both as "use the
    /// per-character path instead".
    pub fn shape(&self, text: &str) -> Vec<ShapedGlyph> {
        if text.is_empty() || text.len() > c_int::MAX as usize {
            return Vec::new();
        }
        // SAFETY: the buffer is created and destroyed in this scope; text
        // pointer/length describe a live UTF-8 slice; infos/positions point at
        // buffer-owned arrays of the returned length, read before destroy.
        unsafe {
            let buffer = hb_buffer_create();
            if buffer.is_null() {
                return Vec::new();
            }
            hb_buffer_add_utf8(
                buffer,
                text.as_ptr() as *const c_char,
                text.len() as c_int,
                0,
                text.len() as c_int,
            );
            hb_buffer_guess_segment_properties(buffer);
            hb_shape(self.font.as_ptr(), buffer, ptr::null(), 0);

            let mut n_info: c_uint = 0;
            let mut n_pos: c_uint = 0;
            let infos = hb_buffer_get_glyph_infos(buffer, &mut n_info);
            let positions = hb_buffer_get_glyph_positions(buffer, &mut n_pos);
            let n = n_info.min(n_pos) as usize;
            let mut out = Vec::with_capacity(n);
            if !infos.is_null() && !positions.is_null() {
                for i in 0..n {
                    let info = &*infos.add(i);
                    let pos = &*positions.add(i);
                    out.push(ShapedGlyph {
                        id: info.codepoint,
                        x_advance: pos.x_advance as f32 / 64.0,
                        x_offset: pos.x_offset as f32 / 64.0,
                        y_offset: pos.y_offset as f32 / 64.0,
                    });
                }
            }
            hb_buffer_destroy(buffer);
            out
        }
    }
}

impl Drop for Shaper {
    fn drop(&mut self) {
        // SAFETY: the handle came from hb_ft_font_create_referenced and is
        // destroyed exactly once.
        unsafe { hb_font_destroy(self.font.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_advance_is_zero_empty_and_grows_with_length() {
        let face = Face::open_default(32).expect("a default font");
        assert_eq!(text_advance(&face, ""), 0.0);
        let one = text_advance(&face, "m");
        let two = text_advance(&face, "mm");
        assert!(one > 0.0);
        assert!(two > one, "two glyphs advance further than one");
    }

    #[test]
    fn text_advance_ascii_fast_path_matches_the_cluster_path() {
        let face = Face::open_default(32).expect("a default font");
        let text = "Hello, world 123 <> []";
        let fast = text_advance(&face, text);
        let clustered: f32 = grapheme::graphemes(text)
            .map(|(_, cluster)| face.cluster_advance(cluster))
            .sum();
        assert_eq!(
            fast, clustered,
            "the ASCII shortcut must not change measurement"
        );
    }
}
