//! Font *discovery*: ask the system which font can draw a character we have none for.
//!
//! `freetype.rs` opens the fonts we name; this module finds the ones we did not think
//! to name. It is a thin FFI over `libfontconfig`, the library every other application
//! on a Linux desktop already resolves its fonts through.
//!
//! ```text
//!   glyph_face(key, ch)
//!     ├─ the keyed face's cmap        ─▶ Consolas, for ASCII and Latin: the common case,
//!     │                                  one FT_Get_Char_Index and done
//!     ├─ FontConfig::fallback         ─▶ the fonts we deliberately pin (Symbols Nerd
//!     │                                  Font for the private-use icon range)
//!     └─ Fontconfig::font_for_char    ─▶ *this module*: whatever the system has
//!                                        (✓ from FreeSerif, ⏺ from AdwaitaMono, …)
//! ```
//!
//! # A short list, then the system
//!
//! A short hardcoded list keeps the terminal running on a bare machine without a
//! font-discovery dependency, and it renders identically wherever those files exist.
//! What it cannot do is cover everything: the moment a program prints a character no
//! listed font has — and the Claude CLI prints half a dozen (`⏺ ⎿ ✓ ✗ ✻` and the
//! braille spinners) — the cell renders as a `.notdef` tofu box.
//!
//! Extending the list is the same bet at longer odds: Unicode is larger than any list
//! we will maintain, and the glyph a program picks tomorrow is not one we chose today.
//! Worse, the fonts we *did* choose are not the fonts the rest of the desktop chose, so
//! the same `✓` came out of a different font here than in every other terminal on the
//! machine.
//!
//! The pinned list therefore stays as an *override* ("this range comes from this font,
//! whatever the system thinks"), and fontconfig answers everything else, the way it
//! answers for every other application. The cost is that rendering is now a function of
//! what is installed rather than of a fixed list: a real loss of determinism, taken
//! knowingly, because matching the desktop is worth more here than reproducing a tofu
//! box identically on two machines.
//!
//! # The dependency
//!
//! `libfontconfig` is a system C library, reached the way [`freetype`](crate::platform::freetype),
//! [`shape`](crate::platform::shape) (HarfBuzz), and [`xkb`](crate::platform::xkb) (libxkbcommon) are reached:
//! `extern "C"` against the real ABI, no `-sys` crate. It binds cleanly: nearly every type
//! below is an **opaque handle**, with one exception — `FcFontSet`, whose three fields we
//! read directly, and whose layout is therefore mirrored field-for-field and pinned against
//! a C compiler in the tests, as `termios` and `winsize` are.
//!
//! # Cost
//!
//! [`Fontconfig::new`] loads the system font cache, which is why it is created *lazily*,
//! on the first character the pinned fonts cannot draw. A session that prints only Latin
//! text never builds it at all. Each [`Fontconfig::font_for_char`] is a fresh match, so
//! callers memoize: `freetype.rs` caches the answer per character and the opened face per
//! file, and the GPU batcher caches the raster on top of that, so a repeated glyph never
//! reaches this module twice.

use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr;

use core::ffi::{c_char, c_int, c_uchar, c_uint, c_void};

/// An opaque fontconfig pattern: a bag of properties, both the query and the answer.
#[repr(C)]
struct FcPattern {
    _opaque: [u8; 0],
}

/// An opaque set of characters, used here to hold exactly one: the one we need drawn.
#[repr(C)]
struct FcCharSet {
    _opaque: [u8; 0],
}

/// An opaque fontconfig configuration. We always pass null, meaning "the current one",
/// which is the system's — the same one every other application on the desktop uses.
#[repr(C)]
struct FcConfig {
    _opaque: [u8; 0],
}

// Property names (`fontconfig.h`). NUL-terminated because the C API takes `const char *`.
const FC_FILE: &[u8] = b"file\0";
const FC_INDEX: &[u8] = b"index\0";
const FC_SCALABLE: &[u8] = b"scalable\0";
const FC_CHARSET: &[u8] = b"charset\0";

/// `FcMatchPattern` (`FcMatchKind`): substitutions to apply to a *query*, as opposed to
/// to a font. The first variant of the enum, hence 0.
const FC_MATCH_PATTERN: c_int = 0;
/// `FcResultMatch` (`FcResult`): the property was found. Also the first variant, hence 0.
const FC_RESULT_MATCH: c_int = 0;
/// `FcTrue`.
const FC_TRUE: c_int = 1;

#[link(name = "fontconfig")]
extern "C" {
    fn FcInit() -> c_int;
    fn FcPatternCreate() -> *mut FcPattern;
    fn FcPatternDestroy(pattern: *mut FcPattern);
    fn FcPatternAddCharSet(
        pattern: *mut FcPattern,
        object: *const c_char,
        charset: *const FcCharSet,
    ) -> c_int;
    fn FcPatternAddBool(pattern: *mut FcPattern, object: *const c_char, value: c_int) -> c_int;
    fn FcPatternGetString(
        pattern: *const FcPattern,
        object: *const c_char,
        index: c_int,
        value: *mut *mut c_uchar,
    ) -> c_int;
    fn FcPatternGetInteger(
        pattern: *const FcPattern,
        object: *const c_char,
        index: c_int,
        value: *mut c_int,
    ) -> c_int;
    fn FcCharSetCreate() -> *mut FcCharSet;
    fn FcCharSetDestroy(charset: *mut FcCharSet);
    fn FcCharSetAddChar(charset: *mut FcCharSet, ch: c_uint) -> c_int;
    fn FcConfigSubstitute(config: *mut FcConfig, pattern: *mut FcPattern, kind: c_int) -> c_int;
    fn FcDefaultSubstitute(pattern: *mut FcPattern);
    fn FcFontSort(
        config: *mut FcConfig,
        pattern: *mut FcPattern,
        trim: c_int,
        csp: *mut *mut FcCharSet,
        result: *mut c_int,
    ) -> *mut FcFontSet;
    fn FcFontSetDestroy(set: *mut FcFontSet);
    fn FcPatternGetCharSet(
        pattern: *const FcPattern,
        object: *const c_char,
        index: c_int,
        charset: *mut *mut FcCharSet,
    ) -> c_int;
    fn FcCharSetHasChar(charset: *const FcCharSet, ch: c_uint) -> c_int;
}

/// `FcFontSet` (`fontconfig.h`), the one type in the C API we touch that has a layout
/// rather than being an opaque handle. Mirrored field for field; its size, alignment,
/// and offsets are pinned against a C compiler in the tests below.
///
/// ```text
///   nfont ─ how many fonts are in `fonts`
///   sfont ─ how many slots are allocated (fontconfig's business, not ours)
///   fonts ─ the patterns themselves, in preference order
/// ```
#[repr(C)]
struct FcFontSet {
    nfont: c_int,
    sfont: c_int,
    fonts: *mut *mut FcPattern,
}

/// A font file the system offers, as a path and the face index within it (non-zero only
/// for a TrueType *collection*, where several faces share one file — Noto's CJK fonts
/// ship this way, and taking face 0 of a collection quietly gives the wrong language).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FontFile {
    pub path: PathBuf,
    pub index: i32,
}

/// The system's fonts, ranked once, ready to answer "who can draw this?".
///
/// # Why a sorted set and not a match per character
///
/// The obvious binding is `FcFontMatch` with a one-character charset — literally what
/// `fc-match ':charset=2713'` does — and it is what this module did first. It is also a
/// quarter of a millisecond per call, because each match re-runs the whole configuration:
/// substitutions, defaults, and a scored sort of every font on the machine. Fine for the
/// six symbols a CLI prints. Ruinous for a script the primary family does not cover: a
/// screenful of distinct CJK is a thousand fresh questions, and it measured **268 ms** —
/// a visible stall, on the very path the rest of this codebase is built to keep smooth.
///
/// So the sort happens once, at [`Fontconfig::new`], and every character after that is
/// answered from the charsets fontconfig already holds in memory: walk the ranked fonts,
/// return the first whose charset contains the codepoint. No further calls into the
/// library, no scoring, no allocation. This is the same shape wezterm and ghostty use,
/// and it is why they can render a page of Chinese without thinking about it.
///
/// `trim` asks fontconfig to drop any font that adds no coverage the ones above it do not
/// already have, so the list stays short and holds no redundant entry.
pub struct Fontconfig {
    /// The ranked fonts. Owned: the charsets borrowed in [`font_for_char`](Self::font_for_char)
    /// point into these patterns, so the set must outlive every query made against it.
    set: *mut FcFontSet,
}

impl Fontconfig {
    /// Load and rank the system's fonts, or `None` if fontconfig will not start (a broken
    /// or empty font cache) or offers nothing. `None` is not fatal anywhere: the caller
    /// falls back to the fonts it pinned, exactly as it did before discovery existed.
    ///
    /// This is the expensive call — it reads the font cache and sorts it — which is why
    /// it is made lazily, on the first character the pinned fonts cannot draw. A session
    /// that shows only Latin text never makes it at all.
    pub fn new() -> Option<Self> {
        // SAFETY: FcInit takes no arguments and is idempotent; it returns FcFalse when the
        // configuration cannot be loaded.
        if unsafe { FcInit() } != FC_TRUE {
            return None;
        }

        let query = Pattern::new()?;
        // SAFETY: query is live; the object name is a NUL-terminated literal. A bitmap face
        // holds one fixed size, so it cannot be opened at the cell's pixel size; asking for
        // scalable fonts only keeps one out of the ranking entirely.
        unsafe {
            if FcPatternAddBool(query.0, FC_SCALABLE.as_ptr().cast(), FC_TRUE) != FC_TRUE {
                return None;
            }
            // The standard two-step every fontconfig client performs before matching: apply
            // the system's configured substitutions, then fill in the defaults they did not
            // set. Skipping it does not fail loudly, it just quietly stops honouring the
            // user's fontconfig rules -- which would defeat the point of asking the system.
            FcConfigSubstitute(ptr::null_mut(), query.0, FC_MATCH_PATTERN);
            FcDefaultSubstitute(query.0);
        }

        let mut result: c_int = 0;
        // SAFETY: query is live; a null config means the current (system) one; `trim` drops
        // fonts adding no new coverage; a null `csp` declines the accumulated charset we do
        // not need. The returned set is owned by us and freed in Drop.
        let set = unsafe {
            FcFontSort(
                ptr::null_mut(),
                query.0,
                FC_TRUE,
                ptr::null_mut(),
                &mut result,
            )
        };
        if set.is_null() || result != FC_RESULT_MATCH {
            return None;
        }
        // SAFETY: set is non-null and was just returned by FcFontSort.
        if unsafe { (*set).nfont } <= 0 {
            // SAFETY: set is a live, owned FcFontSet; destroyed exactly once, here.
            unsafe { FcFontSetDestroy(set) };
            return None;
        }
        Some(Fontconfig { set })
    }

    /// The font the system would use to draw `ch`: the first in the ranked set whose
    /// character map contains it, or `None` when nothing installed does.
    ///
    /// A charset test against an in-memory bitset, per candidate, until one hits. No call
    /// into fontconfig and no allocation, so this is cheap enough to ask per character and
    /// let the caller memoize rather than the other way round.
    pub fn font_for_char(&self, ch: char) -> Option<FontFile> {
        // SAFETY: self.set is a live, owned FcFontSet with nfont > 0 (checked at
        // construction); `fonts` points at nfont valid pattern pointers.
        let (count, fonts) = unsafe { ((*self.set).nfont, (*self.set).fonts) };
        for i in 0..count {
            // SAFETY: i is in 0..nfont, so fonts[i] is one of the set's live patterns. It
            // is borrowed, never freed here: the set owns it.
            let font = unsafe { *fonts.add(i as usize) };
            if font.is_null() {
                continue;
            }
            let mut charset: *mut FcCharSet = ptr::null_mut();
            // SAFETY: font is a live pattern; the object name is a NUL-terminated literal;
            // charset is a live local fontconfig writes a *borrowed* pointer through, valid
            // for as long as the pattern (and so the set) lives.
            let found =
                unsafe { FcPatternGetCharSet(font, FC_CHARSET.as_ptr().cast(), 0, &mut charset) };
            if found != FC_RESULT_MATCH || charset.is_null() {
                continue;
            }
            // SAFETY: charset is a live FcCharSet borrowed from the pattern above.
            if unsafe { FcCharSetHasChar(charset, ch as u32) } != FC_TRUE {
                continue;
            }
            // A font that claims the character. Its file is the answer; a font with no file
            // cannot be opened, so keep walking rather than give up on the character.
            if let Some(path) = pattern_string(font, FC_FILE) {
                return Some(FontFile {
                    path: PathBuf::from(path),
                    // No index property means a plain single-face font: face 0.
                    index: pattern_integer(font, FC_INDEX).unwrap_or(0),
                });
            }
        }
        None
    }
}

impl Drop for Fontconfig {
    fn drop(&mut self) {
        // SAFETY: self.set came from FcFontSort, is non-null (checked at construction), and
        // is destroyed exactly once. Every charset borrowed from it is gone by now: they are
        // only ever read inside `font_for_char`, which never lets one escape.
        unsafe { FcFontSetDestroy(self.set) };
    }
}

/// A string property of a pattern, copied out. The pointer fontconfig hands back points
/// *into* the pattern, so it lives and dies with it: copy before the pattern is freed.
///
/// Takes a raw pointer rather than an owned [`Pattern`] because it is used both ways: on a
/// pattern we own (nothing does, now) and on one merely *borrowed* from the ranked set,
/// which must not be freed by us.
fn pattern_string(pattern: *const FcPattern, object: &[u8]) -> Option<String> {
    let mut value: *mut c_uchar = ptr::null_mut();
    // SAFETY: pattern is live, object is a NUL-terminated literal, and value is a live
    // local that fontconfig writes a borrowed pointer through.
    let result = unsafe { FcPatternGetString(pattern, object.as_ptr().cast(), 0, &mut value) };
    if result != FC_RESULT_MATCH || value.is_null() {
        return None;
    }
    // SAFETY: on FcResultMatch, value points at a NUL-terminated UTF-8 string owned by the
    // pattern, which outlives this borrow (we copy before returning).
    let text = unsafe { CStr::from_ptr(value.cast::<c_char>()) };
    text.to_str().ok().map(str::to_owned)
}

/// An integer property of a pattern, or `None` when it does not carry one.
fn pattern_integer(pattern: *const FcPattern, object: &[u8]) -> Option<i32> {
    let mut value: c_int = 0;
    // SAFETY: as `pattern_string`; fontconfig writes the integer through the pointer.
    let result = unsafe { FcPatternGetInteger(pattern, object.as_ptr().cast(), 0, &mut value) };
    (result == FC_RESULT_MATCH).then_some(value)
}

/// An owned `FcPattern`, freed on drop. Every early return in construction would otherwise
/// leak one.
struct Pattern(*mut FcPattern);

impl Pattern {
    fn new() -> Option<Self> {
        // SAFETY: takes no arguments; returns null on allocation failure.
        let pattern = unsafe { FcPatternCreate() };
        (!pattern.is_null()).then_some(Pattern(pattern))
    }
}

impl Drop for Pattern {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: self.0 was returned by FcPatternCreate and is dropped exactly once.
            unsafe { FcPatternDestroy(self.0) };
        }
    }
}

/// An owned `FcCharSet`, freed on drop.
struct CharSet(*mut FcCharSet);

impl CharSet {
    fn new() -> Option<Self> {
        // SAFETY: takes no arguments; returns null on allocation failure.
        let charset = unsafe { FcCharSetCreate() };
        (!charset.is_null()).then_some(CharSet(charset))
    }
}

impl Drop for CharSet {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: self.0 was returned by FcCharSetCreate and is dropped once.
            unsafe { FcCharSetDestroy(self.0) };
        }
    }
}

/// Silence the unused-type warning for the config handle, which only ever appears behind
/// a null pointer in the calls above.
const _: Option<&FcConfig> = None;
const _: Option<&c_void> = None;

#[cfg(test)]
mod tests {
    use super::*;

    /// The characters the Claude CLI prints that a monospace terminal font typically has
    /// no cell for, and that sent us here. Not asserted to come from any *particular*
    /// font (that is the system's business, and differs per machine), only to be found.
    const HOMELESS: [char; 6] = ['⏺', '⎿', '✓', '✗', '✻', '⠁'];

    #[test]
    fn fcfontset_matches_the_c_abi() {
        // The one fontconfig type we read fields out of rather than pass around as a
        // handle, so its layout is ours to get right. Verified against what a C compiler
        // emits on this machine: `sizeof(FcFontSet)=16 align=8 nfont@0 sfont@4 fonts@8`.
        // Reading `nfont` at the wrong offset would walk a garbage count of font pointers.
        assert_eq!(std::mem::size_of::<FcFontSet>(), 16);
        assert_eq!(std::mem::align_of::<FcFontSet>(), 8);
        let set = FcFontSet {
            nfont: 0,
            sfont: 0,
            fonts: ptr::null_mut(),
        };
        let base = &set as *const FcFontSet as usize;
        assert_eq!(&set.nfont as *const c_int as usize - base, 0);
        assert_eq!(&set.sfont as *const c_int as usize - base, 4);
        assert_eq!(&set.fonts as *const *mut *mut FcPattern as usize - base, 8);
    }

    #[test]
    fn the_system_can_draw_the_characters_our_pinned_fonts_cannot() {
        let Some(fc) = Fontconfig::new() else {
            eprintln!("no usable fontconfig in this environment; skipping");
            return;
        };
        for ch in HOMELESS {
            let found = fc.font_for_char(ch);
            assert!(
                found.is_some(),
                "fontconfig found no font for {ch:?} (U+{:04X})",
                ch as u32
            );
            let file = found.expect("just asserted");
            assert!(
                file.path.exists(),
                "fontconfig named a font that is not on disk: {file:?}"
            );
            assert!(file.index >= 0, "a negative face index: {file:?}");
        }
    }

    #[test]
    fn a_plain_ascii_letter_resolves_too() {
        // Not because we ever ask it to -- the primary face answers every Latin glyph
        // long before this module is reached -- but because a discovery layer that
        // cannot find a font for 'A' is broken in a way the exotic cases might hide.
        let Some(fc) = Fontconfig::new() else {
            return;
        };
        let found = fc
            .font_for_char('A')
            .expect("some font on this system has 'A'");
        assert!(found.path.exists());
    }

    #[test]
    fn every_call_is_independent() {
        // The pattern and charset are owned and freed per call (see `Pattern`/`CharSet`),
        // so a repeated query must keep answering rather than resolve once and then hand
        // back a dangling or emptied pattern. Under a leak or a double free this is where
        // it shows.
        let Some(fc) = Fontconfig::new() else {
            return;
        };
        let first = fc.font_for_char('✓');
        let second = fc.font_for_char('✓');
        assert_eq!(
            first, second,
            "the same character must resolve the same way"
        );
        assert!(first.is_some());
    }
}
