//! Font discovery through `libfontconfig`.
//!
//! This module resolves configured family names to files and finds fallback faces for
//! characters the selected fonts cannot draw.
//!
//! ```text
//!   glyph_face(key, ch)
//!     ├─ selected face cmap         ─▶ configured grid or UI font
//!     ├─ FontConfig::fallback       ─▶ configured symbol fonts
//!     └─ Fontconfig::font_for_char  ─▶ installed system fonts
//! ```
//!
//! Family queries apply the current fontconfig configuration. Fallback discovery ranks
//! scalable fonts once, then scans their retained character maps. `freetype.rs` memoizes
//! the selected face per character and opens each discovered file once per size.
//!
//! All C types are opaque except `FcFontSet`, whose layout is pinned against a C compiler
//! in the tests.

use std::ffi::{CStr, CString, OsString};
use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;
use std::ptr;

use core::ffi::{c_char, c_int, c_uchar, c_uint, c_void};

use crate::platform::freetype::FontStyle;

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
/// which is the system's — the same one every other application uses.
#[repr(C)]
struct FcConfig {
    _opaque: [u8; 0],
}

// Property names (`fontconfig.h`). NUL-terminated because the C API takes `const char *`.
const FC_FILE: &[u8] = b"file\0";
const FC_INDEX: &[u8] = b"index\0";
const FC_SCALABLE: &[u8] = b"scalable\0";
const FC_CHARSET: &[u8] = b"charset\0";
const FC_FAMILY: &[u8] = b"family\0";
const FC_WEIGHT: &[u8] = b"weight\0";
const FC_SLANT: &[u8] = b"slant\0";

// Fontconfig's weight and slant values from fontconfig.h.
const FC_WEIGHT_REGULAR: c_int = 80;
const FC_WEIGHT_MEDIUM: c_int = 100;
const FC_WEIGHT_BOLD: c_int = 200;
const FC_SLANT_ROMAN: c_int = 0;
const FC_SLANT_ITALIC: c_int = 100;

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
    fn FcPatternAddString(
        pattern: *mut FcPattern,
        object: *const c_char,
        value: *const c_uchar,
    ) -> c_int;
    fn FcPatternAddInteger(pattern: *mut FcPattern, object: *const c_char, value: c_int) -> c_int;
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
    fn FcFontMatch(
        config: *mut FcConfig,
        pattern: *mut FcPattern,
        result: *mut c_int,
    ) -> *mut FcPattern;
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

/// A resolved font path and its FreeType face index.
///
/// The index may select a face in a collection or a named variable-font instance.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FontFile {
    pub path: PathBuf,
    pub index: i32,
}

impl FontFile {
    /// Face zero of `path`.
    pub fn at(path: &str) -> Self {
        Self {
            path: PathBuf::from(path),
            index: 0,
        }
    }
}

/// A matched file and the family names fontconfig assigns to it.
pub(super) struct FamilyMatch {
    pub file: FontFile,
    pub families: Vec<String>,
}

const GENERIC_FAMILIES: [&str; 5] = ["monospace", "sans-serif", "serif", "system-ui", "emoji"];
const UNCONFIGURED_FAMILY: &str = "__bnkterm_unconfigured_family__";

/// Whether `family` names a role rather than a typeface.
fn is_generic(family: &str) -> bool {
    GENERIC_FAMILIES
        .iter()
        .any(|generic| generic.eq_ignore_ascii_case(family))
}

/// Resolve `family_name` and `style` through the current fontconfig rules.
///
/// Generic aliases accept the selected family. Named families return `None` when the
/// matched pattern does not list that name, allowing the next configured candidate to run.
pub fn font_for_family(family_name: &str, style: FontStyle) -> Option<FontFile> {
    match_family(family_name, style).map(|matched| matched.file)
}

/// Resolve a family while retaining the matched family names for style checks.
pub(super) fn match_family(family_name: &str, style: FontStyle) -> Option<FamilyMatch> {
    // SAFETY: FcInit takes no arguments and is idempotent; it returns FcFalse when the
    // configuration cannot be loaded.
    if unsafe { FcInit() } != FC_TRUE {
        return None;
    }

    let (weight, slant) = weight_and_slant(style);
    let (query, configured_families) = configured_family_query(family_name, weight, slant)?;
    // SAFETY: query is live and has already received configured substitutions.
    unsafe { FcDefaultSubstitute(query.0) };

    let mut result: c_int = 0;
    // SAFETY: query is live and substituted; a null config means the current (system)
    // one; result is a live local. The returned pattern is *owned* by us, which is what
    // the Pattern wrapper is for -- unlike the borrowed patterns inside a sorted set.
    let matched = unsafe { FcFontMatch(ptr::null_mut(), query.0, &mut result) };
    if matched.is_null() || result != FC_RESULT_MATCH {
        return None;
    }
    let matched = Pattern(matched);

    let families = pattern_strings(matched.0, FC_FAMILY);
    if !is_generic(family_name)
        && !families
            .iter()
            .any(|name| name.eq_ignore_ascii_case(family_name))
        && !configured_alias_matches(family_name, weight, slant, &configured_families, &families)
    {
        return None;
    }

    Some(FamilyMatch {
        file: FontFile {
            path: pattern_path(matched.0, FC_FILE)?,
            // No index property means a plain single-face font: face 0.
            index: pattern_integer(matched.0, FC_INDEX).unwrap_or(0),
        },
        families,
    })
}

/// A family query after configured substitutions but before default properties.
fn configured_family_query(
    family_name: &str,
    weight: c_int,
    slant: c_int,
) -> Option<(Pattern, Vec<String>)> {
    // A family name with an interior NUL is not a family any font has.
    let family = CString::new(family_name).ok()?;
    let query = Pattern::new()?;
    // SAFETY: query is live; the object names are NUL-terminated literals and `family` is
    // a live CString that fontconfig copies out of rather than borrows.
    unsafe {
        if FcPatternAddString(query.0, FC_FAMILY.as_ptr().cast(), family.as_ptr().cast()) != FC_TRUE
        {
            return None;
        }
        FcPatternAddInteger(query.0, FC_WEIGHT.as_ptr().cast(), weight);
        FcPatternAddInteger(query.0, FC_SLANT.as_ptr().cast(), slant);
        FcConfigSubstitute(ptr::null_mut(), query.0, FC_MATCH_PATTERN);
    }
    let families = pattern_strings(query.0, FC_FAMILY);
    Some((query, families))
}

/// Whether configured family substitutions introduced the family that won the match.
fn configured_alias_matches(
    requested: &str,
    weight: c_int,
    slant: c_int,
    configured: &[String],
    matched: &[String],
) -> bool {
    let Some((_, baseline)) = configured_family_query(UNCONFIGURED_FAMILY, weight, slant) else {
        return false;
    };
    alias_targets(requested, configured, &baseline)
        .iter()
        .any(|target| {
            matched
                .iter()
                .any(|family| family.eq_ignore_ascii_case(target))
        })
}

/// Families added or reordered specifically for `requested`, excluding global defaults.
fn alias_targets<'a>(
    requested: &str,
    configured: &'a [String],
    baseline: &[String],
) -> Vec<&'a str> {
    let configured: Vec<&str> = configured
        .iter()
        .map(String::as_str)
        .filter(|family| !family.eq_ignore_ascii_case(requested))
        .collect();
    let baseline: Vec<&str> = baseline
        .iter()
        .map(String::as_str)
        .filter(|family| !family.eq_ignore_ascii_case(UNCONFIGURED_FAMILY))
        .collect();
    let mut unmatched = baseline.clone();
    let mut targets = Vec::new();
    for family in &configured {
        if let Some(index) = unmatched
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(family))
        {
            unmatched.remove(index);
        } else {
            targets.push(*family);
        }
    }
    let same_order = configured.len() == baseline.len()
        && configured
            .iter()
            .zip(&baseline)
            .all(|(left, right)| left.eq_ignore_ascii_case(right));
    if targets.is_empty() && !same_order {
        if let Some(first) = configured.first() {
            targets.push(*first);
        }
    }
    targets
}

/// Map a terminal style to fontconfig's weight and slant scales.
fn weight_and_slant(style: FontStyle) -> (c_int, c_int) {
    match style {
        FontStyle::Regular => (FC_WEIGHT_REGULAR, FC_SLANT_ROMAN),
        FontStyle::Medium => (FC_WEIGHT_MEDIUM, FC_SLANT_ROMAN),
        FontStyle::Bold => (FC_WEIGHT_BOLD, FC_SLANT_ROMAN),
        FontStyle::Italic => (FC_WEIGHT_REGULAR, FC_SLANT_ITALIC),
        FontStyle::BoldItalic => (FC_WEIGHT_BOLD, FC_SLANT_ITALIC),
    }
}

/// The system's scalable fonts, ranked once for fallback lookup.
///
/// Later lookups scan retained character maps instead of running a new match. Fontconfig
/// trims fonts that add no coverage beyond earlier entries.
pub struct Fontconfig {
    /// The ranked fonts. Owned: the charsets borrowed in [`font_for_char`](Self::font_for_char)
    /// point into these patterns, so the set must outlive every query made against it.
    set: *mut FcFontSet,
}

impl Fontconfig {
    /// Rank system fallback fonts, or `None` when fontconfig is unavailable or empty.
    ///
    /// [`crate::platform::freetype::Fonts`] calls this lazily when its selected faces
    /// cannot draw a character.
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
            if let Some(path) = pattern_path(font, FC_FILE) {
                return Some(FontFile {
                    path,
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
    pattern_string_at(pattern, object, 0)
}

/// Every string value of a pattern property, in fontconfig preference order.
fn pattern_strings(pattern: *const FcPattern, object: &[u8]) -> Vec<String> {
    (0..)
        .map_while(|index| pattern_string_at(pattern, object, index))
        .collect()
}

/// The `index`th value of a string property, for the properties that carry several (a
/// font's family names, one per language it declares).
fn pattern_string_at(pattern: *const FcPattern, object: &[u8], index: c_int) -> Option<String> {
    let mut value: *mut c_uchar = ptr::null_mut();
    // SAFETY: pattern is live, object is a NUL-terminated literal, and value is a live
    // local that fontconfig writes a borrowed pointer through.
    let result = unsafe { FcPatternGetString(pattern, object.as_ptr().cast(), index, &mut value) };
    if result != FC_RESULT_MATCH || value.is_null() {
        return None;
    }
    // SAFETY: on FcResultMatch, value points at a NUL-terminated UTF-8 string owned by the
    // pattern, which outlives this borrow (we copy before returning).
    let text = unsafe { CStr::from_ptr(value.cast::<c_char>()) };
    text.to_str().ok().map(str::to_owned)
}

/// A filesystem property copied without interpreting its Unix path bytes as text.
fn pattern_path(pattern: *const FcPattern, object: &[u8]) -> Option<PathBuf> {
    let mut value: *mut c_uchar = ptr::null_mut();
    // SAFETY: pattern is live, object is a NUL-terminated literal, and value is a live
    // local that fontconfig writes a borrowed pointer through.
    let result = unsafe { FcPatternGetString(pattern, object.as_ptr().cast(), 0, &mut value) };
    if result != FC_RESULT_MATCH || value.is_null() {
        return None;
    }
    // SAFETY: on FcResultMatch, value points at a NUL-terminated byte string owned by
    // the pattern, which remains live while the bytes are copied.
    let value = unsafe { CStr::from_ptr(value.cast::<c_char>()) };
    Some(PathBuf::from(OsString::from_vec(value.to_bytes().to_vec())))
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

    /// Symbols used to exercise system fallback discovery.
    const HOMELESS: [char; 6] = ['⏺', '⎿', '✓', '✗', '✻', '⠁'];

    #[test]
    fn fcfontset_matches_the_c_abi() {
        // Values emitted by a C compiler for fontconfig's only nonopaque type used here.
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
        let Some(fc) = Fontconfig::new() else {
            return;
        };
        let found = fc
            .font_for_char('A')
            .expect("some font on this system has 'A'");
        assert!(found.path.exists());
    }

    #[test]
    fn the_system_resolves_its_generic_monospace() {
        if Fontconfig::new().is_none() {
            eprintln!("no usable fontconfig in this environment; skipping");
            return;
        }
        let mono = font_for_family("monospace", FontStyle::Regular)
            .expect("fontconfig resolves the monospace alias");
        assert!(
            mono.path.exists(),
            "the file it named must be on disk: {}",
            mono.path.display()
        );
    }

    #[test]
    fn every_style_resolves_to_a_file_that_exists() {
        if Fontconfig::new().is_none() {
            return;
        }
        for style in [
            FontStyle::Regular,
            FontStyle::Medium,
            FontStyle::Bold,
            FontStyle::Italic,
            FontStyle::BoldItalic,
        ] {
            let file =
                font_for_family("monospace", style).unwrap_or_else(|| panic!("{style:?} resolves"));
            assert!(file.path.exists(), "{style:?} named a missing file");
        }
    }

    #[test]
    fn an_uninstalled_family_is_a_miss_not_a_substitute() {
        // FcFontMatch substitutes a nearby font, but named candidates must remain misses.
        if Fontconfig::new().is_none() {
            return;
        }
        assert!(font_for_family("NoSuchFamilyExistsAnywhere", FontStyle::Regular).is_none());
    }

    #[test]
    fn a_generic_alias_takes_whatever_it_resolves_to() {
        if Fontconfig::new().is_none() {
            return;
        }
        for generic in GENERIC_FAMILIES {
            if generic == "emoji" {
                continue;
            }
            let file = font_for_family(generic, FontStyle::Regular)
                .unwrap_or_else(|| panic!("the {generic} alias resolves"));
            assert!(file.path.exists());
        }
    }

    #[test]
    fn configured_alias_targets_exclude_the_global_fallbacks() {
        let strings = |names: &[&str]| -> Vec<String> {
            names.iter().map(|name| (*name).to_string()).collect()
        };
        let baseline = strings(&[UNCONFIGURED_FAMILY, "Default Sans", "Last Resort"]);

        let missing = strings(&["Missing Family", "Default Sans", "Last Resort"]);
        assert!(alias_targets("Missing Family", &missing, &baseline).is_empty());

        let alias = strings(&[
            "Chosen Mono",
            "My Terminal Font",
            "Default Sans",
            "Last Resort",
        ]);
        assert_eq!(
            alias_targets("My Terminal Font", &alias, &baseline),
            ["Chosen Mono"]
        );

        let unavailable = strings(&[
            "Unavailable Target",
            "My Terminal Font",
            "Default Sans",
            "Last Resort",
        ]);
        let targets = alias_targets("My Terminal Font", &unavailable, &baseline);
        assert!(!targets.contains(&"Default Sans"));
    }

    #[test]
    fn a_named_family_that_is_installed_comes_back() {
        if Fontconfig::new().is_none() {
            return;
        }
        let Some(mono) = font_for_family("monospace", FontStyle::Regular) else {
            return;
        };
        let Some(name) = installed_family_name(&mono) else {
            return;
        };
        let by_name = font_for_family(&name, FontStyle::Regular)
            .unwrap_or_else(|| panic!("{name} is installed, so it resolves by name"));
        assert_eq!(by_name.path, mono.path);
    }

    /// An installed family name derived without assuming the system's font selection.
    fn installed_family_name(file: &FontFile) -> Option<String> {
        let fc = Fontconfig::new()?;
        // SAFETY: the set is live and owned by `fc`; the patterns inside are borrowed.
        let (count, fonts) = unsafe { ((*fc.set).nfont, (*fc.set).fonts) };
        for i in 0..count {
            // SAFETY: i is in 0..nfont, so fonts[i] is one of the set's live patterns.
            let font = unsafe { *fonts.add(i as usize) };
            if font.is_null() {
                continue;
            }
            if pattern_path(font, FC_FILE).as_ref() == Some(&file.path) {
                return pattern_string(font, FC_FAMILY);
            }
        }
        None
    }

    #[test]
    fn a_family_name_with_an_interior_nul_is_refused() {
        assert!(font_for_family("mono\0space", FontStyle::Regular).is_none());
    }

    #[test]
    fn a_file_property_preserves_non_utf8_path_bytes() {
        use std::os::unix::ffi::OsStrExt;

        let pattern = Pattern::new().expect("a fontconfig pattern");
        let path = b"/tmp/bnkterm-font-\xff.ttf\0";
        // SAFETY: pattern is live, both byte strings are NUL-terminated, and
        // fontconfig copies the property value.
        let added =
            unsafe { FcPatternAddString(pattern.0, FC_FILE.as_ptr().cast(), path.as_ptr().cast()) };
        assert_eq!(added, FC_TRUE);

        let resolved = pattern_path(pattern.0, FC_FILE).expect("the path property");
        assert_eq!(resolved.as_os_str().as_bytes(), &path[..path.len() - 1]);
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
