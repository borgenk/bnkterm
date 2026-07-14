//! Grapheme cluster segmentation, following the Unicode UAX #29 rules.
//!
//! The editor moves the cursor and deletes by *user-perceived characters*, not
//! by Unicode scalars: Backspace over a flag, a skin-toned wave, or a ZWJ family
//! emoji removes the whole thing, and a base letter plus its combining accent
//! travel together. That is exactly a grapheme cluster.
//!
//! UAX #29 decides cluster boundaries from each scalar's Grapheme_Cluster_Break
//! property (plus Extended_Pictographic for emoji and Indic_Conjunct_Break for
//! Brahmic conjuncts). Staying crate-free, all three property tables are
//! generated from the Unicode Character Database into `grapheme_tables.rs` (see
//! `tools/gen_grapheme_tables.py`) and binary-searched here, so classification is
//! complete rather than a hand-curated subset. The full break rule set
//! GB1..GB13, including GB9c (the Indic conjunct break added in Unicode 15.1), is
//! implemented.
//!
//! ```text
//! "a"  + U+0301 (combining acute)      -> one cluster  "á"
//! 👨 + ZWJ + 👩 + ZWJ + 👧             -> one cluster  (family)
//! 🇳 + 🇴 (regional indicators)         -> one cluster  (flag, paired)
//! 👋 + U+1F3FD (skin tone)             -> one cluster
//! ```

use core::cmp::Ordering;

/// The Grapheme_Cluster_Break property value of a scalar, as far as the rules
/// below need to distinguish. Extended_Pictographic is tracked separately (a
/// scalar can be `Other` for GCB yet Extended_Pictographic for emoji rules).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Gcb {
    Other,
    Cr,
    Lf,
    Control,
    Extend,
    Zwj,
    RegionalIndicator,
    Prepend,
    SpacingMark,
    L,
    V,
    T,
    Lv,
    Lvt,
}

/// The Indic_Conjunct_Break property value of a scalar, for rule GB9c. Scalars
/// not in the table are treated as having no InCB value (`incb` returns `None`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Incb {
    Consonant,
    Extend,
    Linker,
}

/// Byte offsets of every grapheme cluster boundary in `s`, always including 0
/// and `s.len()`. Consecutive entries bound one cluster.
/// A lazy iterator over the grapheme clusters of a string, each yielded as its
/// byte offset and substring. It runs the UAX #29 break state machine forward one
/// scalar at a time, so no boundary list is ever materialised, the layout measure
/// path segments millions of clusters and must not allocate a `Vec` per line.
pub struct Graphemes<'a> {
    s: &'a str,
    iter: core::str::CharIndices<'a>,
    /// Byte offset of the cluster being accumulated.
    start: usize,
    done: bool,
    state: BreakState,
}

/// The UAX #29 break machine, carried one scalar at a time.
///
/// A terminal needs this *incrementally* and cannot have it any other way. The grid is
/// handed one scalar at a time as bytes arrive off the pty, and it cannot look ahead: the
/// rest of a cluster may not have been written yet, may arrive in the next read, or may
/// never arrive at all because the program crashed mid-emoji. So the question can never be
/// "where are the boundaries in this string?" — it has to be "does a boundary fall
/// immediately *before* this scalar, given everything I have seen?", which is what this
/// answers.
///
/// [`Graphemes`] runs the same machine over a string it already holds, so a cluster the
/// grid assembles and a cluster the renderer segments can never disagree about where it
/// ends.
#[derive(Clone, Copy, Default)]
pub struct BreakState {
    prev: Option<Gcb>,
    /// Consecutive Regional_Indicator scalars ending at `prev` (raw run length): a
    /// break falls between two RIs only when this is even.
    ri_run: u32,
    /// GB11: the run ending at `prev` is an Extended_Pictographic then zero or more
    /// Extend (`ep_then_extend`), possibly closed by a ZWJ (`zwj_after_ep`).
    ep_then_extend: bool,
    zwj_after_ep: bool,
    /// GB9c: the run ending at `prev` is an InCB=Consonant then only InCB
    /// Extend/Linker (`incb_run`), including at least one Linker (`incb_linker`).
    incb_run: bool,
    incb_linker: bool,
}

impl BreakState {
    /// Whether a cluster boundary falls immediately before `c`, folding `c` into the
    /// state either way.
    ///
    /// `false` for the very first scalar it ever sees: there is no cluster in front of it
    /// to be broken away from. A caller starting a fresh run of text calls [`reset`] and
    /// gets that behaviour again.
    pub fn breaks_before(&mut self, c: char) -> bool {
        let g = gcb(c);
        let ep = is_extended_pictographic(c);
        let ci = incb(c);
        let gb9c = self.incb_run && self.incb_linker && ci == Some(Incb::Consonant);
        let brk = self
            .prev
            .is_some_and(|p| should_break(p, g, ep, self.ri_run, self.zwj_after_ep, gb9c));
        self.advance(g, ep, ci);
        brk
    }

    /// Forget everything: the next scalar starts a cluster rather than continuing one.
    /// The grid calls this whenever the run of printing is interrupted — a control byte,
    /// a cursor move — because a cluster cannot span one.
    pub fn reset(&mut self) {
        *self = BreakState::default();
    }

    /// Fold one scalar's break properties into the carried state (the state-update
    /// half of GB1..GB13), in the order the rules require: `ri_run` reads the
    /// previous scalar, then `prev` advances.
    fn advance(&mut self, g: Gcb, ep: bool, ci: Option<Incb>) {
        self.ri_run = if g == Gcb::RegionalIndicator {
            if self.prev == Some(Gcb::RegionalIndicator) {
                self.ri_run + 1
            } else {
                1
            }
        } else {
            0
        };
        if ep {
            self.ep_then_extend = true;
            self.zwj_after_ep = false;
        } else if g == Gcb::Extend {
            self.zwj_after_ep = false; // ep_then_extend carries through Extend*
        } else if g == Gcb::Zwj {
            self.zwj_after_ep = self.ep_then_extend;
            self.ep_then_extend = false;
        } else {
            self.ep_then_extend = false;
            self.zwj_after_ep = false;
        }
        match ci {
            Some(Incb::Consonant) => {
                self.incb_run = true;
                self.incb_linker = false;
            }
            Some(Incb::Linker) => self.incb_linker |= self.incb_run,
            Some(Incb::Extend) => {}
            None => {
                self.incb_run = false;
                self.incb_linker = false;
            }
        }
        self.prev = Some(g);
    }
}

impl<'a> Iterator for Graphemes<'a> {
    type Item = (usize, &'a str);

    fn next(&mut self) -> Option<(usize, &'a str)> {
        if self.done {
            return None;
        }
        loop {
            let Some((i, c)) = self.iter.next() else {
                // The final cluster runs to the end of the string.
                self.done = true;
                return (self.start < self.s.len()).then(|| (self.start, &self.s[self.start..]));
            };
            // A break before `c` closes the cluster at `[start, i)`; `c` then opens
            // the next one. State advances for `c` either way.
            let brk = self.state.breaks_before(c);
            if brk {
                let out = (self.start, &self.s[self.start..i]);
                self.start = i;
                return Some(out);
            }
        }
    }
}

/// Whether a cluster boundary falls between adjacent scalars `prev` and `cur`.
/// Implements UAX #29 rules GB3..GB999 in order; the first match wins. `gb9c` is
/// the precomputed Indic-conjunct condition (it needs scan state the caller
/// tracks).
fn should_break(
    prev: Gcb,
    cur: Gcb,
    cur_ep: bool,
    ri_run: u32,
    zwj_after_ep: bool,
    gb9c: bool,
) -> bool {
    use Gcb::*;
    // GB3: do not break between CR and LF.
    if prev == Cr && cur == Lf {
        return false;
    }
    // GB4 / GB5: always break around Control, CR, and LF.
    if matches!(prev, Control | Cr | Lf) || matches!(cur, Control | Cr | Lf) {
        return true;
    }
    // GB6 / GB7 / GB8: keep Hangul syllables together.
    if prev == L && matches!(cur, L | V | Lv | Lvt) {
        return false;
    }
    if matches!(prev, Lv | V) && matches!(cur, V | T) {
        return false;
    }
    if matches!(prev, Lvt | T) && cur == T {
        return false;
    }
    // GB9 / GB9a / GB9b: never break before Extend, ZWJ, or SpacingMark, nor
    // after Prepend.
    if matches!(cur, Extend | Zwj) {
        return false;
    }
    if cur == SpacingMark {
        return false;
    }
    if prev == Prepend {
        return false;
    }
    // GB9c: keep an Indic conjunct joined (Consonant [Extend Linker]* Linker
    // [Extend Linker]* x Consonant). The left side is precomputed in `gb9c`.
    if gb9c {
        return false;
    }
    // GB11: keep an emoji ZWJ sequence (Ext_Pict Extend* ZWJ x Ext_Pict) joined.
    if prev == Zwj && cur_ep && zwj_after_ep {
        return false;
    }
    // GB12 / GB13: break between regional indicators only on even boundaries, so
    // they pair up into flags.
    if prev == RegionalIndicator && cur == RegionalIndicator {
        return ri_run.is_multiple_of(2);
    }
    // GB999: otherwise, break.
    true
}

/// The byte offset of the first cluster boundary strictly after `i`. `i` is
/// assumed to be a boundary itself (the cursor always sits on one). Cluster starts
/// plus the string end are the boundaries, so the first start past `i` is it, or
/// the end when `i` sits in the last cluster.
pub fn next_boundary(s: &str, i: usize) -> usize {
    graphemes(s)
        .map(|(off, _)| off)
        .find(|&off| off > i)
        .unwrap_or(s.len())
}

/// The byte offset of the last cluster boundary strictly before `i`.
pub fn prev_boundary(s: &str, i: usize) -> usize {
    graphemes(s)
        .map(|(off, _)| off)
        .take_while(|&off| off < i)
        .last()
        .unwrap_or(0)
}

/// Whether `c` counts as a word character for word motion (Ctrl+arrow, word
/// delete) and whole-word find: an alphanumeric or `_`. The single definition the
/// two share, so they can never drift apart.
pub fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The grapheme clusters of `s`, lazily: each yielded as its byte offset and
/// substring, with nothing allocated.
pub fn graphemes(s: &str) -> Graphemes<'_> {
    Graphemes {
        s,
        iter: s.char_indices(),
        start: 0,
        done: false,
        state: BreakState::default(),
    }
}

// ---------------------------------------------------------------------------
// Property classification, by binary search over the generated UCD tables in
// grapheme_tables.rs (GCB_RANGES and EXTENDED_PICTOGRAPHIC). Both are sorted,
// non-overlapping range lists, so a codepoint maps to its property in O(log n).
// ---------------------------------------------------------------------------

/// Find the range covering `u` in a `(lo, hi, _)` table sorted by `lo`.
fn search<T: Copy>(table: &[(u32, u32, T)], u: u32) -> Option<T> {
    table
        .binary_search_by(|&(lo, hi, _)| {
            if u < lo {
                Ordering::Greater
            } else if u > hi {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .ok()
        .map(|i| table[i].2)
}

fn gcb(c: char) -> Gcb {
    search(GCB_RANGES, c as u32).unwrap_or(Gcb::Other)
}

fn incb(c: char) -> Option<Incb> {
    search(INCB_RANGES, c as u32)
}

/// Whether `c` has the Unicode `Extended_Pictographic` property, i.e. is (or
/// can start) an emoji-form cluster. Public for the font layer, which uses it
/// to route emoji clusters to the color emoji face.
pub fn is_extended_pictographic(c: char) -> bool {
    let u = c as u32;
    EXTENDED_PICTOGRAPHIC
        .binary_search_by(|&(lo, hi)| {
            if u < lo {
                Ordering::Greater
            } else if u > hi {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

include!("grapheme_tables.rs");

#[cfg(test)]
mod tests {
    use super::*;

    /// Cluster boundaries as a list, for compact assertions.
    fn bounds(s: &str) -> Vec<usize> {
        graphemes(s)
            .map(|(o, _)| o)
            .chain(std::iter::once(s.len()))
            .collect()
    }

    #[test]
    fn ascii_is_one_cluster_per_byte() {
        assert_eq!(bounds("abc"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn precomposed_and_combining_accents() {
        // Precomposed é (U+00E9, 2 bytes) is one scalar, one cluster.
        assert_eq!(bounds("é"), vec![0, 2]);
        // Decomposed: 'e' + combining acute (U+0301, 2 bytes) is one cluster.
        let decomposed = "e\u{0301}";
        assert_eq!(decomposed.len(), 3);
        assert_eq!(bounds(decomposed), vec![0, 3]);
    }

    #[test]
    fn emoji_is_a_single_cluster() {
        // 😀 is U+1F600, four UTF-8 bytes.
        assert_eq!(bounds("a😀"), vec![0, 1, 5]);
    }

    #[test]
    fn zwj_family_is_one_cluster() {
        // 👨‍👩‍👧 = man ZWJ woman ZWJ girl: 4 + 3 + 4 + 3 + 4 = 18 bytes, one cluster.
        let family = "👨\u{200D}👩\u{200D}👧";
        assert_eq!(family.len(), 18);
        assert_eq!(bounds(family), vec![0, 18]);
        assert_eq!(next_boundary(family, 0), 18);
        assert_eq!(prev_boundary(family, 18), 0);
    }

    #[test]
    fn regional_indicators_pair_into_flags() {
        // 🇳🇴 = one flag (two RIs), 8 bytes; 🇳🇴🇸🇪 = two flags.
        assert_eq!(bounds("🇳🇴"), vec![0, 8]);
        assert_eq!(bounds("🇳🇴🇸🇪"), vec![0, 8, 16]);
    }

    #[test]
    fn skin_tone_modifier_joins_its_base() {
        // 👋 (U+1F44B) + medium skin tone (U+1F3FD): one cluster, 8 bytes.
        let wave = "👋\u{1F3FD}";
        assert_eq!(wave.len(), 8);
        assert_eq!(bounds(wave), vec![0, 8]);
    }

    #[test]
    fn indic_conjunct_joins_via_gb9c() {
        // Devanagari KA + VIRAMA + KA forms one conjunct cluster (GB9c). Each
        // scalar is three UTF-8 bytes.
        let conjunct = "\u{0915}\u{094D}\u{0915}";
        assert_eq!(conjunct.len(), 9);
        assert_eq!(bounds(conjunct), vec![0, 9]);
        // A longer chain (KA vir KA vir KA) stays one cluster, exercising the
        // consonant-restart in the run state.
        let chain = "\u{0915}\u{094D}\u{0915}\u{094D}\u{0915}";
        assert_eq!(bounds(chain), vec![0, chain.len()]);
    }

    #[test]
    fn two_consonants_without_a_linker_still_break() {
        // KA + KA with no virama between is two clusters: GB9c requires a Linker.
        let two = "\u{0915}\u{0915}";
        assert_eq!(bounds(two), vec![0, 3, 6]);
    }

    #[test]
    fn full_tables_cover_scripts_beyond_the_old_curated_set() {
        // Combining marks from scripts the earlier hand-curated table did not
        // list now join their base, because the property tables are the full
        // UCD. Tibetan KA + vowel sign I:
        let tibetan = "\u{0F40}\u{0F72}";
        assert_eq!(bounds(tibetan), vec![0, tibetan.len()]);
        // Balinese letter A + vowel sign ULU, a different block:
        let balinese = "\u{1B05}\u{1B36}";
        assert_eq!(bounds(balinese), vec![0, balinese.len()]);
    }

    #[test]
    fn newline_is_its_own_cluster() {
        // Backspace at the join must remove the '\n' alone, merging the lines.
        assert_eq!(bounds("a\nb"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn graphemes_splits_into_substrings() {
        let g: Vec<_> = graphemes("a😀b").collect();
        assert_eq!(g, vec![(0, "a"), (1, "😀"), (5, "b")]);
    }

    #[test]
    fn boundary_navigation_steps_whole_clusters() {
        let s = "👋\u{1F3FD}x"; // wave+tone (8 bytes), then x
        assert_eq!(next_boundary(s, 0), 8);
        assert_eq!(next_boundary(s, 8), 9);
        assert_eq!(prev_boundary(s, 9), 8);
        assert_eq!(prev_boundary(s, 8), 0);
        // At the ends, navigation clamps.
        assert_eq!(prev_boundary(s, 0), 0);
        assert_eq!(next_boundary(s, 9), 9);
    }
}
