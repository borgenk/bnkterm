//! Character cell width: how many terminal columns a scalar value occupies when
//! printed. The grid needs this number to advance the cursor and to lay wide and
//! combining characters into cells correctly; getting it wrong corrupts a screen
//! the way the `grid` module header warns about.
//!
//! Three answers:
//!
//! ```text
//!   0  combining marks (Mn, Me), format characters (Cf), and the Hangul
//!      conjoining jamo: they compose onto the previous cell and take no column.
//!   1  the ordinary case: one glyph, one column.
//!   2  East Asian Wide and Fullwidth: CJK and most emoji, drawn across two
//!      columns (a WIDE_LEADER cell plus a WIDE_SPACER placeholder in the grid).
//! ```
//!
//! A generated two-level table maps each scalar to a width. Unicode is divided
//! into 256-scalar pages and identical pages share one body, keeping lookup to
//! two indexed loads without carrying a 1.1 MiB flat table. The generator also
//! retains the source ranges in test builds as an exhaustive correctness oracle.
//! Control characters never reach `width`: the parser
//! dispatches C0/C1 as actions, so only printable scalar values are measured.
//! xterm is the reference for the two wcwidth tweaks the generator bakes in
//! (SOFT HYPHEN is width 1, not 0; the Hangul jamo are width 0 though not marks).

#[cfg(test)]
use core::cmp::Ordering;

/// Columns `c` occupies when printed: 0, 1, or 2. Zero-width wins over wide, so a
/// combining mark that also carries an East Asian width still measures 0.
/// Display columns occupied by one already-segmented grapheme cluster.
///
/// This is the *other* width function, and the two must never disagree: [`width`] measures
/// a scalar, this measures a user-perceived character. A terminal that measures a cluster
/// one way when it decides which cells to occupy and another way when it draws them puts
/// the cursor somewhere the glyphs are not, and every subsequent column is wrong. So this
/// lives here, beside the scalar width, and both the grid and the renderer call it —
/// rather than each keeping a rule of its own, which is precisely how they used to differ.
///
/// A combining sequence inherits its base width. Three cases do not:
///
/// - an **emoji presentation selector** (`U+FE0F`) says "draw the preceding character as
///   an emoji", and an emoji is two cells wide even when the bare character is one (`☀`
///   is narrow, `☀️` is not);
/// - a **keycap** (`U+20E3`) likewise, which is how `1️⃣` is two cells and `1` is one;
/// - a **flag** is two regional indicators, each of which is *narrow* on its own. Nothing
///   about the pair's scalar widths says two, and yet two is what it draws as.
pub fn cluster_width(cluster: &str) -> u8 {
    let mut chars = cluster.chars();
    let Some(first) = chars.next() else {
        return 0;
    };
    if cluster.contains('\u{20e3}')
        || cluster.contains('\u{fe0f}')
        || (matches!(first, '\u{1f1e6}'..='\u{1f1ff}') && chars.next().is_some())
    {
        return 2;
    }
    cluster.chars().map(width).max().unwrap_or(0)
}

pub fn width(c: char) -> u8 {
    let cp = u32::from(c);
    let page_number = usize::try_from(cp >> WIDTH_PAGE_SHIFT).unwrap_or_default();
    let page = WIDTH_PAGE_INDEX
        .get(page_number)
        .copied()
        .map(usize::from)
        .and_then(|index| WIDTH_PAGES.get(index));
    let page_mask = 1_u32
        .checked_shl(WIDTH_PAGE_SHIFT)
        .unwrap_or(1)
        .saturating_sub(1);
    let offset = usize::try_from(cp & page_mask).unwrap_or_default();
    page.and_then(|values| values.get(offset))
        .copied()
        .unwrap_or(1)
}

/// Whether `cp` falls inside one of the sorted, non-overlapping `ranges`.
#[cfg(test)]
fn in_ranges(ranges: &[(u32, u32)], cp: u32) -> bool {
    ranges
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                Ordering::Greater
            } else if cp > hi {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

#[cfg(test)]
fn range_width(c: char) -> u8 {
    let cp = u32::from(c);
    if in_ranges(ZERO_WIDTH_RANGES, cp) {
        0
    } else if in_ranges(WIDE_RANGES, cp) {
        2
    } else {
        1
    }
}

include!("width_tables.rs");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_is_one_column() {
        assert_eq!(width('A'), 1);
        assert_eq!(width(' '), 1);
        assert_eq!(width('~'), 1);
        assert_eq!(width('0'), 1);
    }

    #[test]
    fn combining_marks_are_zero() {
        assert_eq!(width('\u{0300}'), 0); // combining grave accent
        assert_eq!(width('\u{0301}'), 0); // combining acute accent
        assert_eq!(width('\u{20DD}'), 0); // combining enclosing circle (Me)
    }

    #[test]
    fn format_characters_are_zero() {
        assert_eq!(width('\u{200B}'), 0); // zero width space
        assert_eq!(width('\u{200D}'), 0); // zero width joiner
        assert_eq!(width('\u{FEFF}'), 0); // zero width no-break space / BOM
    }

    #[test]
    fn cjk_and_kana_are_wide() {
        assert_eq!(width('世'), 2);
        assert_eq!(width('中'), 2);
        assert_eq!(width('\u{4E00}'), 2); // first CJK unified ideograph
        assert_eq!(width('\u{3042}'), 2); // hiragana A
        assert_eq!(width('\u{AC00}'), 2); // hangul syllable GA
    }

    #[test]
    fn fullwidth_and_halfwidth_forms() {
        assert_eq!(width('\u{FF21}'), 2); // FULLWIDTH LATIN CAPITAL A
        assert_eq!(width('\u{FF61}'), 1); // HALFWIDTH IDEOGRAPHIC FULL STOP (H)
    }

    #[test]
    fn common_emoji_are_wide() {
        assert_eq!(width('\u{1F600}'), 2); // grinning face
        assert_eq!(width('\u{1F680}'), 2); // rocket
    }

    #[test]
    fn wcwidth_tweaks_hold() {
        // SOFT HYPHEN is gc=Cf but width 1 in a terminal, not 0.
        assert_eq!(width('\u{00AD}'), 1);
        // Hangul conjoining jamo are gc=Lo but must be width 0 to compose.
        assert_eq!(width('\u{1160}'), 0);
        assert_eq!(width('\u{11FF}'), 0);
    }

    #[test]
    fn tables_are_sorted_and_disjoint() {
        // The binary search relies on this; a bad generator run must go red.
        for table in [WIDE_RANGES, ZERO_WIDTH_RANGES] {
            for pair in table.windows(2) {
                let (_, prev_hi) = pair[0];
                let (lo, hi) = pair[1];
                assert!(lo <= hi, "reversed range ({lo:#x}, {hi:#x})");
                assert!(prev_hi < lo, "unsorted or overlapping at {lo:#x}");
            }
        }
    }

    #[test]
    fn pinned_to_the_vendored_unicode_version() {
        assert_eq!(UNICODE_VERSION, "18.0.0");
    }

    #[test]
    fn width_never_panics_over_all_scalar_values() {
        // Exhaustive equivalence makes the retained ranges an oracle for every
        // generated page entry as well as pinning the no-panic guarantee.
        for cp in 0..=0x10FFFFu32 {
            if let Some(c) = char::from_u32(cp) {
                let w = width(c);
                assert!(w <= 2);
                assert_eq!(w, range_width(c), "page lookup differs at U+{cp:04X}");
            }
        }
    }
}
