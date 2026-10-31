//! The cell and its storage: the packed layout, the attribute set, the style table's
//! collection, and the bounds on what the grid holds.

use crate::grid::*;
use crate::testutil::feed;

#[test]
fn cell_layout_is_pinned() {
    // A screenful plus scrollback is a lot of cells; growth is a decision, not an
    // accident. The stored cell is exactly eight bytes and every one is spoken for:
    // a `u32` holding the scalar in 21 bits with `CellWidth` in the two the scalar
    // could never use, then two bytes naming a rendition and two naming a hyperlink.
    // Anything that pushes this to 12 costs the grid 50%.
    assert_eq!(std::mem::size_of::<PackedCell>(), 8);
    assert_eq!(std::mem::align_of::<PackedCell>(), 4);
    assert_eq!(std::mem::size_of::<StyleId>(), 2);
    assert_eq!(std::mem::size_of::<LinkId>(), 2);
    assert_eq!(std::mem::size_of::<Attrs>(), 2);

    // The resolved `Cell` is the shape callers work in, not the shape stored, so it is
    // free to be comfortable. It is pinned only so that nobody mistakes it for the one
    // the grid is made of.
    assert_eq!(std::mem::size_of::<Cell>(), 20);
}

#[test]
fn a_packed_cell_round_trips_every_field() {
    // The packing is bit manipulation on the hottest type in the codebase, so it is
    // worth proving rather than eyeballing: the scalar and the width share a `u32` and
    // must not bleed into each other at either end of the range.
    for rune in ['\0', ' ', 'a', '世', '\u{10FFFF}', '\u{FFFF}'] {
        for width in [CellWidth::Narrow, CellWidth::Leader, CellWidth::Spacer] {
            let c = PackedCell::new(rune, width, StyleId::DEFAULT, LinkId::NONE);
            assert_eq!(c.rune(), rune, "rune survives {width:?}");
            assert_eq!(c.width(), width, "width survives {rune:?}");
        }
    }
    // The blank is a space in the default rendition, and is what an empty grid is full of.
    assert_eq!(PackedCell::BLANK.rune(), ' ');
    assert_eq!(PackedCell::BLANK.width(), CellWidth::Narrow);
    assert_eq!(PackedCell::BLANK.style_id(), StyleId::DEFAULT);
}

#[test]
fn a_full_scrollback_ring_never_outgrows_its_limit() {
    // The ring is filled by pushing past the cap and popping the front back off, so it
    // is momentarily one row longer than it is allowed to be. Allocated at exactly the
    // limit, that one push doubles the header array and the doubling is never given
    // back: 0.53 MiB held for nothing at the default 10k ring. The spare slot
    // ([`buffer::scrollback_capacity`]) is what stops it, and this is what stops the
    // spare slot from being tidied away again.
    const LIMIT: usize = 64;
    let mut s = Screen::with_scrollback(20, 4, LIMIT);
    let want = buffer::scrollback_capacity(LIMIT);

    // Well past the cap, so eviction has been running for a while.
    for _ in 0..LIMIT * 3 {
        feed(&mut s, b"line\r\n");
    }
    assert_eq!(s.scrollback_len(), LIMIT);
    assert_eq!(s.primary.scrollback.capacity(), want);

    // A width change rebuilds the ring from scratch, which is where this last came
    // back: reflow sized the new deque to the rows it was putting in it.
    s.resize(11, 4);
    for _ in 0..LIMIT * 3 {
        feed(&mut s, b"line\r\n");
    }
    assert_eq!(s.scrollback_len(), LIMIT);
    assert_eq!(s.primary.scrollback.capacity(), want);
}

#[test]
fn a_full_grid_holds_exactly_the_bytes_it_should() {
    // The footprint gate compares this number exactly, so it is worth
    // knowing here that every byte of it is accounted for rather than merely stable.
    // A grid is its rows and nothing else: cells at 16 bytes, one `Row` header per row
    // in each ring, and the tab table.
    const COLS: usize = 20;
    const ROWS: usize = 4;
    const LIMIT: usize = 64;
    let mut s = Screen::with_scrollback(COLS, ROWS, LIMIT);
    for _ in 0..LIMIT * 3 {
        feed(&mut s, b"line\r\n");
    }
    assert_eq!(s.scrollback_len(), LIMIT);

    // The *stored* cell, not the resolved one callers see: the grid is made of packed
    // cells, and using the wrong one here is exactly the confusion the split invites.
    let cell = std::mem::size_of::<PackedCell>();
    let row = std::mem::size_of::<buffer::Row>();
    let tabs = buffer::default_tabs(COLS).capacity();
    // Both buffers: the primary's full ring plus live screen, and the alt screen's
    // live rows (it keeps no history, and its ring still carries the spare slot).
    let primary =
        (LIMIT + ROWS) * COLS * cell + buffer::scrollback_capacity(LIMIT) * row + ROWS * row + tabs;
    let alt = ROWS * COLS * cell + buffer::scrollback_capacity(0) * row + ROWS * row + tabs;
    assert_eq!(s.storage_bytes(), primary + alt);
}

#[test]
fn dead_styles_are_collected_and_the_grid_keeps_its_colours() {
    // The mark-and-sweep, end to end. A screen that churns through renditions must not
    // burn out the id space, and the sweep must renumber every surviving cell to match
    // the compacted table — a sweep that reclaimed ids but forgot to renumber would
    // repaint the whole screen in someone else's colours.
    let mut s = Screen::with_scrollback(8, 2, 0);

    // Fill the visible screen with one distinctive rendition, then scroll far more
    // renditions than the screen can hold past it, so almost all of them die.
    feed(&mut s, b"\x1b[38;5;99mkeep\r\n");
    for i in 0..300u16 {
        feed(&mut s, format!("\x1b[38;5;{}mx\r\n", i % 256).as_bytes());
    }
    // With no scrollback, everything but the last two rows is gone, so the table must
    // not have grown to hold the ~256 renditions that passed through it.
    s.collect_styles();
    assert!(
        s.styles.styles.len() <= 4,
        "sweep kept {} renditions for a 2-row screen",
        s.styles.styles.len()
    );

    // And the cells that survived still resolve to what they were written with.
    feed(&mut s, b"\x1b[38;5;42mAB");
    let before = s.cell(1, 0);
    assert_eq!(before.fg, Color::Indexed(42));
    s.collect_styles();
    assert_eq!(s.cell(1, 0), before, "a swept cell kept its rendition");
    assert_eq!(s.cell(1, 1).fg, Color::Indexed(42));
}

#[test]
fn a_style_sweep_mid_print_does_not_strand_the_pen() {
    // Interning into a full style table sweeps it, and the sweep renumbers ids. A wrap's
    // line feed and an insert's blank both intern, so a print path that took the pen's
    // id before them wrote the rest of its text with an id that now names another style,
    // or none. Truecolour output (half-block images, video) fills the table routinely.
    fn nearly_full(cols: usize, rows: usize) -> Screen {
        let mut s = Screen::with_scrollback(cols, rows, 0);
        for i in 0..STYLE_LIMIT - 1 {
            let dead = Style {
                fg: Color::Rgb((i >> 8) as u8, i as u8, 7),
                ..Style::default()
            };
            let _ = s.styles.intern(dead);
        }
        s
    }
    let pen = b"\x1b[38;2;9;9;9;48;2;1;2;3m";

    // A run of ASCII that wraps: the wrap's blank is what fills the table.
    let mut s = nearly_full(4, 3);
    feed(&mut s, &[&pen[..], b"abcdx"].concat());
    for (row, col) in [(0, 0), (1, 0)] {
        let cell = s.cell(row, col);
        assert_eq!(
            (cell.fg, cell.bg),
            (Color::Rgb(9, 9, 9), Color::Rgb(1, 2, 3)),
            "cell ({row}, {col}) keeps the pen"
        );
    }

    // Insert mode: the blank the insert opens is interned after the pen was.
    let mut s = nearly_full(4, 3);
    feed(&mut s, &[b"\x1b[4h".as_slice(), pen, b"x"].concat());
    let cell = s.cell(0, 0);
    assert_eq!(
        (cell.fg, cell.bg),
        (Color::Rgb(9, 9, 9), Color::Rgb(1, 2, 3))
    );
}

#[test]
fn a_spent_style_table_degrades_without_losing_text() {
    // The id space is a `u16`, so a child printing nothing but distinct truecolour
    // renditions can exhaust it. The rule is then the one a spent `LinkId` space
    // follows: the text is still there and still correct, it just draws unstyled.
    let mut table = StyleTable::default();
    let style_for = |i: usize| Style {
        fg: Color::Rgb((i >> 16) as u8, (i >> 8) as u8, i as u8),
        ..Style::default()
    };
    for i in 0..STYLE_LIMIT {
        assert!(table.intern(style_for(i)).is_some(), "id {i} should fit");
    }
    assert_eq!(table.styles.len(), STYLE_LIMIT);
    // One past the end fails rather than wrapping an id or growing past the u16.
    assert_eq!(table.intern(style_for(STYLE_LIMIT)), None);
    // The default costs no id at all, so it is still available on a full table — which
    // is exactly what makes "draw it unstyled" a safe thing to fall back to.
    assert_eq!(table.intern(Style::default()), Some(StyleId::DEFAULT));

    // And a screen whose table is spent still prints, in the default rendition.
    //
    // "Spent" means more than "full": a full table normally triggers a sweep, and a
    // sweep that frees anything makes room. The state the fallback is for is a table
    // full of *live* renditions, which the guard records so a child printing SGR into
    // one does not buy a walk over every cell with each sequence.
    let mut s = Screen::new(4, 2);
    s.styles = table;
    s.styles.swept_dry_at = Some((s.epoch, s.primary.evicted));
    s.pen.fg = Color::Rgb(1, 2, 3);
    assert_eq!(s.pen_style_id(), StyleId::DEFAULT);
    feed(&mut s, b"ok");
    assert_eq!(s.cell(0, 0).rune, 'o');
    assert_eq!(s.cell(0, 1).rune, 'k');
    assert_eq!(s.cell(0, 0).fg, Color::Default, "unstyled, not miscoloured");
}

#[test]
fn a_style_sweep_reclaims_the_whole_id_space() {
    // The sweep is what stops the exhaustion above from being permanent. Nothing here
    // is printed, so every rendition that churned through is dead and all of it comes
    // back — the property that lets a long-lived session keep recolouring forever.
    let mut s = Screen::new(4, 2);
    for i in 0..STYLE_LIMIT {
        s.pen.fg = Color::Rgb((i >> 16) as u8, (i >> 8) as u8, i as u8);
        let _ = s.pen_style_id();
    }
    assert_eq!(s.styles.styles.len(), STYLE_LIMIT);
    s.collect_styles();
    // Only the pen's own current rendition survives: it is a root even unprinted.
    assert_eq!(s.styles.styles.len(), 1);
}

#[test]
fn blank_is_a_default_space() {
    assert_eq!(Cell::BLANK, Cell::default());
    assert_eq!(Cell::BLANK.rune, ' ');
    assert_eq!(Cell::BLANK.fg, Color::Default);
    assert!(Cell::BLANK.attrs.is_empty());
}

#[test]
fn attrs_insert_remove_set_roundtrip() {
    let mut a = Attrs::empty();
    assert!(a.is_empty());
    a.insert(Attrs::BOLD);
    a.insert(Attrs::ITALIC);
    assert!(a.contains(Attrs::BOLD));
    assert!(a.contains(Attrs::ITALIC));
    assert!(!a.contains(Attrs::UNDERLINE));
    // contains(A|B) is all-of, not any-of.
    assert!(a.contains(Attrs::BOLD | Attrs::ITALIC));
    assert!(!a.contains(Attrs::BOLD | Attrs::UNDERLINE));
    a.remove(Attrs::BOLD);
    assert!(!a.contains(Attrs::BOLD));
    a.remove(Attrs::ITALIC);
    assert!(a.is_empty());
}

#[test]
fn every_flag_is_a_distinct_bit() {
    // No two flags share a bit, and none is zero (which would alias empty).
    let mut seen = 0u16;
    for (flag, _) in Attrs::ALL {
        assert_ne!(flag.0, 0);
        assert_eq!(seen & flag.0, 0, "overlapping attr bit");
        seen |= flag.0;
    }
}

#[test]
fn attrs_debug_lists_flags() {
    assert_eq!(format!("{:?}", Attrs::empty()), "Attrs(-)");
    assert_eq!(format!("{:?}", Attrs::BOLD), "Attrs(BOLD)");
    // Listed in bit order regardless of insertion order.
    let a = Attrs::ITALIC | Attrs::BOLD;
    assert_eq!(format!("{a:?}"), "Attrs(BOLD|ITALIC)");
}

#[test]
fn wide_halves_report_their_role() {
    let leader = Cell {
        rune: '世',
        width: CellWidth::Leader,
        ..Cell::BLANK
    };
    assert!(leader.is_wide_leader());
    assert!(!leader.is_wide_spacer());

    let spacer = Cell {
        width: CellWidth::Spacer,
        ..Cell::BLANK
    };
    assert!(spacer.is_wide_spacer());
    assert!(!spacer.is_wide_leader());

    // The role is layout, not rendition: both halves of a wide glyph share one style,
    // which is exactly why it had to leave `Attrs` before a style could be interned.
    assert_eq!(leader.style(), spacer.style());
    assert!(!Cell::BLANK.is_wide_leader());
    assert!(!Cell::BLANK.is_wide_spacer());
}

/// The cap a screen is given is the cap it keeps, in both directions: a ring that
/// is told to hold two lines evicts the third, and one told to hold none keeps
/// nothing at all.
#[test]
fn with_scrollback_caps_history_where_it_is_told_to() {
    let mut s = Screen::with_scrollback(4, 1, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd");
    assert_eq!(s.scrollback_len(), 2, "the ring holds its limit, not more");
    s.scroll_view_to_top();
    assert_eq!(s.view_cell(0, 0).rune, 'b', "the oldest rows evicted");

    // A zero-line history must keep nothing at all, where the default screen would
    // keep it all: the alt screen is built exactly this way.
    let mut none = Screen::with_scrollback(4, 1, 0);
    feed(&mut none, b"a\r\nb\r\nc");
    assert_eq!(none.scrollback_len(), 0);
    assert_eq!(none.view_cell(0, 0).rune, 'c');
}
