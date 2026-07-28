//! The grid's tests. See the `mod tests` declaration in `grid/mod.rs` for why they are one
//! module rather than one per sibling implementation file.

use super::buffer::MAX_LOGICAL_COLS;
use super::osc::base64_decode;
use super::*;
use crate::color::{Ground, Rgb};

#[test]
fn cell_layout_is_pinned() {
    // A screenful plus scrollback is a lot of cells; growth is a decision,
    // not an accident. char(4) + Color(4) + Color(4) + Attrs(2) + LinkId(2),
    // aligned to char's 4 bytes, is exactly 16 — the hyperlink id was fitted into
    // the two bytes the cell was already padding away, so OSC 8 cost the grid
    // nothing. Anything that pushes this to 20 has to justify a 25% bigger grid.
    assert_eq!(std::mem::size_of::<Attrs>(), 2);
    assert_eq!(std::mem::size_of::<LinkId>(), 2);
    assert_eq!(std::mem::size_of::<Cell>(), 16);
    assert_eq!(std::mem::align_of::<Cell>(), 4);
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
    a.set(Attrs::ITALIC, false);
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
    let mut leader = Cell::new('世');
    leader.attrs.insert(Attrs::WIDE_LEADER);
    assert!(leader.is_wide_leader());
    assert!(!leader.is_wide_spacer());

    let mut spacer = Cell::BLANK;
    spacer.attrs.insert(Attrs::WIDE_SPACER);
    assert!(spacer.is_wide_spacer());
    assert!(!spacer.is_wide_leader());
}

// ---- Screen ------------------------------------------------------------

fn print_str(s: &mut Screen, text: &str) {
    for c in text.chars() {
        s.print(c);
    }
}

#[test]
fn print_advances_and_reads_back() {
    let mut s = Screen::new(10, 3);
    print_str(&mut s, "hi");
    assert_eq!(s.cursor(), (0, 2));
    assert_eq!(s.cell(0, 0).rune, 'h');
    assert_eq!(s.cell(0, 1).rune, 'i');
    assert_eq!(s.row_string(0).trim_end(), "hi");
}

#[test]
fn pending_wrap_is_the_last_column_rule() {
    // Exactly `cols` characters must fill the row without wrapping; the wrap
    // is deferred until the next glyph arrives.
    let mut s = Screen::new(4, 3);
    print_str(&mut s, "abcd");
    // Cursor pinned to the last column, wrap pending, still on row 0.
    assert_eq!(s.cursor(), (0, 3));
    assert_eq!(s.row_string(0).trim_end(), "abcd");
    // The next glyph triggers the wrap and lands on row 1.
    s.print('e');
    assert_eq!(s.cursor(), (1, 1));
    assert_eq!(s.cell(1, 0).rune, 'e');
    // The wrapped row is flagged for reflow/selection.
    assert!(s.row_wraps(0));
}

#[test]
fn no_autowrap_pins_at_the_last_column() {
    let mut s = Screen::new(4, 3);
    s.set_mode(7, true, false); // DECAWM off
    print_str(&mut s, "abcdef");
    // Everything past the edge overwrites the last cell; no wrap.
    assert_eq!(s.cursor(), (0, 3));
    assert_eq!(s.cell(0, 3).rune, 'f');
    assert_eq!(s.row_string(1).trim_end(), "");
}

#[test]
fn wide_char_takes_two_cells() {
    let mut s = Screen::new(10, 3);
    s.print('世');
    assert!(s.cell(0, 0).is_wide_leader());
    assert_eq!(s.cell(0, 0).rune, '世');
    assert!(s.cell(0, 1).is_wide_spacer());
    assert_eq!(s.cursor(), (0, 2));
    // row_string skips the spacer, so it reads as one glyph.
    assert_eq!(s.row_string(0).trim_end(), "世");
}

#[test]
fn wide_char_at_right_edge_wraps_whole() {
    // Two columns wide, one column left: the glyph must move to the next row
    // rather than split across the edge.
    let mut s = Screen::new(3, 3);
    print_str(&mut s, "ab"); // cursor now at col 2 (last), pending wrap
    s.print('世');
    // The wide glyph lands on row 1, not straddling the edge of row 0.
    assert_eq!(s.cell(1, 0).rune, '世');
    assert!(s.cell(1, 0).is_wide_leader());
    assert!(s.cell(1, 1).is_wide_spacer());
    assert_eq!(s.cursor(), (1, 2));
}

#[test]
fn overwriting_a_wide_half_cleans_its_partner() {
    let mut s = Screen::new(10, 3);
    s.print('世'); // leader at 0, spacer at 1
    s.move_to(0, 0);
    s.print('x'); // overwrite the leader
    assert_eq!(s.cell(0, 0).rune, 'x');
    assert!(!s.cell(0, 0).is_wide_leader());
    // The orphaned spacer is cleared to a blank.
    assert!(!s.cell(0, 1).is_wide_spacer());
    assert_eq!(s.cell(0, 1), Cell::BLANK);
}

#[test]
fn combining_mark_attaches_to_base() {
    let mut s = Screen::new(10, 3);
    s.print('e');
    s.print('\u{0301}'); // combining acute accent
                         // The base cell is unchanged; the mark is in the side table.
    assert_eq!(s.cell(0, 0).rune, 'e');
    assert_eq!(s.cursor(), (0, 1)); // zero-width, cursor did not advance
    assert_eq!(s.marks_at(0, 0).collect::<Vec<_>>(), ['\u{0301}']);
    assert_eq!(s.row_string(0).trim_end(), "e\u{0301}");
}

#[test]
fn combining_mark_after_wide_char_hits_the_leader() {
    let mut s = Screen::new(10, 3);
    s.print('か'); // wide: leader at 0, spacer at 1, cursor at 2
    s.print('\u{3099}'); // combining voiced sound mark -> が
                         // The mark composes onto the leader (col 0), not the spacer (col 1).
    assert_eq!(s.marks_at(0, 0).collect::<Vec<_>>(), ['\u{3099}']);
    assert!(s.marks_at(0, 1).next().is_none());
}

#[test]
fn carriage_return_and_line_feed() {
    let mut s = Screen::new(10, 3);
    print_str(&mut s, "ab");
    s.carriage_return();
    assert_eq!(s.cursor(), (0, 0));
    s.line_feed();
    assert_eq!(s.cursor(), (1, 0));
}

#[test]
fn line_feed_at_bottom_scrolls_into_scrollback() {
    let mut s = Screen::new(4, 2);
    print_str(&mut s, "aa");
    s.line_feed();
    s.carriage_return();
    print_str(&mut s, "bb");
    s.line_feed(); // at bottom: scroll up
    s.carriage_return();
    print_str(&mut s, "cc");
    // "aa" scrolled off, "bb"/"cc" are the two visible rows.
    assert_eq!(s.row_string(0).trim_end(), "bb");
    assert_eq!(s.row_string(1).trim_end(), "cc");
}

#[test]
fn the_scrollbar_reads_the_view_as_a_scroll_down_from_the_oldest_line() {
    // Three lines retired into history behind a 2-row screen: 5 lines of content, a
    // 2-line viewport, so the view can scroll 3.
    let mut s = Screen::new(4, 2);
    for line in ["aa", "bb", "cc", "dd", "ee"] {
        print_str(&mut s, line);
        s.line_feed();
        s.carriage_return();
    }
    assert_eq!(s.scrollback_len(), 4);
    assert_eq!(
        s.scroll_extent(),
        (6, 2),
        "content is history plus the screen"
    );

    // At the live bottom the offset is 0, but the *position* is the whole history:
    // the thumb sits at the end of its travel, not the start.
    assert_eq!(s.view_offset(), 0);
    assert_eq!(s.scroll_position(), 4);

    // Scrolling back walks the position down toward the oldest line.
    s.scroll_view_up(3);
    assert_eq!((s.view_offset(), s.scroll_position()), (3, 1));
    s.scroll_view_to_top();
    assert_eq!(s.scroll_position(), 0, "the oldest line kept");
}

#[test]
fn a_thumb_drop_puts_the_view_where_it_landed() {
    let mut s = Screen::new(4, 2);
    for line in ["aa", "bb", "cc", "dd", "ee"] {
        print_str(&mut s, line);
        s.line_feed();
        s.carriage_return();
    }
    // `scroll_view_to` is the inverse of `scroll_position`: every position round-trips.
    for position in 0..=s.scrollback_len() as i32 {
        s.scroll_view_to(position);
        assert_eq!(s.scroll_position(), position);
    }
    // A drag flung past either end of the track rests at the top or the live bottom
    // rather than running off the content.
    s.scroll_view_to(-100);
    assert_eq!(
        s.view_offset(),
        s.scrollback_len(),
        "clamped to the oldest line"
    );
    s.scroll_view_to(9_999);
    assert_eq!(s.view_offset(), 0, "clamped to the live bottom");
}

#[test]
fn the_alt_screen_reports_nothing_to_scroll() {
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd");
    assert!(s.scrollback_len() > 0, "the primary screen has history");
    feed(&mut s, b"\x1b[?1049h"); // enter the alt screen
                                  // The history is still behind it, but the alt screen does not show it, so it
                                  // reports content == viewport: unscrollable, which is what hides the bar.
    assert_eq!(s.scroll_extent(), (2, 2));
    assert_eq!(s.scroll_position(), 0);
    // And a thumb cannot drag a view that does not scroll.
    s.scroll_view_to(2);
    assert_eq!(s.view_offset(), 0);
}

#[test]
fn cursor_moves_clamp_to_bounds() {
    let mut s = Screen::new(5, 4);
    s.move_to(2, 3);
    assert_eq!(s.cursor(), (2, 3));
    s.move_up(10);
    assert_eq!(s.cursor(), (0, 3));
    s.move_back(10);
    assert_eq!(s.cursor(), (0, 0));
    s.move_down(10);
    assert_eq!(s.cursor(), (3, 0));
    s.move_forward(10);
    assert_eq!(s.cursor(), (3, 4));
    // Out-of-range CUP clamps rather than trapping.
    s.move_to(99, 99);
    assert_eq!(s.cursor(), (3, 4));
}

#[test]
fn origin_mode_confines_addressing_to_the_region() {
    let mut s = Screen::new(10, 10);
    s.set_scroll_region(2, 5);
    s.set_mode(6, true, true); // DECOM on; also homes cursor to region top
    assert_eq!(s.cursor(), (2, 0));
    s.move_to(0, 0); // row 0 is relative to the region top (row 2)
    assert_eq!(s.cursor(), (2, 0));
    s.move_to(99, 0); // clamps to the region bottom, not the screen bottom
    assert_eq!(s.cursor(), (5, 0));
}

#[test]
fn erase_line_modes() {
    let mut s = Screen::new(6, 2);
    print_str(&mut s, "abcdef");
    s.move_to(0, 3);
    s.erase_line(0); // cursor to end
    assert_eq!(s.row_string(0).trim_end(), "abc");
    print_str(&mut s, "");
    s.move_to(0, 0);
    print_str(&mut s, "abcdef");
    s.move_to(0, 3);
    s.erase_line(1); // start to cursor inclusive
    assert_eq!(s.cell(0, 0).rune, ' ');
    assert_eq!(s.cell(0, 3).rune, ' ');
    assert_eq!(s.cell(0, 4).rune, 'e');
    s.erase_line(2);
    assert_eq!(s.row_string(0).trim_end(), "");
}

#[test]
fn erase_display_below_and_all() {
    let mut s = Screen::new(4, 3);
    print_str(&mut s, "aa");
    s.line_feed();
    s.carriage_return();
    print_str(&mut s, "bb");
    s.line_feed();
    s.carriage_return();
    print_str(&mut s, "cc");
    s.move_to(1, 0);
    s.erase_display(0); // cursor to end of screen
    assert_eq!(s.row_string(0).trim_end(), "aa");
    assert_eq!(s.row_string(1).trim_end(), "");
    assert_eq!(s.row_string(2).trim_end(), "");
    s.erase_display(2);
    assert_eq!(s.dump().trim(), "");
}

#[test]
fn insert_and_delete_chars() {
    let mut s = Screen::new(6, 1);
    print_str(&mut s, "abcdef");
    s.move_to(0, 1);
    s.delete_chars(2); // remove "bc"
    assert_eq!(s.row_string(0).trim_end(), "adef");
    s.move_to(0, 1);
    s.insert_chars(2); // open two blanks after 'a'
    assert_eq!(s.cell(0, 1).rune, ' ');
    assert_eq!(s.cell(0, 2).rune, ' ');
    assert_eq!(s.cell(0, 3).rune, 'd');
}

#[test]
fn erase_chars_does_not_shift() {
    let mut s = Screen::new(6, 1);
    print_str(&mut s, "abcdef");
    s.move_to(0, 2);
    s.erase_chars(2);
    assert_eq!(s.cell(0, 2).rune, ' ');
    assert_eq!(s.cell(0, 3).rune, ' ');
    assert_eq!(s.cell(0, 4).rune, 'e');
}

/// An erase cannot split a wide glyph: half of one is not a thing that can be drawn.
/// Landing on either half takes the other with it, so the range grows in whichever
/// direction it has to.
#[test]
fn an_erase_cannot_split_a_wide_glyph() {
    // The leader: erasing it must take the spacer, or the spacer is left holding a
    // column for a rune that is gone.
    let mut s = Screen::new(6, 1);
    print_str(&mut s, "a\u{4e00}b");
    s.move_to(0, 1);
    s.erase_chars(1);
    assert_eq!(s.cell(0, 1).rune, ' ');
    assert!(s.cell(0, 2).attrs.is_empty(), "the spacer was not orphaned");
    assert_eq!(s.cell(0, 0).rune, 'a', "its neighbours are untouched");
    assert_eq!(s.cell(0, 3).rune, 'b');

    // The spacer: erasing it must reach *back* for the leader, or the leader keeps
    // drawing two columns into the one cell it still owns.
    let mut s = Screen::new(6, 1);
    print_str(&mut s, "a\u{4e00}b");
    s.move_to(0, 2);
    s.erase_chars(1);
    assert_eq!(s.cell(0, 1).rune, ' ', "the leader went with its spacer");
    assert!(s.cell(0, 1).attrs.is_empty());
    assert_eq!(s.cell(0, 0).rune, 'a');
    assert_eq!(s.cell(0, 3).rune, 'b');
}

/// The same rule through EL, which shares the erase path: a range that starts on a
/// spacer reaches back one cell for its leader.
#[test]
fn erase_line_cannot_split_a_wide_glyph() {
    let mut s = Screen::new(6, 1);
    print_str(&mut s, "a\u{4e00}bcd");
    s.move_to(0, 2); // the spacer
    s.erase_line(0);
    assert_eq!(s.row_string(0).trim_end(), "a");
    assert!(s.cell(0, 1).attrs.is_empty(), "no orphaned leader survives");
}

/// Every shift cuts the row twice, and a wide glyph straddling either cut would be
/// sliced in half. ICH cuts at the cursor and at the last cell the right edge lets
/// survive; DCH cuts at the cursor and at the first cell pulled in over it.
#[test]
fn a_shift_cannot_split_a_wide_glyph() {
    // ICH with the cursor on a spacer: the leader would stay while its second half
    // slid away from it.
    let mut s = Screen::new(6, 1);
    print_str(&mut s, "a\u{4e00}b");
    s.move_to(0, 2);
    s.insert_chars(1);
    assert!(s.cell(0, 1).attrs.is_empty(), "no leader left behind");
    assert_eq!(s.cell(0, 1).rune, ' ');

    // ICH pushing a glyph against the right edge: the edge eats the spacer, and the
    // leader must not be left on the last column with nothing to draw into.
    let mut s = Screen::new(4, 1);
    print_str(&mut s, "ab\u{4e00}");
    s.move_to(0, 0);
    s.insert_chars(1);
    assert_eq!(s.row_string(0).trim_end(), " ab");
    assert!(
        s.cell(0, 3).attrs.is_empty(),
        "no orphan on the last column"
    );

    // DCH deleting a leader: its spacer would slide into the leader's place.
    let mut s = Screen::new(6, 1);
    print_str(&mut s, "a\u{4e00}b");
    s.move_to(0, 1);
    s.delete_chars(1);
    assert_eq!(s.row_string(0).trim_end(), "a b");
    assert!(s.cell(0, 1).attrs.is_empty(), "no spacer left behind");

    // DCH with the cursor on a spacer: the leader behind it would be left drawing
    // across ground it no longer owns.
    let mut s = Screen::new(6, 1);
    print_str(&mut s, "a\u{4e00}b");
    s.move_to(0, 2);
    s.delete_chars(1);
    assert_eq!(s.row_string(0).trim_end(), "a b");
    assert!(s.cell(0, 1).attrs.is_empty());
}

/// A one-column screen cannot hold a wide glyph, and must not pretend to.
///
/// The invariant everything else here rests on is that a leader has its spacer. On a
/// single-column grid the spacer has nowhere to go — the wrap cannot help and the
/// no-autowrap back-up lands on the same cell — so marking the cell a leader would
/// break the invariant on the one screen where it cannot be kept. The glyph goes in
/// clipped instead.
#[test]
fn a_single_column_screen_holds_no_wide_glyph() {
    // Autowrap moves it to the next row (equally narrow); without autowrap the
    // back-up lands on the same cell. Both must place a plain cell, wherever it ends
    // up: the invariant is that no half of a pair is ever left alone.
    for autowrap in [b"\x1b[?7h".as_slice(), b"\x1b[?7l"] {
        let mut s = Screen::new(1, 2);
        feed(&mut s, autowrap);
        print_str(&mut s, "\u{4e00}");
        for row in 0..2 {
            let cell = s.cell(row, 0);
            assert!(
                !cell.is_wide_leader(),
                "{autowrap:?} r{row}: a leader with no spacer to own"
            );
            assert!(
                !cell.is_wide_spacer(),
                "{autowrap:?} r{row}: an orphan spacer"
            );
        }
        assert!(
            (0..2).any(|r| s.cell(r, 0).rune == '\u{4e00}'),
            "{autowrap:?}: the glyph is still on the screen, clipped"
        );
    }
}

/// A shift that touches no wide glyph moves the pair along whole, cut or no cut.
#[test]
fn a_shift_moves_a_wide_glyph_along_intact() {
    let mut s = Screen::new(4, 1);
    print_str(&mut s, "a\u{4e00}");
    s.move_to(0, 0);
    s.insert_chars(1);
    // The pair straddled neither cut, so it simply slid one column right.
    assert_eq!(s.cell(0, 2).rune, '\u{4e00}');
    assert!(s.cell(0, 2).is_wide_leader());
    assert!(s.cell(0, 3).is_wide_spacer());
}

#[test]
fn insert_delete_lines_respect_scroll_region() {
    // The classic scroll-region interaction: IL/DL scroll only within
    // [top, bottom], leaving lines outside the region alone.
    let mut s = Screen::new(3, 5);
    for (r, text) in ["00", "11", "22", "33", "44"].iter().enumerate() {
        s.move_to(r, 0);
        print_str(&mut s, text);
    }
    s.set_scroll_region(1, 3); // rows 1..=3
    s.move_to(1, 0);
    s.insert_lines(1); // push rows 1,2 down to 2,3; row 4 (outside) untouched
    assert_eq!(s.row_string(0).trim_end(), "00");
    assert_eq!(s.row_string(1).trim_end(), ""); // new blank line
    assert_eq!(s.row_string(2).trim_end(), "11");
    assert_eq!(s.row_string(3).trim_end(), "22");
    assert_eq!(s.row_string(4).trim_end(), "44"); // below the region, intact
    s.move_to(1, 0);
    s.delete_lines(1); // pull rows 2,3 back up
    assert_eq!(s.row_string(1).trim_end(), "11");
    assert_eq!(s.row_string(2).trim_end(), "22");
    assert_eq!(s.row_string(4).trim_end(), "44");
}

#[test]
fn scroll_up_down_within_region() {
    let mut s = Screen::new(3, 4);
    for (r, text) in ["00", "11", "22", "33"].iter().enumerate() {
        s.move_to(r, 0);
        print_str(&mut s, text);
    }
    s.scroll_up(1);
    assert_eq!(s.row_string(0).trim_end(), "11");
    assert_eq!(s.row_string(3).trim_end(), "");
    s.scroll_down(1);
    assert_eq!(s.row_string(0).trim_end(), "");
    assert_eq!(s.row_string(1).trim_end(), "11");
}

#[test]
fn reverse_index_scrolls_at_top() {
    let mut s = Screen::new(3, 3);
    s.move_to(0, 0);
    print_str(&mut s, "aa");
    s.move_to(1, 0);
    print_str(&mut s, "bb");
    s.move_to(0, 0);
    s.reverse_index(); // at the top margin: scroll down
    assert_eq!(s.row_string(0).trim_end(), ""); // new blank line at top
    assert_eq!(s.row_string(1).trim_end(), "aa");
    assert_eq!(s.row_string(2).trim_end(), "bb");
}

#[test]
fn tabs_land_on_stops() {
    let mut s = Screen::new(20, 1);
    s.tab();
    assert_eq!(s.cursor(), (0, 8));
    s.tab();
    assert_eq!(s.cursor(), (0, 16));
    s.clear_tab_stop(3); // clear all
    s.move_to(0, 0);
    s.tab();
    assert_eq!(s.cursor(), (0, 19)); // no stops: go to last column
}

/// A colour the terminal was told to use is the colour it reports back.
///
/// The same self-oracle shape as the DECRQSS one below, over the other query path: a
/// theme-aware program (every modern editor) sets a palette entry, asks what it is,
/// and paints to match. No expected string is written here — the property is that
/// `set` and `query` agree, and it holds for every colour, every index, and both
/// terminators.
#[test]
fn a_colour_query_answers_with_the_colour_that_was_set() {
    let mut s = Screen::new(4, 1);
    for index in [0u16, 7, 8, 15, 16, 128, 196, 255] {
        for rgb in [
            Rgb::new(0, 0, 0),
            Rgb::new(255, 255, 255),
            Rgb::new(0x12, 0x34, 0x56),
            Rgb::new(1, 2, 3),
        ] {
            let mut set = Vec::new();
            set.extend_from_slice(format!("\x1b]4;{index};").as_bytes());
            crate::color::write_x11_color(rgb, &mut set);
            set.push(0x07);
            feed(&mut s, &set);
            s.take_responses();

            feed(&mut s, format!("\x1b]4;{index};?\x07").as_bytes());
            let reply = s.take_responses();
            let text = String::from_utf8_lossy(&reply).to_string();
            let body = text
                .strip_prefix(&format!("\u{1b}]4;{index};"))
                .and_then(|t| t.strip_suffix('\u{7}'))
                .unwrap_or_else(|| panic!("index {index}: bad reply {text:?}"));
            assert_eq!(
                crate::color::parse_x11_color(body.as_bytes()),
                Some(rgb),
                "index {index} was set to {rgb:?} and reported {body:?}"
            );
        }
    }

    // The reply mirrors the terminator it was asked with: a client that sent BEL may
    // only be listening for BEL, and one that sent ST for ST.
    feed(&mut s, b"\x1b]4;9;?\x1b\\");
    let reply = s.take_responses();
    assert!(reply.ends_with(b"\x1b\\"), "ST asked, ST answered");
    assert!(!reply.contains(&0x07));
    feed(&mut s, b"\x1b]4;9;?\x07");
    let reply = s.take_responses();
    assert!(reply.ends_with(&[0x07]), "BEL asked, BEL answered");
}

/// DECRQSS answers with a sequence that *reproduces* the setting, so the answer can
/// be checked against itself: ask, replay the reply into a fresh screen, and the two
/// must end up in the same state.
///
/// A self-oracle, and worth more than a table of expected strings would be. Nobody
/// writes down what the reply "should" say, so nobody can write down something wrong
/// and bless it; the property is that a program which saves a setting, changes it,
/// and puts it back gets what it had — which is the entire reason DECRQSS exists.
#[test]
fn a_decrqss_answer_reproduces_the_state_it_reports() {
    for sgr in [
        &b""[..],
        b"\x1b[1m",
        b"\x1b[1;4;31m",
        b"\x1b[3;7;9m",
        b"\x1b[38;2;255;0;128m",
        b"\x1b[48;5;196m",
        b"\x1b[4:3m",
        b"\x1b[4:2;38;5;9;48;2;1;2;3m",
        b"\x1b[0m",
    ] {
        // Set the pen and print through it, so the cell carries what the pen was.
        let mut asked = Screen::new(4, 1);
        feed(&mut asked, sgr);
        feed(&mut asked, b"\x1bP$qm\x1b\\");
        let reply = asked.take_responses();
        print_str(&mut asked, "x");

        // The reply is `DCS 1 $ r <sgr> ST`; take the <sgr> body out of it.
        let text = String::from_utf8_lossy(&reply).to_string();
        let body = text
            .strip_prefix("\u{1b}P1$r")
            .and_then(|t| t.strip_suffix("\u{1b}\\"))
            .unwrap_or_else(|| panic!("{sgr:?}: not a valid DECRQSS answer: {text:?}"));

        // Replay it into a screen that has never seen the original sequence.
        let mut replayed = Screen::new(4, 1);
        feed(&mut replayed, format!("\u{1b}[{body}").as_bytes());
        print_str(&mut replayed, "x");

        assert_eq!(
            replayed.cell(0, 0),
            asked.cell(0, 0),
            "{sgr:?} was reported as {body:?}, which does not reproduce it"
        );
    }
}

/// A resize is a window being dragged, not a program asking to lose the tab stops it
/// set. They survive it, in both directions.
#[test]
fn tab_stops_survive_a_resize() {
    let mut s = Screen::new(25, 3);
    // A custom stop at column 3, alongside the defaults at 0, 8, 16, 24.
    feed(&mut s, b"\x1b[1;4H\x1bH");
    s.resize(80, 3);

    let stops = |s: &mut Screen| -> Vec<usize> {
        let mut out = Vec::new();
        s.move_to(0, 0);
        loop {
            s.tab();
            let (_, col) = s.cursor();
            if out.last() == Some(&col) {
                break; // parked at the last column: no further stops
            }
            out.push(col);
            if out.len() > 16 {
                break;
            }
        }
        out
    };
    assert_eq!(
        stops(&mut s),
        vec![3, 8, 16, 24, 32, 40, 48, 56, 64, 72, 79],
        "the custom stop survived, and the defaults extend into the new width"
    );

    // And narrowing does not forget the stops it merely hid: widening finds them.
    s.resize(25, 3);
    assert_eq!(stops(&mut s), vec![3, 8, 16, 24]);
    s.resize(80, 3);
    assert_eq!(
        stops(&mut s),
        vec![3, 8, 16, 24, 32, 40, 48, 56, 64, 72, 79]
    );
}

/// "Clear every tab stop" has to mean every one, including in columns the window has
/// not grown into yet.
///
/// The table used to be sized to the width, so `TBC 3` could only clear as far as the
/// screen went and a later widening seeded fresh defaults into the columns it had
/// emptied — the stops came back from the dead. A program that clears the stops and
/// sets its own gets someone else's every time the window is dragged wider.
#[test]
fn clearing_every_tab_stop_survives_a_widening() {
    let first_stop = |s: &mut Screen| -> usize {
        s.move_to(0, 0);
        s.tab();
        s.cursor().1
    };
    let mut s = Screen::new(80, 3);
    feed(&mut s, b"\x1b[3g");
    assert_eq!(
        first_stop(&mut s),
        79,
        "no stops left: HT parks at the margin"
    );

    s.resize(100, 3);
    assert_eq!(
        first_stop(&mut s),
        99,
        "the cleared stops stayed cleared past the old width"
    );

    // And a stop set after the clear is the only one there, at either width.
    feed(&mut s, b"\x1b[1;11H\x1bH");
    assert_eq!(first_stop(&mut s), 10);
    s.resize(120, 3);
    assert_eq!(first_stop(&mut s), 10);
}

/// TBC defines 0 (this column) and 3 (all). ECMA-48's other parameters speak of
/// line tab stops, which no terminal in use has, and xterm drops them — so a stop
/// must survive them. vttest's tab screen tests exactly this, by sending both at a
/// live stop and requiring the two lines it then prints to come out identical.
#[test]
fn tbc_ignores_the_parameters_it_does_not_define() {
    let mut s = Screen::new(20, 1);
    s.move_to(0, 8); // a default tab stop
    for mode in [1, 2, 4, 5, 9] {
        s.clear_tab_stop(mode);
    }
    s.move_to(0, 0);
    s.tab();
    assert_eq!(s.cursor(), (0, 8), "the stop survived every undefined TBC");

    // 0 is the one that clears this column, and the default when absent.
    s.move_to(0, 8);
    s.clear_tab_stop(0);
    s.move_to(0, 0);
    s.tab();
    assert_eq!(s.cursor(), (0, 16), "TBC 0 cleared the stop at column 8");
}

#[test]
fn sgr_sets_the_pen_and_print_applies_it() {
    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b[1;31m"); // bold, red foreground
    s.print('x');
    let c = s.cell(0, 0);
    assert!(c.attrs.contains(Attrs::BOLD));
    assert_eq!(c.fg, Color::Ansi(1));
    feed(&mut s, b"\x1b[0m"); // reset
    s.print('y');
    let c = s.cell(0, 1);
    assert!(c.attrs.is_empty());
    assert_eq!(c.fg, Color::Default);
}

#[test]
fn sgr_extended_colors() {
    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b[38;5;200m"); // 256-color foreground
    assert_eq!(s.pen.fg, Color::Indexed(200));
    feed(&mut s, b"\x1b[48;2;10;20;30m"); // truecolor background
    assert_eq!(s.pen.bg, Color::Rgb(10, 20, 30));
    feed(&mut s, b"\x1b[90m"); // bright black foreground -> ANSI 8
    assert_eq!(s.pen.fg, Color::Ansi(8));
}

/// Screen with grapheme clustering (`?2027`) turned on, as a program that asked for it
/// would have it.
fn clustering(cols: usize, rows: usize) -> Screen {
    let mut s = Screen::new(cols, rows);
    feed(&mut s, b"\x1b[?2027h");
    s
}

#[test]
fn osc_7_takes_the_shells_word_for_the_directory() {
    let mut s = Screen::new(10, 2);
    assert_eq!(s.reported_cwd(), None);
    feed(&mut s, b"\x1b]7;file://host/home/testuser/projects\x07");
    assert_eq!(s.reported_cwd(), Some("/home/testuser/projects"));

    // A filename may hold any byte a filename may hold, which is why the path is
    // encoded at all: this one has a space and a semicolon in it, and the semicolon
    // would otherwise have looked like the end of the field.
    feed(&mut s, b"\x1b]7;file://host/tmp/a%20b%3Bc\x07");
    assert_eq!(s.reported_cwd(), Some("/tmp/a b;c"));

    // A bare path (some shells send one) is accepted; a stray `%` is a literal `%`,
    // because a directory that is merely unusual must not be mangled.
    feed(&mut s, b"\x1b]7;/plain/path\x07");
    assert_eq!(s.reported_cwd(), Some("/plain/path"));
    feed(&mut s, b"\x1b]7;file://h/100%pure\x07");
    assert_eq!(s.reported_cwd(), Some("/100%pure"));
}

#[test]
fn osc_133_marks_where_each_command_began_and_how_it_ended() {
    // The shell tells the terminal what it could never work out for itself: where one
    // command ended and the next began. Everything good downstream falls out of this.
    let mut s = Screen::new(20, 6);
    feed(&mut s, b"\x1b]133;A\x07$ ls\r\n"); // a prompt on row 0
    feed(&mut s, b"\x1b]133;C\x07"); // output starts on row 1
    feed(&mut s, b"a.txt\r\n");
    feed(&mut s, b"\x1b]133;D;0\x07"); // and it succeeded

    feed(&mut s, b"\x1b]133;A\x07$ false\r\n"); // a second prompt on row 2
    feed(&mut s, b"\x1b]133;C\x07\x1b]133;D;1\x07"); // which failed

    let prompts = s.prompts();
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0].row, s.abs_row(0));
    assert_eq!(prompts[0].output, Some(s.abs_row(1)));
    assert_eq!(prompts[0].exit, Some(0), "ls worked");
    assert_eq!(prompts[1].row, s.abs_row(2));
    assert_eq!(prompts[1].exit, Some(1), "false did not");
}

#[test]
fn the_exit_code_belongs_to_the_command_that_just_ended() {
    // These are the bytes a real bash emits, in the order it emits them. The ordering
    // is the subtlety: `D` reports the status of the command that *just finished*, and
    // it arrives immediately before the `A` that starts the next prompt — so the code
    // attaches to the prompt behind it, never the one in front.
    let mut s = Screen::new(20, 6);
    feed(&mut s, b"\x1b]133;A\x07\x1b]7;file://h/tmp\x07$ false\r\n");
    feed(&mut s, b"\x1b]133;D;1\x07"); // `false` failed...
    feed(&mut s, b"\x1b]133;A\x07$ "); // ...and only now does the next prompt open

    let prompts = s.prompts();
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0].exit, Some(1), "the failure landed on `false`");
    assert_eq!(prompts[1].exit, None, "and not on the prompt after it");
    assert_eq!(s.reported_cwd(), Some("/tmp"));
}

#[test]
fn a_prompt_redrawn_in_place_is_still_the_same_prompt() {
    // zsh with syntax highlighting redraws its prompt on every keystroke, re-marking
    // the same row each time. Counting those as separate prompts would fill the list
    // with hundreds of copies of the one you are typing at.
    let mut s = Screen::new(20, 4);
    for _ in 0..50 {
        feed(&mut s, b"\x1b]133;A\x07");
    }
    assert_eq!(s.prompts().len(), 1);
}

#[test]
fn a_prompt_is_marked_on_the_cursor_row_even_while_the_view_is_scrolled_back() {
    // The shell marks a prompt when it prints one, and that is wherever the *cursor*
    // is — which has nothing to do with what the user happens to be looking at. Run a
    // slow command, scroll back to read while it works, and the prompt it prints on
    // finishing must still be marked on the row it was printed on. Reading the mark
    // off the display instead lands it `view_offset` rows up in history, on a line the
    // command printed, which then misdirects both the prompt jump and the reflow's
    // freeze point.
    let mut s = Screen::with_scrollback(20, 4, 100);
    feed(&mut s, b"\x1b]133;A\x07$ slow\r\n");
    feed(&mut s, b"\x1b]133;C\x07");
    for i in 0..10 {
        feed(&mut s, format!("out {i}\r\n").as_bytes());
    }
    s.scroll_view_up(5);
    assert_eq!(s.view_offset(), 5, "the user is reading history");

    feed(&mut s, b"\x1b]133;D;0\x07\x1b]133;A\x07$ ");
    let live = s
        .primary
        .abs_of(s.primary.scrollback.len() + s.primary.cursor.row);
    assert_eq!(
        s.prompts().last().map(|p| p.row),
        Some(live),
        "the mark names the row the prompt was printed on"
    );
}

#[test]
fn jumping_to_a_prompt_scrolls_past_everything_the_command_printed() {
    // The payoff, and the thing a scrollbar can never do: a scrollbar knows how far you
    // have come, this knows *what* you have come past. Ten screens of build log go by
    // in one keystroke, because the terminal knows where the command that printed them
    // started.
    let mut s = Screen::new(20, 4);
    feed(&mut s, b"\x1b]133;A\x07$ build\r\n");
    for i in 0..30 {
        feed(&mut s, format!("log line {i}\r\n").as_bytes());
    }
    feed(&mut s, b"\x1b]133;A\x07$ ");

    let first = s.prompts()[0].row;
    assert!(s.view_offset() == 0, "we are at the live bottom");
    assert!(s.scroll_to_prompt(true), "there is a prompt behind us");
    assert_eq!(s.abs_row(0), first, "and it is now the top line on screen");
    // Read through the *view*, not the live screen: the whole point is that we are
    // looking at history, and `row_string` would show what is live under it.
    let seen: String = (0..20).map(|c| s.view_cell(0, c).rune).collect();
    assert_eq!(seen.trim_end(), "$ build");

    // Nothing further back to find.
    assert!(!s.scroll_to_prompt(true));
    // And forward again, to the prompt we left. It sits near the live bottom, so it
    // cannot be lifted to the top row — there is not enough text under it to fill the
    // screen — and it comes to rest as high as the history allows. What matters is that
    // it is on screen.
    let last = s.prompts()[1].row;
    assert!(s.scroll_to_prompt(false));
    assert!(
        s.display_row(last).is_some(),
        "the prompt we jumped to is visible"
    );
}

#[test]
fn marks_are_dropped_when_the_rows_they_name_stop_meaning_anything() {
    // They hold row ids, and a reset renumbers the world. Offering to jump to a prompt
    // that is no longer there is worse than offering nothing — it is the same rule the
    // selection follows, for the same reason.
    let mut s = Screen::new(20, 4);
    feed(&mut s, b"\x1b]133;A\x07$ ls\r\n");
    assert_eq!(s.prompts().len(), 1);
    feed(&mut s, b"\x1b[2J"); // a full erase ends the epoch
    assert!(s.prompts().is_empty());
}

#[test]
fn without_clustering_an_astronaut_is_two_emoji_in_four_columns() {
    // The default, and it must not change: `wcwidth` counts scalars, so the woman is
    // two columns, the ZWJ is none, and the rocket is two more. An application that
    // measures its output that way — which is most of them — needs the terminal to
    // agree, and this is the terminal agreeing.
    let mut s = Screen::new(10, 1);
    print_str(&mut s, "\u{1F469}\u{200D}\u{1F680}"); // 👩‍🚀
    assert_eq!(s.cursor(), (0, 4), "four columns");
    assert_eq!(s.cell(0, 0).rune, '\u{1F469}', "a woman");
    assert_eq!(s.cell(0, 2).rune, '\u{1F680}', "and, separately, a rocket");
}

/// An ASCII byte can be the *base* of a cluster, so the bulk-write fast path has to
/// stand aside under `?2027`.
///
/// `#` is one column. `#️⃣` is the same byte plus a presentation selector and a
/// keycap, and it is two columns and one character. The bulk path places cells
/// without anchoring a cluster, so a run through it left the VS16 nothing to join and
/// the keycap came out half width — invisible to every test that only prints
/// non-ASCII bases, which is what an astronaut is.
#[test]
fn an_ascii_base_still_starts_a_cluster() {
    for base in ['#', '1', '*'] {
        let mut s = clustering(10, 1);
        print_str(&mut s, &format!("{base}\u{fe0f}\u{20e3}"));
        assert_eq!(s.cursor(), (0, 2), "{base}\u{fe0f}\u{20e3} is two columns");
        assert_eq!(s.cell(0, 0).rune, base);
        assert!(s.cell(0, 0).is_wide_leader(), "{base}: the base widened");
        assert!(s.cell(0, 1).is_wide_spacer());
    }

    // A whole ASCII run still works, and the last byte of it is what the next scalar
    // joins: the fallback must not lose the anchor either.
    let mut s = clustering(10, 1);
    print_str(&mut s, "ab#\u{fe0f}");
    assert_eq!(s.cursor(), (0, 4), "'a','b' one column each, '#'+VS16 two");
    assert!(s.cell(0, 2).is_wide_leader());

    // And with the mode off, the same bytes are plain wcwidth again: a zero-width
    // selector attaches to a one-column '#'.
    let mut s = Screen::new(10, 1);
    print_str(&mut s, "#\u{fe0f}\u{20e3}");
    assert_eq!(
        s.cursor(),
        (0, 1),
        "?2027 off: scalars are counted, not clusters"
    );
}

#[test]
fn clustering_makes_an_astronaut_one_character_in_two_columns() {
    // The same three scalars, counted as the one character a user sees. The rocket does
    // not start a cell of its own: it joins the cluster already in the one behind it,
    // and the whole thing is two columns — which is what an application that enabled the
    // mode is also computing, so the two agree about where the next column is.
    let mut s = clustering(10, 1);
    print_str(&mut s, "\u{1F469}\u{200D}\u{1F680}");
    assert_eq!(s.cursor(), (0, 2), "two columns, not four");
    assert_eq!(s.cell(0, 0).rune, '\u{1F469}');
    assert!(s.cell(0, 0).is_wide_leader());
    assert!(s.cell(0, 1).is_wide_spacer());
    // The tail of the cluster rides on the base cell, which is exactly the shape the
    // renderer already shapes as one glyph. The astronaut is drawn, not the woman and
    // the rocket.
    assert_eq!(
        s.marks_at(0, 0).collect::<Vec<_>>(),
        ['\u{200D}', '\u{1F680}'],
        "the joiner and the rocket joined the woman"
    );
    // And nothing is left in the columns the scalars would have taken.
    assert_eq!(s.cell(0, 2).rune, ' ');
}

#[test]
fn a_cluster_that_grows_wide_takes_the_column_beside_it() {
    // The fiddly case, and the reason a cluster's width cannot be known when it starts.
    // `☀` is one column. `☀️` — the very same character with an emoji presentation
    // selector after it — is two. The sun was already sitting in a narrow cell by the
    // time the selector arrived, so the cell has to grow under it.
    let mut s = clustering(10, 1);
    print_str(&mut s, "\u{2600}"); // ☀ alone: narrow
    assert_eq!(s.cursor(), (0, 1));
    assert!(!s.cell(0, 0).is_wide_leader());

    print_str(&mut s, "\u{FE0F}"); // ...and now make it an emoji
    assert_eq!(s.cursor(), (0, 2), "it grew into the column beside it");
    assert!(s.cell(0, 0).is_wide_leader());
    assert!(s.cell(0, 1).is_wide_spacer());
}

#[test]
fn a_flag_is_two_narrow_scalars_that_add_up_to_a_wide_character() {
    // Neither regional indicator is wide. The pair of them is an emoji, and there is
    // nothing in either scalar's width that says so — which is exactly why the width of
    // a cluster is a property of the cluster and not a sum of its parts.
    let mut s = clustering(10, 1);
    print_str(&mut s, "\u{1F1F3}\u{1F1F4}"); // 🇳🇴
    assert_eq!(s.cursor(), (0, 2));
    assert!(s.cell(0, 0).is_wide_leader());
    assert_eq!(s.marks_at(0, 0).collect::<Vec<_>>(), ['\u{1F1F4}']);
}

#[test]
fn a_cluster_cannot_span_a_cursor_move_or_a_control_byte() {
    // The anchor is only good while the cursor has not moved. Otherwise a ZWJ arriving
    // after a jump across the screen would graft a rocket onto whatever happened to be
    // sitting there — a cell that may since have been overwritten by something else
    // entirely.
    let mut s = clustering(10, 2);
    print_str(&mut s, "\u{1F469}"); // a woman at (0,0)
    feed(&mut s, b"\x1b[2;5H"); // jump away
    print_str(&mut s, "\u{200D}\u{1F680}"); // a joiner and a rocket, elsewhere
    assert!(
        s.marks_at(0, 0).next().is_none(),
        "the woman did not acquire a rocket from across the screen"
    );

    // Same for a control byte: a newline ends the cluster.
    let mut s = clustering(10, 2);
    print_str(&mut s, "\u{1F469}");
    feed(&mut s, b"\r\n");
    print_str(&mut s, "\u{1F680}");
    assert!(s.marks_at(0, 0).next().is_none());
    assert_eq!(s.cell(1, 0).rune, '\u{1F680}', "the rocket stands alone");
}

#[test]
fn clustering_is_reported_so_a_program_can_agree_with_us() {
    // The mode only works if the application knows about it: both width answers are
    // legitimate, and the disaster is not picking the wrong one, it is the two ends
    // picking differently. So DECRQM has to tell the truth about this one above all.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b[?2027$p");
    assert_eq!(s.take_responses(), b"\x1b[?2027;2$y", "known, and off");
    feed(&mut s, b"\x1b[?2027h\x1b[?2027$p");
    assert_eq!(s.take_responses(), b"\x1b[?2027;1$y", "known, and on");
}

#[test]
fn rep_repeats_the_last_character_and_nothing_else() {
    // ncurses paints a run of one character with this rather than sending it a
    // thousand times, so it turns up in real output more than its obscurity suggests.
    let mut s = Screen::new(10, 3);
    feed(&mut s, b"-\x1b[4b"); // a dash, then four more
    assert_eq!(s.row_string(0).trim_end(), "-----");

    // It repeats the last *printed* character, so an escape sequence in between
    // changes the style and not what gets repeated.
    feed(&mut s, b"\x1b[2;1Hx\x1b[31m\x1b[2b");
    assert_eq!(s.row_string(1).trim_end(), "xxx");
    assert_eq!(
        s.cell(1, 2).fg,
        Color::Ansi(1),
        "the repeat takes the new pen"
    );

    // But a control byte ends the run: a REP straight after a newline has nothing to
    // repeat, and must print nothing rather than a screenful of the line above.
    feed(&mut s, b"\r\n\x1b[5b");
    assert_eq!(s.row_string(2).trim_end(), "");
    assert_eq!(s.cursor(), (2, 0));

    // A bare `CSI b` is one repeat, like every other count.
    feed(&mut s, b"z\x1b[b");
    assert_eq!(s.row_string(2).trim_end(), "zz");
}

#[test]
fn mode_12_blinks_the_cursor_without_reshaping_it() {
    // DECSCUSR binds shape and blink together in one parameter, so a program that
    // wants to change only the blink has no way to say it. `?12` is that way.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b[4 q"); // a steady underline
    assert_eq!(s.cursor_style(), CursorStyle::Underline);
    assert!(!s.cursor_blinks());

    feed(&mut s, b"\x1b[?12h");
    assert!(s.cursor_blinks(), "blinking now");
    assert_eq!(
        s.cursor_style(),
        CursorStyle::Underline,
        "still an underline"
    );

    feed(&mut s, b"\x1b[?12l");
    assert!(!s.cursor_blinks());
    // And it reports honestly, like every other mode we act on.
    feed(&mut s, b"\x1b[?12$p");
    assert_eq!(s.take_responses(), b"\x1b[?12;2$y");
}

#[test]
fn hts_sets_a_tab_stop_and_cbt_walks_back_to_it() {
    // `set_tab_stop` existed but nothing ever called it: a program could clear tab
    // stops and never set them, so `ESC H` did nothing and only the default every-8
    // stops existed. CBT (`CSI Z`) had no arm at all.
    let mut s = Screen::new(40, 2);
    feed(&mut s, b"\x1b[3g"); // TBC 3: clear every stop, default or otherwise
    feed(&mut s, b"\x1b[1;5H\x1bH"); // column 5 (1-based): set a stop here
    feed(&mut s, b"\x1b[1;20H\x1bH"); // and another at column 20
                                      // A tab from home now lands on the first stop we set, not on column 8.
    feed(&mut s, b"\x1b[1;1H\t");
    assert_eq!(s.cursor(), (0, 4));
    feed(&mut s, b"\t");
    assert_eq!(s.cursor(), (0, 19));
    // And CBT walks back through them.
    feed(&mut s, b"\x1b[Z");
    assert_eq!(s.cursor(), (0, 4));
    feed(&mut s, b"\x1b[Z");
    assert_eq!(
        s.cursor(),
        (0, 0),
        "past the first stop it stops at column 0"
    );
    feed(&mut s, b"\x1b[Z");
    assert_eq!(s.cursor(), (0, 0), "and does not run off the left edge");
}

#[test]
fn decstr_resets_the_settings_and_keeps_the_screen() {
    // The point of a soft reset: put the *settings* back without throwing away what
    // is on the screen. A program that has wedged the terminal sends this to hand a
    // sane state back to the shell, which is why RIS would be too big a hammer.
    let mut s = Screen::new(10, 4);
    print_str(&mut s, "keep me");
    feed(&mut s, b"\x1b[1;31m"); // bold red
    feed(&mut s, b"\x1b[?1h"); // DECCKM: application cursor keys
    feed(&mut s, b"\x1b[?6h"); // DECOM: origin mode
    feed(&mut s, b"\x1b[?25l"); // hide the cursor
    feed(&mut s, b"\x1b[4h"); // IRM: insert mode
    feed(&mut s, b"\x1b[2;3r"); // a scroll region
    feed(&mut s, b"\x1b="); // DECKPAM: application keypad
    feed(&mut s, b"\x1b[3;2H"); // park the cursor somewhere
    feed(&mut s, b"\x1b[!p"); // DECSTR

    assert_eq!(s.row_string(0).trim_end(), "keep me", "the screen survived");
    assert_eq!(s.cursor(), (2, 1), "and so did the cursor position");
    assert!(s.pen.attrs.is_empty(), "rendition reset");
    assert_eq!(s.pen.fg, Color::Default);
    assert!(s.cursor_visible());
    assert!(!s.app_cursor_keys());
    assert!(!s.keypad_app());
    assert!(!s.origin_mode);
    assert!(!s.insert_mode);
    let region = (s.active().scroll_top, s.active().scroll_bottom);
    assert_eq!(region, (0, 3), "margins back to the full screen");

    // The DEC table says autowrap is *reset* by DECSTR. xterm leaves it on, and a
    // terminal that stops wrapping the moment a program soft-resets would mangle
    // every long line the shell prints afterwards. We follow xterm.
    assert!(s.autowrap, "autowrap stays on: xterm, not the DEC table");
    feed(&mut s, b"\x1b[1;1Habcdefghijkl"); // 12 chars into a 10-column row
    assert!(s.row_wraps(0), "so a long line still wraps");
    assert_eq!(s.row_string(1).trim_end(), "kl");
}

#[test]
fn decstr_drops_the_saved_cursor() {
    // DECSC state resets to "home position", so a DECRC after a soft reset must not
    // resurrect a cursor from before it.
    let mut s = Screen::new(10, 4);
    feed(&mut s, b"\x1b[3;5H\x1b7"); // park and save
    feed(&mut s, b"\x1b[!p"); // soft reset
    feed(&mut s, b"\x1b[1;1H\x1b8"); // home, then restore
    assert_eq!(s.cursor(), (0, 0), "there was nothing to restore");
}

#[test]
fn a_private_marker_does_not_execute_its_ansi_namesake() {
    // Three sequences that used to be executed as entirely different commands,
    // because the final byte was matched without looking at the marker in front of
    // it. Each must now be a no-op: dropping an unknown sequence is always safe,
    // guessing at it is not.
    let mut s = Screen::new(10, 5);

    // XTRESTORE (`CSI ? Ps r`) restores saved private modes. It used to land on
    // DECSTBM and reset the scroll region, homing the cursor.
    feed(&mut s, b"\x1b[2;4r"); // a real DECSTBM: rows 2..4
    feed(&mut s, b"\x1b[5;3H"); // park the cursor inside it
    feed(&mut s, b"\x1b[?1r"); // XTRESTORE (restore DECCKM) — must do nothing
    assert_eq!(s.cursor(), (4, 2), "the cursor did not move");
    let region = (s.active().scroll_top, s.active().scroll_bottom);
    assert_eq!(region, (1, 3), "the scroll region survived");

    // XTSMGRAPHICS (`CSI ? Pi;Pa;Pv S`) is a sixel geometry query. It used to land
    // on SU and scroll the screen.
    let mut s = Screen::new(10, 3);
    feed(&mut s, b"top\r\nmid\r\nlow");
    feed(&mut s, b"\x1b[?1;1;0S");
    assert_eq!(s.row_string(0).trim_end(), "top", "nothing scrolled");

    // `CSI > 4 h` is not a mode set; it used to turn on insert mode (IRM).
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b[>4h");
    feed(&mut s, b"ab");
    feed(&mut s, b"\x1b[1;1H");
    feed(&mut s, b"X");
    assert_eq!(
        s.row_string(0).trim_end(),
        "Xb",
        "X overwrote, not inserted"
    );
}

#[test]
fn an_extended_color_reads_the_sub_parameter_form_too() {
    // The same colours, spelled with colons. Which spelling a program uses depends
    // on what its terminfo told it, so both have to work.
    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b[38:5:200m");
    assert_eq!(s.pen.fg, Color::Indexed(200));
    // With the (unused) colour-space slot, as the ITU form has it.
    feed(&mut s, b"\x1b[48:2::10:20:30m");
    assert_eq!(s.pen.bg, Color::Rgb(10, 20, 30));
    // And without it, which is just as common.
    feed(&mut s, b"\x1b[38:2:1:2:3m");
    assert_eq!(s.pen.fg, Color::Rgb(1, 2, 3));
}

#[test]
fn a_curly_underline_no_longer_swallows_the_colours_with_it() {
    // The bug this all started from, and the exact sequence nvim sends to underline
    // a diagnostic. The whole CSI used to be dropped on sight of the first colon, so
    // the *colour* went with it: an error came out unstyled and uncoloured.
    //
    // We still draw only a solid underline, so the style is discarded — but cleanly,
    // one parameter at a time, instead of taking the rest of the sequence down with
    // it.
    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b[4:3;58:2::255:0:0;38:2::0:255:0m");
    assert!(s.pen.attrs.contains(Attrs::UNDERLINE), "underlined");
    assert_eq!(s.pen.fg, Color::Rgb(0, 255, 0), "and the colour survived");
}

#[test]
fn an_underline_carries_its_style_without_growing_a_cell() {
    // The design decision, pinned. A `Cell` is 16 bytes and the damage diff compares
    // two screenfuls of them every painted frame; the style rides in the spare bits of
    // the attribute bitfield precisely so squiggling one diagnostic does not cost the
    // whole grid 25%. If this test starts failing, someone has paid that price.
    assert_eq!(std::mem::size_of::<Cell>(), 16);

    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b[4:3max"); // the curly underline nvim marks an error with
    let cell = s.cell(0, 0);
    assert!(cell.attrs.contains(Attrs::UNDERLINE));
    assert_eq!(cell.attrs.underline_style(), UnderlineStyle::Curly);

    for (seq, want) in [
        (&b"\x1b[4m"[..], UnderlineStyle::Single),
        (&b"\x1b[4:1m"[..], UnderlineStyle::Single),
        (&b"\x1b[4:2m"[..], UnderlineStyle::Double),
        (&b"\x1b[4:3m"[..], UnderlineStyle::Curly),
        (&b"\x1b[4:4m"[..], UnderlineStyle::Dotted),
        (&b"\x1b[4:5m"[..], UnderlineStyle::Dashed),
        // A style we have never heard of is still an underline: better a straight
        // line where a program wanted a fancy one than no mark at all.
        (&b"\x1b[4:9m"[..], UnderlineStyle::Single),
    ] {
        feed(&mut s, b"\x1b[0m");
        feed(&mut s, seq);
        assert!(s.pen.attrs.contains(Attrs::UNDERLINE), "{seq:?}");
        assert_eq!(s.pen.attrs.underline_style(), want, "{seq:?}");
    }
}

#[test]
fn turning_the_underline_off_takes_its_style_with_it() {
    // Otherwise a stale style would sit in the bits, and the next bare `SGR 4` would
    // come out curly because something squiggled an hour ago.
    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b[4:3m"); // curly
    feed(&mut s, b"\x1b[24m"); // underline off
    assert!(!s.pen.attrs.contains(Attrs::UNDERLINE));
    assert!(
        s.pen.attrs.is_empty(),
        "no attribute survives, style included"
    );
    feed(&mut s, b"\x1b[4m"); // a plain underline, and it had better be plain
    assert_eq!(s.pen.attrs.underline_style(), UnderlineStyle::Single);

    // `4:0` is the other way to say it.
    feed(&mut s, b"\x1b[4:3m\x1b[4:0m");
    assert!(!s.pen.attrs.contains(Attrs::UNDERLINE));
    assert!(s.pen.attrs.is_empty());
}

#[test]
fn an_underline_style_of_zero_turns_the_underline_off() {
    // `4:0` is "no underline" — the one sub-parameter of 4 that is not an underline.
    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b[4m");
    assert!(s.pen.attrs.contains(Attrs::UNDERLINE));
    feed(&mut s, b"\x1b[4:0m");
    assert!(!s.pen.attrs.contains(Attrs::UNDERLINE));
    // Every other style is *an* underline, whatever we end up drawing for it.
    for style in [b"\x1b[4:1m", b"\x1b[4:2m", b"\x1b[4:3m", b"\x1b[4:4m"] {
        feed(&mut s, b"\x1b[4:0m");
        feed(&mut s, style);
        assert!(s.pen.attrs.contains(Attrs::UNDERLINE), "{style:?}");
    }
}

#[test]
fn an_unimplemented_extended_color_is_consumed_not_misread() {
    // SGR 58 is the underline colour, which we have nowhere to store. The danger is
    // not that we drop it — it is that we drop it *without consuming its arguments*,
    // in which case the `2` reads as dim, the `0` as a full reset, and the rest of
    // the line comes out in the wrong style. Consume, then ignore.
    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b[31m"); // red, to be left alone
    feed(&mut s, b"\x1b[58;2;255;0;0;1m"); // underline colour, then bold
    assert!(
        s.pen.attrs.contains(Attrs::BOLD),
        "the bold after it landed"
    );
    assert!(
        !s.pen.attrs.contains(Attrs::DIM),
        "the 2 was not read as dim"
    );
    assert_eq!(s.pen.fg, Color::Ansi(1), "and nothing reset the colour");
    // The same in the sub-parameter spelling, which consumes no extra parameters.
    feed(&mut s, b"\x1b[0;32;58:2::1:2:3;3m");
    assert!(s.pen.attrs.contains(Attrs::ITALIC));
    assert_eq!(s.pen.fg, Color::Ansi(2));
}

#[test]
fn alt_screen_is_separate_and_restores() {
    let mut s = Screen::new(6, 2);
    print_str(&mut s, "main");
    s.set_mode(1049, true, true); // enter alt, save cursor
    assert!(s.is_alt());
    assert_eq!(s.dump().trim(), ""); // alt starts cleared
    assert_eq!(s.cursor(), (0, 4), "the cursor carries onto the alt screen");
    s.move_to(0, 0);
    print_str(&mut s, "alt");
    assert_eq!(s.row_string(0).trim_end(), "alt");
    s.set_mode(1049, true, false); // leave alt, restore cursor
    assert!(!s.is_alt());
    assert_eq!(s.row_string(0).trim_end(), "main"); // primary intact
    assert_eq!(s.cursor(), (0, 4)); // cursor restored to after "main"
}

/// The alt screen is a different set of cells, not a different cursor: switching
/// carries the cursor across rather than homing it. `?1049h` saves the cursor and
/// clears the alt, and a program that restores expects to land where it started.
#[test]
fn the_cursor_carries_onto_the_alt_screen() {
    let mut s = Screen::new(10, 3);
    s.move_to(2, 5);
    feed(&mut s, b"\x1b[?1049h");
    assert_eq!(s.cursor(), (2, 5), "not homed to the origin");
    feed(&mut s, b"x");
    assert_eq!(s.cell(2, 5).rune, 'x');
}

/// The cursor is one cursor, and it carries *both* ways.
///
/// Carrying it only on the way in would be a cursor that is shared entering the alt
/// screen and per-buffer leaving it. `?1049` cannot tell, because its restore lands
/// on top; `?47`/`?1047` — what a program using the alt screen without the
/// save/restore pair sends — can.
#[test]
fn the_cursor_carries_off_the_alt_screen_as_well_as_onto_it() {
    let mut s = Screen::new(10, 3);
    print_str(&mut s, "main");
    assert_eq!(s.cursor(), (0, 4));

    feed(&mut s, b"\x1b[?1047h");
    assert_eq!(s.cursor(), (0, 4), "carried onto the alt screen");
    feed(&mut s, b"\x1b[3;1H");
    assert_eq!(s.cursor(), (2, 0));

    feed(&mut s, b"\x1b[?1047l");
    assert_eq!(
        s.cursor(),
        (2, 0),
        "and back off it: one cursor, not one per buffer"
    );
    // The primary's cells are of course untouched by any of it.
    assert_eq!(s.row_string(0).trim_end(), "main");
}

/// Each screen owns its DECSC slot, and it survives a trip to the other screen and
/// back: a program may save on the alt screen, leave, return, and restore.
#[test]
fn each_screen_keeps_its_own_saved_cursor_across_a_switch() {
    let mut s = Screen::new(10, 4);
    feed(&mut s, b"\x1b[?1049h"); // to the alt screen
    s.move_to(3, 7);
    feed(&mut s, b"\x1b7"); // DECSC on the alt screen
    feed(&mut s, b"\x1b[?1049l"); // back to the primary
    feed(&mut s, b"\x1b[?1049h"); // and to the alt screen again
    feed(&mut s, b"\x1b[1;1H"); // somewhere else entirely
    feed(&mut s, b"\x1b8"); // DECRC finds what the alt screen saved
    assert_eq!(s.cursor(), (3, 7));
}

#[test]
fn save_and_restore_cursor() {
    let mut s = Screen::new(10, 5);
    s.move_to(2, 4);
    feed(&mut s, b"\x1b[1m");
    s.save_cursor();
    s.move_to(0, 0);
    feed(&mut s, b"\x1b[0m");
    s.restore_cursor();
    assert_eq!(s.cursor(), (2, 4));
    assert!(s.pen.attrs.contains(Attrs::BOLD));
}

#[test]
fn reset_returns_to_power_on_state() {
    let mut s = Screen::new(6, 3);
    feed(&mut s, b"\x1b[31m");
    print_str(&mut s, "junk");
    s.set_scroll_region(1, 2);
    s.reset();
    assert_eq!(s.dump().trim(), "");
    assert_eq!(s.cursor(), (0, 0));
    assert_eq!(s.pen.fg, Color::Default);
    assert_eq!(s.dimensions(), (6, 3));
}

#[test]
fn steady_scroll_is_allocation_free_after_warmup() {
    // Once scrollback is full, scrolling must recycle row storage. We can at
    // least assert the ring stays bounded and correct over many scrolls.
    let mut s = Screen::new(4, 3);
    for i in 0..50_000u32 {
        s.carriage_return();
        let d = char::from(b'0' + (i % 10) as u8);
        s.print(d);
        s.line_feed();
    }
    // Never grows past the visible height; the bottom holds the last digit.
    assert_eq!(s.dimensions(), (4, 3));
    assert_eq!(s.primary.lines.len(), 3);
    assert!(s.primary.scrollback.len() <= DEFAULT_SCROLLBACK);
}

// ---- end-to-end: bytes through the parser into the grid ----------------

fn feed(s: &mut Screen, bytes: &[u8]) {
    let mut p = crate::vt::Parser::new();
    p.advance_bytes(s, bytes);
}

#[test]
fn violent_resize_never_panics_and_keeps_invariants() {
    // "Resize violently narrow -> wide -> narrow" reportedly crashed the live app. This
    // drives the same at the grid level: thousands of resizes across the whole width/
    // height range, a shell prompt with OSC 133 marks (so the freeze path runs), wide
    // glyphs (so the new_cols == 1 degenerate path runs), and a shell-style redraw
    // interleaved. Any out-of-bounds or overflow in reflow surfaces here as a panic.
    let mut s = Screen::new(80, 24);
    let mut p = crate::vt::Parser::new();
    p.advance_bytes(
        &mut s,
        "a wide 世 line and more text to wrap around\r\n".as_bytes(),
    );
    p.advance_bytes(
        &mut s,
        b"\x1b]133;A\x07user@host ~/some/long/path  main\r\n> ",
    );

    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed >> 33
    };
    for _ in 0..4000 {
        let cols = 1 + (rng() % 220) as usize;
        let rows = 1 + (rng() % 70) as usize;
        s.resize(cols, rows);
        let (c, r) = s.dimensions();
        assert_eq!((c, r), (cols, rows), "size drifted");
        let (cr, cc) = s.cursor();
        assert!(cr < r && cc < c, "cursor {cr},{cc} escaped {c}x{r}");
        // The wide-glyph invariant must survive every rewrap: a leader always keeps its
        // spacer in the next column, and a spacer always follows its leader. A one-column
        // reflow used to break both by splitting the pair across two rows.
        for row in 0..r {
            for col in 0..c {
                let cell = s.cell(row, col);
                if cell.is_wide_leader() {
                    assert!(
                        col + 1 < c && s.cell(row, col + 1).is_wide_spacer(),
                        "leader at {row},{col} lost its spacer at {c}x{r}"
                    );
                }
                if cell.is_wide_spacer() {
                    assert!(
                        col > 0 && s.cell(row, col - 1).is_wide_leader(),
                        "orphan spacer at {row},{col} at {c}x{r}"
                    );
                }
            }
        }
        // Interleave a shell SIGWINCH redraw and fresh marks, as a live session would.
        match rng() % 4 {
            0 => p.advance_bytes(&mut s, b"\r\x1b[J\x1b]133;A\x07user@host ~/p  main\r\n> "),
            1 => p.advance_bytes(&mut s, "wide 世世世 and text\r\n".as_bytes()),
            _ => {}
        }
    }
}

#[test]
fn sgr_colors_through_the_parser() {
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b[1;31mhi\x1b[0mok");
    assert_eq!(s.cell(0, 0).rune, 'h');
    assert!(s.cell(0, 0).attrs.contains(Attrs::BOLD));
    assert_eq!(s.cell(0, 0).fg, Color::Ansi(1));
    // After the reset, 'o' carries the default rendition again.
    assert_eq!(s.cell(0, 2).rune, 'o');
    assert!(s.cell(0, 2).attrs.is_empty());
    assert_eq!(s.cell(0, 2).fg, Color::Default);
}

#[test]
fn cursor_addressing_and_erase_through_the_parser() {
    let mut s = Screen::new(10, 3);
    feed(&mut s, b"line1\r\nline2");
    feed(&mut s, b"\x1b[H"); // home
    assert_eq!(s.cursor(), (0, 0));
    feed(&mut s, b"\x1b[2;3Hx"); // row 2, col 3 (1-based)
    assert_eq!(s.cell(1, 2).rune, 'x');
    feed(&mut s, b"\x1b[2J"); // erase whole display
    assert_eq!(s.dump().trim(), "");
}

#[test]
fn wrap_via_parser_flags_the_row() {
    let mut s = Screen::new(3, 3);
    feed(&mut s, b"abcde");
    assert_eq!(s.row_string(0).trim_end(), "abc");
    assert_eq!(s.row_string(1).trim_end(), "de");
    assert!(s.row_wraps(0));
}

#[test]
fn box_drawing_charset() {
    let mut s = Screen::new(10, 1);
    // ESC ( 0 designates G0 = DEC Special Graphics; lqk -> ┌─┐; ESC ( B back.
    feed(&mut s, b"\x1b(0lqk\x1b(Bx");
    assert_eq!(s.cell(0, 0).rune, '┌');
    assert_eq!(s.cell(0, 1).rune, '─');
    assert_eq!(s.cell(0, 2).rune, '┐');
    assert_eq!(s.cell(0, 3).rune, 'x');
}

/// 0x5F is "Blank" in DEC's chart, not an underscore. A path or an identifier
/// printed while the line-drawing set is designated is where this shows: every
/// `_` in it is a blank on a real VT.
#[test]
fn the_dec_graphics_underscore_is_a_blank() {
    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b(0a_b\x1b(B_");
    assert_eq!(s.cell(0, 0).rune, '▒');
    assert_eq!(s.cell(0, 1).rune, ' ', "0x5F is Blank in the graphics set");
    assert_eq!(s.cell(0, 2).rune, '␉');
    assert_eq!(
        s.cell(0, 3).rune,
        '_',
        "and an ordinary underscore in ASCII"
    );
}

/// DECSC/DECRC save and restore the character set shift state, which the VT spec
/// lists alongside the cursor and the rendition. A program that saves, designates
/// the line-drawing set to paint a frame, and restores is asking for the ASCII
/// mapping back; without it every letter after the restore is a box glyph.
/// The drawing happens on row 1 so that DECRC's cursor restore (back to row 0) does
/// not simply overwrite the cell under test.
#[test]
fn decsc_restores_the_charset_designation() {
    let mut s = Screen::new(10, 2);
    // Save at (0,0) in ASCII, designate G0 = line drawing, draw on row 1, restore.
    feed(&mut s, b"\x1b7\x1b(0\x1b[2;1Hq\x1b8q");
    assert_eq!(
        s.cell(1, 0).rune,
        '─',
        "drawn while graphics were designated"
    );
    assert_eq!(
        s.cell(0, 0).rune,
        'q',
        "the restore put the ASCII designation back"
    );
}

/// The shift state (SO/SI), not only the designation, is part of what DECSC holds.
#[test]
fn decsc_restores_the_shift_state() {
    let mut s = Screen::new(10, 2);
    // G1 = graphics. Save while shifted in, shift out, draw, restore, print.
    feed(&mut s, b"\x1b)0\x1b7\x0e\x1b[2;1Hq\x1b8q");
    assert_eq!(s.cell(1, 0).rune, '─', "drawn while shifted out to G1");
    assert_eq!(s.cell(0, 0).rune, 'q', "the restore shifted back in to G0");

    // And the other way round: saved while shifted out, the restore shifts out again.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b)0\x0e\x1b7\x0f\x1b[2;1Hq\x1b8q");
    assert_eq!(s.cell(1, 0).rune, 'q', "drawn while shifted in to ASCII");
    assert_eq!(s.cell(0, 0).rune, '─', "the restore shifted back out to G1");
}

#[test]
fn shift_out_in_switches_charset() {
    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b)0"); // designate G1 = special graphics
    feed(&mut s, b"a\x0eq\x0fb"); // 'a', SO, 'q'->─, SI, 'b'
    assert_eq!(s.cell(0, 0).rune, 'a');
    assert_eq!(s.cell(0, 1).rune, '─');
    assert_eq!(s.cell(0, 2).rune, 'b');
}

#[test]
fn alt_screen_via_parser() {
    let mut s = Screen::new(6, 2);
    feed(&mut s, b"main");
    feed(&mut s, b"\x1b[?1049h\x1b[H"); // the cursor carries over, so home it
    assert!(s.is_alt());
    feed(&mut s, b"alt");
    assert_eq!(s.row_string(0).trim_end(), "alt");
    feed(&mut s, b"\x1b[?1049l");
    assert_eq!(s.row_string(0).trim_end(), "main");
}

#[test]
fn osc_sets_title_through_the_parser() {
    let mut s = Screen::new(10, 1);
    feed(&mut s, b"\x1b]0;my title\x07");
    assert_eq!(s.title(), "my title");
    feed(&mut s, b"\x1b]2;other\x1b\\");
    assert_eq!(s.title(), "other");
}

#[test]
fn scroll_region_index_via_parser() {
    let mut s = Screen::new(4, 4);
    feed(&mut s, b"\x1b[2;3r"); // region rows 2..=3 (1-based) -> index 1..=2
    feed(&mut s, b"\x1b[2;1HAA");
    feed(&mut s, b"\x1b[3;1HBB");
    feed(&mut s, b"\x1b[3;1H\n"); // LF at the region bottom scrolls the region
    assert_eq!(s.row_string(1).trim_end(), "BB");
    assert_eq!(s.row_string(2).trim_end(), "");
}

#[test]
fn decaln_fills_with_e() {
    let mut s = Screen::new(3, 2);
    feed(&mut s, b"\x1b#8");
    assert_eq!(s.row_string(0), "EEE");
    assert_eq!(s.row_string(1), "EEE");
    assert_eq!(s.cursor(), (0, 0));
}

/// DECALN "sets the margins to the extremes of the page, and moves the cursor to
/// the home position". A stale scroll region left behind would confine every later
/// scroll to it.
#[test]
fn decaln_resets_the_scroll_margins() {
    let mut s = Screen::new(3, 4);
    s.set_scroll_region(1, 2);
    feed(&mut s, b"\x1b#8");
    assert_eq!(s.cursor(), (0, 0));

    // With the margins back at the extremes, a scroll moves the whole page: fill
    // the rows, then one more line feed from the bottom scrolls row 0 off.
    feed(&mut s, b"\x1b[4;1Ha\n");
    assert_eq!(s.row_string(0), "EEE");
    assert_eq!(
        s.row_string(3),
        "   ",
        "the whole page scrolled, not a region"
    );
}

#[test]
fn wide_and_combining_via_parser() {
    let mut s = Screen::new(10, 1);
    feed(&mut s, "aか\u{3099}b".as_bytes()); // 'a', wide か + combining mark, 'b'
    assert_eq!(s.cell(0, 0).rune, 'a');
    assert_eq!(s.cell(0, 1).rune, 'か');
    assert!(s.cell(0, 1).is_wide_leader());
    assert_eq!(s.marks_at(0, 1).collect::<Vec<_>>(), ['\u{3099}']);
    assert_eq!(s.cell(0, 3).rune, 'b');
}

/// The structural invariants that must hold after any input whatsoever, or `None`.
///
/// Returns the break rather than asserting it, and that is what makes the fuzz
/// find promotable: an assertion cannot be asked "*would* this input have failed?",
/// which is the question a shrinker has to ask a few thousand times.
fn invariant_break(bytes: &[u8], cols: usize, rows: usize) -> Option<String> {
    let mut s = Screen::new(cols, rows);
    let mut p = crate::vt::Parser::new();
    // Fed in chunks, so the incremental path is what gets checked: a break that only
    // appears mid-stream is exactly the kind worth catching.
    for chunk in bytes.chunks(4096) {
        p.advance_bytes(&mut s, chunk);
        if s.dimensions() != (cols, rows) {
            return Some(format!("size drifted to {:?}", s.dimensions()));
        }
        let (cr, cc) = s.cursor();
        if cr >= rows || cc >= cols {
            return Some(format!("cursor {cr},{cc} is outside {cols}x{rows}"));
        }
        if s.primary.lines.len() != rows || s.alt.lines.len() != rows {
            return Some("a buffer stopped holding exactly `rows` lines".to_string());
        }
        if s.primary.scrollback.len() > DEFAULT_SCROLLBACK {
            return Some(format!(
                "scrollback grew to {} past its {DEFAULT_SCROLLBACK} cap",
                s.primary.scrollback.len()
            ));
        }
        // A wide glyph is a leader (the rune) plus a spacer (its second column); an
        // erase or shift that cuts the pair and leaves an orphan is exactly the
        // corruption these ICH/DCH/ECH paths were hardened against, and it is
        // invisible to the size/cursor/count checks above. Every visible row of both
        // buffers, and the scrollback a wide glyph can scroll into, has to stay paired.
        for (name, buf) in [("primary", &s.primary), ("alt", &s.alt)] {
            for row in buf.lines.iter().chain(buf.scrollback.iter()) {
                if let Some(why) = wide_pair_break(&row.cells) {
                    return Some(format!("{name}: {why}"));
                }
            }
        }
    }
    None
}

/// Where a row's wide-glyph pairing is broken, or `None`. A leader must be followed by
/// a spacer, and a spacer must be preceded by a leader; either half standing alone is
/// an orphan that draws as garbage.
fn wide_pair_break(row: &[Cell]) -> Option<String> {
    for (i, cell) in row.iter().enumerate() {
        if cell.is_wide_leader() && !row.get(i + 1).is_some_and(|c| c.is_wide_spacer()) {
            return Some(format!("wide leader at col {i} has no spacer to its right"));
        }
        if cell.is_wide_spacer()
            && !i
                .checked_sub(1)
                .and_then(|p| row.get(p))
                .is_some_and(|c| c.is_wide_leader())
        {
            return Some(format!("wide spacer at col {i} has no leader to its left"));
        }
    }
    None
}

#[test]
fn random_bytes_into_screen_keep_invariants() {
    // The whole pipeline under fuzz: ~1.2 MB of arbitrary bytes through the parser
    // into a real Screen must never panic and must keep the grid's structural
    // invariants.
    //
    // On a break the input is shrunk before it is reported, which is the difference
    // between a find and a fixed bug: "an invariant broke somewhere in 1.2 MB, 200
    // chunks deep" is a sentence, and the six bytes this prints instead are a
    // scenario you can paste into `tests/scenarios.rs` and keep forever.
    // A small grid on purpose: 24x8 puts every wrap, scroll and margin edge within
    // a few characters of wherever the cursor is, so the interesting collisions
    // happen constantly instead of once a megabyte.
    let (cols, rows) = (24, 8);
    let bytes = crate::fuzz::Stream::new(0xDEAD_BEEF_CAFE_1234).bytes(300 * 4096);

    let Some(why) = invariant_break(&bytes, cols, rows) else {
        return;
    };
    let minimal = crate::fuzz::shrink(&bytes, |b| invariant_break(b, cols, rows).is_some());
    let because = invariant_break(&minimal, cols, rows).unwrap_or_else(|| why.clone());
    panic!(
        "the grid's structural invariants broke: {because}\n\n\
         Shrunk from {} bytes to {}:\n\n    \
         let mut s = Screen::new({cols}, {rows});\n    \
         feed(&mut s, {});\n\n\
         Fix the bug, then keep this: it belongs in tests/scenarios.rs as a scenario \
         so the seed that found it never has to find it twice.",
        bytes.len(),
        minimal.len(),
        crate::fuzz::as_byte_literal(&minimal),
    );
}

/// Feed `bytes` two ways — `advance_bytes` (which bulk-writes printable-ASCII
/// runs) and one byte at a time through `advance` (the per-char path) — and
/// assert the resulting screens are identical. Any divergence is a bug in the
/// bulk `print_ascii` path.
fn assert_bulk_equiv(cols: usize, rows: usize, bytes: &[u8]) {
    let mut bulk = Screen::new(cols, rows);
    let mut per = Screen::new(cols, rows);
    let mut pb = crate::vt::Parser::new();
    let mut pc = crate::vt::Parser::new();
    pb.advance_bytes(&mut bulk, bytes);
    for &b in bytes {
        pc.advance(&mut per, b);
    }
    assert_eq!(bulk.snapshot(), per.snapshot(), "snapshot mismatch");
    assert_eq!(bulk.cursor(), per.cursor(), "cursor mismatch");
}

#[test]
fn bulk_ascii_equivalence_targeted() {
    // A plain run that soft-wraps across several short rows, then a hard newline.
    assert_bulk_equiv(10, 3, b"hello world this wraps over rows\r\nnext line");
    // SGR pen changes split the run into differently-styled cells mid-stream.
    assert_bulk_equiv(20, 3, b"norm\x1b[31mred\x1b[1;32mboldgreen\x1b[0mback");
    // DEC Special Graphics via ESC ( 0 remaps ASCII to line-drawing: the bulk
    // path must fall back and map each glyph. ESC ( B restores identity ASCII.
    assert_bulk_equiv(20, 3, b"\x1b(0lqqqk abc\x1b(Bplain");
    // SO/SI shift GL to G1 (designated special-graphics) and back.
    assert_bulk_equiv(20, 3, b"\x1b)0ab\x0eqqwwee\x0fnormal");
    // Insert mode (CSI 4h) shifts existing cells right as each char lands.
    assert_bulk_equiv(20, 3, b"12345\x1b[H\x1b[4hABC");
    // No autowrap (CSI ?7l): the cursor sticks at the last column and overwrites.
    assert_bulk_equiv(8, 3, b"\x1b[?7labcdefghijk");
    // Tab, carriage return, and backspace interleaved with runs.
    assert_bulk_equiv(20, 3, b"ab\tcd\re\x08fgh");
    // Cursor addressing lands the cursor mid-row before a run.
    assert_bulk_equiv(20, 4, b"\x1b[2;5Hplaced here and wrapping onward");
}

#[test]
fn decoded_text_run_equivalence_targeted() {
    assert_bulk_equiv(
        10,
        4,
        "é and 日本語 mixed with ASCII across rows".as_bytes(),
    );
    assert_bulk_equiv(8, 4, "a\u{301}b界\u{302}c café".as_bytes());
    // Wide glyphs immediately before, on, and after the right edge.
    assert_bulk_equiv(5, 4, "abc界x\r\nabcd界y\r\nabcde界z".as_bytes());
    // The no-autowrap edge backs a wide glyph up over the final two cells.
    assert_bulk_equiv(5, 3, "\x1b[?7labcd界日本".as_bytes());
    // Modes whose semantics are intentionally scalar still pass through the
    // same run callback and must map/shift/cluster exactly as before.
    assert_bulk_equiv(12, 3, "\x1b[4hAé界".as_bytes());
    assert_bulk_equiv(12, 3, "\x1b(0qéx\x1b(B日本".as_bytes());
    assert_bulk_equiv(12, 3, "\x1b[?2027h#\u{fe0f}\u{20e3} 日本".as_bytes());
}

#[test]
fn bulk_ascii_matches_per_char_under_fuzz() {
    // A small grid so runs cross rows often, driven by a deterministic stream
    // dense with ASCII runs but salted with the full byte range (ESC sequences,
    // controls, high/UTF-8 bytes). Every 4 KiB block must leave both screens
    // identical — the strongest guard that the bulk path changed nothing.
    let mut bulk = Screen::new(8, 4);
    let mut per = Screen::new(8, 4);
    let mut pb = crate::vt::Parser::new();
    let mut pc = crate::vt::Parser::new();
    let mut seed: u64 = 0x0BAD_C0DE_1234_5678;
    let mut buf = [0u8; 4096];
    for block in 0..300 {
        for b in buf.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (seed >> 33) as u32;
            // ~1 in 6 bytes is full-range (keeps escapes/controls/UTF-8 in the
            // mix); the rest are printable ASCII, so runs are long enough to wrap.
            *b = if r.is_multiple_of(6) {
                (r >> 8) as u8
            } else {
                0x20 + ((r >> 8) % 0x5f) as u8
            };
        }
        pb.advance_bytes(&mut bulk, &buf);
        for &byte in buf.iter() {
            pc.advance(&mut per, byte);
        }
        assert_eq!(bulk.snapshot(), per.snapshot(), "block {block} snapshot");
        assert_eq!(bulk.cursor(), per.cursor(), "block {block} cursor");
    }
}

#[test]
fn resize_keeps_the_line_count_and_content() {
    let mut s = Screen::new(10, 3);
    feed(&mut s, b"abc\r\ndef\r\nghi");
    s.resize(6, 5);
    assert_eq!(s.dimensions(), (6, 5));
    // Existing rows keep their text; growth adds blank rows at the bottom.
    assert_eq!(s.row_string(0).trim_end(), "abc");
    assert_eq!(s.row_string(2).trim_end(), "ghi");
    assert_eq!(s.row_string(4).trim_end(), "");
    // Both screens track the new size.
    assert_eq!(s.primary.lines.len(), 5);
    assert_eq!(s.alt.lines.len(), 5);
}

#[test]
fn narrowing_wraps_and_widening_rejoins() {
    // The reflow contract, replacing the old truncate-and-pad behaviour: a line that
    // no longer fits is wrapped into scrollback, not cut off, and widening past its
    // length pulls it back into one row rather than padding a lost tail with blanks.
    let mut s = Screen::new(8, 1);
    feed(&mut s, b"abcdefgh");
    s.resize(4, 1);
    // The tail is the visible row; the head wrapped up into scrollback, nothing lost.
    assert_eq!(s.row_string(0).trim_end(), "efgh");
    assert_eq!(s.scrollback_len(), 1);
    s.scroll_view_up(1);
    assert_eq!(
        s.view_cell(0, 0).rune,
        'a',
        "the head is in history, not truncated away"
    );
    s.scroll_view_to_bottom();
    // Widening back rejoins the two segments into one row.
    s.resize(8, 1);
    assert_eq!(s.row_string(0).trim_end(), "abcdefgh");
    assert_eq!(s.dimensions(), (8, 1));
}

#[test]
fn shrinking_rows_scrolls_the_cursor_line_into_view() {
    // Cursor on the last row: shrinking must scroll the top away (into
    // scrollback), not drop the cursor's line off the bottom.
    let mut s = Screen::new(6, 4);
    feed(&mut s, b"r0\r\nr1\r\nr2\r\nr3");
    let (cr, _) = s.cursor();
    assert_eq!(cr, 3);
    s.resize(6, 2);
    assert_eq!(s.dimensions(), (6, 2));
    let (cr, _) = s.cursor();
    assert!(cr < 2, "cursor stayed on screen");
    // The last-written line is still visible; the top scrolled into history.
    assert_eq!(s.row_string(1).trim_end(), "r3");
    assert!(!s.primary.scrollback.is_empty());
}

#[test]
fn resize_clamps_the_cursor_and_resets_the_scroll_region() {
    let mut s = Screen::new(10, 10);
    feed(&mut s, b"\x1b[3;8r"); // DECSTBM: region rows 3..8
    feed(&mut s, b"\x1b[9;9H"); // cursor near the old bottom-right
    s.resize(5, 5);
    let (cr, cc) = s.cursor();
    assert!(
        cr < 5 && cc < 5,
        "cursor {cr},{cc} clamped into the new grid"
    );
    // A resize resets DECSTBM to the full screen (xterm behavior): a full-
    // screen scroll now moves all five rows.
    assert_eq!((s.primary.scroll_top, s.primary.scroll_bottom), (0, 4));
}

#[test]
fn scrolling_up_reveals_scrollback_rows() {
    // Four lines on a two-row screen: "a","b" scroll into history, "c","d"
    // stay live. The view walks up through the combined history-plus-screen.
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd");
    assert_eq!(s.scrollback_len(), 2);
    // Pinned to the bottom: the live rows.
    assert_eq!(s.view_cell(0, 0).rune, 'c');
    assert_eq!(s.view_cell(1, 0).rune, 'd');
    // One line up: "b" over "c".
    s.scroll_view_up(1);
    assert!(s.is_scrolled());
    assert_eq!(s.view_cell(0, 0).rune, 'b');
    assert_eq!(s.view_cell(1, 0).rune, 'c');
    // Two lines up: the very top, "a" over "b".
    s.scroll_view_up(1);
    assert_eq!(s.view_cell(0, 0).rune, 'a');
    assert_eq!(s.view_cell(1, 0).rune, 'b');
    // Back to the bottom.
    s.scroll_view_to_bottom();
    assert!(!s.is_scrolled());
    assert_eq!(s.view_cell(0, 0).rune, 'c');
}

#[test]
fn scroll_is_clamped_to_the_history() {
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd"); // two lines of scrollback
    s.scroll_view_up(999);
    assert_eq!(s.view_offset(), 2, "cannot scroll above the oldest line");
    s.scroll_view_down(999);
    assert_eq!(s.view_offset(), 0, "cannot scroll below the live bottom");
    s.scroll_view_to_top();
    assert_eq!(s.view_offset(), 2);
}

#[test]
fn alt_screen_has_no_scrollback_view() {
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd");
    s.scroll_view_up(2);
    feed(&mut s, b"\x1b[?1049h"); // enter the alt screen
    assert_eq!(s.view_offset(), 0, "the alt screen pins to the bottom");
    s.scroll_view_up(5);
    assert!(!s.is_scrolled(), "scrolling is inert on the alt screen");
}

#[test]
fn leaving_the_alt_screen_resets_the_view() {
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd");
    s.scroll_view_up(2);
    feed(&mut s, b"\x1b[?1049h\x1b[?1049l"); // enter then leave the alt screen
    assert_eq!(s.view_offset(), 0, "back to a live view, not stale history");
}

#[test]
fn resize_preserves_the_scrolled_view() {
    // A width change reflows, and the view a user has scrolled up to follows the
    // content across the rewrap instead of snapping to the bottom: the same line stays
    // under their eye.
    let mut s = Screen::new(5, 3);
    feed(&mut s, b"a\r\nb\r\nc\r\nd\r\ne");
    s.scroll_view_up(2);
    assert_eq!(s.view_cell(0, 0).rune, 'a');
    s.resize(8, 4);
    assert!(s.is_scrolled(), "the reflow kept the view scrolled");
    assert_eq!(
        s.view_cell(0, 0).rune,
        'a',
        "the same line is still at the top after the reflow"
    );
}

#[test]
fn a_resize_to_the_same_size_changes_nothing_at_all() {
    // Reachable from a sub-cell drag of the window edge, or a scale change: the window's
    // guard lets anything through whose *pixel* size moved, so the grid is asked to
    // resize to the dimensions it already has. Every path below this used to run anyway,
    // resetting state the child owns — and the child is never told, because the pixel
    // fields of `winsize` are always sent as zero, so the kernel sees a byte-identical
    // struct and raises no SIGWINCH. The program cannot know to repaint what we broke.
    let mut s = Screen::new(20, 8);

    // A program reserving a status line, exactly as tmux or vim-with-a-split does.
    feed(&mut s, b"\x1b[1;5r"); // DECSTBM: rows 1..5, leaving row 6 as a status line
    feed(&mut s, b"\x1b[6;1HSTATUS");
    assert_eq!(s.row_string(5).trim_end(), "STATUS");

    // A user scrolled back into history, and a pending wrap parked at the right edge.
    feed(&mut s, b"\x1b[1;1H");
    for i in 0..12 {
        feed(&mut s, format!("line{i}\r\n").as_bytes());
    }
    s.scroll_view_up(3);
    let scrolled_to = s.view_offset();
    assert!(scrolled_to > 0, "the view is up in history");
    feed(&mut s, b"\x1b[3;20Hx"); // print in the last column: wrap is now pending

    assert!(s.active().cursor.pending_wrap, "and a wrap is pending");

    let effect = s.resize(20, 8);
    assert!(
        matches!(effect, ResizeEffect::Stable),
        "nothing moved, so the selection stands"
    );
    assert_eq!(
        (s.active().scroll_top, s.active().scroll_bottom),
        (0, 4),
        "the scroll region is the child's, and the child was not consulted"
    );
    assert!(
        s.active().cursor.pending_wrap,
        "the deferred wrap still belongs to the glyph that parked it"
    );
    assert_eq!(
        s.view_offset(),
        scrolled_to,
        "the history view is where the user left it"
    );

    // The behavioural half of the margins, and the failure the user actually sees: a
    // line feed at the region's bottom margin must scroll *inside* the region and leave
    // the cursor there. With the margins reset it walks onto row 6 instead, and the next
    // thing the program prints lands on top of its own status line.
    feed(&mut s, b"\x1b[5;1H\nOUT");
    assert_eq!(
        s.row_string(5).trim_end(),
        "STATUS",
        "the status line is intact"
    );
    assert_eq!(
        s.row_string(4).trim_end(),
        "OUT",
        "and the output stayed inside the scroll region"
    );
}

#[test]
fn reflow_widen_rejoins_a_wrapped_line() {
    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcdef");
    // On four columns it soft-wrapped across two rows.
    assert_eq!(s.row_string(0).trim_end(), "abcd");
    assert_eq!(s.row_string(1).trim_end(), "ef");
    // Widening past the line's length pulls the two rows back into one.
    s.resize(8, 3);
    assert_eq!(s.row_string(0).trim_end(), "abcdef");
    assert_eq!(s.row_string(1).trim_end(), "");
}

#[test]
fn reflow_narrow_splits_a_long_line() {
    let mut s = Screen::new(6, 4);
    feed(&mut s, b"hello world");
    s.resize(4, 4);
    // The one logical line breaks cleanly across rows, content preserved in order.
    assert_eq!(s.row_string(0).trim_end(), "hell");
    assert_eq!(s.row_string(1).trim_end(), "o wo");
    assert_eq!(s.row_string(2).trim_end(), "rld");
}

#[test]
fn reflow_round_trips_through_a_narrow_width() {
    let mut s = Screen::new(10, 4);
    feed(&mut s, b"the quick brown fox");
    s.resize(5, 4);
    // Widening past the logical length recovers the original single line.
    s.resize(20, 4);
    assert_eq!(s.row_string(0).trim_end(), "the quick brown fox");
}

#[test]
fn reflow_keeps_the_cursor_on_its_character() {
    let mut s = Screen::new(4, 3);
    // "abcdef" wraps to "abcd"/"ef"; park the cursor on 'b' at row 0, col 1.
    feed(&mut s, b"abcdef\x1b[1;2H");
    let (r, c) = s.cursor();
    assert_eq!(s.cell(r, c).rune, 'b');
    s.resize(8, 3);
    let (r, c) = s.cursor();
    assert_eq!(
        s.cell(r, c).rune,
        'b',
        "the cursor rode the reflow to its char"
    );
}

#[test]
fn reflow_never_splits_a_wide_glyph() {
    let mut s = Screen::new(10, 4);
    feed(&mut s, "aかb".as_bytes()); // 'a', wide か (two columns), 'b'
    s.resize(2, 4);
    // The pair could not fit in the last column of row 0, so it was pushed whole onto
    // its own row rather than split — no half glyph is ever left behind.
    assert_eq!(s.cell(1, 0).rune, 'か');
    assert!(s.cell(1, 0).is_wide_leader());
    assert!(s.cell(1, 1).is_wide_spacer());
    assert_eq!(s.cell(0, 0).rune, 'a');
    assert!(!s.cell(0, 1).is_wide_leader() && !s.cell(0, 1).is_wide_spacer());
}

#[test]
fn reflow_clips_a_wide_glyph_at_a_single_column() {
    // At one column a wide glyph has nowhere to put its spacer, so it is stored clipped —
    // an ordinary single cell — exactly as the printer does at width one. It must never
    // become a leader stranded without its spacer, nor an orphan spacer on its own row.
    let mut s = Screen::new(10, 4);
    feed(&mut s, "aか".as_bytes()); // 'a', wide か
    s.resize(1, 4);
    assert_eq!(s.row_string(0).trim_end(), "a");
    assert_eq!(s.cell(1, 0).rune, 'か');
    assert!(!s.cell(1, 0).is_wide_leader(), "clipped, not a leader");
    assert!(!s.cell(1, 0).is_wide_spacer());
    // No third row carrying an orphaned spacer.
    assert_eq!(s.row_string(2).trim_end(), "");
}

#[test]
fn reflow_widen_drops_wide_glyph_wrap_padding() {
    // "aかb" at two columns leaves a blank after 'a' because か could not fit beside it and
    // wrapped down. That blank is wrap padding, not text: widening must rejoin the line with
    // か flush against 'a', not preserve a phantom gap between them.
    let mut s = Screen::new(2, 4);
    feed(&mut s, "aかb".as_bytes());
    s.resize(10, 4);
    assert_eq!(s.cell(0, 0).rune, 'a');
    assert_eq!(s.cell(0, 1).rune, 'か');
    assert!(s.cell(0, 1).is_wide_leader());
    assert!(s.cell(0, 2).is_wide_spacer());
    assert_eq!(s.cell(0, 3).rune, 'b');
}

#[test]
fn reflow_keeps_the_cursor_live_when_rows_below_it_do_not_fit() {
    // A narrow multiplies the row count. If the content below the cursor then needs more
    // rows than the screen has, the live window must still contain the cursor (output is
    // written there); the rows below it that no longer fit are dropped. Repro: the cursor
    // parked high on 'b' with a full line beneath it, narrowed so they cannot all fit — the
    // cursor used to detach onto unrelated content two rows down.
    let mut s = Screen::new(4, 4);
    feed(&mut s, b"abcdef\r\nWXYZ"); // "abcd"/"ef", then "WXYZ"
    feed(&mut s, b"\x1b[1;2H"); // cursor onto 'b' at row 0, col 1
    let (r, c) = s.cursor();
    assert_eq!(s.cell(r, c).rune, 'b');
    s.resize(2, 4);
    let (r, c) = s.cursor();
    assert_eq!(
        s.cell(r, c).rune,
        'b',
        "the cursor stayed on its character instead of detaching to unrelated content"
    );
}

#[test]
fn reflow_carries_combining_marks() {
    let mut s = Screen::new(6, 3);
    // 'e' at col 4 carries a combining mark; the line is "abcdef".
    feed(&mut s, "abcde\u{0301}f".as_bytes());
    assert_eq!(s.marks_at(0, 4).collect::<Vec<_>>(), ['\u{0301}']);
    s.resize(3, 3);
    // "abc"/"def": 'e' moved to row 1, col 1, and its mark rode along.
    assert_eq!(s.cell(1, 1).rune, 'e');
    assert_eq!(s.marks_at(1, 1).collect::<Vec<_>>(), ['\u{0301}']);
}

#[test]
fn reflow_keeps_hyperlinks() {
    let mut s = Screen::new(10, 3);
    feed(
        &mut s,
        b"\x1b]8;;https://example.com\x1b\\linktext\x1b]8;;\x1b\\",
    );
    let link = s.cell(0, 0).link;
    assert!(link.is_set(), "the text is inside an OSC 8 anchor");
    s.resize(4, 3);
    // The link rides on the cell, so a rewrap keeps it without any span fixups.
    assert_eq!(s.cell(0, 0).link, link);
}

#[test]
fn reflow_trims_plain_trailing_blanks() {
    let mut s = Screen::new(10, 3);
    feed(&mut s, b"hi");
    s.resize(4, 3);
    // A short line stays one row: the trailing blanks are padding, not wrapped content.
    assert_eq!(s.scrollback_len(), 0);
    assert_eq!(s.row_string(0).trim_end(), "hi");
    assert_eq!(s.row_string(1).trim_end(), "");
}

#[test]
fn reflow_keeps_coloured_trailing_cells() {
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b[41m    \x1b[0m"); // four red-background spaces
    s.resize(6, 2);
    // A coloured space is content, not padding: the trim leaves it, so the bar survives.
    assert_eq!(
        s.cell(0, 3).bg,
        Color::Ansi(1),
        "the coloured cell rode the reflow"
    );
}

#[test]
fn reflow_caps_a_giant_logical_line() {
    // A line longer than MAX_LOGICAL_COLS is broken at the cap and never re-joins across
    // the seam, so a very wide window cannot force an unbounded single logical line.
    let big = vec![b'x'; 1100];
    let mut s = Screen::new(80, 5);
    feed(&mut s, &big);
    s.resize(1200, 5);
    let first = s.row_string(0).trim_end().len();
    assert!(
        (MAX_LOGICAL_COLS..1100).contains(&first),
        "the first row is capped near {MAX_LOGICAL_COLS}, got {first}"
    );
    assert!(
        !s.row_string(1).trim_end().is_empty(),
        "the remainder stayed on a second row instead of collapsing to one"
    );
}

#[test]
fn reflow_leaves_the_alt_screen_clamped() {
    let mut s = Screen::new(8, 3);
    feed(&mut s, b"\x1b[?1049h"); // enter the alt screen
    feed(&mut s, b"abcdefgh");
    s.resize(4, 3);
    // The alt screen clamps like an ordinary resize; a full-screen program owns it and
    // repaints on SIGWINCH, so its content must never be re-wrapped.
    assert_eq!(s.row_string(0).trim_end(), "abcd");
    assert_eq!(s.row_string(1).trim_end(), "");
}

#[test]
fn height_only_resize_does_not_reflow() {
    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcdef"); // "abcd"/"ef"
    s.resize(4, 6); // width unchanged: the cheap path, no rewrap
    assert_eq!(s.row_string(0).trim_end(), "abcd");
    assert_eq!(
        s.row_string(1).trim_end(),
        "ef",
        "still split, not rejoined"
    );
}

#[test]
fn growing_height_pulls_scrollback_down() {
    // Growing the screen keeps the bottom line pinned to the window's bottom edge and
    // reveals history above, instead of padding blank rows below the content.
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd"); // "a","b" in history; "c","d" live
    assert_eq!(s.scrollback_len(), 2);
    s.resize(5, 4); // height only, grow by two
    assert_eq!(s.row_string(0).trim_end(), "a");
    assert_eq!(s.row_string(1).trim_end(), "b");
    assert_eq!(s.row_string(2).trim_end(), "c");
    assert_eq!(s.row_string(3).trim_end(), "d");
    assert_eq!(s.scrollback_len(), 0, "history was pulled onto the screen");
    // The cursor rode down with its line, still on the last row.
    assert_eq!(s.cursor().0, 3);
}

#[test]
fn growing_height_past_history_pads_below() {
    // Only one line of history but two new rows: the one is pulled down, the last row
    // is a blank pad below the content.
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc"); // "a" in history; "b","c" live
    assert_eq!(s.scrollback_len(), 1);
    s.resize(5, 4);
    assert_eq!(s.row_string(0).trim_end(), "a");
    assert_eq!(s.row_string(1).trim_end(), "b");
    assert_eq!(s.row_string(2).trim_end(), "c");
    assert_eq!(s.row_string(3).trim_end(), "");
    assert_eq!(s.scrollback_len(), 0);
}

#[test]
fn reflow_returns_a_remap_that_tracks_a_cell() {
    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcdef"); // "abcd" | "ef"; 'f' is at (abs_row(1), 1)
    let f = s.abs_row(1);
    let ResizeEffect::Reflowed(remap) = s.resize(8, 3) else {
        panic!("a width change reflows");
    };
    // After widening, "abcdef" is one row and 'f' has moved to column 5.
    let (row, col) = remap.point(f, 1).expect("the cell survived the reflow");
    assert_eq!(col, 5);
    let dr = s.display_row(row).expect("its row is on screen");
    assert_eq!(s.row_string(dr).chars().nth(col), Some('f'));
}

#[test]
fn a_height_only_resize_reports_stable_ids() {
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd");
    assert!(matches!(s.resize(5, 4), ResizeEffect::Stable));
}

#[test]
fn reflow_carries_prompt_marks() {
    let mut s = Screen::new(4, 4);
    feed(&mut s, b"\x1b]133;A\x07ab\r\n"); // a prompt "ab" on row 0
    feed(&mut s, b"\x1b]133;C\x07cdefgh"); // output "cdefgh" wraps to "cdef" | "gh"
    assert_eq!(s.prompts()[0].row, s.abs_row(0));
    s.resize(8, 4); // widen: "ab" stays row 0, "cdefgh" rejoins below it
                    // The prompt still names the row holding "ab", and its output the rejoined line.
    let p = s.prompts()[0];
    let prompt_row = s.display_row(p.row).expect("prompt row on screen");
    assert_eq!(s.row_string(prompt_row).trim_end(), "ab");
    let output_row = s
        .display_row(p.output.expect("output marked"))
        .expect("output row on screen");
    assert_eq!(s.row_string(output_row).trim_end(), "cdefgh");
}

#[test]
fn reflow_leaves_the_live_prompt_for_the_shell_to_redraw() {
    // Repro of the resize-mangles-the-zsh-prompt bug. A full-width prompt bar reflows to
    // more rows than the shell regenerates it to on SIGWINCH. The shell redraws by moving
    // up its OWN row count and clearing to the end of the screen; if the terminal reflowed
    // the live prompt to more rows, the shell's clear misses the extra row and it is left
    // as residue on the static line above (exactly the mangled `ps` line in the report).
    // So the terminal must not reflow the current prompt region (from the last OSC 133
    // mark to the cursor) — the shell owns it and repaints it.
    let mut s = Screen::new(12, 4);
    feed(&mut s, b"top\r\n"); // static output, above the prompt
    feed(&mut s, b"second\r\n");
    feed(&mut s, b"\x1b]133;A\x07"); // the current prompt begins here
    feed(&mut s, b"BAR@FULLWID:\r\n"); // a bar that fills all 12 columns
    feed(&mut s, b"> "); // the input line; the cursor rests at column 2

    s.resize(8, 4); // narrow: the bar's 12 cells would reflow onto two rows

    // The shell's SIGWINCH redraw: it regenerates a one-row bar for 8 columns and,
    // believing its prompt is two rows (bar + input), moves up one from the input line,
    // clears to the end of the screen, and repaints.
    feed(&mut s, b"\x1b[1A\r\x1b[J"); // up one, column zero, clear to end of screen
    feed(&mut s, b"BARSHORT\r\n> "); // the regenerated one-row bar, then the input line

    // The row above the repainted bar must be the static "second", not a leftover half of
    // the old bar that the shell's clear could not reach.
    assert_eq!(
        s.row_string(1).trim_end(),
        "second",
        "a reflowed prompt row leaked as residue above the shell's redraw"
    );
}

#[test]
fn reflow_survives_narrow_until_wrap_then_widen() {
    // The user's precise repro: a full screen, narrow until the prompt label wraps to a
    // second row, then widen again. Each resize is followed by the shell's SIGWINCH
    // redraw (move up its previous prompt height, clear to end of screen, repaint).
    let mut s = Screen::new(8, 4);
    feed(&mut s, b"o0\r\no1\r\n"); // output filling toward the bottom
    feed(&mut s, b"\x1b]133;A\x07"); // the idle prompt begins
    feed(&mut s, b"LABEL_XY"); // an 8-column-wide label (fills the row)
    feed(&mut s, b"\r\n> "); // the input line; cursor at column 2

    // Narrow to 5: the 8-cell label no longer fits on one row.
    s.resize(5, 4);
    // Shell redraw, previous prompt height 2 (label + input): up 1, clear, repaint.
    feed(&mut s, b"\x1b[1A\r\x1b[J\x1b]133;A\x07LABEL_XY\r\n> ");

    // Widen to 10: the label fits on one row again.
    s.resize(10, 4);
    // Shell redraw, previous prompt height 3 (wrapped label + input): up 2, clear, repaint.
    feed(&mut s, b"\x1b[2A\r\x1b[J\x1b]133;A\x07LABEL_XY\r\n> ");

    // The label must appear on exactly one row — no wrapped-half residue left behind.
    let rows: Vec<String> = (0..4)
        .map(|r| s.row_string(r).trim_end().to_string())
        .collect();
    let with_label = rows.iter().filter(|r| r.contains("LABEL")).count();
    assert_eq!(
        with_label, 1,
        "label residue after narrow-then-widen: {rows:?}"
    );
}

#[test]
fn reflow_survives_narrow_then_widen_without_remarking_the_prompt() {
    // As above, but the shell does NOT re-emit OSC 133;A on its SIGWINCH redraw (it only
    // marks a fresh prompt, not a repaint). The mark must still track the label across the
    // reflow, or the freeze protects the wrong rows.
    let mut s = Screen::new(8, 4);
    feed(&mut s, b"o0\r\no1\r\n");
    feed(&mut s, b"\x1b]133;A\x07");
    feed(&mut s, b"LABEL_XY");
    feed(&mut s, b"\r\n> ");

    s.resize(5, 4);
    feed(&mut s, b"\x1b[1A\r\x1b[JLABEL_XY\r\n> "); // redraw, no OSC 133;A

    s.resize(10, 4);
    feed(&mut s, b"\x1b[2A\r\x1b[JLABEL_XY\r\n> "); // redraw, no OSC 133;A

    let rows: Vec<String> = (0..4)
        .map(|r| s.row_string(r).trim_end().to_string())
        .collect();
    let with_label = rows.iter().filter(|r| r.contains("LABEL")).count();
    assert_eq!(with_label, 1, "label residue without re-marking: {rows:?}");
}

#[test]
#[ignore = "known reflow-vs-shell limit without shell integration; alacritty has it too. \
            The fix is OSC 133 (POWERLEVEL9K_TERM_SHELL_INTEGRATION=true), covered by \
            reflow_leaves_the_live_prompt_for_the_shell_to_redraw."]
fn reflow_does_not_mangle_a_prompt_without_osc133() {
    // The real-world repro, captured from zsh 5.9 under a PTY: a full-width prompt bar,
    // NO OSC 133 marks (most prompts do not emit them), then a resize. zsh's SIGWINCH
    // redraw moves up its own tracked prompt height, clears to end of screen, and
    // repaints. If the terminal reflowed the full-width bar to more rows than zsh tracked,
    // zsh's clear misses the extra row and it survives as residue — the mangled line.
    let mut s = Screen::new(12, 4);
    feed(&mut s, b"o0\r\no1\r\n"); // static output above
    feed(&mut s, b"\x1b[44mBARBARBARBAR\x1b[49m\r\n> "); // a blue full-width bar, then input

    s.resize(6, 4); // narrow: the 12-cell bar no longer fits on one row

    // zsh's actual redraw bytes (up one, clear to end of screen, repaint a 6-wide bar).
    feed(&mut s, b"\r\r\x1b[A\x1b[J\x1b[44mBARBAR\x1b[49m\r\n> ");

    // The bar must occupy exactly one row: no leftover half above the repaint.
    let rows: Vec<String> = (0..4)
        .map(|r| s.row_string(r).trim_end().to_string())
        .collect();
    let bars = rows.iter().filter(|r| r.contains("BAR")).count();
    assert_eq!(
        bars, 1,
        "prompt-bar residue after a no-OSC133 resize: {rows:?}"
    );
}

#[test]
fn shell_integration_fixes_the_full_width_bar_resize() {
    // The same full-width bar and the same real zsh redraw bytes as the ignored repro
    // above, but WITH the OSC 133 prompt mark shell integration emits (the p10k
    // `POWERLEVEL9K_TERM_SHELL_INTEGRATION=true` path). The bar is now frozen — clamped to
    // one row rather than rewrapped — so it matches zsh's one-row repaint and no residue is
    // left. This is the fix kitty ships and alacritty lacks.
    let mut s = Screen::new(12, 4);
    feed(&mut s, b"o0\r\no1\r\n");
    feed(&mut s, b"\x1b]133;A\x07\x1b[44mBARBARBARBAR\x1b[49m\r\n> "); // marked prompt

    s.resize(6, 4);
    feed(&mut s, b"\r\r\x1b[A\x1b[J\x1b[44mBARBAR\x1b[49m\r\n> "); // real zsh redraw

    let rows: Vec<String> = (0..4)
        .map(|r| s.row_string(r).trim_end().to_string())
        .collect();
    let bars = rows.iter().filter(|r| r.contains("BAR")).count();
    assert_eq!(
        bars, 1,
        "shell integration should leave no bar residue: {rows:?}"
    );
}

#[test]
fn reflow_still_reflows_output_above_the_idle_prompt() {
    // Freezing the live prompt must not disable the feature: a finished command's output
    // above the idle prompt still reflows (the whole point). "docker ps"-style — wrapped
    // output, then a fresh idle prompt below it.
    let mut s = Screen::new(4, 4);
    feed(&mut s, b"\x1b]133;A\x07p\r\n"); // an earlier prompt
    feed(&mut s, b"\x1b]133;C\x07abcdef\r\n"); // its output wraps: "abcd" | "ef"
    feed(&mut s, b"\x1b]133;D;0\x07"); // the command finished
    feed(&mut s, b"\x1b]133;A\x07> "); // a fresh idle prompt (no output yet)

    s.resize(8, 4); // widen

    // The finished output rejoined onto one row above the frozen prompt.
    assert_eq!(s.row_string(1).trim_end(), "abcdef");
    assert_eq!(s.row_string(2).trim_end(), ">"); // the idle prompt, clamped, left for zsh
    let (cr, cc) = s.cursor();
    assert_eq!((cr, cc), (2, 2), "cursor still sits in the input");
}

#[test]
fn output_holds_a_scrolled_view_in_place() {
    // The regression test for "you cannot scroll back while the child is printing":
    // a row entering history under a scrolled view must move the offset with it, or
    // the text under the user's eye slides away.
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd"); // "a","b" in history; "c","d" live
    s.scroll_view_up(2); // the top of history: "a" over "b"
    assert_eq!(s.view_cell(0, 0).rune, 'a');

    feed(&mut s, b"\r\ne"); // "c" enters history while the view is scrolled

    assert_eq!(
        s.view_offset(),
        3,
        "the offset followed the content into history"
    );
    assert_eq!(s.view_cell(0, 0).rune, 'a', "the visible text did not move");
    assert_eq!(s.view_cell(1, 0).rune, 'b');
}

#[test]
fn a_pinned_view_still_follows_the_tail() {
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd");
    assert_eq!(s.view_offset(), 0);
    feed(&mut s, b"\r\ne");
    assert_eq!(
        s.view_offset(),
        0,
        "a view at the bottom stays at the bottom"
    );
    assert_eq!(
        s.view_cell(0, 0).rune,
        'd',
        "and keeps showing the newest rows"
    );
    assert_eq!(s.view_cell(1, 0).rune, 'e');
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

#[test]
fn a_scrolled_view_holds_at_the_top_as_history_evicts() {
    // Once the ring is full every push also evicts, so the offset saturates and the
    // oldest line the user can see slides out from under them (as xterm does). The
    // view must hold at the top, not jump.
    let mut s = Screen::with_scrollback(5, 2, 3);
    feed(&mut s, b"a\r\nb\r\nc\r\nd\r\ne"); // "a".."c" in history (full), "d","e" live
    assert_eq!(s.scrollback_len(), 3);
    s.scroll_view_to_top();
    assert_eq!(s.view_offset(), 3);
    assert_eq!(s.view_cell(0, 0).rune, 'a');

    feed(&mut s, b"\r\nf"); // "d" pushes in, "a" evicts

    assert_eq!(s.view_offset(), 3, "the offset saturates at the full ring");
    assert_eq!(s.scrollback_len(), 3);
    assert_eq!(
        s.view_cell(0, 0).rune,
        'b',
        "the oldest line slid out of view"
    );
    assert_eq!(s.view_cell(1, 0).rune, 'c');
}

#[test]
fn a_scroll_region_does_not_feed_a_scrolled_view() {
    // With a top margin below row 0 the rows leaving the region are discarded, not
    // retained, so there is no new history for the view to follow.
    let mut s = Screen::new(5, 3);
    feed(&mut s, b"a\r\nb\r\nc\r\nd"); // "a" in history; "b","c","d" live
    s.scroll_view_up(1);
    assert_eq!(s.view_offset(), 1);

    feed(&mut s, b"\x1b[2;3r"); // DECSTBM: region rows 2..3 (1-based)
    feed(&mut s, b"\x1b[3;1Hx\r\ny"); // a line feed at the bottom margin

    assert_eq!(s.scrollback_len(), 1, "a region scroll retains nothing");
    assert_eq!(s.view_offset(), 1, "so the view has nothing to follow");
    assert_eq!(s.view_cell(0, 0).rune, 'a');
}

#[test]
fn multi_row_scroll_moves_the_view_by_the_rows_that_entered_history() {
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd"); // "a","b" history; "c","d" live
    s.scroll_view_up(1);
    feed(&mut s, b"\r\ne\r\nf\r\ng\r\nh\r\ni"); // five line feeds
    assert_eq!(
        s.view_offset(),
        6,
        "the offset tracks rows entering history, not line feeds"
    );
    assert_eq!(s.view_offset(), s.scrollback_len() - 1);
    assert_eq!(s.view_cell(0, 0).rune, 'b', "the view held its content");
}

#[test]
fn su_retains_the_rows_that_leave_the_top_of_the_screen() {
    // SU with no top margin scrolls content off the top of the screen, and those
    // rows are history, not litter (xterm's `top_marg == 0`, and the same in
    // alacritty and ghostty). bnkterm used to drop them.
    let mut s = Screen::new(5, 3);
    feed(&mut s, b"a\r\nb\r\nc"); // fills the screen, no history yet
    assert_eq!(s.scrollback_len(), 0);

    feed(&mut s, b"\x1b[2S"); // SU 2

    assert_eq!(
        s.scrollback_len(),
        2,
        "the rows that left the top were kept"
    );
    s.scroll_view_to_top();
    assert_eq!(s.view_cell(0, 0).rune, 'a');
    assert_eq!(s.view_cell(1, 0).rune, 'b');
    assert_eq!(
        s.view_cell(2, 0).rune,
        'c',
        "and the live screen scrolled up"
    );
}

#[test]
fn su_inside_a_scroll_region_discards() {
    // With a top margin the rows leaving the region never reach the top of the
    // screen, so they were never history.
    let mut s = Screen::new(5, 3);
    feed(&mut s, b"a\r\nb\r\nc");
    feed(&mut s, b"\x1b[2;3r"); // DECSTBM: region rows 2..3
    feed(&mut s, b"\x1b[1S");
    assert_eq!(s.scrollback_len(), 0, "a region scroll retains nothing");
    assert_eq!(
        s.row_string(0).trim_end(),
        "a",
        "and rows above it hold still"
    );
}

#[test]
fn su_moves_a_scrolled_view_with_its_content() {
    let mut s = Screen::new(5, 3);
    feed(&mut s, b"a\r\nb\r\nc\r\nd"); // "a" in history; "b","c","d" live
    s.scroll_view_up(1);
    assert_eq!(s.view_cell(0, 0).rune, 'a');

    feed(&mut s, b"\x1b[2S"); // "b","c" enter history under a scrolled view

    assert_eq!(s.view_offset(), 3, "the offset followed the content");
    assert_eq!(
        s.view_cell(0, 0).rune,
        'a',
        "and the visible text held still"
    );
}

#[test]
fn dl_at_the_top_of_the_screen_discards_rather_than_retains() {
    // A deliberate divergence from xterm and alacritty, matching ghostty: DL is an
    // editing command, so the lines the application removed stay removed. See
    // [`Screen::delete_lines`].
    let mut s = Screen::new(5, 3);
    feed(&mut s, b"a\r\nb\r\nc");
    feed(&mut s, b"\x1b[1;1H\x1b[1M"); // home, DL 1
    assert_eq!(
        s.scrollback_len(),
        0,
        "the deleted line is gone, not archived"
    );
    assert_eq!(s.row_string(0).trim_end(), "b");
}

#[test]
fn erasing_the_scrollback_returns_the_view_to_the_bottom() {
    // ED 3 (what `clear` and `tput reset` emit) drops the history the view was
    // showing. An offset left pointing into an empty ring is a view of nothing.
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd");
    s.scroll_view_up(2);
    feed(&mut s, b"\x1b[3J");
    assert_eq!(s.scrollback_len(), 0);
    assert_eq!(s.view_offset(), 0, "the view cannot outlive the history");
    assert!(!s.is_scrolled());
}

#[test]
fn link_at_finds_a_url_and_spans_exactly_its_cells() {
    let mut s = Screen::new(40, 2);
    feed(&mut s, b"see https://example.com/a now");
    let mut probe = LinkProbe::default();
    // Cols 4..=24 hold the URL ("see " is 4 cells, the URL 21).
    let hit = s.link_at(0, 10, &mut probe).unwrap();
    assert_eq!(hit, ((0, 4), (0, 24)));
    assert_eq!(probe.url(), "https://example.com/a");
    // Every cell of the URL resolves to the same span, so the underline does not
    // flicker as the pointer crosses it.
    for col in 4..=24 {
        assert_eq!(s.link_at(0, col, &mut probe), Some(hit), "col {col}");
    }
    // The prose on either side is not a link.
    assert_eq!(s.link_at(0, 3, &mut probe), None, "the space before");
    assert_eq!(s.link_at(0, 25, &mut probe), None, "the space after");
    assert_eq!(probe.url(), "", "a miss leaves no stale URL behind");
}

#[test]
fn link_at_joins_a_soft_wrapped_url() {
    // A URL that runs off the right margin continues on the next row; half of it
    // is not a link, so the scan must read the logical line, not the display row.
    let mut s = Screen::new(16, 3);
    feed(&mut s, b"go https://example.com/xy");
    assert!(s.row_wraps(0));
    let mut probe = LinkProbe::default();
    // Hovering either half yields the whole link, spanning the wrap.
    let hit = ((0, 3), (1, 8));
    assert_eq!(s.link_at(0, 5, &mut probe), Some(hit), "the first row");
    assert_eq!(probe.url(), "https://example.com/xy");
    assert_eq!(s.link_at(1, 2, &mut probe), Some(hit), "past the wrap");
    assert_eq!(probe.url(), "https://example.com/xy");
}

#[test]
fn link_at_reads_the_scrollback_when_scrolled() {
    // The probe is a view concern: scrolled up, row 0 is history, and a link there
    // must be found at the cell it is actually *shown* in.
    let mut s = Screen::new(30, 2);
    feed(&mut s, b"https://example.com/old\r\nb\r\nc");
    s.scroll_view_up(2);
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 5, &mut probe), Some(((0, 0), (0, 22))));
    assert_eq!(probe.url(), "https://example.com/old");
}

#[test]
fn link_at_covers_both_halves_of_a_wide_glyph() {
    // A CJK character in an IRI occupies two cells. Probing the spacer must find
    // the link (the right half of a character is the character), and the span must
    // cover the spacer, or the underline would stop half a glyph short.
    let mut s = Screen::new(20, 2);
    feed(&mut s, "https://x.com/世".as_bytes());
    assert!(s.cell(0, 14).is_wide_leader());
    assert!(s.cell(0, 15).is_wide_spacer());
    let mut probe = LinkProbe::default();
    let hit = ((0, 0), (0, 15));
    assert_eq!(s.link_at(0, 14, &mut probe), Some(hit), "the leader");
    assert_eq!(s.link_at(0, 15, &mut probe), Some(hit), "the spacer");
    assert_eq!(probe.url(), "https://x.com/世");
}

#[test]
fn link_at_ignores_a_column_past_the_grid() {
    let mut s = Screen::new(30, 2);
    feed(&mut s, b"https://example.com/a");
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 99, &mut probe), None);
}

// ---- OSC 8 hyperlinks ---------------------------------------------------
//
// The anchor form throughout: `ESC ] 8 ; <params> ; <URI> ESC \` opens, the cells
// printed after it belong to the link, and `ESC ] 8 ; ; ESC \` closes.

/// `ESC ] 8 ; ; <uri> ESC \` — an anchor around `label`, closed after it.
fn anchor(uri: &str, label: &str) -> Vec<u8> {
    format!("\x1b]8;;{uri}\x1b\\{label}\x1b]8;;\x1b\\").into_bytes()
}

#[test]
fn an_anchor_makes_text_that_is_not_a_url_a_link() {
    // The whole point of OSC 8: the label need not look like a URL. No text scan
    // could ever find this one, which is exactly why the child marked it up.
    let mut s = Screen::new(20, 2);
    let mut bytes = anchor("https://example.com/a", "click me");
    bytes.extend_from_slice(b" nope");
    feed(&mut s, &bytes);

    let mut probe = LinkProbe::default();
    let hit = ((0, 0), (0, 7)); // "click me" is 8 cells
    for col in 0..=7 {
        assert_eq!(s.link_at(0, col, &mut probe), Some(hit), "col {col}");
        assert_eq!(probe.url(), "https://example.com/a");
    }
    assert_eq!(
        s.link_at(0, 8, &mut probe),
        None,
        "the space past the anchor"
    );
    assert_eq!(s.link_at(0, 10, &mut probe), None, "and the text after it");
    assert_eq!(probe.url(), "", "a miss leaves no stale URL behind");
}

#[test]
fn an_anchor_outranks_the_text_underneath_it() {
    // The label happens to be a URL, and a *different* one from the target. The
    // child said where these cells go; the scanner does not get a vote.
    let mut s = Screen::new(40, 2);
    feed(
        &mut s,
        &anchor("https://real.example/x", "https://decoy.example"),
    );
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 3, &mut probe), Some(((0, 0), (0, 20))));
    assert_eq!(probe.url(), "https://real.example/x");
}

#[test]
fn an_anchor_we_would_refuse_to_open_is_not_a_link_at_all() {
    // A child prints whatever it likes. A scheme outside `OPENABLE_SCHEMES` is
    // rejected when the anchor *opens*, so the label stays ordinary text and never
    // underlines. Decorating a link and then refusing to follow it is the bug this
    // exists to prevent (see `browser::OPENABLE_SCHEMES`).
    for uri in [
        "javascript:alert(1)",
        "data:text/html,<script>x</script>",
        "ftp://host/x",
        "-flag-that-xdg-open-would-eat",
    ] {
        let mut s = Screen::new(20, 2);
        feed(&mut s, &anchor(uri, "click"));
        let mut probe = LinkProbe::default();
        assert_eq!(s.cell(0, 0).link, LinkId::NONE, "{uri}");
        assert_eq!(s.link_at(0, 2, &mut probe), None, "{uri}");
    }
}

#[test]
fn an_sgr_reset_does_not_close_an_anchor() {
    // `ls --hyperlink` with colors sets a color inside the anchor and emits SGR 0
    // after the name while *still* inside it. If a rendition reset closed the link,
    // the most widespread OSC 8 producer there is would half-work. The link rides
    // on the pen but is not an SGR attribute; only `OSC 8 ; ;` closes it.
    let mut s = Screen::new(20, 2);
    feed(
        &mut s,
        b"\x1b]8;;https://example.com/d\x1b\\\x1b[34mdir\x1b[0m/\x1b]8;;\x1b\\",
    );
    let mut probe = LinkProbe::default();
    assert_eq!(
        s.link_at(0, 3, &mut probe),
        Some(((0, 0), (0, 3))),
        "the `/` printed after the reset is still inside the anchor"
    );
    assert_eq!(probe.url(), "https://example.com/d");
    assert_eq!(s.cell(0, 0).fg, Color::Ansi(4), "the color still applied");
    assert_eq!(
        s.cell(0, 3).fg,
        Color::Default,
        "and the reset still reset it"
    );
}

#[test]
fn an_anchor_joins_across_a_soft_wrap() {
    // Ten label cells on an eight-column grid: the anchor is one link across the
    // margin, not two, exactly as a wrapped bare URL is.
    let mut s = Screen::new(8, 3);
    feed(&mut s, &anchor("https://example.com/w", "abcdefghij"));
    let mut probe = LinkProbe::default();
    let hit = ((0, 0), (1, 1));
    assert_eq!(s.link_at(0, 4, &mut probe), Some(hit), "before the wrap");
    assert_eq!(s.link_at(1, 1, &mut probe), Some(hit), "after it");
    assert_eq!(probe.url(), "https://example.com/w");
}

#[test]
fn two_anchors_sharing_one_url_stay_two_links() {
    // Interning by URL hands both runs the same id. The *run* is what bounds a
    // link, so the plain cell between them still splits it into two hover targets.
    let mut s = Screen::new(20, 2);
    let mut bytes = anchor("https://e.com/a", "ab");
    bytes.extend_from_slice(b" ");
    bytes.extend(anchor("https://e.com/a", "cd"));
    feed(&mut s, &bytes);

    assert!(s.cell(0, 0).link.is_set());
    assert_eq!(s.cell(0, 0).link, s.cell(0, 3).link, "interned once");
    let mut probe = LinkProbe::default();
    assert_eq!(
        s.link_at(0, 0, &mut probe),
        Some(((0, 0), (0, 1))),
        "the first"
    );
    assert_eq!(
        s.link_at(0, 4, &mut probe),
        Some(((0, 3), (0, 4))),
        "the second"
    );
    assert_eq!(s.link_at(0, 2, &mut probe), None, "the gap between them");
}

#[test]
fn an_anchor_covers_both_halves_of_a_wide_glyph() {
    // The spacer carries the link too, so the run never breaks mid-character and
    // probing the right half of a wide glyph finds the glyph's link.
    let mut s = Screen::new(20, 2);
    feed(&mut s, &anchor("https://e.com/w", "世a"));
    assert!(s.cell(0, 0).is_wide_leader());
    assert!(s.cell(0, 1).is_wide_spacer());
    let mut probe = LinkProbe::default();
    let hit = ((0, 0), (0, 2));
    assert_eq!(s.link_at(0, 0, &mut probe), Some(hit), "the leader");
    assert_eq!(s.link_at(0, 1, &mut probe), Some(hit), "the spacer");
    assert_eq!(probe.url(), "https://e.com/w");
}

#[test]
fn erasing_a_cell_takes_it_out_of_its_anchor() {
    // Erased ground is inside no link (`blank_cell` carries `LinkId::NONE`), so the
    // run breaks there and the halves become separate links — the honest reading
    // once the label is no longer contiguous.
    let mut s = Screen::new(20, 2);
    feed(&mut s, &anchor("https://e.com/a", "abcde"));
    feed(&mut s, b"\x1b[1;3H\x1b[X"); // ECH one cell at col 2
    assert_eq!(s.cell(0, 2).link, LinkId::NONE);

    let mut probe = LinkProbe::default();
    assert_eq!(
        s.link_at(0, 0, &mut probe),
        Some(((0, 0), (0, 1))),
        "the left half"
    );
    assert_eq!(s.link_at(0, 2, &mut probe), None, "the erased cell");
    assert_eq!(
        s.link_at(0, 4, &mut probe),
        Some(((0, 3), (0, 4))),
        "the right half"
    );
}

#[test]
fn the_id_param_is_ignored_and_an_empty_uri_closes() {
    // `id=` is a hint that two runs are one link. We intern by URL, which already
    // gives identical URLs one identity, so the hint buys nothing — and honouring a
    // child-chosen id would let it fuse unrelated runs into one hover target.
    let mut s = Screen::new(20, 2);
    feed(
        &mut s,
        b"\x1b]8;id=xyz;https://e.com/a\x1b\\ab\x1b]8;;\x1b\\cd",
    );
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 1, &mut probe), Some(((0, 0), (0, 1))));
    assert_eq!(probe.url(), "https://e.com/a");
    assert_eq!(s.link_at(0, 2, &mut probe), None, "the empty URI closed it");
}

#[test]
fn a_full_link_table_collects_dead_ids_and_keeps_linking() {
    // Burn the whole u16 id space on anchors that then scroll out of history. The
    // next one must still work: without the collector the table would stay full for
    // the life of the process and OSC 8 would quietly stop linking anything.
    let mut s = Screen::new(20, 2);
    let mut bytes = Vec::new();
    for i in 0..LINK_LIMIT {
        bytes.extend(anchor(&format!("https://e.com/{i}"), "x"));
        bytes.extend_from_slice(b"\r\n");
    }
    feed(&mut s, &bytes);
    assert_eq!(s.links.urls.len(), LINK_LIMIT, "the id space is spent");

    // Row 0 shows the last anchor printed; row 1 is where the cursor now sits.
    let last = format!("https://e.com/{}", LINK_LIMIT - 1);
    let mut probe = LinkProbe::default();
    assert_eq!(s.link_at(0, 0, &mut probe), Some(((0, 0), (0, 0))));
    assert_eq!(probe.url(), last);

    // One more distinct URL: intern finds the table full, collects, and succeeds.
    feed(&mut s, &anchor("https://e.com/fresh", "NEW"));
    assert!(
        s.links.urls.len() < LINK_LIMIT,
        "the ids of the scrolled-out anchors came back"
    );
    assert_eq!(s.link_at(1, 1, &mut probe), Some(((1, 0), (1, 2))));
    assert_eq!(probe.url(), "https://e.com/fresh");

    // And the survivors were *renumbered*, not orphaned: a cell that outlived the
    // collection still resolves to the URL it always had.
    assert_eq!(s.link_at(0, 0, &mut probe), Some(((0, 0), (0, 0))));
    assert_eq!(
        probe.url(),
        last,
        "a survivor kept its URL across the collection"
    );
}

/// The absolute cell under display `(row, col)`, which is what a pointer resolves to.
fn at(s: &Screen, row: usize, col: usize) -> (AbsRow, usize) {
    (s.abs_row(row), col)
}

#[test]
fn selection_text_extracts_and_joins_rows() {
    let mut s = Screen::new(10, 3);
    feed(&mut s, b"hello\r\nworld\r\nfoo");
    // A single-row partial selection: cols 0..=4 of row 0.
    assert_eq!(s.selection_text(at(&s, 0, 0), at(&s, 0, 4)), "hello");
    // Across rows: end of row 0 through part of row 1, joined by a newline;
    // trailing blanks on the first row are trimmed.
    assert_eq!(s.selection_text(at(&s, 0, 0), at(&s, 1, 2)), "hello\nwor");
    // A mid-row start.
    assert_eq!(s.selection_text(at(&s, 0, 2), at(&s, 0, 4)), "llo");
    // Either order: the endpoints sort into reading order.
    assert_eq!(s.selection_text(at(&s, 1, 2), at(&s, 0, 0)), "hello\nwor");
}

#[test]
fn selection_joins_a_soft_wrapped_line() {
    // A line longer than the width soft-wraps; selecting across the wrap must
    // copy it as one unbroken logical line (no newline at the wrap point).
    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcdef"); // wraps: "abcd" | "ef"
    assert!(s.row_wraps(0));
    assert_eq!(s.selection_text(at(&s, 0, 0), at(&s, 1, 1)), "abcdef");
}

#[test]
fn a_resize_rewraps_a_soft_wrapped_line() {
    // Reflow re-lays a wrapped line to the new width: it wraps only while it does not
    // fit, and rejoins onto one row once it does. (Before reflow this checked the wrap
    // link merely survived in place; now the line actively re-flows.)
    for new_cols in [3, 4, 6, 12] {
        let mut s = Screen::new(4, 3);
        feed(&mut s, b"abcdef"); // wraps: "abcd" | "ef"
        s.resize(new_cols, 3);
        if new_cols < 6 {
            assert!(
                s.row_wraps(0),
                "{new_cols} cols: still too narrow, so it wraps"
            );
        } else {
            assert!(!s.row_wraps(0), "{new_cols} cols: it rejoined onto one row");
            assert_eq!(s.row_string(0).trim_end(), "abcdef");
        }
    }
}

#[test]
fn a_resized_wrapped_line_copies_unbroken() {
    // Drag over an old wrapped line after a resize and the clipboard must not break
    // mid-sentence. A widen rejoins it onto one row; a narrow re-wraps it; both still
    // copy as one unbroken logical line.
    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcdef"); // "abcd" | "ef"
    s.resize(6, 3); // widen: rejoins onto one row
    assert!(!s.row_wraps(0));
    assert_eq!(s.selection_text(at(&s, 0, 0), at(&s, 0, 5)), "abcdef");

    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcdef");
    s.resize(3, 3); // narrow: re-wraps to "abc" | "def"
    assert!(s.row_wraps(0));
    assert_eq!(s.selection_text(at(&s, 0, 0), at(&s, 1, 2)), "abcdef");
}

#[test]
fn a_recycled_row_never_inherits_a_wrap() {
    // A row's storage is reused rather than reallocated, so a stale wrap link is the
    // opposite failure mode: it glues two unrelated lines into one on copy. `DL`
    // recycles the deleted row straight to the bottom of the region, which is the
    // shortest path from a wrapped row to a blank one wearing its flag.
    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcdef"); // row 0 "abcd" wraps into row 1 "ef"
    assert!(s.row_wraps(0));
    feed(&mut s, b"\x1b[H\x1b[M"); // home, then delete row 0
    assert_eq!(s.row_string(0).trim_end(), "ef");
    assert!(!s.row_wraps(2), "the recycled row is not still wrapped");
    // Two fresh lines in the recycled rows copy as two lines, not one.
    feed(&mut s, b"\x1b[2;1Hxy\r\nzw");
    assert_eq!(s.selection_text(at(&s, 1, 0), at(&s, 2, 1)), "xy\nzw");
}

#[test]
fn rewriting_the_last_column_ends_the_wrap() {
    // The wrap link is only true while the text that wrapped is still the text in
    // the final column. Overwrite it (here without wrapping again) and the line
    // ends there, so a copy across the two rows takes the newline back.
    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcdef"); // "abcd" wraps into "ef"
    feed(&mut s, b"\x1b[1;4Hz"); // print 'z' over the last column of row 0
    assert!(!s.row_wraps(0));
    assert_eq!(s.selection_text(at(&s, 0, 0), at(&s, 1, 1)), "abcz\nef");
}

#[test]
fn erasing_the_tail_of_a_wrapped_row_ends_the_wrap() {
    // EL 0 (cursor to end of line) takes the wrapped text with it; so does DCH,
    // which blanks the tail as it shifts the rest left. Neither leaves a line that
    // still continues onto the next row.
    for erase in [&b"\x1b[1;3H\x1b[K"[..], &b"\x1b[1;3H\x1b[2P"[..]] {
        let mut s = Screen::new(4, 3);
        feed(&mut s, b"abcdef");
        assert!(s.row_wraps(0));
        feed(&mut s, erase);
        assert!(!s.row_wraps(0), "{erase:?} ends the line");
    }
}

#[test]
fn a_hard_newline_at_the_margin_is_not_a_wrap() {
    // The case the flag exists to tell apart: text that exactly fills the row and
    // then ends with a real newline is two lines, and must copy as two.
    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcd\r\nef");
    assert!(!s.row_wraps(0));
    assert_eq!(s.selection_text(at(&s, 0, 0), at(&s, 1, 1)), "abcd\nef");
}

#[test]
fn selection_reads_scrollback_when_scrolled() {
    let mut s = Screen::new(6, 2);
    feed(&mut s, b"one\r\ntwo\r\nthree\r\nfour");
    s.scroll_view_up(2); // top: "one" over "two"
    assert_eq!(s.selection_text(at(&s, 0, 0), at(&s, 0, 2)), "one");
}

#[test]
fn a_selection_copies_its_rows_after_output_scrolls_them_away() {
    // The absolute-row payoff: the same ids, resolved after the grid has scrolled
    // under them, still name the lines the user picked — including once they have
    // scrolled out of sight entirely.
    let mut s = Screen::new(6, 2);
    feed(&mut s, b"one\r\ntwo");
    let (start, end) = (at(&s, 0, 0), at(&s, 0, 2)); // "one", on the live screen
    assert_eq!(s.selection_text(start, end), "one");

    feed(&mut s, b"\r\nthree\r\nfour\r\nfive"); // "one" is pushed deep into history

    assert_eq!(s.display_row(start.0), None, "it is off screen now");
    assert_eq!(
        s.selection_text(start, end),
        "one",
        "but the ids still name the line it was made over"
    );
}

#[test]
fn a_row_id_rides_along_with_its_content() {
    // The invariant the whole model rests on: the id names a *line*, so as the line
    // moves up the screen and into history, the id keeps resolving to it.
    let mut s = Screen::new(6, 3);
    feed(&mut s, b"one\r\ntwo\r\nthree");
    let one = s.abs_row(0);
    assert_eq!(s.display_row(one), Some(0));

    feed(&mut s, b"\r\nfour"); // "one" is pushed into history
    assert_eq!(s.scrollback_len(), 1);
    assert_eq!(s.display_row(one), None, "off the top of the live band");
    assert!(s.row_exists(one), "but still in history");
    assert_eq!(
        s.abs_cell(one, 0).rune,
        'o',
        "and it is still the same line"
    );

    s.scroll_view_up(1); // scroll it back into view
    assert_eq!(
        s.display_row(one),
        Some(0),
        "the same id, now on screen again"
    );
    assert_eq!(s.abs_row(0), one);
}

#[test]
fn ordinary_output_never_breaks_the_row_epoch() {
    // Printing, wrapping, and scrolling into history are the whole steady state of a
    // terminal. If any of them ended the epoch, a selection could not survive output
    // and the feature would be a lie.
    let mut s = Screen::new(4, 2);
    let epoch = s.row_epoch();
    feed(&mut s, b"abcdefgh\r\nmore\r\ntext\r\nstill going");
    assert!(s.scrollback_len() > 0, "it really did scroll");
    assert_eq!(s.row_epoch(), epoch, "and no id was invalidated");
}

#[test]
fn tearing_a_row_out_of_the_middle_ends_the_row_epoch() {
    // A region scroll and a DL both delete a row from the middle of the stream: every
    // row below shifts up, so the ids no longer name the same lines and anything
    // holding one has to be told.
    let mut s = Screen::new(5, 3);
    feed(&mut s, b"a\r\nb\r\nc");

    let epoch = s.row_epoch();
    feed(&mut s, b"\x1b[2;3r\x1b[3;1H\r\n"); // DECSTBM, then a feed at the margin
    assert_ne!(s.row_epoch(), epoch, "a region scroll discarded a row");

    let epoch = s.row_epoch();
    feed(&mut s, b"\x1b[r\x1b[1;1H\x1b[1M"); // full screen, home, DL
    assert_ne!(s.row_epoch(), epoch, "and DL deleted one");
}

#[test]
fn a_top_region_that_stops_short_of_the_bottom_renumbers_the_rows_below_it() {
    // The subtle one. The region starts at the top, so the row leaving it retires into
    // history and keeps its id — but the blank filling in behind it lands at the
    // region's bottom margin, which is *mid-stream*, so every row below the region
    // shifts up an id while sitting perfectly still on screen. (A status line under a
    // scroll region is exactly this shape.) Reporting only the push would leave a
    // selection on that status line quietly pointing at the blank instead.
    let mut s = Screen::new(8, 4);
    feed(&mut s, b"a\r\nb\r\nc\r\nstatus");
    feed(&mut s, b"\x1b[1;3r"); // DECSTBM rows 1..3: the top three, not the last
    let status = s.abs_row(3);
    assert_eq!(s.abs_cell(status, 0).rune, 's');

    let epoch = s.row_epoch();
    feed(&mut s, b"\x1b[3;1H\r\n"); // a line feed at the region's bottom margin

    assert_eq!(
        s.scrollback_len(),
        1,
        "the row off the top was still retained"
    );
    assert_eq!(
        s.row_string(3).trim_end(),
        "status",
        "and the status line never moved"
    );
    assert_ne!(
        s.abs_cell(status, 0).rune,
        's',
        "yet its old id now names the blank the scroll left behind"
    );
    assert_ne!(
        s.row_epoch(),
        epoch,
        "so the ids below the region must be declared dead"
    );
}

#[test]
fn a_full_screen_scroll_keeps_every_id() {
    // The contrast that makes the case above legible: with the region running to the
    // last row, the blank appends to the *end* of the stream and nothing is renumbered
    // — which is why ordinary output can leave a selection alone.
    let mut s = Screen::new(8, 4);
    feed(&mut s, b"a\r\nb\r\nc\r\nlast");
    let last = s.abs_row(3);
    let epoch = s.row_epoch();

    feed(&mut s, b"\r\nmore");

    assert_eq!(s.scrollback_len(), 1);
    assert_eq!(s.row_epoch(), epoch, "no id was invalidated");
    assert_eq!(s.abs_cell(last, 0).rune, 'l', "and they all still hold");
    assert_eq!(s.display_row(last), Some(2), "one row higher, same line");
}

#[test]
fn scrolling_the_content_down_ends_the_row_epoch() {
    // The mirror of tearing a row out of the top: RI, SD and IL all slide the content
    // *down* while the ids stay where they are, so the line that was AbsRow(n) is now
    // at AbsRow(n+1). Nothing is pushed to history and nothing obviously "leaves", so
    // it is easy to miss — but a selection held across one would drift a row, which is
    // exactly the silent wrongness absolute rows exist to prevent.
    let mut s = Screen::new(5, 3);

    feed(&mut s, b"a\r\nb\r\nc");
    let epoch = s.row_epoch();
    feed(&mut s, b"\x1b[1;1H\x1b[1L"); // IL at the top: everything slides down
    assert_eq!(s.row_string(1).trim_end(), "a", "the content moved down");
    assert_ne!(
        s.row_epoch(),
        epoch,
        "so the ids no longer name those lines"
    );

    let epoch = s.row_epoch();
    feed(&mut s, b"\x1b[2T"); // SD
    assert_ne!(s.row_epoch(), epoch);

    let epoch = s.row_epoch();
    feed(&mut s, b"\x1b[1;1H\x1bM"); // RI at the top margin
    assert_ne!(s.row_epoch(), epoch);

    // But an RI that merely moves the cursor up displaces nothing.
    let epoch = s.row_epoch();
    feed(&mut s, b"\x1b[3;1H\x1bM");
    assert_eq!(s.row_epoch(), epoch, "no rows moved, no ids invalidated");
}

#[test]
fn the_alt_screen_and_a_reset_end_the_row_epoch() {
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb");

    let epoch = s.row_epoch();
    feed(&mut s, b"\x1b[?1049h"); // a different buffer: ids mean nothing there
    assert_ne!(s.row_epoch(), epoch);

    // The alt screen keeps no history, so its scrolls discard rows and end the epoch
    // too: nothing on it can be tracked across a scroll.
    let epoch = s.row_epoch();
    feed(&mut s, b"x\r\ny\r\nz");
    assert_ne!(s.row_epoch(), epoch, "an alt-screen scroll discards");

    let epoch = s.row_epoch();
    feed(&mut s, b"\x1bc"); // RIS
    assert_ne!(s.row_epoch(), epoch, "a reset never rewinds identity");
}

#[test]
fn erasing_the_history_keeps_the_live_rows_ids() {
    // ED 3 drops history off the *front* of the stream, which is an eviction like any
    // other: the live rows have not moved, so they keep the ids they had. (Getting
    // this wrong would renumber them and quietly alias a stale id onto a live line.)
    let mut s = Screen::new(5, 2);
    feed(&mut s, b"a\r\nb\r\nc\r\nd"); // "a","b" history; "c","d" live
    let c = s.abs_row(0);
    let gone = s.primary.abs_of(0);
    assert!(s.row_exists(gone));

    feed(&mut s, b"\x1b[3J");

    assert_eq!(s.scrollback_len(), 0);
    assert!(!s.row_exists(gone), "the history rows are gone for good");
    assert_eq!(s.display_row(c), Some(0), "the live rows kept their ids");
    assert_eq!(s.abs_row(0), c);
}

#[test]
fn word_at_spans_a_token_and_stops_at_boundaries() {
    let mut s = Screen::new(20, 1);
    feed(&mut s, b"cd /usr/bin (ok)");
    let abs = s.abs_row(0);
    // A click anywhere in "cd" selects just "cd" (cols 0..=1); a space bounds it.
    assert_eq!(s.word_at(abs, 1), ((abs, 0), (abs, 1)));
    // The path is one word: '/' is not a boundary, so double-click grabs it whole.
    let path = s.word_at(abs, 5);
    assert_eq!(s.selection_text(path.0, path.1), "/usr/bin");
    // A parenthesis bounds the token, and clicking the space between words
    // selects just that cell.
    assert_eq!(
        s.word_at(abs, 11),
        ((abs, 11), (abs, 11)),
        "the space is its own cell"
    );
    let ok = s.word_at(abs, 13);
    assert_eq!(s.selection_text(ok.0, ok.1), "ok");
}

#[test]
fn line_at_spans_a_soft_wrapped_logical_line() {
    // "abcdef" in a 4-wide grid wraps to "abcd" | "ef"; a triple-click on either
    // display row selects the whole logical line, edge to edge.
    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcdef");
    let (r0, r1) = (s.abs_row(0), s.abs_row(1));
    assert_eq!(s.line_at(r0), ((r0, 0), (r1, 3)));
    assert_eq!(s.line_at(r1), ((r0, 0), (r1, 3)));
    let line = s.line_at(r1);
    assert_eq!(s.selection_text(line.0, line.1), "abcdef");
}

#[test]
fn line_at_follows_a_wrap_that_starts_above_the_visible_band() {
    // The walk runs over the stream, not the window: the head of this logical line
    // has scrolled into history, and a triple-click on its tail must still take it
    // whole. (In display rows the walk stopped at the top of the screen and copied
    // half a line.)
    let mut s = Screen::new(4, 2);
    feed(&mut s, b"abcdefgh"); // "abcd" | "efgh", filling both rows
    feed(&mut s, b"\r\nx"); // scrolls "abcd" into history
    assert_eq!(s.scrollback_len(), 1);
    let tail = s.abs_row(0); // "efgh", now the top visible row
    let line = s.line_at(tail);
    assert_eq!(
        s.selection_text(line.0, line.1),
        "abcdefgh",
        "the line was taken whole, across the top of the screen"
    );
}

#[test]
fn device_attributes_and_status_queries_get_answers() {
    let mut s = Screen::new(80, 24);
    // Primary DA.
    feed(&mut s, b"\x1b[c");
    assert_eq!(s.take_responses(), b"\x1b[?1;2c");
    assert!(s.take_responses().is_empty(), "the queue drains");
    // Secondary DA.
    feed(&mut s, b"\x1b[>c");
    assert_eq!(s.take_responses(), b"\x1b[>0;0;0c");
    // DSR status: OK.
    feed(&mut s, b"\x1b[5n");
    assert_eq!(s.take_responses(), b"\x1b[0n");
}

#[test]
fn xtversion_says_who_we_are() {
    // It gets us onto nobody's allowlist — that is not what it is for. It is what a
    // terminal is supposed to say when asked, and silence is how you stay unknown.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b[>0q");
    let reply = s.take_responses();
    let text = String::from_utf8_lossy(&reply).into_owned();
    assert!(text.starts_with("\x1bP>|bnkterm "), "{text:?}");
    assert!(text.ends_with("\x1b\\"), "{text:?}");
    assert!(text.contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn decrqss_answers_with_the_sequence_that_would_restore_the_setting() {
    // A program saves a setting, changes it, and puts it back without ever knowing
    // what it was, so the reply is a sequence to replay, not a description.
    let mut s = Screen::new(10, 3);
    feed(&mut s, b"\x1bP$qm\x1b\\"); // "what is SGR?"
    assert_eq!(s.take_responses(), b"\x1bP1$r0m\x1b\\");

    feed(&mut s, b"\x1b[1;4;38;5;200m"); // bold, underlined, colour 200
    feed(&mut s, b"\x1bP$qm\x1b\\");
    assert_eq!(s.take_responses(), b"\x1bP1$r0;1;4;38;5;200m\x1b\\");

    // The scroll region, 1-based like the sequence that sets it.
    feed(&mut s, b"\x1b[2;3r");
    feed(&mut s, b"\x1bP$qr\x1b\\");
    assert_eq!(s.take_responses(), b"\x1bP1$r2;3r\x1b\\");

    // A setting we do not report is answered "not supported" — a real answer, so the
    // program stops waiting. Claiming support and inventing a value would be worse
    // than silence: it would be restored verbatim.
    feed(&mut s, b"\x1bP$q\"p\x1b\\"); // DECSCL
    assert_eq!(s.take_responses(), b"\x1bP0$r\x1b\\");
}

#[test]
fn xtgettcap_answers_what_our_terminfo_would_have_said() {
    // The query that lets a program learn what we can do *without* a terminfo entry
    // installed for us — which matters, because we ship none and claim to be xterm.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1bP+q544e\x1b\\"); // "TN" — terminal name
    assert_eq!(s.take_responses(), b"\x1bP1+r544e=626e6b7465726d\x1b\\");

    feed(&mut s, b"\x1bP+q524742\x1b\\"); // "RGB" — do you mean it about truecolor?
    assert_eq!(s.take_responses(), b"\x1bP1+r524742=382f382f38\x1b\\");

    // Several at once, and an unknown one answered 0 rather than ignored.
    feed(&mut s, b"\x1bP+q436f;7a7a7a\x1b\\"); // "Co" (colors), then "zzz"
    assert_eq!(
        s.take_responses(),
        b"\x1bP1+r436f=323536\x1b\\\x1bP0+r7a7a7a\x1b\\"
    );

    // Not hex at all: still an answer, still not a guess.
    feed(&mut s, b"\x1bP+qnothex\x1b\\");
    assert_eq!(s.take_responses(), b"\x1bP0+r\x1b\\");

    // This reply is the *only* one in the terminal that puts child-supplied bytes back on
    // the pty (the name returns verbatim, because it is already hex on the wire). What
    // makes that safe is `hex_decode` refusing anything but hex digits *before* the echo,
    // so a name carrying a control byte is answered without being repeated. Asserted here
    // rather than argued in prose, because it is the property that would
    // silently stop holding if that validator were ever loosened.
    for hostile in [
        &b"\x1bP+q54\x074e\x1b\\"[..],     // BEL inside the name
        &b"\x1bP+q54\x0d\x0a4e\x1b\\"[..], // CR LF inside the name
        &b"\x1bP+q544\x1b\\"[..],          // odd length, so half a byte
    ] {
        feed(&mut s, hostile);
        let reply = s.take_responses();
        assert_eq!(
            reply, b"\x1bP0+r\x1b\\",
            "refused without echo: {hostile:?}"
        );
        assert!(
            !reply.iter().any(|b| matches!(b, 0x00..=0x06 | 0x08..=0x1a)),
            "no control byte from the request reached the reply: {reply:?}"
        );
    }
}

#[test]
fn a_dcs_we_do_not_answer_is_swallowed_whole() {
    // Sixel arrives here (`DCS q …`), and so does anything else we have no use for.
    // The payload must never leak onto the screen as text.
    let mut s = Screen::new(20, 2);
    feed(&mut s, b"ab\x1bPq#0;2;0;0;0#0~~@@vv@@~~@@~~$\x1b\\cd");
    assert_eq!(s.row_string(0).trim_end(), "abcd");
    assert!(s.take_responses().is_empty());
}

#[test]
fn in_band_resize_reports_only_to_a_program_that_asked() {
    let mut s = Screen::new(80, 24);
    s.set_pixel_size(640, 384);

    // Nobody asked: a resize says nothing. A program that never enabled `?2048` would
    // read the report as garbage on its input.
    s.resize(100, 30);
    assert!(s.take_responses().is_empty());

    // Enabling it reports immediately, which is the point: a program handed a fresh
    // pty has no other way to ask how big it is.
    feed(&mut s, b"\x1b[?2048h");
    assert_eq!(s.take_responses(), b"\x1b[48;30;100;384;640t");

    // And every resize after it. This is the news SIGWINCH cannot carry across an ssh
    // hop or a tmux, because the signal stops at this side of the pty.
    s.set_pixel_size(800, 400);
    s.resize(120, 40);
    s.report_size();
    assert_eq!(s.take_responses(), b"\x1b[48;40;120;400;800t");

    feed(&mut s, b"\x1b[?2048l");
    s.report_size();
    assert!(s.take_responses().is_empty(), "and it can be turned off");
}

#[test]
fn bel_rings_the_bell_and_disturbs_nothing_else() {
    // BEL was dropped on the floor entirely. It is also the *documented fallback* for
    // desktop notifications in the CLIs that will not send them to a terminal they do
    // not recognise, so ignoring it broke the workaround for a thing we cannot fix.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"ab\x07cd");
    assert!(s.take_bell(), "it rang");
    assert!(!s.take_bell(), "and taking it clears it");
    assert_eq!(s.row_string(0).trim_end(), "abcd", "and printed nothing");
    assert_eq!(s.cursor(), (0, 4), "and moved no cursor");
}

#[test]
fn focus_events_are_reported_only_to_a_program_that_asked() {
    // A program that never enabled `?1004` would read `CSI I` as a stray key press,
    // so silence is not politeness here, it is correctness.
    let mut s = Screen::new(10, 2);
    s.report_focus(true);
    s.report_focus(false);
    assert!(s.take_responses().is_empty(), "nobody asked");

    feed(&mut s, b"\x1b[?1004h");
    s.report_focus(true);
    assert_eq!(s.take_responses(), b"\x1b[I");
    s.report_focus(false);
    assert_eq!(s.take_responses(), b"\x1b[O");

    // And it can be turned back off.
    feed(&mut s, b"\x1b[?1004l");
    s.report_focus(true);
    assert!(s.take_responses().is_empty());
}

#[test]
fn osc_52_puts_the_childs_text_on_the_clipboard() {
    // How nvim over SSH, tmux's copy mode and Claude Code's `/copy` reach the system
    // clipboard: they cannot see the compositor, and the terminal can.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b]52;c;aGVsbG8=\x07"); // "hello"
    assert_eq!(
        s.take_clipboard_writes(),
        vec![(ClipboardTarget::Clipboard, b"hello".to_vec())]
    );
    assert!(s.take_clipboard_writes().is_empty(), "the queue drains");

    // `p` is the primary selection, and a program may name several at once.
    feed(&mut s, b"\x1b]52;p;aGk=\x07");
    assert_eq!(
        s.take_clipboard_writes(),
        vec![(ClipboardTarget::Primary, b"hi".to_vec())]
    );
    feed(&mut s, b"\x1b]52;pc;aGk=\x07");
    assert_eq!(
        s.take_clipboard_writes(),
        vec![
            (ClipboardTarget::Clipboard, b"hi".to_vec()),
            (ClipboardTarget::Primary, b"hi".to_vec()),
        ]
    );
    // An empty selection field means the clipboard.
    feed(&mut s, b"\x1b]52;;aGk=\x07");
    assert_eq!(
        s.take_clipboard_writes(),
        vec![(ClipboardTarget::Clipboard, b"hi".to_vec())]
    );
}

#[test]
fn osc_52_never_reads_the_clipboard_back() {
    // The protocol defines a read, and we refuse it on purpose.
    //
    // A terminal cannot tell which program printed a byte. `cat` a hostile file and
    // it can send this; the reply goes straight back down the pty to whatever is
    // reading. That turns "display some text" into "exfiltrate the clipboard", which
    // routinely holds passwords. The answer is silence, not a reply.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b]52;c;?\x07");
    assert!(s.take_responses().is_empty(), "no answer, ever");
    assert!(s.take_clipboard_writes().is_empty(), "and nothing written");
}

#[test]
fn a_malformed_osc_52_is_dropped_rather_than_half_decoded() {
    // The child writes this, so it is hostile input. A decoder that improvises on bad
    // bytes is how you smuggle bytes past it.
    let mut s = Screen::new(10, 2);
    for junk in [
        &b"\x1b]52;c;not base64!!\x07"[..],
        &b"\x1b]52;c;aGVsbG8\x3d\x3d\x3d\x07"[..], // three padding characters
        &b"\x1b]52;c;a\x07"[..],                   // a lone leftover character
        &b"\x1b]52;c;aG=sbG8=\x07"[..],            // padding in the middle
    ] {
        feed(&mut s, junk);
        assert!(
            s.take_clipboard_writes().is_empty(),
            "refused: {:?}",
            std::str::from_utf8(junk)
        );
    }
}

#[test]
fn base64_decodes_what_a_terminal_actually_receives() {
    assert_eq!(base64_decode(b"aGVsbG8="), Some(b"hello".to_vec()));
    assert_eq!(base64_decode(b"aGVsbG8h"), Some(b"hello!".to_vec()));
    assert_eq!(base64_decode(b"eA=="), Some(b"x".to_vec()));
    assert_eq!(base64_decode(b""), Some(Vec::new()));
    // Padding is optional, and line breaks (mail-era encoders wrap) are skipped.
    assert_eq!(base64_decode(b"aGVsbG8"), Some(b"hello".to_vec()));
    assert_eq!(base64_decode(b"aGVs\nbG8="), Some(b"hello".to_vec()));
    // Every byte of the alphabet, including the two that are not alphanumeric.
    assert_eq!(base64_decode(b"+/8="), Some(vec![0xfb, 0xff]));
    // And the refusals.
    assert_eq!(base64_decode(b"a"), None);
    assert_eq!(base64_decode(b"a==="), None);
    assert_eq!(base64_decode(b"aa=a"), None);
    assert_eq!(base64_decode(b"////_"), None);

    // Padding only ever *completes* a four-character group, so a group that ended early
    // cannot have been padded. These are refusals rather than short decodes, and the
    // distinction is not academic: the padding used to be counted as a data character, so
    // `AA=` decoded to *two* bytes where the same input without the `=` gives one — the
    // decoder inventing a byte out of a malformed length, which is exactly what the strict
    // contract above exists to prevent.
    assert_eq!(base64_decode(b"AA="), None, "a padded two-character group");
    assert_eq!(base64_decode(b"A="), None, "a padded one-character group");
    assert_eq!(
        base64_decode(b"AAA=="),
        None,
        "padding past the group boundary"
    );
    // The well-formed neighbours of each, so the refusals above cannot be over-broad.
    assert_eq!(base64_decode(b"AA"), Some(vec![0x00]));
    assert_eq!(base64_decode(b"AAA"), Some(vec![0x00, 0x00]));
    assert_eq!(base64_decode(b"AAA="), Some(vec![0x00, 0x00]));
    assert_eq!(base64_decode(b"AA=="), Some(vec![0x00]));
}

#[test]
fn a_flood_of_queries_cannot_mint_unbounded_replies() {
    // The child decides how many questions it asks, so reply generation has to be
    // bounded here rather than trusting the far side to drain. A file full of `\x1b[c`
    // is ~21,800 device-attributes queries per 64 KiB, each answered — and the `cat`
    // printing it never reads its stdin, so nothing consumes the answers.
    let mut s = Screen::new(80, 24);
    let flood = b"\x1b[c".repeat(40_000);
    feed(&mut s, &flood);
    // A soft cap by design: the budget is checked before an answer is built, never
    // during, so the last one runs past the line rather than being cut in half. The
    // overshoot is one reply, and the flood it replaces is ~280 KB of answers.
    let len = s.responses().len();
    assert!(len >= RESPONSE_MAX, "the cap is where the flood stopped");
    assert!(
        len < RESPONSE_MAX + 1024,
        "and it overshoots by at most the one answer already under way, got {len}"
    );

    // Capped, not truncated: what is there is whole answers. A half-written reply is a
    // malformed escape sequence in the child's input, which is worse than no reply.
    let one = b"\x1b[?1;2c";
    assert!(
        !s.responses().is_empty(),
        "and the early ones were answered"
    );
    assert_eq!(
        s.responses().len() % one.len(),
        0,
        "the cap fell on an answer boundary"
    );
    assert!(s.responses().chunks(one.len()).all(|c| c == one));

    // Draining resets the budget: it is a per-parse bound, not a lifetime one, so a
    // long-lived shell asking a normal question an hour later still gets an answer.
    s.clear_responses();
    feed(&mut s, b"\x1b[c");
    assert_eq!(s.responses(), one);
}

#[test]
fn decrqm_reports_the_modes_we_have_and_admits_the_ones_we_do_not() {
    // Of the five modes nvim asks about before drawing anything, we now implement all
    // but one. The honest answer for that one is 0 ("I do not know this mode") — never
    // 2, which would mean "I know it and it is off" and invite nvim to turn it on and
    // then rely on it.
    let mut s = Screen::new(10, 2);
    // `?2031` (colour-scheme change notification) is the one left, and we say so.
    feed(&mut s, b"\x1b[?2031$p");
    assert_eq!(s.take_responses(), b"\x1b[?2031;0$y", "not ours to claim");

    // The ones we do implement report their real state, and follow it as it changes.
    feed(&mut s, b"\x1b[?7$p"); // DECAWM, on by default
    assert_eq!(s.take_responses(), b"\x1b[?7;1$y");
    feed(&mut s, b"\x1b[?7l\x1b[?7$p"); // turn it off, ask again
    assert_eq!(s.take_responses(), b"\x1b[?7;2$y");

    feed(&mut s, b"\x1b[?2004$p"); // bracketed paste, off by default
    assert_eq!(s.take_responses(), b"\x1b[?2004;2$y");
    feed(&mut s, b"\x1b[?2004h\x1b[?2004$p");
    assert_eq!(s.take_responses(), b"\x1b[?2004;1$y");

    // Synchronized output: we *do* implement it, so it reports honestly rather than
    // 0 — which is the whole point of answering, since a mode nobody can discover is
    // a mode nobody will ever use.
    feed(&mut s, b"\x1b[?2026$p");
    assert_eq!(s.take_responses(), b"\x1b[?2026;2$y");
    feed(&mut s, b"\x1b[?2026h\x1b[?2026$p");
    assert_eq!(s.take_responses(), b"\x1b[?2026;1$y");

    // Same for the rest of what this round added, which is the whole point of keeping
    // the two tables in step: a mode we act on is a mode we report.
    feed(&mut s, b"\x1b[?1004$p");
    assert_eq!(s.take_responses(), b"\x1b[?1004;2$y");
    feed(&mut s, b"\x1b[?2048$p");
    assert_eq!(s.take_responses(), b"\x1b[?2048;2$y");
    // Enabling `?2048` also reports the size, so drain that before asking again.
    feed(&mut s, b"\x1b[?2048h");
    let _ = s.take_responses();
    feed(&mut s, b"\x1b[?2048$p");
    assert_eq!(s.take_responses(), b"\x1b[?2048;1$y");

    // The ANSI space answers on its own form, without the `?`.
    feed(&mut s, b"\x1b[4$p"); // IRM
    assert_eq!(s.take_responses(), b"\x1b[4;2$y");
    feed(&mut s, b"\x1b[4h\x1b[4$p");
    assert_eq!(s.take_responses(), b"\x1b[4;1$y");
    // And an ANSI mode we do not know is 0, like any other.
    feed(&mut s, b"\x1b[20$p"); // LNM
    assert_eq!(s.take_responses(), b"\x1b[20;0$y");
}

#[test]
fn osc_11_answers_what_the_background_actually_is() {
    // The reply nvim and helix wait for. They ask whether they are sitting on a dark
    // or a light background and choose an entire colour scheme from the answer; a
    // terminal that says nothing does not get a default, it gets a guess.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b]11;?\x07");
    // The default background is #282c34, in the 16-bit-per-channel form xterm uses.
    assert_eq!(s.take_responses(), b"\x1b]11;rgb:2828/2c2c/3434\x07");

    // Foreground and cursor answer on their own numbers.
    feed(&mut s, b"\x1b]10;?\x07");
    assert_eq!(s.take_responses(), b"\x1b]10;rgb:ffff/ffff/ffff\x07");
    feed(&mut s, b"\x1b]12;?\x07");
    assert_eq!(s.take_responses(), b"\x1b]12;rgb:ffff/0000/7878\x07");
}

#[test]
fn a_color_reply_uses_the_terminator_it_was_asked_with() {
    // A client that asked with ST may not be listening for BEL, and vice versa.
    // Mirroring the request is the one answer that is right for both.
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b]11;?\x1b\\"); // asked with ST
    assert_eq!(s.take_responses(), b"\x1b]11;rgb:2828/2c2c/3434\x1b\\");
    feed(&mut s, b"\x1b]11;?\x07"); // asked with BEL
    assert_eq!(s.take_responses(), b"\x1b]11;rgb:2828/2c2c/3434\x07");
}

#[test]
fn osc_10_11_12_set_the_colours_and_the_query_follows() {
    let mut s = Screen::new(10, 2);
    feed(&mut s, b"\x1b]11;#ff0000\x07"); // a red background
    assert_eq!(s.theme().bg, Rgb::new(0xff, 0, 0));
    feed(&mut s, b"\x1b]11;?\x07");
    assert_eq!(s.take_responses(), b"\x1b]11;rgb:ffff/0000/0000\x07");
    // A default cell now paints on the new background: the change is not cosmetic,
    // it is the palette the whole grid resolves against.
    assert_eq!(
        Color::Default.resolve(s.theme(), Ground::Background),
        Rgb::new(0xff, 0, 0)
    );
    // `OSC 111` puts it back.
    feed(&mut s, b"\x1b]111\x07");
    assert_eq!(s.theme().bg, Theme::default().bg);
}

#[test]
fn osc_4_sets_and_queries_any_palette_entry() {
    let mut s = Screen::new(10, 2);
    // A base16 theme script repaints far past the ANSI 16, which is exactly why the
    // palette is stored rather than computed.
    feed(&mut s, b"\x1b]4;1;rgb:aa/bb/cc;17;#010203\x07");
    assert_eq!(s.theme().indexed(1), Rgb::new(0xaa, 0xbb, 0xcc));
    assert_eq!(s.theme().indexed(17), Rgb::new(1, 2, 3));
    // And an SGR colour resolves through it.
    assert_eq!(
        Color::Ansi(1).resolve(s.theme(), Ground::Foreground),
        Rgb::new(0xaa, 0xbb, 0xcc)
    );
    feed(&mut s, b"\x1b]4;1;?\x07");
    assert_eq!(s.take_responses(), b"\x1b]4;1;rgb:aaaa/bbbb/cccc\x07");
    // `OSC 104` puts the named entries back without touching fg/bg/cursor.
    feed(&mut s, b"\x1b]11;#ff0000\x07");
    feed(&mut s, b"\x1b]104;1\x07");
    assert_eq!(s.theme().indexed(1), Theme::default().indexed(1));
    assert_eq!(s.theme().indexed(17), Rgb::new(1, 2, 3), "only 1 was reset");
    assert_eq!(s.theme().bg, Rgb::new(0xff, 0, 0), "and the bg survived");
    // With no index, the whole indexed palette resets — still not fg/bg/cursor.
    feed(&mut s, b"\x1b]104\x07");
    assert_eq!(s.theme().indexed(17), Theme::default().indexed(17));
    assert_eq!(s.theme().bg, Rgb::new(0xff, 0, 0));
}

#[test]
fn a_malformed_colour_is_ignored_not_guessed_at() {
    // The child writes these, so they are hostile input like everything else.
    let mut s = Screen::new(10, 2);
    let before = *s.theme();
    for junk in [
        &b"\x1b]11;lightgoldenrodyellow\x07"[..], // an X11 name: we do not ship rgb.txt
        &b"\x1b]11;rgb:zz/zz/zz\x07"[..],
        &b"\x1b]11;#12345\x07"[..], // not divisible by three
        &b"\x1b]11;\x07"[..],
        &b"\x1b]4;999;#ffffff\x07"[..], // index out of range
        &b"\x1b]4;notanumber;#ffffff\x07"[..],
    ] {
        feed(&mut s, junk);
    }
    assert_eq!(*s.theme(), before, "nothing moved");
    assert!(s.take_responses().is_empty(), "and nothing was answered");
}

#[test]
fn the_kitty_keyboard_stack_pushes_and_pops() {
    let mut s = Screen::new(80, 24);
    // Nothing pushed: legacy.
    assert_eq!(s.kitty_flags(), KittyFlags::NONE);
    // `CSI > 1 u`: the push every application starts with.
    feed(&mut s, b"\x1b[>1u");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
    // A nested program pushes its own, then pops back to its parent's.
    feed(&mut s, b"\x1b[>0u");
    assert_eq!(s.kitty_flags(), KittyFlags::NONE);
    feed(&mut s, b"\x1b[<u");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
    // Popping past the bottom empties the stack rather than underflowing.
    feed(&mut s, b"\x1b[<99u");
    assert_eq!(s.kitty_flags(), KittyFlags::NONE);
}

#[test]
fn the_kitty_query_answers_with_the_flags_really_in_force() {
    let mut s = Screen::new(80, 24);
    feed(&mut s, b"\x1b[?u");
    assert_eq!(s.take_responses(), b"\x1b[?0u");
    feed(&mut s, b"\x1b[>1u\x1b[?u");
    assert_eq!(s.take_responses(), b"\x1b[?1u");
    // An application asking for everything (0b11111) is told exactly what it will
    // get: disambiguate and event types, which we implement, and not alternate keys,
    // all-keys-as-escape-codes or associated text, which we do not. Reporting those as
    // set would leave it waiting for reports that never come.
    feed(&mut s, b"\x1b[>31u\x1b[?u");
    assert_eq!(s.take_responses(), b"\x1b[?3u");
    assert_eq!(
        s.kitty_flags(),
        KittyFlags::DISAMBIGUATE | KittyFlags::REPORT_EVENT_TYPES
    );
}

#[test]
fn the_kitty_set_form_replaces_sets_and_clears() {
    let mut s = Screen::new(80, 24);
    // `CSI = <flags> ; <mode> u`. Mode 1 (the default) replaces.
    feed(&mut s, b"\x1b[=1;1u");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
    // Mode 3 clears the named bits.
    feed(&mut s, b"\x1b[=1;3u");
    assert_eq!(s.kitty_flags(), KittyFlags::NONE);
    // Mode 2 sets them.
    feed(&mut s, b"\x1b[=1;2u");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
    // An unknown mode changes nothing.
    feed(&mut s, b"\x1b[=1;9u");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
}

#[test]
fn a_kitty_push_loop_cannot_grow_the_stack_without_bound() {
    // The child controls the depth, so it is hostile input like any other. The cap
    // holds and the most recent intent (the top) survives.
    let mut s = Screen::new(80, 24);
    for _ in 0..(KITTY_STACK_LIMIT * 4) {
        feed(&mut s, b"\x1b[>0u");
    }
    feed(&mut s, b"\x1b[>1u");
    assert_eq!(s.kitty_stack.len(), KITTY_STACK_LIMIT);
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
}

#[test]
fn a_bare_csi_u_still_restores_the_cursor() {
    // SCORC and the kitty protocol share the final byte `u`, and only the private
    // marker tells them apart. A parser that got this wrong would break either the
    // cursor save/restore or the keyboard protocol.
    let mut s = Screen::new(80, 24);
    feed(&mut s, b"\x1b[5;3H\x1b[s"); // save at row 5, col 3
    feed(&mut s, b"\x1b[1;1H\x1b[u"); // home, then restore
    assert_eq!(s.cursor(), (4, 2));
    assert_eq!(s.kitty_flags(), KittyFlags::NONE, "SCORC pushed no flags");
    assert!(s.take_responses().is_empty(), "and answered no query");
}

#[test]
fn xtmodkeys_sets_and_resets_the_modify_other_keys_level() {
    let mut s = Screen::new(80, 24);
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Off);
    feed(&mut s, b"\x1b[>4;2m");
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Level2);
    feed(&mut s, b"\x1b[>4;1m");
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Level1);
    // No level given resets, which is how a program turns it back off on exit.
    feed(&mut s, b"\x1b[>4m");
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Off);
    // A resource other than 4 (modifyCursorKeys, say) is not ours to act on.
    feed(&mut s, b"\x1b[>4;2m\x1b[>1;2m");
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Level2);
}

#[test]
fn a_reset_takes_the_keyboard_protocols_with_it() {
    // RIS is the escape hatch for a program that died mid-mode. If the flags
    // survived it, the shell underneath would keep receiving CSI u for its keys.
    let mut s = Screen::new(80, 24);
    feed(&mut s, b"\x1b[>1u\x1b[>4;2m");
    assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
    feed(&mut s, b"\x1bc"); // RIS
    assert_eq!(s.kitty_flags(), KittyFlags::NONE);
    assert_eq!(s.modify_other_keys(), ModifyOtherKeys::Off);
}

#[test]
fn cursor_position_report_is_one_based() {
    let mut s = Screen::new(80, 24);
    feed(&mut s, b"\x1b[3;10H"); // move to row 3, col 10 (1-based)
    feed(&mut s, b"\x1b[6n"); // request CPR
    assert_eq!(s.take_responses(), b"\x1b[3;10R");
    // The extended (?6n) form carries a trailing page number.
    feed(&mut s, b"\x1b[?6n");
    assert_eq!(s.take_responses(), b"\x1b[?3;10;1R");
}

#[test]
fn decscusr_sets_the_cursor_shape_and_blink() {
    let mut s = Screen::new(10, 2);
    // Power-on default is a steady block (see `CursorAppearance`).
    assert_eq!(s.cursor_style(), CursorStyle::Block);
    assert!(!s.cursor_blinks());
    feed(&mut s, b"\x1b[4 q"); // steady underline
    assert_eq!(s.cursor_style(), CursorStyle::Underline);
    assert!(!s.cursor_blinks());
    feed(&mut s, b"\x1b[5 q"); // blinking bar
    assert_eq!(s.cursor_style(), CursorStyle::Bar);
    assert!(s.cursor_blinks());
    feed(&mut s, b"\x1b[1 q"); // blinking block, asked for explicitly
    assert_eq!(s.cursor_style(), CursorStyle::Block);
    assert!(s.cursor_blinks());
    feed(&mut s, b"\x1b[7 q"); // out of range: ignored, the last style stands
    assert_eq!(s.cursor_style(), CursorStyle::Block);
    assert!(s.cursor_blinks());
}

/// `DECSCUSR 0` means "the terminal's default", not xterm's literal blinking
/// block. TUIs (codex, anything on crossterm's `SetCursorStyle::DefaultUserShape`)
/// emit it while restoring the terminal on exit, so reading it as a blink request
/// leaves a blinking cursor behind in the shell long after the program is gone.
#[test]
fn decscusr_zero_restores_the_power_on_cursor() {
    let mut s = Screen::new(10, 2);
    for style in [&b"\x1b[5 q"[..], b"\x1b[1 q", b"\x1b[3 q"] {
        feed(&mut s, style);
        assert!(s.cursor_blinks(), "the program asked for a blink");
        feed(&mut s, b"\x1b[0 q");
        assert_eq!(s.cursor_style(), CursorStyle::Block);
        assert!(!s.cursor_blinks(), "DECSCUSR 0 restores the steady default");
    }
}

#[test]
fn mouse_modes_track_the_dec_private_toggles() {
    use crate::mouse::MouseProtocol;
    let mut s = Screen::new(10, 4);
    assert!(!s.mouse_mode().reports(), "off by default");
    feed(&mut s, b"\x1b[?1000h");
    assert_eq!(s.mouse_mode().protocol, MouseProtocol::Press);
    feed(&mut s, b"\x1b[?1002h"); // a higher level supersedes
    assert_eq!(s.mouse_mode().protocol, MouseProtocol::ButtonEvent);
    feed(&mut s, b"\x1b[?1006h"); // SGR is an orthogonal encoding flag
    assert!(s.mouse_mode().sgr);
    assert_eq!(s.mouse_mode().protocol, MouseProtocol::ButtonEvent);
    feed(&mut s, b"\x1b[?1002l"); // disabling turns reporting off
    assert!(!s.mouse_mode().reports());
    assert!(s.mouse_mode().sgr, "the encoding flag is independent");
}

#[test]
fn narrowing_blanks_a_dangling_wide_leader() {
    // A wide glyph whose spacer is cut off must not leave a half-cell behind.
    let mut s = Screen::new(4, 1);
    feed(&mut s, "aa\u{6f22}".as_bytes()); // 'a','a', then 漢 as leader+spacer at cols 2,3
    assert!(s.cell(0, 2).is_wide_leader());
    s.resize(3, 1); // drops the spacer at col 3, leaving the leader at col 2
    assert!(
        !s.cell(0, 2).is_wide_leader(),
        "the orphaned leader was blanked"
    );
    assert_eq!(s.cell(0, 2).rune, ' ');
}
