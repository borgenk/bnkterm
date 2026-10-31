//! Editing: cursor motion, erase, insert and delete, scrolling and scroll regions, tab
//! stops, the bell, and the pen that SGR sets.

use crate::grid::tests::print_str;
use crate::grid::*;
use crate::testutil::feed;

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
fn only_a_scroll_paints_the_pen_background_into_a_row() {
    let row_bg =
        |s: &Screen, row: usize| -> Vec<Color> { (0..4).map(|c| s.cell(row, c).bg).collect() };
    let mut s = Screen::new(4, 3);
    feed(&mut s, b"\x1b[44m\n");
    assert_eq!(
        row_bg(&s, 1),
        [Color::Default; 4],
        "LF onto a row leaves its colour"
    );
    feed(&mut s, b"\n\n");
    assert_eq!(
        row_bg(&s, 2),
        [Color::Ansi(4); 4],
        "LF at the bottom fills the new row"
    );
    feed(&mut s, b"\x1b[41m\x1b[2;1H\x1bM");
    assert_eq!(
        row_bg(&s, 0),
        [Color::Default; 4],
        "RI onto a row leaves its colour"
    );
    feed(&mut s, b"\x1bM");
    assert_eq!(
        row_bg(&s, 0),
        [Color::Ansi(1); 4],
        "RI at the top fills the new row"
    );
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
    // The design decision, pinned. The underline shape rides in the spare bits of the
    // attribute bitfield rather than in a field of its own, so squiggling one
    // diagnostic costs the grid nothing at all. Now that the rendition is interned it
    // costs even less than it used to — a shape is a property of a `Style` that a
    // thousand cells can share — but `Attrs` still has to fit in its two bytes, because
    // widening it widens every interned rendition.
    assert_eq!(std::mem::size_of::<Attrs>(), 2);
    assert_eq!(std::mem::size_of::<PackedCell>(), 8);

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
fn ed_3_erases_the_scrollback_and_nothing_else() {
    // `CSI 3 J` is xterm's Erase Saved Lines. It is why `clear(1)` sends both `2J` and
    // `3J`: the pair is complementary, not redundant, so a `3` that also wiped the screen
    // would make the `2` pointless. kitty, alacritty and ghostty all agree.
    let mut s = Screen::new(6, 3);
    feed(&mut s, b"old1\r\nold2\r\nkeep1\r\nkeep2\r\nkeep3");
    assert!(s.scrollback_len() > 0, "there is history to erase");
    let before: Vec<String> = (0..3).map(|r| s.row_string(r)).collect();
    let cursor = s.cursor();

    feed(&mut s, b"\x1b[3J");

    assert_eq!(s.scrollback_len(), 0, "the history is gone");
    let after: Vec<String> = (0..3).map(|r| s.row_string(r)).collect();
    assert_eq!(after, before, "and the display is byte-identical");
    assert_eq!(s.cursor(), cursor, "the cursor did not move either");
}

#[test]
fn an_unknown_erase_parameter_erases_nothing() {
    // xterm's ED and EL switches hit `default:` and do nothing. Falling back to mode 0
    // turns a sequence the program never meant into an erase, and the wrong guess is
    // destructive: too little erased shows stale text until the next repaint, too much
    // has already destroyed it.
    for seq in [b"\x1b[7J".as_slice(), b"\x1b[9J", b"\x1b[4K", b"\x1b[99K"] {
        let mut s = Screen::new(6, 3);
        feed(&mut s, b"abcdef\r\nghijkl\r\nmnopqr\x1b[2;3H");
        let before: Vec<String> = (0..3).map(|r| s.row_string(r)).collect();
        let cursor = s.cursor();
        feed(&mut s, seq);
        let after: Vec<String> = (0..3).map(|r| s.row_string(r)).collect();
        assert_eq!(
            after,
            before,
            "{:?} erased something",
            String::from_utf8_lossy(seq)
        );
        assert_eq!(s.cursor(), cursor);
    }
}

#[test]
fn sgr_21_draws_a_double_underline_and_leaves_bold_alone() {
    // ECMA-48 and xterm ctlseqs: "Ps = 21 -> Doubly-underlined". Turning bold *off* is
    // 22. The grid could already carry and report the style, so this arm was the only
    // thing between a program asking for a double underline and getting one.
    let mut s = Screen::new(4, 1);
    feed(&mut s, b"\x1b[1;21mX");
    let cell = s.cell(0, 0);
    assert!(cell.attrs.contains(Attrs::BOLD), "21 is not 'bold off'");
    assert!(cell.attrs.contains(Attrs::UNDERLINE));
    assert_eq!(cell.attrs.underline_style(), UnderlineStyle::Double);

    // And 22 still is bold off, which is the arm 21 was doing the job of.
    feed(&mut s, b"\x1b[22mY");
    assert!(!s.cell(0, 1).attrs.contains(Attrs::BOLD));

    // The style survives a DECRQSS round trip, since that is how a program restores it.
    feed(&mut s, b"\x1b[0m\x1b[21m\x1bP$qm\x1b\\");
    let reply = String::from_utf8_lossy(s.responses()).into_owned();
    assert!(
        reply.contains("4:2"),
        "DECRQSS reports the shape: {reply:?}"
    );
}

#[test]
fn an_invalid_scroll_region_is_refused_whole() {
    // xterm's `CASE_DECSTBM` acts only `if (bottom > top)`, so a bad request leaves the
    // margins *and the cursor* exactly as they were; alacritty and ghostty agree.
    // Resetting to the full screen instead hands hostile or corrupted input a way to
    // silently drop a program's scroll region and move its cursor.
    for bad in [b"\x1b[5;5r".as_slice(), b"\x1b[6;3r", b"\x1b[9;9r"] {
        let mut s = Screen::new(10, 8);
        feed(&mut s, b"\x1b[2;6r\x1b[3;7H"); // a real region, cursor parked inside it
        let region = (s.active().scroll_top, s.active().scroll_bottom);
        let cursor = s.cursor();
        feed(&mut s, bad);
        assert_eq!(
            (s.active().scroll_top, s.active().scroll_bottom),
            region,
            "{:?} dropped the region",
            String::from_utf8_lossy(bad)
        );
        assert_eq!(
            s.cursor(),
            cursor,
            "{:?} moved the cursor",
            String::from_utf8_lossy(bad)
        );
    }

    // A valid one still applies, and still homes the cursor.
    let mut s = Screen::new(10, 8);
    feed(&mut s, b"\x1b[4;4H\x1b[2;6r");
    assert_eq!((s.active().scroll_top, s.active().scroll_bottom), (1, 5));
    assert_eq!(s.cursor(), (0, 0), "a valid DECSTBM homes the cursor");
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
