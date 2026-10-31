//! The scrollback view and row identity: where the view rests as output arrives, the
//! scrollbar's extent, and when row ids keep or lose their meaning.

use crate::grid::tests::print_str;
use crate::grid::*;
use crate::testutil::feed;

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
