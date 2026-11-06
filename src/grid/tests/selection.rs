//! Reading text back out: a selection's text, and the word and logical line under a
//! cell.

use crate::grid::tests::{at, selection_text};
use crate::grid::*;
use crate::testutil::feed;

#[test]
fn selection_text_extracts_and_joins_rows() {
    let mut s = Screen::new(10, 3);
    feed(&mut s, b"hello\r\nworld\r\nfoo");
    // A single-row partial selection: cols 0..=4 of row 0.
    assert_eq!(selection_text(&s, at(&s, 0, 0), at(&s, 0, 4)), "hello");
    // Across rows: end of row 0 through part of row 1, joined by a newline;
    // trailing blanks on the first row are trimmed.
    assert_eq!(selection_text(&s, at(&s, 0, 0), at(&s, 1, 2)), "hello\nwor");
    // A mid-row start.
    assert_eq!(selection_text(&s, at(&s, 0, 2), at(&s, 0, 4)), "llo");
    // Either order: the endpoints sort into reading order.
    assert_eq!(selection_text(&s, at(&s, 1, 2), at(&s, 0, 0)), "hello\nwor");
}

#[test]
fn selection_joins_a_soft_wrapped_line() {
    // A line longer than the width soft-wraps; selecting across the wrap must
    // copy it as one unbroken logical line (no newline at the wrap point).
    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcdef"); // wraps: "abcd" | "ef"
    assert!(s.row_wraps(0));
    assert_eq!(selection_text(&s, at(&s, 0, 0), at(&s, 1, 1)), "abcdef");
}

#[test]
fn a_space_at_the_wrap_point_survives_a_copy() {
    // The space lands in the last column and the next word wraps: "ab " | "cd". That
    // space is text, so the copy must not glue the two words together.
    let mut s = Screen::new(3, 3);
    feed(&mut s, b"ab cd");
    assert!(s.row_wraps(0));
    assert_eq!(selection_text(&s, at(&s, 0, 0), at(&s, 1, 1)), "ab cd");
}

#[test]
fn wide_glyph_wrap_padding_is_not_copied() {
    // "aかb" at two columns: か cannot fit beside 'a', so it wraps and leaves the last
    // column blank. That blank is padding, not a space in the text.
    let mut s = Screen::new(2, 4);
    feed(&mut s, "aかb".as_bytes());
    assert!(s.row_wraps(0));
    assert_eq!(selection_text(&s, at(&s, 0, 0), at(&s, 2, 0)), "aかb");
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
    assert_eq!(selection_text(&s, at(&s, 0, 0), at(&s, 0, 5)), "abcdef");

    let mut s = Screen::new(4, 3);
    feed(&mut s, b"abcdef");
    s.resize(3, 3); // narrow: re-wraps to "abc" | "def"
    assert!(s.row_wraps(0));
    assert_eq!(selection_text(&s, at(&s, 0, 0), at(&s, 1, 2)), "abcdef");
}

#[test]
fn selection_reads_scrollback_when_scrolled() {
    let mut s = Screen::new(6, 2);
    feed(&mut s, b"one\r\ntwo\r\nthree\r\nfour");
    s.scroll_view_up(2); // top: "one" over "two"
    assert_eq!(selection_text(&s, at(&s, 0, 0), at(&s, 0, 2)), "one");
}

#[test]
fn a_selection_copies_its_rows_after_output_scrolls_them_away() {
    // The absolute-row payoff: the same ids, resolved after the grid has scrolled
    // under them, still name the lines the user picked — including once they have
    // scrolled out of sight entirely.
    let mut s = Screen::new(6, 2);
    feed(&mut s, b"one\r\ntwo");
    let (start, end) = (at(&s, 0, 0), at(&s, 0, 2)); // "one", on the live screen
    assert_eq!(selection_text(&s, start, end), "one");

    feed(&mut s, b"\r\nthree\r\nfour\r\nfive"); // "one" is pushed deep into history

    assert_eq!(s.display_row(start.0), None, "it is off screen now");
    assert_eq!(
        selection_text(&s, start, end),
        "one",
        "but the ids still name the line it was made over"
    );
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
    assert_eq!(selection_text(&s, path.0, path.1), "/usr/bin");
    // A parenthesis bounds the token, and clicking the space between words
    // selects just that cell.
    assert_eq!(
        s.word_at(abs, 11),
        ((abs, 11), (abs, 11)),
        "the space is its own cell"
    );
    let ok = s.word_at(abs, 13);
    assert_eq!(selection_text(&s, ok.0, ok.1), "ok");
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
    assert_eq!(selection_text(&s, line.0, line.1), "abcdef");
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
        selection_text(&s, line.0, line.1),
        "abcdefgh",
        "the line was taken whole, across the top of the screen"
    );
}
