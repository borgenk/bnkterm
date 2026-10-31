//! Resize and reflow: rewrapping on a width change, what a height change keeps, and what
//! rides along with the text (the cursor, hyperlinks, prompt marks).

use crate::grid::buffer::MAX_LOGICAL_COLS;
use crate::grid::tests::anchor;
use crate::grid::*;
use crate::testutil::feed;

#[test]
fn the_saved_cursor_follows_its_row_through_a_resize() {
    // A shell saves its cursor at the prompt (DECSC) and restores it when SIGWINCH says
    // the window changed (DECRC), so the saved position has to move with the row it was
    // saved on. The width path (reflow) carried it; the height-only path did not, so a
    // restore after a height change landed on whatever row now sat at the old index.
    let restored_row = |s: &mut Screen| {
        feed(s, b"\x1b8");
        let (row, _) = s.cursor();
        s.row_string(row).trim_end().to_string()
    };

    // Taller: history comes back down onto the screen, pushing every row down.
    let mut s = Screen::new(10, 4);
    feed(&mut s, b"a\r\nb\r\nc\r\nd\r\ne\r\nf\x1b7");
    s.resize(10, 6);
    assert_eq!(restored_row(&mut s), "f", "after growing");

    // Shorter: the top scrolls into history, pulling every row up.
    let mut s = Screen::new(10, 6);
    feed(&mut s, b"a\r\nb\r\nc\r\nd\x1b7\r\ne\r\nf");
    s.resize(10, 4);
    assert_eq!(restored_row(&mut s), "d", "after shrinking");

    // Wider: a wrapped line joins, pulling the rows after it up.
    let mut s = Screen::new(5, 4);
    feed(&mut s, b"aaaaabbbbb\r\nc\x1b7\r\nd");
    s.resize(10, 4);
    assert_eq!(restored_row(&mut s), "c", "after a width reflow");
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
fn reflow_keeps_a_trailing_blank_inside_an_anchor_open_or_refused() {
    // A space printed inside an anchor is part of the link, so like a coloured blank it is
    // content, not padding: narrowed, it wraps onto the next row rather than being trimmed.
    for uri in ["https://e.com/a", "javascript:x"] {
        let mut s = Screen::new(10, 3);
        feed(&mut s, &anchor(uri, "ab  "));
        s.resize(3, 3);
        assert!(s.row_wraps(0), "{uri}");
        assert_eq!(s.cell(1, 0).link, s.cell(0, 0).link, "{uri}");
    }
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

// The reflow hands a row straight over when its text already fits the new width, on the
// grounds that laying it out again would land it exactly where it is. These pin that the
// shortcut and the full unwrap/rewrap agree on where a row's text ends — the one thing
// that decides which of the two a row takes.

#[test]
fn a_narrow_into_padding_alone_keeps_the_row_whole() {
    let mut s = Screen::new(20, 3);
    feed(&mut s, b"hello");
    // The cut lands in the trailing blanks, so the text is untouched and unwrapped.
    s.resize(8, 3);
    assert_eq!(s.row_string(0).trim_end(), "hello");
    assert_eq!(s.row_string(1).trim_end(), "");
    assert_eq!(s.scrollback_len(), 0);
}

#[test]
fn a_narrow_that_reaches_the_text_still_rewraps() {
    let mut s = Screen::new(20, 3);
    feed(&mut s, b"hello world");
    // One column short of the text's end is a real rewrap, not a padding trim.
    s.resize(10, 3);
    assert_eq!(s.row_string(0).trim_end(), "hello worl");
    assert_eq!(s.row_string(1).trim_end(), "d");
}

#[test]
fn a_coloured_blank_past_the_new_width_counts_as_text() {
    let mut s = Screen::new(12, 3);
    // Two red spaces at columns 8-9: blank runes, but a coloured cell is content.
    feed(&mut s, b"abcdefgh\x1b[41m  \x1b[0m");
    s.resize(9, 3);
    // Narrowing past them must wrap them onto the next row rather than trim them away.
    assert_eq!(s.row_string(0).trim_end(), "abcdefgh");
    assert_eq!(
        s.cell(1, 0).bg,
        Color::Ansi(1),
        "the coloured blank wrapped instead of being dropped as padding"
    );
}

#[test]
fn a_mark_on_a_blank_past_the_new_width_counts_as_text() {
    let mut s = Screen::new(12, 3);
    // A combining acute on the space at column 5, so that blank carries content.
    feed(&mut s, "abcde \u{0301}".as_bytes());
    s.resize(5, 3);
    assert_eq!(s.row_string(0).trim_end(), "abcde");
    assert_eq!(
        s.marks_at(1, 0).collect::<Vec<_>>(),
        ['\u{0301}'],
        "the marked blank wrapped, mark and all"
    );
}

#[test]
fn a_row_whose_text_ends_exactly_at_the_new_width_stays_one_row() {
    let mut s = Screen::new(12, 3);
    feed(&mut s, b"abcdefgh");
    // Eight columns of text into eight columns: the last cell fits, nothing wraps.
    s.resize(8, 3);
    assert_eq!(s.row_string(0).trim_end(), "abcdefgh");
    assert_eq!(s.row_string(1).trim_end(), "");
}

#[test]
fn a_narrow_that_would_split_a_wide_glyph_pushes_the_pair_down() {
    let mut s = Screen::new(12, 3);
    feed(&mut s, "abc\u{4e00}".as_bytes()); // wide glyph in columns 3-4
    s.resize(4, 3);
    // Column 3 is the last, and the pair cannot straddle the edge, so it goes down whole.
    assert_eq!(s.row_string(0).trim_end(), "abc");
    assert_eq!(s.cell(1, 0).rune, '\u{4e00}');
}

#[test]
fn a_widen_leaves_a_short_row_exactly_where_it_was() {
    let mut s = Screen::new(8, 3);
    feed(&mut s, b"one\r\ntwo\r\nthree");
    s.resize(40, 3);
    // Nothing wrapped at eight columns, so widening rejoins nothing and moves nothing.
    assert_eq!(s.row_string(0).trim_end(), "one");
    assert_eq!(s.row_string(1).trim_end(), "two");
    assert_eq!(s.row_string(2).trim_end(), "three");
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
