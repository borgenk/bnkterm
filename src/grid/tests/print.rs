//! Printing: characters into cells, the deferred wrap, wide and combining characters,
//! grapheme clustering, the character sets, and the bulk print paths agreeing with the
//! per-character one.

use crate::dev::fuzz::Rng;
use crate::grid::buffer::MAX_COMBINING_PER_CELL;
use crate::grid::tests::{at, print_str, selection_text, wide_pair_break};
use crate::grid::*;
use crate::testutil::feed;

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
fn a_cell_keeps_its_cap_of_marks_when_two_cells_take_them_in_turn() {
    // Zalgo piles thousands of marks on a cell, and each cell keeps only its cap. The
    // cap is per cell, so marks arriving for two cells in turn fill both to it.
    let mut s = Screen::new(10, 3);
    feed(&mut s, b"ab");
    for _ in 0..MAX_COMBINING_PER_CELL + 8 {
        feed(&mut s, "\u{0301}\x1b[D\u{0302}\x1b[C".as_bytes());
    }
    assert_eq!(s.marks_at(0, 0).count(), MAX_COMBINING_PER_CELL);
    assert_eq!(s.marks_at(0, 1).count(), MAX_COMBINING_PER_CELL);
    assert!(s.marks_at(0, 0).all(|m| m == '\u{0302}'));
    assert!(s.marks_at(0, 1).all(|m| m == '\u{0301}'));
}

#[test]
fn a_cell_written_over_at_its_cap_takes_marks_again() {
    // Overwriting a cell drops its marks, so the new character starts from none.
    let mut s = Screen::new(10, 3);
    let zalgo = "\u{0301}".repeat(MAX_COMBINING_PER_CELL + 8);
    feed(&mut s, format!("a{zalgo}").as_bytes());
    assert_eq!(s.marks_at(0, 0).count(), MAX_COMBINING_PER_CELL);
    feed(&mut s, "\x1b[Db\u{0302}".as_bytes());
    assert_eq!(s.marks_at(0, 0).collect::<Vec<_>>(), ['\u{0302}']);
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

/// Screen with grapheme clustering (`?2027`) turned on, as a program that asked for it
/// would have it.
fn clustering(cols: usize, rows: usize) -> Screen {
    let mut s = Screen::new(cols, rows);
    feed(&mut s, b"\x1b[?2027h");
    s
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
fn a_cluster_growing_wide_does_not_orphan_the_pair_it_grows_into() {
    // Growing is a *write* into the column beside the cluster, and that column may be
    // occupied. Here it holds the leader of a wide pair, so laying the spacer over it
    // would strand the old pair's own spacer one column further right — an orphan, which
    // draws as garbage and is precisely what `wide_pair_break` calls corruption.
    let mut s = clustering(10, 1);
    feed(&mut s, b"\x1b[2G"); // 世 at columns 1-2...
    print_str(&mut s, "\u{4E16}");
    feed(&mut s, b"\x1b[1G"); // ...and back to column 0
    print_str(&mut s, "\u{2600}\u{FE0F}"); // ☀ then the selector: the cell grows into 1

    assert!(s.cell(0, 0).is_wide_leader());
    assert!(s.cell(0, 1).is_wide_spacer());
    assert_eq!(
        wide_pair_break(&s.primary.lines[0].cells),
        None,
        "the pair the cluster grew into must be broken cleanly, not half-overwritten"
    );
    assert_eq!(s.cell(0, 2).rune, ' ', "世's stranded half is blanked");
    assert!(s.marks_at(0, 1).next().is_none(), "and its marks are gone");
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
fn wide_and_combining_via_parser() {
    let mut s = Screen::new(10, 1);
    feed(&mut s, "aか\u{3099}b".as_bytes()); // 'a', wide か + combining mark, 'b'
    assert_eq!(s.cell(0, 0).rune, 'a');
    assert_eq!(s.cell(0, 1).rune, 'か');
    assert!(s.cell(0, 1).is_wide_leader());
    assert_eq!(s.marks_at(0, 1).collect::<Vec<_>>(), ['\u{3099}']);
    assert_eq!(s.cell(0, 3).rune, 'b');
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

/// Drive `bytes` through both the bulk and the per-char path and require the two screens
/// to stay identical, block by block. A small grid so runs cross rows often.
fn assert_bulk_equiv_over(bytes: &[u8], label: &str) {
    let mut bulk = Screen::new(8, 4);
    let mut per = Screen::new(8, 4);
    let mut pb = crate::vt::Parser::new();
    let mut pc = crate::vt::Parser::new();
    for (block, chunk) in bytes.chunks(4096).enumerate() {
        pb.advance_bytes(&mut bulk, chunk);
        for &byte in chunk {
            pc.advance(&mut per, byte);
        }
        assert_eq!(
            bulk.snapshot(),
            per.snapshot(),
            "{label} block {block} snapshot"
        );
        assert_eq!(bulk.cursor(), per.cursor(), "{label} block {block} cursor");
    }
}

#[test]
fn breaking_a_wide_glyph_in_the_final_column_ends_the_soft_wrap_on_every_print_path() {
    // A soft-wrapped row whose last two cells hold a wide glyph. A print over the glyph's
    // leader blanks its spacer, which is the final cell, so the line stops there as it
    // does for any write to that cell. The bulk ASCII run, the bulk text run and the
    // per-character path must agree; the fuzz corpus never places a print there.
    let setup = "abcdef\u{4e00}g".as_bytes();
    for (tail, row) in [
        ("\x1b[1;7Hx", "abcdefx "),
        ("\x1b[1;6Hxy", "abcdexy "),
        ("\x1b[1;7H\u{e9}", "abcdef\u{e9} "),
    ] {
        let mut bulk = Screen::new(8, 4);
        let mut per = Screen::new(8, 4);
        let (mut pb, mut pc) = (crate::vt::Parser::new(), crate::vt::Parser::new());
        pb.advance_bytes(&mut bulk, setup);
        assert!(bulk.row_wraps(0), "the setup soft-wraps the glyph's row");
        pb.advance_bytes(&mut bulk, tail.as_bytes());
        for &byte in setup.iter().chain(tail.as_bytes()) {
            pc.advance(&mut per, byte);
        }
        for (path, s) in [("bulk", &bulk), ("per char", &per)] {
            assert_eq!(s.row_string(0), row, "{path} {tail:?}");
            assert!(
                !s.row_wraps(0),
                "{path} {tail:?}: the wrap ends at the blanked cell"
            );
        }
    }
}

#[test]
fn bulk_ascii_matches_per_char_under_fuzz() {
    // Two corpora, because they find different things and neither subsumes the other.
    //
    // The uniform one is dense with long ASCII runs salted with full-range bytes, which
    // is what puts run boundaries and SWAR word edges in awkward places. It is also
    // nearly blind to the escape machinery: it reaches insert mode about once per 1.2 MB
    // and `?2027` never, and those are exactly the modes that take the printer *off* its
    // bulk path — so on its own it was comparing two fast paths to each other.
    let mut uniform = Vec::with_capacity(300 * 4096);
    let mut rng = Rng::new(0x0BAD_C0DE_1234_5678);
    for _ in 0..300 * 4096 {
        let r = rng.next() as u32;
        // ~1 in 6 bytes is full-range (keeps escapes/controls/UTF-8 in the
        // mix); the rest are printable ASCII, so runs are long enough to wrap.
        uniform.push(if r.is_multiple_of(6) {
            (r >> 8) as u8
        } else {
            0x20 + ((r >> 8) % 0x5f) as u8
        });
    }
    assert_bulk_equiv_over(&uniform, "uniform");

    // The structured one sets the modes, walks the DCS states, and interrupts sequences
    // mid-flight, so the comparison actually spans the slow paths.
    let structured = crate::dev::fuzz::Stream::new(0x5EED_1234_ABCD_0001).bytes(300 * 4096);
    assert_bulk_equiv_over(&structured, "structured");
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
    assert_eq!(selection_text(&s, at(&s, 1, 0), at(&s, 2, 1)), "xy\nzw");
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
    assert_eq!(selection_text(&s, at(&s, 0, 0), at(&s, 1, 1)), "abcz\nef");
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
    assert_eq!(selection_text(&s, at(&s, 0, 0), at(&s, 1, 1)), "abcd\nef");
}
