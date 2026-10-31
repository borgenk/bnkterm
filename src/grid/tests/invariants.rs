//! The grid's structural invariants under arbitrary input and violent resizes.

use crate::dev::fuzz::Rng;
use crate::grid::tests::wide_pair_break;
use crate::grid::*;

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

    let mut rng = Rng::new(0x9e3779b97f4a7c15);
    for _ in 0..4000 {
        let cols = 1 + (rng.next() % 220) as usize;
        let rows = 1 + (rng.next() % 70) as usize;
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
        match rng.next() % 4 {
            0 => p.advance_bytes(&mut s, b"\r\x1b[J\x1b]133;A\x07user@host ~/p  main\r\n> "),
            1 => p.advance_bytes(&mut s, "wide 世世世 and text\r\n".as_bytes()),
            _ => {}
        }
    }
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
    let bytes = crate::dev::fuzz::Stream::new(0xDEAD_BEEF_CAFE_1234).bytes(300 * 4096);

    let Some(why) = invariant_break(&bytes, cols, rows) else {
        return;
    };
    let minimal = crate::dev::fuzz::shrink(&bytes, |b| invariant_break(b, cols, rows).is_some());
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
        crate::dev::fuzz::as_byte_literal(&minimal),
    );
}
