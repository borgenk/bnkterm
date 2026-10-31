//! The grid's tests, filed by the behaviour they pin (see the `mod tests` declaration in
//! `grid/mod.rs`). This file holds the helpers more than one of them uses.

mod cells;
mod edit;
mod invariants;
mod links;
mod modes;
mod osc;
mod print;
mod reflow;
mod reply;
mod selection;
mod view;

use crate::grid::*;

pub(super) fn print_str(s: &mut Screen, text: &str) {
    for c in text.chars() {
        s.print(c);
    }
}

/// Where a row's wide-glyph pairing is broken, or `None`. A leader must be followed by
/// a spacer, and a spacer must be preceded by a leader; either half standing alone is
/// an orphan that draws as garbage.
pub(super) fn wide_pair_break(row: &[PackedCell]) -> Option<String> {
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

/// `ESC ] 8 ; ; <uri> ESC \` — an anchor around `label`, closed after it.
pub(super) fn anchor(uri: &str, label: &str) -> Vec<u8> {
    format!("\x1b]8;;{uri}\x1b\\{label}\x1b]8;;\x1b\\").into_bytes()
}

/// The absolute cell under display `(row, col)`, which is what a pointer resolves to.
pub(super) fn at(s: &Screen, row: usize, col: usize) -> (AbsRow, usize) {
    (s.abs_row(row), col)
}

/// The selected text as a value. The terminal appends into a buffer it reuses across a
/// drag; a test wants the string on its own to compare against.
pub(super) fn selection_text(s: &Screen, a: (AbsRow, usize), b: (AbsRow, usize)) -> String {
    let mut out = String::new();
    s.selection_text_into(a, b, &mut out);
    out
}
