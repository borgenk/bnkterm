//! The screen model: the cell grid, cursor, scrollback, and the operations the VT parser
//! drives. This is the type `bytes -> vt::Parser -> grid::Screen` feeds, and where
//! terminal *correctness* lives. The parser holds no grid state and the grid holds no
//! parser state: this module knows only semantic operations (print a rune, move the
//! cursor, erase, scroll), never bytes.
//!
//! This file is the vocabulary — every type the grid is built from, and [`Screen`] itself.
//! The operations are split across sibling modules by what they do, all of them flat:
//!
//! ```text
//!   the parser's side              mod.rs      the types and Screen's fields
//!   ─────────────────              buffer.rs   the ring: Row, Buffer, scroll, reflow
//!   vt::Parser
//!       │
//!       ▼                          what a sequence turns into:
//!   perform.rs ──────────────┬───▶ print.rs    characters into cells
//!   (syntax -> meaning)      ├───▶ edit.rs     motion, erase, scroll, modes, SGR, resize
//!                            ├───▶ osc.rs      title, palette, links, cwd, prompt marks
//!                            └───▶ reply.rs    what we answer (DA, DSR, DECRQM, DECRQSS)
//!
//!   the app's side ─────────────▶ view.rs      reading the grid back out: display vs
//!                                              stream coordinates, selection, hyperlinks
//! ```
//!
//! `view.rs` is the one that answers *outward* — the painter and the app ask it questions
//! and it changes nothing — which is why it is a peer of the parser-driven four rather
//! than one of them.
//!
//! Two cell shapes corrupt a screen when handled wrong, so both are stated once here and
//! respected everywhere:
//!
//!   * Wide characters (CJK, most emoji) occupy two columns. The left column carries the
//!     rune and the [`Attrs::WIDE_LEADER`] attr; the right column is a
//!     [`Attrs::WIDE_SPACER`] placeholder the cursor steps over and the renderer skips.
//!     Overwriting either half cleans up its orphaned partner.
//!
//!   * Combining marks (an accent after its base, a Hangul jamo stack) are rare, so a
//!     [`Cell`] stores only the base rune inline and extra marks overflow into a small
//!     per-`Row` side list keyed by column. It travels with the row when it scrolls, and
//!     the common all-single-codepoint row pays nothing.

mod buffer;
mod edit;
mod osc;
mod perform;
mod print;
mod reply;
mod view;

/// The grid's suite. One module rather than one per sibling above, because these tests are
/// overwhelmingly *end to end* — feed bytes, assert the resulting screen — so a single test
/// routinely exercises `perform`, `edit`, `print` and `view` at once and belongs to no one
/// of them. Splitting them along the implementation's seams would file each test under
/// whichever module it happens to touch first.
#[cfg(test)]
mod tests;

// The sibling modules reach these through `use super::*`, so this is the one list of what
// the whole grid speaks — including the two `buffer` types `Screen`'s own fields name.
use buffer::{Buffer, Pen};

use crate::color::{self, Color, Theme};
use crate::input::{KittyFlags, ModifyOtherKeys};
use crate::mouse::{MouseMode, MouseProtocol};
use crate::platform::grapheme;
use crate::vt::{Params, Perform};
use crate::width::width;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::fmt;

/// Default lines of scrollback the primary screen keeps. Overridable by config
/// later (phase 4); the alternate screen keeps none.
pub(crate) const DEFAULT_SCROLLBACK: usize = 10_000;

/// The index of a line in the unbounded stream of everything the child has printed:
/// history first (oldest at 0), then the live screen. Unlike a display row it does not
/// move when the screen scrolls — output pushes a row up out of the live screen and
/// into history, and the id rides along with the content.
///
/// That is the whole point of it. A selection stored in display coordinates has to be
/// thrown away on every scroll (display row 4 is a different line afterwards), which is
/// why child output used to drop it; a selection stored in `AbsRow` survives, because
/// the row it names is still the row the user picked.
///
/// Ids are only meaningful within one [`RowEpoch`]: see there for the operations that
/// renumber the grid, and [`Screen::display_row`] for resolving one back to a row on
/// screen (`None` once it has scrolled out of the visible band or aged out of history).
///
/// ```text
///   evicted=2        │ the stream (ids never reused, never shifted)
///   ─────────────────┼──────────────────────────────────────────────
///   AbsRow(0)   gone │ pushed out of the front of the ring
///   AbsRow(1)   gone │
///   AbsRow(2)        │ ┐ history (scrollback)
///   AbsRow(3)        │ ┘
///   AbsRow(4)        │ ┐ live screen (rows)
///   AbsRow(5)        │ ┘
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct AbsRow(u64);

impl AbsRow {
    /// The line after this one. `None` only at the end of the id space, which a
    /// terminal printing a line per nanosecond would reach in about six hundred years.
    pub fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(AbsRow)
    }

    /// The line before this one, or `None` at the very start of the stream.
    pub fn prev(self) -> Option<Self> {
        self.0.checked_sub(1).map(AbsRow)
    }
}

/// The identity regime [`AbsRow`] ids live in, bumped whenever they stop meaning what
/// they meant. Two things do that, and only these two:
///
/// - **A row is discarded out of the middle of the grid** rather than retired into
///   history: a scroll inside a `DECSTBM` region, a `DL`, or any scroll on the alt
///   screen (which keeps no history at all). Every row below the hole shifts up, so the
///   ids no longer name the same lines.
/// - **The content is destroyed wholesale**: `RIS`, a resize (rows re-index and we do
///   not re-wrap), an alt-screen switch, or an `ED` that erases the whole display.
///
/// Note what is *not* here: ordinary output. A row retiring into history keeps its id,
/// and a row evicted off the front of a full ring only makes ids below `evicted`
/// unresolvable — it shifts nothing. That asymmetry is what lets a selection (and the
/// viewport) survive a printing child, which is the entire feature.
///
/// Anything holding row ids across output stores the epoch it minted them in and is
/// dropped when the epoch moves on, rather than resolving them against a grid that has
/// since been renumbered.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct RowEpoch(u64);

impl RowEpoch {
    fn next(self) -> Self {
        RowEpoch(self.0.wrapping_add(1))
    }
}

/// A cell's rendition attributes: the SGR styles plus the two layout bits that
/// mark a wide character's halves. A bitfield newtype (no `bitflags` crate) so it
/// is one `u16`, `Copy`, and cheap to compare in the damage diff.
///
/// Soft wrap is deliberately *not* here: it describes a line, not a cell (see
/// [`Row::wrapped`](buffer::Row::wrapped)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Attrs(u16);

/// How an underline is drawn: `SGR 4:n`, where `4:0` is no underline at all, `4:1` the
/// plain one, and the rest are the shapes an editor uses to mean something. The curly one
/// is why this exists — every LSP in use marks an error by squiggling under it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum UnderlineStyle {
    #[default]
    Single,
    Double,
    Curly,
    Dotted,
    Dashed,
}

impl Attrs {
    pub const BOLD: Attrs = Attrs(1 << 0);
    pub const DIM: Attrs = Attrs(1 << 1);
    pub const ITALIC: Attrs = Attrs(1 << 2);
    pub const UNDERLINE: Attrs = Attrs(1 << 3);
    pub const REVERSE: Attrs = Attrs(1 << 4);
    pub const STRIKE: Attrs = Attrs(1 << 5);
    /// SGR 8 (conceal): the cell keeps its rune but the renderer draws it in the
    /// background color, so it is invisible yet still selectable and copyable.
    pub const HIDDEN: Attrs = Attrs(1 << 6);
    /// The left half of a wide (2-column) character; carries the rune.
    pub const WIDE_LEADER: Attrs = Attrs(1 << 7);
    /// The right half of a wide character; a placeholder the cursor skips.
    pub const WIDE_SPACER: Attrs = Attrs(1 << 8);

    /// Bits 9-11: how the underline is drawn ([`UnderlineStyle`]), not *whether* it is —
    /// that is still [`UNDERLINE`](Self::UNDERLINE), and the style only means anything
    /// alongside it.
    ///
    /// It lives in the bitfield's spare bits because it *fits*, and that is the whole
    /// design decision: a `Cell` is 16 bytes and the damage diff compares two screenfuls
    /// of them every painted frame, so carrying the style in a new field would have cost
    /// the entire grid 25% to decorate the handful of cells an editor squiggles. The
    /// underline *colour* (SGR 58) does not fit, which is exactly why it is still parsed,
    /// consumed and dropped rather than stored.
    const UNDERLINE_STYLE_SHIFT: u16 = 9;
    const UNDERLINE_STYLE_MASK: u16 = 0b111 << Attrs::UNDERLINE_STYLE_SHIFT;

    /// How the underline on this cell is drawn. Meaningless without
    /// [`UNDERLINE`](Self::UNDERLINE), and [`UnderlineStyle::Single`] for every cell that
    /// never asked for anything else.
    pub fn underline_style(self) -> UnderlineStyle {
        match (self.0 & Attrs::UNDERLINE_STYLE_MASK) >> Attrs::UNDERLINE_STYLE_SHIFT {
            1 => UnderlineStyle::Double,
            2 => UnderlineStyle::Curly,
            3 => UnderlineStyle::Dotted,
            4 => UnderlineStyle::Dashed,
            _ => UnderlineStyle::Single,
        }
    }

    pub fn set_underline_style(&mut self, style: UnderlineStyle) {
        let bits = match style {
            UnderlineStyle::Single => 0,
            UnderlineStyle::Double => 1,
            UnderlineStyle::Curly => 2,
            UnderlineStyle::Dotted => 3,
            UnderlineStyle::Dashed => 4,
        };
        self.0 = (self.0 & !Attrs::UNDERLINE_STYLE_MASK) | (bits << Attrs::UNDERLINE_STYLE_SHIFT);
    }

    /// All flags, in bit order, with their names, for `Debug` and for tests.
    const ALL: [(Attrs, &'static str); 9] = [
        (Attrs::BOLD, "BOLD"),
        (Attrs::DIM, "DIM"),
        (Attrs::ITALIC, "ITALIC"),
        (Attrs::UNDERLINE, "UNDERLINE"),
        (Attrs::REVERSE, "REVERSE"),
        (Attrs::STRIKE, "STRIKE"),
        (Attrs::HIDDEN, "HIDDEN"),
        (Attrs::WIDE_LEADER, "WIDE_LEADER"),
        (Attrs::WIDE_SPACER, "WIDE_SPACER"),
    ];

    pub const fn empty() -> Self {
        Attrs(0)
    }

    /// No attributes at all. The style bits do not count: they say how an underline
    /// *would* be drawn, and with no `UNDERLINE` there is none, so a cell carrying only a
    /// stale style is as plain as one carrying nothing.
    pub const fn is_empty(self) -> bool {
        self.0 & !Attrs::UNDERLINE_STYLE_MASK == 0
    }

    /// Whether every flag in `other` is set here.
    pub const fn contains(self, other: Attrs) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Attrs) {
        self.0 |= other.0;
    }

    pub fn remove(&mut self, other: Attrs) {
        self.0 &= !other.0;
    }

    /// Set or clear `other` by a boolean, the shape SGR wants (e.g. bold on 1,
    /// off on 22).
    pub fn set(&mut self, other: Attrs, on: bool) {
        if on {
            self.insert(other);
        } else {
            self.remove(other);
        }
    }

    /// The bits that decide how a *run* of cells is drawn: the two that pick a face
    /// (bold, italic) and the two rules laid over it (underline, with its shape, and
    /// strike).
    ///
    /// What is left out is as deliberate as what is in. `REVERSE` and `DIM` are folded
    /// into the resolved colour, which the painter compares separately; `HIDDEN` draws
    /// as a space and so leaves a run intact; the wide-pair bits break a run by standing
    /// a glyph alone, before any of this is asked.
    const RUN_MASK: u16 = Attrs::BOLD.0
        | Attrs::ITALIC.0
        | Attrs::UNDERLINE.0
        | Attrs::STRIKE.0
        | Attrs::UNDERLINE_STYLE_MASK;

    /// Just those bits, normalised, so two cells can be compared for "would draw the
    /// same" in one `u16` test.
    ///
    /// The shape bits drop out when there is no underline to draw, which is the same
    /// reading [`Self::is_empty`] takes: a cell carrying only a stale shape is not a
    /// different-looking cell, and must not split a run.
    ///
    /// One masked compare rather than a test per attribute because the painter asks this
    /// per cell, a quarter of a million times a frame.
    pub fn run_key(self) -> Attrs {
        let mask = if self.contains(Attrs::UNDERLINE) {
            Attrs::RUN_MASK
        } else {
            Attrs::RUN_MASK & !Attrs::UNDERLINE_STYLE_MASK
        };
        Attrs(self.0 & mask)
    }
}

impl std::ops::BitOr for Attrs {
    type Output = Attrs;
    fn bitor(self, rhs: Attrs) -> Attrs {
        Attrs(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Attrs {
    fn bitor_assign(&mut self, rhs: Attrs) {
        self.0 |= rhs.0;
    }
}

/// Lists the set flags (`Attrs(BOLD|ITALIC)`), so a grid dump in a golden test
/// reads like the styling it represents instead of a hex mask.
impl fmt::Debug for Attrs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Attrs(")?;
        if self.is_empty() {
            f.write_str("-")?;
        } else {
            let mut first = true;
            for (flag, name) in Attrs::ALL {
                if self.contains(flag) {
                    if !first {
                        f.write_str("|")?;
                    }
                    f.write_str(name)?;
                    first = false;
                }
            }
            // The style is not a flag, and a curly underline that printed the same as a
            // straight one would be invisible in a golden dump — which is where a
            // regression in it would otherwise hide.
            if self.contains(Attrs::UNDERLINE) && self.underline_style() != UnderlineStyle::Single {
                write!(f, ":{:?}", self.underline_style())?;
            }
        }
        f.write_str(")")
    }
}

/// How many distinct hyperlinks one screen can hold at once. The id space is a
/// `u16` with zero reserved for "no link", so this is all of it; running out is not
/// fatal (see [`Screen::collect_links`]).
const LINK_LIMIT: usize = u16::MAX as usize - 1;

/// How many bytes of query replies may accumulate before further answers are dropped
/// (see [`Screen::respond`]). Sized far above any real conversation: the chattiest
/// startup handshake in the wild is nvim's, a few hundred bytes, and this is two
/// hundred times that. Only a flood reaches it.
const RESPONSE_MAX: usize = 64 * 1024;

/// Which OSC 8 hyperlink a cell belongs to: an index into the screen's [`LinkTable`],
/// or [`LinkId::NONE`] for the overwhelming majority of cells, which are in none.
///
/// The URL is interned rather than stored on the cell for two reasons. A `Cell` must
/// stay small and `Copy` (the damage diff compares two screenfuls every painted
/// frame), and a `u16` costs the grid *nothing*: `Cell` was already padding two bytes
/// out to `char`'s alignment, so hyperlinks fit in the hole. And the same URL is
/// printed over and over (`ls --hyperlink` repeats a directory's links on every run),
/// so one copy shared by every cell that cites it is also simply less memory.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct LinkId(u16);

impl LinkId {
    /// The cell is not inside an OSC 8 hyperlink. Zero, so a blank cell — and any
    /// `Default` cell — is linkless without anyone saying so.
    pub const NONE: LinkId = LinkId(0);

    /// Whether this cell is inside a hyperlink at all. The check the hover probe runs
    /// first, before it considers scanning text.
    pub fn is_set(self) -> bool {
        self != LinkId::NONE
    }
}

/// The URLs of the OSC 8 hyperlinks a screen is holding, each interned once and cited
/// by [`LinkId`].
///
/// Ids are 1-based, so `urls[i]` is the URL of `LinkId(i + 1)` and zero is free to
/// mean "no link". The URL is stored twice — once in `urls` to go id → URL in O(1)
/// for the hover probe, once as the `index` key to go URL → id on intern so a link
/// printed a thousand times is interned once. That duplication is bounded (the id
/// space caps the table at [`LINK_LIMIT`] entries) and buys both directions their
/// natural cost, which a single structure could not.
#[derive(Default)]
struct LinkTable {
    urls: Vec<String>,
    index: HashMap<String, LinkId>,
}

impl LinkTable {
    /// The id for `url`, interning it if this is the first time we have seen it.
    /// `None` when the id space is exhausted, which is the caller's cue to collect the
    /// dead ids and try once more ([`Screen::intern_link`]).
    fn intern(&mut self, url: &str) -> Option<LinkId> {
        if let Some(&id) = self.index.get(url) {
            return Some(id);
        }
        if self.urls.len() >= LINK_LIMIT {
            return None;
        }
        // Ids are 1-based and the table is capped below `u16::MAX`, so this fits.
        let id = LinkId(u16::try_from(self.urls.len() + 1).ok()?);
        self.urls.push(url.to_string());
        self.index.insert(url.to_string(), id);
        Some(id)
    }

    /// The URL behind `id`, or `None` for [`LinkId::NONE`] and any id this table does
    /// not hold.
    fn url(&self, id: LinkId) -> Option<&str> {
        let slot = usize::from(id.0).checked_sub(1)?;
        self.urls.get(slot).map(String::as_str)
    }

    /// Drop every URL `live` does not mark and renumber the survivors, returning the
    /// old-id → new-id map (indexed by the old id's raw value) that the caller must
    /// then apply to every cell it kept. `live[i]` speaks for `LinkId(i)`, so slot
    /// zero is [`LinkId::NONE`] and is never a URL.
    fn compact(&mut self, live: &[bool]) -> Vec<LinkId> {
        let mut remap = vec![LinkId::NONE; self.urls.len() + 1];
        let old = std::mem::take(&mut self.urls);
        self.index.clear();
        // Counts only survivors, so it is bounded by the table we came in with and
        // cannot pass `LINK_LIMIT`, let alone wrap.
        let mut next: u16 = 0;
        for (slot, url) in old.into_iter().enumerate() {
            if live.get(slot + 1) != Some(&true) {
                continue;
            }
            next += 1;
            let id = LinkId(next);
            if let Some(entry) = remap.get_mut(slot + 1) {
                *entry = id;
            }
            self.index.insert(url.clone(), id);
            self.urls.push(url);
        }
        remap
    }
}

/// One grid cell: the base rune of its grapheme cluster, its foreground and
/// background colors, its rendition attributes, and the hyperlink it belongs to.
/// Small, `Copy`, and `Eq` so the per-frame damage diff can compare two screenfuls
/// cheaply.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cell {
    /// The base rune. Combining marks, when present, live in the `Row`'s side
    /// list keyed by column (see the module header).
    pub rune: char,
    pub fg: Color,
    pub bg: Color,
    pub attrs: Attrs,
    /// The OSC 8 hyperlink this cell is inside, [`LinkId::NONE`] for most cells.
    /// Carried per cell rather than as a span so that every operation that moves or
    /// overwrites a cell (insert, delete, scroll, erase) keeps the link right for
    /// free, with no ranges to fix up.
    pub link: LinkId,
}

impl Cell {
    /// The blank cell: a space in the default colors with no attributes. This is
    /// the grid's initial fill; an erase writes a space in the *current* bg.
    pub const BLANK: Cell = Cell {
        rune: ' ',
        fg: Color::Default,
        bg: Color::Default,
        attrs: Attrs::empty(),
        link: LinkId::NONE,
    };

    /// A cell carrying `rune` in the default colors, no attributes. Handy for
    /// tests and for plain printable output before any SGR is seen.
    pub fn new(rune: char) -> Self {
        Cell {
            rune,
            ..Cell::BLANK
        }
    }

    /// Whether this cell is the right-half placeholder of a wide character.
    pub fn is_wide_spacer(self) -> bool {
        self.attrs.contains(Attrs::WIDE_SPACER)
    }

    /// Whether this cell is the left half (the rune) of a wide character.
    pub fn is_wide_leader(self) -> bool {
        self.attrs.contains(Attrs::WIDE_LEADER)
    }
}

impl Default for Cell {
    fn default() -> Self {
        Cell::BLANK
    }
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cell({:?}, fg={:?}, bg={:?}, {:?}",
            self.rune, self.fg, self.bg, self.attrs
        )?;
        // Only when there is one, so the overwhelmingly common linkless cell reads
        // exactly as it always has.
        if self.link.is_set() {
            write!(f, ", link={}", self.link.0)?;
        }
        f.write_str(")")
    }
}

/// The cursor's drawn shape, set by `DECSCUSR` (`CSI Ps SP q`). The renderer maps
/// this to how it paints the cursor; whether it blinks is tracked separately.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CursorStyle {
    #[default]
    Block,
    Underline,
    Bar,
}

/// How the cursor looks: its shape and whether it blinks. `DECSCUSR` sets both at
/// once, so they travel together, and the derived [`Default`] is the one place that
/// says what bnkterm's cursor looks like at power-on: a **steady** block.
///
/// That default is also what `DECSCUSR 0` restores, which is the whole reason this
/// pair has a name. xterm documents `Ps = 0` as a *blinking* block, but no modern
/// terminal reads it that way: it means "back to the terminal's own default", and
/// TUIs emit it while restoring the terminal on exit. Taking xterm literally there
/// means a program that politely puts the cursor back leaves it blinking forever.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct CursorAppearance {
    style: CursorStyle,
    blink: bool,
}

impl CursorAppearance {
    const fn new(style: CursorStyle, blink: bool) -> Self {
        CursorAppearance { style, blink }
    }
}

/// A designated character set. bnkterm supports ASCII and the DEC Special
/// Graphics (line-drawing) set, which is what box-drawing TUIs rely on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Charset {
    Ascii,
    DecSpecialGraphics,
}

/// Translates [`AbsRow`]-anchored positions from a buffer's geometry before a
/// [`Buffer::reflow`] to after it, so a text selection and the OSC 133 prompt marks follow
/// the cells they name across a rewrap. Built by `reflow`, applied by [`Screen::resize`].
///
/// It resolves a cell by its logical position: an old id and column name a logical line and
/// an offset into it, and the reflow recorded which new row that offset landed on.
pub struct RowRemap {
    /// The buffer's evicted count when the reflow ran. Both the old and the new ids are
    /// numbered from it (a reflow reassigns ids but keeps the oldest surviving row's), so the
    /// arithmetic is a stream-index add on either side.
    evicted: u64,
    new_cols: usize,
    /// How many logical lines the reflowed region unwrapped into. An old row whose line index
    /// is at least this was a trailing blank the reflow absorbed, and maps to nothing.
    lines_len: usize,
    /// Per old stream row *in the reflowed region*: the logical line it joined and the offset
    /// its first column sits at within that line.
    old: Vec<(usize, usize)>,
    /// First new stream index of each logical line; `[lines_len]` is the reflowed length.
    line_first: Vec<usize>,
    /// Per new stream row in the reflowed region: the logical offset of its first cell.
    new_start: Vec<usize>,
    /// Old stream index where the frozen live-prompt region began: rows at or below it were
    /// clamped (one old row → one new row), not rewrapped, so the shell's SIGWINCH redraw
    /// finds the prompt at the row count it expects. Equal to the old row count when nothing
    /// was frozen.
    frozen_from: usize,
    /// New stream index the frozen region begins at (the number of rows the reflowed region
    /// produced).
    reflowed_len: usize,
}

impl RowRemap {
    /// The new stream index and column of the cell at old stream index `old_idx`, column
    /// `col`, or `None` if its row was a trailing blank the reflow absorbed.
    fn locate(&self, old_idx: usize, col: usize) -> Option<(usize, usize)> {
        if old_idx >= self.frozen_from {
            // The clamped live-prompt region: one old row maps straight to one new row at the
            // same column, only bounded to the new width.
            let new_idx = self.reflowed_len + (old_idx - self.frozen_from);
            return Some((new_idx, col.min(self.new_cols - 1)));
        }
        let (line, base) = *self.old.get(old_idx)?;
        if line >= self.lines_len {
            return None;
        }
        let off = base + col;
        let (start, end) = (self.line_first[line], self.line_first[line + 1]);
        // A line's rows carry contiguous offset ranges, so the last row whose start is not
        // past `off` is the one holding it; an offset beyond the content clamps to the last.
        let mut chosen = start;
        for k in start..end {
            if self.new_start[k] <= off {
                chosen = k;
            } else {
                break;
            }
        }
        Some((
            chosen,
            off.saturating_sub(self.new_start[chosen])
                .min(self.new_cols - 1),
        ))
    }

    /// Where the cell `(abs, col)` named before the reflow now sits, or `None` if its row was
    /// absorbed as a trailing blank or has since aged out of history.
    pub fn point(&self, abs: AbsRow, col: usize) -> Option<(AbsRow, usize)> {
        let old_idx = usize::try_from(abs.0.checked_sub(self.evicted)?).ok()?;
        let (new_idx, new_col) = self.locate(old_idx, col)?;
        let id = self
            .evicted
            .saturating_add(u64::try_from(new_idx).unwrap_or(u64::MAX));
        Some((AbsRow(id), new_col))
    }

    /// The new id of the row `abs` named, for a mark that anchors a row rather than a cell.
    pub fn row(&self, abs: AbsRow) -> Option<AbsRow> {
        self.point(abs, 0).map(|(r, _)| r)
    }
}

/// What a [`Screen::resize`] did to the row ids a caller may be holding (a text selection).
/// The grid's own prompt marks are carried internally; this tells the caller how to treat
/// its own.
pub enum ResizeEffect {
    /// Ids were unchanged (a height-only change on the primary screen): keep them as they are.
    Stable,
    /// Ids were renumbered by a width reflow but are translatable: remap each through the
    /// [`RowRemap`], re-stamping with the new [`Screen::row_epoch`].
    Reflowed(RowRemap),
    /// Ids were invalidated with no mapping (a resize on the alt screen, which only clamps):
    /// drop the anchored state.
    Reset,
}

/// The reusable working set of a hyperlink probe (see [`Screen::link_at`]): the
/// logical line being scanned, the map back from its bytes to the cells they came
/// from, and the URL the last hit found.
///
/// It is owned by the caller and reused because a probe is not a rare event: it runs
/// for every cell the pointer crosses, and again whenever output scrolls the screen
/// under a parked pointer. Rebuilding these three buffers in place keeps that path
/// allocation-free once they have reached their size, so hovering costs the scan and
/// nothing else.
#[derive(Default)]
pub struct LinkProbe {
    /// The logical line under the pointer: its soft-wrapped display rows joined, wide
    /// spacers dropped. Exactly the text [`crate::platform::link`] scans.
    text: String,
    /// Where each rune of `text` came from: its byte offset, and the display cell that
    /// printed it. Ascending in both, so a byte offset maps back to a cell by binary
    /// search (see [`LinkProbe::cell_at`]).
    runes: Vec<(usize, (usize, usize))>,
    /// The URL of the most recent hit (empty after a miss).
    url: String,
}

impl LinkProbe {
    /// The URL the last [`Screen::link_at`] hit found; empty after a miss.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// The display cell whose rune covers byte offset `at` in [`Self::text`]: the last
    /// rune that starts at or before it, so an offset landing inside a multi-byte rune
    /// still resolves to the cell that printed it.
    fn cell_at(&self, at: usize) -> Option<(usize, usize)> {
        let past = self.runes.partition_point(|(offset, _)| *offset <= at);
        self.runes.get(past.checked_sub(1)?).map(|&(_, cell)| cell)
    }
}

/// The terminal grid: primary and alternate buffers, the active graphic rendition
/// (pen), and the terminal-wide modes. This is the type the VT parser drives (via
/// the `Perform` impl below); the renderer reads the visible cells back out.
/// Positions in its API are 0-based (the parser converts the 1-based numbers
/// escape sequences carry).
pub struct Screen {
    primary: Buffer,
    alt: Buffer,
    on_alt: bool,
    pen: Pen,
    autowrap: bool,        // DECAWM (?7)
    origin_mode: bool,     // DECOM (?6)
    cursor_visible: bool,  // DECTCEM (?25)
    insert_mode: bool,     // IRM (4)
    bracketed_paste: bool, // ?2004
    app_cursor_keys: bool, // DECCKM (?1)
    keypad_app: bool,      // DECKPAM/DECKPNM, for the input encoder (phase 3)
    /// The two designated charsets and whether GL points at G1 (SO) not G0 (SI).
    g0: Charset,
    g1: Charset,
    gl_is_g1: bool,
    title: String,
    /// How many lines the *view* is scrolled up into the primary scrollback (0 =
    /// pinned to the live bottom). A display concern, not terminal state: writes
    /// always land on the live screen; only what the renderer shows shifts. The
    /// alt screen has no scrollback, so it forces this to 0.
    view_offset: usize,
    /// The regime the current [`AbsRow`] ids belong to. Anything that stores rows
    /// across output compares against it and drops what it holds when it moves on.
    epoch: RowEpoch,
    /// Mouse reporting the child asked for (`?1000`/`?1002`/`?1003`/`?1006`); the
    /// app reads it to decide whether a pointer event goes to the child or drives
    /// local selection/scroll.
    mouse: MouseMode,
    /// How the cursor is drawn, from `DECSCUSR`. Opens as a steady block and returns
    /// there on `DECSCUSR 0`; see [`CursorAppearance`]. A program can still ask for a
    /// blink explicitly (`DECSCUSR 1`/`3`/`5`). Where the cursor *is* lives in the
    /// active buffer; this is only what it looks like.
    cursor_appearance: CursorAppearance,
    /// Bytes to write back to the child in answer to a query (DA, DSR). The grid
    /// holds no PTY, so it queues its replies as data here; the app drains them
    /// after each parse and writes them to the master fd. Kept tiny (queries are
    /// rare), reused across parses.
    responses: Vec<u8>,
    /// The URLs of the OSC 8 hyperlinks on screen and in scrollback, which cells cite
    /// by [`LinkId`]. Shared by both buffers: the alt screen has its own cells but not
    /// its own link namespace, so an id means the same thing whichever buffer holds it
    /// and switching buffers costs nothing.
    links: LinkTable,
    /// The kitty keyboard protocol's flag stacks, **one per screen**. It is a stack
    /// because a full-screen program pushes the flags it wants on entry and pops them on
    /// exit, so it cannot strand the terminal in a mode the shell underneath it does not
    /// understand — and a program that dies without popping is cleaned up by whatever
    /// pushed beneath it. The top entry is what is in force; an empty stack is legacy.
    ///
    /// Per screen because kitty's protocol says so, and for the reason the whole stack
    /// exists: a program that dies in the alt screen must not leave the *main* screen's
    /// shell speaking a protocol it does not know. `nvim` sends `?1049h` then `CSI > 1 u`,
    /// is `SIGKILL`ed, and never sends `CSI < 1 u`; the shell's `?1049l` comes back to a
    /// main screen where `Esc` is `CSI 27u` and `Ctrl+C` is `CSI 99;5u`. zsh vi-mode never
    /// leaves insert, fzf bindings misfire, and `Ctrl+C` prints garbage.
    ///
    /// Two fields rather than a save/restore in `switch_alt`, which is deliberate: the
    /// bug this fixes *was* a switch that forgot a field, so the fix is a shape where
    /// there is nothing to forget. [`Screen::kitty_stack`] picks by `on_alt`, exactly as
    /// [`Screen::active`] picks the buffer.
    ///
    /// The child chooses the depth, so these are attacker-controlled and therefore capped
    /// ([`KITTY_STACK_LIMIT`]); a program in a push loop must not grow the terminal's
    /// memory without bound.
    kitty_primary: Vec<KittyFlags>,
    kitty_alt: Vec<KittyFlags>,
    /// xterm's `modifyOtherKeys` level (`CSI > 4 ; Pv m`). Not a stack: XTMODKEYS has no
    /// push/pop, a program just sets a level and sets it back.
    modify_other_keys: ModifyOtherKeys,
    /// The colour palette. Terminal state, not renderer state: a program can read it
    /// (`OSC 4/10/11/12` with a `?`) and write it, so it has to live where the escape
    /// sequences can reach it. The renderer borrows it at paint time.
    ///
    /// Boxed, and the reason is measured rather than stylistic. A 256-entry palette is
    /// ~780 bytes, and inline it pushed the fields the *parser* touches on every byte —
    /// the pen, the cursor, the mode flags — apart across cache lines. The palette is cold
    /// to the parser and warm only to the painter, so it goes behind a pointer and the hot
    /// fields close ranks. See the note in `perf/` before undoing this.
    theme: Box<Theme>,
    /// The working directory the shell reported (`OSC 7`), if it has.
    ///
    /// The app can also read this out of `/proc/<pid>/cwd`, and does — but only for a child
    /// on *this* machine. A shell on the far end of an ssh hop has no /proc we can see, and
    /// is the case where knowing the directory is worth the most. So when the shell says,
    /// we believe the shell.
    cwd: Option<String>,
    /// The prompts the shell has marked (`OSC 133;A`), oldest first, and how the command
    /// after each one turned out.
    ///
    /// Marks name rows by [`AbsRow`], so they survive output scrolling under them — and
    /// they are dropped wholesale when the row epoch ends, because at that point the ids
    /// they hold name lines that no longer exist. Exactly the rule the selection follows,
    /// and for exactly the same reason.
    prompts: Vec<Prompt>,
    /// Grapheme clustering (`?2027`): a user-perceived character occupies one cell, rather
    /// than each scalar occupying its own.
    ///
    /// Off by default, and it has to be. Both answers are legitimate — `wcwidth` says the
    /// astronaut is four columns (woman 2 + ZWJ 0 + rocket 2) and a clustering terminal
    /// says two — and an application that measures one way while the terminal measures the
    /// other puts its cursor where the glyphs are not. So the terminal cannot simply pick
    /// the better answer: it has to *say* which one it uses, and only change it for a
    /// program that asked. That is the whole reason this is a mode and not a fix.
    grapheme_clustering: bool,
    /// The UAX #29 machine, carried across scalars, and the cell the cluster it is
    /// building lives in. A cluster arrives one scalar at a time and cannot be looked
    /// ahead of: the rest of it may be in the next read off the pty, or may never come.
    cluster: grapheme::BreakState,
    cluster_anchor: Option<ClusterAnchor>,
    /// The last character actually printed, which is all REP (`CSI b`) has to repeat.
    /// Cleared by any C0 control, so a REP after a newline repeats nothing rather than
    /// a screenful of whatever ended the line above.
    last_printed: Option<char>,
    /// Synchronized output (`?2026`): the child has asked us to hold the frame until it
    /// finishes drawing, so it is never seen half-painted. The grid keeps *updating*
    /// while this is set — only presentation waits. The grid owns no clock, so the
    /// timeout that stops a dead child freezing the window lives in the app.
    synchronized: bool,
    /// Text the child has asked to put on a selection (`OSC 52`). The grid owns no
    /// compositor connection, so it queues the request here the same way it queues query
    /// replies, and the app drains it and does the Wayland half.
    clipboard_writes: Vec<(ClipboardTarget, Vec<u8>)>,
    /// The child rang the bell (BEL, `0x07`) since the app last looked. What a bell
    /// *means* is the app's business (a flash, a mark on the tab); all the grid knows is
    /// that it was rung.
    bell: bool,
    /// Focus reporting (`?1004`): the child wants to be told when the window gains or
    /// loses focus, so it can dim an inactive pane or pause an animation.
    focus_events: bool,
    /// In-band resize (`?2048`): the child wants the new size as an escape sequence,
    /// not only as a SIGWINCH.
    in_band_resize: bool,
    /// The window's size in pixels, carried by the in-band resize report. The grid has no
    /// other use for pixels and never computes them; the app sets this.
    pixel_size: (u32, u32),
}

/// Which of the three named colours an `OSC 10/11/12` is about.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NamedColor {
    Foreground,
    Background,
    Cursor,
}

/// A prompt the shell marked, and the fate of the command typed at it.
///
/// This is what `OSC 133` buys: the terminal stops seeing an undifferentiated river of
/// text and starts knowing where one command ended and the next began. Everything good
/// downstream — jump to the last prompt, select a command's output, colour a failure —
/// falls out of knowing just this.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Prompt {
    /// The row the prompt starts on.
    pub row: AbsRow,
    /// Where the command's output begins (`OSC 133;C`), once it has.
    pub output: Option<AbsRow>,
    /// The command's exit status (`OSC 133;D;<code>`), once it has one. `Some(0)` is a
    /// success; anything else is the shell telling us it went wrong.
    pub exit: Option<i32>,
}

/// How many prompts to remember. The shell writes these, and a shell in a loop is still a
/// program the terminal must not let grow its memory without bound. Far more than a
/// scrollback's worth of prompts, so in practice the ring ages them out first.
const PROMPT_LIMIT: usize = 4096;

/// The cell a grapheme cluster is being built in, and where the cursor was left after it.
///
/// The cursor position is what makes the anchor safe: a cluster cannot span a cursor move,
/// so if the cursor is no longer where this cluster left it, the cluster is over and the
/// next scalar starts a new one. That check costs nothing and needs no hooks in the twenty
/// places that move a cursor — any of which would otherwise have been a way to leave a
/// stale anchor pointing at a cell that has since been overwritten.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ClusterAnchor {
    row: usize,
    col: usize,
    after: (usize, usize),
}

/// Which selection an `OSC 52` write is for. The X11 letters: `c` is the clipboard,
/// `p` (and `s`, the primary/secondary muddle nobody kept straight) the primary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ClipboardTarget {
    Clipboard,
    Primary,
}

/// How deep the kitty keyboard stack may go before a push starts dropping the oldest
/// entry. kitty itself uses a small fixed depth for the same reason: the stack exists
/// so a program can nest a mode over its parent's, and nothing legitimate nests deeply.
/// Dropping from the *bottom* keeps the most recent (the innermost program's) intent,
/// which is the one that matters.
const KITTY_STACK_LIMIT: usize = 16;

impl Screen {
    pub fn new(cols: usize, rows: usize) -> Self {
        Screen {
            primary: Buffer::new(cols, rows, DEFAULT_SCROLLBACK),
            alt: Buffer::new(cols, rows, 0),
            on_alt: false,
            pen: Pen::default(),
            autowrap: true,
            origin_mode: false,
            cursor_visible: true,
            insert_mode: false,
            bracketed_paste: false,
            app_cursor_keys: false,
            keypad_app: false,
            g0: Charset::Ascii,
            g1: Charset::Ascii,
            gl_is_g1: false,
            title: String::new(),
            view_offset: 0,
            epoch: RowEpoch::default(),
            mouse: MouseMode::default(),
            cursor_appearance: CursorAppearance::default(),
            responses: Vec::new(),
            links: LinkTable::default(),
            kitty_primary: Vec::new(),
            kitty_alt: Vec::new(),
            modify_other_keys: ModifyOtherKeys::default(),
            theme: Box::new(Theme::default()),
            cwd: None,
            prompts: Vec::new(),
            grapheme_clustering: false,
            cluster: grapheme::BreakState::default(),
            cluster_anchor: None,
            last_printed: None,
            synchronized: false,
            clipboard_writes: Vec::new(),
            bell: false,
            focus_events: false,
            in_band_resize: false,
            pixel_size: (0, 0),
        }
    }

    /// A screen whose history holds `limit` lines, so a test can watch the ring fill
    /// and evict without pushing [`DEFAULT_SCROLLBACK`] lines through the parser.
    #[cfg(test)]
    fn with_scrollback(cols: usize, rows: usize, limit: usize) -> Self {
        Screen {
            primary: Buffer::new(cols, rows, limit),
            ..Screen::new(cols, rows)
        }
    }

    fn active(&self) -> &Buffer {
        if self.on_alt {
            &self.alt
        } else {
            &self.primary
        }
    }

    fn active_mut(&mut self) -> &mut Buffer {
        if self.on_alt {
            &mut self.alt
        } else {
            &mut self.primary
        }
    }

    /// The kitty flag stack of the screen currently showing. Same selector as
    /// [`Self::active`], on the same flag, so the two can never disagree about which
    /// screen we are on.
    pub(super) fn kitty_stack(&self) -> &Vec<KittyFlags> {
        if self.on_alt {
            &self.kitty_alt
        } else {
            &self.kitty_primary
        }
    }

    pub(super) fn kitty_stack_mut(&mut self) -> &mut Vec<KittyFlags> {
        if self.on_alt {
            &mut self.kitty_alt
        } else {
            &mut self.kitty_primary
        }
    }

    /// The cell an erase or scroll fills with: a space in the current background
    /// (background-color erase, as xterm does), default foreground, no attrs.
    ///
    /// Deliberately *not* the pen's hyperlink, even while one is open: erased ground
    /// is not inside the anchor. Blanks that an insert (ICH) opens in the middle of a
    /// link are likewise outside it, which is what splits the run the hover probe
    /// walks — exactly the behaviour you want, since the text either side of the gap
    /// is no longer one label.
    fn blank_cell(&self) -> Cell {
        Cell {
            rune: ' ',
            fg: Color::Default,
            bg: self.pen.bg,
            attrs: Attrs::empty(),
            link: LinkId::NONE,
        }
    }
}
