//! The screen model: the cell grid, cursor, scrollback, and the operations the
//! VT parser drives. This is the type `bytes -> vt::Parser -> grid::Screen` feeds,
//! and where terminal *correctness* lives. The parser holds no grid state and the
//! grid holds no parser state: this file knows only semantic operations (print a
//! rune, move the cursor, erase, scroll), never bytes.
//!
//! Structure, chosen for the access pattern:
//!
//! ```text
//!   Screen ── primary: Buffer ─┐
//!         └── alt:     Buffer ─┴─ each Buffer is a ring of Rows:
//!
//!     scrollback (VecDeque<Row>)   visible (VecDeque<Row>, len == rows)
//!     ┌───┬───┬───┐               ┌───┬───┬───┬───┐
//!     │ … │ … │old│  ◀── scroll ──│row│row│row│row│
//!     └───┴───┴───┘               └───┴───┴───┴───┘
//! ```
//!
//! A scroll rotates *row headers* (a pointer move), never the cells: a
//! full-screen scroll is a `pop_front` + `push_back` (O(1)); a region scroll
//! moves only the headers inside the region. The evicted row's cell buffer is
//! recycled into the new blank row, so a steady scroll allocates nothing.
//!
//! Two cell shapes corrupt a screen when handled wrong:
//!
//!   * Wide characters (CJK, most emoji) occupy two columns. The left column
//!     carries the rune and the `WIDE_LEADER` attr; the right column is a
//!     `WIDE_SPACER` placeholder the cursor steps over and the renderer skips.
//!     Overwriting either half cleans up its orphaned partner (`write_cell`).
//!
//!   * Combining marks (an accent after its base, a Hangul jamo stack) are rare,
//!     so a `Cell` stores only the base rune inline and extra marks overflow into
//!     a small per-`Row` side list keyed by column. It travels with the row when
//!     it scrolls, and the common all-single-codepoint row pays nothing.

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

/// Columns between default tab stops.
const TAB_WIDTH: usize = 8;

/// The widest a single logical line may grow before reflow treats the boundary as a
/// hard break. It bounds both the intermediate buffer and the per-line rewrap cost
/// against a child that prints megabytes with no newline: a line longer than this is
/// rewrapped in independent blocks that never re-join across the seam. wezterm uses the
/// same 1024; the visible cost is a seam every 1024 columns on such a line, invisible on
/// any real output. See [`Buffer::reflow`].
const MAX_LOGICAL_COLS: usize = 1024;

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

/// What a scroll did to the rows it moved, which is all the rest of the terminal needs to
/// know about it.
///
/// - `pushed`: rows that *retired into history* off the top of the screen. A scrolled-back
///   viewport must follow them up ([`Screen::follow_history`]).
/// - `renumbered`: the ids stopped naming the same lines, so the [`RowEpoch`] must end.
///
/// The two are independent, and the reason is worth stating once. Rows live in one stream,
/// `scrollback ++ lines`, and an [`AbsRow`] is a position in it. A row retiring into
/// history does not move within that stream — it goes from the front of `lines` to the
/// back of `scrollback`, which is the very same slot — so a *full-screen* scroll keeps
/// every id: it only appends a blank at the far end.
///
/// ```text
///   stream:  [ h0 h1 | r0 r1 r2 ]        r0 retires, blank appends at the end
///            [ h0 h1   r0 | r1 r2 __ ]   every surviving row kept its slot
/// ```
///
/// Anything else that moves rows *inside* the stream renumbers it, and there are two such
/// moves. A row can be torn out of the middle — a scroll inside a `DECSTBM` region, a
/// `DL`, any scroll on the historyless alt screen, or any scroll *down* — and everything
/// past the hole shifts. Or a top-anchored region can stop short of the last row, in which
/// case the blank that fills in behind the scroll is inserted mid-stream and shifts every
/// row *below the region*, even though nothing about those rows moved on screen:
///
/// ```text
///   region 0..=1 of a 4-row screen, r0 retiring into history
///   stream:  [ h0 | r0 r1 r2 r3 ]
///            [ h0   r0 | r1 __ r2 r3 ]   ← the blank landed mid-stream:
///                            ↑↑  r2 and r3 each shifted up an id, sitting still
/// ```
///
/// That last case is the subtle one: it pushes to history *and* renumbers, so a program
/// holding a status line below a scroll region would silently drag a selection onto the
/// blank if only `pushed` were reported.
#[derive(Clone, Copy, Default)]
struct Scrolled {
    pushed: usize,
    renumbered: bool,
}

/// A cell's rendition attributes: the SGR styles plus the two layout bits that
/// mark a wide character's halves. A bitfield newtype (no `bitflags` crate) so it
/// is one `u16`, `Copy`, and cheap to compare in the damage diff.
///
/// Soft wrap is deliberately *not* here: it describes a line, not a cell (see
/// [`Row::wrapped`]).
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

/// The cursor: a position plus the deferred-wrap flag that implements xterm's
/// last-column rule. `pending_wrap` is set after a glyph lands in the final
/// column; the wrap to the next line is delayed until the next glyph actually
/// arrives, so writing exactly `cols` characters does not scroll.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct Cursor {
    row: usize,
    col: usize,
    pending_wrap: bool,
}

/// The current graphic rendition (SGR state): the colors and attributes printed
/// cells receive, plus the OSC 8 hyperlink they fall inside. `Copy` so a
/// save/restore (DECSC/DECRC) is a plain move.
///
/// The link rides on the pen because that is exactly what it is: a mode the child
/// turns on, prints under, and turns off, no different from bold. It is *not* an SGR
/// attribute, though, so `SGR 0` (reset) must leave it alone — closing a hyperlink is
/// `OSC 8 ; ; ST` and nothing else, and a program that resets colors mid-anchor (as
/// any colored `ls` listing does) still expects the anchor to hold.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Pen {
    fg: Color,
    bg: Color,
    attrs: Attrs,
    link: LinkId,
}

impl Pen {
    /// SGR 0: default colors, no attributes, and the hyperlink left exactly as it was.
    /// See the type's header for why the link survives a rendition reset — a colored
    /// `ls --hyperlink` listing emits `SGR 0` between entries *inside* an open anchor,
    /// so clearing the link here would break the most common OSC 8 producer there is.
    fn reset_rendition(&mut self) {
        // Destructured rather than `..Pen::default()` so that adding a field to `Pen`
        // is a compile error here: whoever adds it has to say whether SGR 0 clears it.
        let Pen {
            fg,
            bg,
            attrs,
            link: _,
        } = Pen::default();
        self.fg = fg;
        self.bg = bg;
        self.attrs = attrs;
    }
}

impl Default for Pen {
    fn default() -> Self {
        Pen {
            fg: Color::Default,
            bg: Color::Default,
            attrs: Attrs::empty(),
            link: LinkId::NONE,
        }
    }
}

/// A saved cursor (DECSC): position, pen, and origin mode, restored by DECRC.
#[derive(Clone, Copy)]
struct Saved {
    cursor: Cursor,
    pen: Pen,
    origin: bool,
    /// The character set shift state: the G0 and G1 designations and which of them GL
    /// is currently mapped to. DECSC saves it and DECRC restores it, which is not
    /// optional decoration — see [`Screen::save_cursor`].
    charsets: Charsets,
}

/// The character set shift state: what G0 and G1 are designated as, and which one GL
/// reads through (SI selects G0, SO selects G1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Charsets {
    g0: Charset,
    g1: Charset,
    gl_is_g1: bool,
}

/// The most combining marks a single cell retains. Unicode's Stream-Safe Text
/// format caps a grapheme at 30, so this sits just above it: real accented and
/// stacked text is never clipped, but an unbounded Zalgo stream cannot grow one
/// cell's side table (or the render string it feeds) without limit.
const MAX_COMBINING_PER_CELL: usize = 32;

/// One combining mark riding along with the base cell in `col`, in arrival order.
/// Flattened into the row's [`Row::combining`] list rather than owned per marked
/// cell, so the whole row's marks live in one reusable heap buffer (see the field
/// docs for why that matters).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CombiningMark {
    col: usize,
    ch: char,
}

/// One row of cells plus its rare combining marks. Cells are contiguous, so a
/// row is cheap to scan and write. Combining marks live in a tiny side list keyed
/// by column so they ride along when the row scrolls and cost nothing on the
/// common all-single-codepoint row.
#[derive(Clone, Debug)]
struct Row {
    cells: Vec<Cell>,
    /// The row's combining marks, flattened: one `(col, ch)` pair per mark rather
    /// than an owned `Vec<char>` per marked cell. Flattening keeps the side table a
    /// single heap buffer that [`Row::reset`] clears in place, so a recycled scroll
    /// row reuses its capacity and a warmed combining/emoji line allocates nothing.
    /// A column's marks read back by filtering on `col`, which preserves the order
    /// they arrived in.
    combining: Vec<CombiningMark>,
    /// Autowrap carried this line onto the next row: the two are one logical line,
    /// so a selection spanning them copies as unbroken text and a triple-click takes
    /// both. A hard newline leaves this false.
    ///
    /// It lives on the row rather than on the row's last cell because it describes the
    /// *line*, and a cell is addressed by a column that a resize moves: widening would
    /// strand the flag mid-row and narrowing would truncate it away, and either one
    /// silently breaks a wrapped line in half on the next copy. We do not re-wrap on
    /// resize ([`Buffer::resize`]), so nothing would put it back.
    ///
    /// The invariant: true only while the text that wrapped is still the text in the
    /// final column. Every write or erase reaching that column clears it.
    wrapped: bool,
}

impl Row {
    fn filled(cols: usize, cell: Cell) -> Self {
        Row {
            cells: vec![cell; cols],
            combining: Vec::new(),
            wrapped: false,
        }
    }

    /// Reset every cell to `blank` and drop combining marks, reusing the existing
    /// allocation so a recycled scroll row never allocates. The wrap link goes with
    /// the content: a recycled row must not inherit one and glue two unrelated lines
    /// together.
    fn reset(&mut self, cols: usize, blank: Cell) {
        if self.cells.len() == cols {
            self.cells.iter_mut().for_each(|c| *c = blank);
        } else {
            self.cells.clear();
            self.cells.resize(cols, blank);
        }
        self.combining.clear();
        self.wrapped = false;
    }

    /// Blank a wide glyph straddling the boundary *before* `col`, so an operation that
    /// cuts the row there cannot leave half a glyph behind.
    ///
    /// A pair straddles the cut exactly when the cell at `col` is a spacer: its leader is
    /// the cell before it, on the far side of the line about to be drawn.
    ///
    /// ```text
    ///   split before col 2        the pair straddles the cut, so both halves go
    ///   ┌───┬───┬───┬───┐         ┌───┬───┬───┬───┐
    ///   │ a │ 一────▶│ b │   ──▶  │ a │   │   │ b │
    ///   └───┴───┴─┃─┴───┘         └───┴───┴───┴───┘
    ///             ┃ the cut
    /// ```
    ///
    /// Every shift is a pair of cuts (see [`Buffer::insert_blanks`] and
    /// [`Buffer::delete_chars`]), which is why this is stated once here rather than
    /// open-coded per operation: each one gets it independently wrong otherwise.
    fn split_wide_at(&mut self, col: usize, blank: Cell) {
        if !self.cells.get(col).is_some_and(|c| c.is_wide_spacer()) {
            return;
        }
        let Some(lead) = col.checked_sub(1) else {
            return;
        };
        if let Some(slice) = self.cells.get_mut(lead..=col) {
            slice.iter_mut().for_each(|c| *c = blank);
        }
        self.combining.retain(|m| m.col != lead && m.col != col);
    }

    /// The combining marks attached to `col`, in arrival order; empty if the cell
    /// carries none. Yields `char` so a caller appends them straight onto a run
    /// string without materializing a temporary collection.
    fn marks(&self, col: usize) -> impl Iterator<Item = char> + '_ {
        self.combining
            .iter()
            .filter(move |m| m.col == col)
            .map(|m| m.ch)
    }

    /// Whether `col` carries any combining mark. Cheaper than `marks(col).next()` at
    /// the ink test, and reads clearly there.
    fn has_marks(&self, col: usize) -> bool {
        self.combining.iter().any(|m| m.col == col)
    }

    fn add_mark(&mut self, col: usize, mark: char) {
        // Clamp how many marks one cell can stack. Unicode's Stream-Safe format caps a
        // grapheme at 30 combining marks; a hostile stream (Zalgo) piles on thousands,
        // which would grow this row's side table and every render string it feeds
        // without bound. Beyond the cap the extra marks are dropped: invisible clutter
        // no reader needs, and the bound keeps retained memory and render work finite.
        if self.combining.iter().filter(|m| m.col == col).count() >= MAX_COMBINING_PER_CELL {
            return;
        }
        self.combining.push(CombiningMark { col, ch: mark });
    }

    fn clear_marks(&mut self, col: usize) {
        self.combining.retain(|m| m.col != col);
    }

    /// Grow or shrink the row to `new_cols`, padding with blanks or truncating.
    /// Truncation drops any combining marks past the new edge and blanks a wide
    /// glyph's leader whose spacer just fell off, so no half of a wide cell is
    /// ever left dangling. [`Row::wrapped`] is line state, not cell state, so it
    /// survives untouched. This is the column half of a terminal resize; it does
    /// not re-wrap soft-wrapped content (see [`Buffer::resize`]).
    fn resize_cols(&mut self, new_cols: usize) {
        let old = self.cells.len();
        if new_cols < old {
            self.cells.truncate(new_cols);
            self.combining.retain(|m| m.col < new_cols);
            if let Some(last) = self.cells.last_mut() {
                if last.is_wide_leader() {
                    *last = Cell::BLANK;
                }
            }
        } else if new_cols > old {
            self.cells.resize(new_cols, Cell::BLANK);
        }
    }

    /// Refill this row in place from `cells` (at most `cols` of them), padding to `cols`
    /// with blanks, and set the soft-wrap link. Reuses the existing cell allocation so a
    /// reflow recycles rows instead of allocating fresh. Combining marks are cleared; the
    /// caller re-adds any the new columns carry.
    fn refill(&mut self, cols: usize, cells: &[Cell], wrapped: bool) {
        self.cells.clear();
        self.cells.extend_from_slice(cells);
        self.cells.resize(cols, Cell::BLANK);
        self.combining.clear();
        self.wrapped = wrapped;
    }
}

/// A logical line reassembled from a run of soft-wrapped physical rows: the flat cell
/// sequence, with its combining marks re-keyed from per-row columns to the logical
/// column. This is reflow's intermediate — [`Buffer::reflow`] unwraps the stream into
/// these, then rewraps them at the new width. It carries no "was this a hard newline"
/// flag on purpose: a line ends either at a real newline or at [`MAX_LOGICAL_COLS`], and
/// both cases rewrap the same way (the last row is `wrapped = false`), so the seam a cap
/// break leaves behind is indistinguishable from a newline and will not re-join on a
/// later reflow — which is exactly the bound the cap is there to keep.
#[derive(Default)]
struct LogicalLine {
    cells: Vec<Cell>,
    combining: Vec<CombiningMark>,
}

impl LogicalLine {
    /// Drop trailing default-blank cells (and any marks that sat on them). A cell with a
    /// non-default background or a combining mark is content, not padding, and stops the
    /// trim: a coloured prompt bar drawn with spaces, or a marked blank, must survive a
    /// reflow rather than be pulled up into the line above it. A wide glyph's spacer is
    /// not `Cell::BLANK` (it carries `WIDE_SPACER`), so a trailing wide pair is never
    /// half-trimmed.
    fn trim_trailing_blanks(&mut self) {
        let mut end = self.cells.len();
        while end > 0
            && self.cells[end - 1] == Cell::BLANK
            && !self.combining.iter().any(|m| m.col == end - 1)
        {
            end -= 1;
        }
        self.cells.truncate(end);
        self.combining.retain(|m| m.col < end);
    }
}

/// Take a row from the recycle `pool` (or allocate one when it is empty) and fill it from
/// `cells` padded to `cols`, with the given soft-wrap link and no combining marks. This is
/// reflow's row factory: draining the old stream leaves a pool of `Row`s whose cell
/// allocations this reuses, so a rewrap does not allocate a row per line.
fn take_row(pool: &mut Vec<Row>, cols: usize, cells: &[Cell], wrapped: bool) -> Row {
    let mut row = pool.pop().unwrap_or_else(|| Row::filled(cols, Cell::BLANK));
    row.refill(cols, cells, wrapped);
    row
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

/// One screen buffer: the visible rows (a ring so a scroll rotates row headers,
/// never cells), the scrollback ring behind it, the cursor, the DECSTBM scroll
/// region, and the tab stops. Ring mechanics live here; the terminal *semantics*
/// (what each escape means) live on `Screen`.
struct Buffer {
    cols: usize,
    rows: usize,
    /// Visible rows, always exactly `rows` long.
    lines: VecDeque<Row>,
    /// History above the screen, oldest at the front, capped at `scrollback_limit`.
    scrollback: VecDeque<Row>,
    scrollback_limit: usize,
    /// How many rows have left the front of the stream for good: evicted from a full
    /// ring, or dropped when `ED 3` cleared the history. It is the offset that turns an
    /// [`AbsRow`] into an index into `scrollback ++ lines`, and the reason an id is
    /// never reused: the front of the stream only ever moves forward.
    evicted: u64,
    cursor: Cursor,
    /// What DECSC last stored *for this buffer*. Each screen owns its own slot, and it
    /// outlives a trip to the other screen and back: a program may save on the alt
    /// screen, leave, return, and restore, and it expects to find what it saved.
    saved: Option<Saved>,
    /// DECSTBM scroll region, inclusive, within `0..rows`.
    scroll_top: usize,
    scroll_bottom: usize,
    /// Tab stops; `tabs[c]` is true where a horizontal tab lands.
    tabs: Vec<bool>,
}

impl Buffer {
    fn new(cols: usize, rows: usize, scrollback_limit: usize) -> Self {
        let cols = cols.max(1);
        let rows = rows.max(1);
        let mut lines = VecDeque::with_capacity(rows);
        for _ in 0..rows {
            lines.push_back(Row::filled(cols, Cell::BLANK));
        }
        Buffer {
            cols,
            rows,
            lines,
            scrollback: VecDeque::with_capacity(scrollback_limit),
            scrollback_limit,
            evicted: 0,
            cursor: Cursor::default(),
            saved: None,
            scroll_top: 0,
            scroll_bottom: rows - 1,
            tabs: default_tabs(cols),
        }
    }

    fn line(&self, row: usize) -> Option<&Row> {
        self.lines.get(row)
    }

    fn cell(&self, row: usize, col: usize) -> Cell {
        self.lines
            .get(row)
            .and_then(|r| r.cells.get(col))
            .copied()
            .unwrap_or(Cell::BLANK)
    }

    /// The row at `idx` in this buffer's stream — history first (oldest at 0), then the
    /// live screen — which is the one column of rows every other row lookup is a view
    /// onto. `None` past the live bottom.
    fn stream_row(&self, idx: usize) -> Option<&Row> {
        if idx < self.scrollback.len() {
            self.scrollback.get(idx)
        } else {
            self.lines.get(idx - self.scrollback.len())
        }
    }

    /// The stream index of `abs`, or `None` once it has been evicted off the front.
    fn stream_index(&self, abs: AbsRow) -> Option<usize> {
        usize::try_from(abs.0.checked_sub(self.evicted)?).ok()
    }

    /// The row `abs` names, or `None` if it has aged out of history or does not exist
    /// yet (an id past the live bottom).
    fn abs_row(&self, abs: AbsRow) -> Option<&Row> {
        self.stream_row(self.stream_index(abs)?)
    }

    /// The id of the row at stream index `idx`.
    fn abs_of(&self, idx: usize) -> AbsRow {
        AbsRow(
            self.evicted
                .saturating_add(idx.try_into().unwrap_or(u64::MAX)),
        )
    }

    /// One past the newest row's id: the live bottom of the stream.
    fn abs_end(&self) -> AbsRow {
        self.abs_of(self.scrollback.len() + self.rows)
    }

    /// The row shown at display position `display_row` when the view is scrolled
    /// `offset` lines up into scrollback. The window of `rows` starts `offset` lines
    /// above the live top. `offset` is assumed already clamped to `0..=scrollback
    /// .len()`. Returns `None` past the end (a short final window).
    fn view_row(&self, display_row: usize, offset: usize) -> Option<&Row> {
        let base = self.scrollback.len().checked_sub(offset)?;
        self.stream_row(base + display_row)
    }

    /// Write `cell` at (row, col) with no wide-pair or combining-mark bookkeeping.
    /// Replacing a row's final cell replaces the text that wrapped out of it, so the
    /// line stops there: the wrap link goes (autowrap sets it again if the new text
    /// wraps in its turn).
    fn set_raw(&mut self, row: usize, col: usize, cell: Cell) {
        let Some(r) = self.lines.get_mut(row) else {
            return;
        };
        let Some(slot) = r.cells.get_mut(col) else {
            return;
        };
        *slot = cell;
        if col + 1 == r.cells.len() {
            r.wrapped = false;
        }
    }

    /// Write `cell` at (row, col), first breaking any wide pair it straddles so a
    /// half-overwritten wide glyph never leaves an orphan on screen, and dropping
    /// any combining marks the overwritten cell carried.
    fn write_cell(&mut self, row: usize, col: usize, cell: Cell) {
        let existing = self.cell(row, col);
        if existing.is_wide_leader() {
            self.set_raw(row, col + 1, Cell::BLANK);
        } else if existing.is_wide_spacer() && col > 0 {
            self.set_raw(row, col - 1, Cell::BLANK);
            if let Some(r) = self.lines.get_mut(row) {
                r.clear_marks(col - 1);
            }
        }
        self.set_raw(row, col, cell);
        if let Some(r) = self.lines.get_mut(row) {
            r.clear_marks(col);
        }
    }

    /// Fill `[start_col, start_col + run.len())` on `row` with `run`'s bytes as
    /// single-width cells sharing `pen` — the bulk equivalent of one
    /// [`write_cell`](Self::write_cell) per byte, but with the row (a `VecDeque`
    /// index) resolved once instead of three times per cell. This is the ASCII
    /// print hot path, so the redundant per-cell work is what it strips.
    ///
    /// The wide-pair break only has to look at the two edges: a wide character
    /// wholly inside the run has both halves overwritten (no orphan), so only a
    /// pair the run *straddles* can leave a partner outside it — a spacer at the
    /// left edge orphans its leader to the left, a leader at the right edge orphans
    /// its spacer to the right. Combining marks under the run drop in a single
    /// retain. The `bulk_ascii_matches_per_char_under_fuzz` test pins this against
    /// the byte-at-a-time path.
    fn fill_ascii_run(&mut self, row: usize, start_col: usize, run: &[u8], pen: Pen) {
        let cols = self.cols;
        let end_col = start_col + run.len();
        let Some(r) = self.lines.get_mut(row) else {
            return;
        };
        // Left edge: overwriting a wide spacer orphans its leader one cell left.
        if start_col > 0 && r.cells.get(start_col).is_some_and(|c| c.is_wide_spacer()) {
            if let Some(slot) = r.cells.get_mut(start_col - 1) {
                *slot = Cell::BLANK;
            }
            r.clear_marks(start_col - 1);
        }
        // Right edge: overwriting a wide leader orphans its spacer one cell right.
        if end_col < cols && r.cells.get(end_col - 1).is_some_and(|c| c.is_wide_leader()) {
            if let Some(slot) = r.cells.get_mut(end_col) {
                *slot = Cell::BLANK;
            }
        }
        for (k, &byte) in run.iter().enumerate() {
            if let Some(slot) = r.cells.get_mut(start_col + k) {
                *slot = Cell {
                    rune: char::from(byte),
                    fg: pen.fg,
                    bg: pen.bg,
                    attrs: pen.attrs,
                    link: pen.link,
                };
            }
        }
        r.combining
            .retain(|m| m.col < start_col || m.col >= end_col);
        // As in `set_raw`: a run reaching the final column replaces whatever wrapped out
        // of it. The caller re-wraps this row if the run itself runs off the edge.
        if end_col >= r.cells.len() {
            r.wrapped = false;
        }
    }

    /// Fill one row segment from decoded width-1/2 scalars. Widths were already
    /// classified by [`Screen::print_text_run`], and the caller guarantees the
    /// complete segment fits on this row. Like [`Self::fill_ascii_run`], this
    /// resolves the row once, repairs only wide pairs straddling the two edges,
    /// and removes combining marks under the whole overwritten interval once.
    fn fill_text_run(
        &mut self,
        row: usize,
        start_col: usize,
        chars: &[char],
        widths: &[u8],
        pen: Pen,
    ) {
        let columns = widths
            .iter()
            .fold(0usize, |sum, width| sum.saturating_add(usize::from(*width)));
        let end_col = start_col.saturating_add(columns).min(self.cols);
        let Some(r) = self.lines.get_mut(row) else {
            return;
        };
        if start_col > 0
            && r.cells
                .get(start_col)
                .is_some_and(|cell| cell.is_wide_spacer())
        {
            if let Some(slot) = r.cells.get_mut(start_col - 1) {
                *slot = Cell::BLANK;
            }
            r.clear_marks(start_col - 1);
        }
        if end_col < r.cells.len()
            && r.cells
                .get(end_col.saturating_sub(1))
                .is_some_and(|cell| cell.is_wide_leader())
        {
            if let Some(slot) = r.cells.get_mut(end_col) {
                *slot = Cell::BLANK;
            }
        }

        let mut col = start_col;
        for (&c, &cell_width) in chars.iter().zip(widths) {
            let wide = cell_width == 2 && col.saturating_add(1) < r.cells.len();
            if let Some(slot) = r.cells.get_mut(col) {
                *slot = Cell {
                    rune: c,
                    fg: pen.fg,
                    bg: pen.bg,
                    attrs: if wide {
                        pen.attrs | Attrs::WIDE_LEADER
                    } else {
                        pen.attrs
                    },
                    link: pen.link,
                };
            }
            if wide {
                if let Some(slot) = r.cells.get_mut(col.saturating_add(1)) {
                    *slot = Cell {
                        rune: ' ',
                        fg: pen.fg,
                        bg: pen.bg,
                        attrs: pen.attrs | Attrs::WIDE_SPACER,
                        link: pen.link,
                    };
                }
            }
            col = col.saturating_add(usize::from(cell_width));
        }
        r.combining
            .retain(|mark| mark.col < start_col || mark.col >= end_col);
        if end_col >= r.cells.len() {
            r.wrapped = false;
        }
    }

    /// Scroll `[top, bottom]` up by `n`, feeding `blank` rows in at the bottom.
    /// When `to_scrollback` and the region reaches the top of the screen, the
    /// rows leaving the top are retained in scrollback; otherwise they are
    /// discarded. Either way the row storage is recycled, so a steady scroll
    /// allocates nothing once scrollback is full.
    ///
    /// Reports what left, as a [`Scrolled`]: rows that *retired into history* keep their
    /// [`AbsRow`] and pull a scrolled-back viewport along with them
    /// ([`Screen::follow_history`]); rows that were *discarded* are torn out of the
    /// middle of the stream, shifting every id below them, which ends the
    /// [`RowEpoch`]. The push count is a count of pushes, not of net growth: once the
    /// ring is full every push also evicts, and that difference is what makes a view
    /// parked at the top hold still while history slides out from under it.
    fn scroll_up_range(
        &mut self,
        top: usize,
        bottom: usize,
        n: usize,
        blank: Cell,
        to_scrollback: bool,
    ) -> Scrolled {
        let mut out = Scrolled::default();
        if top > bottom || bottom >= self.rows {
            return out;
        }
        let n = n.min(bottom - top + 1);
        for _ in 0..n {
            let Some(leaving) = self.lines.remove(top) else {
                return out;
            };
            let mut recycled = if to_scrollback && top == 0 && self.scrollback_limit > 0 {
                out.pushed += 1;
                self.scrollback.push_back(leaving);
                if self.scrollback.len() > self.scrollback_limit {
                    self.evicted += 1;
                    self.scrollback
                        .pop_front()
                        .unwrap_or_else(|| Row::filled(self.cols, blank))
                } else {
                    Row::filled(self.cols, blank)
                }
            } else {
                // Torn out of the middle of the stream: everything past the hole shifts.
                out.renumbered = true;
                leaving
            };
            recycled.reset(self.cols, blank);
            self.lines.insert(bottom, recycled);
        }
        // A retiring row keeps its slot in the stream, but the blank replacing it is
        // inserted at the region's bottom margin. Only when that is the last row of the
        // screen does it land at the end of the stream and leave the ids alone; short of
        // it, every row below the region shifts up one while sitting perfectly still.
        if out.pushed > 0 && bottom + 1 < self.rows {
            out.renumbered = true;
        }
        out
    }

    /// Scroll `[top, bottom]` down by `n`, feeding `blank` rows in at the top.
    /// Never touches scrollback (only content leaving the top of the full screen
    /// enters history).
    ///
    /// Every row it moves is [`Scrolled::discarded`]: the content slides *down* while the
    /// ids stay where they are, so the row that was `AbsRow(n)` now holds what used to be
    /// above it, and the row pushed off the bottom of the region is gone. That shifts the
    /// content out from under anything holding an id, which ends the [`RowEpoch`] — the
    /// mirror image of a region scroll up.
    fn scroll_down_range(&mut self, top: usize, bottom: usize, n: usize, blank: Cell) -> Scrolled {
        let mut out = Scrolled::default();
        if top > bottom || bottom >= self.rows {
            return out;
        }
        let n = n.min(bottom - top + 1);
        for _ in 0..n {
            let Some(leaving) = self.lines.remove(bottom) else {
                return out;
            };
            out.renumbered = true;
            let mut recycled = leaving;
            recycled.reset(self.cols, blank);
            self.lines.insert(top, recycled);
        }
        out
    }

    /// Push a row off the top into scrollback, evicting the oldest if the ring is
    /// full. A no-op when this buffer keeps no history (the alt screen), so its
    /// shrunk-away rows are simply dropped.
    fn push_history(&mut self, row: Row) {
        if self.scrollback_limit == 0 {
            return;
        }
        self.scrollback.push_back(row);
        while self.scrollback.len() > self.scrollback_limit {
            self.scrollback.pop_front();
            self.evicted += 1;
        }
    }

    /// Drop the whole history (`ED 3`). The rows are gone from the front of the stream
    /// like any eviction, so the *live* rows keep the ids they already had: an id names
    /// a line, and nothing here moved a line.
    fn clear_history(&mut self) {
        self.evicted += u64::try_from(self.scrollback.len()).unwrap_or(u64::MAX);
        self.scrollback.clear();
    }

    /// Resize the buffer to `new_cols` x `new_rows`. Columns truncate or pad every row
    /// (visible and scrollback); rows grow by pulling recent history back down onto the
    /// screen (bottom stays pinned, scrollback fills the new space above) and only pad
    /// blanks below once history runs out, and shrink by dropping rows below the cursor
    /// first, then scrolling the top into scrollback so the cursor stays on screen. The
    /// scroll region resets to the full screen (as xterm does on resize) and the cursor is
    /// re-clamped.
    ///
    /// Deliberately does NOT re-wrap soft-wrapped lines on a width change — that is
    /// [`Buffer::reflow`]'s job, called instead of this when the width changes. This is the
    /// same-width path (a height change or a scale-driven metrics change), where re-wrapping
    /// would be wasted work.
    fn resize(&mut self, new_cols: usize, new_rows: usize) {
        let new_cols = new_cols.max(1);
        let new_rows = new_rows.max(1);

        if new_cols != self.cols {
            for r in self.lines.iter_mut().chain(self.scrollback.iter_mut()) {
                r.resize_cols(new_cols);
            }
            self.cols = new_cols;
            // Tab stops are not geometry: they are state a program set, and a window
            // resize is not a program asking to lose it. `HTS` at column 3 has to still
            // be there after a drag, so the table is never rebuilt — it already covers
            // [`MAX_TAB_COLUMNS`], and only a screen wider than that needs more. Growing
            // it is the one case where fresh defaults are right: those columns have never
            // existed, so nothing has ever said anything about them.
            for c in self.tabs.len()..new_cols {
                self.tabs.push(c % TAB_WIDTH == 0);
            }
            if self.cursor.col >= new_cols {
                self.cursor.col = new_cols - 1;
            }
        }

        match new_rows.cmp(&self.rows) {
            Ordering::Greater => {
                let mut grow = new_rows - self.rows;
                // Growing pulls recent history back down onto the screen so the bottom line
                // stays pinned to the window's bottom edge and the new space reveals
                // scrollback above, as xterm and wezterm do — not blank rows shoved in below
                // the content. The stream `scrollback ++ lines` is unchanged, only its split
                // point moves, so every row keeps its id. Blank rows fill in below only once
                // history runs out.
                while grow > 0 {
                    let Some(row) = self.scrollback.pop_back() else {
                        break;
                    };
                    self.lines.push_front(row);
                    self.cursor.row += 1;
                    grow -= 1;
                }
                for _ in 0..grow {
                    self.lines.push_back(Row::filled(new_cols, Cell::BLANK));
                }
            }
            Ordering::Less => {
                let mut excess = self.rows - new_rows;
                // Rows strictly below the cursor go first (they hold nothing the
                // user is looking at); only then does the top scroll into history.
                let below = self.rows.saturating_sub(self.cursor.row + 1);
                let from_bottom = excess.min(below);
                for _ in 0..from_bottom {
                    self.lines.pop_back();
                }
                excess -= from_bottom;
                for _ in 0..excess {
                    if let Some(top) = self.lines.pop_front() {
                        self.push_history(top);
                    }
                    self.cursor.row = self.cursor.row.saturating_sub(1);
                }
            }
            Ordering::Equal => {}
        }

        self.rows = new_rows;
        self.scroll_top = 0;
        self.scroll_bottom = new_rows - 1;
        if self.cursor.row >= new_rows {
            self.cursor.row = new_rows - 1;
        }
        self.cursor.pending_wrap = false;
    }

    /// Reflow the buffer to `new_cols` x `new_rows`, re-wrapping soft-wrapped lines to the
    /// new width instead of clamping them. This is the widen-rejoins / narrow-splits
    /// behaviour: a `docker ps` line that wrapped narrow pulls back together when the
    /// window widens, and a long line breaks cleanly when it narrows. Only the primary
    /// buffer calls this; the alt screen keeps [`Buffer::resize`], because a full-screen
    /// program owns that canvas and repaints on `SIGWINCH`.
    ///
    /// The cursor and the scrolled view are preserved by recording each as a logical
    /// `(line, offset)` before the rewrap and locating it again after. `view_offset` is
    /// how far the view is scrolled up into this buffer's scrollback (0 = pinned to the
    /// live bottom); the returned value is that same view re-derived for the new geometry.
    ///
    /// ```text
    ///   unwrap                      rewrap to new_cols        re-split (cursor on screen)
    ///   physical rows ──▶ logical lines ──▶ physical rows ──▶ scrollback + live screen
    ///        (wrapped flag              (chunks of new_cols,      (bottom new_rows are
    ///         joins the runs)            wide glyph never split)   live; rest is history)
    /// ```
    ///
    /// The cursor and the scrolled view are carried across internally. The returned
    /// [`RowRemap`] carries everything anchored by [`AbsRow`] from the outside — a text
    /// selection and the OSC 133 prompt marks — by translating each old id to where its cell
    /// now sits; the caller ([`Screen::resize`]) applies it.
    ///
    /// `frozen_from` is the old stream index where the live prompt region begins (the last
    /// idle shell prompt). Rows at or below it are **clamped**, not rewrapped: the shell
    /// repaints its prompt on `SIGWINCH` and clears by its own row count, so re-wrapping a
    /// full-width prompt bar to a different number of rows would leave the extra row as
    /// uncleared residue above it. Pass the old row count to reflow everything (no prompt to
    /// protect).
    fn reflow(
        &mut self,
        new_cols: usize,
        new_rows: usize,
        view_offset: usize,
        frozen_from: usize,
    ) -> (usize, RowRemap) {
        let new_cols = new_cols.max(1);
        let new_rows = new_rows.max(1);
        let old_evicted = self.evicted;
        let old_scrollback = self.scrollback.len();

        // Anchors, as absolute indices into the current stream (`scrollback ++ lines`).
        // The cursor always sits on a live row; the view top matters only when scrolled.
        let cursor_idx = self.scrollback.len() + self.cursor.row;
        let cursor_col = self.cursor.col;
        let track_view = view_offset > 0;
        let view_idx = self.scrollback.len().saturating_sub(view_offset);

        // Drain the whole stream into one Vec so every Row is ours to recycle.
        let mut old_rows: Vec<Row> = Vec::with_capacity(self.scrollback.len() + self.lines.len());
        old_rows.extend(self.scrollback.drain(..));
        old_rows.extend(self.lines.drain(..));
        let old_count = old_rows.len();
        let frozen_from = frozen_from.min(old_count);
        // Split off the live prompt region; the head is what actually reflows.
        let frozen_rows: Vec<Row> = old_rows.split_off(frozen_from);

        // --- unwrap the reflowed head: join soft-wrapped runs into logical lines, recording
        // for every old row the logical line it joined and the offset its first column sits
        // at (`old`), so any id can later be resolved to a logical position ---
        let mut lines: Vec<LogicalLine> = Vec::new();
        let mut old: Vec<(usize, usize)> = Vec::with_capacity(old_rows.len());
        let mut cur = LogicalLine::default();
        let mut pending = false; // `cur` has rows not yet closed into `lines`
        for (idx, row) in old_rows.iter().enumerate() {
            old.push((lines.len(), cur.cells.len()));
            let base = cur.cells.len();
            cur.cells.extend_from_slice(&row.cells);
            for m in &row.combining {
                cur.combining.push(CombiningMark {
                    col: base + m.col,
                    ch: m.ch,
                });
            }
            // A soft-wrapped row whose continuation begins with a wide glyph left its last
            // column blank because the pair could not fit there (the printer's last-column
            // rule). That blank is wrap padding, not text: drop it so a widen rejoins the glyph
            // flush against the text before it instead of preserving it as a phantom space.
            if row.wrapped
                && cur.cells.last() == Some(&Cell::BLANK)
                && old_rows
                    .get(idx + 1)
                    .and_then(|next| next.cells.first())
                    .is_some_and(|c| c.is_wide_leader())
            {
                cur.cells.pop();
            }
            pending = true;
            // Close on a hard newline, or when the run reaches the cap (a forced seam).
            if !row.wrapped || cur.cells.len() >= MAX_LOGICAL_COLS {
                cur.trim_trailing_blanks();
                lines.push(std::mem::take(&mut cur));
                pending = false;
            }
        }
        // A trailing wrapped run with no terminator cannot arise (the bottom live row is
        // never wrapped past the screen), but close any remainder rather than lose it.
        if pending {
            cur.trim_trailing_blanks();
            lines.push(cur);
        }

        // Empty rows below the cursor are screen padding, not content. Drop the trailing ones
        // so the reflow lets the content rise to fill the screen from the bottom, instead of
        // stranding it above blank lines (and needlessly into scrollback) when a narrow
        // multiplies the row count. Only when nothing is frozen: with a live prompt below, the
        // cursor is down in it and the blank rows above are the shell's own spacing.
        if frozen_rows.is_empty() {
            let cursor_line = old.get(cursor_idx).map(|a| a.0).unwrap_or(0);
            let mut keep = lines.len();
            while keep > cursor_line + 1 && lines[keep - 1].cells.is_empty() {
                keep -= 1;
            }
            lines.truncate(keep);
        }

        // --- rewrap the head: lay each logical line into new_cols-wide rows, recycling the
        // drained rows, and record each line's first new index and every new row's start ---
        let mut pool = old_rows;
        let mut new_stream: Vec<Row> = Vec::with_capacity(old_count);
        let mut line_first: Vec<usize> = Vec::with_capacity(lines.len() + 1);
        let mut new_start: Vec<usize> = Vec::with_capacity(old_count);

        for line in lines.iter() {
            line_first.push(new_stream.len());
            let cells = &line.cells;
            if cells.is_empty() {
                // An empty logical line is a bare newline: it still shows one blank row.
                new_start.push(0);
                new_stream.push(take_row(&mut pool, new_cols, &[], false));
            } else {
                let mut i = 0;
                while i < cells.len() {
                    let mut end = (i + new_cols).min(cells.len());
                    // Never split a wide glyph from its spacer at the wrap: if a leader
                    // would be this row's last column with its spacer on the next, push
                    // the whole pair down (the printer's own last-column rule).
                    if end < cells.len() && cells[end - 1].is_wide_leader() {
                        end -= 1;
                    }
                    // Degenerate new_cols == 1 against a wide glyph: place it alone rather
                    // than loop forever.
                    if end <= i {
                        end = i + 1;
                    }
                    // A wide glyph whose spacer cannot share its row (only at new_cols == 1)
                    // is stored clipped: the leader alone with its WIDE_LEADER attr stripped,
                    // and its spacer dropped, exactly as the printer records a wide glyph at a
                    // single column. Leaving the leader marked would strand it without the
                    // spacer the grid's invariant demands, and copying the spacer would orphan
                    // it onto the next row.
                    let clip_wide = cells[i].is_wide_leader() && end == i + 1;
                    let next_i =
                        if clip_wide && cells.get(i + 1).is_some_and(|c| c.is_wide_spacer()) {
                            i + 2
                        } else {
                            end
                        };
                    let wrapped = next_i < cells.len();
                    let mut row = if clip_wide {
                        let mut clipped = cells[i];
                        clipped.attrs.remove(Attrs::WIDE_LEADER);
                        take_row(&mut pool, new_cols, &[clipped], wrapped)
                    } else {
                        take_row(&mut pool, new_cols, &cells[i..end], wrapped)
                    };
                    for m in &line.combining {
                        if m.col >= i && m.col < end {
                            row.combining.push(CombiningMark {
                                col: m.col - i,
                                ch: m.ch,
                            });
                        }
                    }
                    new_start.push(i);
                    new_stream.push(row);
                    i = next_i;
                }
            }
        }
        line_first.push(new_stream.len());

        // The live prompt region follows, clamped to the new width one row per row so its row
        // count is exactly what the shell's redraw expects.
        let reflowed_len = new_stream.len();
        for mut row in frozen_rows {
            row.resize_cols(new_cols);
            new_stream.push(row);
        }

        let remap = RowRemap {
            evicted: old_evicted,
            new_cols,
            lines_len: lines.len(),
            old,
            line_first,
            new_start,
            frozen_from,
            reflowed_len,
        };

        // --- re-split: the live screen is the bottom new_rows rows, the rest is history,
        // the cursor clamped into the screen ---
        let total = new_stream.len();
        let (cursor_new_idx, cursor_new_col) = remap
            .locate(cursor_idx, cursor_col)
            .unwrap_or((total.saturating_sub(1), 0));
        let view_new = track_view
            .then(|| remap.locate(view_idx, 0).map(|(i, _)| i))
            .flatten();
        // The live screen is a window of new_rows rows over the rewrapped stream. Pin it to the
        // bottom so the most recent output shows, but never above the cursor: a narrow can push
        // more rows below the cursor than fit, and the cursor must stay live because output
        // writes there. When that happens the window slides up to the cursor and the rows below
        // it that no longer fit are dropped, as a height shrink drops rows below the cursor to
        // keep it on screen.
        let live_top = total.saturating_sub(new_rows).min(cursor_new_idx);
        let live_end = live_top.saturating_add(new_rows).min(total);

        let mut lines_out: VecDeque<Row> = VecDeque::with_capacity(new_rows);
        let mut scrollback_out: VecDeque<Row> = VecDeque::with_capacity(live_top);
        for (idx, row) in new_stream.into_iter().enumerate() {
            if idx < live_top {
                scrollback_out.push_back(row);
            } else if idx < live_end {
                lines_out.push_back(row);
            }
            // idx >= live_end: a row below the cursor the shrunk screen cannot hold; dropped.
        }
        while lines_out.len() < new_rows {
            lines_out.push_back(take_row(&mut pool, new_cols, &[], false));
        }

        self.scrollback = scrollback_out;
        self.lines = lines_out;
        self.cols = new_cols;
        self.rows = new_rows;
        self.cursor.row = cursor_new_idx.saturating_sub(live_top).min(new_rows - 1);
        self.cursor.col = cursor_new_col.min(new_cols - 1);
        // Preserve pending-wrap only if the cursor still sits in the last column; anywhere
        // else there is now room, so the deferred wrap no longer applies.
        self.cursor.pending_wrap = self.cursor.pending_wrap && self.cursor.col == new_cols - 1;
        // Carry the DECSC saved cursor across the reflow the same way. This is the piece
        // alacritty leaves out (it only clamps the column) and the reason a shell that saves
        // its cursor at the prompt and restores it on `SIGWINCH` — the shell-integration path
        // — lands its redraw right instead of on a stale row. See `Screen::resize`.
        if let Some(saved) = self.saved.as_mut() {
            let old_idx = old_scrollback + saved.cursor.row;
            match remap.locate(old_idx, saved.cursor.col) {
                Some((idx, col)) => {
                    saved.cursor.row = idx.saturating_sub(live_top).min(new_rows - 1);
                    saved.cursor.col = col;
                    saved.cursor.pending_wrap =
                        saved.cursor.pending_wrap && saved.cursor.col == new_cols - 1;
                }
                None => {
                    saved.cursor.row = saved.cursor.row.min(new_rows - 1);
                    saved.cursor.col = saved.cursor.col.min(new_cols - 1);
                }
            }
        }
        self.scroll_top = 0;
        self.scroll_bottom = new_rows - 1;
        // Tab stops are state a program set, not geometry: a widen only mints defaults for
        // columns that never existed, as `resize` does.
        for c in self.tabs.len()..new_cols {
            self.tabs.push(c % TAB_WIDTH == 0);
        }

        // The rewrap can multiply the row count past the ring's cap (narrowing); evict the
        // oldest, as `push_history` does, so scrollback stays bounded.
        while self.scrollback.len() > self.scrollback_limit {
            self.scrollback.pop_front();
            self.evicted = self.evicted.saturating_add(1);
        }

        // The view anchor named content by logical position; place it back at display row
        // 0. If the rewrap pulled it down into the live screen, pin to the bottom.
        let scrollback_len = self.scrollback.len();
        let new_view = match view_new {
            Some(idx) => live_top.saturating_sub(idx).min(scrollback_len),
            None => 0,
        };
        (new_view, remap)
    }

    /// Overwrite every cell of `row` with `fill` (a blank for the erases, an 'E' for
    /// DECALN), dropping its marks and its wrap link.
    fn clear_line_full(&mut self, row: usize, fill: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            r.cells.iter_mut().for_each(|c| *c = fill);
            r.combining.clear();
            r.wrapped = false;
        }
    }

    /// Blank cells `[start, end)` of `row`, leaving the rest untouched. A range that
    /// reaches the final column erases the text that wrapped, so the line ends here.
    ///
    /// The range grows over a wide glyph's other half at either edge, and that is not
    /// politeness — it is the only answer that can be drawn. A wide glyph is a leader
    /// carrying the rune plus a spacer holding its second column, and neither half means
    /// anything alone:
    ///
    /// ```text
    ///   erase just the leader        erase just the spacer
    ///   ┌───┬───┬───┐                ┌───┬───┬───┐
    ///   │ a │   │▒▒▒│                │ a │ 一────▶│      the leader still draws two
    ///   └───┴───┴───┘                └───┴───┴───┘      columns, over the erased cell
    ///           ↑ a spacer holding
    ///             a column for a rune
    ///             that is gone
    /// ```
    ///
    /// So landing on a spacer takes its leader, and ending on a leader takes its spacer.
    fn clear_line_range(&mut self, row: usize, start: usize, end: usize, blank: Cell) {
        let Some(r) = self.lines.get_mut(row) else {
            return;
        };
        let len = r.cells.len();
        let (mut lo, mut hi) = (start, end.min(len));
        if lo >= hi {
            return;
        }
        if r.cells.get(lo).is_some_and(|c| c.is_wide_spacer()) {
            lo = lo.saturating_sub(1);
        }
        if r.cells.get(hi - 1).is_some_and(|c| c.is_wide_leader()) {
            hi = (hi + 1).min(len);
        }
        if let Some(slice) = r.cells.get_mut(lo..hi) {
            slice.iter_mut().for_each(|c| *c = blank);
        }
        r.combining.retain(|m| m.col < lo || m.col >= hi);
        if hi == len {
            r.wrapped = false;
        }
    }

    /// ICH: shift `[col, len)` right by `n`, blanking the `n` opened cells; cells
    /// pushed past the right edge are lost, including whatever wrapped out of the
    /// final column, so the wrap link goes with it.
    ///
    /// The shift cuts the row twice — at the cursor, where the move begins, and at the
    /// last cell that survives it, where the right edge eats the rest — and a wide glyph
    /// straddling either cut would lose half of itself (see [`Row::split_wide_at`]).
    fn insert_blanks(&mut self, row: usize, col: usize, n: usize, blank: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            let len = r.cells.len();
            if col >= len {
                return;
            }
            let n = n.min(len - col);
            r.split_wide_at(col, blank);
            r.split_wide_at(len - n, blank);
            r.cells.copy_within(col..len - n, col + n);
            if let Some(slice) = r.cells.get_mut(col..col + n) {
                slice.iter_mut().for_each(|c| *c = blank);
            }
            r.combining.retain(|m| m.col < col || m.col + n < len);
            r.combining.iter_mut().for_each(|m| {
                if m.col >= col {
                    m.col += n
                }
            });
            r.wrapped = false;
        }
    }

    /// DCH: shift `[col+n, len)` left by `n`, blanking the `n` cells at the right.
    /// Blanking the tail ends the line there, so the wrap link goes too.
    ///
    /// Two cuts again, mirroring ICH's: at the cursor, where the deletion begins, and at
    /// the first cell pulled in over it (see [`Row::split_wide_at`]).
    fn delete_chars(&mut self, row: usize, col: usize, n: usize, blank: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            let len = r.cells.len();
            if col >= len {
                return;
            }
            let n = n.min(len - col);
            r.split_wide_at(col, blank);
            r.split_wide_at(col + n, blank);
            r.cells.copy_within(col + n..len, col);
            if let Some(slice) = r.cells.get_mut(len - n..len) {
                slice.iter_mut().for_each(|c| *c = blank);
            }
            r.combining.retain(|m| m.col < col || m.col >= col + n);
            r.combining.iter_mut().for_each(|m| {
                if m.col >= col + n {
                    m.col -= n
                }
            });
            r.wrapped = false;
        }
    }

    fn clear_all(&mut self, fill: Cell) {
        for row in 0..self.rows {
            self.clear_line_full(row, fill);
        }
    }
}

/// A designated character set. bnkterm supports ASCII and the DEC Special
/// Graphics (line-drawing) set, which is what box-drawing TUIs rely on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Charset {
    Ascii,
    DecSpecialGraphics,
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
    /// The kitty keyboard protocol's flag stack. It is a stack because a full-screen
    /// program pushes the flags it wants on entry and pops them on exit,
    /// so it cannot strand the terminal in a mode the shell underneath it does not
    /// understand — and a program that dies without popping is cleaned up by whatever
    /// pushed beneath it. The top entry is what is in force; an empty stack is legacy.
    ///
    /// The child chooses the depth, so this is attacker-controlled and therefore capped
    /// ([`KITTY_STACK_LIMIT`]); a program in a push loop must not grow the terminal's
    /// memory without bound.
    kitty_stack: Vec<KittyFlags>,
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
            kitty_stack: Vec::new(),
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

    // ---- accessors ----------------------------------------------------------

    pub fn dimensions(&self) -> (usize, usize) {
        let b = self.active();
        (b.cols, b.rows)
    }

    pub fn is_alt(&self) -> bool {
        self.on_alt
    }

    /// The cursor position as `(row, col)`, 0-based.
    pub fn cursor(&self) -> (usize, usize) {
        let c = self.active().cursor;
        (c.row, c.col)
    }

    pub fn cursor_visible(&self) -> bool {
        self.cursor_visible
    }

    pub fn bracketed_paste(&self) -> bool {
        self.bracketed_paste
    }

    /// The mouse reporting the child asked for (off when it wants none).
    pub fn mouse_mode(&self) -> MouseMode {
        self.mouse
    }

    /// The cursor shape the child selected via `DECSCUSR` (default block).
    pub fn cursor_style(&self) -> CursorStyle {
        self.cursor_appearance.style
    }

    /// Whether the child asked the cursor to blink (default false; see
    /// [`CursorAppearance`]).
    pub fn cursor_blinks(&self) -> bool {
        self.cursor_appearance.blink
    }

    /// The bytes queued to write back to the child (DA/DSR answers). Empty in the
    /// common case (no query since the last parse). Borrowed rather than taken so the
    /// caller writes it straight through and then [`clear_responses`](Self::clear_responses)
    /// in place, keeping the queue's capacity instead of leaving a zero-cap vec the
    /// next reply must regrow.
    pub fn responses(&self) -> &[u8] {
        &self.responses
    }

    /// Drop the queued reply bytes after they have been written, retaining capacity.
    pub fn clear_responses(&mut self) {
        self.responses.clear();
    }

    /// Take the queued reply bytes, leaving the queue empty. A read-and-clear
    /// convenience for tests; the app writes via [`responses`](Self::responses) and
    /// [`clear_responses`](Self::clear_responses) instead, to keep the capacity.
    #[cfg(test)]
    pub fn take_responses(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.responses)
    }

    pub fn app_cursor_keys(&self) -> bool {
        self.app_cursor_keys
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    /// The cell at visible `(row, col)`; a blank cell if out of range.
    pub fn cell(&self, row: usize, col: usize) -> Cell {
        self.active().cell(row, col)
    }

    /// Combining marks attached to visible `(row, col)`, in arrival order; empty if
    /// none.
    pub fn marks_at(&self, row: usize, col: usize) -> impl Iterator<Item = char> + '_ {
        self.active()
            .line(row)
            .into_iter()
            .flat_map(move |r| r.marks(col))
    }

    // ---- scrollback viewport ------------------------------------------------

    /// How many lines the view is scrolled up into scrollback (0 = live bottom).
    /// Always 0 on the alt screen, which keeps no history.
    pub fn view_offset(&self) -> usize {
        if self.on_alt {
            0
        } else {
            self.view_offset
        }
    }

    /// Whether the view is currently showing history rather than the live bottom.
    pub fn is_scrolled(&self) -> bool {
        self.view_offset() > 0
    }

    /// How many lines of scrollback the primary screen holds.
    pub fn scrollback_len(&self) -> usize {
        self.primary.scrollback.len()
    }

    /// Lines of history the *view* can scroll through: the primary's scrollback, or none
    /// at all on the alt screen, which does not show it (the history is still there
    /// behind it, which is why this is not [`Self::scrollback_len`]).
    ///
    /// Every scrollbar question is answered from here, so "the alt screen does not
    /// scroll" is stated once and the extent, the position, and a thumb drop cannot
    /// disagree about it.
    fn scrollable_history(&self) -> i32 {
        if self.on_alt {
            0
        } else {
            self.scrollback_len() as i32
        }
    }

    /// What the scrollbar needs to size its thumb, in lines: how much content there is to
    /// scroll through (the history plus the screen), and how much of it is on screen.
    ///
    /// On the alt screen the content is exactly the viewport, so it reports as
    /// unscrollable — which is what hides the bar there, with no special case in the
    /// painter or the pointer.
    pub fn scroll_extent(&self) -> (i32, i32) {
        let rows = self.dimensions().1 as i32;
        (self.scrollable_history() + rows, rows)
    }

    /// How far down the whole history-plus-screen the view sits, in lines: 0 at the
    /// oldest line kept, the full history at the live bottom.
    ///
    /// This is [`Self::view_offset`] read the other way round. The offset counts lines
    /// *up* from the live bottom, because that is the direction the user scrolls back; a
    /// thumb travels *down* a track, so the scrollbar wants the complement.
    pub fn scroll_position(&self) -> i32 {
        self.scrollable_history() - self.view_offset() as i32
    }

    /// Put the view where a scrollbar thumb dropped it: `position` lines down the whole
    /// history-plus-screen, the inverse of [`Self::scroll_position`]. Clamped, so a drag
    /// flung past either end of the track rests at the oldest line or the live bottom.
    pub fn scroll_view_to(&mut self, position: i32) {
        let history = self.scrollable_history();
        self.view_offset = (history - position).clamp(0, history) as usize;
    }

    /// Scroll the view up into history by `n` lines, clamped to the top. A no-op on
    /// the alt screen.
    pub fn scroll_view_up(&mut self, n: usize) {
        if self.on_alt {
            return;
        }
        self.view_offset = (self.view_offset + n).min(self.primary.scrollback.len());
    }

    /// Scroll the view back down toward the live bottom by `n` lines.
    pub fn scroll_view_down(&mut self, n: usize) {
        self.view_offset = self.view_offset.saturating_sub(n);
    }

    /// Jump the view to the oldest scrollback line.
    pub fn scroll_view_to_top(&mut self) {
        if !self.on_alt {
            self.view_offset = self.primary.scrollback.len();
        }
    }

    /// Pin the view back to the live bottom (what fresh input does).
    pub fn scroll_view_to_bottom(&mut self) {
        self.view_offset = 0;
    }

    /// Keep a scrolled-back view anchored to the content it is showing after `pushed`
    /// rows left the screen for scrollback. The offset counts lines above the live
    /// bottom, so every row entering history moves the text under the user's eye one
    /// line further up; the offset has to grow with it or the view drifts. Output
    /// therefore never yanks the view back to the bottom: only the user does, by
    /// typing, pasting, or asking for the bottom.
    ///
    /// A view already at the live bottom stays there — `view_offset == 0` *is* what
    /// "follow the tail" means. The clamp to `scrollback.len()` is the full-ring case:
    /// history evicts from the front as fast as it grows, so a view parked at the very
    /// top holds while the oldest lines slide out from under it (as xterm does).
    fn follow_history(&mut self, pushed: usize) {
        if pushed == 0 || self.view_offset == 0 {
            return;
        }
        self.view_offset = (self.view_offset + pushed).min(self.primary.scrollback.len());
    }

    /// Settle the two things a scroll leaves behind: the viewport follows the rows that
    /// retired into history, and a scroll that renumbered the stream ends the
    /// [`RowEpoch`] (see [`Scrolled`] for both, and for why a scroll can do both at once).
    ///
    /// Every scroll goes through here, so those two rules are stated once.
    fn after_scroll(&mut self, scrolled: Scrolled) {
        self.follow_history(scrolled.pushed);
        if scrolled.renumbered {
            self.break_row_identity();
        }
    }

    // ---- absolute rows ------------------------------------------------------

    /// The regime the grid's [`AbsRow`] ids currently belong to. A holder of row ids
    /// (the selection) keeps the epoch it minted them in and drops them when this
    /// changes, rather than resolving stale ids against a renumbered grid.
    pub fn row_epoch(&self) -> RowEpoch {
        self.epoch
    }

    /// End the current identity regime: the rows the outstanding ids named are gone or
    /// renumbered. See [`RowEpoch`] for the (short) list of things that do this.
    fn break_row_identity(&mut self) {
        self.epoch = self.epoch.next();
        // The marks hold row ids, and the ids no longer name the lines they named. Keeping
        // them would mean offering to jump to a prompt that is not there any more — the
        // same reason the selection is dropped here, and the same fix.
        self.prompts.clear();
    }

    /// The id of the line currently shown at display `row`.
    ///
    /// Display row 0 sits `view_offset` lines above the live screen, so it is that far
    /// back into history; add the rows already evicted off the front and the display row
    /// itself, and the result names a line rather than a position.
    pub fn abs_row(&self, row: usize) -> AbsRow {
        let b = self.active();
        let top = b.scrollback.len().saturating_sub(self.view_offset());
        b.abs_of(top + row)
    }

    /// Where `abs` sits on screen right now, or `None` when it is not in the visible
    /// band — scrolled off above or below it, or aged out of history entirely. The
    /// painter uses the `None` to clip: a selection whose top has scrolled away still
    /// paints the part you can see.
    pub fn display_row(&self, abs: AbsRow) -> Option<usize> {
        let b = self.active();
        let top = b.scrollback.len().saturating_sub(self.view_offset());
        let idx = b.stream_index(abs)?;
        let row = idx.checked_sub(top)?;
        (row < b.rows).then_some(row)
    }

    /// Whether `abs` still names a line that exists: not yet evicted off the front of
    /// history, not past the live bottom. (A row can exist without being on screen.)
    pub fn row_exists(&self, abs: AbsRow) -> bool {
        let b = self.active();
        abs >= b.abs_of(0) && abs < b.abs_end()
    }

    /// The cell at absolute `(row, col)`, blank when the row no longer exists. This is
    /// the selection's view of the grid: it reads the content the user picked, whatever
    /// the viewport has since done.
    fn abs_cell(&self, row: AbsRow, col: usize) -> Cell {
        self.active()
            .abs_row(row)
            .and_then(|r| r.cells.get(col))
            .copied()
            .unwrap_or(Cell::BLANK)
    }

    /// Combining marks at absolute `(row, col)`.
    fn abs_marks(&self, row: AbsRow, col: usize) -> impl Iterator<Item = char> + '_ {
        self.active()
            .abs_row(row)
            .into_iter()
            .flat_map(move |r| r.marks(col))
    }

    /// The cell shown at display `(row, col)` honouring the scroll offset: the live
    /// cell when pinned to the bottom, else the scrollback row scrolled into view.
    pub fn view_cell(&self, row: usize, col: usize) -> Cell {
        let off = self.view_offset();
        if off == 0 {
            return self.active().cell(row, col);
        }
        self.active()
            .view_row(row, off)
            .and_then(|r| r.cells.get(col))
            .copied()
            .unwrap_or(Cell::BLANK)
    }

    /// Combining marks at display `(row, col)` honouring the scroll offset, in
    /// arrival order; empty if none.
    pub fn view_marks(&self, row: usize, col: usize) -> impl Iterator<Item = char> + '_ {
        self.view_line(row)
            .into_iter()
            .flat_map(move |r| r.marks(col))
    }

    /// Whether display `(row, col)` carries any combining mark, honouring the scroll
    /// offset. The ink test uses this instead of `view_marks(..).next()` so a caller
    /// can ask the yes/no without starting an iterator it will not drain.
    pub fn view_has_marks(&self, row: usize, col: usize) -> bool {
        self.view_line(row).is_some_and(|r| r.has_marks(col))
    }

    /// The row shown at display `row` honouring the scroll offset: the live row when
    /// pinned to the bottom, else the scrollback row scrolled into view.
    fn view_line(&self, row: usize) -> Option<&Row> {
        let off = self.view_offset();
        if off == 0 {
            self.active().line(row)
        } else {
            self.active().view_row(row, off)
        }
    }

    /// The text of a linear selection, `a`..`b` inclusive in absolute `(row, col)`
    /// cells (either order). Rows join with `\n`, except a soft-wrapped row joins with
    /// nothing (the two rows are one logical line, so a selection across a wrap copies
    /// as unbroken text). Trailing blanks on a row are dropped, wide spacers skipped,
    /// combining marks kept.
    ///
    /// Absolute rows, so it copies the lines the user picked no matter where the
    /// viewport has drifted to since — including lines that have scrolled out of sight
    /// entirely. Rows that have aged out of history read blank.
    pub fn selection_text(&self, a: (AbsRow, usize), b: (AbsRow, usize)) -> String {
        let (start, end) = if a <= b { (a, b) } else { (b, a) };
        let cols = self.dimensions().0;
        let last_col = cols.saturating_sub(1);
        let buf = self.active();
        // Clip to the rows that still exist: the head of an old selection may have aged
        // out of history, and its tail cannot run past the live bottom.
        let start_row = start.0.max(buf.abs_of(0));
        let Some(end_row) = buf.abs_end().prev().map(|last| end.0.min(last)) else {
            return String::new();
        };
        let mut out = String::new();
        let mut row = start_row;
        while row <= end_row {
            let first = if row == start.0 { start.1 } else { 0 };
            let last = if row == end.0 { end.1 } else { last_col };
            // Append this row straight onto the shared buffer rather than into a fresh
            // per-row `String`, then drop its trailing spaces in place — no further back
            // than where the row began, so an earlier row's content is untouched.
            let row_start = out.len();
            for col in first..=last.min(last_col) {
                let cell = self.abs_cell(row, col);
                if cell.is_wide_spacer() {
                    continue;
                }
                out.push(cell.rune);
                out.extend(self.abs_marks(row, col));
            }
            while out.len() > row_start && out.ends_with(' ') {
                out.pop();
            }
            if row != end_row {
                // A soft wrap continues the same logical line: no newline.
                if !self.wraps(row) {
                    out.push('\n');
                }
            }
            let Some(next) = row.next() else { break };
            row = next;
        }
        out
    }

    /// Whether absolute `row` soft-wrapped into the next one, so the two are one
    /// logical line. False for a row that has aged out of history.
    fn wraps(&self, row: AbsRow) -> bool {
        self.active().abs_row(row).is_some_and(|r| r.wrapped)
    }

    /// Whether the line on the live screen at display `row` soft-wrapped into the next.
    fn row_wraps(&self, row: usize) -> bool {
        self.active().line(row).is_some_and(|r| r.wrapped)
    }

    /// The inclusive cell range of the word at display `(row, col)`, for a
    /// double-click selection. A word is a maximal run of non-boundary runes (see
    /// [`is_word_boundary`]); a wide glyph's spacer continues its leader's word.
    /// Clicking a boundary cell (whitespace, a bracket) selects just that cell.
    pub fn word_at(&self, row: AbsRow, col: usize) -> ((AbsRow, usize), (AbsRow, usize)) {
        let cols = self.dimensions().0;
        if col >= cols {
            return ((row, col), (row, col));
        }
        let is_word = |c: usize| {
            let cell = self.abs_cell(row, c);
            cell.is_wide_spacer() || !is_word_boundary(cell.rune)
        };
        if !is_word(col) {
            return ((row, col), (row, col));
        }
        let mut start = col;
        while start > 0 && is_word(start - 1) {
            start -= 1;
        }
        let mut end = col;
        while end + 1 < cols && is_word(end + 1) {
            end += 1;
        }
        ((row, start), (row, end))
    }

    /// The inclusive cell range of the whole logical line at absolute `row`, for a
    /// triple-click selection: it spans every row a soft wrap joined (the `WRAPPED` flag
    /// on a row's last cell continues it into the next), edge to edge.
    ///
    /// The walk runs over the stream, not the visible band, so triple-clicking the first
    /// row on screen still takes the part of the line that wrapped in from above it.
    pub fn line_at(&self, row: AbsRow) -> ((AbsRow, usize), (AbsRow, usize)) {
        let cols = self.dimensions().0;
        let last_col = cols.saturating_sub(1);
        let b = self.active();
        let (first_row, end_row) = (b.abs_of(0), b.abs_end());
        let mut start = row;
        while let Some(above) = start.prev() {
            if start <= first_row || !self.wraps(above) {
                break;
            }
            start = above;
        }
        let mut end = row;
        while let Some(below) = end.next() {
            if below >= end_row || !self.wraps(end) {
                break;
            }
            end = below;
        }
        ((start, 0), (end, last_col))
    }

    /// The display rows the logical line at display `row` spans, clipped to the visible
    /// band. A hyperlink is a *hover* concern — re-probed on every pointer motion, never
    /// stored across output, and underlined row by row on screen — so unlike a selection
    /// it works in display rows and stops at the edge of what is drawn, even where the
    /// logical line runs on past it. The clamps are safe because `row` is on screen and
    /// the line contains it: a bound that does not resolve is off the band on that side.
    fn line_rows_on_screen(&self, row: usize) -> (usize, usize) {
        let rows = self.dimensions().1;
        let ((first, _), (last, _)) = self.line_at(self.abs_row(row));
        (
            self.display_row(first).unwrap_or(0),
            self.display_row(last).unwrap_or(rows.saturating_sub(1)),
        )
    }

    /// The hyperlink under display `(row, col)`: the inclusive cell range its text
    /// occupies, in reading order, with the URL left in `probe` for
    /// [`LinkProbe::url`] to read. `None` when that cell is not inside one.
    ///
    /// The scan runs over the whole *logical* line (soft wraps joined, the same walk
    /// [`Self::line_at`] does), because a URL printed near the right margin routinely
    /// continues on the next display row, and half a URL is not a URL. So the range
    /// returned is contiguous in reading order and may span rows — the first row from
    /// its start column to the right edge, then whole rows, then the last row to its
    /// end column — which is exactly the geometry the painter already highlights for a
    /// selection.
    ///
    /// Wide glyphs are handled at both ends: probing a spacer probes its leader (the
    /// right half of a character is the character), and a range ending on a leader is
    /// widened over its spacer, so the underline never stops half a glyph short.
    ///
    /// An OSC 8 anchor short-circuits all of that. When the cell carries a [`LinkId`],
    /// the child has *told* us these cells are this link, so there is nothing to infer:
    /// the extent is the run of cells sharing the id and the URL is the one it interned.
    /// The text scan is only ever the fallback for output that was never marked up.
    pub fn link_at(
        &self,
        row: usize,
        col: usize,
        probe: &mut LinkProbe,
    ) -> Option<((usize, usize), (usize, usize))> {
        let (cols, _) = self.dimensions();
        if col >= cols {
            return None;
        }
        // The right half of a wide glyph belongs to the glyph on its left.
        let col = if self.view_cell(row, col).is_wide_spacer() {
            col.saturating_sub(1)
        } else {
            col
        };

        probe.text.clear();
        probe.runes.clear();
        probe.url.clear();

        let id = self.view_cell(row, col).link;
        if id.is_set() {
            probe.url.push_str(self.links.url(id)?);
            return Some(self.anchor_span(row, col, id));
        }
        // Byte offset of the probed cell's rune, recorded as that rune is pushed.
        let mut at = None;
        let (first, last) = self.line_rows_on_screen(row);
        for r in first..=last {
            for c in 0..cols {
                let cell = self.view_cell(r, c);
                if cell.is_wide_spacer() {
                    continue;
                }
                if (r, c) == (row, col) {
                    at = Some(probe.text.len());
                }
                probe.runes.push((probe.text.len(), (r, c)));
                probe.text.push(cell.rune);
                probe.text.extend(self.view_marks(r, c));
            }
        }

        let found = crate::platform::link::find_at(&probe.text, at?)?;
        // `found.end` is exclusive, so the last rune inside the link is the one
        // covering the byte before it.
        let start = probe.cell_at(found.start)?;
        let mut end = probe.cell_at(found.end.checked_sub(1)?)?;
        probe.url.push_str(probe.text.get(found)?);
        if self.view_cell(end.0, end.1).is_wide_leader() && end.1 + 1 < cols {
            end.1 += 1;
        }
        Some((start, end))
    }

    /// The extent of the OSC 8 anchor `id` around `(row, col)`: the unbroken run of
    /// cells carrying that same id, walked in reading order across the whole logical
    /// line, so an anchor that soft-wraps at the margin underlines whole.
    ///
    /// The run — rather than "every cell with this id" — is what makes two anchors that
    /// happen to share a URL (and so an id) stay two links: the plain cells between
    /// them break the run. It is also why an ICH-opened gap splits an anchor, which is
    /// the honest reading once the label is no longer contiguous.
    ///
    /// ```text
    ///        col 0                     cols-1
    ///   row  ┌───────────────────────────────┐
    ///    r   │ $ ls  [docs····················│  ← id 7 starts, hits the margin
    ///   r+1  │····/spec.md]  README.md       │  ← id 7 continues, then plain cells
    ///        └───────────────────────────────┘
    ///          the span runs start → end in reading order, exactly the geometry
    ///          `push_hover_rule` already knows how to underline row by row.
    /// ```
    fn anchor_span(&self, row: usize, col: usize, id: LinkId) -> ((usize, usize), (usize, usize)) {
        let (cols, _) = self.dimensions();
        let (first, last) = self.line_rows_on_screen(row);
        // Walk the logical line as one flat sequence of cells, so crossing a soft wrap
        // needs no special case: cell `n` is at (first + n / cols, n % cols).
        let at = |n: usize| self.view_cell(first + n / cols, n % cols);
        let cell_of = |n: usize| (first + n / cols, n % cols);
        let total = (last - first + 1) * cols;
        let here = (row - first) * cols + col;

        let mut start = here;
        while start > 0 && at(start - 1).link == id {
            start -= 1;
        }
        let mut end = here;
        while end + 1 < total && at(end + 1).link == id {
            end += 1;
        }
        (cell_of(start), cell_of(end))
    }

    /// A visible row as text: base runes with their combining marks, wide spacers
    /// skipped. For selection/copy later, and for readable test assertions.
    pub fn row_string(&self, row: usize) -> String {
        let b = self.active();
        let Some(r) = b.line(row) else {
            return String::new();
        };
        let mut s = String::new();
        for (col, cell) in r.cells.iter().enumerate() {
            if cell.is_wide_spacer() {
                continue;
            }
            s.push(cell.rune);
            s.extend(r.marks(col));
        }
        s
    }

    /// The whole visible screen as text, one row per line, trailing blanks
    /// trimmed. A debugging and golden-diff convenience.
    pub fn dump(&self) -> String {
        let (_, rows) = self.dimensions();
        (0..rows)
            .map(|r| {
                let mut line = self.row_string(r);
                let trimmed = line.trim_end();
                line.truncate(trimmed.len());
                line
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A complete, human-diffable snapshot of the terminal state: dimensions,
    /// cursor, mode flags, the visible grid as text, and every non-default cell's
    /// colors and attributes. The golden-oracle tests assert this field-for-field,
    /// so a regression reads like a screenshot with a style key beneath it. A
    /// blank default cell is the omission: it shows as a space in `grid:` and
    /// carries no `style:` line.
    ///
    /// A row that soft-wrapped into the next is flagged `wrapped` beside its text.
    /// That is not cosmetic: the flag is what a copy across the wrap depends on, and
    /// it is invisible in the grid text (both a soft wrap and a hard newline just end
    /// the row), so without it a regression could not be seen here at all.
    pub fn snapshot(&self) -> String {
        use std::fmt::Write;
        let (cols, rows) = self.dimensions();
        let (cr, cc) = self.cursor();
        let mut out = String::new();
        let _ = writeln!(out, "size {cols}x{rows}");
        let _ = writeln!(out, "cursor {cr},{cc}");
        let _ = writeln!(
            out,
            "flags alt={} autowrap={} origin={} insert={} cursor_vis={} \
             bracketed={} appcursor={} keypad={}",
            self.on_alt,
            self.autowrap,
            self.origin_mode,
            self.insert_mode,
            self.cursor_visible,
            self.bracketed_paste,
            self.app_cursor_keys,
            self.keypad_app,
        );
        let _ = writeln!(out, "title {:?}", self.title);
        out.push_str("grid:\n");
        for r in 0..rows {
            let cont = if self.row_wraps(r) { " wrapped" } else { "" };
            let _ = writeln!(out, "|{}|{cont}", self.row_string(r));
        }
        out.push_str("style:\n");
        for r in 0..rows {
            for c in 0..cols {
                let cell = self.cell(r, c);
                if cell.fg != Color::Default || cell.bg != Color::Default || !cell.attrs.is_empty()
                {
                    let _ = writeln!(
                        out,
                        "r{r}c{c} {:?} fg={:?} bg={:?} {:?}",
                        cell.rune, cell.fg, cell.bg, cell.attrs
                    );
                }
            }
        }
        // What we said back. Half of what a terminal *does* is answer questions, and none
        // of it shows up on the grid — a wrong reply to `OSC 11` leaves the screen
        // pixel-identical and the editor's colours wrong. So a golden test that cannot
        // see the replies cannot see half the terminal. Only emitted when there are any,
        // so a fixture that asks nothing looks exactly as it did before.
        if !self.responses.is_empty() {
            out.push_str("replies:\n");
            for reply in split_replies(&self.responses) {
                let _ = writeln!(out, "{}", escape_reply(reply));
            }
        }
        out
    }

    // ---- internal helpers ---------------------------------------------------

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

    // ---- printing -----------------------------------------------------------

    /// Print one scalar value at the cursor, applying wide-character and wrap
    /// semantics. Combining marks (width 0) attach to the preceding cell.
    pub fn print(&mut self, c: char) {
        // Under `?2027` a scalar that continues the cluster in the cell behind us joins
        // it, whatever its own width — that is the whole difference between counting
        // scalars and counting characters. The rocket of an astronaut is two columns wide
        // on its own and no columns wide as part of the astronaut.
        if self.grapheme_clustering && self.extend_cluster(c) {
            return;
        }
        self.print_width(c, usize::from(width(c)));
    }

    /// Place one scalar whose width is already known. Decoded batches use this
    /// for complex fallbacks so width-table searches are never repeated.
    fn print_width(&mut self, c: char, cw: usize) {
        if cw == 0 {
            self.put_combining(c);
            return;
        }
        let cols = self.active().cols;
        if cols == 0 {
            return;
        }
        // Deferred wrap: a prior glyph filled the last column (xterm last-column
        // rule); perform the wrap now, before placing this glyph.
        if self.active().cursor.pending_wrap {
            self.wrap_line();
        }
        // A wide glyph with a single column left cannot fit.
        if cw == 2 && self.active().cursor.col + 1 >= cols {
            if self.autowrap {
                self.wrap_line();
            } else {
                // No autowrap: back up so it overwrites the last two columns.
                self.active_mut().cursor.col = cols.saturating_sub(2);
            }
        }

        let pen = self.pen;
        let insert = self.insert_mode;
        let (row, col) = {
            let cur = self.active().cursor;
            (cur.row, cur.col)
        };
        if insert {
            let blank = self.blank_cell();
            self.active_mut().insert_blanks(row, col, cw, blank);
        }
        // A wide glyph needs two columns and there is exactly one screen narrow enough to
        // deny it: a single-column grid, where the wrap has nowhere to go and the back-up
        // for no-autowrap lands on the same cell. Marking it a leader there would set the
        // one invariant this grid keeps — a leader always has its spacer — against a
        // spacer that cannot exist. It goes in as an ordinary cell instead: clipped, and
        // structurally sound.
        let fits_wide = cw == 2 && col + 1 < cols;
        let leader = Cell {
            rune: c,
            fg: pen.fg,
            bg: pen.bg,
            attrs: if fits_wide {
                pen.attrs | Attrs::WIDE_LEADER
            } else {
                pen.attrs
            },
            link: pen.link,
        };
        {
            let b = self.active_mut();
            b.write_cell(row, col, leader);
            if fits_wide {
                // The spacer carries the link too, so the run the hover probe walks
                // never breaks in the middle of a wide glyph.
                let spacer = Cell {
                    rune: ' ',
                    fg: pen.fg,
                    bg: pen.bg,
                    attrs: pen.attrs | Attrs::WIDE_SPACER,
                    link: pen.link,
                };
                b.write_cell(row, col + 1, spacer);
            }
        }

        let autowrap = self.autowrap;
        let b = self.active_mut();
        let cols = b.cols;
        if b.cursor.col + cw >= cols {
            b.cursor.col = cols.saturating_sub(1);
            b.cursor.pending_wrap = autowrap;
        } else {
            b.cursor.col += cw;
            b.cursor.pending_wrap = false;
        }
        self.last_printed = Some(c);
        if self.grapheme_clustering {
            self.set_cluster_anchor(row, col);
        }
    }

    /// Remember the cell a continuing scalar would join.
    ///
    /// `#[inline(never)]`, and that is not a style choice. `print` is the per-character hot
    /// path, and inlining this into it grew the function past whatever threshold the
    /// compiler lays code out around: the escape-heavy benchmark lost 10% to it *while the
    /// mode was switched off and this never ran once*. Code that cannot execute still costs
    /// what it displaces.
    #[inline(never)]
    fn set_cluster_anchor(&mut self, row: usize, col: usize) {
        let cursor = self.active().cursor;
        self.cluster_anchor = Some(ClusterAnchor {
            row,
            col,
            after: (cursor.row, cursor.col),
        });
    }

    /// Bulk-write a run of printable ASCII (each width 1) at the cursor. Equivalent
    /// to calling [`print`](Self::print) once per byte, but hoisting the per-char
    /// width lookup, wrap check, and cursor math out of the inner loop: a screenful
    /// of plain text becomes a few row-fills instead of thousands of single prints.
    ///
    /// Caller guarantees (upheld by `<Screen as Perform>::print_ascii`): every byte
    /// is `0x20..=0x7e`, the active charset is identity ASCII, and insert mode is
    /// off, so no per-char glyph mapping, wide-cell, or shift handling is needed.
    /// `write_cell` still runs per cell, so wide-pair and combining-mark cleanup at
    /// the run's edges is preserved.
    fn print_ascii_run(&mut self, bytes: &[u8]) {
        let cols = self.active().cols;
        if cols == 0 {
            return;
        }
        let pen = self.pen;
        let autowrap = self.autowrap;

        let mut rest = bytes;
        while !rest.is_empty() {
            // Take any deferred wrap the previous cell left before placing more.
            if self.active().cursor.pending_wrap {
                self.wrap_line();
            }
            let (row, start_col) = {
                let c = self.active().cursor;
                (c.row, c.col)
            };
            // Fill to the end of the row, or until the run ends. `room >= 1`: the
            // cursor column is always `< cols`, and a wrap just reset it to 0.
            let room = cols - start_col;
            let take = room.min(rest.len());
            let (run, tail) = rest.split_at(take);
            self.active_mut().fill_ascii_run(row, start_col, run, pen);
            rest = tail;
            // Advance the cursor exactly as the per-char path would after `take`
            // cells: park at the last column with a deferred wrap when the row filled.
            let end_col = start_col + take;
            let b = self.active_mut();
            if end_col >= cols {
                b.cursor.col = cols - 1;
                b.cursor.pending_wrap = autowrap;
            } else {
                b.cursor.col = end_col;
                b.cursor.pending_wrap = false;
            }
        }
        // REP repeats the last character printed, and the bulk path prints too.
        self.last_printed = bytes.last().map(|&b| char::from(b));
    }

    /// Consume already-decoded text in bounded pieces, classifying every scalar
    /// once and writing maximal positive-width prefixes a row at a time. Width-0
    /// marks and awkward right-edge wide glyphs go through [`Self::print`], which
    /// remains the semantic oracle for complex placement.
    ///
    /// The scratch widths are fixed and initialized: no allocation and no
    /// uninitialized-memory `unsafe`. Its capacity is only a work quantum;
    /// callers may supply an arbitrarily long slice without changing semantics.
    fn print_text_run(&mut self, chars: &[char]) {
        const BATCH: usize = 128;

        let mut rest = chars;
        let mut widths = [0u8; BATCH];
        while !rest.is_empty() {
            let take = rest.len().min(BATCH);
            let Some(chunk) = rest.get(..take) else {
                return;
            };
            let Some(chunk_widths) = widths.get_mut(..take) else {
                return;
            };
            let Some(&first) = chunk.first() else {
                return;
            };
            let first_width = width(first);
            if first_width == 0 {
                self.print_scalar_partitioned_run(chunk);
                rest = rest.get(take..).unwrap_or_default();
                continue;
            }
            if let Some(slot) = chunk_widths.first_mut() {
                *slot = first_width;
            }
            for (slot, &c) in chunk_widths.iter_mut().skip(1).zip(chunk.iter().skip(1)) {
                *slot = width(c);
            }
            let zero_width = chunk_widths
                .iter()
                .filter(|&&cell_width| cell_width == 0)
                .count();
            if zero_width.saturating_mul(2) >= chunk.len() {
                self.print_scalar_partitioned_run(chunk);
            } else {
                self.print_classified_run(chunk, chunk_widths);
            }
            rest = rest.get(take..).unwrap_or_default();
        }
    }

    /// Preserve the established shape for a run dominated by complex marks:
    /// non-ASCII scalars use the scalar oracle, while each printable ASCII scalar
    /// retains the same byte-style callback the parser used before batching.
    /// Mark-heavy text normally alternates one base with one or more marks, so
    /// collecting spans here only adds a second scan and a scratch copy.
    fn print_scalar_partitioned_run(&mut self, chars: &[char]) {
        for &c in chars {
            if c.is_ascii() {
                let byte = u8::try_from(u32::from(c)).unwrap_or(b'?');
                self.print_ascii_run(std::slice::from_ref(&byte));
            } else {
                self.print_width(c, usize::from(width(c)));
            }
        }
    }

    fn print_classified_run(&mut self, chars: &[char], widths: &[u8]) {
        let mut at = 0usize;
        while let (Some(&c), Some(&classified)) = (chars.get(at), widths.get(at)) {
            if classified == 0 {
                self.print_width(c, 0);
                at += 1;
                continue;
            }
            let cols = self.active().cols;
            if cols == 0 {
                return;
            }
            if self.active().cursor.pending_wrap {
                self.wrap_line();
            }
            let (row, start_col) = {
                let cursor = self.active().cursor;
                (cursor.row, cursor.col)
            };
            let room = cols.saturating_sub(start_col);
            let mut end = at;
            let mut columns = 0usize;
            while let Some(&cell_width) = widths.get(end) {
                if cell_width == 0 {
                    break;
                }
                let next = columns.saturating_add(usize::from(cell_width));
                if next > room {
                    break;
                }
                columns = next;
                end += 1;
            }

            // A two-cell glyph with one column remaining needs the full scalar
            // edge policy (wrap first, or back up under no-autowrap).
            if end == at {
                self.print_width(c, usize::from(classified));
                at += 1;
                continue;
            }

            let Some(segment) = chars.get(at..end) else {
                return;
            };
            let Some(segment_widths) = widths.get(at..end) else {
                return;
            };
            let pen = self.pen;
            self.active_mut()
                .fill_text_run(row, start_col, segment, segment_widths, pen);

            let end_col = start_col.saturating_add(columns);
            let autowrap = self.autowrap;
            let buffer = self.active_mut();
            if end_col >= cols {
                buffer.cursor.col = cols.saturating_sub(1);
                buffer.cursor.pending_wrap = autowrap;
            } else {
                buffer.cursor.col = end_col;
                buffer.cursor.pending_wrap = false;
            }
            self.last_printed = segment.last().copied();
            at = end;
        }
    }

    /// Try to join `c` onto the grapheme cluster already in the cell behind the cursor.
    /// Returns whether it did, in which case the caller has nothing left to do.
    ///
    /// This is where `?2027` actually lives. Everything else about the mode is
    /// bookkeeping; the decision is here, and it is made one scalar at a time because that
    /// is how a terminal receives them. There is no lookahead: when the woman arrives we
    /// do not know whether a ZWJ and a rocket are coming, so she is printed as herself,
    /// and the ZWJ and the rocket *join her* when they turn up. A cluster is never held
    /// back waiting to see whether it is finished — a program that writes half an emoji and
    /// crashes must still leave half an emoji on screen.
    ///
    /// The anchor carries where the cursor was when the cluster last grew, and that one
    /// check is the *whole* invalidation rule. If the cursor is not where the cluster left
    /// it, the cluster is over.
    ///
    /// Nothing else needs to say so. A control byte moves the cursor; a run of ASCII moves
    /// the cursor; a cursor motion sequence moves the cursor — so every one of them breaks
    /// the anchor by construction, and none of them needs a hook here. The hooks were
    /// written first and then measured: they sat in `execute` and in the bulk-ASCII print
    /// run, which are the two hottest functions in the terminal, and cost the
    /// escape-heavy benchmark 7.5% *while the mode was switched off*. They were also
    /// redundant. Both facts point the same way.
    #[inline(never)]
    fn extend_cluster(&mut self, c: char) -> bool {
        if !self.grapheme_clustering {
            return false;
        }
        let cursor = self.active().cursor;
        let anchor = match self.cluster_anchor {
            Some(a) if a.after == (cursor.row, cursor.col) => Some(a),
            // The cursor moved, or there is nothing to continue: whatever run of text we
            // were in has ended, and this scalar starts a fresh cluster.
            _ => {
                self.cluster.reset();
                self.cluster_anchor = None;
                None
            }
        };
        // The machine is fed every scalar, whether it joins or starts something.
        let joins = !self.cluster.breaks_before(c);
        let Some(anchor) = anchor else {
            return false;
        };
        if !joins {
            return false;
        }
        self.grow_cluster(anchor, c);
        true
    }

    /// Add `c` to the cluster at `anchor`, and widen the cell if the cluster has grown
    /// from one column to two.
    ///
    /// The widening is the fiddly half, and it is unavoidable: a cluster's width is not
    /// known until it ends. `☀` is one column, and `☀️` — the very same character with an
    /// emoji presentation selector after it — is two. The base was already placed in a
    /// narrow cell by the time the selector arrived, so the cell has to grow under it. The
    /// same goes for a flag: a regional indicator is narrow on its own and a pair of them
    /// is an emoji.
    ///
    /// At the right margin there is nowhere to grow into, and the cluster stays narrow
    /// rather than wrapping: a character that has already been drawn cannot be moved to the
    /// next line without the cursor arithmetic on the far end of the pty disagreeing about
    /// where everything after it went.
    #[inline(never)]
    fn grow_cluster(&mut self, anchor: ClusterAnchor, c: char) {
        let (row, col) = (anchor.row, anchor.col);
        let cols = self.active().cols;
        let was = self.cluster_text(row, col);
        let before = crate::width::cluster_width(&was);

        if let Some(r) = self.active_mut().lines.get_mut(row) {
            r.add_mark(col, c);
        }
        let now = self.cluster_text(row, col);
        let after = crate::width::cluster_width(&now);

        // One column to two: promote the cell to a wide leader and lay a spacer beside it,
        // the same shape a wide character has had all along, so nothing downstream has to
        // learn a new one.
        if before < 2 && after == 2 && col + 1 < cols {
            let leader = self.active().cell(row, col);
            let spacer = Cell {
                rune: ' ',
                fg: leader.fg,
                bg: leader.bg,
                attrs: leader.attrs | Attrs::WIDE_SPACER,
                link: leader.link,
            };
            let b = self.active_mut();
            b.set_raw(
                row,
                col,
                Cell {
                    attrs: leader.attrs | Attrs::WIDE_LEADER,
                    ..leader
                },
            );
            b.set_raw(row, col + 1, spacer);
            b.cursor.col = (col + 2).min(cols.saturating_sub(1));
            b.cursor.pending_wrap = col + 2 >= cols;
        }
        let cursor = self.active().cursor;
        self.cluster_anchor = Some(ClusterAnchor {
            row,
            col,
            after: (cursor.row, cursor.col),
        });
        self.last_printed = Some(c);
    }

    /// The full text of the cluster in a cell: its base rune and every mark that has
    /// joined it. This is what both the width rule and the shaper are handed, so the
    /// number of columns it takes and the glyph drawn in them come from the same string.
    fn cluster_text(&self, row: usize, col: usize) -> String {
        let mut out = String::new();
        let b = self.active();
        out.push(b.cell(row, col).rune);
        if let Some(r) = b.line(row) {
            out.extend(r.marks(col));
        }
        out
    }

    /// Attach a zero-width combining mark to the base cell to the left of where
    /// the next glyph would land, composing onto a wide glyph's leader (not its
    /// spacer). Dropped only when there is no cell to attach to (column 0).
    fn put_combining(&mut self, mark: char) {
        let cols = self.active().cols;
        let cur = self.active().cursor;
        let (row, mut col) = if cur.pending_wrap {
            (cur.row, cols.saturating_sub(1))
        } else if cur.col > 0 {
            (cur.row, cur.col - 1)
        } else {
            return;
        };
        if self.active().cell(row, col).is_wide_spacer() && col > 0 {
            col -= 1;
        }
        if let Some(r) = self.active_mut().lines.get_mut(row) {
            r.add_mark(col, mark);
        }
    }

    /// The soft wrap: link the row being left to the next one, then move to the
    /// start of that next line (scrolling if at the bottom of the region).
    fn wrap_line(&mut self) {
        let row = self.active().cursor.row;
        if let Some(r) = self.active_mut().lines.get_mut(row) {
            r.wrapped = true;
        }
        self.active_mut().cursor.pending_wrap = false;
        self.line_feed();
        self.carriage_return();
    }

    // ---- C0 controls --------------------------------------------------------

    pub fn carriage_return(&mut self) {
        let b = self.active_mut();
        b.cursor.col = 0;
        b.cursor.pending_wrap = false;
    }

    /// LF / IND: move down one row, scrolling the region up at the bottom margin.
    pub fn line_feed(&mut self) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        b.cursor.pending_wrap = false;
        let mut scrolled = Scrolled::default();
        if b.cursor.row == b.scroll_bottom {
            let (top, bottom) = (b.scroll_top, b.scroll_bottom);
            scrolled = b.scroll_up_range(top, bottom, 1, blank, top == 0);
        } else if b.cursor.row + 1 < b.rows {
            b.cursor.row += 1;
        }
        self.after_scroll(scrolled);
    }

    /// RI: move up one row, scrolling the region down at the top margin.
    pub fn reverse_index(&mut self) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        b.cursor.pending_wrap = false;
        let mut scrolled = Scrolled::default();
        if b.cursor.row == b.scroll_top {
            let (top, bottom) = (b.scroll_top, b.scroll_bottom);
            scrolled = b.scroll_down_range(top, bottom, 1, blank);
        } else if b.cursor.row > 0 {
            b.cursor.row -= 1;
        }
        self.after_scroll(scrolled);
    }

    /// NEL: carriage return plus line feed.
    pub fn next_line(&mut self) {
        self.line_feed();
        self.carriage_return();
    }

    pub fn backspace(&mut self) {
        let b = self.active_mut();
        if b.cursor.col > 0 {
            b.cursor.col -= 1;
        }
        b.cursor.pending_wrap = false;
    }

    /// HT: advance to the next tab stop, or the last column.
    pub fn tab(&mut self) {
        let b = self.active_mut();
        let mut c = b.cursor.col + 1;
        while c < b.cols && !b.tabs.get(c).copied().unwrap_or(false) {
            c += 1;
        }
        b.cursor.col = c.min(b.cols.saturating_sub(1));
        b.cursor.pending_wrap = false;
    }

    /// HTS: set a tab stop at the cursor column.
    pub fn set_tab_stop(&mut self) {
        let b = self.active_mut();
        let col = b.cursor.col;
        if let Some(stop) = b.tabs.get_mut(col) {
            *stop = true;
        }
    }

    /// REP (`CSI Ps b`): print the last character again, `n` times.
    ///
    /// ncurses reaches for this to paint a run of one character without sending it a
    /// thousand times, so it turns up in real output more than its obscurity suggests.
    /// It repeats the last *printed* character and nothing else: a control byte, an escape
    /// sequence, or a fresh line all leave nothing to repeat, and then it does nothing —
    /// which is the behaviour a program relies on when it uses REP right after a newline
    /// and expects no output rather than a screenful of the last thing on the line above.
    pub fn repeat_last(&mut self, n: usize) {
        let Some(c) = self.last_printed else {
            return;
        };
        for _ in 0..n {
            self.print(c);
        }
    }

    /// CBT: move back `n` tab stops, stopping at column 0. The mirror of [`tab`](Self::tab).
    pub fn back_tab(&mut self, n: usize) {
        let b = self.active_mut();
        for _ in 0..n.max(1) {
            let Some(mut c) = b.cursor.col.checked_sub(1) else {
                break;
            };
            while c > 0 && !b.tabs.get(c).copied().unwrap_or(false) {
                c -= 1;
            }
            b.cursor.col = c;
        }
        b.cursor.pending_wrap = false;
    }

    /// TBC: clear the tab stop at the cursor (mode 0, the default) or every stop
    /// (mode 3).
    ///
    /// Every other parameter is ignored, which is the whole behavior and not a
    /// shortcut. ECMA-48 also defines 1, 2, 4 and 5 in terms of *line* tab stops, which
    /// no terminal in use has; xterm implements 0 and 3 and drops the rest, and vttest's
    /// tab screen checks exactly that by sending `CSI 1 g` and `CSI 2 g` at a live tab
    /// stop and requiring the stop to survive. Treating an unknown parameter as 0 would
    /// silently delete it.
    pub fn clear_tab_stop(&mut self, mode: u16) {
        let b = self.active_mut();
        match mode {
            0 => {
                let col = b.cursor.col;
                if let Some(stop) = b.tabs.get_mut(col) {
                    *stop = false;
                }
            }
            3 => b.tabs.iter_mut().for_each(|t| *t = false),
            _ => {}
        }
    }

    // ---- cursor movement ----------------------------------------------------

    pub fn move_up(&mut self, n: usize) {
        let b = self.active_mut();
        let n = n.max(1);
        let limit = if b.cursor.row >= b.scroll_top {
            b.scroll_top
        } else {
            0
        };
        b.cursor.row = b.cursor.row.saturating_sub(n).max(limit);
        b.cursor.pending_wrap = false;
    }

    pub fn move_down(&mut self, n: usize) {
        let b = self.active_mut();
        let n = n.max(1);
        let limit = if b.cursor.row <= b.scroll_bottom {
            b.scroll_bottom
        } else {
            b.rows - 1
        };
        b.cursor.row = (b.cursor.row + n).min(limit);
        b.cursor.pending_wrap = false;
    }

    pub fn move_forward(&mut self, n: usize) {
        let b = self.active_mut();
        let n = n.max(1);
        b.cursor.col = (b.cursor.col + n).min(b.cols.saturating_sub(1));
        b.cursor.pending_wrap = false;
    }

    pub fn move_back(&mut self, n: usize) {
        let b = self.active_mut();
        let n = n.max(1);
        b.cursor.col = b.cursor.col.saturating_sub(n);
        b.cursor.pending_wrap = false;
    }

    /// CUP / HVP: move to `(row, col)`, 0-based, honoring origin mode (which
    /// makes rows relative to and bounded by the scroll region).
    pub fn move_to(&mut self, row: usize, col: usize) {
        let origin = self.origin_mode;
        let b = self.active_mut();
        b.cursor.row = if origin {
            (b.scroll_top + row).min(b.scroll_bottom)
        } else {
            row.min(b.rows - 1)
        };
        b.cursor.col = col.min(b.cols - 1);
        b.cursor.pending_wrap = false;
    }

    /// CHA / HPA: move to an absolute column.
    pub fn move_to_col(&mut self, col: usize) {
        let b = self.active_mut();
        b.cursor.col = col.min(b.cols - 1);
        b.cursor.pending_wrap = false;
    }

    /// VPA: move to a row (origin-aware, like `move_to`).
    pub fn move_to_row(&mut self, row: usize) {
        let origin = self.origin_mode;
        let b = self.active_mut();
        b.cursor.row = if origin {
            (b.scroll_top + row).min(b.scroll_bottom)
        } else {
            row.min(b.rows - 1)
        };
        b.cursor.pending_wrap = false;
    }

    // ---- erase --------------------------------------------------------------

    /// EL: erase in line. 0 = cursor to end, 1 = start to cursor, 2 = whole line.
    pub fn erase_line(&mut self, mode: u16) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (row, col, cols) = (b.cursor.row, b.cursor.col, b.cols);
        let (start, end) = match mode {
            1 => (0, col + 1),
            2 => (0, cols),
            _ => (col, cols),
        };
        b.clear_line_range(row, start, end, blank);
        b.cursor.pending_wrap = false;
    }

    /// ED: erase in display. 0 = cursor to end, 1 = start to cursor, 2 = all,
    /// 3 = all plus scrollback.
    pub fn erase_display(&mut self, mode: u16) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (row, col, rows, cols) = (b.cursor.row, b.cursor.col, b.rows, b.cols);
        match mode {
            1 => {
                for r in 0..row {
                    b.clear_line_full(r, blank);
                }
                b.clear_line_range(row, 0, col + 1, blank);
            }
            2 | 3 => {
                b.clear_all(blank);
                if mode == 3 {
                    b.clear_history();
                }
            }
            _ => {
                b.clear_line_range(row, col, cols, blank);
                for r in (row + 1)..rows {
                    b.clear_line_full(r, blank);
                }
            }
        }
        b.cursor.pending_wrap = false;
        if mode == 2 || mode == 3 {
            // The display the user was looking at (and may have selected) is gone.
            self.break_row_identity();
        }
        if mode == 3 {
            // The history the view was scrolled into no longer exists, and an offset
            // pointing past the end of an empty ring is not a view of anything.
            self.view_offset = 0;
        }
    }

    /// ECH: erase `n` cells from the cursor, without shifting the rest.
    /// ECH: blank `n` cells from the cursor without shifting anything.
    ///
    /// Cancels a deferred wrap, as every operator that rewrites the cells around the
    /// cursor must: the flag is a claim that the next glyph belongs on the next row *on
    /// behalf of the text already here*, and this just erased that text. Honouring it
    /// afterwards would wrap for a glyph that no longer exists. Unconditional, and
    /// before any bounds check refuses the erase itself — a rule about the cursor does
    /// not wait on whether the cells were in range.
    pub fn erase_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        b.cursor.pending_wrap = false;
        let (row, col) = (b.cursor.row, b.cursor.col);
        b.clear_line_range(row, col, col + n.max(1), blank);
    }

    // ---- insert / delete ----------------------------------------------------

    /// ICH: insert `n` blanks at the cursor, shifting the rest of the line right.
    pub fn insert_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        // Cancels a deferred wrap: ICH shifts the text the flag was deferring for out
        // from under the cursor (see `erase_chars`).
        b.cursor.pending_wrap = false;
        let (row, col) = (b.cursor.row, b.cursor.col);
        b.insert_blanks(row, col, n.max(1), blank);
    }

    /// DCH: delete `n` cells at the cursor, shifting the rest of the line left.
    pub fn delete_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        // Cancels a deferred wrap, as ICH does and for the same reason.
        b.cursor.pending_wrap = false;
        let (row, col) = (b.cursor.row, b.cursor.col);
        b.delete_chars(row, col, n.max(1), blank);
    }

    /// IL: insert `n` blank lines at the cursor row, within the scroll region.
    pub fn insert_lines(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        if b.cursor.row < b.scroll_top || b.cursor.row > b.scroll_bottom {
            return;
        }
        let (top, bottom) = (b.cursor.row, b.scroll_bottom);
        let scrolled = b.scroll_down_range(top, bottom, n.max(1), blank);
        b.cursor.col = 0;
        b.cursor.pending_wrap = false;
        self.after_scroll(scrolled);
    }

    /// DL: delete `n` lines at the cursor row, within the scroll region.
    ///
    /// The deleted rows are discarded, never retained, even with the cursor at row 0.
    /// This is a deliberate pick in a genuine split: xterm and alacritty push them into
    /// history (xterm's `DeleteLine` saves when `cur_row == 0`), ghostty does not. DL is
    /// an editing command, not a scroll — the application is *removing* those lines, and
    /// a shell redrawing a multi-line prompt at the top of the screen should not dribble
    /// prompt fragments into the scrollback. If a real program is ever found to depend on
    /// the xterm behavior, this is the line to change.
    pub fn delete_lines(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        if b.cursor.row < b.scroll_top || b.cursor.row > b.scroll_bottom {
            return;
        }
        let (top, bottom) = (b.cursor.row, b.scroll_bottom);
        let scrolled = b.scroll_up_range(top, bottom, n.max(1), blank, false);
        b.cursor.col = 0;
        b.cursor.pending_wrap = false;
        self.after_scroll(scrolled);
    }

    // ---- scrolling ----------------------------------------------------------

    /// DECSTBM: set the scroll region to `[top, bottom]` (0-based, inclusive).
    /// An empty or inverted region resets to the full screen. Homes the cursor.
    pub fn set_scroll_region(&mut self, top: usize, bottom: usize) {
        {
            let b = self.active_mut();
            let bottom = bottom.min(b.rows - 1);
            if top < bottom {
                b.scroll_top = top;
                b.scroll_bottom = bottom;
            } else {
                b.scroll_top = 0;
                b.scroll_bottom = b.rows - 1;
            }
        }
        self.move_to(0, 0);
    }

    /// SU: scroll the region up `n` lines (content moves up, blanks feed in at the
    /// bottom). Rows leaving the *top of the screen* enter history, exactly as they do
    /// for a line feed at the bottom margin: xterm retains them whenever the top margin
    /// is 0 (`xtermScroll`'s `top_marg == 0`), and alacritty and ghostty agree. A region
    /// with a top margin discards instead — those rows never touch the top of the
    /// screen, so they were never history.
    pub fn scroll_up(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (top, bottom) = (b.scroll_top, b.scroll_bottom);
        let scrolled = b.scroll_up_range(top, bottom, n.max(1), blank, top == 0);
        self.after_scroll(scrolled);
    }

    /// SD: scroll the region down `n` lines (content moves down).
    pub fn scroll_down(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (top, bottom) = (b.scroll_top, b.scroll_bottom);
        let scrolled = b.scroll_down_range(top, bottom, n.max(1), blank);
        self.after_scroll(scrolled);
    }

    // ---- save / restore -----------------------------------------------------

    /// DECSC: save the cursor, pen, origin mode, and character set shift state.
    ///
    /// The charsets are part of the saved state by the VT spec's own list ("cursor
    /// position, graphic rendition, character set shift state, state of wrap flag,
    /// state of origin mode, state of selective erase"), and leaving them out is
    /// visible immediately: a shell that saves the cursor, switches to the line-drawing
    /// set to paint a frame, then restores, is telling the terminal to put the ASCII
    /// mapping back. Without that, every letter it prints afterwards comes out as box
    /// glyphs.
    pub fn save_cursor(&mut self) {
        let (pen, origin, charsets) = (self.pen, self.origin_mode, self.charsets());
        let b = self.active_mut();
        b.saved = Some(Saved {
            cursor: b.cursor,
            pen,
            origin,
            charsets,
        });
    }

    /// DECRC: restore what DECSC saved, or home the cursor if nothing was saved.
    pub fn restore_cursor(&mut self) {
        let saved = self.active().saved;
        match saved {
            Some(s) => {
                self.pen = s.pen;
                self.origin_mode = s.origin;
                self.set_charsets(s.charsets);
                let b = self.active_mut();
                b.cursor = s.cursor;
                b.cursor.row = b.cursor.row.min(b.rows - 1);
                b.cursor.col = b.cursor.col.min(b.cols - 1);
            }
            None => self.move_to(0, 0),
        }
    }

    fn charsets(&self) -> Charsets {
        Charsets {
            g0: self.g0,
            g1: self.g1,
            gl_is_g1: self.gl_is_g1,
        }
    }

    fn set_charsets(&mut self, c: Charsets) {
        self.g0 = c.g0;
        self.g1 = c.g1;
        self.gl_is_g1 = c.gl_is_g1;
    }

    // ---- rendition (SGR) ----------------------------------------------------

    /// Apply an SGR sequence, updating the pen. An empty parameter list is a reset
    /// (SGR 0). Handles ANSI-16, bright, 256 (`38;5;n` and `38:5:n`), and truecolor
    /// (`38;2;r;g;b` and `38:2::r:g:b`) for both foreground (38) and background (48).
    ///
    /// The rule for everything we do not implement is **consume, then ignore**, and it
    /// is load-bearing rather than pedantic. An extended-colour introducer owns the
    /// parameters that follow it, so failing to consume `58;2;255;0;0` would not merely
    /// skip an underline colour: the `2` would be read as *dim*, the `0` as *reset*, and
    /// the rest of the line would come out in the wrong style. Skipping a parameter we
    /// do not understand is only safe once we know how many belong to it.
    pub fn sgr(&mut self, params: &Params) {
        if params.is_empty() {
            self.pen.reset_rendition();
            return;
        }
        let mut i = 0;
        while i < params.len() {
            let Some(param) = params.get(i) else { break };
            let p = param.first().copied().unwrap_or(0);
            match p {
                0 => self.pen.reset_rendition(),
                1 => self.pen.attrs.insert(Attrs::BOLD),
                2 => self.pen.attrs.insert(Attrs::DIM),
                3 => self.pen.attrs.insert(Attrs::ITALIC),
                // `4` is underline; `4:n` picks a style (1 single, 2 double, 3 curly,
                // 4 dotted, 5 dashed) and `4:0` turns it off. We draw only a solid
                // underline for now, so a style other than 0 is an underline of some
                // kind and that is what the cell records. Storing *which* style needs a
                // wider `Attrs` and a cell field; until then a curly underline shows up
                // as a straight one, which is the right kind of wrong.
                // `4` underlines; `4:n` says with what shape. `4:0` is the one
                // sub-parameter of 4 that is not an underline at all.
                4 => match param.get(1).copied() {
                    Some(0) => {
                        self.pen.attrs.remove(Attrs::UNDERLINE);
                        self.pen.attrs.set_underline_style(UnderlineStyle::Single);
                    }
                    style => {
                        // An unknown style is still an underline: better a straight line
                        // where a program wanted a fancy one than no mark at all.
                        let style = match style {
                            Some(2) => UnderlineStyle::Double,
                            Some(3) => UnderlineStyle::Curly,
                            Some(4) => UnderlineStyle::Dotted,
                            Some(5) => UnderlineStyle::Dashed,
                            _ => UnderlineStyle::Single,
                        };
                        self.pen.attrs.insert(Attrs::UNDERLINE);
                        self.pen.attrs.set_underline_style(style);
                    }
                },
                7 => self.pen.attrs.insert(Attrs::REVERSE),
                8 => self.pen.attrs.insert(Attrs::HIDDEN),
                9 => self.pen.attrs.insert(Attrs::STRIKE),
                21 => self.pen.attrs.remove(Attrs::BOLD),
                22 => self.pen.attrs.remove(Attrs::BOLD | Attrs::DIM),
                23 => self.pen.attrs.remove(Attrs::ITALIC),
                24 => {
                    self.pen.attrs.remove(Attrs::UNDERLINE);
                    self.pen.attrs.set_underline_style(UnderlineStyle::Single);
                }
                27 => self.pen.attrs.remove(Attrs::REVERSE),
                28 => self.pen.attrs.remove(Attrs::HIDDEN),
                29 => self.pen.attrs.remove(Attrs::STRIKE),
                30..=37 => self.pen.fg = Color::Ansi((p - 30) as u8),
                38 => {
                    let (color, advance) = parse_ext_color(param, params, i);
                    if let Some(color) = color {
                        self.pen.fg = color;
                    }
                    i += advance;
                }
                39 => self.pen.fg = Color::Default,
                40..=47 => self.pen.bg = Color::Ansi((p - 40) as u8),
                48 => {
                    let (color, advance) = parse_ext_color(param, params, i);
                    if let Some(color) = color {
                        self.pen.bg = color;
                    }
                    i += advance;
                }
                49 => self.pen.bg = Color::Default,
                // Underline colour. We have nowhere on a cell to put it, but it takes
                // the same arguments as 38/48, so it must be consumed all the same.
                58 => {
                    let (_, advance) = parse_ext_color(param, params, i);
                    i += advance;
                }
                90..=97 => self.pen.fg = Color::Ansi((p - 90 + 8) as u8),
                100..=107 => self.pen.bg = Color::Ansi((p - 100 + 8) as u8),
                _ => {}
            }
            i += 1;
        }
    }

    // ---- modes --------------------------------------------------------------

    /// DECRQM (`CSI ? Ps $ p`, and `CSI Ps $ p` for the ANSI modes): "do you know this
    /// mode, and is it on?" Answered with DECRPM, `CSI ? Ps ; Pm $ y`.
    ///
    /// This is how anything we implement ever gets *used*. nvim asks five of these
    /// before it draws a single character, and a terminal that stays silent is told
    /// nothing and assumes nothing — so a mode we add and never report is a mode nobody
    /// will ever turn on.
    ///
    /// The values are 0 not recognised, 1 set, 2 reset, 3 permanently set, 4 permanently
    /// reset. Answering *honestly* is the whole job: a mode we do not implement must
    /// report **0**, never 2. Reporting 2 says "I know that mode, and it is currently
    /// off", which invites the program to switch it on and then depend on it. A lie here
    /// is worse than the silence it replaces.
    fn report_mode(&mut self, mode: u16, private: bool) {
        let state = match self.mode_state(mode, private) {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        };
        self.respond(if private { b"\x1b[?" } else { b"\x1b[" });
        push_decimal(&mut self.responses, u32::from(mode));
        self.responses.push(b';');
        push_decimal(&mut self.responses, state);
        self.respond(b"$y");
    }

    /// Whether we implement `mode`, and if so whether it is currently on. `None` means
    /// we do not know it, which is exactly what DECRQM needs to hear and the one thing
    /// it must never be told falsely.
    ///
    /// Every mode [`set_mode`](Self::set_mode) acts on appears here, and nothing else
    /// does. If you teach the terminal a new mode, teach this at the same time — the two
    /// lists disagreeing is how a terminal ends up claiming a feature it does not have.
    fn mode_state(&self, mode: u16, private: bool) -> Option<bool> {
        if !private {
            return match mode {
                4 => Some(self.insert_mode), // IRM
                _ => None,
            };
        }
        Some(match mode {
            1 => self.app_cursor_keys,
            6 => self.origin_mode,
            7 => self.autowrap,
            25 => self.cursor_visible,
            47 | 1047 | 1049 => self.on_alt,
            1000 => self.mouse.protocol == MouseProtocol::Press,
            1002 => self.mouse.protocol == MouseProtocol::ButtonEvent,
            1003 => self.mouse.protocol == MouseProtocol::AnyEvent,
            1006 => self.mouse.sgr,
            12 => self.cursor_appearance.blink,
            1004 => self.focus_events,
            2048 => self.in_band_resize,
            2004 => self.bracketed_paste,
            2026 => self.synchronized,
            2027 => self.grapheme_clustering,
            _ => return None,
        })
    }

    /// Set or reset a mode. `private` distinguishes the DEC private modes
    /// (`?`-prefixed, like DECAWM) from the ANSI modes (like IRM).
    pub fn set_mode(&mut self, mode: u16, private: bool, enable: bool) {
        if private {
            match mode {
                1 => self.app_cursor_keys = enable,
                6 => {
                    self.origin_mode = enable;
                    self.move_to(0, 0);
                }
                7 => self.autowrap = enable,
                25 => self.cursor_visible = enable,
                47 | 1047 => self.switch_alt(enable),
                1049 => {
                    if enable {
                        self.save_cursor();
                        self.switch_alt(true);
                    } else {
                        self.switch_alt(false);
                        self.restore_cursor();
                    }
                }
                // `?12` blinks the cursor without saying anything about its shape, which
                // is the one thing DECSCUSR cannot do: its parameters bind the two
                // together. A program that wants a blinking bar and finds a steady block
                // has to be able to say only "blink".
                12 => self.cursor_appearance.blink = enable,
                1004 => self.focus_events = enable,
                2048 => self.set_in_band_resize(enable),
                2004 => self.bracketed_paste = enable,
                2026 => self.synchronized = enable,
                2027 => {
                    self.grapheme_clustering = enable;
                    // Whatever was on screen was measured the other way; start clean.
                    self.cluster.reset();
                    self.cluster_anchor = None;
                }
                // Mouse reporting: the ?1000/?1002/?1003 levels are mutually
                // exclusive (enabling one, or disabling any, sets the level);
                // ?1006 is the orthogonal SGR encoding flag.
                1000 => self.set_mouse_protocol(MouseProtocol::Press, enable),
                1002 => self.set_mouse_protocol(MouseProtocol::ButtonEvent, enable),
                1003 => self.set_mouse_protocol(MouseProtocol::AnyEvent, enable),
                1006 => self.mouse.sgr = enable,
                _ => {}
            }
        } else if mode == 4 {
            self.insert_mode = enable;
        }
    }

    /// Set or clear a mouse-reporting level. Disabling any level turns reporting
    /// off (programs toggle a single level, so this is the common, correct case).
    fn set_mouse_protocol(&mut self, protocol: MouseProtocol, enable: bool) {
        self.mouse.protocol = if enable { protocol } else { MouseProtocol::Off };
    }

    /// DECSCUSR: the `Ps` argument selects both the shape and whether it blinks (odd
    /// blinks, even is steady). `Ps = 0` asks for the terminal's own default, so it
    /// restores bnkterm's power-on look rather than xterm's documented blinking block
    /// (see [`CursorAppearance`]). An unknown `Ps` is ignored.
    fn set_cursor_style(&mut self, ps: u16) {
        self.cursor_appearance = match ps {
            0 => CursorAppearance::default(),
            1 => CursorAppearance::new(CursorStyle::Block, true),
            2 => CursorAppearance::new(CursorStyle::Block, false),
            3 => CursorAppearance::new(CursorStyle::Underline, true),
            4 => CursorAppearance::new(CursorStyle::Underline, false),
            5 => CursorAppearance::new(CursorStyle::Bar, true),
            6 => CursorAppearance::new(CursorStyle::Bar, false),
            _ => return,
        };
    }

    /// DA (Send Device Attributes): answer a program's "what are you?" probe.
    /// Primary (`CSI c`) reports a VT100 with the Advanced Video Option, the
    /// conservative, universally-understood identity (real capabilities come from
    /// `TERM`); secondary (`CSI > c`) gives a benign version triple. Answering at
    /// all is the point: a program that queries and gets nothing can hang.
    fn device_attributes(&mut self, private: u8) {
        match private {
            0 => self.respond(b"\x1b[?1;2c"),
            b'>' => self.respond(b"\x1b[>0;0;0c"),
            _ => {}
        }
    }

    /// DSR (Device Status Report): `5 n` asks if we are OK (yes), `6 n` asks for
    /// the cursor position (CPR). The `?6 n` private form is the extended report
    /// (DECXCPR) some programs use. The position is 1-based and origin-mode aware.
    fn device_status(&mut self, params: &Params, private: u8) {
        let ps = params.value(0);
        match (private, ps) {
            (0, 5) => self.respond(b"\x1b[0n"),
            (0, 6) => {
                let (row, col) = self.report_position();
                self.respond(b"\x1b[");
                push_decimal(&mut self.responses, row);
                self.responses.push(b';');
                push_decimal(&mut self.responses, col);
                self.responses.push(b'R');
            }
            (b'?', 6) => {
                let (row, col) = self.report_position();
                self.respond(b"\x1b[?");
                push_decimal(&mut self.responses, row);
                self.responses.push(b';');
                push_decimal(&mut self.responses, col);
                self.responses.extend_from_slice(b";1R");
            }
            _ => {}
        }
    }

    /// The cursor position for a CPR, 1-based, relative to the scroll region when
    /// origin mode is on (as the report expects).
    fn report_position(&self) -> (u32, u32) {
        let (mut row, col) = self.cursor();
        if self.origin_mode {
            row = row.saturating_sub(self.active().scroll_top);
        }
        (row as u32 + 1, col as u32 + 1)
    }

    /// Queue bytes to be written back to the child.
    fn respond(&mut self, bytes: &[u8]) {
        self.responses.extend_from_slice(bytes);
    }

    /// Enter or leave the alternate screen. Entering clears it and carries the cursor
    /// across unchanged (it is one cursor shared by both buffers, see below); the
    /// primary buffer is untouched, so leaving reveals it intact.
    fn switch_alt(&mut self, enable: bool) {
        if enable == self.on_alt {
            return;
        }
        // Entering or leaving the alt screen shows a live view, never stale
        // history; the alt screen has no scrollback to scroll anyway. The rows now
        // belong to a different buffer, so no id minted against the old one survives.
        self.view_offset = 0;
        self.break_row_identity();
        if enable {
            let blank = self.blank_cell();
            // The cursor carries across the switch rather than homing: the alt screen
            // is a different set of cells, not a different cursor. `?1049h` is where
            // this shows — it saves the cursor, switches, and clears, and a program
            // that then restores expects to land where it started, not at the origin.
            // The deferred-wrap flag rides along with it for the same reason.
            let cursor = self.active().cursor;
            self.alt.clear_all(blank);
            self.alt.cursor = cursor;
            self.alt.scroll_top = 0;
            self.alt.scroll_bottom = self.alt.rows - 1;
            self.on_alt = true;
        } else {
            // And back the same way. The cursor is one cursor: xterm keeps a single
            // position across both buffers, so leaving the alt screen has to carry it
            // home as much as entering it did. Carrying it only one way would be a
            // cursor that is shared going in and per-buffer coming out, which is the
            // half-and-half model this is here to remove.
            //
            // `?1049l` restores a saved cursor immediately after this and so cannot
            // tell; `?47l` and `?1047l` do not, and are what a program using the alt
            // screen without the save/restore pair sends.
            let cursor = self.alt.cursor;
            self.primary.cursor = cursor;
            self.on_alt = false;
        }
    }

    // ---- misc ---------------------------------------------------------------

    /// RIS: hard reset to the power-on state, keeping the current dimensions.
    pub fn reset(&mut self) {
        let (cols, rows) = self.dimensions();
        let epoch = self.epoch.next();
        *self = Screen::new(cols, rows);
        // A fresh `Screen` starts at epoch zero, which would make ids minted before the
        // reset look current again. Identity moves forward across a reset, never back.
        self.epoch = epoch;
    }

    /// DECSTR (`CSI ! p`): soft reset. Puts the *settings* back to their defaults while
    /// leaving the screen, the scrollback and the cursor position alone — which is why
    /// programs reach for it instead of RIS: it recovers a sane terminal without
    /// throwing away what is on it.
    ///
    /// One deliberate departure from the DEC table, which lists autowrap as reset ("No
    /// autowrap"). **xterm leaves autowrap on**, and a terminal that stops wrapping
    /// after a soft reset breaks the very shell the program handed control back to.
    /// xterm.js once implemented the letter of the spec here and had to undo it. When
    /// the spec and xterm disagree, programs were written against xterm.
    ///
    /// The saved cursor goes too (DECSC is reset to "home position"), so a stray DECRC
    /// afterwards cannot restore a cursor from before the reset.
    pub fn soft_reset(&mut self) {
        self.pen = Pen::default();
        self.cursor_visible = true;
        self.insert_mode = false;
        self.origin_mode = false;
        self.autowrap = true; // xterm, not DEC. See above.
        self.app_cursor_keys = false;
        self.keypad_app = false;
        self.g0 = Charset::Ascii;
        self.g1 = Charset::Ascii;
        self.gl_is_g1 = false;
        let b = self.active_mut();
        b.saved = None;
        // The margins go back to the full screen, but *without* homing the cursor the
        // way an explicit DECSTBM would: a soft reset is not a cursor move.
        b.scroll_top = 0;
        b.scroll_bottom = b.rows.saturating_sub(1);
        b.cursor.pending_wrap = false;
    }

    /// Resize both screens to `cols` x `rows` (clamped to at least 1x1). The app calls
    /// this when the window's pixel size divided by the cell size yields a new grid; it
    /// then sends the child the matching `TIOCSWINSZ`.
    ///
    /// A width change re-wraps the primary screen and its scrollback ([`Buffer::reflow`]),
    /// so a widened window pulls soft-wrapped lines back together and a narrowed one breaks
    /// them; the cursor and a scrolled-up view are carried across. A height-only change
    /// keeps the cheap path — no rewrap is needed when the width is the same, and it grows by
    /// pulling history back onto the screen. The alt screen never reflows: a full-screen
    /// program owns it and repaints on `SIGWINCH`.
    ///
    /// Anchored state follows the text where it can. The primary's own OSC 133 prompt marks
    /// are translated internally; the returned [`ResizeEffect`] tells the caller how to carry
    /// its text selection (translate, keep, or drop). The row-id epoch ends on a reflow (or
    /// any alt-screen resize), so any holder that is *not* translated is still dropped by the
    /// usual prune — the translation is the addition, not a removal of that safety net.
    pub fn resize(&mut self, cols: usize, rows: usize) -> ResizeEffect {
        let cols = cols.max(1);
        let rows = rows.max(1);
        let remap = if cols == self.primary.cols {
            // Height-only (or a no-op): re-clamp rows (a grow pulls history back down) and
            // re-pin the view. The primary's ids stay put, so prompts need no translation.
            self.primary.resize(cols, rows);
            self.view_offset = 0;
            None
        } else {
            // Width changed: rewrap the primary, re-deriving the view, and translate the
            // primary's prompt marks through the reflow.
            //
            // The live prompt is left for the shell to repaint: freeze from the last idle
            // prompt mark (one with no command output yet — the input the user is sitting at)
            // down to the bottom, never below the cursor. Everything above it, including a
            // finished command's output, still reflows.
            let old_count = self.primary.scrollback.len() + self.primary.rows;
            let cursor_idx = self.primary.scrollback.len() + self.primary.cursor.row;
            let frozen_from = self
                .prompts
                .iter()
                .rev()
                .find(|p| p.output.is_none())
                .and_then(|p| self.primary.stream_index(p.row))
                .map(|idx| idx.min(cursor_idx))
                .unwrap_or(old_count);
            let view = self.view_offset();
            let (new_view, remap) = self.primary.reflow(cols, rows, view, frozen_from);
            self.view_offset = new_view;
            self.prompts.retain_mut(|p| match remap.row(p.row) {
                Some(row) => {
                    p.row = row;
                    p.output = p.output.and_then(|o| remap.row(o));
                    true
                }
                None => false,
            });
            Some(remap)
        };
        self.alt.resize(cols, rows);

        if self.on_alt {
            // A selection was over the alt screen, which only clamps and renumbers: there is
            // nothing to translate it against, so end the epoch and tell the caller to drop.
            self.epoch = self.epoch.next();
            ResizeEffect::Reset
        } else if let Some(remap) = remap {
            // Primary width reflow: the ids were renumbered but are translatable. End the
            // epoch (the safety net for any holder not translated) and hand back the map.
            self.epoch = self.epoch.next();
            ResizeEffect::Reflowed(remap)
        } else {
            // Height-only on the primary: the ids stayed put, so a selection stands as-is.
            ResizeEffect::Stable
        }
    }

    /// OSC 0/2: set the window title.
    pub fn set_title(&mut self, title: String) {
        self.title = title;
    }

    /// OSC 8: open or close a hyperlink on the pen. `pt` is everything after the `8;`,
    /// which the spec shapes as `params ; URI`:
    ///
    /// ```text
    ///   ESC ] 8 ; ; https://example.com  ESC \   the anchor opens
    ///   text the user sees                       these cells carry the link
    ///   ESC ] 8 ; ;                       ESC \   an empty URI closes it
    /// ```
    ///
    /// `params` is a `key=value:key=value` list whose one defined key is `id=`, a hint
    /// that two separate runs are one logical link. We ignore it: interning by URL
    /// already gives identical URLs one identity, which is the only thing the hint was
    /// for, and honouring an attacker-chosen id would let a child fuse two unrelated
    /// runs into one hover target.
    ///
    /// The URI is vetted **here**, when the anchor opens, not at click time. That is
    /// the rule [`crate::platform::browser::OPENABLE_SCHEMES`] states: a link we
    /// decorate and then refuse to follow is worse than one we never decorated. So a
    /// `javascript:` or `data:` anchor is not a link at all — it prints as plain text
    /// and never underlines, rather than underlining and then doing nothing. Anything
    /// we cannot make sense of (bad UTF-8, an unopenable scheme, an empty URI) closes
    /// whatever anchor was open, which is also the safe reading of a malformed
    /// sequence.
    fn set_hyperlink(&mut self, pt: &[u8]) {
        let mut fields = pt.splitn(2, |&b| b == b';');
        let _params = fields.next();
        let uri = fields.next().unwrap_or(&[]);
        let opened = std::str::from_utf8(uri)
            .ok()
            .filter(|uri| crate::platform::browser::can_open(uri));
        self.pen.link = match opened {
            Some(uri) => self.intern_link(uri),
            None => LinkId::NONE,
        };
    }

    /// The id for `url`, collecting dead ids first if the table has filled up. Falling
    /// all the way through to [`LinkId::NONE`] means 65534 *live* links are on screen
    /// and in scrollback at once, at which point the newest simply is not clickable —
    /// a degradation, never a failure, and the text still reads.
    fn intern_link(&mut self, url: &str) -> LinkId {
        if let Some(id) = self.links.intern(url) {
            return id;
        }
        self.collect_links();
        self.links.intern(url).unwrap_or(LinkId::NONE)
    }

    /// Reclaim the ids of hyperlinks no cell carries any more.
    ///
    /// Links die when the cells citing them do: scrollback evicts its oldest rows, the
    /// alt screen is thrown away wholesale on exit, an erase blanks a line. Without
    /// this, a session that prints an unbounded stream of *distinct* links (an
    /// `ls --hyperlink` of a large tree, run again and again) would burn through the
    /// `u16` id space and quietly stop linking anything until the terminal restarted.
    ///
    /// A textbook mark-and-sweep, run only when [`LinkTable::intern`] reports the table
    /// full — so the walk over every cell is an *event*, once per 65534 distinct links,
    /// and never a per-frame cost. The roots are every cell in both buffers (visible
    /// and scrollback) plus the pen and both saved cursors, because an anchor can be
    /// open with no text printed under it yet.
    fn collect_links(&mut self) {
        // `live[i]` speaks for `LinkId(i)`; slot 0 is `LinkId::NONE` and is never a URL.
        let mut live = vec![false; self.links.urls.len() + 1];
        let mark = |id: LinkId, live: &mut Vec<bool>| {
            if let Some(slot) = live.get_mut(usize::from(id.0)) {
                *slot = true;
            }
        };
        for buf in [&self.primary, &self.alt] {
            for row in buf.scrollback.iter().chain(buf.lines.iter()) {
                for cell in &row.cells {
                    mark(cell.link, &mut live);
                }
            }
            if let Some(saved) = &buf.saved {
                mark(saved.pen.link, &mut live);
            }
        }
        mark(self.pen.link, &mut live);

        let remap = self.links.compact(&live);
        let renumber = |id: LinkId| {
            remap
                .get(usize::from(id.0))
                .copied()
                .unwrap_or(LinkId::NONE)
        };
        for buf in [&mut self.primary, &mut self.alt] {
            for row in buf.scrollback.iter_mut().chain(buf.lines.iter_mut()) {
                for cell in &mut row.cells {
                    cell.link = renumber(cell.link);
                }
            }
            if let Some(saved) = &mut buf.saved {
                saved.pen.link = renumber(saved.pen.link);
            }
        }
        self.pen.link = renumber(self.pen.link);
    }

    /// DECKPAM/DECKPNM keypad mode, read by the input encoder (phase 3).
    pub fn keypad_app(&self) -> bool {
        self.keypad_app
    }

    /// Whether the bell has rung since this was last asked, clearing it. The grid has no
    /// opinion on what a bell should *do* — that is the app's call, and on a modern
    /// desktop it is a visual mark rather than a beep.
    pub fn take_bell(&mut self) -> bool {
        std::mem::take(&mut self.bell)
    }

    /// `?2048`: report the terminal's size in band, as an escape sequence, rather than
    /// only through SIGWINCH.
    ///
    /// The signal is not enough on its own. It reaches the process group on *this* side of
    /// the pty and nothing else, so a program on the far end of an ssh hop, or behind
    /// tmux, learns nothing — and a program that has just been handed a pty has no way to
    /// ask. Enabling the mode reports the size immediately for exactly that reason: the
    /// first report is the one that answers "what am I attached to?".
    fn set_in_band_resize(&mut self, enable: bool) {
        self.in_band_resize = enable;
        if enable {
            self.report_size();
        }
    }

    /// `CSI 48 ; rows ; cols ; height ; width t`, the in-band size report. Silent unless
    /// the child asked for it (`?2048`), which the app calls on every resize.
    pub fn report_size(&mut self) {
        if !self.in_band_resize {
            return;
        }
        let (cols, rows) = self.dimensions();
        let (w, h) = self.pixel_size;
        self.respond(b"\x1b[48;");
        push_decimal(&mut self.responses, rows as u32);
        self.responses.push(b';');
        push_decimal(&mut self.responses, cols as u32);
        self.responses.push(b';');
        push_decimal(&mut self.responses, h);
        self.responses.push(b';');
        push_decimal(&mut self.responses, w);
        self.responses.push(b't');
    }

    /// The window's pixel size, which the in-band resize report carries alongside the
    /// cell count. The grid does not otherwise care about pixels; it is told.
    pub fn set_pixel_size(&mut self, width: u32, height: u32) {
        self.pixel_size = (width, height);
    }

    /// Tell the child the window gained or lost focus (`CSI I` / `CSI O`), if it asked to
    /// be told (`?1004`). Silent otherwise: a program that never enabled focus reporting
    /// would read these as stray key presses.
    pub fn report_focus(&mut self, focused: bool) {
        if !self.focus_events {
            return;
        }
        self.respond(if focused { b"\x1b[I" } else { b"\x1b[O" });
    }

    /// Whether the child is mid-frame under synchronized output (`?2026`), and would
    /// rather we showed nothing than showed it half-drawn.
    pub fn synchronized(&self) -> bool {
        self.synchronized
    }

    /// The palette in force. The renderer borrows this at paint time; a program can have
    /// changed any of it through `OSC 4/10/11/12`.
    pub fn theme(&self) -> &Theme {
        &self.theme
    }

    /// `OSC 10/11/12`: set or *query* the default foreground, background, or cursor
    /// colour.
    ///
    /// The query is the reason this matters, and it is the most useful reply a terminal
    /// gives. An editor asks `OSC 11 ; ? ST` to find out whether it is sitting on a dark
    /// or a light background, and picks its whole colour scheme from the answer. A
    /// terminal that stays silent does not get a default — it gets a *guess*, and half
    /// the time the guess is wrong and every colour in the editor is subtly off.
    ///
    /// A program may batch several requests in one sequence (`OSC 10 ; ? ; ? ST` asks for
    /// the foreground and then the background), so the payload is a list, and each field
    /// steps to the next colour in the sequence fg → bg → cursor.
    fn osc_named_color(&mut self, first: NamedColor, pt: &[u8], bel: bool) {
        let mut which = Some(first);
        for field in pt.split(|&b| b == b';') {
            let Some(target) = which else { break };
            let slot = match target {
                NamedColor::Foreground => &mut self.theme.fg,
                NamedColor::Background => &mut self.theme.bg,
                NamedColor::Cursor => &mut self.theme.cursor,
            };
            if field == b"?" {
                let (color, code) = (*slot, osc_color_code(target));
                self.respond(b"\x1b]");
                push_decimal(&mut self.responses, code);
                self.responses.push(b';');
                color::write_x11_color(color, &mut self.responses);
                self.end_osc(bel);
            } else if let Some(color) = color::parse_x11_color(field) {
                *slot = color;
            }
            which = match target {
                NamedColor::Foreground => Some(NamedColor::Background),
                NamedColor::Background => Some(NamedColor::Cursor),
                NamedColor::Cursor => None,
            };
        }
    }

    /// `OSC 4 ; index ; spec` sets a palette entry; `OSC 4 ; index ; ?` queries one.
    /// Several pairs may ride in one sequence, which is how a theme script repaints the
    /// whole palette in a single write.
    fn osc_palette(&mut self, pt: &[u8], bel: bool) {
        let mut fields = pt.split(|&b| b == b';');
        while let (Some(index), Some(spec)) = (fields.next(), fields.next()) {
            let Some(index) = parse_u8(index) else {
                continue;
            };
            if spec == b"?" {
                self.respond(b"\x1b]4;");
                push_decimal(&mut self.responses, u32::from(index));
                self.responses.push(b';');
                color::write_x11_color(self.theme.indexed(index), &mut self.responses);
                self.end_osc(bel);
            } else if let Some(color) = color::parse_x11_color(spec) {
                self.theme.set_indexed(index, color);
            }
        }
    }

    /// `OSC 104`: reset the named palette entries, or the whole palette when the payload
    /// is empty.
    fn osc_reset_palette(&mut self, pt: &[u8]) {
        if pt.is_empty() {
            let (fg, bg, cursor) = (self.theme.fg, self.theme.bg, self.theme.cursor);
            self.theme.reset_palette();
            // `OSC 104` is about the *indexed* palette; the three named colours have
            // their own resets (110/111/112) and must survive this one.
            self.theme.fg = fg;
            self.theme.bg = bg;
            self.theme.cursor = cursor;
            return;
        }
        for field in pt.split(|&b| b == b';') {
            if let Some(index) = parse_u8(field) {
                self.theme.reset_indexed(index);
            }
        }
    }

    /// `OSC 7 ; file://<host>/<path>`: the shell reports its working directory.
    ///
    /// The path is percent-encoded, because a directory may contain any byte a filename
    /// may contain, including the `;` that separates OSC fields. A malformed escape is
    /// left as written rather than guessed at.
    fn osc_cwd(&mut self, pt: &[u8]) {
        let path = match pt.strip_prefix(b"file://") {
            // Strip the host: it is the machine the shell is on, which we cannot do
            // anything useful with, and the path begins at the slash that follows it.
            Some(rest) => match rest.iter().position(|&b| b == b'/') {
                Some(i) => rest.get(i..).unwrap_or(&[]),
                None => return,
            },
            // A bare path, which some shells send. Accept it.
            None if pt.starts_with(b"/") => pt,
            None => return,
        };
        self.cwd = Some(percent_decode(path));
    }

    /// `OSC 133`: the shell tells us where a prompt is, where the command's output starts,
    /// and how it ended.
    ///
    /// ```text
    ///   OSC 133 ; A ST          a prompt starts here
    ///   OSC 133 ; B ST          the prompt ends and what you type begins
    ///   OSC 133 ; C ST          the command's output starts here
    ///   OSC 133 ; D ; 1 ST      the command finished, and it failed
    /// ```
    ///
    /// Without this a terminal sees an undifferentiated river of text and can only offer
    /// you a scrollbar. With it, it knows where one command ended and the next began, and
    /// everything good follows from that: jump back to the last prompt, select a command's
    /// output, mark the ones that failed.
    fn osc_shell_mark(&mut self, pt: &[u8]) {
        let mut fields = pt.split(|&b| b == b';');
        let kind = fields.next().unwrap_or(&[]);
        match kind {
            b"A" => {
                let row = self.abs_row(self.active().cursor.row);
                // A shell that redraws its prompt in place (zsh does, on every keystroke
                // with syntax highlighting) re-marks the same row. It is the same prompt.
                if self.prompts.last().map(|p| p.row) == Some(row) {
                    return;
                }
                if self.prompts.len() >= PROMPT_LIMIT {
                    self.prompts.remove(0);
                }
                self.prompts.push(Prompt {
                    row,
                    output: None,
                    exit: None,
                });
            }
            b"C" => {
                let row = self.abs_row(self.active().cursor.row);
                if let Some(p) = self.prompts.last_mut() {
                    p.output.get_or_insert(row);
                }
            }
            b"D" => {
                // The exit code is optional: a shell that reports only "it finished" is
                // still telling us something worth knowing.
                let exit = fields
                    .next()
                    .and_then(|f| std::str::from_utf8(f).ok())
                    .and_then(|f| f.trim().parse::<i32>().ok());
                if let Some(p) = self.prompts.last_mut() {
                    p.exit = exit.or(p.exit);
                }
            }
            // `B` (the prompt ends, input begins) we parse and do not need: nothing we
            // offer keys off it, and inventing a use for it would be inventing a feature.
            _ => {}
        }
    }

    /// The prompts the shell has marked, oldest first.
    pub fn prompts(&self) -> &[Prompt] {
        &self.prompts
    }

    /// Scroll so the prompt nearest above (or below) the top of the view comes into sight,
    /// and answer whether one was found.
    ///
    /// This is the payoff, and it is the thing a scrollbar can never do: a scrollbar knows
    /// how far you have come, and this knows *what* you have come past. Ten screens of a
    /// build log scroll by in one keystroke because the terminal knows where the command
    /// that produced them started.
    ///
    /// The prompt lands at the top of the screen, which is where you want it: the command
    /// and everything it printed are then below it, in the order they happened. A prompt
    /// close to the live bottom cannot be lifted that far — there is not enough text under
    /// it to fill the screen — so it comes to rest as high as the history allows, which is
    /// the same thing every scrollbar in the world does at the end of its travel.
    pub fn scroll_to_prompt(&mut self, backward: bool) -> bool {
        let top = self.abs_row(0);
        let target = if backward {
            self.prompts.iter().rev().find(|p| p.row < top)
        } else {
            self.prompts.iter().find(|p| p.row > top)
        };
        let Some(row) = target.map(|p| p.row) else {
            return false;
        };
        let b = self.active();
        let Some(index) = b.stream_index(row) else {
            return false;
        };
        // The offset is measured back from the live top of the screen, which is where the
        // scrollback ends.
        let offset = b.scrollback.len().saturating_sub(index);
        self.view_offset = offset.min(self.scrollback_len());
        true
    }

    /// The working directory the shell last reported (`OSC 7`).
    pub fn reported_cwd(&self) -> Option<&str> {
        self.cwd.as_deref()
    }

    /// `OSC 52 ; <selection> ; <base64>`: the child puts text on the clipboard.
    ///
    /// This is how a program reaches the system clipboard when it cannot reach the
    /// compositor itself — `nvim` over SSH, tmux's copy mode, Claude Code's `/copy`. The
    /// terminal is the only process in the pipeline that *is* on your desktop, so it has
    /// to be the one to do it.
    ///
    /// # Reads are refused, and that is deliberate
    ///
    /// The protocol also defines a *read* (`OSC 52 ; c ; ?`), which answers with the
    /// clipboard's contents. We do not implement it, and will not.
    ///
    /// A terminal cannot tell which program printed a byte. Anything that reaches your
    /// screen can send this — a `cat` of a hostile file, a log line, a compiler error
    /// quoting attacker-controlled text — and the reply goes straight back down the pty
    /// to whatever is reading it. That turns "display some text" into "exfiltrate the
    /// user's clipboard", which routinely holds passwords. xterm ships reads disabled for
    /// exactly this reason and it is the right call.
    ///
    /// The *write* is a smaller version of the same problem (a hostile file can clobber
    /// your clipboard, which is annoying but not a disclosure), and it is the half every
    /// real program actually needs, so it stays.
    fn osc_clipboard(&mut self, pt: &[u8]) {
        let mut fields = pt.splitn(2, |&b| b == b';');
        let selection = fields.next().unwrap_or(&[]);
        let payload = fields.next().unwrap_or(&[]);
        if payload == b"?" {
            return; // A read. See above: never.
        }
        let Some(text) = base64_decode(payload) else {
            return;
        };
        // The X11 selection letters. An empty field means the clipboard, and a program
        // may name several at once (`OSC 52 ; pc ; …`), so honour each one it lists.
        let letters = if selection.is_empty() {
            &b"c"[..]
        } else {
            selection
        };
        let mut clipboard = false;
        let mut primary = false;
        for &letter in letters {
            match letter {
                b'c' => clipboard = true,
                b'p' | b's' => primary = true,
                _ => {}
            }
        }
        if clipboard {
            self.clipboard_writes
                .push((ClipboardTarget::Clipboard, text.clone()));
        }
        if primary {
            self.clipboard_writes.push((ClipboardTarget::Primary, text));
        }
    }

    /// Take the clipboard writes the child has asked for, leaving the queue empty. The
    /// app drains this after each parse batch, exactly as it drains query replies.
    pub fn take_clipboard_writes(&mut self) -> Vec<(ClipboardTarget, Vec<u8>)> {
        std::mem::take(&mut self.clipboard_writes)
    }

    /// Close a reply that opened with `OSC`, with the same terminator the request used.
    /// A client that asked with BEL may only be listening for BEL.
    fn end_osc(&mut self, bel: bool) {
        if bel {
            self.responses.push(0x07);
        } else {
            self.respond(b"\x1b\\");
        }
    }

    /// The kitty keyboard flags in force: the top of the stack, or none when the child
    /// has pushed nothing and the legacy encoding applies.
    pub fn kitty_flags(&self) -> KittyFlags {
        self.kitty_stack.last().copied().unwrap_or(KittyFlags::NONE)
    }

    /// The `modifyOtherKeys` level the child asked for.
    pub fn modify_other_keys(&self) -> ModifyOtherKeys {
        self.modify_other_keys
    }

    /// XTVERSION (`CSI > 0 q`): "what terminal are you?" Answered `DCS > | bnkterm <ver> ST`.
    ///
    /// This does not get us onto anyone's allowlist — the CLIs that gate features on a
    /// terminal's name match it against a fixed list we are not on, and we do not intend
    /// to impersonate someone to get on it (see the `TERM_PROGRAM` note in `app.rs`). But
    /// it is what a terminal is *supposed* to say when asked, and the only way a program
    /// could recognise us deliberately rather than not at all.
    fn xtversion(&mut self) {
        self.respond(b"\x1bP>|bnkterm ");
        self.respond(env!("CARGO_PKG_VERSION").as_bytes());
        self.respond(b"\x1b\\");
    }

    /// DECRQSS (`DCS $ q <setting> ST`): "what is this setting currently set to?" The
    /// answer is the sequence that would *reproduce* it — `DCS 1 $ r 0 m ST` says "SGR is
    /// at its default" — so a program can save a setting, change it, and put it back
    /// without knowing what it was.
    ///
    /// An unknown request is answered `DCS 0 $ r ST`, which means "I do not support that
    /// query". Answering 1 with an empty or invented setting would be worse than silence:
    /// the program would take the reply at face value and restore garbage.
    fn decrqss(&mut self, setting: &[u8]) {
        match setting {
            b"m" => {
                // SGR. We report the *pen*, which is what a program restoring a rendition
                // needs; the attribute bits are spelled out in the order xterm uses.
                self.respond(b"\x1bP1$r0");
                let attrs = self.pen.attrs;
                for (bit, code) in [
                    (Attrs::BOLD, b"1".as_slice()),
                    (Attrs::DIM, b"2"),
                    (Attrs::ITALIC, b"3"),
                    // The underline carries its shape with it. Reporting a bare `4` for a
                    // curly one answers a program's "what is this set to?" with a
                    // different setting: it saves the reply, changes the style, replays
                    // it, and its squiggle has quietly become a straight line. The whole
                    // point of DECRQSS is that the answer reproduces the state.
                    (
                        Attrs::UNDERLINE,
                        Self::underline_sgr(attrs.underline_style()),
                    ),
                    (Attrs::REVERSE, b"7"),
                    (Attrs::HIDDEN, b"8"),
                    (Attrs::STRIKE, b"9"),
                ] {
                    if attrs.contains(bit) {
                        self.responses.push(b';');
                        self.respond(code);
                    }
                }
                self.push_sgr_color(true);
                self.push_sgr_color(false);
                self.respond(b"m\x1b\\");
            }
            b"r" => {
                // DECSTBM, the scroll region, 1-based as the sequence that sets it.
                let (top, bottom) = {
                    let b = self.active();
                    (b.scroll_top, b.scroll_bottom)
                };
                self.respond(b"\x1bP1$r");
                push_decimal(&mut self.responses, top as u32 + 1);
                self.responses.push(b';');
                push_decimal(&mut self.responses, bottom as u32 + 1);
                self.respond(b"r\x1b\\");
            }
            _ => self.respond(b"\x1bP0$r\x1b\\"),
        }
    }

    /// The SGR parameter that sets an underline of this shape, for a DECRQSS reply.
    ///
    /// Plain underline reports as `4` rather than `4:1`, because that is what everything
    /// has always written and a program restoring it should get back what it sent.
    fn underline_sgr(style: UnderlineStyle) -> &'static [u8] {
        match style {
            UnderlineStyle::Single => b"4",
            UnderlineStyle::Double => b"4:2",
            UnderlineStyle::Curly => b"4:3",
            UnderlineStyle::Dotted => b"4:4",
            UnderlineStyle::Dashed => b"4:5",
        }
    }

    /// One colour of the pen, as the SGR parameters that would set it, appended to a
    /// DECRQSS reply. Nothing is appended for a default colour: `SGR 0` already said it.
    fn push_sgr_color(&mut self, foreground: bool) {
        let color = if foreground { self.pen.fg } else { self.pen.bg };
        let base = if foreground { 30 } else { 40 };
        match color {
            Color::Default => {}
            Color::Ansi(i) if i < 8 => {
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(base + i));
            }
            Color::Ansi(i) => {
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(base + 60 + (i & 7)));
            }
            Color::Indexed(i) => {
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(base + 8));
                self.respond(b";5;");
                push_decimal(&mut self.responses, u32::from(i));
            }
            Color::Rgb(r, g, b) => {
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(base + 8));
                self.respond(b";2;");
                push_decimal(&mut self.responses, u32::from(r));
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(g));
                self.responses.push(b';');
                push_decimal(&mut self.responses, u32::from(b));
            }
        }
    }

    /// XTGETTCAP (`DCS + q <hex-encoded names> ST`): "what does your terminfo say about
    /// these capabilities?" Names and values travel hex-encoded, which is how a value
    /// containing an escape sequence survives the trip.
    ///
    /// This is the query that lets a program learn what we can do *without* a terminfo
    /// entry installed for us — which matters, because we ship none and claim
    /// `xterm-256color`, so a terminfo lookup describes xterm rather than bnkterm.
    ///
    /// An unknown capability is answered with `DCS 0 + r <name> ST`, which is the
    /// protocol's way of saying "I do not have that" — and it is a real answer, not a
    /// silence, so the program stops waiting.
    fn xtgettcap(&mut self, data: &[u8]) {
        for name in data.split(|&b| b == b';') {
            let Some(decoded) = hex_decode(name) else {
                self.respond(b"\x1bP0+r\x1b\\");
                continue;
            };
            let value: Option<&[u8]> = match decoded.as_slice() {
                b"TN" | b"name" => Some(b"bnkterm".as_slice()),
                b"Co" | b"colors" => Some(b"256".as_slice()),
                // Truecolor. The `RGB` capability is how a program learns it can send
                // `38;2;r;g;b` and mean it; `COLORTERM=truecolor` says the same thing to
                // the programs that read the environment instead.
                b"RGB" => Some(b"8/8/8".as_slice()),
                _ => None,
            };
            match value {
                Some(value) => {
                    // The name goes back exactly as it arrived: it is already hex on the
                    // wire, and re-encoding it here would hex the hex.
                    self.respond(b"\x1bP1+r");
                    self.respond(name);
                    self.responses.push(b'=');
                    hex_encode(value, &mut self.responses);
                    self.respond(b"\x1b\\");
                }
                None => {
                    self.respond(b"\x1bP0+r");
                    self.respond(name);
                    self.respond(b"\x1b\\");
                }
            }
        }
    }

    /// The kitty keyboard protocol's four control sequences, all sharing the final byte
    /// `u` and distinguished by their private marker:
    ///
    /// ```text
    ///   CSI ? u                 query   ─▶ reply CSI ? <flags> u
    ///   CSI = <flags> ; <mode> u set     (mode 1 replace, 2 set bits, 3 clear bits)
    ///   CSI > <flags> u          push
    ///   CSI < <count> u          pop
    /// ```
    ///
    /// The request is masked to [`KittyFlags::SUPPORTED`] on the way in, so the query
    /// reports what bnkterm will really do rather than what was asked for. A program
    /// that wants key-release events and reads back that it is not getting them can
    /// fall back; one that is told yes and then never sees a release would hang.
    fn kitty_keyboard(&mut self, params: &Params, private: u8) {
        match private {
            b'?' => {
                self.respond(b"\x1b[?");
                let flags = self.kitty_flags().bits();
                push_decimal(&mut self.responses, flags);
                self.respond(b"u");
            }
            b'=' => {
                let flags = KittyFlags::from_request(params.value(0));
                let mode = if params.len() > 1 { params.value(1) } else { 1 };
                let current = self.kitty_flags();
                let next = current.apply(flags, mode);
                match self.kitty_stack.last_mut() {
                    Some(top) => *top = next,
                    // Setting flags with nothing pushed still has to take effect, so the
                    // set becomes the stack's first entry.
                    None => self.kitty_stack.push(next),
                }
            }
            b'>' => {
                let flags = KittyFlags::from_request(params.value(0));
                if self.kitty_stack.len() >= KITTY_STACK_LIMIT {
                    self.kitty_stack.remove(0);
                }
                self.kitty_stack.push(flags);
            }
            b'<' => {
                let count = usize::from(
                    if params.is_empty() {
                        1
                    } else {
                        params.value(0)
                    }
                    .max(1),
                );
                let keep = self.kitty_stack.len().saturating_sub(count);
                self.kitty_stack.truncate(keep);
            }
            _ => {}
        }
    }

    /// XTMODKEYS (`CSI > Pp ; Pv m`). `Pp = 4` is `modifyOtherKeys`, the only resource
    /// we implement; the others (`modifyCursorKeys`, `modifyFunctionKeys`) only shuffle
    /// encodings we already emit in their standard form. Omitting `Pv` resets, which is
    /// how xterm defines it and how a program turns the mode back off on exit.
    fn xtmodkeys(&mut self, params: &Params) {
        if params.value(0) != 4 || params.is_empty() {
            return;
        }
        self.modify_other_keys = if params.len() > 1 {
            ModifyOtherKeys::from_param(params.value(1))
        } else {
            ModifyOtherKeys::Off
        };
    }

    /// DECALN: fill the whole screen with 'E', reset the margins to the extremes of the
    /// page, and home the cursor (a vttest alignment pattern; useful for confirming
    /// glyph placement early).
    ///
    /// The margins and the cursor are not incidental: DECALN's own definition is that it
    /// "sets the margins to the extremes of the page, and moves the cursor to the home
    /// position", so a screen left under a stale scroll region after an alignment test
    /// would scroll inside it.
    pub fn decaln(&mut self) {
        let b = self.active_mut();
        b.clear_all(Cell::new('E'));
        b.scroll_top = 0;
        b.scroll_bottom = b.rows.saturating_sub(1);
        b.cursor = Cursor::default();
    }

    /// The charset GL currently maps through (G0 or G1).
    fn active_charset(&self) -> Charset {
        if self.gl_is_g1 {
            self.g1
        } else {
            self.g0
        }
    }

    /// Translate a printable char through the active charset (DEC Special
    /// Graphics remaps `_`..`~` to line-drawing glyphs; ASCII is identity).
    fn map_glyph(&self, c: char) -> char {
        match self.active_charset() {
            Charset::Ascii => c,
            Charset::DecSpecialGraphics => dec_special_graphics(c),
        }
    }
}

/// Tab stops every [`TAB_WIDTH`] columns (the terminal default).
/// How many columns the tab-stop table covers, regardless of how wide the screen
/// currently is.
///
/// The table is deliberately *not* geometry, and sizing it to the width was a bug: `TBC
/// 3` ("clear every stop") can only clear the columns the table has, so a later widening
/// would seed fresh defaults into columns the program had explicitly emptied, and the
/// stops it deleted would come back. Covering a fixed span from the start means "every"
/// means every, whatever the window does afterwards. xterm's array is width-independent
/// (1024 columns) for the same reason.
const MAX_TAB_COLUMNS: usize = 1024;

/// The default table: a stop every [`TAB_WIDTH`] columns, across [`MAX_TAB_COLUMNS`] or
/// the screen's width, whichever is more.
fn default_tabs(cols: usize) -> Vec<bool> {
    (0..cols.max(MAX_TAB_COLUMNS))
        .map(|c| c % TAB_WIDTH == 0)
        .collect()
}

/// Whether a rune bounds a double-click word selection: whitespace, or one of the
/// brackets/quotes that fence a token. The set matches the common terminal default
/// (wezterm/xterm), so double-clicking selects a path or URL whole but stops at a
/// delimiter. A blank cell's rune is a space, so it bounds a word too.
fn is_word_boundary(c: char) -> bool {
    c.is_whitespace() || matches!(c, '{' | '}' | '[' | ']' | '(' | ')' | '"' | '\'' | '`')
}

/// Split the reply buffer into individual replies, so a snapshot shows one per line.
///
/// Every reply we emit begins with `ESC`, which makes the boundary almost unambiguous —
/// the exception being the `ESC \` that *terminates* a DCS reply, which opens nothing and
/// belongs to the reply in front of it.
fn split_replies(responses: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, &byte) in responses.iter().enumerate() {
        let terminator = responses.get(i + 1) == Some(&b'\\');
        if byte == 0x1b && i > start && !terminator {
            if let Some(reply) = responses.get(start..i) {
                out.push(reply);
            }
            start = i;
        }
    }
    if let Some(reply) = responses.get(start..) {
        if !reply.is_empty() {
            out.push(reply);
        }
    }
    out
}

/// One reply, readable: the control bytes named rather than printed, so a golden diff
/// reads like the sequence it is instead of a hex dump.
fn escape_reply(reply: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for &byte in reply {
        match byte {
            0x1b => out.push_str("<ESC>"),
            0x07 => out.push_str("<BEL>"),
            0x20..=0x7e => out.push(char::from(byte)),
            _ => {
                let _ = write!(out, "<{byte:#04x}>");
            }
        }
    }
    out
}

/// Percent-decode a URL path, which is how `OSC 7` carries a directory name.
///
/// It has to be encoded: a filename may contain any byte except `/` and NUL — including
/// the `;` that separates OSC fields and the bytes that would terminate the sequence
/// outright. A `%` that is not followed by two hex digits is a literal `%`, which is what
/// a shell that forgot to encode one produces, and guessing otherwise would mangle a
/// directory that is merely unusual rather than malformed.
///
/// Invalid UTF-8 becomes U+FFFD rather than being refused: a path we cannot render is
/// still a path we can show *something* for, and the alternative is silently having no cwd.
fn percent_decode(input: &[u8]) -> String {
    let mut bytes = Vec::with_capacity(input.len());
    let mut i = 0;
    while let Some(&byte) = input.get(i) {
        let decoded = (byte == b'%')
            .then(|| {
                let hi = (*input.get(i + 1)? as char).to_digit(16)?;
                let lo = (*input.get(i + 2)? as char).to_digit(16)?;
                Some((hi * 16 + lo) as u8)
            })
            .flatten();
        match decoded {
            Some(b) => {
                bytes.push(b);
                i += 3;
            }
            None => {
                bytes.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Decode the hex that XTGETTCAP encodes capability names in. `None` on anything that is
/// not an even run of hex digits, which the caller answers as "I do not have that"
/// rather than guessing at.
fn hex_decode(input: &[u8]) -> Option<Vec<u8>> {
    if input.is_empty() || !input.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(input.len() / 2);
    for pair in input.chunks_exact(2) {
        let hi = (*pair.first()? as char).to_digit(16)?;
        let lo = (*pair.get(1)? as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
    }
    Some(out)
}

/// Encode a capability value as the hex XTGETTCAP expects, which is how a value holding
/// an escape sequence survives being sent inside one.
fn hex_encode(input: &[u8], out: &mut Vec<u8>) {
    for &byte in input {
        for nibble in [byte >> 4, byte & 0x0f] {
            out.push(match nibble {
                0..=9 => b'0' + nibble,
                _ => b'a' + (nibble - 10),
            });
        }
    }
}

/// Decode base64, the encoding `OSC 52` wraps clipboard text in. `None` for anything
/// malformed, which the caller drops.
///
/// Hand-rolled, and easily: base64 is a fixed 4-to-3 byte fold with no lengths, no
/// offsets and no allocation decisions taken from the input, which is exactly the shape
/// of thing the dependency policy says to write rather than depend on. (Image decoding
/// is the counter-example, and the reason the policy exists at all.)
///
/// Strict about what it accepts: the child writes this. Whitespace is skipped (mail-era
/// encoders wrap lines), padding is optional, and anything else is a refusal rather than
/// a guess — a decoder that improvises on bad input is how you smuggle bytes past it.
fn base64_decode(input: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() / 4 * 3 + 3);
    let mut quad = [0u8; 4];
    let mut n = 0;
    let mut padding = 0;
    for &byte in input {
        if byte.is_ascii_whitespace() {
            continue;
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => {
                padding += 1;
                if padding > 2 {
                    return None;
                }
                0
            }
            _ => return None,
        };
        // Padding may only ever be trailing: `A=AA` is malformed, not a clever encoding.
        if padding > 0 && byte != b'=' {
            return None;
        }
        quad[n] = value;
        n += 1;
        if n == 4 {
            let triple = (u32::from(quad[0]) << 18)
                | (u32::from(quad[1]) << 12)
                | (u32::from(quad[2]) << 6)
                | u32::from(quad[3]);
            out.push((triple >> 16) as u8);
            if padding < 2 {
                out.push((triple >> 8) as u8);
            }
            if padding < 1 {
                out.push(triple as u8);
            }
            n = 0;
        }
    }
    match n {
        // A clean end, padded or exactly aligned.
        0 => Some(out),
        // Unpadded input, which the spec allows: finish the partial group.
        2 | 3 => {
            let triple =
                (u32::from(quad[0]) << 18) | (u32::from(quad[1]) << 12) | (u32::from(quad[2]) << 6);
            out.push((triple >> 16) as u8);
            if n == 3 {
                out.push((triple >> 8) as u8);
            }
            Some(out)
        }
        // One leftover character encodes six bits of nothing.
        _ => None,
    }
}

/// The OSC number that names each colour, for building a reply.
fn osc_color_code(which: NamedColor) -> u32 {
    match which {
        NamedColor::Foreground => 10,
        NamedColor::Background => 11,
        NamedColor::Cursor => 12,
    }
}

/// A decimal palette index from an OSC field. `None` for anything that is not a plain
/// number in `0..=255`, which the caller then skips.
fn parse_u8(field: &[u8]) -> Option<u8> {
    if field.is_empty() || field.len() > 3 {
        return None;
    }
    let mut n: u32 = 0;
    for &b in field {
        let digit = (b as char).to_digit(10)?;
        n = n * 10 + digit;
    }
    u8::try_from(n).ok()
}

/// Clamp an SGR color component (`0..=255`) to a byte; the parser already bounds
/// parameters, so this only guards a malformed stream, never a valid one.
fn sgr_component(v: u16) -> u8 {
    v.min(255) as u8
}

/// Parse the extended-colour introducer (`38`, `48`, `58`) at parameter `at`,
/// returning the colour and how many *extra parameters* it consumed.
///
/// The arguments come in one of two spellings, and a terminal has to read both:
///
/// ```text
///   38 ; 2 ; r ; g ; b     five parameters   (xterm's, and the common one)
///   38 : 2 : : r : g : b   one parameter, six sub-parameters (ITU T.416)
///   38 : 2 : r : g : b     one parameter, five sub-parameters (seen in the wild)
/// ```
///
/// The empty slot in the T.416 form is a colour-space id, which nobody uses and we
/// ignore. The sub-parameter form consumes no *extra* parameters — everything it needs
/// is inside the one it started in — which is why the return value counts parameters
/// and not values.
///
/// An unrecognized form consumes nothing: better to skip one parameter than to swallow
/// a `1` that was meant to turn on bold.
fn parse_ext_color(param: &[u16], params: &Params, at: usize) -> (Option<Color>, usize) {
    // The sub-parameter form: the arguments ride inside this parameter.
    if param.len() > 1 {
        return match param.get(1).copied() {
            Some(5) => {
                let n = param.get(2).copied().unwrap_or(0);
                (Some(Color::Indexed(sgr_component(n))), 0)
            }
            Some(2) => {
                // With a colour-space id (6 values) the channels start one later than
                // without it (5 values). Anything shorter is malformed; read what is
                // there and let the missing channels default to zero.
                let base = if param.len() >= 6 { 3 } else { 2 };
                let r = sgr_component(param.get(base).copied().unwrap_or(0));
                let g = sgr_component(param.get(base + 1).copied().unwrap_or(0));
                let b = sgr_component(param.get(base + 2).copied().unwrap_or(0));
                (Some(Color::Rgb(r, g, b)), 0)
            }
            _ => (None, 0),
        };
    }

    // The parameter form: the arguments are the parameters that follow.
    match params.value(at + 1) {
        5 if params.len() > at + 1 => {
            let n = params.value(at + 2);
            (Some(Color::Indexed(sgr_component(n))), 2)
        }
        2 if params.len() > at + 1 => {
            let r = sgr_component(params.value(at + 2));
            let g = sgr_component(params.value(at + 3));
            let b = sgr_component(params.value(at + 4));
            (Some(Color::Rgb(r, g, b)), 4)
        }
        _ => (None, 0),
    }
}

/// The charset an `ESC ( F` / `ESC ) F` designation selects. Only `0` (DEC
/// Special Graphics) differs from ASCII in what we support; `B` (ASCII) and any
/// other designation map to ASCII.
fn charset_from(byte: u8) -> Charset {
    match byte {
        b'0' => Charset::DecSpecialGraphics,
        _ => Charset::Ascii,
    }
}

/// The DEC Special Graphics glyph for a byte in `0x60..=0x7e` (the line-drawing
/// set); any other char passes through unchanged. This is the box-drawing
/// coverage `less`, `mc`, and framed TUIs depend on.
fn dec_special_graphics(c: char) -> char {
    match c {
        // 0x5F is "Blank" in DEC's own chart, not an underscore: the set replaces the
        // ASCII glyph at every position it defines, and this is the one that maps to
        // nothing visible. Leaving it through as `_` puts an underscore in the middle
        // of any line-drawing output that happens to contain one.
        '_' => ' ',
        '`' => '◆',
        'a' => '▒',
        'b' => '␉',
        'c' => '␌',
        'd' => '␍',
        'e' => '␊',
        'f' => '°',
        'g' => '±',
        'h' => '␤',
        'i' => '␋',
        'j' => '┘',
        'k' => '┐',
        'l' => '┌',
        'm' => '└',
        'n' => '┼',
        'o' => '⎺',
        'p' => '⎻',
        'q' => '─',
        'r' => '⎼',
        's' => '⎽',
        't' => '├',
        'u' => '┤',
        'v' => '┴',
        'w' => '┬',
        'x' => '│',
        'y' => '≤',
        'z' => '≥',
        '{' => 'π',
        '|' => '≠',
        '}' => '£',
        '~' => '·',
        _ => c,
    }
}

/// A CSI numeric parameter, or 0 when absent.
fn csi_arg(params: &Params, i: usize) -> u16 {
    params.value(i)
}

/// A CSI count parameter (for cursor moves, repeat counts): absent or 0 means 1.
fn csi_count(params: &Params, i: usize) -> usize {
    usize::from(csi_arg(params, i).max(1))
}

/// A 1-based CSI position parameter as a 0-based index (default 1 maps to 0).
fn csi_index(params: &Params, i: usize) -> usize {
    usize::from(csi_arg(params, i).max(1)) - 1
}

/// Append `n` as decimal ASCII (for building query responses), allocation-free.
fn push_decimal(out: &mut Vec<u8>, mut n: u32) {
    if n == 0 {
        out.push(b'0');
        return;
    }
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    while n > 0 {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    out.extend_from_slice(&tmp[i..]);
}

impl Screen {
    /// The private-marker CSI space: `CSI ? …` (DEC), `CSI > …`, `CSI = …`, `CSI < …`.
    ///
    /// A marker is not a modifier on the ANSI sequence that shares its final byte, it
    /// selects a *different sequence*. Letting one fall through to its ANSI namesake is
    /// how `CSI ? Ps r` (XTRESTORE, "restore saved private modes") used to execute as
    /// DECSTBM — resetting the scroll region and homing the cursor — and how
    /// `CSI ? Pi ; Pa ; Pv S` (XTSMGRAPHICS, a sixel query) used to execute as SU and
    /// scroll the screen out from under the program that asked.
    ///
    /// So the rule here is: match a marker and a final byte *together*, and drop
    /// anything else. Dropping an unknown sequence is always safe. Guessing is not.
    fn csi_private(&mut self, params: &Params, private: u8, action: u8) {
        match (private, action) {
            (b'?', b'h') => {
                for m in params.iter() {
                    self.set_mode(m.first().copied().unwrap_or(0), true, true);
                }
            }
            (b'?', b'l') => {
                for m in params.iter() {
                    self.set_mode(m.first().copied().unwrap_or(0), true, false);
                }
            }
            (b'?', b'n') => self.device_status(params, b'?'),
            (b'>', b'c') => self.device_attributes(b'>'),
            (b'>', b'm') => self.xtmodkeys(params),
            (b'>', b'q') => self.xtversion(),
            // The kitty keyboard protocol: `?` query, `=` set, `>` push, `<` pop. It
            // shares SCORC's final byte, and only the marker tells them apart — which is
            // exactly why this space has to be matched as a pair.
            (_, b'u') => self.kitty_keyboard(params, private),
            _ => {}
        }
    }
}

/// The VT interpretation: turn the parser's syntactic callbacks into the grid
/// operations above. This is the "grid interprets the actions" seam; the parser
/// (`vt.rs`) stays free of any of this meaning.
impl Perform for Screen {
    fn print(&mut self, c: char) {
        let mapped = self.map_glyph(c);
        // The inherent `Screen::print` (wide-char / wrap / combining) takes
        // priority over this trait method, so this is not a recursive call.
        self.print(mapped);
    }

    fn print_run(&mut self, chars: &[char]) {
        if self.insert_mode || self.grapheme_clustering || self.active_charset() != Charset::Ascii {
            for &c in chars {
                let mapped = self.map_glyph(c);
                self.print(mapped);
            }
            return;
        }
        self.print_text_run(chars);
    }

    fn print_ascii(&mut self, bytes: &[u8]) {
        // The bulk write assumes each byte is its own glyph placed one column
        // apart. That holds only under the identity (ASCII) charset, outside
        // insert mode, and outside grapheme clustering; DEC Special Graphics remaps
        // each byte to a line-drawing glyph, insert mode shifts the row per char, and
        // under `?2027` a byte is not its own glyph at all — it may be the *base* of a
        // cluster that the next scalar joins and widens. `#` is one column; `#️⃣` is the
        // same byte, two columns, and one keycap. The bulk path places cells without
        // anchoring a cluster, so a run through it leaves nothing for a following VS16
        // to attach to and the keycap silently loses half its width.
        //
        // The cost is that `?2027` turns off the bulk ASCII path, which is a real
        // slowdown for a program that opts in. That is the right way round: the mode is
        // off by default, so nothing pays for it unasked, and a program that asks for
        // cluster semantics is asking for exactly the thing the fast path cannot do.
        if self.insert_mode || self.grapheme_clustering || self.active_charset() != Charset::Ascii {
            for &b in bytes {
                let mapped = self.map_glyph(char::from(b));
                self.print(mapped);
            }
            return;
        }
        self.print_ascii_run(bytes);
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            0x08 => self.backspace(),        // BS
            0x09 => self.tab(),              // HT
            0x0a..=0x0c => self.line_feed(), // LF, VT, FF
            0x0d => self.carriage_return(),  // CR
            0x0e => self.gl_is_g1 = true,    // SO (shift out to G1)
            0x0f => self.gl_is_g1 = false,   // SI (shift in to G0)
            0x07 => self.bell = true,        // BEL
            _ => {}                          // NUL, XON/XOFF, ...: nothing to draw
        }
        // Whatever it did, it was not printing: there is nothing left for REP to repeat.
        self.last_printed = None;
    }

    fn csi_dispatch(&mut self, params: &Params, intermediates: &[u8], private: u8, action: u8) {
        // The sequences carrying an intermediate byte, and *only* those: the overwhelming
        // majority of CSIs have none, and they should not pay four slice comparisons to
        // discover it. Each is matched on the intermediate, the private marker *and* the
        // final byte together — see `csi_private` for what happens when a final byte is
        // trusted on its own.
        if !intermediates.is_empty() {
            match (intermediates, private, action) {
                ([b'!'], 0, b'p') => self.soft_reset(), // DECSTR
                ([b'$'], 0, b'p') => self.report_mode(params.value(0), false), // DECRQM
                ([b'$'], b'?', b'p') => self.report_mode(params.value(0), true), // DECRQM
                ([b' '], 0, b'q') => self.set_cursor_style(params.value(0)), // DECSCUSR
                // Anything else with an intermediate is dropped rather than misread.
                _ => {}
            }
            return;
        }
        // A private marker selects a different sequence space; it does not modify the
        // ANSI sequence that happens to share the final byte. Splitting them here is
        // what stops `CSI ? Ps r` (XTRESTORE) from executing as DECSTBM.
        if private != 0 {
            self.csi_private(params, private, action);
            return;
        }
        match action {
            b'A' => self.move_up(csi_count(params, 0)),
            b'B' => self.move_down(csi_count(params, 0)),
            b'C' => self.move_forward(csi_count(params, 0)),
            b'D' => self.move_back(csi_count(params, 0)),
            b'E' => {
                self.move_down(csi_count(params, 0));
                self.carriage_return();
            }
            b'F' => {
                self.move_up(csi_count(params, 0));
                self.carriage_return();
            }
            b'G' | b'`' => self.move_to_col(csi_index(params, 0)),
            b'H' | b'f' => self.move_to(csi_index(params, 0), csi_index(params, 1)),
            b'd' => self.move_to_row(csi_index(params, 0)),
            b'I' => {
                for _ in 0..csi_count(params, 0) {
                    self.tab();
                }
            }
            b'Z' => self.back_tab(csi_count(params, 0)), // CBT
            b'b' => self.repeat_last(csi_count(params, 0)), // REP
            b'J' => self.erase_display(csi_arg(params, 0)),
            b'K' => self.erase_line(csi_arg(params, 0)),
            b'L' => self.insert_lines(csi_count(params, 0)),
            b'M' => self.delete_lines(csi_count(params, 0)),
            b'@' => self.insert_chars(csi_count(params, 0)),
            b'P' => self.delete_chars(csi_count(params, 0)),
            b'X' => self.erase_chars(csi_count(params, 0)),
            b'S' => self.scroll_up(csi_count(params, 0)),
            // `CSI Ps T` is SD, but `CSI Ps;Ps;Ps;Ps;Ps T` is xterm's highlight mouse
            // tracking — a different function that merely shares a final byte. bnkterm
            // does not implement it, and *ignoring* it is the whole of the right answer:
            // reading it as a scroll would jerk the page every time a program asked to
            // track the mouse, which is worse than not answering at all.
            b'T' if params.len() == 5 => {}
            b'T' => self.scroll_down(csi_count(params, 0)),
            b'r' => {
                let (_, rows) = self.dimensions();
                let top = csi_index(params, 0);
                let bottom = match csi_arg(params, 1) {
                    0 => rows - 1,
                    v => usize::from(v) - 1,
                };
                self.set_scroll_region(top, bottom);
            }
            b'm' => self.sgr(params),
            b'h' => {
                for m in params.iter() {
                    self.set_mode(m.first().copied().unwrap_or(0), false, true);
                }
            }
            b'l' => {
                for m in params.iter() {
                    self.set_mode(m.first().copied().unwrap_or(0), false, false);
                }
            }
            b's' if params.is_empty() => self.save_cursor(),
            b'u' if params.is_empty() => self.restore_cursor(),
            b'g' => self.clear_tab_stop(csi_arg(params, 0)),
            b'c' => self.device_attributes(0),
            b'n' => self.device_status(params, 0),
            _ => {}
        }
    }

    fn esc_dispatch(&mut self, intermediates: &[u8], byte: u8) {
        match intermediates.first().copied() {
            None => match byte {
                b'c' => self.reset(),            // RIS
                b'D' => self.line_feed(),        // IND
                b'E' => self.next_line(),        // NEL
                b'H' => self.set_tab_stop(),     // HTS
                b'M' => self.reverse_index(),    // RI
                b'7' => self.save_cursor(),      // DECSC
                b'8' => self.restore_cursor(),   // DECRC
                b'=' => self.keypad_app = true,  // DECKPAM
                b'>' => self.keypad_app = false, // DECKPNM
                _ => {}
            },
            Some(b'(') => self.g0 = charset_from(byte), // designate G0
            Some(b')') => self.g1 = charset_from(byte), // designate G1
            Some(b'#') if byte == b'8' => self.decaln(), // DECALN
            _ => {}
        }
    }

    fn dcs_dispatch(&mut self, _params: &Params, intermediates: &[u8], action: u8, data: &[u8]) {
        // Both of these are questions, and both were previously swallowed whole: DCS had
        // no callback at all, so a program that asked one waited for an answer that was
        // never coming.
        match (intermediates, action) {
            ([b'$'], b'q') => self.decrqss(data), // "what is this setting?"
            ([b'+'], b'q') => self.xtgettcap(data), // "what does your terminfo say?"
            _ => {}
        }
    }

    fn osc_dispatch(&mut self, data: &[u8], bel_terminated: bool) {
        // OSC Ps ; Pt
        let mut parts = data.splitn(2, |&b| b == b';');
        let ps = parts.next().unwrap_or(&[]);
        let pt = parts.next().unwrap_or(&[]);
        match ps {
            b"0" | b"2" => self.set_title(String::from_utf8_lossy(pt).into_owned()),
            b"4" => self.osc_palette(pt, bel_terminated),
            b"8" => self.set_hyperlink(pt),
            b"10" => self.osc_named_color(NamedColor::Foreground, pt, bel_terminated),
            b"11" => self.osc_named_color(NamedColor::Background, pt, bel_terminated),
            b"12" => self.osc_named_color(NamedColor::Cursor, pt, bel_terminated),
            b"7" => self.osc_cwd(pt),
            b"52" => self.osc_clipboard(pt),
            b"104" => self.osc_reset_palette(pt),
            b"133" => self.osc_shell_mark(pt),
            b"110" => self.theme.fg = Theme::default().fg,
            b"111" => self.theme.bg = Theme::default().bg,
            b"112" => self.theme.cursor = Theme::default().cursor,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
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
}
