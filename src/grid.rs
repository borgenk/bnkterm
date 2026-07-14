//! The screen model: the cell grid, cursor, scrollback, and the operations the
//! VT parser drives. This is the type `bytes -> vt::Parser -> grid::Screen` feeds,
//! and where terminal *correctness* lives. The parser holds no grid state and the
//! grid holds no parser state: this file knows only semantic operations (print a
//! rune, move the cursor, erase, scroll), never bytes.
//!
//! Structure, chosen for the access pattern (the structure is the
//! optimization):
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
//! Two shapes of the cell earn a comment because they corrupt a screen when
//! gotten wrong:
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

use crate::color::Color;
use crate::input::{KittyFlags, ModifyOtherKeys};
use crate::mouse::{MouseMode, MouseProtocol};
use crate::vt::Perform;
use crate::width::width;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::fmt;

/// Default lines of scrollback the primary screen keeps. Overridable by config
/// later (phase 4); the alternate screen keeps none.
const DEFAULT_SCROLLBACK: usize = 10_000;

/// Columns between default tab stops.
const TAB_WIDTH: usize = 8;

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
    fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(AbsRow)
    }

    /// The line before this one, or `None` at the very start of the stream.
    fn prev(self) -> Option<Self> {
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

    pub const fn is_empty(self) -> bool {
        self.0 == 0
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
}

/// One row of cells plus its rare combining marks. Cells are contiguous, so a
/// row is cheap to scan and write. Combining marks live in a tiny side list keyed
/// by column so they ride along when the row scrolls and cost nothing on the
/// common all-single-codepoint row.
#[derive(Clone, Debug)]
struct Row {
    cells: Vec<Cell>,
    combining: Vec<(usize, Vec<char>)>,
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

    fn marks_at(&self, col: usize) -> Option<&[char]> {
        self.combining
            .iter()
            .find(|(c, _)| *c == col)
            .map(|(_, m)| m.as_slice())
    }

    fn add_mark(&mut self, col: usize, mark: char) {
        match self.combining.iter_mut().find(|(c, _)| *c == col) {
            Some((_, marks)) => marks.push(mark),
            None => self.combining.push((col, vec![mark])),
        }
    }

    fn clear_marks(&mut self, col: usize) {
        self.combining.retain(|(c, _)| *c != col);
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
            self.combining.retain(|(c, _)| *c < new_cols);
            if let Some(last) = self.cells.last_mut() {
                if last.is_wide_leader() {
                    *last = Cell::BLANK;
                }
            }
        } else if new_cols > old {
            self.cells.resize(new_cols, Cell::BLANK);
        }
    }
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
        r.combining.retain(|(c, _)| *c < start_col || *c >= end_col);
        // As in `set_raw`: a run reaching the final column replaces whatever wrapped out
        // of it. The caller re-wraps this row if the run itself runs off the edge.
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

    /// Resize the buffer to `new_cols` x `new_rows`. Columns truncate or pad every
    /// row (visible and scrollback); rows grow by appending blanks at the bottom
    /// and shrink by dropping rows below the cursor first, then scrolling the top
    /// into scrollback so the cursor stays on screen. The scroll region resets to
    /// the full screen (as xterm does on resize) and the cursor is re-clamped.
    ///
    /// Deliberately does NOT re-wrap soft-wrapped lines: full-screen programs
    /// repaint on `SIGWINCH`, so what matters is a correctly sized, uncorrupted
    /// grid, not re-flowed history. Scrollback re-wrap is a later nicety.
    fn resize(&mut self, new_cols: usize, new_rows: usize) {
        let new_cols = new_cols.max(1);
        let new_rows = new_rows.max(1);

        if new_cols != self.cols {
            for r in self.lines.iter_mut().chain(self.scrollback.iter_mut()) {
                r.resize_cols(new_cols);
            }
            self.cols = new_cols;
            self.tabs = default_tabs(new_cols);
            if self.cursor.col >= new_cols {
                self.cursor.col = new_cols - 1;
            }
        }

        match new_rows.cmp(&self.rows) {
            Ordering::Greater => {
                for _ in 0..(new_rows - self.rows) {
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
    fn clear_line_range(&mut self, row: usize, start: usize, end: usize, blank: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            let hi = end.min(r.cells.len());
            if start < hi {
                if let Some(slice) = r.cells.get_mut(start..hi) {
                    slice.iter_mut().for_each(|c| *c = blank);
                }
                r.combining.retain(|(c, _)| *c < start || *c >= hi);
                if hi == r.cells.len() {
                    r.wrapped = false;
                }
            }
        }
    }

    /// ICH: shift `[col, len)` right by `n`, blanking the `n` opened cells; cells
    /// pushed past the right edge are lost, including whatever wrapped out of the
    /// final column, so the wrap link goes with it.
    fn insert_blanks(&mut self, row: usize, col: usize, n: usize, blank: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            let len = r.cells.len();
            if col >= len {
                return;
            }
            let n = n.min(len - col);
            r.cells.copy_within(col..len - n, col + n);
            if let Some(slice) = r.cells.get_mut(col..col + n) {
                slice.iter_mut().for_each(|c| *c = blank);
            }
            r.combining.retain(|(c, _)| *c < col || *c + n < len);
            r.combining.iter_mut().for_each(|(c, _)| {
                if *c >= col {
                    *c += n
                }
            });
            r.wrapped = false;
        }
    }

    /// DCH: shift `[col+n, len)` left by `n`, blanking the `n` cells at the right.
    /// Blanking the tail ends the line there, so the wrap link goes too.
    fn delete_chars(&mut self, row: usize, col: usize, n: usize, blank: Cell) {
        if let Some(r) = self.lines.get_mut(row) {
            let len = r.cells.len();
            if col >= len {
                return;
            }
            let n = n.min(len - col);
            r.cells.copy_within(col + n..len, col);
            if let Some(slice) = r.cells.get_mut(len - n..len) {
                slice.iter_mut().for_each(|c| *c = blank);
            }
            r.combining.retain(|(c, _)| *c < col || *c >= col + n);
            r.combining.iter_mut().for_each(|(c, _)| {
                if *c >= col + n {
                    *c -= n
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
    /// The kitty keyboard protocol's flag stack. The protocol is a *stack* on purpose:
    /// a full-screen program pushes the flags it wants on entry and pops them on exit,
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

    /// Take the bytes queued to write back to the child (DA/DSR answers), leaving
    /// the queue empty. Empty in the common case (no query since the last parse),
    /// so the app can call it after every parse cheaply.
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

    /// Combining marks attached to visible `(row, col)`, if any.
    pub fn marks_at(&self, row: usize, col: usize) -> Option<&[char]> {
        self.active().line(row).and_then(|r| r.marks_at(col))
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
    fn abs_marks(&self, row: AbsRow, col: usize) -> Option<&[char]> {
        self.active().abs_row(row).and_then(|r| r.marks_at(col))
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

    /// Combining marks at display `(row, col)` honouring the scroll offset.
    pub fn view_marks(&self, row: usize, col: usize) -> Option<&[char]> {
        let off = self.view_offset();
        if off == 0 {
            return self.marks_at(row, col);
        }
        self.active()
            .view_row(row, off)
            .and_then(|r| r.marks_at(col))
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
            let mut line = String::new();
            for col in first..=last.min(last_col) {
                let cell = self.abs_cell(row, col);
                if cell.is_wide_spacer() {
                    continue;
                }
                line.push(cell.rune);
                if let Some(marks) = self.abs_marks(row, col) {
                    line.extend(marks);
                }
            }
            out.push_str(line.trim_end_matches(' '));
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
                if let Some(marks) = self.view_marks(r, c) {
                    probe.text.extend(marks);
                }
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
            if let Some(marks) = r.marks_at(col) {
                s.extend(marks);
            }
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
        let cw = usize::from(width(c));
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
        let leader = Cell {
            rune: c,
            fg: pen.fg,
            bg: pen.bg,
            attrs: if cw == 2 {
                pen.attrs | Attrs::WIDE_LEADER
            } else {
                pen.attrs
            },
            link: pen.link,
        };
        {
            let b = self.active_mut();
            b.write_cell(row, col, leader);
            if cw == 2 {
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

    /// TBC: clear the tab stop at the cursor (mode 0) or all stops (mode 3).
    pub fn clear_tab_stop(&mut self, mode: u16) {
        let b = self.active_mut();
        match mode {
            3 => b.tabs.iter_mut().for_each(|t| *t = false),
            _ => {
                let col = b.cursor.col;
                if let Some(stop) = b.tabs.get_mut(col) {
                    *stop = false;
                }
            }
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
    pub fn erase_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (row, col) = (b.cursor.row, b.cursor.col);
        b.clear_line_range(row, col, col + n.max(1), blank);
    }

    // ---- insert / delete ----------------------------------------------------

    /// ICH: insert `n` blanks at the cursor, shifting the rest of the line right.
    pub fn insert_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
        let (row, col) = (b.cursor.row, b.cursor.col);
        b.insert_blanks(row, col, n.max(1), blank);
    }

    /// DCH: delete `n` cells at the cursor, shifting the rest of the line left.
    pub fn delete_chars(&mut self, n: usize) {
        let blank = self.blank_cell();
        let b = self.active_mut();
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

    /// DECSC: save the cursor, pen, and origin mode.
    pub fn save_cursor(&mut self) {
        let (pen, origin) = (self.pen, self.origin_mode);
        let b = self.active_mut();
        b.saved = Some(Saved {
            cursor: b.cursor,
            pen,
            origin,
        });
    }

    /// DECRC: restore what DECSC saved, or home the cursor if nothing was saved.
    pub fn restore_cursor(&mut self) {
        let saved = self.active().saved;
        match saved {
            Some(s) => {
                self.pen = s.pen;
                self.origin_mode = s.origin;
                let b = self.active_mut();
                b.cursor = s.cursor;
                b.cursor.row = b.cursor.row.min(b.rows - 1);
                b.cursor.col = b.cursor.col.min(b.cols - 1);
            }
            None => self.move_to(0, 0),
        }
    }

    // ---- rendition (SGR) ----------------------------------------------------

    /// Apply an SGR sequence, updating the pen. An empty parameter list is a
    /// reset (SGR 0). Handles ANSI-16, bright, 256 (`38;5;n`), and truecolor
    /// (`38;2;r;g;b`) for both foreground (38) and background (48).
    pub fn sgr(&mut self, params: &[u16]) {
        if params.is_empty() {
            self.pen.reset_rendition();
            return;
        }
        let mut i = 0;
        while i < params.len() {
            let p = params[i];
            match p {
                0 => self.pen.reset_rendition(),
                1 => self.pen.attrs.insert(Attrs::BOLD),
                2 => self.pen.attrs.insert(Attrs::DIM),
                3 => self.pen.attrs.insert(Attrs::ITALIC),
                4 => self.pen.attrs.insert(Attrs::UNDERLINE),
                7 => self.pen.attrs.insert(Attrs::REVERSE),
                8 => self.pen.attrs.insert(Attrs::HIDDEN),
                9 => self.pen.attrs.insert(Attrs::STRIKE),
                21 => self.pen.attrs.remove(Attrs::BOLD),
                22 => self.pen.attrs.remove(Attrs::BOLD | Attrs::DIM),
                23 => self.pen.attrs.remove(Attrs::ITALIC),
                24 => self.pen.attrs.remove(Attrs::UNDERLINE),
                27 => self.pen.attrs.remove(Attrs::REVERSE),
                28 => self.pen.attrs.remove(Attrs::HIDDEN),
                29 => self.pen.attrs.remove(Attrs::STRIKE),
                30..=37 => self.pen.fg = Color::Ansi((p - 30) as u8),
                38 => {
                    let (color, advance) = parse_ext_color(&params[i..]);
                    if let Some(color) = color {
                        self.pen.fg = color;
                    }
                    i += advance;
                }
                39 => self.pen.fg = Color::Default,
                40..=47 => self.pen.bg = Color::Ansi((p - 40) as u8),
                48 => {
                    let (color, advance) = parse_ext_color(&params[i..]);
                    if let Some(color) = color {
                        self.pen.bg = color;
                    }
                    i += advance;
                }
                49 => self.pen.bg = Color::Default,
                90..=97 => self.pen.fg = Color::Ansi((p - 90 + 8) as u8),
                100..=107 => self.pen.bg = Color::Ansi((p - 100 + 8) as u8),
                _ => {}
            }
            i += 1;
        }
    }

    // ---- modes --------------------------------------------------------------

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
                2004 => self.bracketed_paste = enable,
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
    fn device_status(&mut self, params: &[u16], private: u8) {
        let ps = params.first().copied().unwrap_or(0);
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

    /// Enter or leave the alternate screen. Entering clears it and homes the
    /// cursor; the primary buffer is untouched, so leaving reveals it intact.
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
            self.alt.clear_all(blank);
            self.alt.cursor = Cursor::default();
            self.alt.scroll_top = 0;
            self.alt.scroll_bottom = self.alt.rows - 1;
            self.alt.saved = None;
            self.on_alt = true;
        } else {
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

    /// Resize both screens to `cols` x `rows` (clamped to at least 1x1). The app
    /// calls this when the window's pixel size divided by the cell size yields a
    /// new grid; it then sends the child the matching `TIOCSWINSZ`. Content is
    /// preserved and re-clamped, not re-wrapped (see [`Buffer::resize`]).
    pub fn resize(&mut self, cols: usize, rows: usize) {
        let cols = cols.max(1);
        let rows = rows.max(1);
        self.primary.resize(cols, rows);
        self.alt.resize(cols, rows);
        // Scrollback was reindexed; a stale offset would point at the wrong rows, and a
        // row id minted against the old geometry names a row that may not exist (and,
        // since we do not re-wrap, may hold different text at a different column).
        self.view_offset = 0;
        self.break_row_identity();
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

    /// The kitty keyboard flags in force: the top of the stack, or none when the child
    /// has pushed nothing and the legacy encoding applies.
    pub fn kitty_flags(&self) -> KittyFlags {
        self.kitty_stack.last().copied().unwrap_or(KittyFlags::NONE)
    }

    /// The `modifyOtherKeys` level the child asked for.
    pub fn modify_other_keys(&self) -> ModifyOtherKeys {
        self.modify_other_keys
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
    fn kitty_keyboard(&mut self, params: &[u16], private: u8) {
        match private {
            b'?' => {
                self.respond(b"\x1b[?");
                let flags = self.kitty_flags().bits();
                push_decimal(&mut self.responses, flags);
                self.respond(b"u");
            }
            b'=' => {
                let flags = KittyFlags::from_request(params.first().copied().unwrap_or(0));
                let mode = params.get(1).copied().unwrap_or(1);
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
                let flags = KittyFlags::from_request(params.first().copied().unwrap_or(0));
                if self.kitty_stack.len() >= KITTY_STACK_LIMIT {
                    self.kitty_stack.remove(0);
                }
                self.kitty_stack.push(flags);
            }
            b'<' => {
                let count = usize::from(params.first().copied().unwrap_or(1).max(1));
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
    fn xtmodkeys(&mut self, params: &[u16]) {
        if params.first().copied() != Some(4) {
            return;
        }
        self.modify_other_keys = match params.get(1) {
            Some(&level) => ModifyOtherKeys::from_param(level),
            None => ModifyOtherKeys::Off,
        };
    }

    /// DECALN: fill the whole screen with 'E' and home the cursor (a vttest
    /// alignment pattern; useful for confirming glyph placement early).
    pub fn decaln(&mut self) {
        let b = self.active_mut();
        b.clear_all(Cell::new('E'));
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
fn default_tabs(cols: usize) -> Vec<bool> {
    (0..cols).map(|c| c % TAB_WIDTH == 0).collect()
}

/// Whether a rune bounds a double-click word selection: whitespace, or one of the
/// brackets/quotes that fence a token. The set matches the common terminal default
/// (wezterm/xterm), so double-clicking selects a path or URL whole but stops at a
/// delimiter. A blank cell's rune is a space, so it bounds a word too.
fn is_word_boundary(c: char) -> bool {
    c.is_whitespace() || matches!(c, '{' | '}' | '[' | ']' | '(' | ')' | '"' | '\'' | '`')
}

/// Clamp an SGR color component (`0..=255`) to a byte; the parser already bounds
/// parameters, so this only guards a malformed stream, never a valid one.
fn sgr_component(v: u16) -> u8 {
    v.min(255) as u8
}

/// Parse a `38`/`48` extended-color introducer at the front of `params`,
/// returning the color and how many *extra* parameters it consumed (0 if the
/// form is unrecognized). Handles `;5;n` (indexed) and `;2;r;g;b` (truecolor).
fn parse_ext_color(params: &[u16]) -> (Option<Color>, usize) {
    match params.get(1).copied() {
        Some(5) => {
            let n = params.get(2).copied().unwrap_or(0);
            (Some(Color::Indexed(sgr_component(n))), 2)
        }
        Some(2) => {
            let r = sgr_component(params.get(2).copied().unwrap_or(0));
            let g = sgr_component(params.get(3).copied().unwrap_or(0));
            let b = sgr_component(params.get(4).copied().unwrap_or(0));
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
fn csi_arg(params: &[u16], i: usize) -> u16 {
    params.get(i).copied().unwrap_or(0)
}

/// A CSI count parameter (for cursor moves, repeat counts): absent or 0 means 1.
fn csi_count(params: &[u16], i: usize) -> usize {
    usize::from(csi_arg(params, i).max(1))
}

/// A 1-based CSI position parameter as a 0-based index (default 1 maps to 0).
fn csi_index(params: &[u16], i: usize) -> usize {
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

    fn print_ascii(&mut self, bytes: &[u8]) {
        // The bulk write assumes each byte is its own glyph placed one column
        // apart. That holds only under the identity (ASCII) charset and outside
        // insert mode; DEC Special Graphics remaps each byte to a line-drawing
        // glyph, and insert mode shifts the row per char. Fall back to the per-char
        // path (glyph mapping + inherent print) for both, keeping this identical to
        // calling `Perform::print` on each byte.
        if self.insert_mode || self.active_charset() != Charset::Ascii {
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
            _ => {}                          // BEL, NUL, ...: nothing to draw
        }
    }

    fn csi_dispatch(&mut self, params: &[u16], intermediates: &[u8], private: u8, action: u8) {
        // DECSCUSR (`CSI Ps SP q`) is the one CSI-with-intermediate we act on; the
        // rest are dropped rather than misreading the final byte.
        if intermediates == [b' '] && action == b'q' {
            self.set_cursor_style(params.first().copied().unwrap_or(0));
            return;
        }
        if !intermediates.is_empty() {
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
            b'J' => self.erase_display(csi_arg(params, 0)),
            b'K' => self.erase_line(csi_arg(params, 0)),
            b'L' => self.insert_lines(csi_count(params, 0)),
            b'M' => self.delete_lines(csi_count(params, 0)),
            b'@' => self.insert_chars(csi_count(params, 0)),
            b'P' => self.delete_chars(csi_count(params, 0)),
            b'X' => self.erase_chars(csi_count(params, 0)),
            b'S' => self.scroll_up(csi_count(params, 0)),
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
            b'm' if private == 0 => self.sgr(params),
            b'm' if private == b'>' => self.xtmodkeys(params),
            b'h' => {
                let dec = private == b'?';
                for &m in params {
                    self.set_mode(m, dec, true);
                }
            }
            b'l' => {
                let dec = private == b'?';
                for &m in params {
                    self.set_mode(m, dec, false);
                }
            }
            b's' if private == 0 && params.is_empty() => self.save_cursor(),
            b'u' if private == 0 && params.is_empty() => self.restore_cursor(),
            // The kitty keyboard protocol shares SCORC's final byte and is told apart by
            // its private marker, so a bare `CSI u` still restores the cursor.
            b'u' if private != 0 => self.kitty_keyboard(params, private),
            b'g' => self.clear_tab_stop(csi_arg(params, 0)),
            b'c' => self.device_attributes(private),
            b'n' => self.device_status(params, private),
            _ => {}
        }
    }

    fn esc_dispatch(&mut self, intermediates: &[u8], byte: u8) {
        match intermediates.first().copied() {
            None => match byte {
                b'c' => self.reset(),            // RIS
                b'D' => self.line_feed(),        // IND
                b'E' => self.next_line(),        // NEL
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

    fn osc_dispatch(&mut self, data: &[u8]) {
        // OSC Ps ; Pt — we handle 0 (icon + title), 2 (title), and 8 (hyperlink).
        let mut parts = data.splitn(2, |&b| b == b';');
        let ps = parts.next().unwrap_or(&[]);
        let pt = parts.next().unwrap_or(&[]);
        match ps {
            b"0" | b"2" => self.set_title(String::from_utf8_lossy(pt).into_owned()),
            b"8" => self.set_hyperlink(pt),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(s.marks_at(0, 0), Some(['\u{0301}'].as_slice()));
        assert_eq!(s.row_string(0).trim_end(), "e\u{0301}");
    }

    #[test]
    fn combining_mark_after_wide_char_hits_the_leader() {
        let mut s = Screen::new(10, 3);
        s.print('か'); // wide: leader at 0, spacer at 1, cursor at 2
        s.print('\u{3099}'); // combining voiced sound mark -> が
                             // The mark composes onto the leader (col 0), not the spacer (col 1).
        assert_eq!(s.marks_at(0, 0), Some(['\u{3099}'].as_slice()));
        assert_eq!(s.marks_at(0, 1), None);
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

    #[test]
    fn sgr_sets_the_pen_and_print_applies_it() {
        let mut s = Screen::new(10, 1);
        s.sgr(&[1, 31]); // bold, red foreground
        s.print('x');
        let c = s.cell(0, 0);
        assert!(c.attrs.contains(Attrs::BOLD));
        assert_eq!(c.fg, Color::Ansi(1));
        s.sgr(&[0]); // reset
        s.print('y');
        let c = s.cell(0, 1);
        assert!(c.attrs.is_empty());
        assert_eq!(c.fg, Color::Default);
    }

    #[test]
    fn sgr_extended_colors() {
        let mut s = Screen::new(10, 1);
        s.sgr(&[38, 5, 200]); // 256-color foreground
        assert_eq!(s.pen.fg, Color::Indexed(200));
        s.sgr(&[48, 2, 10, 20, 30]); // truecolor background
        assert_eq!(s.pen.bg, Color::Rgb(10, 20, 30));
        s.sgr(&[90]); // bright black foreground -> ANSI 8
        assert_eq!(s.pen.fg, Color::Ansi(8));
    }

    #[test]
    fn alt_screen_is_separate_and_restores() {
        let mut s = Screen::new(6, 2);
        print_str(&mut s, "main");
        s.set_mode(1049, true, true); // enter alt, save cursor
        assert!(s.is_alt());
        assert_eq!(s.dump().trim(), ""); // alt starts cleared
        print_str(&mut s, "alt");
        assert_eq!(s.row_string(0).trim_end(), "alt");
        s.set_mode(1049, true, false); // leave alt, restore cursor
        assert!(!s.is_alt());
        assert_eq!(s.row_string(0).trim_end(), "main"); // primary intact
        assert_eq!(s.cursor(), (0, 4)); // cursor restored to after "main"
    }

    #[test]
    fn save_and_restore_cursor() {
        let mut s = Screen::new(10, 5);
        s.move_to(2, 4);
        s.sgr(&[1]);
        s.save_cursor();
        s.move_to(0, 0);
        s.sgr(&[0]);
        s.restore_cursor();
        assert_eq!(s.cursor(), (2, 4));
        assert!(s.pen.attrs.contains(Attrs::BOLD));
    }

    #[test]
    fn reset_returns_to_power_on_state() {
        let mut s = Screen::new(6, 3);
        s.sgr(&[31]);
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
        feed(&mut s, b"\x1b[?1049h");
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

    #[test]
    fn wide_and_combining_via_parser() {
        let mut s = Screen::new(10, 1);
        feed(&mut s, "aか\u{3099}b".as_bytes()); // 'a', wide か + combining mark, 'b'
        assert_eq!(s.cell(0, 0).rune, 'a');
        assert_eq!(s.cell(0, 1).rune, 'か');
        assert!(s.cell(0, 1).is_wide_leader());
        assert_eq!(s.marks_at(0, 1), Some(['\u{3099}'].as_slice()));
        assert_eq!(s.cell(0, 3).rune, 'b');
    }

    #[test]
    fn random_bytes_into_screen_keep_invariants() {
        // The whole pipeline under fuzz: ~1.2 MB of arbitrary bytes through the
        // parser into a real Screen must never panic and must keep the grid's
        // structural invariants (a seeded LCG, no crate).
        let mut s = Screen::new(24, 8);
        let mut p = crate::vt::Parser::new();
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_1234;
        let mut buf = [0u8; 4096];
        for _ in 0..300 {
            for b in buf.iter_mut() {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *b = (seed >> 33) as u8;
            }
            p.advance_bytes(&mut s, &buf);
            let (cols, rows) = s.dimensions();
            assert_eq!((cols, rows), (24, 8));
            let (cr, cc) = s.cursor();
            assert!(cr < rows && cc < cols, "cursor {cr},{cc} out of bounds");
            assert_eq!(s.primary.lines.len(), rows);
            assert_eq!(s.alt.lines.len(), rows);
            assert!(s.primary.scrollback.len() <= DEFAULT_SCROLLBACK);
        }
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
    fn narrowing_truncates_and_widening_pads() {
        let mut s = Screen::new(8, 1);
        feed(&mut s, b"abcdefgh");
        s.resize(4, 1);
        assert_eq!(s.row_string(0).trim_end(), "abcd");
        s.resize(8, 1);
        // Widening pads with blanks; the truncated tail is gone, not restored.
        assert_eq!(s.cell(0, 4).rune, ' ');
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
    fn resize_pins_the_view_to_the_bottom() {
        let mut s = Screen::new(5, 3);
        feed(&mut s, b"a\r\nb\r\nc\r\nd\r\ne");
        s.scroll_view_up(2);
        assert!(s.is_scrolled());
        s.resize(8, 4);
        assert_eq!(s.view_offset(), 0, "a resize reindexes history and re-pins");
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
    fn a_resize_keeps_a_soft_wrapped_line_joined() {
        // The regression that started this: the wrap link used to live on the row's
        // last cell, so widening stranded it mid-row and narrowing truncated it away.
        // Either way the next copy across the wrap grew a newline out of nowhere.
        for new_cols in [4, 6, 12, 3] {
            let mut s = Screen::new(4, 3);
            feed(&mut s, b"abcdef"); // wraps: "abcd" | "ef"
            s.resize(new_cols, 3);
            assert!(s.row_wraps(0), "{new_cols} cols: the wrap link survives");
        }
    }

    #[test]
    fn a_resized_wrapped_line_copies_unbroken() {
        // The user-visible half of the same bug: drag over an old wrapped line after
        // the window has been resized, and the clipboard must not break mid-sentence.
        let mut s = Screen::new(4, 3);
        feed(&mut s, b"abcdef"); // wraps: "abcd" | "ef"
        s.resize(6, 3);
        assert_eq!(s.selection_text(at(&s, 0, 0), at(&s, 1, 1)), "abcdef");
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
        // An application asking for flags we do not implement (here 0b11111: event
        // types, alternate keys, all-keys, associated text) is told the truth about what
        // it will get. Reporting them as set would leave it waiting for key-release
        // events that never come.
        feed(&mut s, b"\x1b[>31u\x1b[?u");
        assert_eq!(s.take_responses(), b"\x1b[?1u");
        assert_eq!(s.kitty_flags(), KittyFlags::DISAMBIGUATE);
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
