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
//!     rune and [`CellWidth::Leader`]; the right column is a [`CellWidth::Spacer`]
//!     placeholder the cursor steps over and the renderer skips. Overwriting either half
//!     cleans up its orphaned partner. The role is *layout*, not rendition, which is why
//!     it sits on the cell rather than in [`Attrs`]: both halves share one [`Style`].
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

    /// Bits 7-9: how the underline is drawn ([`UnderlineStyle`]), not *whether* it is —
    /// that is still [`UNDERLINE`](Self::UNDERLINE), and the style only means anything
    /// alongside it.
    ///
    /// It lives in the bitfield's spare bits because it *fits*, and that is the whole
    /// design decision: carrying the shape in a field of its own would have widened every
    /// cell in the grid to decorate the handful an editor squiggles. The underline
    /// *colour* (SGR 58) does not fit, which is exactly why it is still parsed, consumed
    /// and dropped rather than stored.
    const UNDERLINE_STYLE_SHIFT: u16 = 7;
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
    const ALL: [(Attrs, &'static str); 7] = [
        (Attrs::BOLD, "BOLD"),
        (Attrs::DIM, "DIM"),
        (Attrs::ITALIC, "ITALIC"),
        (Attrs::UNDERLINE, "UNDERLINE"),
        (Attrs::REVERSE, "REVERSE"),
        (Attrs::STRIKE, "STRIKE"),
        (Attrs::HIDDEN, "HIDDEN"),
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
    /// into the resolved colour, which the painter compares separately, and `HIDDEN`
    /// draws as a space and so leaves a run intact. The wide-pair state is not here at
    /// all any more — it is [`CellWidth`], on the cell rather than in the rendition —
    /// and it breaks a run by standing a glyph alone, before any of this is asked.
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

/// The longest URL that can be interned. foot and wezterm both settle around 2 KB, and
/// nothing a human clicks is close: the OSC 8 anchors real tools emit (`ls --hyperlink`,
/// `gcc`'s diagnostics, `delta`) are file URIs of a few dozen to a couple of hundred
/// bytes.
///
/// A cap is needed because the only other bound on one URL is `OSC_MAX` (4096), and
/// **count** was the only thing capped. 65534 distinct 4 KB URIs, one anchored cell each,
/// pins about 537 MB that `collect_links` can never reclaim — because each one is stored
/// twice (once in `urls`, once as the `index` key) and every one is live. That is forty
/// times what the scrollback's own cells cost, from a child that merely prints.
const LINK_URL_MAX: usize = 2048;

/// The most bytes of URL text the table may hold, counting **both** stored copies.
///
/// [`LINK_URL_MAX`] alone leaves the worst case at 65534 × 2 KB × 2 ≈ 260 MB, which is
/// still out of proportion to a terminal, so the table carries a byte budget as well as
/// a count. The number is chosen against what the *text* costs: a full scrollback was
/// ~13 MB of cells when this was set, so links were allowed roughly half of that.
///
/// Packing the cell to 8 bytes has since taken that yardstick to ~9.9 MiB, so the same
/// 8 MB is now roughly *all* of what the cells cost rather than half. Left where it is
/// deliberately — it is still generous rather than tight, which is the property that
/// matters (see below), and shrinking it would newly turn away sessions that fit before.
/// Worth revisiting if the cells shrink again.
///
/// It is generous rather than tight, and deliberately: at a realistic ~60-byte file URI
/// this still admits the entire id space (65534 × 60 × 2 ≈ 7.9 MB), so no legitimate
/// session reaches it. Only a stream of long distinct URIs does, and that one degrades
/// exactly as a spent id space does — the newest link is not clickable, the text still
/// reads.
const LINK_BYTES_MAX: usize = 8 * 1024 * 1024;

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
/// printed a thousand times is interned once. That duplication buys both directions
/// their natural cost, which a single structure could not — and it is why the table is
/// bounded three ways rather than one: by count ([`LINK_LIMIT`], the id space), by the
/// length of any single URL ([`LINK_URL_MAX`]), and by total stored bytes
/// ([`LINK_BYTES_MAX`]). Count alone was not a bound on memory, because the child picks
/// the length too.
#[derive(Default)]
struct LinkTable {
    urls: Vec<String>,
    index: HashMap<String, LinkId>,
    /// Bytes of URL text held, counting both copies. Tracked rather than summed so the
    /// budget check stays O(1) on a path a child can drive.
    bytes: usize,
    /// Where the grid was the last time a sweep reclaimed nothing: `(epoch, evicted)`.
    ///
    /// A sweep walks every cell in both buffers plus the whole scrollback, which is fine
    /// once but ruinous per anchor — and "per anchor" is exactly where a full table of
    /// *live* links lands, because every subsequent intern fails and asks for another
    /// sweep. That was always true of a spent id space; adding a byte budget made the
    /// same state reachable thirty times sooner, so it stops being theoretical.
    ///
    /// Links die when the cells citing them die, and the two things that kill them in
    /// bulk are a row aging out of history (`evicted`) and the row epoch breaking (an
    /// alt-screen switch, `ED 2`, a reset). Neither has moved means nothing worth
    /// sweeping for has happened. A narrower death — one `EL` freeing one link — is
    /// missed until then, which is a degradation of a state that is already a
    /// degradation.
    swept_dry_at: Option<(RowEpoch, u64)>,
}

impl LinkTable {
    /// The id for `url`, interning it if this is the first time we have seen it.
    /// `None` when the table is full — of ids or of bytes — which is the caller's cue to
    /// collect the dead ids and try once more ([`Screen::intern_link`]). A sweep can free
    /// either, so both failures are worth retrying after one.
    ///
    /// A URL longer than [`LINK_URL_MAX`] never reaches here: [`Screen::intern_link`]
    /// turns it away first, because no amount of collecting would make room for it and
    /// the sweep walks every cell in both buffers.
    fn intern(&mut self, url: &str) -> Option<LinkId> {
        if let Some(&id) = self.index.get(url) {
            return Some(id);
        }
        if self.urls.len() >= LINK_LIMIT {
            return None;
        }
        let cost = url.len().checked_mul(2)?;
        if self.bytes.saturating_add(cost) > LINK_BYTES_MAX {
            return None;
        }
        // Ids are 1-based and the table is capped below `u16::MAX`, so this fits.
        let id = LinkId(u16::try_from(self.urls.len() + 1).ok()?);
        self.urls.push(url.to_string());
        self.index.insert(url.to_string(), id);
        self.bytes = self.bytes.saturating_add(cost);
        Some(id)
    }

    /// The heap this table holds: the URL text (already tracked, both copies) plus the
    /// two containers' own capacity. The `HashMap` entry is charged at key-plus-value,
    /// which understates its control bytes slightly and is close enough for a table
    /// that is empty in every session without hyperlinks.
    fn storage_bytes(&self) -> usize {
        self.bytes
            + self.urls.capacity() * std::mem::size_of::<String>()
            + self.index.capacity()
                * (std::mem::size_of::<String>() + std::mem::size_of::<LinkId>())
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
        // Rebuilt from the survivors below, so the byte budget frees exactly what the
        // sweep dropped.
        self.bytes = 0;
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
            self.bytes = self.bytes.saturating_add(url.len().saturating_mul(2));
            self.index.insert(url.clone(), id);
            self.urls.push(url);
        }
        remap
    }
}

/// How a cell sits in a wide (2-column) character, which is a fact about *layout*
/// rather than about rendition.
///
/// It used to be two bits in [`Attrs`], and moving it out is what lets a rendition be
/// interned: the two halves of a wide glyph share one SGR style but are not the same
/// cell, so leaving the pair bits in the style would have split every style in three
/// and made the wide checks — which the cursor asks constantly — a table lookup.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CellWidth {
    /// One column, the overwhelming majority.
    #[default]
    Narrow,
    /// The left half of a wide character; carries the rune.
    Leader,
    /// The right half of a wide character; a placeholder the cursor skips.
    Spacer,
}

/// A cell's rendition: the colours and attributes an SGR sequence sets, and nothing
/// that varies per cell within a run.
///
/// This is the unit that gets interned. It is deliberately *not* on the stored cell:
/// most cells are unstyled, and where there is styling it is overwhelmingly shared, so
/// a screenful of coloured `ls` output resolves to a handful of these.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Style {
    pub fg: Color,
    pub bg: Color,
    pub attrs: Attrs,
}

/// Hashed as two machine words rather than field by field, which is worth the manual
/// impl: this runs once per SGR sequence, and escape-dense output (a coloured `ls`, an
/// `htop` redraw) is nothing *but* SGR sequences.
///
/// The derived impl walks the structure — a discriminant plus up to three payload bytes
/// per [`Color`], then the attributes — so a rendition arrives at the hasher as nine
/// separate writes. Packing each colour into the `u32` it already fits in turns that
/// into two, and measurably: it is the difference between the interning showing up in
/// `parse_escape` and not.
///
/// Deliberately still the default (SipHash) hasher underneath. A `Style` is built from
/// bytes the child chooses, so the map is keyed by attacker-controlled input, exactly
/// like [`LinkTable`]'s. Swapping in a fast non-cryptographic hash here would trade a
/// bounded, measured cost for an unbounded collision attack on the parse path.
impl std::hash::Hash for Style {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(u64::from(self.fg.key()) << 32 | u64::from(self.bg.key()));
        state.write_u16(self.attrs.0);
    }
}

/// A rendition interned in the screen's [`StyleTable`], or [`StyleId::DEFAULT`] for the
/// unstyled cells that are most of any grid.
///
/// The same trade [`LinkId`] makes, for the same reason and with the same machinery: a
/// stored cell holds two bytes naming a rendition instead of ten bytes spelling one
/// out, which is what takes the grid from 16 bytes a cell to 8. Zero is the default
/// rendition, so a blank cell is unstyled without anyone saying so, and the table never
/// has to hold an entry for the commonest case.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct StyleId(u16);

impl StyleId {
    /// Default colours, no attributes. Never stored in the table.
    pub const DEFAULT: StyleId = StyleId(0);
}

/// How many distinct renditions one screen can hold at once, beyond the default. The
/// id space is a `u16` with zero reserved, so this is all of it; running out is not
/// fatal (see [`Screen::collect_styles`]).
const STYLE_LIMIT: usize = u16::MAX as usize - 1;

/// The renditions the grid's cells cite by [`StyleId`].
///
/// Reclaimed by mark-and-sweep rather than reference counting, which is the same call
/// [`LinkTable`] makes and for a sharper version of the same reason. A refcount would
/// tax *every cell write* — the hottest path there is — to reclaim a `u16` id space
/// that a real session never comes close to exhausting; a sweep costs one walk over
/// the grid, once, on the rare occasion the table actually fills. Emacs reaches the
/// same conclusion about realized faces, and for the same reason declines to track
/// dependencies at all: it flushes the cache wholesale instead.
///
/// The cost of choosing sweep is that the table is a high-water mark between sweeps
/// rather than a live count. Bounded at [`STYLE_LIMIT`] entries, which is small against
/// what the cells themselves cost.
#[derive(Default)]
struct StyleTable {
    /// Interned renditions; `styles[i]` is `StyleId(i + 1)`, since zero is the default.
    styles: Vec<Style>,
    index: HashMap<Style, StyleId>,
    /// Where the grid was the last time a sweep reclaimed nothing, exactly as
    /// [`LinkTable::swept_dry_at`] and for the same reason: a full table of *live*
    /// styles must not buy a fresh walk over every cell with each new rendition.
    swept_dry_at: Option<(RowEpoch, u64)>,
}

impl StyleTable {
    /// The id for `style`, interning it if it is new. `None` when the table is full,
    /// which is the caller's cue to sweep and try once more.
    fn intern(&mut self, style: Style) -> Option<StyleId> {
        if style == Style::default() {
            return Some(StyleId::DEFAULT);
        }
        if let Some(&id) = self.index.get(&style) {
            return Some(id);
        }
        if self.styles.len() >= STYLE_LIMIT {
            return None;
        }
        // Ids are 1-based and the table is capped below `u16::MAX`, so this fits.
        let id = StyleId(u16::try_from(self.styles.len() + 1).ok()?);
        self.styles.push(style);
        self.index.insert(style, id);
        Some(id)
    }

    /// The rendition behind `id`. An id this table does not hold resolves to the
    /// default rather than trapping, which keeps every cell read total.
    fn get(&self, id: StyleId) -> Style {
        match usize::from(id.0).checked_sub(1) {
            Some(slot) => self.styles.get(slot).copied().unwrap_or_default(),
            None => Style::default(),
        }
    }

    /// Drop every rendition `live` does not mark and renumber the survivors, returning
    /// the old-id → new-id map (indexed by the old id's raw value) the caller must then
    /// apply to every cell it kept. Slot zero is [`StyleId::DEFAULT`] and maps to itself.
    fn compact(&mut self, live: &[bool]) -> Vec<StyleId> {
        let mut remap = vec![StyleId::DEFAULT; self.styles.len() + 1];
        let old = std::mem::take(&mut self.styles);
        self.index.clear();
        // Counts only survivors, so it is bounded by the table we came in with and
        // cannot pass `STYLE_LIMIT`, let alone wrap.
        let mut next: u16 = 0;
        for (slot, style) in old.into_iter().enumerate() {
            if live.get(slot + 1) != Some(&true) {
                continue;
            }
            next += 1;
            let id = StyleId(next);
            if let Some(entry) = remap.get_mut(slot + 1) {
                *entry = id;
            }
            self.index.insert(style, id);
            self.styles.push(style);
        }
        remap
    }

    /// The heap this table holds: the interned renditions and the lookup index.
    fn storage_bytes(&self) -> usize {
        self.styles.capacity() * std::mem::size_of::<Style>()
            + self.index.capacity()
                * (std::mem::size_of::<Style>() + std::mem::size_of::<StyleId>())
    }
}

/// A cell as it is **stored**: eight bytes, which is what a screenful plus ten thousand
/// rows of scrollback is actually made of.
///
/// ```text
///   packed: u32                      style: u16   link: u16
///   ┌──────────────────────┬───────┐  ┌────────┐  ┌────────┐
///   │ scalar (bits 0..21)  │ w 2b  │  │ StyleId│  │ LinkId │
///   └──────────────────────┴───────┘  └────────┘  └────────┘
/// ```
///
/// The rune and the wide-pair state share a `u32` because a Unicode scalar needs only
/// 21 of its bits, so [`CellWidth`] rides in the space `char` was already wasting and
/// costs the grid nothing. The rendition is two bytes naming an entry in the screen's
/// [`StyleTable`] instead of ten bytes spelling one out.
///
/// Reading one back out needs the table, so the grid hands callers a resolved [`Cell`]
/// and keeps this type to itself.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PackedCell {
    packed: u32,
    style: StyleId,
    pub(super) link: LinkId,
}

impl PackedCell {
    const SCALAR_MASK: u32 = (1 << 21) - 1;
    const WIDTH_SHIFT: u32 = 21;

    /// The blank cell: a space in the default rendition. The grid's initial fill; an
    /// erase writes a space in the *current* bg, which is a different cell.
    pub(super) const BLANK: PackedCell = PackedCell {
        packed: ' ' as u32,
        style: StyleId::DEFAULT,
        link: LinkId::NONE,
    };

    pub(super) fn new(rune: char, width: CellWidth, style: StyleId, link: LinkId) -> Self {
        let w = match width {
            CellWidth::Narrow => 0,
            CellWidth::Leader => 1,
            CellWidth::Spacer => 2,
        };
        PackedCell {
            packed: (rune as u32 & PackedCell::SCALAR_MASK) | (w << PackedCell::WIDTH_SHIFT),
            style,
            link,
        }
    }

    /// A single-column cell carrying `rune` in the default rendition and no link.
    pub(super) fn plain(rune: char) -> Self {
        PackedCell::new(rune, CellWidth::Narrow, StyleId::DEFAULT, LinkId::NONE)
    }

    /// The base rune. Total by construction: the scalar was a valid `char` when it was
    /// packed and the mask cannot turn it into a surrogate, but an unconvertible value
    /// yields a space rather than trapping, because no grid read may panic.
    pub(super) fn rune(self) -> char {
        char::from_u32(self.packed & PackedCell::SCALAR_MASK).unwrap_or(' ')
    }

    pub(super) fn width(self) -> CellWidth {
        match self.packed >> PackedCell::WIDTH_SHIFT {
            1 => CellWidth::Leader,
            2 => CellWidth::Spacer,
            _ => CellWidth::Narrow,
        }
    }

    pub(super) fn style_id(self) -> StyleId {
        self.style
    }

    /// This cell moved to a different place in a wide pair, keeping its rune and
    /// rendition. Reflow demotes a leader whose partner no longer fits beside it; a
    /// grapheme cluster growing from one column to two promotes one the other way.
    pub(super) fn with_width(self, width: CellWidth) -> Self {
        PackedCell::new(self.rune(), width, self.style, self.link)
    }

    pub(super) fn set_style_id(&mut self, id: StyleId) {
        self.style = id;
    }

    /// Whether this cell is the right-half placeholder of a wide character.
    pub(super) fn is_wide_spacer(self) -> bool {
        matches!(self.width(), CellWidth::Spacer)
    }

    /// Whether this cell is the left half (the rune) of a wide character.
    pub(super) fn is_wide_leader(self) -> bool {
        matches!(self.width(), CellWidth::Leader)
    }
}

/// The pen as a stored cell carries it: the rendition already interned, and the
/// hyperlink. What the write paths need and all they need.
///
/// It exists so interning happens **once per run** rather than once per cell. `Buffer`
/// holds no [`StyleTable`] (the table is the `Screen`'s, shared across both buffers),
/// so resolving inside the fill loop would not even be possible without handing the
/// table down into the storage layer, which is exactly the coupling the split avoids.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct PackedPen {
    pub(super) style: StyleId,
    pub(super) link: LinkId,
}

impl Default for PackedCell {
    fn default() -> Self {
        PackedCell::BLANK
    }
}

impl fmt::Debug for PackedCell {
    /// Unpacked, because the packed `u32` is unreadable in a failure message and the
    /// whole point of the type is that nobody has to think in it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PackedCell({:?}, style={}", self.rune(), self.style.0)?;
        if !matches!(self.width(), CellWidth::Narrow) {
            write!(f, ", {:?}", self.width())?;
        }
        if self.link.is_set() {
            write!(f, ", link={}", self.link.0)?;
        }
        f.write_str(")")
    }
}

/// One grid cell, **resolved**: the base rune of its grapheme cluster, its foreground
/// and background colours, its rendition attributes, its place in a wide pair, and the
/// hyperlink it belongs to.
///
/// This is the shape the grid answers questions in, not the shape it stores (see
/// [`PackedCell`], which is half the size and names its rendition by id). Resolving on
/// the way out keeps every reader — the painter, the selection, the tests — working in
/// the terms it actually cares about, and keeps the interning table a detail of the
/// grid rather than something every caller has to hold.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cell {
    /// The base rune. Combining marks, when present, live in the `Row`'s side
    /// list keyed by column (see the module header).
    pub rune: char,
    pub fg: Color,
    pub bg: Color,
    pub attrs: Attrs,
    /// Where this cell sits in a wide character, if it is in one at all.
    pub width: CellWidth,
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
        width: CellWidth::Narrow,
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

    /// The rendition half of this cell, which is what gets interned.
    pub fn style(self) -> Style {
        Style {
            fg: self.fg,
            bg: self.bg,
            attrs: self.attrs,
        }
    }

    /// Whether this cell is the right-half placeholder of a wide character.
    pub fn is_wide_spacer(self) -> bool {
        matches!(self.width, CellWidth::Spacer)
    }

    /// Whether this cell is the left half (the rune) of a wide character.
    pub fn is_wide_leader(self) -> bool {
        matches!(self.width, CellWidth::Leader)
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
        if !matches!(self.width, CellWidth::Narrow) {
            write!(f, ", {:?}", self.width)?;
        }
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
        for (k, &row_start) in (start..end).zip(self.new_start.get(start..end)?) {
            if row_start > off {
                break;
            }
            chosen = k;
        }
        Some((
            chosen,
            off.saturating_sub(*self.new_start.get(chosen)?)
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
    /// The renditions the grid's cells cite by [`StyleId`]. Shared by both buffers for
    /// exactly the reason [`Self::links`] is, and swept the same way.
    styles: StyleTable,
    /// The pen's rendition and its interned id, so printing a run does not hash a
    /// rendition per cell.
    ///
    /// Validated rather than invalidated: [`Screen::pen_style_id`] compares the pen's
    /// current rendition against the one cached here instead of relying on every one of
    /// the many places SGR writes the pen to remember to clear a flag. A stale-by-
    /// omission cache would paint text in the previous colour, which is precisely the
    /// class of bug that is invisible in tests and obvious on screen.
    pen_style: (Style, StyleId),
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
            styles: StyleTable::default(),
            pen_style: (Style::default(), StyleId::DEFAULT),
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
    fn blank_cell(&mut self) -> PackedCell {
        let style = Style {
            fg: Color::Default,
            bg: self.pen.bg,
            attrs: Attrs::empty(),
        };
        PackedCell::new(
            ' ',
            CellWidth::Narrow,
            self.intern_style(style),
            LinkId::NONE,
        )
    }

    /// The pen in the form the write paths store: rendition interned once, ready to be
    /// stamped into every cell of a run.
    pub(super) fn packed_pen(&mut self) -> PackedPen {
        PackedPen {
            style: self.pen_style_id(),
            link: self.pen.link,
        }
    }

    /// Resolve a stored cell into the shape every reader outside the grid works in.
    pub(super) fn resolve(&self, cell: PackedCell) -> Cell {
        let style = self.styles.get(cell.style_id());
        Cell {
            rune: cell.rune(),
            fg: style.fg,
            bg: style.bg,
            attrs: style.attrs,
            width: cell.width(),
            link: cell.link,
        }
    }

    /// The id for `style`, interning it and sweeping for room if it is new.
    ///
    /// Falls back to [`StyleId::DEFAULT`] when the table is full and a sweep cannot
    /// free anything, which degrades exactly as a spent [`LinkId`] space does: the text
    /// is still there and still correct, it just draws unstyled. Sixty-five thousand
    /// *simultaneously live* renditions is not a state any real program reaches.
    fn intern_style(&mut self, style: Style) -> StyleId {
        if let Some(id) = self.styles.intern(style) {
            return id;
        }
        // A sweep is worth doing once, and worth *not* repeating until something could
        // have died since. See [`StyleTable::swept_dry_at`].
        let grid_at = (self.epoch, self.primary.evicted);
        if self.styles.swept_dry_at == Some(grid_at) {
            return StyleId::DEFAULT;
        }
        self.collect_styles();
        match self.styles.intern(style) {
            Some(id) => id,
            None => {
                self.styles.swept_dry_at = Some(grid_at);
                StyleId::DEFAULT
            }
        }
    }

    /// Reclaim the ids of renditions no cell carries any more.
    ///
    /// A textbook mark-and-sweep, run only when [`StyleTable::intern`] reports the table
    /// full, so the walk over every cell is an *event* rather than a per-frame cost. The
    /// roots are every cell in both buffers plus the pen and both saved cursors, because
    /// a rendition can be set with nothing printed under it yet.
    ///
    /// Deliberately not reference counted: see [`StyleTable`].
    fn collect_styles(&mut self) {
        // `live[i]` speaks for `StyleId(i)`; slot 0 is the default and is never stored.
        let mut live = vec![false; self.styles.styles.len() + 1];
        let mark = |id: StyleId, live: &mut Vec<bool>| {
            if let Some(slot) = live.get_mut(usize::from(id.0)) {
                *slot = true;
            }
        };
        for buf in [&self.primary, &self.alt] {
            for row in buf.scrollback.iter().chain(buf.lines.iter()) {
                for cell in &row.cells {
                    mark(cell.style_id(), &mut live);
                }
            }
        }
        // The pen's own cached id. A saved cursor (DECSC) is *not* a root: it stores the
        // rendition unpacked, so restoring re-interns rather than citing a table entry.
        mark(self.pen_style.1, &mut live);

        let remap = self.styles.compact(&live);
        let renumber = |id: StyleId| {
            remap
                .get(usize::from(id.0))
                .copied()
                .unwrap_or(StyleId::DEFAULT)
        };
        for buf in [&mut self.primary, &mut self.alt] {
            for row in buf.scrollback.iter_mut().chain(buf.lines.iter_mut()) {
                for cell in &mut row.cells {
                    let id = cell.style_id();
                    cell.set_style_id(renumber(id));
                }
            }
        }
        self.pen_style.1 = renumber(self.pen_style.1);
    }

    /// The pen's rendition, interned, re-using the cached id while the pen has not
    /// changed. The check is a comparison of the rendition itself (ten bytes) rather
    /// than a flag someone has to remember to clear, so the cache cannot go stale.
    fn pen_style_id(&mut self) -> StyleId {
        let style = self.pen.style();
        if self.pen_style.0 != style {
            let id = self.intern_style(style);
            self.pen_style = (style, id);
        }
        self.pen_style.1
    }
}
