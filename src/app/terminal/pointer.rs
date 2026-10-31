//! The grid under the pointer: a local text selection, mouse reports to a program that
//! grabbed the mouse, and the hyperlink the pointer hovers. The copy side of the clipboard
//! is here too, since a selection is what it copies.
//!
//! A selection lives in absolute rows, so child output scrolling the grid does not move
//! it. [`Drag`] holds a drag's anchor until the button comes up.

use crate::app::message::{PointerEvent, Side, ToWindow};
use crate::app::terminal::TerminalCore;
use crate::error::Result;
use crate::grid::{AbsRow, ResizeEffect, RowEpoch};
use crate::input;
use crate::mouse::{self, MouseButton, MouseKind};
use crate::platform::verdict::{self, Refusal, Verdict};
use crate::pty::Target;
use crate::term_render::{self, CellSpan, LinkHover, Selection};

/// Lines the scrollback view moves per wheel notch, and arrows sent per notch
/// when the wheel falls back to arrow keys on the alt screen.
const WHEEL_LINES: usize = 3;

/// A vertical cell edge: edge `col` on `row` sits just before cell `col`, so `col`
/// runs `0..=cols` with the last edge the row's right margin. A character drag's
/// endpoints are edges, not cells: the pointer rounds to the nearer edge of the
/// cell it is over (see [`Edge::nearest`]), and the drag selects exactly the cells
/// with both edges inside it. Ordered row-major, like the cells between.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Edge {
    row: AbsRow,
    col: usize,
}

impl Edge {
    /// The edge of cell `(row, col)` nearer the pointer: the cell's own start from
    /// its left half, the next cell's start from its right. This rounding is what
    /// forgives a near-miss press — landing in the trailing half of the blank
    /// before a word anchors the selection at the word, not the blank — the same
    /// forgiveness alacritty (at the midline) and ghostty (at 60%) apply.
    fn nearest(row: AbsRow, col: usize, side: Side) -> Edge {
        Edge {
            row,
            col: col.saturating_add((side == Side::Right) as usize),
        }
    }
}

/// The granularity a selection drag extends by and the anchor it pivots around,
/// chosen from the press's click count: a single click drags by character around
/// the pressed cell's nearer [`Edge`]; a double-click drags by word and a
/// triple-click by whole (soft-wrapped) line, each around the clicked unit's
/// inclusive cell range so the anchored unit always stays selected.
#[derive(Clone, Copy)]
enum DragMode {
    Char(Edge),
    /// A word or line drag: the anchored unit's inclusive cell range, and which kind
    /// of unit to snap the far end to.
    Unit {
        range: ((AbsRow, usize), (AbsRow, usize)),
        by: Granularity,
    },
}

/// The unit a word or line drag snaps to under the pointer.
#[derive(Clone, Copy)]
enum Granularity {
    Word,
    Line,
}

impl DragMode {
    /// The last row the anchor names: once it has aged off the front of the ring,
    /// everything the anchor named is behind the oldest line still held, and the
    /// prune drops the drag (see [`TerminalCore::prune_selection`]).
    fn last_row(self) -> AbsRow {
        match self {
            DragMode::Char(edge) => edge.row,
            DragMode::Unit { range, .. } => range.1 .0,
        }
    }
}

/// A left-drag in progress: everything the pointer needs to keep extending it.
///
/// One value rather than two fields, because they are only meaningful together. In
/// particular the `epoch`: a character drag deliberately paints *no* selection until it
/// crosses a cell midline (so a plain click never flashes), yet it is already pinned
/// to an anchor. If the grid renumbers its rows in that window — a resize, an
/// alt-screen switch, a `clear`, a region scroll — the anchor names nothing, and the drag
/// has to be cancelled rather than resumed against whatever row inherited the id. Kept
/// beside the anchor it qualifies, there is no way to check one and forget the other.
#[derive(Clone, Copy)]
pub(super) struct Drag {
    /// The granularity the drag extends by and the anchor it pivots around.
    mode: DragMode,
    /// The identity regime the anchor's rows were minted in.
    epoch: RowEpoch,
}

/// The inclusive cell range a character drag between edges `a` and `b` covers (in
/// either order), or `None` when it covers no whole cell: a press-and-wiggle that
/// crossed no midline, or a drag from one cell's trailing half to the next cell's
/// leading half, which rounds both endpoints to the same edge. The end rounds
/// exactly like the start, so a cell the drag reaches less than halfway into
/// stays out at either end.
///
/// An edge at `cols` is the row's right margin, the same boundary as the next
/// row's edge 0; the arithmetic steps across that seam in both directions (a
/// start on the margin begins on the row below, an end on edge 0 stops at the
/// row above). At the ends of the id space the step has nowhere to go and the
/// range is empty.
fn char_range(a: Edge, b: Edge, cols: usize) -> Option<((AbsRow, usize), (AbsRow, usize))> {
    let (s, e) = if a <= b { (a, b) } else { (b, a) };
    let start = if s.col >= cols {
        (s.row.next()?, 0)
    } else {
        (s.row, s.col)
    };
    let end = match e.col.checked_sub(1) {
        Some(col) => (e.row, col),
        None => (e.row.prev()?, cols.checked_sub(1)?),
    };
    (start <= end).then_some((start, end))
}

impl TerminalCore {
    /// The terminal half of a pointer event: report it to a program grabbing the
    /// mouse, or drive local selection / scrollback scroll / hyperlinks. The window
    /// already mapped the event to a cell and supplied the modifier chord (Shift
    /// forces local use even while a program is reporting the mouse).
    ///
    /// Shift decides who owns a gesture when its button goes down, and the gesture
    /// stays with that owner until the button comes up: letting go of Shift halfway
    /// through a Shift+drag still ends the selection, and pressing it halfway through
    /// a reported drag still reports the release. A release whose press nobody took
    /// (the Ctrl+click that opened a link) does nothing.
    pub(in crate::app) fn pointer(&mut self, event: PointerEvent, mods: input::Mods) -> Result<()> {
        let reports = self.screen.mouse_mode().reports();
        let reporting = reports && !mods.contains(input::Mods::SHIFT);
        // A button the program heard go down keeps the pointer the program's until it
        // comes up, or until the program stops listening.
        let program_gesture = reports && self.mouse_held.is_some();
        match event {
            PointerEvent::Button {
                button,
                pressed,
                col,
                row,
                count,
                side,
            } => {
                if !pressed && button == MouseButton::Left && self.drag.is_some() {
                    // A local drag ends here and offers the text to the clipboard and
                    // primary selection, whatever Shift says now.
                    self.drag = None;
                    self.finish_selection();
                    self.dirty = true;
                } else if reporting || program_gesture {
                    self.mouse_held = pressed.then_some(button);
                    // A press or release starts a new gesture, so the next motion is
                    // news whatever cell it lands in.
                    self.reported_cell = None;
                    let kind = if pressed {
                        MouseKind::Press
                    } else {
                        MouseKind::Release
                    };
                    self.write_mouse(button, kind, col, row, mods, 1)?;
                } else if button == MouseButton::Left {
                    // Ctrl+click follows a hyperlink instead of starting a selection.
                    // It sits inside the local branch because a program grabbing the
                    // mouse owns its clicks, and Shift (which already means "this one is
                    // mine") is what frees a link there.
                    if pressed && mods.contains(input::Mods::CTRL) && self.open_link(row, col) {
                        return Ok(());
                    }
                    // A press begins a local selection at the click's granularity
                    // (character/word/line).
                    if pressed {
                        self.begin_selection(row, col, side, count);
                        self.dirty = true;
                    }
                } else if button == MouseButton::Middle && pressed {
                    // Middle-click pastes the primary selection (the Linux
                    // convention). The window owns the data device, so it does the
                    // receive; the core only asks.
                    self.outbox.push(ToWindow::PastePrimary);
                }
            }
            PointerEvent::Motion { col, row, side } => {
                if self.drag.is_some() {
                    self.extend_selection(row, col, side);
                } else if (reporting || program_gesture) && self.reported_cell != Some((row, col)) {
                    // Report motion to a program that asked for it (drag under ?1002,
                    // any move under ?1003) — once per *cell*, which is the only
                    // resolution the report has. See `reported_cell`.
                    self.reported_cell = Some((row, col));
                    let button = self.mouse_held.unwrap_or(MouseButton::None);
                    self.write_mouse(button, MouseKind::Motion, col, row, mods, 1)?;
                }
                // A hover is live only when the gesture is not already spoken for: a
                // drag is a selection, and a grabbed mouse belongs to the program.
                let live = !reporting && !program_gesture && self.drag.is_none();
                self.track_hover(row, col, live);
            }
            PointerEvent::Left => {
                // Off the grid entirely: the next motion back onto it is a fresh
                // position, even if it lands on the cell the pointer left from.
                self.reported_cell = None;
                self.track_hover(0, 0, false);
            }
            PointerEvent::Wheel {
                down,
                notches,
                col,
                row,
            } => {
                if reporting {
                    let button = if down {
                        MouseButton::WheelDown
                    } else {
                        MouseButton::WheelUp
                    };
                    self.write_mouse(button, MouseKind::Press, col, row, mods, notches)?;
                } else if !self.screen.is_alt() {
                    let lines = notches as usize * WHEEL_LINES;
                    if down {
                        self.screen.scroll_view_down(lines);
                    } else {
                        self.screen.scroll_view_up(lines);
                    }
                    self.flash_scrollbar();
                    // The text moved but the pointer did not: whatever it now rests on
                    // is a different link, or none.
                    self.refresh_hover();
                    self.dirty = true;
                } else {
                    // Alt screen without mouse reporting: wheel becomes arrow keys, the
                    // conventional fallback so wheel-scrolling `less`/`man` works.
                    let key = if down {
                        input::Key::Down
                    } else {
                        input::Key::Up
                    };
                    let modes = input::Modes::from_screen(&self.screen);
                    // Encoded into one buffer and sent once. A touchpad flick coalesces
                    // to several notches, and each notch is several lines, so the naive
                    // shape is a dozen three-byte writes where one does; over ssh that is
                    // a dozen round trips.
                    self.key_buf.clear();
                    for _ in 0..(notches as usize * WHEEL_LINES) {
                        input::encode(key, input::Mods::NONE, modes, &mut self.key_buf);
                    }
                    self.send_key_buf()?;
                }
            }
        }
        Ok(())
    }

    /// Track the hyperlink under the pointer as it moves to display `(row, col)`.
    /// `live` is false when the gesture belongs to something else (a drag, a program
    /// grabbing the mouse) or the pointer has left the grid, which drops the hover.
    ///
    /// The cell check is what keeps this cheap: a pointer crosses many pixels per
    /// cell, so all but the first motion event within a cell returns here.
    fn track_hover(&mut self, row: usize, col: usize, live: bool) {
        if !live {
            self.drop_hover();
            return;
        }
        if self.hover_cell == Some((row, col)) {
            return;
        }
        self.hover_cell = Some((row, col));
        self.refresh_hover();
    }

    /// Re-probe the link under the pointer where it currently rests. Called both when
    /// the pointer moves to a new cell and when the grid changes under a *parked*
    /// pointer: output scrolling the screen moves the text out from under it, so a
    /// hover left alone would underline whatever slid into its place.
    ///
    /// Cheap enough to run on any turn that changed the screen: one logical line
    /// scanned, and no allocation once [`LinkProbe`](crate::grid::LinkProbe)'s buffers
    /// have grown.
    pub(super) fn refresh_hover(&mut self) {
        let Some((row, col)) = self.hover_cell else {
            return;
        };
        let found = self.screen.link_at(row, col, &mut self.probe);
        let hover = found
            .zip(self.probe.verdict())
            .map(|((start, end), verdict)| LinkHover {
                span: CellSpan { start, end },
                verdict,
            });
        self.set_hover(hover);
    }

    /// Forget the hovered link and the cell it was probed from until the pointer next
    /// moves over the grid. For when this tab can no longer say what is under the
    /// pointer: the grid was resized beneath it, or the tab went to the background.
    pub(in crate::app) fn drop_hover(&mut self) {
        self.hover_cell = None;
        self.set_hover(None);
    }

    /// Adopt a new hovered link, just probed into [`Self::probe`], repainting only when it
    /// actually changed (a pointer crossing cells *within* one link must not redraw the
    /// frame). Output can rewrite a link in place, so the same cells with a different
    /// target are a change too: the preview must name what a click would open.
    fn set_hover(&mut self, hover: Option<LinkHover>) {
        let url = if hover.is_some() {
            self.probe.url()
        } else {
            ""
        };
        if self.hover == hover && self.hover_url == url {
            return;
        }
        self.hover = hover;
        self.hover_url.clear();
        self.hover_url.push_str(url);
        match hover {
            Some(hover) => {
                let room = term_render::preview_room(self.screen.dimensions().0, self.geom.metrics);
                crate::link_preview::write(&mut self.preview, url, hover.verdict, room);
            }
            None => self.preview.clear(),
        }
        self.dirty = true;
    }

    /// Follow a Ctrl+clicked hyperlink at display `(row, col)`, returning whether one
    /// was there (in which case the click is consumed and starts no selection).
    ///
    /// The URL is re-probed at the clicked cell rather than read off the hovered span,
    /// so what opens is what was clicked, with no chance of acting on a hover that
    /// output has since invalidated. Only a link whose verdict lets it open reaches the
    /// window: the text came from the child, which can print anything. A refused link
    /// still consumes the click, so Ctrl+click on one does nothing at all.
    ///
    /// A `file:` link is judged once more against the filesystem it names, and opens as
    /// the URL of the path that was checked. A local shell's is judged here
    /// ([`verdict::judge_file`]). A shell on the Flatpak host names host files the sandbox
    /// cannot see, so the window judges its links there, off the event loop
    /// ([`verdict::judge_host_file`]), and hands a refusal back to [`Self::refuse_link`].
    /// Either way a refusal raises the corner notice.
    fn open_link(&mut self, row: usize, col: usize) -> bool {
        if self.screen.link_at(row, col, &mut self.probe).is_none() {
            return false;
        }
        if self.probe.verdict() != Some(Verdict::Open) {
            return true;
        }
        let url = self.probe.url().to_string();
        if !url.starts_with("file:") {
            self.outbox.push(ToWindow::OpenUrl(url));
            return true;
        }
        match self.shell_target {
            Target::FlatpakHost => self.outbox.push(ToWindow::OpenHostFile(url)),
            Target::Local => match verdict::judge_file(&url) {
                Ok(checked) => self.outbox.push(ToWindow::OpenUrl(checked)),
                Err(refusal) => {
                    self.refuse_link(&url, refusal);
                    self.outbox.push(ToWindow::LinkRefused(refusal));
                }
            },
        }
        true
    }

    /// A click found the link to `url` refused by what its file is. When that is a fact
    /// about the file (an executable, a launcher), every anchor to it flips so the hover
    /// shows it from now on. A check that could not run (a missing file) leaves it, and
    /// the next click checks again.
    pub(in crate::app) fn refuse_link(&mut self, url: &str, refusal: Refusal) {
        if refusal != Refusal::Unreachable {
            self.screen.refuse_link(url, refusal);
            self.refresh_hover();
        }
    }

    /// Encode one mouse event under the current mouse mode and write it to the child
    /// (nothing when the mode produces no bytes for it).
    /// `times` is for the wheel, which reports one press per notch and can arrive
    /// several notches at a time from a touchpad flick: they encode into one buffer and
    /// leave as one hand-off rather than one write each.
    fn write_mouse(
        &mut self,
        button: MouseButton,
        kind: MouseKind,
        col: usize,
        row: usize,
        mods: input::Mods,
        times: u32,
    ) -> Result<()> {
        let mode = self.screen.mouse_mode();
        self.key_buf.clear();
        let mut encoded = false;
        for _ in 0..times {
            encoded |= mouse::encode(mode, button, kind, col, row, mods, &mut self.key_buf);
        }
        if encoded {
            self.send_key_buf()?;
        }
        Ok(())
    }

    /// Set the selected region and capture the text under it in the same breath.
    ///
    /// The only writer of either field, and that is the point: the highlight and the
    /// text a copy yields are one promise, and a call site that could move one without
    /// the other would be free to break it. Everything that ends a selection (a prune,
    /// a keystroke, a resize that cannot carry it) passes `None` through here.
    ///
    /// Re-captures on a carry too, which is not redundant: a reflow rewraps the rows, so
    /// the same picked text can join or split across a soft wrap and copy with a
    /// different set of newlines than it did before.
    pub(super) fn set_selection(&mut self, sel: Option<Selection>) {
        let sel = sel.map(|sel| self.whole_glyphs(sel));
        self.selection_text.clear();
        if let Some(sel) = sel {
            self.screen
                .selection_text_into(sel.anchor, sel.head, &mut self.selection_text);
        }
        self.selection = sel;
    }

    /// `sel` with each end widened to a whole wide character, the span both the highlight
    /// and the copy take.
    fn whole_glyphs(&self, sel: Selection) -> Selection {
        let (anchor, head) = self.screen.whole_glyph_span(sel.anchor, sel.head);
        Selection {
            anchor,
            head,
            ..sel
        }
    }

    /// Copy the selection's captured text into the outbox for the window to own on the
    /// clipboard (no-op without a selection or with empty text). The window makes the
    /// data-device request.
    ///
    /// Hands over what was under the highlight when the user drew it, not what is under
    /// it now — see [`Self::selection_text`].
    pub(in crate::app) fn copy_selection(&mut self) {
        self.offer_selection(false);
    }

    /// Queue the captured selection text for the window to own on the clipboard, and on
    /// the primary selection too when `primary` (copy-on-select offers both). A no-op
    /// with no selection or empty text.
    pub(super) fn offer_selection(&mut self, primary: bool) {
        if self.selection_text.is_empty() {
            return;
        }
        let bytes = self.selection_text.as_bytes().to_vec();
        if primary {
            self.outbox.push(ToWindow::OfferPrimary(bytes.clone()));
        }
        self.outbox.push(ToWindow::OfferSelection(bytes));
    }

    /// Drop the selection because the user just sent the child input, matching xterm,
    /// ghostty and alacritty: a keystroke means they have moved on, and leaving a
    /// highlight over cells the child is about to rewrite promises a copy the terminal
    /// would then have to keep.
    ///
    /// The drag is deliberately left alone. If a button is still physically down the
    /// user is still selecting, and cancelling a gesture mid-motion is the more
    /// surprising of the two; the next motion re-draws the highlight from the anchor.
    pub(super) fn end_selection_on_input(&mut self) {
        if self.selection.is_some() {
            self.set_selection(None);
            self.dirty = true;
        }
    }

    /// Begin a selection at display `(row, col)` with the granularity the click
    /// `count` picks: one click a character, two the word, three the whole
    /// (soft-wrapped) line. The clicked unit is the anchor a drag pivots around. A
    /// plain click paints nothing yet (a character drag's selection appears only
    /// once it crosses a cell midline, see [`Self::extend_selection`]), so a single
    /// click never flashes the cell under it; a word/line click is a real selection
    /// at once.
    fn begin_selection(&mut self, row: usize, col: usize, side: Side, count: usize) {
        let abs = self.screen.abs_row(row);
        let (mode, unit) = match count {
            2 => {
                let range = self.screen.word_at(abs, col);
                (
                    DragMode::Unit {
                        range,
                        by: Granularity::Word,
                    },
                    Some(range),
                )
            }
            n if n >= 3 => {
                let range = self.screen.line_at(abs);
                (
                    DragMode::Unit {
                        range,
                        by: Granularity::Line,
                    },
                    Some(range),
                )
            }
            _ => (DragMode::Char(Edge::nearest(abs, col, side)), None),
        };
        let epoch = self.screen.row_epoch();
        self.drag = Some(Drag { mode, epoch });
        self.set_selection(unit.map(|(anchor, head)| Selection {
            anchor,
            head,
            epoch,
        }));
    }

    /// Extend the in-progress drag to display `(row, col)`, snapped to its
    /// granularity: the span runs from the anchored unit to the unit under the
    /// pointer, so a word/line drag never splits a word or line. A character drag
    /// selects the cells between its two rounded edges (see [`char_range`]), so it
    /// stays empty — and reads as a click — until it first crosses a midline, and
    /// a cell at either end joins only once the pointer is over half-way into it.
    fn extend_selection(&mut self, row: usize, col: usize, side: Side) {
        let Some(drag) = self.drag else {
            return;
        };
        let abs = self.screen.abs_row(row);
        let range = match drag.mode {
            DragMode::Char(anchor) => {
                let cols = self.screen.dimensions().0;
                char_range(anchor, Edge::nearest(abs, col, side), cols)
            }
            DragMode::Unit { range: anchor, by } => {
                let unit = match by {
                    Granularity::Word => self.screen.word_at(abs, col),
                    Granularity::Line => self.screen.line_at(abs),
                };
                Some((anchor.0.min(unit.0), anchor.1.max(unit.1)))
            }
        };
        let sel = range.map(|(start, end)| {
            self.whole_glyphs(Selection {
                anchor: start,
                head: end,
                epoch: drag.epoch,
            })
        });
        if self.selection != sel {
            self.set_selection(sel);
            self.dirty = true;
        }
    }

    /// End a left-drag: a selection with text is offered to both the clipboard and
    /// the primary selection (copy-on-select, matching the `copy-on-select =
    /// clipboard` convention), so Ctrl+V and middle-click both paste it. A plain
    /// click leaves no selection at all. A drag over only blank cells keeps its
    /// highlight (so selecting the empty space below the prompt sticks, the way
    /// wezterm/ghostty leave it until the next click or keystroke) but offers nothing,
    /// since there is no text to copy.
    fn finish_selection(&mut self) {
        // Re-capture as the button comes up. The release is the last instant the gesture
        // is the user's, and it is what they were looking at when they let go: a drag
        // left paused over a line the child rewrites in place should copy what was under
        // it then, not what was under it when the pointer last moved.
        let sel = self.selection;
        self.set_selection(sel);
        self.offer_selection(true);
    }

    /// Carry the text selection across a resize per the grid's [`ResizeEffect`]. The ids
    /// stood still (a height change), so keep it; or the reflow renumbered but can translate,
    /// so remap each endpoint to the cell it now names (dropping the whole selection if either
    /// endpoint's cell was absorbed or has aged off the front); or the alt screen renumbered
    /// under it with nothing to translate against, so drop it. A `None` selection stays `None`.
    pub(super) fn carry_selection(&self, effect: ResizeEffect) -> Option<Selection> {
        let sel = self.selection?;
        match effect {
            ResizeEffect::Stable => Some(sel),
            ResizeEffect::Reset => None,
            ResizeEffect::Reflowed(remap) => {
                let anchor = remap.point(sel.anchor.0, sel.anchor.1)?;
                let head = remap.point(sel.head.0, sel.head.1)?;
                Some(Selection {
                    anchor,
                    head,
                    epoch: self.screen.row_epoch(),
                })
            }
        }
    }

    /// Drop the selection, and any drag pinned to it, when the rows they name have
    /// stopped meaning what they meant.
    ///
    /// Two ways that happens, and only two. The grid can end the identity regime those
    /// rows were minted in — a reset, a resize, an alt-screen switch, a full-display
    /// erase, or a scroll that renumbered the stream (see
    /// [`RowEpoch`]). Or the ring can simply outrun them: once the
    /// *last* row of a selection has aged off the front of history, the whole thing is
    /// behind the oldest line the terminal still holds, and there is nothing left to copy.
    /// A selection that has merely scrolled out of *sight* is not pruned — it is still
    /// there, and scrolling back reveals it, as in xterm.
    ///
    /// The drag is checked separately from the selection, and must be: a character drag
    /// that has not yet crossed a cell midline has an anchor but no selection to
    /// speak for it, and resuming it against a renumbered grid would drag out a span from
    /// a row the user never touched.
    ///
    /// Everything else — a printing child, a scroll, a program overwriting the cells
    /// underneath — leaves both alone. What a copy yields is unaffected either way: the
    /// text was captured when the region was drawn ([`Self::selection_text`]), so a
    /// child rewriting those cells changes what is painted under the highlight but never
    /// what the clipboard gets.
    pub(super) fn prune_selection(&mut self) {
        let epoch = self.screen.row_epoch();

        if self
            .drag
            .is_some_and(|d| d.epoch != epoch || !self.screen.row_exists(d.mode.last_row()))
        {
            self.drag = None;
        }
        let stale = self
            .selection
            .is_some_and(|sel| sel.epoch != epoch || !self.screen.row_exists(sel.ordered().1 .0));
        if stale {
            self.set_selection(None);
            self.drag = None;
            self.dirty = true;
        }
    }

    /// The verdict on the hyperlink the pointer rests on, or `None` off a link. The
    /// window reads it to offer the hand over a link that opens and `not_allowed` over
    /// a refused one, so the pointer only promises a click will do something when
    /// Ctrl+clicking actually would.
    pub(in crate::app) fn hovered_link(&self) -> Option<Verdict> {
        self.hover.map(|hover| hover.verdict)
    }

    /// Whether a program has grabbed the mouse with `?1000`/`?1002`/`?1003`. The window
    /// reads it to drop the I-beam: while a program is reporting, a drag is that
    /// program's to interpret, not a text selection, so the pointer must not promise
    /// one. Mirrors the `reporting` test in [`Self::pointer`], minus the Shift
    /// override the window applies itself.
    pub(in crate::app) fn mouse_reporting(&self) -> bool {
        self.screen.mouse_mode().reports()
    }
}

#[cfg(test)]
mod tests {
    use crate::app::terminal::pointer::*;
    use crate::app::terminal::tests::{core_80x24, geom};
    use crate::app::terminal::*;
    use crate::platform::verdict::Refusal;
    use crate::pty::Launch;
    use crate::render::display::DisplayList;
    use crate::testutil::feed;

    /// The plain helpers stay in the cell's left half, whose nearer edge is the
    /// cell's own start — the classic floor mapping; `press_in`/`drag_in` pick a
    /// half explicitly for the edge-rounding cases.
    fn press(core: &mut TerminalCore, button: MouseButton, pressed: bool, col: usize, row: usize) {
        click(core, button, pressed, col, row, 1, Side::Left);
    }

    fn press_in(core: &mut TerminalCore, col: usize, row: usize, side: Side) {
        click(core, MouseButton::Left, true, col, row, 1, side);
    }

    fn click(
        core: &mut TerminalCore,
        button: MouseButton,
        pressed: bool,
        col: usize,
        row: usize,
        count: usize,
        side: Side,
    ) {
        core.pointer(
            PointerEvent::Button {
                button,
                pressed,
                col,
                row,
                count,
                side,
            },
            input::Mods::NONE,
        )
        .unwrap();
    }

    fn drag_to(core: &mut TerminalCore, col: usize, row: usize) {
        drag_in(core, col, row, Side::Left);
    }

    fn drag_in(core: &mut TerminalCore, col: usize, row: usize, side: Side) {
        core.pointer(PointerEvent::Motion { col, row, side }, input::Mods::NONE)
            .unwrap();
    }

    /// What the cells between `a` and `b` hold *now*, which is the thing a capture is
    /// deliberately not: the tests that assert the two have diverged need both.
    fn grid_text(screen: &Screen, a: (AbsRow, usize), b: (AbsRow, usize)) -> String {
        let mut out = String::new();
        screen.selection_text_into(a, b, &mut out);
        out
    }

    /// The text of the sole primary offer in `core`'s outbox, if any.
    fn primary_offer(core: &mut TerminalCore) -> Option<String> {
        core.take_outbox().into_iter().find_map(|m| match m {
            ToWindow::OfferPrimary(bytes) => String::from_utf8(bytes).ok(),
            _ => None,
        })
    }

    /// A blank (non-demo) core with `text` printed at the top-left, for driving the
    /// hyperlink gestures over content a case chooses.
    fn core_showing(text: &str) -> TerminalCore {
        let mut core = core_80x24(false);
        feed(&mut core.screen, text.as_bytes());
        core
    }

    /// Move the pointer to a cell with a modifier chord held.
    fn hover_at(core: &mut TerminalCore, col: usize, row: usize, mods: input::Mods) {
        core.pointer(
            PointerEvent::Motion {
                col,
                row,
                side: Side::Left,
            },
            mods,
        )
        .unwrap();
    }

    /// Press or release a button at a cell with a modifier chord held.
    fn click_mods(
        core: &mut TerminalCore,
        pressed: bool,
        col: usize,
        row: usize,
        mods: input::Mods,
    ) {
        core.pointer(
            PointerEvent::Button {
                button: MouseButton::Left,
                pressed,
                col,
                row,
                count: 1,
                side: Side::Left,
            },
            mods,
        )
        .unwrap();
    }

    /// The URL of the sole open request in `core`'s outbox, if any.
    fn opened_url(core: &mut TerminalCore) -> Option<String> {
        core.take_outbox().into_iter().find_map(|m| match m {
            ToWindow::OpenUrl(url) => Some(url),
            _ => None,
        })
    }

    #[test]
    fn hovering_a_url_marks_it_and_hovering_off_clears_it() {
        // "see https://example.com/a here": the URL sits at cols 4..=24.
        let mut core = core_showing("see https://example.com/a here");
        hover_at(&mut core, 10, 0, input::Mods::NONE);
        assert_eq!(
            core.hovered_link(),
            Some(Verdict::Open),
            "the pointer rests on the link"
        );
        assert_eq!(
            core.hover.map(|h| h.span),
            Some(CellSpan {
                start: (0, 4),
                end: (0, 24)
            })
        );
        // Off the link, onto the prose after it.
        hover_at(&mut core, 27, 0, input::Mods::NONE);
        assert!(core.hovered_link().is_none(), "the prose is not a link");
        // And off the grid entirely: an underline must not outlive the pointer.
        hover_at(&mut core, 10, 0, input::Mods::NONE);
        assert_eq!(core.hovered_link(), Some(Verdict::Open));
        core.pointer(PointerEvent::Left, input::Mods::NONE).unwrap();
        assert!(core.hovered_link().is_none(), "the pointer left the grid");
    }

    #[test]
    fn moving_within_one_link_does_not_redraw() {
        // The hover is repainted only when it *changes*: crossing cells inside one
        // link must not dirty a frame, or a moving pointer would repaint constantly.
        let mut core = core_showing("see https://example.com/a here");
        hover_at(&mut core, 5, 0, input::Mods::NONE);
        assert!(core.dirty, "the link appearing under the pointer redraws");
        core.dirty = false;
        for col in 6..=24 {
            hover_at(&mut core, col, 0, input::Mods::NONE);
        }
        assert!(!core.dirty, "moving along the same link redraws nothing");
        hover_at(&mut core, 25, 0, input::Mods::NONE);
        assert!(core.dirty, "leaving it does");
    }

    #[test]
    fn ctrl_click_opens_a_link_and_starts_no_selection() {
        let mut core = core_showing("see https://example.com/a here");
        click_mods(&mut core, true, 10, 0, input::Mods::CTRL);
        assert_eq!(
            opened_url(&mut core).as_deref(),
            Some("https://example.com/a")
        );
        assert!(core.selection.is_none(), "the click began no selection");
        assert!(core.drag.is_none(), "and no drag is in progress");
    }

    #[test]
    fn a_plain_click_on_a_link_selects_instead_of_opening() {
        // The link only follows on Ctrl+click, so an ordinary click still just places
        // a selection: a stray click can never launch a browser.
        let mut core = core_showing("see https://example.com/a here");
        press(&mut core, MouseButton::Left, true, 10, 0);
        drag_to(&mut core, 12, 0);
        press(&mut core, MouseButton::Left, false, 12, 0);
        let out = core.take_outbox();
        assert!(
            !out.iter().any(|m| matches!(m, ToWindow::OpenUrl(_))),
            "a plain click opens nothing"
        );
        assert!(
            out.iter().any(|m| matches!(m, ToWindow::OfferPrimary(_))),
            "it selects, as it always did"
        );
    }

    #[test]
    fn ctrl_click_off_a_link_still_selects() {
        // Ctrl only means "follow" when there is something to follow; elsewhere the
        // click keeps its ordinary meaning rather than being swallowed.
        let mut core = core_showing("see https://example.com/a here");
        click_mods(&mut core, true, 1, 0, input::Mods::CTRL);
        assert!(core.drag.is_some(), "a selection drag began");
        assert!(opened_url(&mut core).is_none());
    }

    #[test]
    fn a_hostile_scheme_is_never_opened() {
        // The grid's text comes from the child, which can print anything. Neither the
        // hover nor the click may reach a scheme the opener refuses.
        let mut core = core_showing("javascript:alert(1) data:text/html,x");
        hover_at(&mut core, 4, 0, input::Mods::NONE);
        assert!(core.hovered_link().is_none(), "not even underlined");
        click_mods(&mut core, true, 4, 0, input::Mods::CTRL);
        assert!(opened_url(&mut core).is_none(), "and never opened");
    }

    #[test]
    fn a_refused_anchor_hovers_as_refused_and_its_ctrl_click_does_nothing() {
        let mut core =
            core_showing("\x1b]8;;javascript:alert(1)\x1b\\click me\x1b]8;;\x1b\\ after");
        hover_at(&mut core, 3, 0, input::Mods::NONE);
        assert_eq!(core.hovered_link(), Some(Verdict::Refused(Refusal::Scheme)));
        assert_eq!(
            core.hover.map(|h| h.span),
            Some(CellSpan {
                start: (0, 0),
                end: (0, 7)
            })
        );

        click_mods(&mut core, true, 3, 0, input::Mods::CTRL);
        assert!(core.drag.is_none(), "the press began no selection");
        click_mods(&mut core, false, 3, 0, input::Mods::CTRL);
        let out = core.take_outbox();
        assert!(
            !out.iter().any(|m| matches!(m, ToWindow::OpenUrl(_))),
            "nothing opens"
        );
        assert!(core.selection.is_none(), "the click selected nothing");
        assert!(
            !out.iter()
                .any(|m| matches!(m, ToWindow::OfferPrimary(_) | ToWindow::OfferSelection(_))),
            "and offered nothing to the clipboard"
        );
    }

    #[test]
    fn a_program_grabbing_the_mouse_keeps_its_clicks() {
        // Under mouse reporting the pointer belongs to the program (vim, tmux): links
        // neither underline nor open, so its own Ctrl+click still reaches it. Shift is
        // the existing escape hatch that hands the gesture back to the terminal.
        let mut core = core_showing("\x1b[?1000hsee https://example.com/a here");
        assert!(core.screen.mouse_mode().reports());

        hover_at(&mut core, 10, 0, input::Mods::NONE);
        assert!(
            core.hovered_link().is_none(),
            "the program owns the pointer"
        );
        click_mods(&mut core, true, 10, 0, input::Mods::CTRL);
        click_mods(&mut core, false, 10, 0, input::Mods::CTRL);
        assert!(
            opened_url(&mut core).is_none(),
            "the click went to the child"
        );

        // Shift frees the gesture: the link underlines, and Ctrl+Shift+click opens it.
        hover_at(&mut core, 10, 0, input::Mods::SHIFT);
        assert_eq!(
            core.hovered_link(),
            Some(Verdict::Open),
            "Shift takes the pointer back"
        );
        click_mods(
            &mut core,
            true,
            10,
            0,
            input::Mods::CTRL | input::Mods::SHIFT,
        );
        assert_eq!(
            opened_url(&mut core).as_deref(),
            Some("https://example.com/a")
        );
    }

    #[test]
    fn letting_go_of_shift_before_the_button_still_ends_the_selection() {
        // Shift+drag selects text under a program grabbing the mouse, and people let go
        // of Shift a moment before the button. The release still belongs to the
        // selection: it copies, and the program never hears a release it saw no press
        // for.
        let mut core = core_showing("\x1b[?1000h\x1b[?1006hhello world");
        core.key_buf.clear();
        click_mods(&mut core, true, 0, 0, input::Mods::SHIFT);
        hover_at(&mut core, 5, 0, input::Mods::SHIFT);
        hover_at(&mut core, 5, 0, input::Mods::NONE);
        click_mods(&mut core, false, 5, 0, input::Mods::NONE);

        assert!(core.key_buf.is_empty(), "nothing reached the program");
        assert!(core.drag.is_none(), "the drag ended with the button");
        assert_eq!(primary_offer(&mut core).as_deref(), Some("hello"));
        hover_at(&mut core, 8, 0, input::Mods::NONE);
        assert_eq!(
            core.selection_text, "hello",
            "the highlight no longer follows a pointer with no button held"
        );
    }

    #[test]
    fn pressing_shift_mid_drag_leaves_the_gesture_with_the_program() {
        // The reverse: a drag the program heard begin is the program's to the end, so
        // Shift arriving halfway neither hands the pointer to the terminal (no link
        // hover, no selection) nor swallows the release, which would leave the program
        // thinking the button is still down.
        let mut core = core_showing("\x1b[?1002h\x1b[?1006hsee https://example.com/a here");
        core.key_buf.clear();
        click_mods(&mut core, true, 1, 0, input::Mods::NONE);
        assert_eq!(core.key_buf, b"\x1b[<0;2;1M");

        core.key_buf.clear();
        hover_at(&mut core, 10, 0, input::Mods::SHIFT);
        assert_eq!(
            core.key_buf, b"\x1b[<36;11;1M",
            "the drag is still reported"
        );
        assert!(
            core.hovered_link().is_none(),
            "the link under it is not the terminal's"
        );
        assert!(core.drag.is_none());

        core.key_buf.clear();
        click_mods(&mut core, false, 10, 0, input::Mods::SHIFT);
        assert_eq!(core.key_buf, b"\x1b[<4;11;1m", "and so is its release");
        assert!(
            primary_offer(&mut core).is_none(),
            "which offers no selection"
        );
    }

    #[test]
    fn a_url_hiding_a_character_hovers_as_refused_and_never_opens() {
        let mut core = core_showing("see https://example.com/a\u{200B}b here");
        hover_at(&mut core, 10, 0, input::Mods::NONE);
        assert_eq!(core.hovered_link(), Some(Verdict::Refused(Refusal::Hidden)));
        click_mods(&mut core, true, 10, 0, input::Mods::CTRL);
        assert!(opened_url(&mut core).is_none());
        assert!(core.drag.is_none(), "the click was consumed");
    }

    #[test]
    fn an_anchor_disguised_as_another_host_hovers_as_refused_and_never_opens() {
        let mut core =
            core_showing("\x1b]8;;https://evil.example/x\x1b\\https://github.com/x\x1b]8;;\x1b\\");
        hover_at(&mut core, 3, 0, input::Mods::NONE);
        assert_eq!(
            core.hovered_link(),
            Some(Verdict::Refused(Refusal::LabelHost))
        );
        click_mods(&mut core, true, 3, 0, input::Mods::CTRL);
        assert!(opened_url(&mut core).is_none());
    }

    /// A fresh directory for one test's fixtures, removed when it drops, pass or fail.
    struct FixtureDir(std::path::PathBuf);

    impl FixtureDir {
        fn new(name: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |since| since.as_nanos());
            let pid = std::process::id();
            let dir = std::env::temp_dir().join(format!("bnkterm-{name}-{pid}-{nanos}"));
            std::fs::create_dir_all(&dir).unwrap();
            Self(std::fs::canonicalize(dir).unwrap())
        }
    }

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// An OSC 8 anchor to the file at `path`, reading `text`.
    fn file_anchor(path: &std::path::Path, text: &str) -> String {
        format!(
            "\x1b]8;;file://{}\x1b\\{text}\x1b]8;;\x1b\\",
            path.display()
        )
    }

    #[test]
    fn a_missing_file_is_checked_again_on_the_next_click() {
        let fixture = FixtureDir::new("missing");
        let report = fixture.0.join("report.txt");
        let mut core = core_showing(&file_anchor(&report, "report"));

        click_mods(&mut core, true, 2, 0, input::Mods::CTRL);
        let out = core.take_outbox();
        assert!(out
            .iter()
            .any(|m| matches!(m, ToWindow::LinkRefused(Refusal::Unreachable))));
        hover_at(&mut core, 2, 0, input::Mods::NONE);
        assert_eq!(
            core.hovered_link(),
            Some(Verdict::Open),
            "not held against it"
        );

        std::fs::write(&report, b"x").unwrap();
        click_mods(&mut core, true, 2, 0, input::Mods::CTRL);
        assert!(opened_url(&mut core).is_some(), "it opens once it exists");
    }

    #[test]
    fn a_file_link_from_a_shell_on_the_flatpak_host_is_handed_over_to_be_judged_there() {
        // The sandbox cannot see the host's files, so a check here would judge whatever
        // sits at the same path in the sandbox, or nothing.
        use std::os::unix::fs::PermissionsExt;
        let fixture = FixtureDir::new("host");
        let script = fixture.0.join("build.sh");
        std::fs::write(&script, b"x").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut core = core_showing("");
        let launch = Launch::new(vec!["/bin/cat".to_string()], Target::FlatpakHost);
        if core.spawn_shell(&launch).is_err() {
            eprintln!("pty spawn unavailable in this environment; skipping");
            return;
        }
        let shown = format!(
            "{} \x1b]8;;https://example.com/\x1b\\web\x1b]8;;\x1b\\",
            file_anchor(&script, "build")
        );
        feed(&mut core.screen, shown.as_bytes());

        click_mods(&mut core, true, 2, 0, input::Mods::CTRL);
        let out = core.take_outbox();
        assert!(
            !out.iter()
                .any(|m| matches!(m, ToWindow::OpenUrl(_) | ToWindow::LinkRefused(_))),
            "neither opened unchecked nor judged against the sandbox"
        );
        let url = format!("file://{}", script.display());
        assert!(out
            .iter()
            .any(|m| matches!(m, ToWindow::OpenHostFile(handed) if *handed == url)));

        click_mods(&mut core, true, 7, 0, input::Mods::CTRL);
        assert_eq!(
            opened_url(&mut core).as_deref(),
            Some("https://example.com/"),
            "a web link is no host file"
        );
    }

    #[test]
    fn a_file_link_to_a_program_is_refused_at_the_click_and_shows_refused_after() {
        use std::os::unix::fs::PermissionsExt;
        let fixture = FixtureDir::new("click");
        let dir = fixture.0.as_path();
        let script = dir.join("build.sh");
        let notes = dir.join("notes.txt");
        std::fs::write(&script, b"x").unwrap();
        std::fs::write(&notes, b"x").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut core = core_showing(&format!(
            "{} {}",
            file_anchor(&script, "build"),
            file_anchor(&notes, "notes")
        ));

        hover_at(&mut core, 2, 0, input::Mods::NONE);
        assert_eq!(
            core.hovered_link(),
            Some(Verdict::Open),
            "openable until clicked"
        );
        click_mods(&mut core, true, 2, 0, input::Mods::CTRL);
        assert!(core.drag.is_none(), "the click was consumed");
        let out = core.take_outbox();
        assert!(!out.iter().any(|m| matches!(m, ToWindow::OpenUrl(_))));
        assert!(out
            .iter()
            .any(|m| matches!(m, ToWindow::LinkRefused(Refusal::Executable))));
        assert_eq!(
            core.hovered_link(),
            Some(Verdict::Refused(Refusal::Executable)),
            "and refused from now on"
        );

        click_mods(&mut core, true, 7, 0, input::Mods::CTRL);
        assert_eq!(
            opened_url(&mut core).as_deref(),
            Some(crate::platform::file_url::encode(notes.to_str().unwrap()).as_str())
        );
    }

    #[test]
    fn a_resize_drops_the_hover_until_the_pointer_moves() {
        // The grid reflows under a parked pointer, and the cell the pointer is over may
        // move too, so the old hover would underline cells that now hold other text.
        let mut core = core_showing("see https://example.com/a here");
        hover_at(&mut core, 10, 0, input::Mods::NONE);
        assert_eq!(core.hovered_link(), Some(Verdict::Open));
        core.resize(geom(10, 24, 10 * 8, 24 * 16));
        assert_eq!(core.hovered_link(), None);
        hover_at(&mut core, 5, 0, input::Mods::NONE);
        assert_eq!(
            core.hovered_link(),
            Some(Verdict::Open),
            "the next move finds it"
        );
    }

    #[test]
    fn snapping_to_the_live_bottom_re_probes_the_hover() {
        let mut text = String::from("https://example.com/old\r\n");
        for _ in 0..30 {
            text.push_str("plain\r\n");
        }
        let mut core = core_showing(&text);
        core.screen.scroll_view_to_top();
        hover_at(&mut core, 5, 0, input::Mods::NONE);
        assert_eq!(
            core.hovered_link(),
            Some(Verdict::Open),
            "the old URL in history"
        );
        core.paste(b"x".to_vec()).unwrap();
        assert_eq!(
            core.hovered_link(),
            None,
            "the live screen has plain text there"
        );
    }

    #[test]
    fn a_ctrl_click_on_a_link_leaves_the_clipboard_alone() {
        // The press opens the link and takes nothing else. Its release must not finish a
        // selection either: that re-offered whatever was last selected, overwriting
        // whatever the user had copied since.
        let mut core = core_showing("see https://example.com/a here");
        press(&mut core, MouseButton::Left, true, 0, 0);
        drag_to(&mut core, 3, 0);
        press(&mut core, MouseButton::Left, false, 3, 0);
        assert_eq!(primary_offer(&mut core).as_deref(), Some("see"));

        click_mods(&mut core, true, 10, 0, input::Mods::CTRL);
        click_mods(&mut core, false, 10, 0, input::Mods::CTRL);
        let out = core.take_outbox();
        assert!(out.iter().any(|m| matches!(m, ToWindow::OpenUrl(_))));
        assert!(
            !out.iter()
                .any(|m| matches!(m, ToWindow::OfferPrimary(_) | ToWindow::OfferSelection(_))),
            "the old selection was not offered again"
        );
    }

    #[test]
    fn a_resize_that_ends_a_drag_still_copies_its_selection() {
        // A resize ends a drag while the button is still down, so its release finds no
        // drag to finish. What the drag selected is copied all the same.
        let mut core = core_showing("hello world");
        press(&mut core, MouseButton::Left, true, 0, 0);
        drag_to(&mut core, 5, 0);
        core.resize(geom(40, 12, 40 * 8, 12 * 16));
        press(&mut core, MouseButton::Left, false, 5, 0);
        assert_eq!(primary_offer(&mut core).as_deref(), Some("hello"));
    }

    #[test]
    fn mouse_reporting_tracks_the_program_that_grabbed_the_mouse() {
        // What the window reads to drop the I-beam. It must follow the child's toggles
        // exactly, since a stale answer leaves the pointer promising the wrong gesture.
        let mut core = core_showing("");
        assert!(!core.mouse_reporting(), "a bare shell selects text");

        feed(&mut core.screen, b"\x1b[?1000h");
        assert!(core.mouse_reporting(), "the program took the mouse");
        feed(&mut core.screen, b"\x1b[?1000l");
        assert!(!core.mouse_reporting(), "and gave it back on exit");
    }

    #[test]
    fn output_under_a_parked_pointer_re_probes_the_hover() {
        // The pointer sits on a link and the child prints a line, scrolling it up. The
        // underline must follow the text, not stay where the pointer happens to be.
        let mut core = core_showing("https://example.com/a\r\n");
        hover_at(&mut core, 5, 1, input::Mods::NONE);
        assert!(core.hovered_link().is_none(), "row 1 is blank");
        // Print a link onto row 1, where the pointer already rests.
        feed(&mut core.screen, b"https://example.com/b");
        core.after_output();
        assert_eq!(
            core.hovered_link(),
            Some(Verdict::Open),
            "the freshly printed link is under the pointer"
        );
        assert_eq!(
            core.hover.map(|h| h.span),
            Some(CellSpan {
                start: (1, 0),
                end: (1, 20)
            })
        );
    }

    #[test]
    fn the_preview_names_what_the_hovered_link_would_open() {
        use crate::render::display::DrawCmd;
        let mut core = core_showing("https://github.com/a https://evil.example/b");
        let plates = |core: &TerminalCore, ctrl| {
            let (mut list, mut strings) = (DisplayList::new(), Vec::new());
            core.fill_frame_list(&mut list, &mut strings, ctrl);
            list.iter()
                .filter(|c| matches!(c, DrawCmd::RoundRect { .. }))
                .count()
        };

        hover_at(&mut core, 5, 0, input::Mods::NONE);
        assert_eq!(core.preview, "github.com  https://github.com/a");
        assert_eq!(plates(&core, false), 0, "drawn only while Ctrl is held");
        assert_eq!(plates(&core, true), 1);

        hover_at(&mut core, 25, 0, input::Mods::NONE);
        assert_eq!(core.preview, "evil.example  https://evil.example/b");

        // Output rewrites the link under the parked pointer: the same cells, judged the
        // same, but a click would now open another target.
        feed(&mut core.screen, b"\x1b[1;22Hhttps://evil.example/c");
        core.after_output();
        assert_eq!(core.preview, "evil.example  https://evil.example/c");

        hover_at(&mut core, 20, 0, input::Mods::NONE);
        assert!(core.preview.is_empty(), "off a link");
        assert_eq!(plates(&core, true), 0);
    }

    #[test]
    fn a_drag_offers_the_selection_to_clipboard_and_primary() {
        // Dragging across the demo title (row 0 begins "bnkterm") offers that text on
        // release to *both* the clipboard and the primary selection, so Ctrl+V and
        // middle-click both paste it.
        let mut core = core_80x24(true);
        press(&mut core, MouseButton::Left, true, 0, 0);
        drag_in(&mut core, 3, 0, Side::Right); // past col 3's midline, so its cell joins
        press(&mut core, MouseButton::Left, false, 3, 0);
        let out = core.take_outbox();
        let primary = out.iter().find_map(|m| match m {
            ToWindow::OfferPrimary(b) => Some(b.as_slice()),
            _ => None,
        });
        let clipboard = out.iter().find_map(|m| match m {
            ToWindow::OfferSelection(b) => Some(b.as_slice()),
            _ => None,
        });
        assert_eq!(primary, Some(b"bnkt".as_slice()), "primary gets the text");
        assert_eq!(clipboard, Some(b"bnkt".as_slice()), "clipboard too");
    }

    /// Whether one motion event produced a mouse report, and the bytes if it did.
    ///
    /// `key_buf` is where every encoder writes before the bytes are handed off, and it is
    /// where they are observable in a core with no child attached: `pump_writes` discards
    /// the outbound queue when there is no pty to take it, so `out_buf` cannot be read
    /// after the fact.
    fn motion_report(core: &mut TerminalCore, col: usize, row: usize, side: Side) -> Vec<u8> {
        core.key_buf.clear();
        drag_in(core, col, row, side);
        core.key_buf.clone()
    }

    #[test]
    fn motion_is_reported_once_per_cell_not_once_per_pixel() {
        // A mouse reports position in pixels; the protocol reports it in cells. Under
        // `?1003` (or `?1002` with a button held) crossing one 8px column slowly with a
        // 1000 Hz mouse therefore produces dozens of byte-identical reports. Downstream a
        // TUI that repaints per motion report repaints tens of times per cell, and a
        // tmux-over-ssh session pays a round trip for every duplicate. xterm and foot both
        // stay silent until the cell changes.
        let mut core = core_80x24(true);
        core.feed_test_bytes(b"\x1b[?1003h\x1b[?1006h"); // any-motion reporting, SGR

        assert_eq!(
            motion_report(&mut core, 4, 2, Side::Left),
            b"\x1b[<35;5;3M",
            "the first motion into a cell is reported"
        );
        for _ in 0..12 {
            assert!(
                motion_report(&mut core, 4, 2, Side::Left).is_empty(),
                "a further twelve events in the same cell say nothing new"
            );
        }

        assert_eq!(
            motion_report(&mut core, 5, 2, Side::Left),
            b"\x1b[<35;6;3M",
            "crossing a column boundary is news"
        );
        assert_eq!(
            motion_report(&mut core, 5, 3, Side::Left),
            b"\x1b[<35;6;4M",
            "so is a row change"
        );
        assert!(
            motion_report(&mut core, 5, 3, Side::Right).is_empty(),
            "but half a cell is not a cell — the report has no room for it"
        );

        // A gesture boundary re-arms it, so a program hears a fresh position at the start
        // of one rather than inheriting the last one's.
        press(&mut core, MouseButton::Left, true, 5, 3);
        assert_eq!(
            motion_report(&mut core, 5, 3, Side::Left),
            b"\x1b[<32;6;4M",
            "the first motion after a press reports, on the very same cell"
        );
        press(&mut core, MouseButton::Left, false, 5, 3);
        assert!(!motion_report(&mut core, 5, 3, Side::Left).is_empty());

        // Leaving the grid does the same: coming back to the cell it left from is a new
        // position, because in between the pointer was somewhere else entirely.
        assert!(motion_report(&mut core, 5, 3, Side::Left).is_empty());
        core.pointer(PointerEvent::Left, input::Mods::NONE).unwrap();
        assert!(!motion_report(&mut core, 5, 3, Side::Left).is_empty());
    }

    #[test]
    fn a_double_click_selects_the_word() {
        // A double-click inside "bnkterm" selects the whole word, bounded by the
        // trailing space, with no drag needed.
        let mut core = core_80x24(true);
        click(&mut core, MouseButton::Left, true, 2, 0, 2, Side::Left);
        click(&mut core, MouseButton::Left, false, 2, 0, 1, Side::Left);
        assert_eq!(primary_offer(&mut core).as_deref(), Some("bnkterm"));
    }

    #[test]
    fn a_triple_click_selects_the_line() {
        // A triple-click selects the whole logical line (row 0's title), trailing
        // blanks trimmed.
        let mut core = core_80x24(true);
        let row = core.screen.abs_row(0);
        let expected = grid_text(&core.screen, (row, 0), (row, 79));
        click(&mut core, MouseButton::Left, true, 5, 0, 3, Side::Left);
        click(&mut core, MouseButton::Left, false, 5, 0, 1, Side::Left);
        let offered = primary_offer(&mut core);
        assert_eq!(offered.as_deref(), Some(expected.as_str()));
        assert!(
            offered.as_deref().is_some_and(|t| t.starts_with("bnkterm")),
            "the line begins with the title"
        );
    }

    /// The text `core` would copy right now, or `None` with nothing selected.
    fn copied(core: &mut TerminalCore) -> Option<String> {
        core.copy_selection();
        core.take_outbox().into_iter().find_map(|m| match m {
            ToWindow::OfferSelection(bytes) => String::from_utf8(bytes).ok(),
            _ => None,
        })
    }

    /// Double-click the word at display `(row, col)` and leave it selected.
    fn select_word(core: &mut TerminalCore, col: usize, row: usize) {
        click(core, MouseButton::Left, true, col, row, 2, Side::Left);
        click(core, MouseButton::Left, false, col, row, 1, Side::Left);
        core.take_outbox();
    }

    #[test]
    fn a_selection_survives_the_child_printing_under_it() {
        // The other half of the reported bug: selecting text while a program prints used
        // to be impossible, because every burst of output dropped the selection. It now
        // holds, and keeps copying the words it was made over even after they have
        // scrolled off the top of the screen.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        assert_eq!(copied(&mut core).as_deref(), Some("world"));

        // A screenful and a half of output: "hello world" scrolls into history.
        core.feed_test_bytes(b"\r\n");
        for i in 0..30 {
            core.feed_test_bytes(format!("line {i}\r\n").as_bytes());
        }

        assert!(core.selection.is_some(), "output did not evaporate it");
        assert_eq!(
            copied(&mut core).as_deref(),
            Some("world"),
            "and it still names the line it was made over"
        );
    }

    #[test]
    fn a_drag_in_progress_stays_anchored_while_the_child_prints() {
        // The anchor a drag pivots around is absolute too. Here the child scrolls the
        // grid by three rows *between* the press and the drag, so the anchored line moves
        // from display row 5 to display row 2 while the pointer follows it down there. A
        // display-row anchor would still be pinned at row 5 and would drag out three rows
        // of the wrong text; an absolute one is still on the line the user grabbed.
        let mut core = core_showing("\x1b[6;1Halpha beta"); // display row 5
        click(&mut core, MouseButton::Left, true, 0, 5, 1, Side::Left); // press on "alpha"

        core.feed_test_bytes(b"\x1b[24;1H\r\n\r\n\r\n"); // three rows into history

        drag_in(&mut core, 9, 2, Side::Right); // the same line, now three rows higher
        click(&mut core, MouseButton::Left, false, 9, 2, 1, Side::Left);
        assert_eq!(copied(&mut core).as_deref(), Some("alpha beta"));
    }

    #[test]
    fn the_alt_screen_takes_the_selection_with_it() {
        // Rows on the alt screen are a different buffer with no history: an id minted on
        // the primary names nothing there, so the switch ends the epoch and the prune
        // drops the selection rather than highlighting whatever now sits at that row.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        assert!(core.selection.is_some());

        core.feed_test_bytes(b"\x1b[?1049h");

        assert!(core.selection.is_none(), "the selection did not follow");
    }

    #[test]
    fn a_full_screen_erase_takes_the_selection_with_it() {
        // `clear` (ED 2 + ED 3): the text the user picked is gone, so the highlight goes
        // with it rather than sitting over the blank cells that replaced it.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        core.feed_test_bytes(b"\x1b[2J\x1b[3J");
        assert!(core.selection.is_none());
    }

    #[test]
    fn typing_ends_the_selection() {
        // What xterm, ghostty and alacritty all do, for the reason the capture exists: a
        // keystroke is the user moving on, and a highlight left sitting over cells the
        // shell is about to redraw is a promise the terminal then has to keep.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        assert_eq!(copied(&mut core).as_deref(), Some("world"));

        core.key(
            input::Key::plain('x'),
            input::Mods::NONE,
            input::KeyEvent::Press,
        )
        .unwrap();

        assert!(core.selection.is_none(), "typing left the highlight up");
        assert_eq!(copied(&mut core), None, "and left something to copy");
    }

    #[test]
    fn a_drag_across_half_a_wide_character_copies_it_whole() {
        // The drag starts on the right half of `漢` (columns 1-2) and ends on the left half
        // of `字` (columns 4-5). Both are copied whole, over the span the highlight covers.
        let mut core = core_showing("a漢b字c");
        press_in(&mut core, 2, 0, Side::Left);
        drag_in(&mut core, 4, 0, Side::Right);
        press(&mut core, MouseButton::Left, false, 4, 0);

        assert_eq!(copied(&mut core).as_deref(), Some("漢b字"));
        let (start, end) = core.selection.expect("a selection").ordered();
        assert_eq!((start.1, end.1), (1, 5), "the selection covers both pairs");
    }

    #[test]
    fn typing_ends_a_dragged_selection() {
        // The gesture the user actually performs: press, drag, release, then type. The
        // word-click path is a different route into `set_selection`, so it is worth
        // walking the drag one too.
        let mut core = core_showing("hello world");
        press_in(&mut core, 0, 0, Side::Left);
        drag_in(&mut core, 4, 0, Side::Right);
        press(&mut core, MouseButton::Left, false, 4, 0);
        core.take_outbox();
        assert_eq!(copied(&mut core).as_deref(), Some("hello"));

        core.key(
            input::Key::plain('x'),
            input::Mods::NONE,
            input::KeyEvent::Press,
        )
        .unwrap();

        assert!(core.selection.is_none(), "typing left the highlight up");
    }

    #[test]
    fn letting_go_of_a_key_does_not_end_the_selection() {
        // The window swallows Ctrl+Shift+C as a copy chord, but `State::on_key_release`
        // forwards its *release* like any other key's, and under kitty's
        // `REPORT_EVENT_TYPES` that release encodes to bytes of its own. Treating "the
        // encoder produced something" as typing would therefore have the copy chord wipe
        // the selection it was pressed to copy — alacritty's #8509 by another route.
        let mut core = core_showing("hello world");
        core.feed_test_bytes(b"\x1b[>3u"); // disambiguate + report event types
        select_word(&mut core, 6, 0);

        let sent = core
            .key(
                input::Key::plain('c'),
                input::Mods::CTRL | input::Mods::SHIFT,
                input::KeyEvent::Release,
            )
            .unwrap();

        assert!(sent, "the release must encode, or this proves nothing");
        assert_eq!(copied(&mut core).as_deref(), Some("world"));
    }

    #[test]
    fn scrolling_back_to_look_at_a_selection_keeps_it() {
        // The scrollback chords are handled in the window and never reach the child, so
        // they are not typing. Scrolling up to check what you picked must not destroy it.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        core.feed_test_bytes(&[b'\n'; 40]); // into history, so there is somewhere to scroll

        assert!(core.handle_scroll_key(input::Key::PageUp, input::Mods::SHIFT));

        assert_eq!(copied(&mut core).as_deref(), Some("world"));
    }

    #[test]
    fn a_paste_ends_the_selection_too() {
        // A paste is input the child acts on, so it ends the highlight like a keystroke.
        // Middle-click-repeatedly is unharmed: the primary offer went to the window when
        // the drag ended and stands on its own, so losing the highlight retracts nothing.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);

        core.paste(b"ls".to_vec()).unwrap();

        assert!(core.selection.is_none());
    }

    #[test]
    fn the_clipboard_keeps_what_was_highlighted_when_the_child_rewrites_the_cells() {
        // Why the text is captured when the region is drawn rather than read back at copy
        // time. Output deliberately does not end a selection, so a program repainting a
        // line in place — a prompt redraw, a progress bar rewinding with `\r` — used to
        // swap out what Ctrl+Shift+C handed over with the user having done nothing.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        assert_eq!(copied(&mut core).as_deref(), Some("world"));

        // Same rows and same epoch, different characters underneath.
        core.feed_test_bytes(b"\x1b[Hgoodbye moon");
        let sel = core
            .selection
            .expect("an in-place repaint is not one of the things that ends a selection");
        assert_eq!(
            grid_text(&core.screen, sel.anchor, sel.head),
            "e moo",
            "the cells under the highlight really did change"
        );

        assert_eq!(copied(&mut core).as_deref(), Some("world"));
    }

    #[test]
    fn a_paused_drag_copies_what_the_release_showed() {
        // The capture is re-taken as the button comes up: that is the last instant the
        // gesture belongs to the user, and it is what they were looking at when they let
        // go, rather than what the pointer last happened to pass over.
        let mut core = core_showing("abc");
        press_in(&mut core, 0, 0, Side::Left);
        drag_in(&mut core, 2, 0, Side::Right); // across all three, capturing "abc"
        core.feed_test_bytes(b"\x1b[Hxyz");
        core.take_outbox();

        press(&mut core, MouseButton::Left, false, 2, 0);

        assert_eq!(primary_offer(&mut core).as_deref(), Some("xyz"));
    }

    #[test]
    fn a_drag_over_blank_cells_keeps_its_highlight_and_offers_nothing() {
        // Selecting the empty space below the prompt sticks, the way wezterm and ghostty
        // leave it — but blanks trim away to nothing, so the capture is empty and there
        // is no text to put on either clipboard.
        let mut core = core_showing("");
        press_in(&mut core, 0, 3, Side::Left);
        drag_in(&mut core, 4, 3, Side::Right);
        press(&mut core, MouseButton::Left, false, 4, 3);

        assert!(core.selection.is_some(), "the highlight stays");
        assert_eq!(primary_offer(&mut core), None, "with nothing to offer");
        assert_eq!(copied(&mut core), None);
    }

    #[test]
    fn a_resize_carries_the_selection() {
        // A width change reflows and renumbers the rows, but the selection is translated to
        // the cells it was made over rather than dropped: "world" stays selected across it.
        // A resize sends no output, so this is the path that proves the carry does not depend
        // on the child saying something.
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        let picked = core.selection.expect("word selected");
        assert_eq!(grid_text(&core.screen, picked.anchor, picked.head), "world");

        core.resize(geom(40, 12, 40 * 8, 12 * 16));

        let sel = core
            .selection
            .expect("the selection was carried across the reflow");
        assert_eq!(grid_text(&core.screen, sel.anchor, sel.head), "world");
    }

    #[test]
    fn an_alt_screen_resize_drops_the_selection() {
        // A selection over the alt screen has no scrollback to anchor to and the alt screen
        // only clamps (never reflows), so a resize there renumbers it out from under the
        // selection with nothing to translate against: it goes.
        let mut core = core_showing("hello world");
        core.feed_test_bytes(b"\x1b[?1049h"); // enter the alt screen
        core.feed_test_bytes(b"hello world"); // and put a word on it to select
        select_word(&mut core, 6, 0);
        assert!(core.selection.is_some());
        core.resize(geom(40, 12, 40 * 8, 12 * 16));
        assert!(
            core.selection.is_none(),
            "an alt-screen selection is dropped"
        );
    }

    #[test]
    fn a_selection_is_dropped_once_the_ring_outruns_it() {
        // Absolute ids are stable, but not immortal: history is finite, and once the last
        // of a selection's rows has aged off the front of the ring there is nothing left
        // to copy. (A selection that has merely scrolled out of *sight* is kept — see
        // `a_selection_survives_the_child_printing_under_it`.)
        let mut core = core_showing("hello world");
        select_word(&mut core, 6, 0);
        assert!(core.selection.is_some());

        // More line feeds than the ring is deep, in one burst: row 0 is evicted.
        core.feed_test_bytes(&b"\r\n".repeat(11_000));

        assert!(
            core.selection.is_none(),
            "the row it named has aged out of history"
        );
    }

    #[test]
    fn a_pending_character_drag_is_cancelled_when_its_anchor_dies() {
        // A single click pins an anchor but paints no selection until the drag crosses
        // a cell midline (so a plain click never flashes). That leaves a window in which
        // the drag holds row ids that nothing else speaks for: if the grid renumbers in
        // it, the drag has to die too, or the next motion resumes from an anchor that now
        // names a row the user never touched and drags out a span of it.
        let mut core = core_showing("alpha beta");
        click(&mut core, MouseButton::Left, true, 0, 0, 1, Side::Left); // press: anchor, no selection
        assert!(core.selection.is_none());
        assert!(core.drag.is_some());

        core.feed_test_bytes(b"\x1b[?1049h"); // alt screen: every id minted is void

        drag_to(&mut core, 5, 3);
        click(&mut core, MouseButton::Left, false, 5, 3, 1, Side::Left);
        assert!(
            core.selection.is_none(),
            "the drag did not resume from a dead anchor"
        );
        assert_eq!(copied(&mut core), None, "and nothing was offered to copy");
    }

    #[test]
    fn a_bare_click_selects_nothing() {
        // Press and release on the same cell (no drag) selects nothing: no primary
        // offer, and the selection is cleared so no stray cell stays highlighted.
        let mut core = core_80x24(true);
        press(&mut core, MouseButton::Left, true, 0, 0);
        press(&mut core, MouseButton::Left, false, 0, 0);
        assert!(
            !core
                .take_outbox()
                .iter()
                .any(|m| matches!(m, ToWindow::OfferPrimary(_))),
            "a bare click offers nothing"
        );
        assert!(core.selection.is_none(), "and leaves no selection");
    }

    #[test]
    fn a_click_never_highlights_the_clicked_cell() {
        // The single-click "blink": a plain click, even with sub-cell motion that
        // stays on the press's side of the cell's midline, must never create a
        // selection, so the clicked cell does not flash the selection colour.
        let mut core = core_80x24(true);
        press(&mut core, MouseButton::Left, true, 2, 0);
        assert!(core.selection.is_none(), "the press alone shows nothing");
        drag_to(&mut core, 2, 0); // motion that crosses no midline
        assert!(core.selection.is_none(), "same-half motion shows nothing");
        press(&mut core, MouseButton::Left, false, 2, 0);
        assert!(core.selection.is_none(), "and the release leaves nothing");
    }

    #[test]
    fn a_press_in_the_trailing_half_leaves_the_blank_out() {
        // The rounding this exists for: aiming at "authentication" and landing a few
        // pixels early, on the blank before it. The press is in the blank's trailing
        // half, so the anchor rounds forward and the copy starts at the 'a'.
        let mut core = core_showing("run authentication");
        press_in(&mut core, 3, 0, Side::Right); // the blank before 'a', right half
        drag_in(&mut core, 6, 0, Side::Right); // past 't's midline
        press(&mut core, MouseButton::Left, false, 6, 0);
        assert_eq!(primary_offer(&mut core).as_deref(), Some("aut"));
    }

    #[test]
    fn a_cell_joins_the_selection_only_past_its_midline() {
        // Both ends round: entering a cell's near half leaves it out, crossing its
        // midline pulls it in. So a drag can pick exactly one character — which the
        // old inclusive-cell drag could not do at all.
        let mut core = core_showing("abc");
        press_in(&mut core, 1, 0, Side::Left);
        drag_in(&mut core, 1, 0, Side::Right); // across 'b' alone
        assert_eq!(copied(&mut core).as_deref(), Some("b"));

        // Backing off to the near half again empties it.
        drag_in(&mut core, 1, 0, Side::Left);
        assert!(core.selection.is_none());
    }

    #[test]
    fn a_drag_between_two_midlines_selects_nothing() {
        // From one cell's trailing half to the next cell's leading half: both
        // endpoints round to the same edge, no whole cell is covered, and the
        // release offers nothing.
        let mut core = core_showing("run authentication");
        press_in(&mut core, 3, 0, Side::Right);
        drag_in(&mut core, 4, 0, Side::Left);
        assert!(core.selection.is_none());
        press(&mut core, MouseButton::Left, false, 4, 0);
        assert_eq!(primary_offer(&mut core), None);
    }

    #[test]
    fn a_backward_drag_rounds_at_both_ends_too() {
        // Symmetric, unlike ghostty's 60/40 split: dragging right-to-left, the same
        // two edges select the same cells. From before 'd' back past 'b' covers
        // only 'c'.
        let mut core = core_showing("abcde");
        press_in(&mut core, 3, 0, Side::Left);
        drag_in(&mut core, 1, 0, Side::Right);
        assert_eq!(copied(&mut core).as_deref(), Some("c"));
    }

    #[test]
    fn a_press_past_the_last_midline_starts_on_the_next_row() {
        // A row's right margin and the next row's edge 0 are the same boundary:
        // anchoring in the last column's trailing half starts the selection at the
        // top of the row below, not on the last cell of the row pressed.
        let mut core = core_showing("top\r\nsecond");
        press_in(&mut core, 79, 0, Side::Right);
        drag_in(&mut core, 5, 1, Side::Right);
        assert_eq!(copied(&mut core).as_deref(), Some("second"));
    }

    #[test]
    fn a_drag_to_a_rows_first_edge_stops_at_the_row_above() {
        // The mirror seam: ending on a row's edge 0 covers none of that row, so the
        // selection runs to the end of the row above instead.
        let mut core = core_showing("top\r\nsecond");
        press_in(&mut core, 1, 0, Side::Left);
        drag_in(&mut core, 0, 1, Side::Left);
        assert_eq!(copied(&mut core).as_deref(), Some("op"));
    }

    #[test]
    fn a_double_click_in_the_trailing_half_still_takes_the_whole_word() {
        // Edge rounding is a character-drag affair: word (and line) clicks stay
        // greedy, a click anywhere on a word means that word.
        let mut core = core_80x24(true);
        click(&mut core, MouseButton::Left, true, 2, 0, 2, Side::Right);
        click(&mut core, MouseButton::Left, false, 2, 0, 1, Side::Right);
        assert_eq!(primary_offer(&mut core).as_deref(), Some("bnkterm"));
    }

    #[test]
    fn a_drag_over_empty_space_sticks_but_copies_nothing() {
        // Selecting the empty area below the prompt: the drag leaves a live selection
        // (so the highlight sticks after release, matching wezterm/ghostty) yet offers
        // nothing to the clipboard, since blank cells have no text to copy. Row 20 is
        // below the demo's last content row (16), so it is all blank cells.
        let mut core = core_80x24(true);
        press(&mut core, MouseButton::Left, true, 5, 20);
        drag_to(&mut core, 15, 20);
        press(&mut core, MouseButton::Left, false, 15, 20);
        assert!(
            core.selection.is_some(),
            "the empty-space selection stays highlighted"
        );
        assert!(
            !core
                .take_outbox()
                .iter()
                .any(|m| matches!(m, ToWindow::OfferPrimary(_) | ToWindow::OfferSelection(_))),
            "but nothing is offered to copy"
        );
    }

    #[test]
    fn middle_click_requests_a_primary_paste() {
        // With no program grabbing the mouse, a middle-click asks the window to paste
        // the primary selection (the window owns the data device).
        let mut core = core_80x24(true);
        press(&mut core, MouseButton::Middle, true, 5, 5);
        assert!(
            core.take_outbox()
                .iter()
                .any(|m| matches!(m, ToWindow::PastePrimary)),
            "middle-click requests a primary paste"
        );
    }
}
