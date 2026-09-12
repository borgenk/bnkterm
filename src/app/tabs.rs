//! Live terminal tabs between the Wayland window shell and per-tab cores.
//!
//! The main thread remains the sole owner of every parser and grid. Each core has
//! one gather thread and PTY, while this manager owns stable identities, active-tab
//! routing, a shared pump budget, and child teardown.
//!
//! ```text
//!                        Tabs (main thread)
//!                  active index / stable TabId
//!                    /          |          \
//!       TerminalCore(0) TerminalCore(1) TerminalCore(2)
//!          | gatherer       | gatherer       | gatherer
//!          v                v                v
//!        shell            shell            shell
//!
//!   input/frame/title -> active core only
//!   pump/resize/poll  -> every core
//! ```

use std::os::fd::RawFd;
use std::rc::Rc;
use std::time::{Duration, Instant};

use super::message::{ToTerminal, ToWindow};
use super::terminal::{PumpOutcome, TerminalCore};
use crate::color::Theme;
use crate::config::{ShellStartupConfig, TabBarConfig};
use crate::error::Result;
use crate::gather::GatherEnd;
use crate::notice::Notice;
use crate::platform::freetype::Fonts;
use crate::platform::geom::Scale;
use crate::pty::{Launch, ZombieChild};
use crate::render::display::DisplayList;
use crate::tab_bar::{self, BarGeom, Lift, Slot, TabLabel};
use crate::term_render::CellMetrics;

/// One global gather budget per event-loop turn. Foreground output is parsed
/// first; background tabs rotate through whatever remains.
const GATHER_TIME_BUDGET: Duration = Duration::from_millis(2);
const GATHER_BYTE_BUDGET: usize = 1024 * 1024;

/// Stable identity for a tab across opens and closes. Indices may move;
/// identities do not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct TabId(u64);

/// The direction the active tab reorders in: toward index 0 or toward the end.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Reorder {
    Prev,
    Next,
}

/// One managed terminal and its stable identity.
struct TabEntry {
    id: TabId,
    core: TerminalCore,
}

/// A tab drag in progress. The tab is held by identity, not index, because a
/// background tab's shell can exit mid-drag and shift every index under us; an id is
/// immune (the dragged tab is always the active one, so the reorder keeps it active).
///
/// `press_x`, `grab_dx`, and `left` are all device pixels. `lifted` stays false until
/// the pointer has travelled past the lift threshold, so a press that never moves far
/// enough is a plain click, not a lift. `grab_dx` is where inside the block the tab
/// was grabbed; `left` is the block's current (clamped) floating left edge, valid once
/// `lifted`.
struct TabDrag {
    id: TabId,
    press_x: i32,
    grab_dx: i32,
    left: i32,
    lifted: bool,
}

/// The app-shell owner of terminal tabs.
pub(super) struct Tabs {
    entries: Vec<TabEntry>,
    active: usize,
    next_id: u64,
    /// Starting index for the background portion of the next pump turn.
    pump_cursor: usize,
    /// Closed PTY children that have not become waitable yet.
    reaping: Vec<ZombieChild>,
    /// A title, tab count, or selection change requires rebuilding the future bar.
    bar_dirty: bool,
    /// Cell-space layout shared by painting and pointer hit testing.
    bar_slots: Vec<Slot>,
    /// The strip's device-pixel geometry, set by the window each resize (`None`
    /// until the first one, which always precedes a paint that shows the bar).
    bar_geom: Option<BarGeom>,
    /// A mouse drag-to-reorder in flight, if any. Owns the whole gesture: the pointer
    /// only converts a surface coordinate to device x and hands it over.
    drag: Option<TabDrag>,
    /// Strip appearance and layout, the source of truth for `layout`/`fill_bar`.
    cfg: TabBarConfig,
    /// The configured grid colours.
    theme: Rc<Theme>,
    /// How every shell this session spawns is started. Held here because a tab opened an
    /// hour in has to start the way the first one did, integration shims included.
    launch: Launch,
    /// When a shell's own startup is slow enough to say so. Held beside `launch` for the
    /// same reason: it is a property of the session, not of a tab.
    shell_startup: ShellStartupConfig,
    /// The transient corner notice, when one stands. Window-level rather than per-core
    /// because it is chrome over the window, and because the tab it speaks for is the
    /// one that just opened, which is the active one by the time it is raised.
    notice: Option<Notice>,
    /// Messages already translated from per-core facts into window actions.
    outbox: Vec<ToWindow>,
}

impl Tabs {
    /// Wrap the initial terminal core as tab zero, under the given strip config, with the
    /// launch every tab in this session starts its shell with and the budget a shell's own
    /// startup is held to.
    pub(super) fn new(
        core: TerminalCore,
        cfg: TabBarConfig,
        theme: Rc<Theme>,
        launch: Launch,
        shell_startup: ShellStartupConfig,
    ) -> Self {
        let mut tabs = Self {
            entries: Vec::new(),
            active: 0,
            next_id: 0,
            pump_cursor: 0,
            reaping: Vec::new(),
            bar_dirty: false,
            bar_slots: Vec::new(),
            bar_geom: None,
            drag: None,
            cfg,
            theme,
            launch,
            shell_startup,
            notice: None,
            outbox: Vec::new(),
        };
        let id = tabs.allocate_id();
        tabs.entries.push(TabEntry { id, core });
        tabs
    }

    /// Spawn the first tab's shell, once the window has its granted size. Later tabs get
    /// theirs from [`open`](Self::open); both go through the session's launch,
    /// which is why neither caller has to know them.
    pub(super) fn spawn_active_shell(&mut self) -> Result<(usize, usize)> {
        self.entries[self.active].core.spawn_shell(&self.launch)
    }

    /// Number of live tabs.
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no terminal remains (the window is shutting down).
    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The one-row tab bar is hidden for the byte-identical single-tab path.
    pub(super) fn shows_bar(&self) -> bool {
        self.len() >= 2
    }

    /// The visible terminal core.
    pub(super) fn active(&self) -> &TerminalCore {
        &self.entries[self.active].core
    }

    /// The visible terminal core, mutably.
    pub(super) fn active_mut(&mut self) -> &mut TerminalCore {
        &mut self.entries[self.active].core
    }

    /// The visible tab's stable identity.
    pub(super) fn active_id(&self) -> Option<TabId> {
        self.entries.get(self.active).map(|entry| entry.id)
    }

    /// Open a live shell at the current geometry. Existing tabs are not touched
    /// until PTY and gatherer startup both succeed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn open(
        &mut self,
        cols: usize,
        rows: usize,
        metrics: CellMetrics,
        width: u32,
        height: u32,
        pad: i32,
        window_focused: bool,
    ) -> Result<TabId> {
        let mut core = TerminalCore::new(false, cols, rows, metrics, width, height, pad);
        core.set_theme(Rc::clone(&self.theme));
        if let Err(error) = core.spawn_shell(&self.launch) {
            if let Some(child) = core.into_child() {
                self.reaping.push(child);
            }
            return Err(error);
        }
        core.apply(ToTerminal::Focus(window_focused))?;

        if !self.is_empty() {
            self.active_mut().apply(ToTerminal::Focus(false))?;
        }
        let id = self.allocate_id();
        self.entries.push(TabEntry { id, core });
        self.active = self.entries.len() - 1;
        self.bar_dirty = true;
        self.rebuild_bar();
        self.queue_active_title();
        Ok(id)
    }

    /// Add a tab on a static demo grid under `title`. Not [`open`](Self::open),
    /// which spawns a shell and would put the environment into the picture.
    #[cfg(test)]
    pub(super) fn push_demo_tab(
        &mut self,
        metrics: CellMetrics,
        width: u32,
        height: u32,
        pad: i32,
        title: &str,
    ) {
        // The grid size the tabs already agree on, so a new one lands the same shape.
        let (cols, rows) = self.active().dimensions();
        let mut core = TerminalCore::new(true, cols, rows, metrics, width, height, pad);
        core.set_theme(Rc::clone(&self.theme));
        core.feed_test_bytes(format!("\x1b]2;{title}\x07").as_bytes());
        let id = self.allocate_id();
        self.entries.push(TabEntry { id, core });
        self.bar_dirty = true;
        self.rebuild_bar();
    }

    /// Close `id`, repairing the active index and handing its child to the zombie
    /// sweep. Returns true when this removed the last tab.
    pub(super) fn close(&mut self, id: TabId, window_focused: bool) -> bool {
        let Some(index) = self.entries.iter().position(|entry| entry.id == id) else {
            return self.entries.is_empty();
        };
        let was_active = index == self.active;
        let entry = self.entries.remove(index);
        if let Some(child) = entry.core.into_child() {
            self.reaping.push(child);
        }
        self.bar_dirty = true;
        // A drag cannot outlive its own tab, nor the bar itself once a single tab
        // remains and there is nowhere left to drop. A *background* close leaves the
        // drag alone: it holds an id, so it simply continues against the new run.
        if self.drag.as_ref().is_some_and(|drag| drag.id == id) || !self.shows_bar() {
            self.drag = None;
        }

        if self.is_empty() {
            self.active = 0;
            self.pump_cursor = 0;
            self.bar_slots.clear();
            self.outbox.push(ToWindow::Closed);
            return true;
        }

        if index < self.active {
            self.active -= 1;
        } else if was_active {
            // The old right neighbor shifted into `index`; if there was none, use
            // the new last entry.
            self.active = index.min(self.entries.len() - 1);
            let _ = self.active_mut().apply(ToTerminal::Focus(window_focused));
            self.active_mut().dirty = true;
            self.queue_active_title();
        }
        self.pump_cursor %= self.entries.len();
        self.rebuild_bar();
        false
    }

    /// Select a tab by stable identity. Returns whether the active tab changed.
    pub(super) fn select(&mut self, id: TabId, window_focused: bool) -> bool {
        let Some(index) = self.entries.iter().position(|entry| entry.id == id) else {
            return false;
        };
        if index == self.active {
            return false;
        }
        let _ = self.active_mut().apply(ToTerminal::Focus(false));
        self.active = index;
        let _ = self.active_mut().apply(ToTerminal::Focus(window_focused));
        self.active_mut().dirty = true;
        self.bar_dirty = true;
        self.rebuild_bar();
        self.queue_active_title();
        true
    }

    /// Select the previous tab, wrapping at the left edge.
    pub(super) fn prev(&mut self, window_focused: bool) -> bool {
        if self.len() < 2 {
            return false;
        }
        let index = (self.active + self.entries.len() - 1) % self.entries.len();
        let id = self.entries[index].id;
        self.select(id, window_focused)
    }

    /// Select the next tab, wrapping at the right edge.
    pub(super) fn next(&mut self, window_focused: bool) -> bool {
        if self.len() < 2 {
            return false;
        }
        let index = (self.active + 1) % self.entries.len();
        let id = self.entries[index].id;
        self.select(id, window_focused)
    }

    /// Move the active tab to `index` (clamped to the run), sliding the tabs in
    /// between over to fill the gap, and keep it active wherever it lands. Returns
    /// whether the order actually changed. This is the one reorder primitive: the
    /// keyboard's adjacent [`move_active`](Self::move_active) is a wrapper on it, and
    /// a mouse drag hands it an arbitrary landing slot. The tab's content is
    /// untouched; only the bar reflows.
    pub(super) fn move_active_to(&mut self, index: usize) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        let target = index.min(self.entries.len() - 1);
        if target == self.active {
            return false;
        }
        // Remove-and-insert (not a swap): a non-adjacent move must preserve the
        // order of every tab it slides past. For an adjacent move it reduces to the
        // swap the keyboard reorder used to do.
        let entry = self.entries.remove(self.active);
        self.entries.insert(target, entry);
        self.active = target;
        self.bar_dirty = true;
        self.rebuild_bar();
        true
    }

    /// Reorder the active tab one slot toward index 0 (`Prev`) or the end (`Next`),
    /// keeping it the active tab. Reordering does not wrap: unlike selection (which
    /// cycles), flinging a tab past the edge to the far side would be surprising, so
    /// an edge move is a no-op. Returns whether the order changed. The active tab's
    /// content is untouched; only the bar reflows.
    pub(super) fn move_active(&mut self, dir: Reorder) -> bool {
        if self.len() < 2 {
            return false;
        }
        let target = match dir {
            Reorder::Prev if self.active > 0 => self.active - 1,
            Reorder::Next if self.active + 1 < self.entries.len() => self.active + 1,
            _ => return false,
        };
        self.move_active_to(target)
    }

    /// Arm a drag on tab `id`, pressed at device-x `x`. Selection has already made it
    /// the active tab (the pointer path selects on press); this only records where in
    /// the block the grab landed so the float can track the pointer. The tab does not
    /// lift until [`drag_to`](Self::drag_to) sees the pointer cross the threshold, so a
    /// press that never moves stays a plain click. A no-op before the first resize (no
    /// geometry) or for an unknown id.
    pub(super) fn begin_drag(&mut self, id: TabId, x: i32) {
        let Some(index) = self.entries.iter().position(|entry| entry.id == id) else {
            return;
        };
        let Some(bar) = self.bar_geom else {
            return;
        };
        let Some((left, _)) = tab_bar::block_px(&self.bar_slots, &bar, index) else {
            return;
        };
        self.drag = Some(TabDrag {
            id,
            press_x: x,
            grab_dx: x - left,
            left,
            lifted: false,
        });
    }

    /// Advance a live drag to device-x `x`: lift the tab once the pointer has passed
    /// the threshold, float its block under the grab point (clamped to the run), and
    /// reorder when the block's own centre crosses into a new slot. Geometry is read
    /// fresh each call, so a resize or a background tab exiting mid-drag is absorbed
    /// rather than remembered wrongly. A no-op with no drag armed or no geometry.
    pub(super) fn drag_to(&mut self, x: i32) {
        let Some(mut drag) = self.drag.take() else {
            return;
        };
        let Some(bar) = self.bar_geom else {
            return;
        };
        if !drag.lifted {
            // The threshold is half a cell: enough that a click's jitter never lifts
            // the tab, far below the half-block a reorder needs, so it can only gate
            // the lift, never the landing. `metrics.w` is device pixels and already
            // tracks DPI, so this is HiDPI-correct with no extra plumbing.
            let threshold = (bar.metrics.w / 2).max(1);
            if (x - drag.press_x).abs() <= threshold {
                self.drag = Some(drag);
                return;
            }
            drag.lifted = true;
        }
        // Slot zero is never clipped, so its width is the true pitch.
        let pitch = self
            .bar_slots
            .first()
            .map_or(1, |slot| slot.cells.len() as i32 * bar.metrics.w)
            .max(1);
        // A resize since the press can have shrunk the pitch below the grab offset;
        // keep the grip inside the block so the float cannot invert.
        let grab = drag.grab_dx.clamp(0, pitch - 1);
        let n = self.bar_slots.len() as i32;
        // Clamp the float to the run: the tab pins at the ends instead of sliding into
        // the slack, so it can never be dragged somewhere it cannot land.
        let max_left = bar.pad + (n - 1).max(0) * pitch;
        let left = (x - grab).clamp(bar.pad, max_left);
        if left != drag.left {
            drag.left = left;
            self.bar_dirty = true;
        }
        let want = tab_bar::drop_index(&self.bar_slots, &bar, left);
        self.drag = Some(drag);
        // The dragged tab is always the active one, so its current slot is `active`;
        // move only when its centre has crossed into a different block.
        if let Some(want) = want {
            if want != self.active {
                self.move_active_to(want);
            }
        }
    }

    /// End a drag, letting the tab settle from its float back into its block. Keyed by
    /// the caller off the pressed button, so a release anywhere (even dragged off the
    /// grid) ends it. A no-op when no drag was in flight.
    pub(super) fn end_drag(&mut self) {
        if self.drag.take().is_some() {
            self.bar_dirty = true;
        }
    }

    /// Whether a drag is in flight (armed or lifted), so the pointer routes motion here
    /// instead of to the grid beneath the strip.
    pub(super) fn dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Apply the window's current geometry eagerly to every tab so background
    /// output always wraps at the true width, and adopt the strip rectangle the
    /// window computed for this size and top/bottom placement.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resize_all(
        &mut self,
        cols: usize,
        rows: usize,
        width: u32,
        height: u32,
        metrics: CellMetrics,
        label: CellMetrics,
        pad: i32,
        origin_y: i32,
        bar_y: i32,
        bar_h: i32,
        scale: Scale,
    ) -> Result<()> {
        for entry in &mut self.entries {
            entry.core.apply(ToTerminal::Resize {
                cols,
                rows,
                width,
                height,
                metrics,
                pad,
                origin_y,
                scale,
            })?;
            debug_assert_eq!(entry.core.dimensions(), (cols, rows));
        }
        self.bar_geom = Some(BarGeom {
            metrics,
            label,
            surface_width: width as i32,
            pad,
            y: bar_y,
            h: bar_h,
        });
        self.rebuild_bar();
        Ok(())
    }

    /// Every gatherer wake fd, in entry order, for the app's reusable poll set.
    pub(super) fn gather_fds(&self) -> impl Iterator<Item = RawFd> + '_ {
        self.entries.iter().filter_map(|entry| entry.core.poll_fd())
    }

    /// The PTY master of every tab that still owes its child bytes, so the wait ends when
    /// one of them makes room. Empty in the steady state: a write that the kernel took in
    /// full leaves nothing queued, so this registers nothing and costs nothing.
    pub(super) fn write_fds(&self) -> impl Iterator<Item = RawFd> + '_ {
        self.entries
            .iter()
            .filter_map(|entry| entry.core.write_fd())
    }

    /// Push each tab's queued output as far as its child will take it. Every tab, not
    /// just the visible one: a background shell waiting on a paste is owed those bytes
    /// whether or not it is on screen.
    pub(super) fn pump_writes(&mut self) -> Result<()> {
        for entry in &mut self.entries {
            if entry.core.wants_write() {
                entry.core.pump_writes()?;
            }
        }
        Ok(())
    }

    /// Pump the active tab first, then background tabs from a rotating cursor,
    /// sharing one byte/time budget across the whole turn.
    pub(super) fn pump_all(&mut self, window_focused: bool) -> Result<bool> {
        self.sweep_zombies();
        if self.is_empty() {
            return Ok(false);
        }

        let deadline = Instant::now() + GATHER_TIME_BUDGET;
        let mut remaining = GATHER_BYTE_BUDGET;
        let mut more = false;
        let mut ended = Vec::new();

        let active = self.active;
        let outcome = self.entries[active].core.pump(remaining, deadline)?;
        remaining = remaining.saturating_sub(outcome.bytes);
        more |= outcome.more;
        if let Some(end) = outcome.end {
            ended.push((self.entries[active].id, end));
        }
        self.note_settle(active, outcome.bytes, outcome.more);

        // Background tabs share whatever the foreground left of the turn's budget.
        // Under a sustained foreground flood the active tab can consume it all, so
        // hidden grids pause parsing until the flood eases or the user switches
        // (their per-tab gather pools keep draining and then backpressure the
        // child, so no output is lost — only its display is deferred). This is the
        // intended foreground-first bias, not a stall to fix.
        let len = self.entries.len();
        let start = self.pump_cursor % len;
        for offset in 0..len {
            if remaining == 0 || Instant::now() >= deadline {
                break;
            }
            let index = (start + offset) % len;
            if index == active {
                continue;
            }
            let PumpOutcome {
                bytes,
                more: tab_more,
                end,
            } = self.entries[index].core.pump(remaining, deadline)?;
            remaining = remaining.saturating_sub(bytes);
            more |= tab_more;
            self.note_settle(index, bytes, tab_more);
            if let Some(end) = end {
                ended.push((self.entries[index].id, end));
            }
        }
        self.pump_cursor = (self.pump_cursor + 1) % len;

        // Read the startup measurements while their cores are still here: a shell that
        // marks its prompt and exits in the same turn (a startup script, or `exit` typed
        // into a slow shell) is closed below, and taking this afterwards would lose it.
        self.raise_slow_startup_notice();
        // Preserve final title/selection messages before an ended core is removed.
        self.route_core_outboxes();
        for (id, end) in ended {
            if let GatherEnd::ReadError(errno) = end {
                eprintln!("bnkterm: tab PTY read failed: errno {errno}");
            }
            self.close(id, window_focused);
        }
        self.sweep_zombies();

        more |= self.entries.iter().any(|entry| entry.core.has_pending());
        Ok(more)
    }

    /// Raise the corner notice for any tab whose shell has just reached its first prompt
    /// slower than the budget allows.
    ///
    /// Every core is asked, not only the active one: a tab opened and switched away from
    /// still answers, and its shell was still slow. The last one to answer in a turn wins
    /// the corner, which only arises when two tabs open in the same few milliseconds.
    fn raise_slow_startup_notice(&mut self) {
        for index in 0..self.entries.len() {
            let Some(took) = self.entries[index].core.take_startup_time() else {
                continue;
            };
            if let Some(notice) = Notice::shell_startup(took, &self.shell_startup) {
                self.notice = Some(notice);
                // Chrome draws over the active tab's frame, so that is the core whose
                // repaint puts it on screen — whichever tab's shell was the slow one.
                self.mark_active_dirty();
            }
        }
    }

    /// Advance the notice: repaint through its fade, and drop it once the fade is done.
    ///
    /// A standing notice needs nothing — its colours are the same as last frame's — so
    /// only the fade and its end mark the frame dirty. Without that, the hold would
    /// repaint every loop turn for no visible difference.
    pub(super) fn tick_notice_if_due(&mut self) {
        let now = Instant::now();
        let Some(notice) = self.notice.as_ref() else {
            return;
        };
        let (fading, expired) = (notice.fading(now), notice.expired(now));
        if !fading && !expired {
            return;
        }
        if expired {
            self.notice = None;
        }
        self.mark_active_dirty();
    }

    /// Mark the visible core's frame for a rebuild, the lever window chrome has for
    /// getting itself painted.
    fn mark_active_dirty(&mut self) {
        if let Some(entry) = self.entries.get_mut(self.active) {
            entry.core.dirty = true;
        }
    }

    /// When the notice next needs a repaint (the end of its hold, then each fade frame).
    pub(super) fn notice_retry_at(&self) -> Option<Instant> {
        self.notice.as_ref()?.retry_at(Instant::now())
    }

    /// The notice the window should paint, if one stands.
    pub(super) fn notice(&self) -> Option<&Notice> {
        self.notice.as_ref()
    }

    /// Drain and route every core's outbound facts, then return the translated
    /// window actions accumulated so far. The buffer is handed out (leaving the outbox
    /// empty); the caller returns it via [`reclaim_outbox`](Self::reclaim_outbox) after
    /// acting on it, so its capacity is not thrown away each pump.
    pub(super) fn take_outbox(&mut self) -> Vec<ToWindow> {
        self.route_core_outboxes();
        std::mem::take(&mut self.outbox)
    }

    /// Take back the emptied outbox buffer after the caller has acted on it, keeping
    /// its capacity for the next pump. Nothing queues during the window-side drain, so
    /// the outbox is empty here and simply adopts the returned buffer; in the
    /// impossible case a message did arrive, it is kept and the spare buffer dropped.
    pub(super) fn reclaim_outbox(&mut self, drained: Vec<ToWindow>) {
        if self.outbox.is_empty() {
            self.outbox = drained;
        }
    }

    /// Tick only the visible cursor's blink timer.
    pub(super) fn tick_blink_if_due(&mut self) {
        if let Some(entry) = self.entries.get_mut(self.active) {
            entry.core.tick_blink_if_due();
        }
    }

    /// The visible cursor's next blink deadline.
    pub(super) fn blink_deadline(&self) -> Option<Instant> {
        self.entries
            .get(self.active)
            .and_then(|entry| entry.core.blink_deadline())
    }

    /// Carry the visible tab's scrollbar forward a frame. A hidden tab's bar is frozen
    /// wherever it was: nothing is showing it, and it will be ticked (or hidden, if its
    /// history has gone) on the frame that brings it back.
    pub(super) fn tick_scrollbar(&mut self) {
        if let Some(entry) = self.entries.get_mut(self.active) {
            entry.core.tick_scrollbar();
        }
    }

    /// The visible scrollbar's next fade deadline.
    pub(super) fn scrollbar_retry_at(&self) -> Option<Instant> {
        self.entries
            .get(self.active)
            .and_then(|entry| entry.core.scrollbar_retry_at())
    }

    /// Whether the visible scrollbar is mid-fade, so the next compositor frame should
    /// carry it on.
    pub(super) fn scrollbar_animating(&self) -> bool {
        self.entries
            .get(self.active)
            .is_some_and(|entry| entry.core.scrollbar_animating())
    }

    /// Whether the visible terminal has a hyperlink under the pointer, so the window
    /// can offer the hand cursor.
    pub(super) fn hovering_link(&self) -> bool {
        self.entries
            .get(self.active)
            .is_some_and(|entry| entry.core.hovering_link())
    }

    /// Whether the visible terminal's program has grabbed the mouse, so the window can
    /// drop the I-beam over a grid whose drags are not selections.
    pub(super) fn mouse_reporting(&self) -> bool {
        self.entries
            .get(self.active)
            .is_some_and(|entry| entry.core.mouse_reporting())
    }

    /// Whether the active grid or future tab bar needs a frame.
    pub(super) fn needs_frame(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        // A child mid-frame under synchronized output holds the *whole* frame back, tab
        // strip included: the strip is composed over the same grid, so painting it would
        // put exactly the half-drawn screen on display that the child asked us to avoid.
        if self.active_holds_frame() {
            return false;
        }
        self.entries
            .get(self.active)
            .is_some_and(|entry| entry.core.dirty)
            || self.bar_dirty
    }

    /// Whether the visible child is mid-frame under `?2026`.
    fn active_holds_frame(&self) -> bool {
        self.entries
            .get(self.active)
            .is_some_and(|entry| entry.core.holds_frame())
    }

    /// The visible child's synchronized-output deadline, for the event-loop wait: with
    /// no output to wake us, nothing else would.
    pub(super) fn sync_deadline(&self) -> Option<Instant> {
        self.entries
            .get(self.active)
            .and_then(|entry| entry.core.sync_deadline())
    }

    /// Release the visible child's synchronized-output hold once its deadline has passed.
    /// Only the visible tab, because only the visible tab's deadline reaches the
    /// event-loop wait; a hidden tab's stale hold is cleared on the turn after it is
    /// shown, before that wait is computed.
    pub(super) fn tick_sync_if_due(&mut self) {
        if let Some(entry) = self.entries.get_mut(self.active) {
            entry.core.tick_sync_if_due();
        }
    }

    /// The visible child's visual-bell deadline, likewise: the flash has to be taken back
    /// off, and no output is coming to prompt it.
    pub(super) fn bell_deadline(&self) -> Option<Instant> {
        self.entries
            .get(self.active)
            .and_then(|entry| entry.core.bell_deadline())
    }

    /// End the visible child's bell flash if its moment has passed.
    pub(super) fn tick_bell_if_due(&mut self) {
        if let Some(entry) = self.entries.get_mut(self.active) {
            entry.core.tick_bell_if_due();
        }
    }

    /// Push each child its settled resize (`TIOCSWINSZ`). Every tab, not just the visible
    /// one: a window resize reflowed all their grids, so all of them owe their shell the
    /// debounced winsize.
    pub(super) fn flush_winsize_if_due(&mut self) -> crate::error::Result<()> {
        for entry in &mut self.entries {
            entry.core.flush_winsize_if_due()?;
        }
        Ok(())
    }

    /// The soonest pending debounced-resize deadline across all tabs, for the event-loop
    /// wait: with no output coming, nothing else would wake us to deliver it.
    pub(super) fn winsize_deadline(&self) -> Option<Instant> {
        self.entries
            .iter()
            .filter_map(|entry| entry.core.winsize_deadline())
            .min()
    }

    /// Mark the visible terminal for repaint.
    pub(super) fn mark_dirty(&mut self) {
        if let Some(entry) = self.entries.get_mut(self.active) {
            entry.core.dirty = true;
        }
    }

    /// Clear frame dirtiness after a successful present.
    pub(super) fn clear_dirty(&mut self) {
        if let Some(entry) = self.entries.get_mut(self.active) {
            entry.core.dirty = false;
        }
        self.bar_dirty = false;
    }

    /// Compose the visible terminal's display list, then the tab strip over it. The
    /// strip's labels are the proportional interface font, so `fonts` is threaded to
    /// [`tab_bar::fill_bar`] to measure and fit them.
    pub(super) fn fill_frame_list(
        &self,
        out: &mut DisplayList,
        strings: &mut Vec<String>,
        fonts: &Fonts,
    ) {
        self.active().fill_frame_list(out, strings);
        if self.shows_bar() {
            if let Some(bar) = &self.bar_geom {
                // A lifted drag resolves to the slot its id now sits in (a background
                // close may have shifted it) plus the block's floating left edge.
                let lift = self
                    .drag
                    .as_ref()
                    .filter(|drag| drag.lifted)
                    .and_then(|drag| {
                        let slot = self.entries.iter().position(|entry| entry.id == drag.id)?;
                        Some(Lift {
                            slot,
                            left: drag.left,
                        })
                    });
                tab_bar::fill_bar(out, strings, &self.bar_slots, bar, &self.cfg, fonts, lift);
            }
        }
    }

    /// Stable tab identity under a cell column in the bar.
    pub(super) fn tab_at_bar_col(&self, col: usize) -> Option<TabId> {
        let index = tab_bar::hit_test(&self.bar_slots, col)?;
        self.entries.get(index).map(|entry| entry.id)
    }

    /// Allocate a stable identity for a newly opened core.
    fn allocate_id(&mut self) -> TabId {
        let id = TabId(self.next_id);
        // A u64 counter cannot realistically be exhausted (2^64 opens), so wrap
        // instead of carrying a panic on an overflow that can never be reached.
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    /// Queue the active title after a switch/open/active close.
    fn queue_active_title(&mut self) {
        let title = self.active().title().to_string();
        self.outbox.push(ToWindow::Title(title));
    }

    /// Everything a tab has to re-derive from the world outside its byte stream, hung
    /// off the pump so a tab that says nothing costs nothing.
    ///
    /// The tty's mode is re-read on **every batch** that carried bytes. It is one
    /// `TCGETS` against a batch that just cost a parse of up to 1 MiB, and it cannot
    /// wait for the settle: `sudo` restores echo the instant it has the password, and
    /// its command then prints for as long as it likes, so a lock that only cleared on
    /// the settle would sit there for the whole run.
    ///
    /// The `/proc` reads wait for the settle (`!more`), where they are worth their
    /// cost. A core with nothing left pending has likely printed a fresh prompt, so
    /// its working directory may have moved (a bare `cd` reports nothing else) or a
    /// foreground program may have started or exited; when either changed and the bar
    /// is visible, rebuild and repaint it. Both are refreshed even for a lone tab so
    /// they are current the moment a second opens, while the bar work itself is
    /// skipped when there is nothing to draw.
    fn note_settle(&mut self, index: usize, bytes: usize, more: bool) {
        if bytes == 0 {
            return;
        }
        self.entries[index].core.refresh_tty_mode();
        if more {
            return;
        }
        if self.entries[index].core.refresh_process() && self.shows_bar() {
            self.bar_dirty = true;
            self.rebuild_bar();
        }
    }

    fn rebuild_bar(&mut self) {
        if self.entries.is_empty() {
            self.bar_slots.clear();
            return;
        }
        let cols = self.active().dimensions().0;
        // A tab's label (its cwd, and any path prefix) is derived, not stored ready
        // to lend, so gather the owned strings first and let the borrowed
        // `TabLabel`s point into them.
        let titles: Vec<String> = self
            .entries
            .iter()
            .map(|entry| entry.core.tab_label(&self.cfg))
            .collect();
        let labels: Vec<_> = titles
            .iter()
            .enumerate()
            .map(|(index, title)| TabLabel {
                title,
                active: index == self.active,
            })
            .collect();
        // Blocks are sized in terminal cells. Before the first resize there is no
        // geometry; the resulting slots are not painted until that resize rebuilds
        // them, so a placeholder cell width is harmless (the labels, fitted at paint
        // time, never see it).
        let cell_w = self.bar_geom.as_ref().map_or(1, |geom| geom.metrics.w);
        self.bar_slots = tab_bar::layout(cols, &labels, &self.cfg, cell_w);
    }

    /// Translate each core's outbox according to foreground/background routing.
    fn route_core_outboxes(&mut self) {
        let mut title_changed = false;
        for index in 0..self.entries.len() {
            for message in self.entries[index].core.drain_outbox() {
                match message {
                    ToWindow::Title(title) if index == self.active => {
                        title_changed = true;
                        self.outbox.push(ToWindow::Title(title));
                    }
                    ToWindow::Title(_) => {
                        title_changed = true;
                        self.bar_dirty = true;
                    }
                    // Child exit is observed by `pump` as a stream end, not an
                    // outbox message; the manager owns teardown, so a core never
                    // hands us `Closed` to act on here.
                    ToWindow::Closed => {}
                    other => self.outbox.push(other),
                }
            }
        }
        if title_changed {
            self.rebuild_bar();
        }
    }

    /// Retry every child retained after a close, dropping handles that are reaped.
    fn sweep_zombies(&mut self) {
        self.reaping.retain(|child| !child.try_reap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{Key, Mods};
    use crate::pty::{Launch, PollSet, Target};
    use crate::render::display::DrawCmd;

    const METRICS: CellMetrics = CellMetrics {
        size: 16,
        w: 8,
        h: 16,
        baseline: 12,
        ascent: 12,
        descent: 4,
        lock_glyph: true,
    };

    fn core(demo: bool, cols: usize, rows: usize) -> TerminalCore {
        TerminalCore::new(
            demo,
            cols,
            rows,
            METRICS,
            (cols as i32 * METRICS.w) as u32,
            (rows as i32 * METRICS.h) as u32,
            0,
        )
    }

    fn demo_tabs(count: usize) -> Tabs {
        assert!(count > 0);
        let mut tabs = Tabs::new(
            core(true, 80, 24),
            TabBarConfig::default(),
            Rc::new(Theme::default()),
            Launch::new(Vec::new(), Target::Local),
            ShellStartupConfig::default(),
        );
        for _ in 1..count {
            let id = tabs.allocate_id();
            tabs.entries.push(TabEntry {
                id,
                core: core(true, 80, 24),
            });
        }
        tabs
    }

    #[test]
    fn close_repairs_active_index_and_ids_stay_stable() {
        let mut tabs = demo_tabs(3);
        let ids: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();

        assert!(tabs.select(ids[1], true));
        assert_eq!(tabs.active_id(), Some(ids[1]));
        assert!(!tabs.close(ids[1], true));
        assert_eq!(
            tabs.active_id(),
            Some(ids[2]),
            "the right neighbor becomes active"
        );

        assert!(!tabs.close(ids[0], true));
        assert_eq!(
            tabs.active_id(),
            Some(ids[2]),
            "closing left repairs the index without changing identity"
        );
        assert!(tabs.close(ids[2], true));
        assert!(tabs.is_empty());
        assert!(tabs
            .take_outbox()
            .into_iter()
            .any(|message| matches!(message, ToWindow::Closed)));
    }

    /// Drive `tabs` for up to `within` or until its notice is raised, answering whether
    /// one appeared. A real child on a real PTY marks the prompt, so this covers the whole
    /// chain the window relies on: fork → gather → parse → mark → raise.
    fn pump_for_notice(tabs: &mut Tabs, within: Duration) -> bool {
        let stop = Instant::now() + within;
        while Instant::now() < stop {
            let _ = tabs.pump_all(true);
            if tabs.notice().is_some() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }

    /// A core whose child marks one prompt and then stays up, as a shell sitting at its
    /// first prompt does. The sleep bounds it so no stray child can outlive the suite;
    /// hanging up the PTY at drop ends it sooner. `None` where fork/exec is unavailable.
    fn core_marking_a_prompt() -> Option<TerminalCore> {
        let mut core = core(false, 40, 10);
        core.spawn_program(&["/bin/sh", "-c", "printf '\\033]133;A\\007'; sleep 2"])
            .ok()?;
        Some(core)
    }

    #[test]
    fn a_shell_slower_than_the_budget_raises_the_corner_notice() {
        // Budget zero: every startup is over it, so this asserts the wiring rather than
        // the threshold (which `notice` tests directly). Without the `pump_all` hook the
        // measurement would be taken and never reach the window.
        let Some(core) = core_marking_a_prompt() else {
            eprintln!("fork/exec unavailable; skipping the slow-startup notice test");
            return;
        };
        let cfg = ShellStartupConfig {
            warn_after: Some(Duration::ZERO),
            ..ShellStartupConfig::default()
        };
        let mut tabs = Tabs::new(
            core,
            TabBarConfig::default(),
            Rc::new(Theme::default()),
            Launch::new(Vec::new(), Target::Local),
            cfg,
        );
        assert!(
            pump_for_notice(&mut tabs, Duration::from_secs(5)),
            "a shell over the budget never raised its notice"
        );
        // Raising it dirties the visible frame, or the panel would wait for the next
        // unrelated repaint to appear.
        assert!(tabs.active().dirty, "the frame was marked for a rebuild");
    }

    #[test]
    fn a_disabled_budget_stays_silent_through_the_same_pump() {
        // The opt-out, driven through the identical path: same child, same marks, no
        // notice. A user who pays a startup cost on purpose is never told about it.
        let Some(core) = core_marking_a_prompt() else {
            eprintln!("fork/exec unavailable; skipping the disabled-budget test");
            return;
        };
        let cfg = ShellStartupConfig {
            warn_after: None,
            ..ShellStartupConfig::default()
        };
        let mut tabs = Tabs::new(
            core,
            TabBarConfig::default(),
            Rc::new(Theme::default()),
            Launch::new(Vec::new(), Target::Local),
            cfg,
        );
        // A window far longer than the test above needs to see its notice, driving the
        // identical child through the identical path: silence here is the opt-out
        // working, not the prompt failing to arrive.
        assert!(
            !pump_for_notice(&mut tabs, Duration::from_millis(1_500)),
            "a disabled budget raised a notice anyway"
        );
    }

    /// Typing `exit` in the only tab ends its child, and the pump is where that is
    /// noticed. The pump must reap the tab and leave the list empty, because the event
    /// loop reads exactly that to end the turn before it sizes, shapes, or paints a
    /// window with no terminal left in it. A real child on a real PTY, no mocks; skipped
    /// where fork/exec is unavailable so it can never flake.
    #[test]
    fn the_last_child_exiting_empties_the_tabs_through_the_pump() {
        let mut core = core(false, 80, 24);
        // `/bin/true` by name, not through `$SHELL`: this is the one test in the suite
        // that wants a child which exits *immediately*, and routing that through a
        // process-global let any concurrently-spawning test substitute a `cat` that
        // never exits — whereupon this waited out its five-second deadline and failed.
        if core.spawn_program(&["/bin/true"]).is_err() {
            eprintln!("fork/exec unavailable here; skipping the child-exit pump test");
            return;
        }
        let mut tabs = Tabs::new(
            core,
            TabBarConfig::default(),
            Rc::new(Theme::default()),
            Launch::new(Vec::new(), Target::Local),
            ShellStartupConfig::default(),
        );
        assert!(!tabs.is_empty());

        // The child exits at once, but the gather thread still has to see the EOF, so
        // pump until it lands rather than assuming a single turn catches it.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !tabs.is_empty() && Instant::now() < deadline {
            tabs.pump_all(true).expect("pump");
        }

        assert!(
            tabs.is_empty(),
            "the exited child's tab was never reaped by the pump"
        );
        assert!(
            tabs.take_outbox()
                .into_iter()
                .any(|message| matches!(message, ToWindow::Closed)),
            "emptying the tab list must signal the window closed"
        );
        // The loop calls these on its way out; none of them may touch the missing tab.
        assert!(!tabs.needs_frame());
        assert!(!tabs.shows_bar());
        assert_eq!(tabs.active_id(), None);
        assert!(tabs.gather_fds().next().is_none());
    }

    #[test]
    fn next_and_previous_wrap() {
        let mut tabs = demo_tabs(3);
        let ids: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();

        assert!(tabs.prev(false));
        assert_eq!(tabs.active_id(), Some(ids[2]));
        assert!(tabs.next(false));
        assert_eq!(tabs.active_id(), Some(ids[0]));
    }

    #[test]
    fn reorder_moves_the_active_tab_and_does_not_wrap() {
        let mut tabs = demo_tabs(3);
        let ids: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();
        let order = |tabs: &Tabs| tabs.entries.iter().map(|e| e.id).collect::<Vec<_>>();

        // Active is the last tab (open makes the newest active; demo pushes tab 0
        // as active, so start by selecting the middle to reorder from the interior).
        assert!(tabs.select(ids[1], false));
        assert!(tabs.move_active(Reorder::Next));
        assert_eq!(order(&tabs), vec![ids[0], ids[2], ids[1]]);
        assert_eq!(tabs.active_id(), Some(ids[1]), "identity follows the move");

        assert!(tabs.move_active(Reorder::Prev));
        assert_eq!(order(&tabs), vec![ids[0], ids[1], ids[2]]);

        // At an edge the move is a no-op (no wrap), unlike selection.
        assert!(tabs.select(ids[0], false));
        assert!(!tabs.move_active(Reorder::Prev));
        assert_eq!(order(&tabs), vec![ids[0], ids[1], ids[2]]);
        assert!(tabs.select(ids[2], false));
        assert!(!tabs.move_active(Reorder::Next));
        assert_eq!(order(&tabs), vec![ids[0], ids[1], ids[2]]);

        // A single tab cannot reorder.
        let mut one = demo_tabs(1);
        assert!(!one.move_active(Reorder::Next));
    }

    #[test]
    fn move_active_to_clamps_and_keeps_the_tab_active() {
        let mut tabs = demo_tabs(3);
        let ids: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();
        let order = |tabs: &Tabs| tabs.entries.iter().map(|e| e.id).collect::<Vec<_>>();

        // Active starts at tab 0. Move it to the last slot; the others slide left and
        // the identity follows the move.
        assert!(tabs.move_active_to(2));
        assert_eq!(order(&tabs), vec![ids[1], ids[2], ids[0]]);
        assert_eq!(tabs.active_id(), Some(ids[0]), "identity follows the move");

        // Landing on the current slot is a no-op.
        assert!(!tabs.move_active_to(2));

        // Back to the front, order fully restored.
        assert!(tabs.move_active_to(0));
        assert_eq!(order(&tabs), vec![ids[0], ids[1], ids[2]]);

        // An out-of-range index clamps to the last slot rather than trapping.
        assert!(tabs.move_active_to(99));
        assert_eq!(order(&tabs), vec![ids[1], ids[2], ids[0]]);
        assert_eq!(tabs.active, 2);

        // A single tab has nowhere to move, at any index.
        let mut one = demo_tabs(1);
        assert!(!one.move_active_to(0));
        assert!(!one.move_active_to(5));
    }

    /// A multi-tab manager with the strip laid out at the test metrics, so drags have
    /// real device-pixel geometry to work against. `count` equal blocks fill `cols`.
    fn dragging_tabs(count: usize, cols: usize) -> Tabs {
        let mut tabs = demo_tabs(count);
        tabs.resize_all(
            cols,
            10,
            (cols as i32 * METRICS.w) as u32,
            176,
            METRICS,
            METRICS,
            0,
            16,
            0,
            16,
            Scale::ONE,
        )
        .expect("build the bar");
        tabs
    }

    #[test]
    fn a_drag_past_a_block_boundary_reorders_and_the_identity_follows() {
        let mut tabs = dragging_tabs(3, 30); // three equal 10-cell blocks
        let ids: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();
        let pitch = 10 * METRICS.w;
        assert_eq!(tabs.active_id(), Some(ids[0]), "demo tab 0 starts active");

        // Grab tab 0 at its left edge and drag it fully to the right; its own centre
        // lands in the last slot.
        tabs.begin_drag(ids[0], 0);
        tabs.drag_to(2 * pitch + 5);

        assert_eq!(
            tabs.active_id(),
            Some(ids[0]),
            "the dragged tab stays active"
        );
        let order: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();
        assert_eq!(
            order,
            vec![ids[1], ids[2], ids[0]],
            "it reordered to the end"
        );
        assert_eq!(tabs.active, 2);
        assert!(tabs.bar_slots[2].active, "its label is now the last block");

        tabs.end_drag();
        assert!(!tabs.dragging(), "the drag is done");
    }

    #[test]
    fn a_press_that_never_moves_selects_without_reordering() {
        let mut tabs = dragging_tabs(3, 30);
        let ids: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();

        // The pointer path selects on press, then arms the drag; a plain click never
        // calls drag_to, so nothing lifts and nothing reorders.
        assert!(tabs.select(ids[1], false));
        tabs.begin_drag(ids[1], 12 * METRICS.w); // inside block 1
        tabs.end_drag();

        assert_eq!(tabs.active_id(), Some(ids[1]), "the press still selected");
        let order: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();
        assert_eq!(order, vec![ids[0], ids[1], ids[2]], "and never reordered");
    }

    #[test]
    fn a_drag_shorter_than_the_threshold_does_not_lift() {
        let mut tabs = dragging_tabs(2, 20);
        let ids: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();

        tabs.begin_drag(ids[0], 0);
        tabs.bar_dirty = false; // a clean slate to prove the nudge dirties nothing
        tabs.drag_to(1); // one device pixel, below the half-cell (4 px) threshold

        assert!(
            !tabs.bar_dirty,
            "a sub-threshold nudge lifts nothing and repaints nothing"
        );
        let order: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();
        assert_eq!(order, vec![ids[0], ids[1]], "and reorders nothing");
        assert_eq!(tabs.active_id(), Some(ids[0]));
    }

    #[test]
    fn closing_the_dragged_tab_cancels_the_drag() {
        let mut tabs = dragging_tabs(3, 30);
        let ids: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();

        tabs.begin_drag(ids[0], 0);
        tabs.drag_to(2 * 10 * METRICS.w); // lift and drag it to the end
        assert!(tabs.dragging());

        // Its shell exits mid-drag: the drag cannot outlive its own tab.
        tabs.close(ids[0], false);
        assert!(!tabs.dragging(), "the drag is cancelled with its tab");
    }

    #[test]
    fn a_background_tab_closing_mid_drag_keeps_the_drag_on_its_own_tab() {
        let mut tabs = dragging_tabs(3, 30);
        let ids: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();

        // Drag tab 0; it lifts but stays in slot 0 for now.
        tabs.begin_drag(ids[0], 0);
        tabs.drag_to(5);
        assert!(tabs.dragging());
        assert_eq!(tabs.active_id(), Some(ids[0]));

        // A *background* tab's shell exits: every index shifts and the bar re-lays out
        // at a new pitch, but the drag holds an id, so it survives on tab 0.
        tabs.close(ids[2], false);
        assert!(
            tabs.dragging(),
            "a background close does not cancel the drag"
        );
        assert_eq!(tabs.active_id(), Some(ids[0]));

        // It continues against the new two-tab run and still moves tab 0.
        tabs.drag_to(10_000); // pinned to the far right of the new run
        assert_eq!(tabs.active_id(), Some(ids[0]));
        let order: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();
        assert_eq!(order, vec![ids[1], ids[0]]);
    }

    #[test]
    fn background_title_is_swallowed_until_selection() {
        let mut tabs = demo_tabs(2);
        let background = tabs.entries[1].id;
        tabs.entries[1]
            .core
            .feed_test_bytes(b"\x1b]2;background job\x07");

        let routed = tabs.take_outbox();
        assert!(!routed
            .iter()
            .any(|message| matches!(message, ToWindow::Title(_))));
        assert!(tabs.bar_dirty);

        assert!(tabs.select(background, false));
        assert!(tabs
            .take_outbox()
            .into_iter()
            .any(|message| matches!(message, ToWindow::Title(title) if title == "background job")));
    }

    #[test]
    fn resize_updates_every_core() {
        let mut tabs = demo_tabs(3);
        tabs.resize_all(
            100,
            30,
            800,
            480,
            METRICS,
            METRICS,
            0,
            16,
            0,
            16,
            Scale::ONE,
        )
        .expect("resize every demo core");
        assert!(tabs
            .entries
            .iter()
            .all(|entry| entry.core.dimensions() == (100, 30)));
        assert!(tabs.entries.iter().all(|entry| entry.core.origin_y() == 16));
    }

    #[test]
    fn bar_hit_testing_maps_to_stable_ids() {
        let mut tabs = demo_tabs(2);
        let ids: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();
        tabs.resize_all(20, 10, 160, 176, METRICS, METRICS, 0, 16, 0, 16, Scale::ONE)
            .expect("build bar layout");

        // Two tabs in 20 cols land at the floor width of 10 each: 0..10, 10..20.
        assert_eq!(tabs.tab_at_bar_col(2), Some(ids[0]));
        assert_eq!(tabs.tab_at_bar_col(12), Some(ids[1]));
        assert!(tabs.select(ids[1], false));
        assert_eq!(tabs.active_id(), Some(ids[1]));
        assert!(!tabs.close(ids[0], false));
        assert_eq!(tabs.active_id(), Some(ids[1]));
    }

    #[test]
    fn visible_bar_shifts_grid_down_exactly_one_cell() {
        let fonts = Fonts::new(&[METRICS.size]).expect("fonts");
        let one = demo_tabs(1);
        let mut one_list = Vec::new();
        let mut one_strings = Vec::new();
        one.fill_frame_list(&mut one_list, &mut one_strings, &fonts);
        let one_baseline = one_list.iter().find_map(|cmd| match cmd {
            DrawCmd::Cells { baseline, .. } => Some(*baseline),
            _ => None,
        });

        // A top-anchored one-cell strip: grid origin drops one cell, strip at y 0.
        let mut two = demo_tabs(2);
        two.resize_all(
            80,
            23,
            640,
            384,
            METRICS,
            METRICS,
            0,
            METRICS.h,
            0,
            METRICS.h,
            Scale::ONE,
        )
        .expect("shift both grids below the bar");
        let mut two_list = Vec::new();
        let mut two_strings = Vec::new();
        two.fill_frame_list(&mut two_list, &mut two_strings, &fonts);
        let two_baseline = two_list.iter().find_map(|cmd| match cmd {
            DrawCmd::Cells { baseline, .. } => Some(*baseline),
            _ => None,
        });

        assert_eq!(one_baseline, Some(METRICS.ascent));
        assert_eq!(two_baseline, Some(METRICS.h + METRICS.ascent));
        // The tab strip paints its label (a proportional `Text` run) inside the top
        // row, above the shifted-down grid whose runs baseline at METRICS.h + ascent.
        assert!(
            two_list.iter().any(|cmd| {
                matches!(cmd, DrawCmd::Text { baseline, .. } if *baseline < METRICS.h)
            }),
            "the visible strip paints a label above the grid"
        );
    }

    #[test]
    fn every_tab_label_renders_without_a_recycled_prefix() {
        // Regression: a blank grid run returned its space-filled scratch buffer to
        // the pool uncleared, and the bar's next run appended its label to those
        // stale spaces, shoving the inactive tab's label off screen. Build the bar
        // over a populated grid (which produces blank runs to recycle) and assert
        // every label lands as its own clean run, never with recycled bytes glued in
        // front. The label is the proportional interface font, so it is a `Text` run.
        let fonts = Fonts::new(&[METRICS.size]).expect("fonts");
        let mut two = demo_tabs(2);
        let ids: Vec<_> = two.entries.iter().map(|entry| entry.id).collect();
        two.resize_all(
            80,
            23,
            640,
            384,
            METRICS,
            METRICS,
            0,
            METRICS.h,
            0,
            METRICS.h,
            Scale::ONE,
        )
        .expect("build the bar");
        // The runtime path makes the newly opened tab active and tab zero inactive.
        assert!(two.select(ids[1], false));

        let mut list = Vec::new();
        let mut strings = Vec::new();
        two.fill_frame_list(&mut list, &mut strings, &fonts);

        // Both no-PTY demo cores label their tab with the directory fallback "shell".
        // Each lands as its own clean `Text` run: exactly "shell", never " shell" or
        // a recycled prefix. (The demo grid content carries no such string, so a
        // match is unambiguously a bar label.)
        let labels: Vec<&str> = list
            .iter()
            .filter_map(|cmd| match cmd {
                DrawCmd::Text { text, .. } if text.contains("shell") => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            labels,
            vec!["shell", "shell"],
            "each tab paints a clean \"shell\", got {labels:?}"
        );
    }

    #[test]
    fn two_live_pty_streams_land_on_their_own_grids() {
        let mut first = core(false, 40, 10);
        if first.spawn_program(&["/bin/cat"]).is_err() {
            eprintln!("fork/exec unavailable; skipping multi-tab PTY test");
            return;
        }
        let mut tabs = Tabs::new(
            first,
            TabBarConfig::default(),
            Rc::new(Theme::default()),
            Launch::new(vec!["/bin/cat".to_string()], Target::Local),
            ShellStartupConfig::default(),
        );
        if tabs.open(40, 10, METRICS, 320, 160, 0, false).is_err() {
            eprintln!("second PTY unavailable; skipping multi-tab PTY test");
            return;
        }

        tabs.entries[0]
            .core
            .apply(ToTerminal::Key {
                key: Key::plain('a'),
                mods: Mods::NONE,
                event: crate::input::KeyEvent::Press,
            })
            .expect("write tab A");
        tabs.entries[1]
            .core
            .apply(ToTerminal::Key {
                key: Key::plain('b'),
                mods: Mods::NONE,
                event: crate::input::KeyEvent::Press,
            })
            .expect("write tab B");

        let mut poll_set = PollSet::new();
        let mut landed = false;
        for _ in 0..100 {
            poll_set.clear();
            for fd in tabs.gather_fds() {
                poll_set.add(fd);
            }
            poll_set
                .wait(Some(Duration::from_millis(20)))
                .expect("wait for either gatherer");
            tabs.pump_all(false).expect("pump both tabs");
            landed = tabs.entries[0].core.row_string(0).contains('a')
                && tabs.entries[1].core.row_string(0).contains('b');
            if landed {
                break;
            }
        }
        assert!(landed, "each child's echo reaches only its own grid");

        let ids: Vec<_> = tabs.entries.iter().map(|entry| entry.id).collect();
        for id in ids {
            tabs.close(id, false);
        }
        for _ in 0..100 {
            tabs.sweep_zombies();
            if tabs.reaping.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(tabs.reaping.is_empty(), "test children were reaped");
    }
}
