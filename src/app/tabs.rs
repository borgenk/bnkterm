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
use std::time::{Duration, Instant};

use super::message::{ToTerminal, ToWindow};
use super::terminal::{PumpOutcome, TerminalCore};
use crate::config::TabBarConfig;
use crate::error::Result;
use crate::gather::GatherEnd;
use crate::platform::freetype::Fonts;
use crate::pty::ZombieChild;
use crate::render::display::DisplayList;
use crate::tab_bar::{self, BarGeom, Slot, TabLabel};
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
    /// Strip appearance and layout, the source of truth for `layout`/`fill_bar`.
    cfg: TabBarConfig,
    /// Messages already translated from per-core facts into window actions.
    outbox: Vec<ToWindow>,
}

impl Tabs {
    /// Wrap the initial terminal core as tab zero, under the given strip config.
    pub(super) fn new(core: TerminalCore, cfg: TabBarConfig) -> Self {
        let mut tabs = Self {
            entries: Vec::new(),
            active: 0,
            next_id: 0,
            pump_cursor: 0,
            reaping: Vec::new(),
            bar_dirty: false,
            bar_slots: Vec::new(),
            bar_geom: None,
            cfg,
            outbox: Vec::new(),
        };
        let id = tabs.allocate_id();
        tabs.entries.push(TabEntry { id, core });
        tabs
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
        if let Err(error) = core.spawn_shell() {
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
        self.entries.swap(self.active, target);
        self.active = target;
        self.bar_dirty = true;
        self.rebuild_bar();
        true
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

    /// Drain and route every core's outbound facts, then return the translated
    /// window actions accumulated so far.
    pub(super) fn take_outbox(&mut self) -> Vec<ToWindow> {
        self.route_core_outboxes();
        std::mem::take(&mut self.outbox)
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

    /// Whether the visible terminal has a hyperlink under the pointer, so the window
    /// can offer the hand cursor.
    pub(super) fn hovering_link(&self) -> bool {
        self.entries
            .get(self.active)
            .is_some_and(|entry| entry.core.hovering_link())
    }

    /// Whether the active grid or future tab bar needs a frame.
    pub(super) fn needs_frame(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        self.entries
            .get(self.active)
            .is_some_and(|entry| entry.core.dirty)
            || self.bar_dirty
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
                tab_bar::fill_bar(out, strings, &self.bar_slots, bar, &self.cfg, fonts);
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

    /// After a core drained a burst of output with nothing left pending, its shell
    /// has likely printed a fresh prompt, so its working directory may have moved
    /// (a bare `cd` reports nothing else) or a foreground program may have started
    /// or exited. Re-read both, and when either changed and the bar is visible,
    /// rebuild and repaint it. They are refreshed even for a lone tab so they are
    /// current the moment a second opens; the bar work is skipped while there is
    /// nothing to draw.
    fn note_settle(&mut self, index: usize, bytes: usize, more: bool) {
        if bytes == 0 || more {
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
            let messages = self.entries[index].core.take_outbox();
            for message in messages {
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
    use crate::pty::PollSet;
    use crate::render::display::DrawCmd;

    const METRICS: CellMetrics = CellMetrics {
        size: 16,
        w: 8,
        h: 16,
        baseline: 12,
        ascent: 12,
        descent: 4,
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
        let mut tabs = Tabs::new(core(true, 80, 24), TabBarConfig::default());
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
        tabs.resize_all(100, 30, 800, 480, METRICS, METRICS, 0, 16, 0, 16)
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
        tabs.resize_all(20, 10, 160, 176, METRICS, METRICS, 0, 16, 0, 16)
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
            80, 23, 640, 384, METRICS, METRICS, 0, METRICS.h, 0, METRICS.h,
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
            80, 23, 640, 384, METRICS, METRICS, 0, METRICS.h, 0, METRICS.h,
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
        std::env::set_var("SHELL", "/bin/cat");
        let mut first = core(false, 40, 10);
        if first.spawn_shell().is_err() {
            eprintln!("fork/exec unavailable; skipping multi-tab PTY test");
            return;
        }
        let mut tabs = Tabs::new(first, TabBarConfig::default());
        if tabs.open(40, 10, METRICS, 320, 160, 0, false).is_err() {
            eprintln!("second PTY unavailable; skipping multi-tab PTY test");
            return;
        }

        tabs.entries[0]
            .core
            .apply(ToTerminal::Key {
                key: Key::Char('a'),
                mods: Mods::NONE,
            })
            .expect("write tab A");
        tabs.entries[1]
            .core
            .apply(ToTerminal::Key {
                key: Key::Char('b'),
                mods: Mods::NONE,
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
