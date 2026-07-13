//! The overlay scrollbar: its thumb geometry, and the fade/expand state a pointer
//! drives. Kept free of any terminal, so the widget is only ever about scrolling.
//!
//! A scrollable view shows the band `[scroll, scroll + viewport)` of a `content` longer
//! than it, with `scroll` always in `0..=max_scroll`.
//!
//! **The content unit is the caller's.** Every function here is built from *ratios* of
//! `content`, `viewport`, and `scroll` (what fraction of the content is on screen, how
//! far through it we are), so the unit cancels: it only ever scales a track's pixel
//! height. A pixel-scrolled view counts pixels; bnkterm counts **lines**, because a
//! terminal cannot scroll half a row. The one rule is that all three share a unit.
//!
//! bnkterm's mapping, for the primary screen (the alt screen keeps no history, so it is
//! never scrollable and the bar stays hidden there):
//!
//! ```text
//!   content  = scrollback_len + rows       scroll = scrollback_len - view_offset
//!   viewport = rows                        max    = scrollback_len
//! ```
//!
//! which re-reads `view_offset` (lines *above* the live bottom, so 0 is the bottom) as a
//! scroll *down* from the oldest line, which is the direction a thumb travels.

use std::time::{Duration, Instant};

use crate::platform::geom::Rect;

/// The largest valid scroll: 0 when the whole content fits the viewport.
pub fn max_scroll(content: i32, viewport: i32) -> i32 {
    (content - viewport).max(0)
}

/// Whether the content overflows its viewport at all, which is the only case a scrollbar
/// is drawn or grabbable.
pub fn scrollable(content: i32, viewport: i32) -> bool {
    max_scroll(content, viewport) > 0
}

/// The thumb's rectangle inside the bar's `track`, or `None` when the content fits and
/// there is nothing to show. The thumb's length is the visible fraction of the content,
/// floored at `min_len` so a deep scrollback still leaves something to grab, and its
/// travel spans the track exactly: at `max_scroll` its bottom meets the track's bottom.
pub fn thumb(track: Rect, viewport: i32, content: i32, scroll: i32, min_len: i32) -> Option<Rect> {
    let max = max_scroll(content, viewport);
    if max <= 0 || track.h <= 0 || content <= 0 {
        return None;
    }
    let visible = track.h as i64 * viewport as i64 / content as i64;
    let len = (visible as i32).clamp(min_len.min(track.h), track.h);
    let travel = (track.h - len) as i64;
    let y = track.y + (travel * scroll.clamp(0, max) as i64 / max as i64) as i32;
    Some(Rect { y, h: len, ..track })
}

/// The scroll offset that places a thumb of height `thumb_h` with its top at window y
/// `top`: the inverse of [`thumb`]'s placement, clamped to the content. A track with no
/// travel (the thumb fills it) pins the scroll at 0.
pub fn scroll_at_thumb(track: Rect, thumb_h: i32, viewport: i32, content: i32, top: i32) -> i32 {
    let max = max_scroll(content, viewport);
    let travel = track.h - thumb_h;
    if max <= 0 || travel <= 0 {
        return 0;
    }
    let offset = (top - track.y).clamp(0, travel) as i64;
    (offset * max as i64 / travel as i64) as i32
}

/// How long the bar stays lit after the scroll that woke it, before it fades.
const LIT: Duration = Duration::from_millis(900);
/// How long the bar takes to fade in, to fade back out, and to grow from the thin
/// resting indicator to the grabbable slider.
const FADE_IN: Duration = Duration::from_millis(90);
const FADE_OUT: Duration = Duration::from_millis(280);
const EXPAND: Duration = Duration::from_millis(120);
/// A stall longer than this counts as one bounded step, so a long idle cannot make the
/// bar jump most of the way in a single frame.
const STEP_CAP: Duration = Duration::from_millis(50);
/// While the bar animates, the loop retries at least this often, so a frame whose
/// quantised colours did not change (and so presented nothing and armed no compositor
/// callback) still gets a follow-up tick.
const RETRY: Duration = Duration::from_millis(16);

/// Move `cur` toward `target` by one `dt` step of a `dur`-long transition, landing
/// exactly on the target rather than crawling asymptotically toward it.
fn approach(cur: f32, target: f32, dt: Duration, dur: Duration) -> f32 {
    let step = dt.as_secs_f32() / dur.as_secs_f32();
    let gap = target - cur;
    if gap.abs() <= step {
        return target;
    }
    cur + step * gap.signum()
}

/// An overlay scrollbar's live appearance: how lit it is (0 invisible, 1 full) and how
/// far it has grown from the thin resting indicator toward the grabbable slider (0 thin,
/// 1 wide). The bar carries no geometry; the painter derives that from the track and the
/// content each frame.
///
/// Every scroll lights it for [`LIT`] and then it fades away, so the bar confirms a
/// scroll and gets out of the way. The pointer coming near it holds it lit and grows it
/// into something worth grabbing, and a drag pins it there for the duration.
///
/// ```text
///   hidden ──scroll──▶ indicator ──pointer near──▶ slider ──press──▶ dragging
///     ▲                    │                          │                  │
///     └── fades once LIT lapses and the pointer is away ──────────────────┘
/// ```
pub struct Scrollbar {
    lit: f32,
    wide: f32,
    /// Whether the pointer sits in the bar's proximity zone.
    near: bool,
    /// While a drag holds the thumb: where inside it the pointer grabbed, so the thumb
    /// tracks the pointer without snapping its top to it.
    grab: Option<i32>,
    /// When the light from the last scroll expires; `None` once it has.
    lit_until: Option<Instant>,
    last_tick: Option<Instant>,
    animating: bool,
}

impl Default for Scrollbar {
    fn default() -> Self {
        Self::hidden()
    }
}

impl Scrollbar {
    /// A bar out of sight: what one starts as, and what it returns to.
    pub const fn hidden() -> Self {
        Self {
            lit: 0.0,
            wide: 0.0,
            near: false,
            grab: None,
            lit_until: None,
            last_tick: None,
            animating: false,
        }
    }

    /// A scroll happened: light the bar and hold it for [`LIT`].
    pub fn flash(&mut self, now: Instant) {
        self.lit_until = Some(now + LIT);
    }

    /// Whether the pointer is in the bar's proximity zone, which holds it lit and
    /// expanded. Returns whether that changed.
    pub fn set_near(&mut self, near: bool) -> bool {
        let changed = self.near != near;
        self.near = near;
        changed
    }

    /// Take the thumb under the pointer, `grab` pixels below the thumb's top.
    pub fn press(&mut self, grab: i32) {
        self.grab = Some(grab);
    }

    /// Let go of the thumb.
    pub fn release(&mut self) {
        self.grab = None;
    }

    /// Where inside the thumb a live drag grabbed it, or `None` when none is.
    pub fn grab(&self) -> Option<i32> {
        self.grab
    }

    /// Drop the bar out of sight at once: the view stopped being scrollable, so a lit or
    /// half-faded bar has nothing left to point at.
    pub fn hide(&mut self) {
        *self = Self::default();
    }

    /// Advance the fade and the expansion one frame, at `now`. Returns whether the
    /// appearance changed, so the caller can mark a repaint.
    pub fn tick(&mut self, now: Instant) -> bool {
        if self.lit_until.is_some_and(|t| now >= t) {
            self.lit_until = None;
        }
        let dt = self
            .last_tick
            .map_or(Duration::ZERO, |t| now.saturating_duration_since(t))
            .min(STEP_CAP);
        self.last_tick = Some(now);

        let expanded = self.near || self.grab.is_some();
        let held = expanded || self.lit_until.is_some();
        let (lit, wide) = (
            if held { 1.0 } else { 0.0 },
            if expanded { 1.0 } else { 0.0 },
        );
        let was = (self.lit, self.wide);
        self.lit = approach(self.lit, lit, dt, if held { FADE_IN } else { FADE_OUT });
        self.wide = approach(self.wide, wide, dt, EXPAND);
        self.animating = self.lit != lit || self.wide != wide;
        was != (self.lit, self.wide)
    }

    /// How lit the bar is right now: 0 hides it entirely, 1 is full strength.
    pub fn lit(&self) -> f32 {
        self.lit
    }

    /// How far the bar has grown toward the grabbable slider: 0 is the thin resting
    /// indicator, 1 the full-width one.
    pub fn wide(&self) -> f32 {
        self.wide
    }

    /// When the loop should next tick this bar: mid-transition, a short retry; while it
    /// holds lit, the moment that light expires and the fade begins. `None` when it is
    /// settled (hidden, or held open under the pointer).
    pub fn retry_at(&self) -> Option<Instant> {
        if self.animating {
            return self.last_tick.map(|t| t + RETRY);
        }
        self.lit_until
    }

    /// Whether the bar is mid-transition, so the next compositor frame should carry it
    /// further.
    pub fn animating(&self) -> bool {
        self.animating
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_scroll_is_zero_when_content_fits() {
        assert_eq!(max_scroll(300, 600), 0);
        assert_eq!(max_scroll(600, 600), 0);
        assert_eq!(max_scroll(900, 600), 300);
    }

    #[test]
    fn scrollable_only_when_the_content_overflows() {
        // A screen with no history behind it: nothing to scroll, so no bar.
        assert!(!scrollable(24, 24));
        assert!(scrollable(25, 24), "one line of history is enough");
    }

    /// A 600px track down a 600-unit viewport onto 2400 units of content: the thumb is a
    /// quarter of the track, so 150px long with 450px of travel.
    const TRACK: Rect = Rect {
        x: 10,
        y: 0,
        w: 8,
        h: 600,
    };

    #[test]
    fn no_thumb_when_the_content_fits() {
        assert_eq!(thumb(TRACK, 600, 400, 0, 30), None);
        assert_eq!(thumb(TRACK, 600, 600, 0, 30), None);
    }

    #[test]
    fn the_thumb_is_the_visible_fraction_and_spans_the_track() {
        let top = thumb(TRACK, 600, 2400, 0, 30).expect("scrollable");
        assert_eq!(
            (top.x, top.w),
            (TRACK.x, TRACK.w),
            "the thumb fills the lane"
        );
        assert_eq!(
            (top.y, top.h),
            (0, 150),
            "a quarter of the track, at the top"
        );
        let bottom = thumb(TRACK, 600, 2400, max_scroll(2400, 600), 30).expect("scrollable");
        assert_eq!(
            (bottom.y, bottom.h),
            (450, 150),
            "at max scroll the thumb's bottom meets the track's"
        );
        let mid = thumb(TRACK, 600, 2400, 900, 30).expect("scrollable");
        assert_eq!(mid.y, 225, "half the travel at half the scroll");
    }

    #[test]
    fn a_deep_scrollback_keeps_a_grabbable_thumb() {
        // 1% of the track would be 6px; the floor holds it at 30.
        let t = thumb(TRACK, 600, 60_000, 0, 30).expect("scrollable");
        assert_eq!(t.h, 30);
        // And it still reaches the bottom of the track at max scroll.
        let end = thumb(TRACK, 600, 60_000, max_scroll(60_000, 600), 30).expect("scrollable");
        assert_eq!(end.y + end.h, TRACK.y + TRACK.h);
    }

    /// The unit the caller counts content in cancels out, so a terminal's line counts
    /// place the thumb exactly where the equivalent pixel counts would. This is what lets
    /// the same widget serve a pixel-scrolled editor and a line-scrolled grid.
    #[test]
    fn the_content_unit_cancels_so_lines_place_a_thumb_like_pixels() {
        let (rows, history, row_px) = (24, 96, 16);
        let lines = thumb(TRACK, rows, rows + history, 0, 30).expect("scrollable");
        let pixels =
            thumb(TRACK, rows * row_px, (rows + history) * row_px, 0, 30).expect("scrollable");
        assert_eq!(lines, pixels);
        // At the live bottom (max scroll = the whole history), the thumb's bottom meets
        // the track's, counting in lines just as it does in pixels.
        let bottom = thumb(TRACK, rows, rows + history, history, 30).expect("scrollable");
        assert_eq!(bottom.y + bottom.h, TRACK.y + TRACK.h);
    }

    #[test]
    fn dragging_the_thumb_inverts_its_placement() {
        // Every scroll offset maps to a thumb top that maps back to it (within the
        // rounding of one thumb pixel, which is 4 content units here).
        for scroll in (0..=1800).step_by(97) {
            let t = thumb(TRACK, 600, 2400, scroll, 30).expect("scrollable");
            let back = scroll_at_thumb(TRACK, t.h, 600, 2400, t.y);
            assert!(
                (back - scroll).abs() <= 4,
                "thumb at {} mapped back to {back}, not {scroll}",
                t.y
            );
        }
    }

    #[test]
    fn dragging_past_the_track_ends_clamps_to_the_content() {
        assert_eq!(scroll_at_thumb(TRACK, 150, 600, 2400, -200), 0);
        assert_eq!(
            scroll_at_thumb(TRACK, 150, 600, 2400, 5_000),
            max_scroll(2400, 600)
        );
    }

    /// The bar's appearance after `ms` of ticking on from `start`.
    fn advance(bar: &mut Scrollbar, start: Instant, ms: u64) -> (f32, f32) {
        // Step in 16ms frames, the pace the compositor drives.
        for f in 1..=(ms / 16) {
            bar.tick(start + Duration::from_millis(f * 16));
        }
        (bar.lit(), bar.wide())
    }

    #[test]
    fn a_scroll_lights_the_bar_then_it_fades_away() {
        let start = Instant::now();
        let mut bar = Scrollbar::default();
        bar.tick(start);
        assert_eq!(bar.lit(), 0.0, "hidden at rest");
        bar.flash(start);
        let (lit, wide) = advance(&mut bar, start, 100);
        assert_eq!(lit, 1.0, "the fade-in completes");
        assert_eq!(
            wide, 0.0,
            "but stays the thin indicator, with no pointer near"
        );
        // Still lit while the hold lasts, gone once it lapses and the fade runs.
        let (lit, _) = advance(&mut bar, start, 800);
        assert_eq!(lit, 1.0, "held lit for the hold window");
        let (lit, _) = advance(&mut bar, start, 1_300);
        assert_eq!(lit, 0.0, "and faded out after it");
        assert!(!bar.animating(), "settled, so the loop can go idle");
        assert_eq!(bar.retry_at(), None);
    }

    #[test]
    fn the_pointer_holds_the_bar_open_and_widens_it() {
        let start = Instant::now();
        let mut bar = Scrollbar::default();
        bar.set_near(true);
        let (lit, wide) = advance(&mut bar, start, 200);
        assert_eq!((lit, wide), (1.0, 1.0), "lit and grown to the full slider");
        // It stays: a bar under the pointer never fades from under it.
        let (lit, wide) = advance(&mut bar, start, 3_000);
        assert_eq!((lit, wide), (1.0, 1.0));
        assert_eq!(bar.retry_at(), None, "held open costs the loop nothing");
        // The pointer leaves: it shrinks back and fades out.
        bar.set_near(false);
        let (lit, wide) = advance(&mut bar, start + Duration::from_millis(3_000), 400);
        assert_eq!((lit, wide), (0.0, 0.0));
    }

    #[test]
    fn a_drag_holds_the_bar_open_wherever_the_pointer_goes() {
        let start = Instant::now();
        let mut bar = Scrollbar::default();
        bar.set_near(true);
        advance(&mut bar, start, 200);
        // The press grabs the thumb, then the pointer wanders off the lane; the bar must
        // stay lit and wide until the button comes back up.
        bar.press(12);
        assert_eq!(bar.grab(), Some(12));
        bar.set_near(false);
        let (lit, wide) = advance(&mut bar, start, 1_500);
        assert_eq!((lit, wide), (1.0, 1.0), "the drag pins it open");
        bar.release();
        let (lit, wide) = advance(&mut bar, start + Duration::from_millis(1_500), 400);
        assert_eq!((lit, wide), (0.0, 0.0), "and it goes once the drag ends");
    }

    #[test]
    fn a_bar_that_can_no_longer_scroll_disappears_at_once() {
        let start = Instant::now();
        let mut bar = Scrollbar::default();
        bar.flash(start);
        advance(&mut bar, start, 100);
        bar.hide();
        assert_eq!((bar.lit(), bar.wide()), (0.0, 0.0));
        assert_eq!(bar.retry_at(), None);
    }
}
