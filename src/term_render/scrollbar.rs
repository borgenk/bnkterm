//! The overlay scrollbar down the grid's right edge, and the [`ScrollLane`] geometry it is
//! both drawn in and hit-tested against, so the bar is grabbable exactly where it shows.

use crate::color::Rgb;
use crate::grid::Screen;
use crate::platform::geom::{Rect, Scale};
use crate::platform::scroll;
use crate::render::display::DrawCmd;
use crate::term_render::{CellMetrics, Painter, ScrollLane};

/// The overlay scrollbar down the grid's right edge, in *logical* pixels (they pass
/// through [`Scale::px`], so they come out the same physical size at any DPI).
///
/// The bar rests as a thin indicator and grows to [`SCROLL_WIDE`] when the pointer
/// enters its zone: [`SCROLL_NEAR`] of slack to the left of the lane, so it meets the
/// pointer before the pointer has to aim. The thumb is right-aligned in the lane and
/// grows leftward, staying anchored to the window edge, which is where a maximized
/// window's pointer lands.
///
/// The window padding is thinner than the bar and there is no text margin for the
/// slider to hide in, so the expanded slider laps over the last column. That is the price
/// of an *overlay*: reserving a gutter would cost a column and reflow the child on every
/// appearance. The bar is only that wide while the pointer is on it, and only visible at
/// all while there is history to scroll.
const SCROLL_THIN: i32 = 4;
const SCROLL_WIDE: i32 = 10;
const SCROLL_NEAR: i32 = 24;
/// Gap between the lane and the window's right edge.
const SCROLL_INSET: i32 = 3;
/// Shortest the thumb gets, so even a 100k-line scrollback leaves something to grab.
const SCROLL_MIN_THUMB: i32 = 30;

/// The bar's ink, and the coverages it is tinted onto the theme background at: the
/// resting thumb, the thumb once the pointer has lifted it, and the trough behind it.
/// The fade is these falling to zero (see [`Painter::scroll_tint`]).
const SCROLL_INK: Rgb = Rgb::new(0xcf, 0xd8, 0xe3);
const SCROLL_THUMB_COVER: f32 = 0.20;
const SCROLL_THUMB_HOVER_COVER: f32 = 0.36;
const SCROLL_TROUGH_COVER: f32 = 0.05;

impl ScrollLane {
    /// The bar's geometry for a window of `surface` device pixels whose grid starts at
    /// `origin` and runs `rows` deep. The painter and the pointer both build it from the
    /// same inputs, so the bar is grabbable exactly where it is drawn.
    pub fn new(
        surface: (i32, i32),
        origin: (i32, i32),
        metrics: CellMetrics,
        rows: usize,
        scale: Scale,
    ) -> ScrollLane {
        // The column the grid occupies: everything right of the origin, as deep as the
        // rows go.
        let (top, height) = (origin.1, rows as i32 * metrics.h);
        let right = surface.0;
        let wide = scale.px(SCROLL_WIDE);
        let track = Rect {
            x: right - scale.px(SCROLL_INSET) - wide,
            y: top,
            w: wide,
            h: height,
        };
        let from = |x: i32| Rect {
            x,
            y: top,
            w: (right - x).max(0),
            h: height,
        };
        ScrollLane {
            track,
            grab: from(track.x),
            zone: from(track.x - scale.px(SCROLL_NEAR)),
            min_thumb: scale.px(SCROLL_MIN_THUMB),
        }
    }

    /// The thumb as it is drawn for `screen` right now, or `None` when there is nothing
    /// to scroll. One answer, so the rectangle the pointer grabs is the one on screen.
    pub fn thumb(&self, screen: &Screen) -> Option<Rect> {
        let (content, viewport) = screen.scroll_extent();
        scroll::thumb(
            self.track,
            viewport,
            content,
            screen.scroll_position(),
            self.min_thumb,
        )
    }
}

impl Painter<'_> {
    /// The overlay scrollbar down the grid's right edge: a faint trough (only once the
    /// pointer has grown the bar into a slider) and the thumb, both tinted onto the theme
    /// background by how lit the bar is. A bar out of sight, or a screen with no history
    /// behind it (which includes every alt screen), draws nothing.
    pub(super) fn push_scrollbar(&mut self, surface: (i32, i32), rows: usize) {
        let lit = self.scrollbar.lit();
        if lit <= 0.0 {
            return;
        }
        let lane = ScrollLane::new(surface, self.origin, self.metrics, rows, self.scale);
        let track = lane.track;
        let Some(full) = lane.thumb(self.screen) else {
            return;
        };
        let grown = self.scrollbar.wide();
        // The bar grows out of the window edge: the thumb keeps its right edge and widens
        // leftward, and the trough appears under it as it goes.
        if grown > 0.0 {
            self.list.push(DrawCmd::Fill {
                rect: track,
                color: self.scroll_tint(SCROLL_TROUGH_COVER * grown * lit),
            });
        }
        let (thin, wide) = (self.scale.px(SCROLL_THIN), self.scale.px(SCROLL_WIDE));
        let w = thin + ((wide - thin) as f32 * grown).round() as i32;
        let rect = Rect {
            x: full.x + full.w - w,
            w,
            ..full
        };
        // The thumb lifts under the pointer, so a bar that is merely reporting a scroll
        // reads quieter than one asking to be grabbed.
        let cover = SCROLL_THUMB_COVER + (SCROLL_THUMB_HOVER_COVER - SCROLL_THUMB_COVER) * grown;
        self.list.push(DrawCmd::RoundRect {
            rect,
            color: self.scroll_tint(cover * lit),
            radius: w / 2,
        });
    }

    /// [`SCROLL_INK`] over the theme background at coverage `a` in 0..=1.
    ///
    /// The display list draws only opaque fills ([`crate::render::gpu`] reads a command's
    /// colour as `0x00RRGGBB` and hands the GPU an alpha of 1), so chrome that wants to
    /// read as translucent blends itself against the surface it is known to sit on.
    fn scroll_tint(&self, a: f32) -> u32 {
        let parts = (a.clamp(0.0, 1.0) * 255.0).round() as u16;
        self.theme.bg.mix(SCROLL_INK, parts, 255).to_u32()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use crate::term_render::scrollbar::*;
    use crate::term_render::tests::*;
    use crate::term_render::*;
    use crate::testutil::feed;

    /// A bar a scroll has lit, ticked past its fade-in so it draws at full strength but
    /// has not been grown by a pointer (`wide` is still 0: a thin indicator, no trough).
    fn lit_bar() -> Scrollbar {
        let start = Instant::now();
        let mut bar = Scrollbar::default();
        bar.flash(start);
        for f in 1..=8 {
            bar.tick(start + Duration::from_millis(f * 16));
        }
        assert_eq!((bar.lit(), bar.wide()), (1.0, 0.0), "lit, still thin");
        bar
    }

    /// The frame `s` paints with `bar` down its edge.
    fn list_with_bar(s: &Screen, bar: &Scrollbar) -> DisplayList {
        build_display_list(&FrameInputs {
            scrollbar: bar,
            ..inputs(s, &Theme::default())
        })
    }

    /// The scrollbar's thumb: the only rounded rectangle this painter ever draws.
    fn thumb_of(list: &[DrawCmd]) -> Option<Rect> {
        list.iter().find_map(|c| match c {
            DrawCmd::RoundRect { rect, .. } => Some(*rect),
            _ => None,
        })
    }

    /// A 6x2 screen with four lines fed through it, so two rows sit in history and the
    /// view can scroll: content 4 lines, viewport 2, max scroll 2.
    fn scrollable_screen() -> Screen {
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"one\r\ntwo\r\nthree\r\nfour");
        assert_eq!(
            s.scroll_extent(),
            (4, 2),
            "two lines of history behind two rows"
        );
        s
    }

    #[test]
    fn a_bar_out_of_sight_draws_nothing_even_while_scrolled() {
        let mut s = scrollable_screen();
        s.scroll_view_up(2);
        // The bar is state, not a function of the scroll position: until something lights
        // it, viewing history draws no chrome at all.
        assert_eq!(thumb_of(&list_of(&s)), None);
    }

    #[test]
    fn a_lit_bar_draws_a_thumb_in_its_lane() {
        let s = scrollable_screen();
        let list = list_with_bar(&s, &lit_bar());
        let thumb = thumb_of(&list).expect("a lit, scrollable screen draws a thumb");
        // It is drawn where a press would grab it, which is the whole contract between
        // the painter and the window's hit-test.
        let lane = ScrollLane::new(
            (s.dimensions().0 as i32 * M.w, s.dimensions().1 as i32 * M.h),
            (0, 0),
            M,
            s.dimensions().1,
            Scale::ONE,
        );
        assert!(
            thumb.x >= lane.grab.x && thumb.x + thumb.w <= lane.grab.x + lane.grab.w,
            "thumb {thumb:?} sits inside the grab band {:?}",
            lane.grab
        );
        // Thin at rest: no pointer has grown it, so it has not reached the full slider.
        assert_eq!(thumb.w, Scale::ONE.px(SCROLL_THIN));
    }

    #[test]
    fn no_bar_when_there_is_no_history_to_scroll() {
        let mut s = Screen::new(6, 2);
        feed(&mut s, b"one\r\ntwo"); // fills the screen, nothing retired into history
        assert_eq!(s.scroll_extent(), (2, 2), "content fits the viewport");
        assert_eq!(
            thumb_of(&list_with_bar(&s, &lit_bar())),
            None,
            "a lit bar still draws nothing when there is nothing to scroll"
        );
    }

    #[test]
    fn the_alt_screen_never_shows_a_bar() {
        let mut s = scrollable_screen();
        feed(&mut s, b"\x1b[?1049h"); // switch to the alt screen
        assert!(s.is_alt());
        assert_eq!(
            s.scroll_extent(),
            (2, 2),
            "the alt screen keeps no history, so it reports as unscrollable"
        );
        assert_eq!(thumb_of(&list_with_bar(&s, &lit_bar())), None);
    }

    #[test]
    fn the_thumb_travels_the_track_from_the_oldest_line_to_the_live_bottom() {
        let mut s = scrollable_screen();
        let bar = lit_bar();
        let track = ScrollLane::new(
            (s.dimensions().0 as i32 * M.w, s.dimensions().1 as i32 * M.h),
            (0, 0),
            M,
            s.dimensions().1,
            Scale::ONE,
        )
        .track;

        // Pinned to the live bottom (view_offset 0): the thumb's bottom meets the track's.
        let bottom = thumb_of(&list_with_bar(&s, &bar)).expect("scrollable");
        assert_eq!(bottom.y + bottom.h, track.y + track.h);

        // Scrolled all the way back: its top meets the track's top.
        s.scroll_view_to_top();
        let top = thumb_of(&list_with_bar(&s, &bar)).expect("scrollable");
        assert_eq!(top.y, track.y);
    }
}
