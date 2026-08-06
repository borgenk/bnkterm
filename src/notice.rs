//! The transient corner notice: one line the terminal says about itself, then takes back.
//!
//! A terminal has almost nothing to say to its user, and the little it does have is
//! easy to say badly. The bar for putting a word on screen is high: this is the shell's
//! surface, the child owns every cell of the grid, and chrome that persists is chrome
//! that gets in the way. So a notice is a *statement*, not a dialog — it names one fact,
//! stands briefly in a corner, and fades without ever having asked for anything.
//!
//! ```text
//!   ┌──────────────────────────────┬────────────┐
//!   │                              │  notice ▸  │  top-right: the tab strip is at the
//!   │  $ prompt                    └────────────┤  bottom and a fresh prompt starts at
//!   │  █                                        │  the top-left, so this corner is the
//!   │                                           │  one that is free at the moment a tab
//!   ├───────────────────────────────────────────┤  opens, which is when it is raised
//!   │  tab strip                                │
//!   └───────────────────────────────────────────┘
//! ```
//!
//! The one case today is a shell that was slow reaching its first prompt (see
//! [`crate::config::ShellStartupConfig`] for the measurement and why the threshold is
//! absolute). The value is narrower than it looks, and worth stating plainly: this says
//! *the wait was the shell's, not the terminal's*. It does not say why, because a
//! terminal that diagnosed `compinit` would be a terminal that had learned the internals
//! of three shells and would relearn them forever.
//!
//! Fading is the panel and its text mixing toward the background, because the display
//! list carries no alpha (the overlay scrollbar fades the same way). [`paint`] is pure:
//! a notice plus a surface yields draw commands, so it is tested by reading the list
//! back rather than by looking at a window.

use crate::color::{Rgb, Theme};
use crate::config::ShellStartupConfig;
use crate::platform::freetype::{FaceKey, FontStyle};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::render::display::{DisplayList, DrawCmd, RoundedCorners};
use crate::term_render::{self, CellMetrics};
use std::time::{Duration, Instant};

/// How often the fade repaints. The compositor paces frames when one is already in
/// flight; this is the loop's own cadence when it is not, and 60Hz is finer than the
/// eye resolves on a 400ms ramp.
const FADE_FRAME: Duration = Duration::from_millis(16);

/// Mix resolution for the fade: parts-of-total as [`Rgb::mix`] takes them, so the ramp
/// needs no float and lands exactly on the background at the end.
///
/// 255 is a ceiling, not a taste. `Rgb::mix` weights in `u16`, so `channel * total` must
/// stay under 65536 for a full-brightness channel — a 1000-part scale overflows on any
/// channel above 65. A 400ms fade has ~24 frames in it, so 255 steps is far finer than
/// anything that reaches the screen.
const FULL: u16 = 255;

/// A raised notice: what it says, and when it was raised. The timings are copied from
/// the config at raise time rather than borrowed, so a notice is self-contained and the
/// painter needs nothing but this and a surface.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Notice {
    text: String,
    raised: Instant,
    hold: Duration,
    fade: Duration,
}

impl Notice {
    /// The notice for a shell that was slow to its first prompt, or `None` when it was
    /// inside the budget or the budget is disabled.
    ///
    /// Rounded to whole milliseconds: the number is there to be *compared* against a
    /// threshold the user set in milliseconds, and microsecond precision on a figure
    /// that varies run to run would imply a stability it does not have.
    pub(crate) fn shell_startup(took: Duration, cfg: &ShellStartupConfig) -> Option<Self> {
        let warn_after = cfg.warn_after?;
        if took < warn_after {
            return None;
        }
        Some(Self {
            text: format!("shell startup {}ms", took.as_millis()),
            raised: Instant::now(),
            hold: cfg.hold,
            fade: cfg.fade,
        })
    }

    /// Whether the fade has finished and the notice can be dropped.
    pub(crate) fn expired(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.raised) >= self.hold + self.fade
    }

    /// Whether the notice is mid-fade, so this frame's colours differ from the last
    /// one's. A notice that merely stands looks identical frame to frame, and saying so
    /// is what keeps the hold from repainting for nothing.
    pub(crate) fn fading(&self, now: Instant) -> bool {
        let elapsed = now.saturating_duration_since(self.raised);
        elapsed >= self.hold && elapsed < self.hold + self.fade
    }

    /// When the window should next repaint for this notice: the end of the hold while it
    /// still stands, then one frame at a time through the fade, then never. `None` once
    /// it has expired, so a spent notice stops waking the loop.
    pub(crate) fn retry_at(&self, now: Instant) -> Option<Instant> {
        let elapsed = now.saturating_duration_since(self.raised);
        if elapsed < self.hold {
            Some(self.raised + self.hold)
        } else if elapsed < self.hold + self.fade {
            Some(now + FADE_FRAME)
        } else {
            None
        }
    }

    /// How much ink the notice draws with right now, in parts of [`FULL`]: solid through
    /// the hold, then falling linearly to nothing across the fade.
    fn strength(&self, now: Instant) -> u16 {
        let elapsed = now.saturating_duration_since(self.raised);
        let Some(into_fade) = elapsed.checked_sub(self.hold) else {
            return FULL;
        };
        if self.fade.is_zero() || into_fade >= self.fade {
            return 0;
        }
        // Integer arithmetic on millis: the fade is bounded well under a second, so this
        // cannot overflow u32, and the ratio is exact at both ends.
        let gone = (into_fade.as_millis() * u128::from(FULL)) / self.fade.as_millis().max(1);
        FULL.saturating_sub(gone as u16)
    }
}

/// Append the notice to `out`: a rounded panel inset from the top-right corner, with its
/// line drawn on it. Nothing is drawn for `None` or for a spent notice, so a frame
/// without one is byte-identical to a frame built with no notice support at all — which
/// is what keeps the damage diff from seeing a change that is not there.
///
/// Colours are mixed toward the background by the notice's own strength, so the whole
/// panel leaves together rather than the text outliving the panel under it.
pub(crate) fn paint(
    notice: Option<&Notice>,
    out: &mut DisplayList,
    strings: &mut Vec<String>,
    metrics: CellMetrics,
    theme: &Theme,
    surface_w: i32,
    surface_h: i32,
) {
    let Some(notice) = notice else {
        return;
    };
    if metrics.w <= 0 || metrics.h <= 0 {
        return;
    }
    let strength = notice.strength(Instant::now());
    if strength == 0 {
        return;
    }

    let content_cols = cells_wide(&notice.text) as i32;
    let pad_x = metrics.w;
    let pad_y = (metrics.h / 2).max(1);
    let panel_w = content_cols * metrics.w + 2 * pad_x;
    let panel_h = metrics.h + 2 * pad_y;
    // Inset from the corner by one cell, so the panel reads as floating over the grid
    // rather than welded to the window edge.
    let margin = metrics.w;
    let x0 = (surface_w - panel_w - margin).max(0);
    let y0 = margin.min((surface_h - panel_h).max(0));

    // The panel is the same darkened plate the leader overlay uses, so the two pieces of
    // chrome that float over the grid look like one family.
    let panel_bg = theme.bg.mix(Rgb::new(0, 0, 0), 1, 2);
    let faded_bg = theme.bg.mix(panel_bg, strength, FULL);
    let faded_fg = theme.bg.mix(theme.fg, strength, FULL);
    let radius = (metrics.h / 3).max(2);

    out.push(DrawCmd::RoundRect {
        rect: Rect {
            x: x0,
            y: y0,
            w: panel_w,
            h: panel_h,
        },
        color: faded_bg.to_u32(),
        radius,
        corners: RoundedCorners::Both,
    });
    term_render::push_cell_text(
        out,
        strings,
        &notice.text,
        x0 + pad_x,
        y0 + pad_y + metrics.baseline,
        metrics,
        FaceKey::Prose {
            size: metrics.size,
            style: FontStyle::Regular,
        },
        faded_fg.to_u32(),
        faded_bg.to_u32(),
    );
}

/// Cells the text occupies, so the panel is sized in the same fixed pitch the run is
/// drawn in.
fn cells_wide(text: &str) -> usize {
    grapheme::graphemes(text)
        .map(|(_, cluster)| term_render::display_cluster_width(cluster).max(1))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const METRICS: CellMetrics = CellMetrics {
        size: 16,
        w: 8,
        h: 16,
        baseline: 12,
        ascent: 12,
        descent: 4,
        lock_glyph: true,
    };

    fn cfg() -> ShellStartupConfig {
        ShellStartupConfig::default()
    }

    fn raised(text: &str, hold_ms: u64, fade_ms: u64) -> Notice {
        Notice {
            text: text.to_string(),
            raised: Instant::now(),
            hold: Duration::from_millis(hold_ms),
            fade: Duration::from_millis(fade_ms),
        }
    }

    #[test]
    fn a_startup_inside_the_budget_says_nothing() {
        // The threshold is the whole contract: under it there is no notice at all, not
        // a notice drawn at zero strength, so nothing reaches the display list.
        assert_eq!(
            Notice::shell_startup(Duration::from_millis(40), &cfg()),
            None
        );
        assert_eq!(
            Notice::shell_startup(Duration::from_millis(149), &cfg()),
            None
        );
        assert!(Notice::shell_startup(Duration::from_millis(150), &cfg()).is_some());
        assert!(Notice::shell_startup(Duration::from_millis(500), &cfg()).is_some());
    }

    #[test]
    fn a_disabled_budget_never_raises_one() {
        // `None` is what a `0` in a config file means: the user pays this cost knowingly.
        let off = ShellStartupConfig {
            warn_after: None,
            ..cfg()
        };
        for ms in [0, 80, 5_000, 60_000] {
            assert_eq!(Notice::shell_startup(Duration::from_millis(ms), &off), None);
        }
    }

    #[test]
    fn the_line_names_the_measurement_in_whole_milliseconds() {
        // Truncated, not rounded: 212.4ms reads as 212ms. The figure is there to be
        // compared against a threshold set in milliseconds, and a fraction of one would
        // imply a stability a startup time does not have.
        let notice = Notice::shell_startup(Duration::from_micros(212_400), &cfg()).expect("raised");
        assert_eq!(notice.text, "shell startup 212ms");
    }

    #[test]
    fn strength_holds_solid_then_ramps_to_nothing() {
        let notice = raised("x", 100, 100);
        let at = |ms| notice.strength(notice.raised + Duration::from_millis(ms));
        assert_eq!(at(0), FULL, "solid when raised");
        assert_eq!(at(99), FULL, "solid until the hold ends");
        // Half way through the fade, within the one step integer division costs.
        assert!(
            at(150).abs_diff(FULL / 2) <= 1,
            "half way through the fade: {}",
            at(150)
        );
        assert_eq!(at(200), 0, "gone when the fade ends");
        assert_eq!(at(10_000), 0, "and stays gone");
    }

    #[test]
    fn a_spent_notice_stops_waking_the_loop() {
        let notice = raised("x", 100, 100);
        let base = notice.raised;
        // Still standing: wake once, when the hold ends.
        assert_eq!(
            notice.retry_at(base),
            Some(base + Duration::from_millis(100))
        );
        // Mid-fade: wake a frame at a time, so the ramp is drawn rather than stepped.
        assert_eq!(
            notice.retry_at(base + Duration::from_millis(150)),
            Some(base + Duration::from_millis(150) + FADE_FRAME)
        );
        // Spent: no deadline, so an idle terminal goes back to blocking indefinitely.
        assert_eq!(notice.retry_at(base + Duration::from_millis(200)), None);
        assert!(notice.expired(base + Duration::from_millis(200)));
        assert!(!notice.expired(base + Duration::from_millis(199)));
    }

    #[test]
    fn nothing_is_drawn_without_a_notice_or_after_it_expires() {
        let theme = Theme::default();
        let mut out = Vec::new();
        let mut strings = Vec::new();

        paint(None, &mut out, &mut strings, METRICS, &theme, 800, 600);
        assert!(out.is_empty(), "no notice draws nothing");

        // A frame past the fade must be byte-identical to a frame with no notice at all,
        // or the damage diff sees a change every time and repaints forever.
        let spent = raised("x", 0, 0);
        paint(
            Some(&spent),
            &mut out,
            &mut strings,
            METRICS,
            &theme,
            800,
            600,
        );
        assert!(out.is_empty(), "a spent notice draws nothing");
    }

    #[test]
    fn the_panel_sits_inset_in_the_top_right_and_holds_its_text() {
        let theme = Theme::default();
        let mut out = Vec::new();
        let mut strings = Vec::new();
        let notice = raised("shell startup 112ms", 1_000, 100);
        paint(
            Some(&notice),
            &mut out,
            &mut strings,
            METRICS,
            &theme,
            800,
            600,
        );

        let Some(DrawCmd::RoundRect { rect, .. }) = out.first() else {
            panic!("the panel is the first command, so the text lands on top of it");
        };
        // Inset by a cell from the top and the right edge, never off the surface.
        assert_eq!(rect.y, METRICS.w, "one cell down from the top");
        assert_eq!(
            rect.x + rect.w,
            800 - METRICS.w,
            "one cell in from the right edge"
        );
        assert!(rect.x >= 0 && rect.y >= 0);
        // Drawn as fixed-pitch cells like every other run over the grid, and at the
        // plain foreground while the notice still stands at full strength.
        let text: String = out
            .iter()
            .filter_map(|cmd| match cmd {
                DrawCmd::Cells { text, color, .. } => {
                    assert_eq!(*color, theme.fg.to_u32(), "solid during the hold");
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(text, "shell startup 112ms");
    }

    #[test]
    fn a_surface_too_small_for_the_panel_still_lands_on_it() {
        // A window dragged narrower than the notice must not push the panel to a
        // negative origin, which would draw it off-surface or wrap the arithmetic.
        let theme = Theme::default();
        let mut out = Vec::new();
        let mut strings = Vec::new();
        let notice = raised("shell startup 112ms", 1_000, 100);
        paint(
            Some(&notice),
            &mut out,
            &mut strings,
            METRICS,
            &theme,
            20,
            10,
        );

        let Some(DrawCmd::RoundRect { rect, .. }) = out.first() else {
            panic!("panel");
        };
        assert!(
            rect.x >= 0 && rect.y >= 0,
            "clamped onto the surface: {rect:?}"
        );
    }
}
