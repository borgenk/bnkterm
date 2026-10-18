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

use crate::color::Theme;
use crate::config::ShellStartupConfig;
use crate::panel::Panel;
use crate::platform::geom::Rect;
use crate::platform::verdict::Refusal;
use crate::render::display::DisplayList;
use crate::term_render::CellMetrics;
use crate::width;
use std::time::{Duration, Instant};

/// How often the fade repaints. The compositor paces frames when one is already in
/// flight; this is the loop's own cadence when it is not, and 60Hz is finer than the
/// eye resolves on a 400ms ramp.
const FADE_FRAME: Duration = Duration::from_millis(16);

/// Mix resolution for the fade: parts-of-total as [`crate::color::Rgb::mix`] takes them, so the ramp
/// needs no float and lands exactly on the background at the end.
///
/// 255 is a ceiling, not a taste. `Rgb::mix` weights in `u16`, so `channel * total` must
/// stay under 65536 for a full-brightness channel — a 1000-part scale overflows on any
/// channel above 65. A 400ms fade has ~24 frames in it, so 255 steps is far finer than
/// anything that reaches the screen.
const FULL: u16 = 255;

/// Where a notice is in its life at a given moment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Phase {
    /// Standing at full ink. It looks identical frame to frame, so it needs no repaint.
    Holding,
    /// Mid-fade, drawing with this much ink in parts of [`FULL`]; each frame's colours
    /// differ from the last one's.
    Fading(u16),
    /// The fade has finished and the notice can be dropped.
    Gone,
}

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

    /// The notice for a clicked link refused by the rule judged only at the click, held
    /// and faded on the same timings as the startup notice.
    pub(crate) fn link_refused(refusal: Refusal, cfg: &ShellStartupConfig) -> Self {
        Self {
            text: format!("link not opened: {}", refusal.reason()),
            raised: Instant::now(),
            hold: cfg.hold,
            fade: cfg.fade,
        }
    }

    /// The notice's phase at `now`: solid through the hold, then its ink falling
    /// linearly to nothing across the fade, then gone.
    pub(crate) fn phase(&self, now: Instant) -> Phase {
        let elapsed = now.saturating_duration_since(self.raised);
        let Some(into_fade) = elapsed.checked_sub(self.hold) else {
            return Phase::Holding;
        };
        if into_fade >= self.fade {
            return Phase::Gone;
        }
        // Integer arithmetic on millis, so the ratio is exact at both ends. `into_fade`
        // is short of the fade, so `gone` never exceeds FULL.
        let gone = (into_fade.as_millis() * u128::from(FULL)) / self.fade.as_millis().max(1);
        Phase::Fading(u16::try_from(gone).map_or(0, |gone| FULL.saturating_sub(gone)))
    }

    /// When the window should next repaint for this notice: the end of the hold while it
    /// still stands, then one frame at a time through the fade, then never. `None` once
    /// it is gone, so a spent notice stops waking the loop.
    pub(crate) fn retry_at(&self, now: Instant) -> Option<Instant> {
        match self.phase(now) {
            Phase::Holding => Some(self.raised + self.hold),
            Phase::Fading(_) => Some(now + FADE_FRAME),
            Phase::Gone => None,
        }
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
    let strength = match notice.phase(Instant::now()) {
        Phase::Holding => FULL,
        Phase::Fading(0) | Phase::Gone => return,
        Phase::Fading(strength) => strength,
    };

    let panel = Panel::roomy(metrics);
    let panel_w = panel.width(width::text_cells(&notice.text) as i32);
    let panel_h = panel.height(1);
    // Inset from the corner by one cell, so the panel reads as floating over the grid
    // rather than welded to the window edge.
    let margin = metrics.w;
    let x0 = (surface_w - panel_w - margin).max(0);
    let y0 = margin.min((surface_h - panel_h).max(0));

    let rect = Rect {
        x: x0,
        y: y0,
        w: panel_w,
        h: panel_h,
    };
    let faded_fg = theme.bg.mix(theme.fg, strength, FULL);
    let faded_plate = theme.bg.mix(Panel::plate(theme), strength, FULL);
    panel
        .open(out, strings, rect, faded_fg, faded_plate)
        .write(0, 0, &notice.text);
}

#[cfg(test)]
mod tests {
    use crate::notice::*;
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
    fn the_phase_holds_solid_then_ramps_to_nothing() {
        let notice = raised("x", 100, 100);
        let at = |ms| notice.phase(notice.raised + Duration::from_millis(ms));
        assert_eq!(at(0), Phase::Holding, "solid when raised");
        assert_eq!(at(99), Phase::Holding, "solid until the hold ends");
        assert_eq!(at(100), Phase::Fading(FULL), "the fade starts at full ink");
        // Half way through the fade, within the one step integer division costs.
        let Phase::Fading(half) = at(150) else {
            panic!("mid-fade: {:?}", at(150));
        };
        assert!(
            half.abs_diff(FULL / 2) <= 1,
            "half way through the fade: {half}"
        );
        assert_eq!(at(200), Phase::Gone, "gone when the fade ends");
        assert_eq!(at(10_000), Phase::Gone, "and stays gone");
        let unfaded = raised("x", 100, 0);
        assert_eq!(
            unfaded.phase(unfaded.raised + Duration::from_millis(100)),
            Phase::Gone,
            "a zero fade goes straight from the hold to gone"
        );
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
        assert_eq!(notice.phase(base + Duration::from_millis(200)), Phase::Gone);
        assert_ne!(notice.phase(base + Duration::from_millis(199)), Phase::Gone);
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
