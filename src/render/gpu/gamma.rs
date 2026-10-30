//! How a glyph run is painted: its colour, the ink ramp that fades its tail, and the
//! coverage exponent that corrects how linear-light blending weights its anti-aliased
//! edges, from the two [`TextGamma`] dials.

use crate::render::display::Fade;
use crate::render::gpu::TextGamma;

impl Default for TextGamma {
    /// The shipping values, before any environment override (see
    /// `app::config_text_gamma`), so a build's text weight never depends on the
    /// environment it started in.
    fn default() -> Self {
        Self {
            light_on_dark: 1.5,
            dark_on_light: 2.0,
        }
    }
}

/// A glyph run's paint, threaded through the glyph emitters as one value: the
/// foreground `color`, the coverage `exponent` derived once from its contrast with
/// the run background (see [`coverage_exponent`]), and the optional ink ramp that
/// dissolves its tail (see [`Fade`]).
///
/// `exponent` is derived from the run's *own* colour against its background, and the
/// fade deliberately leaves that colour alone: were the fade a colour mix toward the
/// background instead, it would collapse the very contrast this exponent reads, and a
/// label's stroke weight would drift along its own tail.
#[derive(Clone, Copy)]
pub(super) struct Paint {
    pub(super) color: u32,
    pub(super) exponent: f32,
    pub(super) fade: Option<Fade>,
}

impl Paint {
    pub(super) fn new(color: u32, bg: u32, fade: Option<Fade>, gamma: TextGamma) -> Self {
        Paint {
            color,
            exponent: coverage_exponent(color, bg, gamma),
            fade,
        }
    }
}

/// The power the fragment shader raises a run's glyph coverage to, from the run's
/// foreground and background colours. Above 1 thins the anti-aliased edges, below 1
/// thickens them, and exactly 1 leaves the raster alone.
///
/// Linear-light compositing (the colour attachment is an sRGB view, so blending is
/// gamma-correct) misweights anti-aliased text in *opposite* directions depending on
/// which side is lighter, so the two cases get their own dial (see
/// [`TextGamma::default`]):
///
/// - **Light on dark**, the usual terminal text: rendered heavier than the
///   gamma-space stacks most terminals use, so thin it by
///   [`TextGamma::light_on_dark`] flat. The magnitude of the contrast does not enter:
///   the overweighting is there at any contrast.
/// - **Dark on light** (a reverse-video paste highlight, a light theme, the active
///   tab's dark label on its light block): washed out instead, so thicken it by
///   raising [`TextGamma::dark_on_light`] to a negative power, reaching its inverse
///   at maximum contrast. This one *does* scale with contrast, because the washout
///   does. Dark-on-light is never thinned, however faint the contrast: an earlier
///   `1 + 2d` ramp only crossed into thickening past half-contrast, so a dark label
///   on a mid-light block (this tab bar's turquoise) came out thin.
///
/// Computing this per run rather than per pixel keeps the policy in Rust, where it is
/// testable, and leaves the shader one `pow` instead of two.
///
/// Only the sign of `d` (which side is lighter) and a soft magnitude matter, so a
/// perceptual (gamma-space) luma is enough.
fn coverage_exponent(fg: u32, bg: u32, gamma: TextGamma) -> f32 {
    let d = luma(fg) - luma(bg);
    if d >= 0.0 {
        gamma.light_on_dark
    } else {
        // `d` is already in `[-1, 0)`; `max` is a defensive clamp, so the exponent
        // can never fall below the dial's inverse however the colours are chosen.
        gamma.dark_on_light.powf(d.max(-1.0))
    }
}

/// Rec. 709 relative luminance of a `0x00RRGGBB` colour in `[0, 1]`, taken in the
/// gamma-encoded byte space as a perceptual stand-in (enough to pick the contrast
/// direction and a soft strength; see [`coverage_exponent`]).
fn luma(color: u32) -> f32 {
    let r = ((color >> 16) & 0xff) as f32;
    let g = ((color >> 8) & 0xff) as f32;
    let b = (color & 0xff) as f32;
    (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
}

#[cfg(test)]
mod tests {
    use crate::render::gpu::gamma::*;
    use crate::render::gpu::tests::*;
    use crate::render::gpu::*;

    /// Build one `Text` run and return its glyph quads' `extra` vectors.
    fn run_extras(
        fonts: &Fonts,
        text: &str,
        color: u32,
        bg: u32,
        fade: Option<Fade>,
    ) -> Vec<[f32; 4]> {
        let list = vec![DrawCmd::Text {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 200,
                h: 20,
            },
            x: 0,
            baseline: 16,
            face: FaceKey::Prose {
                size: 16,
                style: crate::platform::freetype::FontStyle::Regular,
            },
            color,
            bg,
            fade,
            text: text.to_string(),
        }];
        let mut cache = GlyphCache::default();
        build_frame(fonts, &list, &mut cache, GAMMA)
            .vertices
            .iter()
            .filter(|v| v.mode == MODE_GLYPH)
            .map(|v| v.extra)
            .collect()
    }

    #[test]
    fn a_faded_run_carries_its_ink_ramp_without_disturbing_its_contrast() {
        let fonts = Fonts::new(&[16]).expect("default font");
        // The tab bar's dark-on-light case: the label's own colours, plus a ramp.
        let (fg, bg) = (0x0012_3028, 0x008a_beb7);
        let fade = Fade { from: 40, to: 72 };
        let faded = run_extras(&fonts, "hello", fg, bg, Some(fade));
        let solid = run_extras(&fonts, "hello", fg, bg, None);
        assert!(!faded.is_empty(), "the run emitted glyphs");
        assert_eq!(faded.len(), solid.len());

        // Every glyph in the run carries the same span, so the ramp is continuous
        // across glyph boundaries rather than quantised to them.
        assert!(
            faded
                .iter()
                .all(|e| e[1] == fade.from as f32 && e[2] == fade.to as f32),
            "the fade span reaches every glyph quad"
        );
        // An unfaded run leaves an empty span, which the shader reads as "no fade".
        assert!(
            solid.iter().all(|e| e[2] <= e[1]),
            "no fade means an empty span, not a hard cut at zero"
        );
        // The load-bearing invariant: fading changes the ink, never the contrast the
        // coverage exponent is derived from. Were the fade a colour mix toward the
        // background (as it once was), this exponent would drift toward 1.0 exactly as
        // the tail faded, and the tail would carry ~2x the anti-aliased ink of the
        // head: a smudge on a light tab rather than a dissolve.
        let want = coverage_exponent(fg, bg, GAMMA);
        assert!(
            faded.iter().chain(&solid).all(|e| e[0] == want),
            "the run's coverage exponent is untouched by the fade"
        );
    }

    #[test]
    fn coverage_exponent_thins_light_on_dark_and_thickens_dark_on_light() {
        let white = 0x00ff_ffff;
        let black = 0x0000_0000;
        // Light on dark is thinned by the light_on_dark dial, flat: linear-light
        // compositing overweights it at *any* contrast, so the correction does not
        // soften as the contrast falls.
        assert_eq!(coverage_exponent(white, black, GAMMA), 2.0);
        assert_eq!(coverage_exponent(white, 0x0080_8080, GAMMA), 2.0);
        // Dark on white (a reverse-video paste highlight) is thickened instead: the
        // exponent drops below 1, bottoming at the dial's inverse at full contrast.
        assert_eq!(coverage_exponent(black, white, GAMMA), 0.5);
        assert!(
            coverage_exponent(black, white, GAMMA) < coverage_exponent(0x0060_6060, white, GAMMA)
        );
        // Dark text on a mid-light block (the tab bar's #123028 on #8abeb7) is
        // thickened, not left near-untouched: dark-on-light thickens in proportion
        // to contrast, so a moderately lighter background still gets real weight.
        let tab = coverage_exponent(0x0012_3028, 0x008a_beb7, GAMMA);
        assert!(tab < 0.76, "moderate dark-on-light thickens, got {tab}");
        // Equal luminance takes the thinning branch rather than a discontinuity.
        assert_eq!(coverage_exponent(0x0044_4444, 0x0044_4444, GAMMA), 2.0);
        // A run is thinned at most to the dial and thickened at most to its inverse,
        // so the shader's pow never sees an exponent that could erase or blot text.
        for &(fg, bg) in &[(white, black), (black, white), (0x0012_3456, 0x00fe_dcba)] {
            let e = coverage_exponent(fg, bg, GAMMA);
            assert!((0.5..=2.0).contains(&e), "exponent {e} in range");
        }
    }

    #[test]
    fn each_gamma_dial_moves_only_its_own_contrast_direction() {
        // The bug the split exists to prevent. With one dial governing both
        // corrections, softening it slid *both* toward the no-correction exponent of
        // 1.0 at once, which means opposite visual directions: light-on-dark got
        // bolder while dark-on-light got thinner. So lowering "the gamma" to embolden
        // the grid silently thinned the active tab's dark label. Each dial must now
        // move its own direction and leave the other alone.
        let grid = (0x00d8_dee9, 0x0028_2c34); // light on dark
        let label = (0x0012_3028, 0x008a_beb7); // the active tab's dark label
        let base = GAMMA;

        // Softening the light-on-dark dial emboldens light-on-dark text (a lower
        // exponent keeps more coverage) and leaves dark-on-light text alone.
        let softer_light = TextGamma {
            light_on_dark: 1.5,
            ..base
        };
        assert!(
            coverage_exponent(grid.0, grid.1, softer_light)
                < coverage_exponent(grid.0, grid.1, base),
            "the light-on-dark dial emboldens light-on-dark text"
        );
        assert_eq!(
            coverage_exponent(label.0, label.1, softer_light),
            coverage_exponent(label.0, label.1, base),
            "and must not touch the dark-on-light label"
        );

        // Symmetrically, the dark-on-light dial moves only dark-on-light text.
        let softer_dark = TextGamma {
            dark_on_light: 1.5,
            ..base
        };
        assert!(
            coverage_exponent(label.0, label.1, softer_dark)
                > coverage_exponent(label.0, label.1, base),
            "softening the dark-on-light dial thins the dark-on-light label"
        );
        assert_eq!(
            coverage_exponent(grid.0, grid.1, softer_dark),
            coverage_exponent(grid.0, grid.1, base),
            "and must not touch light-on-dark grid text"
        );
    }
}
