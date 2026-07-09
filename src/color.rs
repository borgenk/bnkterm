//! The terminal color model: an SGR color as a `Color` enum, the fixed 256-entry
//! xterm palette math, and the `Theme` that turns any `Color` into a concrete
//! `Rgb` at paint time.
//!
//! A cell stores a `Color`, not an `Rgb`, on purpose: the same escape sequence
//! has to track the live theme, so `Color::Ansi(1)` follows whatever "red" the
//! theme names and only becomes pixels when it is drawn. The four cases mirror
//! exactly what SGR can say about a color:
//!
//! ```text
//!   Default          SGR 39 / 49          the theme's default fg or bg
//!   Ansi(0..16)      SGR 30-37, 90-97      one of the 16 named slots (theme)
//!   Indexed(0..256)  SGR 38;5;n            the fixed 6x6x6 cube + gray ramp
//!   Rgb(r,g,b)       SGR 38;2;r;g;b        24-bit truecolor, theme-independent
//! ```
//!
//! xterm is the reference: the default 16-color palette and the
//! cube/ramp arithmetic below match it to the byte, because that is the palette
//! every program was written and color-tested against.

/// A resolved 24-bit color: the result of looking a [`Color`] up in a [`Theme`].
/// This is what the renderer ultimately wants; `Color` is what a cell stores.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Pack into `0x00RRGGBB`, the pixel convention the GPU path uses. The seam
    /// where a resolved cell color meets the renderer in phase 2.
    pub fn to_u32(self) -> u32 {
        (u32::from(self.r) << 16) | (u32::from(self.g) << 8) | u32::from(self.b)
    }
}

/// A color as an escape sequence names it. Small (`Copy`, one tag byte plus at
/// most a three-byte payload) so a `Cell` stays cheap to copy and compare.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Color {
    /// The terminal default (SGR 39 for foreground, 49 for background); resolves
    /// against the theme's default, which differs by ground.
    #[default]
    Default,
    /// One of the 16 named ANSI slots, `0..16` (0-7 standard, 8-15 bright). The
    /// theme owns their actual values, so a palette change moves them together.
    Ansi(u8),
    /// An index into the fixed 256-color palette (`38;5;n`). 0-15 alias the ANSI
    /// slots, 16-231 are the 6x6x6 color cube, 232-255 the grayscale ramp.
    Indexed(u8),
    /// 24-bit truecolor (`38;2;r;g;b`), independent of the theme.
    Rgb(u8, u8, u8),
}

/// Which default a [`Color::Default`] resolves to. A terminal's default
/// foreground and background are different colors, so resolution must know which
/// side of the cell it is coloring.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ground {
    Foreground,
    Background,
}

impl Color {
    /// Resolve to concrete pixels against a theme. Never panics: an out-of-range
    /// `Ansi` index is masked into `0..16` rather than trapping, keeping the
    /// no-panic rule intact even on a color the parser should never emit.
    pub fn resolve(self, theme: &Theme, ground: Ground) -> Rgb {
        match self {
            Color::Default => match ground {
                Ground::Foreground => theme.fg,
                Ground::Background => theme.bg,
            },
            // `& 0x0f` keeps the index in `0..16` so it can never trap; for the
            // valid `0..16` an ANSI color is ever built with, it is the identity.
            Color::Ansi(i) => theme.ansi[usize::from(i) & 0x0f],
            Color::Indexed(i) => theme.indexed(i),
            Color::Rgb(r, g, b) => Rgb::new(r, g, b),
        }
    }
}

/// The palette: the 16 named ANSI colors plus the default fore/background and
/// cursor color. Everything a [`Color`] needs to become an [`Rgb`]; the 256-color
/// cube and ramp are computed, not stored, since they are fixed by xterm.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Theme {
    pub fg: Rgb,
    pub bg: Rgb,
    pub cursor: Rgb,
    pub ansi: [Rgb; 16],
}

impl Theme {
    /// Resolve a 256-palette index. Total over all 256 inputs and allocation-free.
    ///
    /// ```text
    ///   0..16    the 16 ANSI slots (theme-owned)
    ///   16..232  a 6x6x6 cube; channel level 0 -> 0, else 55 + level*40
    ///            (i.e. 0, 95, 135, 175, 215, 255)
    ///   232..256 a 24-step gray ramp, 8 + step*10 (8, 18, ..., 238)
    /// ```
    pub fn indexed(&self, i: u8) -> Rgb {
        match i {
            0..=15 => self.ansi[usize::from(i)],
            16..=231 => {
                let n = i - 16;
                Rgb::new(
                    cube_level(n / 36),
                    cube_level((n / 6) % 6),
                    cube_level(n % 6),
                )
            }
            232..=255 => {
                let level = 8 + (i - 232) * 10;
                Rgb::new(level, level, level)
            }
        }
    }
}

/// One channel of the 6x6x6 color cube. `level` is `0..6`; the six steps are the
/// xterm values 0, 95, 135, 175, 215, 255.
fn cube_level(level: u8) -> u8 {
    if level == 0 {
        0
    } else {
        55 + level * 40
    }
}

impl Default for Theme {
    /// The default palette: Tomorrow Night (a common built-in default), a white
    /// foreground on `#282c34`, and a `#ff0078` cursor. These mirror a familiar
    /// mainstream-terminal look so bnkterm renders as expected out of the box;
    /// config makes them overridable later (phase 4), not baked in.
    fn default() -> Self {
        Theme {
            fg: Rgb::new(0xff, 0xff, 0xff),
            bg: Rgb::new(0x28, 0x2c, 0x34),
            cursor: Rgb::new(0xff, 0x00, 0x78),
            ansi: [
                Rgb::new(0x1d, 0x1f, 0x21), // 0  black
                Rgb::new(0xcc, 0x66, 0x66), // 1  red
                Rgb::new(0xb5, 0xbd, 0x68), // 2  green
                Rgb::new(0xf0, 0xc6, 0x74), // 3  yellow
                Rgb::new(0x81, 0xa2, 0xbe), // 4  blue
                Rgb::new(0xb2, 0x94, 0xbb), // 5  magenta
                Rgb::new(0x8a, 0xbe, 0xb7), // 6  cyan
                Rgb::new(0xc5, 0xc8, 0xc6), // 7  white
                Rgb::new(0x66, 0x66, 0x66), // 8  bright black
                Rgb::new(0xd5, 0x4e, 0x53), // 9  bright red
                Rgb::new(0xb9, 0xca, 0x4a), // 10 bright green
                Rgb::new(0xe7, 0xc5, 0x47), // 11 bright yellow
                Rgb::new(0x7a, 0xa6, 0xda), // 12 bright blue
                Rgb::new(0xc3, 0x97, 0xd8), // 13 bright magenta
                Rgb::new(0x70, 0xc0, 0xb1), // 14 bright cyan
                Rgb::new(0xea, 0xea, 0xea), // 15 bright white
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_is_one_tag_plus_three_bytes() {
        // A cell holds two of these, so the size is load-bearing; pin it.
        assert_eq!(std::mem::size_of::<Color>(), 4);
        assert_eq!(std::mem::align_of::<Color>(), 1);
    }

    #[test]
    fn default_color_resolves_per_ground() {
        let t = Theme::default();
        assert_eq!(Color::Default.resolve(&t, Ground::Foreground), t.fg);
        assert_eq!(Color::Default.resolve(&t, Ground::Background), t.bg);
    }

    #[test]
    fn ansi_follows_the_theme() {
        let t = Theme::default();
        assert_eq!(Color::Ansi(1).resolve(&t, Ground::Foreground), t.ansi[1]);
        assert_eq!(Color::Ansi(15).resolve(&t, Ground::Background), t.ansi[15]);
        // Out-of-range never traps; it masks back into 0..16.
        assert_eq!(
            Color::Ansi(0x1f).resolve(&t, Ground::Foreground),
            t.ansi[15]
        );
    }

    #[test]
    fn truecolor_is_theme_independent() {
        let t = Theme::default();
        assert_eq!(
            Color::Rgb(10, 20, 30).resolve(&t, Ground::Foreground),
            Rgb::new(10, 20, 30)
        );
    }

    #[test]
    fn indexed_matches_xterm_palette() {
        let t = Theme::default();
        // 0..16 alias the ANSI slots.
        assert_eq!(t.indexed(0), t.ansi[0]);
        assert_eq!(t.indexed(15), t.ansi[15]);
        // Cube corners: 16 is pure black, 231 is pure white.
        assert_eq!(t.indexed(16), Rgb::new(0, 0, 0));
        assert_eq!(t.indexed(231), Rgb::new(255, 255, 255));
        // A known interior point: 226 is xterm "yellow" (max r, max g, min b).
        assert_eq!(t.indexed(226), Rgb::new(255, 255, 0));
        // Ramp ends: 232 is the darkest gray (8), 255 the lightest (238).
        assert_eq!(t.indexed(232), Rgb::new(8, 8, 8));
        assert_eq!(t.indexed(255), Rgb::new(238, 238, 238));
    }

    #[test]
    fn cube_levels_are_the_xterm_six() {
        let steps: Vec<u8> = (0..6).map(cube_level).collect();
        assert_eq!(steps, vec![0, 95, 135, 175, 215, 255]);
    }

    #[test]
    fn rgb_packs_to_0x00rrggbb() {
        assert_eq!(Rgb::new(0x12, 0x34, 0x56).to_u32(), 0x0012_3456);
        assert_eq!(Rgb::new(0xff, 0xff, 0xff).to_u32(), 0x00ff_ffff);
    }

    #[test]
    fn indexed_never_panics_over_the_whole_range() {
        let t = Theme::default();
        // The standing no-panic guarantee: every one of the 256 indices resolves.
        for i in 0..=255u8 {
            let _ = t.indexed(i);
        }
    }
}
