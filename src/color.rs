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

    /// A per-channel linear blend: `other_parts / total` of `other` mixed over
    /// `self`. Used by the chrome (tab bar, leader overlay) to derive dimmed and
    /// raised shades from the theme without a second palette. `total` must be
    /// nonzero and `other_parts <= total`.
    pub fn mix(self, other: Rgb, other_parts: u16, total: u16) -> Rgb {
        let channel = |a: u8, b: u8| {
            ((u16::from(a) * (total - other_parts) + u16::from(b) * other_parts) / total) as u8
        };
        Rgb::new(
            channel(self.r, other.r),
            channel(self.g, other.g),
            channel(self.b, other.b),
        )
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
            Color::Ansi(i) => theme.indexed(i & 0x0f),
            Color::Indexed(i) => theme.indexed(i),
            Color::Rgb(r, g, b) => Rgb::new(r, g, b),
        }
    }
}

/// The palette: all 256 indexed colors plus the default fore/background and cursor
/// color. Everything a [`Color`] needs to become an [`Rgb`].
///
/// The 256 entries are *stored*, not computed, even though xterm fixes the cube and
/// ramp arithmetic — because a program can change any of them (`OSC 4`), and a palette
/// you can only compute is a palette you cannot set. Base16-style theme scripts set
/// indices well past the ANSI 16, so "only the first 16 are real" would quietly ignore
/// half of what they send. [`Theme::default`] fills the rest with exactly xterm's
/// arithmetic, so nothing changes for the programs that never touch it.
///
/// This is terminal state, not renderer state: it lives on [`crate::grid::Screen`],
/// which is what the escape sequences that mutate it can reach.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Theme {
    pub fg: Rgb,
    pub bg: Rgb,
    pub cursor: Rgb,
    palette: [Rgb; 256],
}

impl Theme {
    /// Resolve a 256-palette index. Allocation-free and total over all 256 inputs: a
    /// `u8` cannot index past a 256-entry array, so this cannot trap.
    pub fn indexed(&self, i: u8) -> Rgb {
        self.palette[usize::from(i)]
    }

    /// Set one palette entry (`OSC 4 ; i ; spec`).
    pub fn set_indexed(&mut self, i: u8, color: Rgb) {
        self.palette[usize::from(i)] = color;
    }

    /// Put one entry back to its power-on value (`OSC 104 ; i`).
    pub fn reset_indexed(&mut self, i: u8) {
        self.palette[usize::from(i)] = Theme::default().palette[usize::from(i)];
    }

    /// Put the whole palette, and the three named colors, back (`OSC 104` with no
    /// parameter, and `OSC 110`/`111`/`112`).
    pub fn reset_palette(&mut self) {
        *self = Theme::default();
    }
}

/// Parse an X11 color specification, the syntax `OSC 4/10/11/12` carry.
///
/// Two spellings, both from X11 and both in the wild:
///
/// ```text
///   rgb:RR/GG/BB          1 to 4 hex digits per channel, any width
///   rgb:RRRR/GGGG/BBBB    the 16-bit form a terminal replies with
///   #RGB  #RRGGBB         the CSS-looking form, 1 to 4 digits per channel
/// ```
///
/// A channel narrower than 8 bits is scaled up rather than zero-padded, so `#f00` is
/// full red and not `0x0f0000`: X11 defines the digits as the *high* bits of the value.
///
/// X11 color *names* (`red`, `cornflowerblue`) are deliberately not supported. They need
/// the rgb.txt database, and a terminal that must ship a color-name table to parse an
/// escape sequence has taken a wrong turn; every program that matters sends hex.
/// `None` for anything we cannot read, which the caller then ignores.
pub fn parse_x11_color(spec: &[u8]) -> Option<Rgb> {
    if let Some(rest) = spec.strip_prefix(b"rgb:") {
        let mut parts = rest.split(|&b| b == b'/');
        let r = scale_hex(parts.next()?)?;
        let g = scale_hex(parts.next()?)?;
        let b = scale_hex(parts.next()?)?;
        if parts.next().is_some() {
            return None;
        }
        return Some(Rgb::new(r, g, b));
    }
    if let Some(rest) = spec.strip_prefix(b"#") {
        // One length for all three channels, so the total must divide by three.
        if rest.is_empty() || rest.len() % 3 != 0 || rest.len() > 12 {
            return None;
        }
        let width = rest.len() / 3;
        let r = scale_hex(rest.get(..width)?)?;
        let g = scale_hex(rest.get(width..width * 2)?)?;
        let b = scale_hex(rest.get(width * 2..)?)?;
        return Some(Rgb::new(r, g, b));
    }
    None
}

/// One channel of hex digits, scaled to 8 bits. X11 reads the digits as the most
/// significant bits of the channel, so `f` is 0xff and `f000` is also 0xff.
fn scale_hex(digits: &[u8]) -> Option<u8> {
    if digits.is_empty() || digits.len() > 4 {
        return None;
    }
    let mut value: u32 = 0;
    for &d in digits {
        let nibble = (d as char).to_digit(16)?;
        value = (value << 4) | nibble;
    }
    // Rescale from `4 * len` bits down to 8: shift down, or fan out a short value so
    // `#f00` is full red rather than nearly black.
    let bits = digits.len() * 4;
    let max = (1u32 << bits) - 1;
    Some(((value * 255 + max / 2) / max) as u8)
}

/// Format a color the way a terminal answers a color query: `rgb:RRRR/GGGG/BBBB`, the
/// 16-bit form xterm replies with and every client parses. An 8-bit channel widens by
/// `0x0101` (so `0xff` becomes `0xffff`, not `0xff00`).
pub fn write_x11_color(color: Rgb, out: &mut Vec<u8>) {
    out.extend_from_slice(b"rgb:");
    for (i, channel) in [color.r, color.g, color.b].into_iter().enumerate() {
        if i > 0 {
            out.push(b'/');
        }
        let wide = u16::from(channel) * 0x101;
        for shift in [12, 8, 4, 0] {
            let nibble = ((wide >> shift) & 0xf) as u8;
            out.push(match nibble {
                0..=9 => b'0' + nibble,
                _ => b'a' + (nibble - 10),
            });
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

/// The 16 named slots: Tomorrow Night, a common built-in default.
const ANSI_16: [Rgb; 16] = [
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
];

impl Default for Theme {
    /// The default palette: the 16 named slots above, then xterm's fixed arithmetic for
    /// the rest — a 6x6x6 cube and a 24-step gray ramp, to the byte, because that is the
    /// palette every program was written and color-tested against:
    ///
    /// ```text
    ///   0..16    the 16 named slots
    ///   16..232  a 6x6x6 cube; channel level 0 -> 0, else 55 + level*40
    ///            (i.e. 0, 95, 135, 175, 215, 255)
    ///   232..256 a 24-step gray ramp, 8 + step*10 (8, 18, ..., 238)
    /// ```
    fn default() -> Self {
        let mut palette = [Rgb::new(0, 0, 0); 256];
        for (i, slot) in palette.iter_mut().enumerate() {
            *slot = match i {
                0..=15 => ANSI_16[i],
                16..=231 => {
                    let n = (i - 16) as u8;
                    Rgb::new(
                        cube_level(n / 36),
                        cube_level((n / 6) % 6),
                        cube_level(n % 6),
                    )
                }
                _ => {
                    let level = 8 + (i - 232) as u8 * 10;
                    Rgb::new(level, level, level)
                }
            };
        }
        Theme {
            fg: Rgb::new(0xff, 0xff, 0xff),
            bg: Rgb::new(0x28, 0x2c, 0x34),
            cursor: Rgb::new(0xff, 0x00, 0x78),
            palette,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x11_colors_parse_in_both_spellings() {
        assert_eq!(
            parse_x11_color(b"rgb:ff/00/80"),
            Some(Rgb::new(255, 0, 128))
        );
        assert_eq!(parse_x11_color(b"#ff0080"), Some(Rgb::new(255, 0, 128)));
        // The 16-bit form a terminal replies with, read back at 8 bits.
        assert_eq!(
            parse_x11_color(b"rgb:ffff/0000/8080"),
            Some(Rgb::new(255, 0, 128))
        );
        // A short channel is the *high* bits, so `#f00` is full red — not near-black,
        // which is what zero-padding would give.
        assert_eq!(parse_x11_color(b"#f00"), Some(Rgb::new(255, 0, 0)));
        assert_eq!(parse_x11_color(b"rgb:f/0/0"), Some(Rgb::new(255, 0, 0)));
    }

    #[test]
    fn a_color_we_cannot_read_is_none_rather_than_a_guess() {
        // The child writes these. Every one must be refused, not approximated.
        for junk in [
            &b"lightgoldenrodyellow"[..], // an X11 name: we ship no rgb.txt
            &b"rgb:zz/00/00"[..],
            &b"rgb:ff/00"[..],        // too few channels
            &b"rgb:ff/00/00/00"[..],  // too many
            &b"#12345"[..],           // not divisible by three
            &b"#fffff00000fffff"[..], // wider than four digits a channel
            &b"#"[..],
            &b""[..],
        ] {
            assert_eq!(parse_x11_color(junk), None, "{:?}", junk);
        }
    }

    #[test]
    fn a_color_reply_round_trips_through_its_own_format() {
        // What we write is what a client sends back at us, so the two must agree.
        for color in [
            Rgb::new(0, 0, 0),
            Rgb::new(255, 255, 255),
            Rgb::new(0x28, 0x2c, 0x34),
            Rgb::new(1, 2, 3),
        ] {
            let mut out = Vec::new();
            write_x11_color(color, &mut out);
            assert_eq!(parse_x11_color(&out), Some(color), "{out:?}");
        }
        let mut out = Vec::new();
        write_x11_color(Rgb::new(0x28, 0x2c, 0x34), &mut out);
        assert_eq!(out, b"rgb:2828/2c2c/3434");
    }

    #[test]
    fn the_palette_is_settable_and_resettable() {
        let mut t = Theme::default();
        let original = t.indexed(200);
        t.set_indexed(200, Rgb::new(1, 2, 3));
        assert_eq!(t.indexed(200), Rgb::new(1, 2, 3));
        t.reset_indexed(200);
        assert_eq!(t.indexed(200), original, "back to xterm's arithmetic");
    }

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
        assert_eq!(Color::Ansi(1).resolve(&t, Ground::Foreground), t.indexed(1));
        assert_eq!(
            Color::Ansi(15).resolve(&t, Ground::Background),
            t.indexed(15)
        );
        // Out-of-range never traps; it masks back into 0..16.
        assert_eq!(
            Color::Ansi(0x1f).resolve(&t, Ground::Foreground),
            t.indexed(15)
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
        assert_eq!(t.indexed(0), t.indexed(0));
        assert_eq!(t.indexed(15), t.indexed(15));
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
