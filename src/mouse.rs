//! Mouse reporting: turn a pointer event into the bytes a program that asked for
//! the mouse (tmux, htop, vim, less) expects. The sibling of [`crate::input`] for
//! the other input device, and pure in the same way, so the whole encoding is
//! pinned by unit tests with no window.
//!
//! A program turns reporting on with a DEC private mode and picks how far it
//! wants to hear about the pointer:
//!
//! ```text
//!   ?1000  press/release only          (MouseProtocol::Press)
//!   ?1002  + motion while a button is down (drag)   (ButtonEvent)
//!   ?1003  + all motion                 (AnyEvent)
//!   ?1006  encode with the SGR scheme (no 223-column limit)   (sgr)
//! ```
//!
//! [`crate::grid::Screen`] tracks the live [`MouseMode`] (the app reads it off the
//! screen each event); this module only encodes. Two wire formats:
//!
//! - **X10** (default): `CSI M Cb Cx Cy`, each a single byte biased by 32, so a
//!   coordinate past column 223 cannot be represented (we clamp).
//! - **SGR** (`?1006`): `CSI < b ; x ; y M` for a press/motion and `... m` for a
//!   release, decimal and unbounded. The modern default programs prefer.

use crate::input::Mods;

/// Which physical button an event names. `None` is "no button", used for the
/// hover motion the any-event protocol reports. Wheel notches arrive as presses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
    None,
    WheelUp,
    WheelDown,
}

/// What happened to the button: a press, a release, or a move (a drag when a
/// button is held, a bare move otherwise).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MouseKind {
    Press,
    Release,
    Motion,
}

/// How much of the pointer a program asked to hear about (the mutually-exclusive
/// `?1000`/`?1002`/`?1003` levels).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum MouseProtocol {
    /// Reporting off.
    #[default]
    Off,
    /// `?1000`: presses and releases only.
    Press,
    /// `?1002`: presses, releases, and motion while a button is held.
    ButtonEvent,
    /// `?1003`: presses, releases, and all motion.
    AnyEvent,
}

/// The live mouse-reporting state: the protocol level and whether the SGR
/// encoding (`?1006`) is on. `Off` means the terminal handles the mouse locally
/// (selection, wheel-scroll).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct MouseMode {
    pub protocol: MouseProtocol,
    pub sgr: bool,
}

impl MouseMode {
    /// Whether a program is currently listening for the mouse.
    pub fn reports(self) -> bool {
        self.protocol != MouseProtocol::Off
    }
}

/// The largest coordinate the byte-based X10 encoding can carry: `32 + 1 + 222 =
/// 255`. Columns past this are clamped rather than wrapping into a control byte.
const X10_MAX_COORD: usize = 223;

/// Encode a mouse event into `out` for the given mode, returning whether anything
/// was written. Motion is dropped when the protocol does not ask for it (so an
/// idle mouse over a `?1000` program stays silent), and nothing is written when
/// reporting is off. `col`/`row` are 0-based cells. `out` is a caller-reused
/// buffer, so a stream of drag events allocates nothing.
pub fn encode(
    mode: MouseMode,
    button: MouseButton,
    kind: MouseKind,
    col: usize,
    row: usize,
    mods: Mods,
    out: &mut Vec<u8>,
) -> bool {
    if !mode.reports() {
        return false;
    }
    if kind == MouseKind::Motion {
        match mode.protocol {
            MouseProtocol::Press => return false,
            // A drag needs a held button; a bare hover is only for any-event.
            MouseProtocol::ButtonEvent if button == MouseButton::None => return false,
            _ => {}
        }
    }

    let mut cb = base_code(button) + modifier_bits(mods);
    if kind == MouseKind::Motion {
        cb += 32; // the motion bit
    }

    if mode.sgr {
        out.extend_from_slice(b"\x1b[<");
        push_num(out, cb);
        out.push(b';');
        push_num(out, col as u32 + 1);
        out.push(b';');
        push_num(out, row as u32 + 1);
        // A release is the same code with a lowercase final byte.
        out.push(if kind == MouseKind::Release {
            b'm'
        } else {
            b'M'
        });
    } else {
        // X10 cannot say *which* button released, so a release is always code 3
        // (plus the modifier bits); the motion bit never rides a release.
        let code = if kind == MouseKind::Release {
            3 + modifier_bits(mods)
        } else {
            cb
        };
        let cx = (col + 1).min(X10_MAX_COORD) as u32;
        let cy = (row + 1).min(X10_MAX_COORD) as u32;
        out.extend_from_slice(b"\x1b[M");
        out.push((code + 32) as u8);
        out.push((cx + 32) as u8);
        out.push((cy + 32) as u8);
    }
    true
}

/// A convenience wrapper returning a fresh buffer (`None` when nothing is
/// reported), for tests and simple call sites.
pub fn encoded(
    mode: MouseMode,
    button: MouseButton,
    kind: MouseKind,
    col: usize,
    row: usize,
    mods: Mods,
) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    encode(mode, button, kind, col, row, mods, &mut out).then_some(out)
}

/// The base button field of the report code: the three buttons, "no button", and
/// the two wheel directions (which set the high bit, `64`).
fn base_code(button: MouseButton) -> u32 {
    match button {
        MouseButton::Left => 0,
        MouseButton::Middle => 1,
        MouseButton::Right => 2,
        MouseButton::None => 3,
        MouseButton::WheelUp => 64,
        MouseButton::WheelDown => 65,
    }
}

/// The modifier bits folded into the report code: Shift 4, Alt(Meta) 8, Ctrl 16.
fn modifier_bits(mods: Mods) -> u32 {
    let mut b = 0;
    if mods.contains(Mods::SHIFT) {
        b += 4;
    }
    if mods.contains(Mods::ALT) {
        b += 8;
    }
    if mods.contains(Mods::CTRL) {
        b += 16;
    }
    b
}

/// Append `n` as decimal ASCII, allocation-free.
fn push_num(out: &mut Vec<u8>, mut n: u32) {
    if n == 0 {
        out.push(b'0');
        return;
    }
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    while n > 0 {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    out.extend_from_slice(&tmp[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRESS: MouseMode = MouseMode {
        protocol: MouseProtocol::Press,
        sgr: false,
    };
    const SGR: MouseMode = MouseMode {
        protocol: MouseProtocol::ButtonEvent,
        sgr: true,
    };

    fn enc(mode: MouseMode, b: MouseButton, k: MouseKind, col: usize, row: usize) -> Vec<u8> {
        encoded(mode, b, k, col, row, Mods::NONE).unwrap_or_default()
    }

    #[test]
    fn off_reports_nothing() {
        assert!(encoded(
            MouseMode::default(),
            MouseButton::Left,
            MouseKind::Press,
            0,
            0,
            Mods::NONE
        )
        .is_none());
    }

    #[test]
    fn x10_press_biases_button_and_coordinates_by_32() {
        // Left press at cell (0,0): code 0+32=' ', x 1+32='!', y 1+32='!'.
        assert_eq!(
            enc(PRESS, MouseButton::Left, MouseKind::Press, 0, 0),
            b"\x1b[M !!"
        );
        // Right press at (2,3): code 2+32='"', x 3+32='#', y 4+32='$'.
        assert_eq!(
            enc(PRESS, MouseButton::Right, MouseKind::Press, 2, 3),
            b"\x1b[M\"#$"
        );
    }

    #[test]
    fn x10_release_is_always_button_three() {
        // Which button released is not encodable in X10: code 3+32='#'.
        assert_eq!(
            enc(PRESS, MouseButton::Left, MouseKind::Release, 0, 0),
            b"\x1b[M#!!"
        );
        assert_eq!(
            enc(PRESS, MouseButton::Right, MouseKind::Release, 0, 0),
            b"\x1b[M#!!"
        );
    }

    #[test]
    fn x10_coordinates_clamp_at_223() {
        // A cell past the addressable range pins to 223 (+1+32 = 255 = 0xff).
        let out = enc(PRESS, MouseButton::Left, MouseKind::Press, 500, 500);
        assert_eq!(out[3], 0x20, "button byte");
        assert_eq!(out[4], 0xff, "x clamped");
        assert_eq!(out[5], 0xff, "y clamped");
    }

    #[test]
    fn sgr_press_and_release_differ_by_final_byte() {
        // Left press at (10,4): CSI < 0 ; 11 ; 5 M
        assert_eq!(
            enc(SGR, MouseButton::Left, MouseKind::Press, 10, 4),
            b"\x1b[<0;11;5M"
        );
        // Same, released: lowercase m, and the button is preserved (unlike X10).
        assert_eq!(
            enc(SGR, MouseButton::Left, MouseKind::Release, 10, 4),
            b"\x1b[<0;11;5m"
        );
    }

    #[test]
    fn wheel_sets_the_high_bit() {
        assert_eq!(
            enc(SGR, MouseButton::WheelUp, MouseKind::Press, 0, 0),
            b"\x1b[<64;1;1M"
        );
        assert_eq!(
            enc(SGR, MouseButton::WheelDown, MouseKind::Press, 0, 0),
            b"\x1b[<65;1;1M"
        );
    }

    #[test]
    fn motion_sets_bit_five_and_respects_the_protocol() {
        // A drag (left held) under button-event reporting: code 0 + 32 = 32.
        assert_eq!(
            enc(SGR, MouseButton::Left, MouseKind::Motion, 0, 0),
            b"\x1b[<32;1;1M"
        );
        // Press-only reporting drops motion entirely.
        assert!(encoded(
            PRESS,
            MouseButton::Left,
            MouseKind::Motion,
            0,
            0,
            Mods::NONE
        )
        .is_none());
        // Button-event reporting drops a *bare* hover (no button held).
        assert!(encoded(SGR, MouseButton::None, MouseKind::Motion, 0, 0, Mods::NONE).is_none());
        // Any-event reporting keeps the bare hover: code 3 + 32 = 35.
        let any = MouseMode {
            protocol: MouseProtocol::AnyEvent,
            sgr: true,
        };
        assert_eq!(
            enc(any, MouseButton::None, MouseKind::Motion, 0, 0),
            b"\x1b[<35;1;1M"
        );
    }

    #[test]
    fn modifiers_add_to_the_code() {
        // Ctrl+Left press under SGR: 0 + 16.
        assert_eq!(
            encoded(SGR, MouseButton::Left, MouseKind::Press, 0, 0, Mods::CTRL).unwrap(),
            b"\x1b[<16;1;1M"
        );
        // Shift(4) + Alt(8) = 12.
        assert_eq!(
            encoded(
                SGR,
                MouseButton::Left,
                MouseKind::Press,
                0,
                0,
                Mods::SHIFT | Mods::ALT
            )
            .unwrap(),
            b"\x1b[<12;1;1M"
        );
    }
}
