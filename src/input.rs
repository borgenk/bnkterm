//! The keyboard encoder: a logical key press plus its modifiers and the
//! terminal's current modes become the exact bytes an application on the other
//! end of the PTY expects. This is the mirror image of the VT parser: `vt.rs`
//! turns the child's output bytes into grid actions; `input.rs` turns the user's
//! key presses into the child's input bytes. It is a pure function, so the whole
//! xterm key-encoding contract is pinned by unit tests with no window or PTY.
//!
//! ```text
//!   wl_keyboard.key ─(app + xkb)─▶ Key + Mods ─encode(·, modes)─▶ bytes ─▶ PTY
//! ```
//!
//! # The encoding, in one place
//!
//! bnkterm follows **xterm**, the reference every program was written against.
//! Two shapes cover almost everything:
//!
//! - **Text keys** (`Char`, Enter, Tab, Backspace, Escape): Ctrl folds the
//!   character into its C0 control byte (`Ctrl+A` → `0x01`), and Alt (the
//!   `metaSendsEscape` convention) prefixes an `ESC`. So `Alt+Ctrl+A` is
//!   `ESC 0x01`.
//! - **Named keys** (arrows, Home/End, Insert/Delete/Page, F-keys): a CSI or SS3
//!   sequence, with any modifiers folded into xterm's numeric *modifier
//!   parameter* `1 + shift + alt·2 + ctrl·4 + super·8`. Unmodified keys use the
//!   short form (`CSI A`, or `SS3 A` for arrows/Home/End while the application
//!   cursor-key mode `DECCKM` is set); a modified key always takes the long CSI
//!   form (`CSI 1 ; 5 A` for `Ctrl+Up`). The [`Mods`] bits are ordered to make
//!   the parameter fall out as `1 + mods.0` (see [`Mods`]).
//!
//! Anything the encoder does not recognise produces no bytes, so an unmapped key
//! is silently inert rather than sending garbage down the PTY.

use crate::grid::Screen;

/// A logical key press, already resolved by the layout. `Char` carries the
/// character the layout produced with Shift applied but *not* Ctrl or Alt: the
/// encoder applies those, because their transformation (a control byte, an ESC
/// prefix) is the terminal's job, not the layout's. The named keys are the ones
/// with dedicated escape sequences.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    /// A layout-produced character (Shift already reflected, e.g. `'A'`, `'@'`).
    Char(char),
    Enter,
    Tab,
    Backspace,
    Escape,
    Up,
    Down,
    Right,
    Left,
    Home,
    End,
    Insert,
    Delete,
    PageUp,
    PageDown,
    /// Function key `n`, 1-based. F1-F4 encode as SS3; F5-F12 as CSI tilde codes.
    Function(u8),
    /// The keypad's Enter, which sends `SS3 M` while the application keypad mode
    /// (`DECKPAM`) is set and a plain `CR` otherwise.
    KeypadEnter,
}

/// The active modifier chord. The bit values are deliberately laid out so the
/// raw byte is xterm's modifier parameter minus one: `Shift = 1`, `Alt = 2`,
/// `Ctrl = 4`, `Super = 8`. Then [`modifier_param`] is just `1 + mods.0`, and a
/// bare press (no modifier) is `mods.0 == 0`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Mods(u8);

impl Mods {
    pub const NONE: Mods = Mods(0);
    pub const SHIFT: Mods = Mods(1 << 0);
    pub const ALT: Mods = Mods(1 << 1);
    pub const CTRL: Mods = Mods(1 << 2);
    pub const SUPER: Mods = Mods(1 << 3);

    /// Whether every bit in `other` is set here.
    pub const fn contains(self, other: Mods) -> bool {
        self.0 & other.0 == other.0
    }

    /// Build from the four booleans an input backend reports.
    pub const fn new(shift: bool, alt: bool, ctrl: bool, super_: bool) -> Self {
        Mods((shift as u8) | ((alt as u8) << 1) | ((ctrl as u8) << 2) | ((super_ as u8) << 3))
    }
}

impl std::ops::BitOr for Mods {
    type Output = Mods;
    fn bitor(self, rhs: Mods) -> Mods {
        Mods(self.0 | rhs.0)
    }
}

/// The terminal modes that change what a key encodes to. Both are DEC private
/// modes the child turns on and off; the app reads them off the [`Screen`] each
/// key press so the encoding tracks the program's current state (a `vim` in the
/// alt screen, a shell at its prompt).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Modes {
    /// `DECCKM` (`?1`): arrows and Home/End send SS3 rather than CSI.
    pub app_cursor: bool,
    /// `DECKPAM`: the keypad sends its application-mode sequences.
    pub app_keypad: bool,
}

impl Modes {
    /// The input-relevant modes as the screen currently holds them.
    pub fn from_screen(screen: &Screen) -> Self {
        Modes {
            app_cursor: screen.app_cursor_keys(),
            app_keypad: screen.keypad_app(),
        }
    }
}

/// Encode one key press into the bytes to write to the PTY, appended to `out`.
/// Nothing is appended for a key the terminal does not send (a lone modifier, an
/// out-of-range function key). `out` is a caller-owned buffer, reused across key
/// presses; keystrokes are human-paced, not a hot path, but reusing the buffer
/// keeps even that allocation-free.
pub fn encode(key: Key, mods: Mods, modes: Modes, out: &mut Vec<u8>) {
    match key {
        Key::Char(c) => encode_char(c, mods, out),
        Key::Enter => {
            alt_prefix(mods, out);
            out.push(b'\r');
        }
        Key::Tab => {
            if mods.contains(Mods::SHIFT) {
                // Back-tab (CBT). Shift is consumed by the sequence itself.
                out.extend_from_slice(b"\x1b[Z");
            } else {
                alt_prefix(mods, out);
                out.push(b'\t');
            }
        }
        Key::Backspace => {
            alt_prefix(mods, out);
            // xterm: Backspace is DEL; Ctrl+Backspace is BS.
            out.push(if mods.contains(Mods::CTRL) {
                0x08
            } else {
                0x7f
            });
        }
        Key::Escape => {
            alt_prefix(mods, out);
            out.push(0x1b);
        }
        Key::Up => cursor_key(b'A', mods, modes, out),
        Key::Down => cursor_key(b'B', mods, modes, out),
        Key::Right => cursor_key(b'C', mods, modes, out),
        Key::Left => cursor_key(b'D', mods, modes, out),
        Key::Home => cursor_key(b'H', mods, modes, out),
        Key::End => cursor_key(b'F', mods, modes, out),
        Key::Insert => tilde_key(2, mods, out),
        Key::Delete => tilde_key(3, mods, out),
        Key::PageUp => tilde_key(5, mods, out),
        Key::PageDown => tilde_key(6, mods, out),
        Key::Function(n) => function_key(n, mods, out),
        Key::KeypadEnter => {
            if modes.app_keypad {
                out.extend_from_slice(b"\x1bOM");
            } else {
                alt_prefix(mods, out);
                out.push(b'\r');
            }
        }
    }
}

/// A convenience wrapper returning a fresh buffer, for call sites (and tests)
/// that do not thread one through.
pub fn encoded(key: Key, mods: Mods, modes: Modes) -> Vec<u8> {
    let mut out = Vec::new();
    encode(key, mods, modes, &mut out);
    out
}

/// A layout character with Ctrl and Alt applied. Ctrl folds the character to its
/// C0 control byte when one exists; Alt prefixes an `ESC` before whatever the
/// non-Alt encoding would be (`metaSendsEscape`).
fn encode_char(c: char, mods: Mods, out: &mut Vec<u8>) {
    if mods.contains(Mods::CTRL) {
        if let Some(b) = ctrl_byte(c) {
            alt_prefix(mods, out);
            out.push(b);
            return;
        }
        // Ctrl with no control mapping (e.g. Ctrl+1): fall through to the plain
        // character, still honouring an Alt prefix.
    }
    alt_prefix(mods, out);
    let mut buf = [0u8; 4];
    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
}

/// The C0 control byte for `Ctrl+c`, or `None` when Ctrl has no effect on this
/// character. The mappable set is the classic one: the letters, `@` through `_`
/// (which `& 0x1f` sends to `0x00`-`0x1f`), plus `Space` → NUL and `?` → DEL.
fn ctrl_byte(c: char) -> Option<u8> {
    match c {
        ' ' => Some(0x00),
        '?' => Some(0x7f),
        'a'..='z' => Some((c as u8) & 0x1f),
        '@'..='_' => Some((c as u8) & 0x1f),
        _ => None,
    }
}

/// An arrow or Home/End key. Unmodified, it is `CSI <final>` normally or
/// `SS3 <final>` under the application cursor-key mode; modified, it is always
/// `CSI 1 ; <param> <final>`.
fn cursor_key(final_byte: u8, mods: Mods, modes: Modes, out: &mut Vec<u8>) {
    if mods.0 == 0 {
        out.push(0x1b);
        out.push(if modes.app_cursor { b'O' } else { b'[' });
        out.push(final_byte);
    } else {
        out.extend_from_slice(b"\x1b[1;");
        push_num(out, modifier_param(mods));
        out.push(final_byte);
    }
}

/// An Insert/Delete/Page key: `CSI <n> ~`, or `CSI <n> ; <param> ~` when
/// modified. These do not vary with the cursor-key mode.
fn tilde_key(n: u32, mods: Mods, out: &mut Vec<u8>) {
    out.extend_from_slice(b"\x1b[");
    push_num(out, n);
    if mods.0 != 0 {
        out.push(b';');
        push_num(out, modifier_param(mods));
    }
    out.push(b'~');
}

/// A function key. F1-F4 are `SS3 P..S` (or `CSI 1 ; <param> P..S` when
/// modified); F5-F12 are tilde codes (`CSI 15 ~` .. `CSI 24 ~`). Keys outside
/// 1-12 produce nothing.
fn function_key(n: u8, mods: Mods, out: &mut Vec<u8>) {
    match n {
        1..=4 => {
            let final_byte = b'P' + (n - 1);
            if mods.0 == 0 {
                out.push(0x1b);
                out.push(b'O');
                out.push(final_byte);
            } else {
                out.extend_from_slice(b"\x1b[1;");
                push_num(out, modifier_param(mods));
                out.push(final_byte);
            }
        }
        5..=12 => {
            // The xterm tilde codes for F5..F12, with their two gaps.
            const CODES: [u32; 8] = [15, 17, 18, 19, 20, 21, 23, 24];
            tilde_key(CODES[(n - 5) as usize], mods, out);
        }
        _ => {}
    }
}

/// xterm's modifier parameter: `1 + shift + alt·2 + ctrl·4 + super·8`. The
/// [`Mods`] bits are laid out to make this `1 + mods.0`.
fn modifier_param(mods: Mods) -> u32 {
    1 + u32::from(mods.0)
}

/// Prefix an `ESC` when Alt is held, the `metaSendsEscape` convention that turns
/// `Alt+x` into `ESC x`.
fn alt_prefix(mods: Mods, out: &mut Vec<u8>) {
    if mods.contains(Mods::ALT) {
        out.push(0x1b);
    }
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

/// Map a raw Linux evdev keycode (as `wl_keyboard.key` reports it, before the
/// libxkbcommon `+8` offset) to a named [`Key`], or `None` for a key that
/// carries text (which the app resolves to a [`Key::Char`] through the layout).
/// Only the keys with dedicated escape sequences are named here; everything else
/// is text. The codes are from `linux/input-event-codes.h`, a kernel ABI.
pub fn key_from_keycode(keycode: u32) -> Option<Key> {
    Some(match keycode {
        keycode::ESC => Key::Escape,
        keycode::BACKSPACE => Key::Backspace,
        keycode::TAB => Key::Tab,
        keycode::ENTER => Key::Enter,
        keycode::KP_ENTER => Key::KeypadEnter,
        keycode::INSERT => Key::Insert,
        keycode::DELETE => Key::Delete,
        keycode::HOME => Key::Home,
        keycode::END => Key::End,
        keycode::PAGEUP => Key::PageUp,
        keycode::PAGEDOWN => Key::PageDown,
        keycode::UP => Key::Up,
        keycode::DOWN => Key::Down,
        keycode::LEFT => Key::Left,
        keycode::RIGHT => Key::Right,
        keycode::F1..=keycode::F10 => Key::Function((keycode - keycode::F1 + 1) as u8),
        keycode::F11 => Key::Function(11),
        keycode::F12 => Key::Function(12),
        _ => return None,
    })
}

/// Raw Linux evdev keycodes (`linux/input-event-codes.h`), the layout-independent
/// hardware codes `wl_keyboard.key` delivers. Kept separate from the layout's
/// keysyms: a keysym follows the user's layout, these do not, so a named key
/// (an arrow, a function key) is identified by its physical code and text keys
/// by their resolved character.
mod keycode {
    pub const ESC: u32 = 1;
    pub const BACKSPACE: u32 = 14;
    pub const TAB: u32 = 15;
    pub const ENTER: u32 = 28;
    pub const F1: u32 = 59;
    pub const F10: u32 = 68;
    pub const KP_ENTER: u32 = 96;
    pub const F11: u32 = 87;
    pub const F12: u32 = 88;
    pub const HOME: u32 = 102;
    pub const UP: u32 = 103;
    pub const PAGEUP: u32 = 104;
    pub const LEFT: u32 = 105;
    pub const RIGHT: u32 = 106;
    pub const END: u32 = 107;
    pub const DOWN: u32 = 108;
    pub const PAGEDOWN: u32 = 109;
    pub const INSERT: u32 = 110;
    pub const DELETE: u32 = 111;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode with default (all-off) modes, the common case.
    fn enc(key: Key, mods: Mods) -> Vec<u8> {
        encoded(key, mods, Modes::default())
    }

    #[test]
    fn plain_text_is_its_utf8() {
        assert_eq!(enc(Key::Char('a'), Mods::NONE), b"a");
        assert_eq!(enc(Key::Char('A'), Mods::NONE), b"A"); // Shift already in the char
        assert_eq!(enc(Key::Char('€'), Mods::NONE), "€".as_bytes());
    }

    #[test]
    fn ctrl_folds_letters_to_control_bytes() {
        assert_eq!(enc(Key::Char('a'), Mods::CTRL), vec![0x01]);
        assert_eq!(enc(Key::Char('c'), Mods::CTRL), vec![0x03]);
        assert_eq!(enc(Key::Char('z'), Mods::CTRL), vec![0x1a]);
        // The symbol block @.._ and the two specials.
        assert_eq!(enc(Key::Char('@'), Mods::CTRL), vec![0x00]);
        assert_eq!(enc(Key::Char('['), Mods::CTRL), vec![0x1b]);
        assert_eq!(enc(Key::Char('\\'), Mods::CTRL), vec![0x1c]);
        assert_eq!(enc(Key::Char(']'), Mods::CTRL), vec![0x1d]);
        assert_eq!(enc(Key::Char(' '), Mods::CTRL), vec![0x00]);
        assert_eq!(enc(Key::Char('?'), Mods::CTRL), vec![0x7f]);
    }

    #[test]
    fn ctrl_without_a_mapping_sends_the_plain_char() {
        // Ctrl+1 has no control byte on this path; the digit goes through.
        assert_eq!(enc(Key::Char('1'), Mods::CTRL), b"1");
    }

    #[test]
    fn alt_prefixes_escape() {
        assert_eq!(enc(Key::Char('a'), Mods::ALT), vec![0x1b, b'a']);
        // Alt+Ctrl+A is ESC then the control byte.
        assert_eq!(
            enc(Key::Char('a'), Mods::ALT | Mods::CTRL),
            vec![0x1b, 0x01]
        );
    }

    #[test]
    fn enter_tab_backspace_escape() {
        assert_eq!(enc(Key::Enter, Mods::NONE), vec![b'\r']);
        assert_eq!(enc(Key::Tab, Mods::NONE), vec![b'\t']);
        assert_eq!(enc(Key::Tab, Mods::SHIFT), b"\x1b[Z"); // back-tab
        assert_eq!(enc(Key::Backspace, Mods::NONE), vec![0x7f]);
        assert_eq!(enc(Key::Backspace, Mods::CTRL), vec![0x08]);
        assert_eq!(enc(Key::Backspace, Mods::ALT), vec![0x1b, 0x7f]);
        assert_eq!(enc(Key::Escape, Mods::NONE), vec![0x1b]);
        assert_eq!(enc(Key::Escape, Mods::ALT), vec![0x1b, 0x1b]);
    }

    #[test]
    fn arrows_plain_and_application_mode() {
        assert_eq!(enc(Key::Up, Mods::NONE), b"\x1b[A");
        assert_eq!(enc(Key::Left, Mods::NONE), b"\x1b[D");
        let app = Modes {
            app_cursor: true,
            ..Modes::default()
        };
        assert_eq!(encoded(Key::Up, Mods::NONE, app), b"\x1bOA");
        assert_eq!(encoded(Key::Right, Mods::NONE, app), b"\x1bOC");
    }

    #[test]
    fn modified_arrows_take_the_csi_parameter_form() {
        // Ctrl+Up -> param 5; the CSI form is used even under app-cursor mode.
        assert_eq!(enc(Key::Up, Mods::CTRL), b"\x1b[1;5A");
        assert_eq!(enc(Key::Left, Mods::SHIFT), b"\x1b[1;2D");
        assert_eq!(enc(Key::Right, Mods::ALT), b"\x1b[1;3C");
        assert_eq!(enc(Key::Down, Mods::SHIFT | Mods::CTRL), b"\x1b[1;6B");
        let app = Modes {
            app_cursor: true,
            ..Modes::default()
        };
        assert_eq!(encoded(Key::Up, Mods::CTRL, app), b"\x1b[1;5A");
    }

    #[test]
    fn home_and_end() {
        assert_eq!(enc(Key::Home, Mods::NONE), b"\x1b[H");
        assert_eq!(enc(Key::End, Mods::NONE), b"\x1b[F");
        assert_eq!(enc(Key::Home, Mods::CTRL), b"\x1b[1;5H");
    }

    #[test]
    fn tilde_keys() {
        assert_eq!(enc(Key::Insert, Mods::NONE), b"\x1b[2~");
        assert_eq!(enc(Key::Delete, Mods::NONE), b"\x1b[3~");
        assert_eq!(enc(Key::PageUp, Mods::NONE), b"\x1b[5~");
        assert_eq!(enc(Key::PageDown, Mods::NONE), b"\x1b[6~");
        // Modified: the parameter slots between the number and the tilde.
        assert_eq!(enc(Key::Delete, Mods::CTRL), b"\x1b[3;5~");
        assert_eq!(enc(Key::PageUp, Mods::SHIFT), b"\x1b[5;2~");
    }

    #[test]
    fn function_keys() {
        assert_eq!(enc(Key::Function(1), Mods::NONE), b"\x1bOP");
        assert_eq!(enc(Key::Function(4), Mods::NONE), b"\x1bOS");
        assert_eq!(enc(Key::Function(5), Mods::NONE), b"\x1b[15~");
        assert_eq!(enc(Key::Function(10), Mods::NONE), b"\x1b[21~");
        assert_eq!(enc(Key::Function(12), Mods::NONE), b"\x1b[24~");
        // Modified.
        assert_eq!(enc(Key::Function(1), Mods::SHIFT), b"\x1b[1;2P");
        assert_eq!(enc(Key::Function(5), Mods::CTRL), b"\x1b[15;5~");
        // Out of range: nothing.
        assert!(enc(Key::Function(0), Mods::NONE).is_empty());
        assert!(enc(Key::Function(13), Mods::NONE).is_empty());
    }

    #[test]
    fn keypad_enter_follows_the_keypad_mode() {
        assert_eq!(enc(Key::KeypadEnter, Mods::NONE), vec![b'\r']);
        let app = Modes {
            app_keypad: true,
            ..Modes::default()
        };
        assert_eq!(encoded(Key::KeypadEnter, Mods::NONE, app), b"\x1bOM");
    }

    #[test]
    fn modifier_parameter_is_one_plus_the_bits() {
        // The bit layout is chosen so the xterm parameter is 1 + mods.0.
        assert_eq!(modifier_param(Mods::NONE), 1);
        assert_eq!(modifier_param(Mods::SHIFT), 2);
        assert_eq!(modifier_param(Mods::ALT), 3);
        assert_eq!(modifier_param(Mods::CTRL), 5);
        assert_eq!(modifier_param(Mods::SUPER), 9);
        assert_eq!(modifier_param(Mods::SHIFT | Mods::CTRL), 6);
        assert_eq!(modifier_param(Mods::new(true, true, true, true)), 16);
    }

    #[test]
    fn keycodes_map_to_named_keys() {
        assert_eq!(key_from_keycode(103), Some(Key::Up));
        assert_eq!(key_from_keycode(108), Some(Key::Down));
        assert_eq!(key_from_keycode(28), Some(Key::Enter));
        assert_eq!(key_from_keycode(1), Some(Key::Escape));
        assert_eq!(key_from_keycode(110), Some(Key::Insert));
        assert_eq!(key_from_keycode(59), Some(Key::Function(1)));
        assert_eq!(key_from_keycode(68), Some(Key::Function(10)));
        assert_eq!(key_from_keycode(87), Some(Key::Function(11)));
        assert_eq!(key_from_keycode(88), Some(Key::Function(12)));
        assert_eq!(key_from_keycode(96), Some(Key::KeypadEnter));
        // A letter key carries text, so it is not a named key here.
        assert_eq!(key_from_keycode(30), None); // 'a' on a US layout
    }

    #[test]
    fn a_reused_buffer_appends_each_press() {
        // The app threads one buffer through a burst of keys; encode appends.
        let mut out = Vec::new();
        encode(Key::Char('h'), Mods::NONE, Modes::default(), &mut out);
        encode(Key::Char('i'), Mods::NONE, Modes::default(), &mut out);
        encode(Key::Enter, Mods::NONE, Modes::default(), &mut out);
        assert_eq!(out, b"hi\r");
    }
}
