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
//!
//! # The hole in the legacy encoding, and the two protocols that fill it
//!
//! The legacy encoding cannot express most of the keyboard. `Shift+Enter` is the
//! famous one: there is no byte for it, so xterm sends plain `CR` and the
//! application cannot tell it from `Enter`. The same goes for `Ctrl+Enter`,
//! `Ctrl+I` versus `Tab`, and `Esc` versus the `ESC` that opens every escape
//! sequence. Two protocols fix this, and an application turns one on when it wants
//! exact keys:
//!
//! - **kitty's keyboard protocol** ([`KittyFlags`]): `CSI <code> ; <mods> u`, so
//!   `Shift+Enter` is `CSI 13;2u`. Pushed with `CSI > 1 u`. This is the modern one;
//!   neovim, helix and Claude Code all speak it.
//! - **xterm's `modifyOtherKeys`** ([`ModifyOtherKeys`]): `CSI 27 ; <mods> ; <code> ~`,
//!   so `Shift+Enter` is `CSI 27;2;13~`. Enabled with `CSI > 4 ; 2 m`. Older, and
//!   what tmux drives with `extended-keys on`.
//!
//! So there are three encodings, and the one in force is whichever the *application*
//! asked for (see [`encode`] for the precedence):
//!
//! ```text
//!                    ┌─ kitty flags pushed?  ──▶ CSI u        (CSI 13;2u)
//!   Key + Mods ──────┼─ modifyOtherKeys set? ──▶ CSI 27 ~     (CSI 27;2;13~)
//!                    └─ neither (the default) ─▶ legacy xterm (CR)
//! ```
//!
//! # Shift+Enter without a protocol
//!
//! Nothing negotiates with a terminal it does not recognise, and bnkterm is on
//! nobody's list (see the `TERM_PROGRAM` note in `app.rs`), so the legacy path has to
//! carry `Shift+Enter` on its own. It sends **LF** (`0x0a`), which is exactly what
//! `Ctrl+J` sends — the newline every CLI that cares already accepts, and the one
//! Claude Code documents as working in every terminal. It degrades perfectly:
//! readline binds `LF` to accept-line just like `CR`, so `Shift+Enter` still runs the
//! command at a shell prompt, and `vim` and `less` treat the two alike. It is the one
//! byte that is *distinguishable* to an application that looks, and *identical* to one
//! that does not.

use crate::grid::Screen;

/// A logical key press, already resolved by the layout. `Char` carries the
/// character the layout produced with Shift applied but *not* Ctrl or Alt: the
/// encoder applies those, because their transformation (a control byte, an ESC
/// prefix) is the terminal's job, not the layout's. The named keys are the ones
/// with dedicated escape sequences.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    /// A layout-produced character. `typed` is the character the key produced with
    /// Shift and the layout applied (`'A'`, `'@'`), and is what is sent as text.
    /// `base` is the same physical key with no modifiers at all (`'a'`, `'2'`).
    ///
    /// The CSI-u protocols name a key by its `base` codepoint plus a modifier
    /// bitmask, which is what keeps `Ctrl+Shift+2` distinct from `Ctrl+@` on a layout
    /// where they are the same physical key. The two characters are equal for most
    /// keys, which is what [`Key::plain`] is for.
    Char {
        typed: char,
        base: char,
    },
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

impl Key {
    /// A character key whose base is itself: the overwhelmingly common case (every
    /// unshifted key, and every letter, since a letter's base is just its lowercase).
    /// The app uses the full [`Key::Char`] form when xkb reports a different unshifted
    /// character for the key, which is how `Shift+2` keeps `'2'` as its identity.
    pub fn plain(c: char) -> Key {
        Key::Char { typed: c, base: c }
    }
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

    /// Whether *any* bit in `other` is set here. The question the CSI-u encodings ask:
    /// Ctrl, Alt or Super held means the key produced no text, so it needs a sequence.
    pub const fn any(self, other: Mods) -> bool {
        self.0 & other.0 != 0
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

/// The kitty keyboard protocol's progressive-enhancement flags, as an application
/// pushes them with `CSI > <flags> u`. A bitfield newtype, not a bare `u8`, so a
/// caller cannot pass the wrong number.
///
/// bnkterm implements [`DISAMBIGUATE`](Self::DISAMBIGUATE), the flag that matters
/// and the one every application starts with; the rest of the protocol's flags
/// (event types, alternate keys, all-keys-as-escape-codes, associated text) are
/// deliberately *not* accepted, and [`Screen::kitty_flags`] reports only what is
/// really in force. An application that asks for more and reads back less gets a
/// truthful answer and degrades; one that is told yes and served no key-release
/// events would sit there waiting for them.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct KittyFlags(u8);

impl KittyFlags {
    /// `0b1`: report the keys the legacy encoding cannot express (`Esc`, `Ctrl+key`,
    /// `Alt+key`, and any modified `Enter`/`Tab`/`Backspace`) as `CSI u` sequences.
    pub const DISAMBIGUATE: KittyFlags = KittyFlags(0b1);

    /// The flags bnkterm actually honours. An application's request is masked with
    /// this, so what we store is what we do.
    pub const SUPPORTED: KittyFlags = KittyFlags(0b1);

    pub const NONE: KittyFlags = KittyFlags(0);

    pub const fn contains(self, other: KittyFlags) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Build from an application's requested bits, dropping everything we do not
    /// implement.
    pub const fn from_request(bits: u16) -> KittyFlags {
        KittyFlags((bits as u8) & KittyFlags::SUPPORTED.0)
    }

    /// The bits, for the `CSI ? <flags> u` reply to a query.
    pub const fn bits(self) -> u32 {
        self.0 as u32
    }

    /// `CSI = <flags> ; <mode> u`: mode 1 replaces, 2 sets the given bits, 3 clears
    /// them. An unknown mode is ignored, as the protocol requires.
    pub fn apply(self, flags: KittyFlags, mode: u16) -> KittyFlags {
        match mode {
            1 => flags,
            2 => KittyFlags(self.0 | flags.0),
            3 => KittyFlags(self.0 & !flags.0),
            _ => self,
        }
    }
}

/// xterm's `modifyOtherKeys` level, set by `CSI > 4 ; <level> m` (XTMODKEYS).
///
/// The levels are xterm's own: level 1 encodes only the modified keys that have *no*
/// legacy encoding at all, leaving the well-known ones (`Ctrl+C` → `0x03`, `Tab`,
/// `Backspace`) alone; level 2 drops those exceptions too, which is what lets an
/// application tell `Ctrl+I` from `Tab` — and `Shift+Enter` from `Enter`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ModifyOtherKeys {
    /// The default: every key takes its legacy encoding.
    #[default]
    Off,
    /// Keys with no legacy encoding at all (`Ctrl+1`, `Super+x`) become `CSI 27 ~`.
    Level1,
    /// Every modified text key becomes `CSI 27 ~`, including the well-known ones.
    Level2,
}

impl ModifyOtherKeys {
    /// From the `Pv` of `CSI > 4 ; Pv m`. Levels above 2 are not defined; xterm treats
    /// an unknown value as a reset, and so do we.
    pub fn from_param(level: u16) -> ModifyOtherKeys {
        match level {
            1 => ModifyOtherKeys::Level1,
            2 => ModifyOtherKeys::Level2,
            _ => ModifyOtherKeys::Off,
        }
    }
}

/// The terminal modes that change what a key encodes to: the two DEC private modes
/// the child turns on and off, plus whichever keyboard protocol it has negotiated.
/// The app reads them off the [`Screen`] each key press so the encoding tracks the
/// program's current state (a `vim` in the alt screen, a shell at its prompt).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Modes {
    /// `DECCKM` (`?1`): arrows and Home/End send SS3 rather than CSI.
    pub app_cursor: bool,
    /// `DECKPAM`: the keypad sends its application-mode sequences.
    pub app_keypad: bool,
    /// The kitty keyboard flags currently on top of the application's stack.
    pub kitty: KittyFlags,
    /// The `modifyOtherKeys` level the application asked for.
    pub modify_other_keys: ModifyOtherKeys,
}

impl Modes {
    /// The input-relevant modes as the screen currently holds them.
    pub fn from_screen(screen: &Screen) -> Self {
        Modes {
            app_cursor: screen.app_cursor_keys(),
            app_keypad: screen.keypad_app(),
            kitty: screen.kitty_flags(),
            modify_other_keys: screen.modify_other_keys(),
        }
    }
}

/// Encode one key press into the bytes to write to the PTY, appended to `out`.
/// Nothing is appended for a key the terminal does not send (a lone modifier, an
/// out-of-range function key). `out` is a caller-owned buffer, reused across key
/// presses; keystrokes are human-paced, not a hot path, but reusing the buffer
/// keeps even that allocation-free.
///
/// Which of the three encodings applies is the application's choice, not ours, and
/// the precedence is: kitty's protocol if it pushed flags, else `modifyOtherKeys` if
/// it set a level, else legacy xterm. An application that turns on both (Claude Code
/// does, to cover terminals that speak only one) gets the kitty encoding, which is
/// the more precise of the two and the one it would have preferred.
///
/// The keys with their own escape sequences — the arrows, Home/End, the editing and
/// function keys — are *not* touched by either protocol. They were never ambiguous,
/// both specs leave them alone, and every application already parses them.
pub fn encode(key: Key, mods: Mods, modes: Modes, out: &mut Vec<u8>) {
    if modes.kitty.contains(KittyFlags::DISAMBIGUATE) {
        encode_kitty(key, mods, modes, out);
    } else if modes.modify_other_keys != ModifyOtherKeys::Off {
        encode_modify_other_keys(key, mods, modes, out);
    } else {
        encode_legacy(key, mods, modes, out);
    }
}

/// The legacy xterm encoding: what every application gets until it asks for better.
fn encode_legacy(key: Key, mods: Mods, modes: Modes, out: &mut Vec<u8>) {
    match key {
        Key::Char { typed, .. } => encode_char(typed, mods, out),
        Key::Enter => {
            // The one deliberate departure from xterm, which sends CR here and leaves
            // the application blind. See the module header: LF is distinguishable to
            // anything that looks and identical to anything that does not.
            if mods.contains(Mods::SHIFT) {
                alt_prefix(mods, out);
                out.push(b'\n');
            } else {
                alt_prefix(mods, out);
                out.push(b'\r');
            }
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
        _ => encode_named(key, mods, modes, out),
    }
}

/// The keys both protocols leave in their legacy form: the ones that already carry
/// their modifiers in a CSI parameter and were never ambiguous.
fn encode_named(key: Key, mods: Mods, modes: Modes, out: &mut Vec<u8>) {
    match key {
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
        // The text keys never reach here: each encoding handles them itself.
        Key::Char { .. } | Key::Enter | Key::Tab | Key::Backspace | Key::Escape => {}
    }
}

/// The kitty keyboard protocol under the disambiguate flag.
///
/// The rule, from the spec: every key that does not produce text is reported as
/// `CSI <code> ; <mods> u`, *except* that unmodified `Enter`, `Tab` and `Backspace`
/// keep their legacy control bytes (a program reading a line still wants a `\r`).
/// `Esc` is reported as `CSI 27u` even unmodified — telling the `Esc` key apart from
/// the `ESC` that opens an escape sequence is the whole reason the flag exists.
///
/// A key that *does* produce text still sends that text: `a` is `a`, and `Shift+a` is
/// `A`. Only Ctrl, Alt and Super (which produce no text) push a character key into the
/// `CSI u` form, where it is named by its `base` codepoint and the modifier bitmask.
fn encode_kitty(key: Key, mods: Mods, modes: Modes, out: &mut Vec<u8>) {
    match key {
        Key::Char { typed, base } => {
            if mods.any(Mods::CTRL | Mods::ALT | Mods::SUPER) {
                csi_u(base as u32, mods, out);
            } else {
                push_utf8(typed, out);
            }
        }
        Key::Enter | Key::Tab | Key::Backspace => {
            let code = text_key_code(key);
            if mods.0 == 0 {
                out.push(legacy_text_byte(key));
            } else {
                csi_u(code, mods, out);
            }
        }
        Key::Escape => csi_u(KEY_ESCAPE, mods, out),
        _ => encode_named(key, mods, modes, out),
    }
}

/// xterm's `modifyOtherKeys`: the same disambiguation as kitty's, in xterm's older
/// shape, `CSI 27 ; <mods> ; <code> ~`.
///
/// The two levels differ only in how much legacy they are willing to break. Level 1
/// keeps every well-known encoding (`Ctrl+C` is still `0x03`, `Shift+Enter` is still a
/// newline) and only rescues keys that legacy cannot express at all, such as `Ctrl+1`.
/// Level 2 gives up the exceptions, so `Ctrl+C` becomes `CSI 27;5;99~` and
/// `Shift+Enter` becomes `CSI 27;2;13~`. An application that asks for level 2 has said
/// it would rather parse keys than receive control bytes; tmux's `extended-keys on`
/// and Claude Code both ask for exactly that.
fn encode_modify_other_keys(key: Key, mods: Mods, modes: Modes, out: &mut Vec<u8>) {
    let level = modes.modify_other_keys;
    match key {
        Key::Char { typed, base } => {
            // Shift alone is not "other": it already produced its character.
            if !mods.any(Mods::CTRL | Mods::ALT | Mods::SUPER) {
                push_utf8(typed, out);
                return;
            }
            let well_known = mods.contains(Mods::CTRL) && ctrl_byte(typed).is_some()
                || mods.contains(Mods::ALT) && !mods.contains(Mods::SUPER);
            if level == ModifyOtherKeys::Level1 && well_known {
                encode_char(typed, mods, out);
            } else {
                csi_27(base as u32, mods, out);
            }
        }
        Key::Enter | Key::Tab | Key::Backspace | Key::Escape => {
            // Level 1 leaves these alone entirely: they are xterm's named exceptions.
            if mods.0 == 0 || level == ModifyOtherKeys::Level1 {
                encode_legacy(key, mods, modes, out);
            } else {
                csi_27(text_key_code(key), mods, out);
            }
        }
        _ => encode_named(key, mods, modes, out),
    }
}

/// The kitty protocol's code for a text key, which is just its ASCII value.
const KEY_ESCAPE: u32 = 27;

fn text_key_code(key: Key) -> u32 {
    match key {
        Key::Enter => 13,
        Key::Tab => 9,
        Key::Backspace => 127,
        Key::Escape => KEY_ESCAPE,
        _ => 0,
    }
}

/// The byte an unmodified text key sends in every encoding.
fn legacy_text_byte(key: Key) -> u8 {
    match key {
        Key::Enter => b'\r',
        Key::Tab => b'\t',
        Key::Backspace => 0x7f,
        Key::Escape => 0x1b,
        _ => 0,
    }
}

/// `CSI <code> ; <mods> u`, the kitty form. The modifier parameter is omitted when
/// nothing is held: it defaults to 1, and `CSI 27u` is what the spec shows.
fn csi_u(code: u32, mods: Mods, out: &mut Vec<u8>) {
    out.extend_from_slice(b"\x1b[");
    push_num(out, code);
    if mods.0 != 0 {
        out.push(b';');
        push_num(out, modifier_param(mods));
    }
    out.push(b'u');
}

/// `CSI 27 ; <mods> ; <code> ~`, the xterm `modifyOtherKeys` form. Both parameters are
/// always present here; xterm's parser wants the modifier in the middle.
fn csi_27(code: u32, mods: Mods, out: &mut Vec<u8>) {
    out.extend_from_slice(b"\x1b[27;");
    push_num(out, modifier_param(mods));
    out.push(b';');
    push_num(out, code);
    out.push(b'~');
}

/// Append `c` as UTF-8.
fn push_utf8(c: char, out: &mut Vec<u8>) {
    let mut buf = [0u8; 4];
    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
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
    push_utf8(c, out);
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
        assert_eq!(enc(Key::plain('a'), Mods::NONE), b"a");
        assert_eq!(enc(Key::plain('A'), Mods::NONE), b"A"); // Shift already in the char
        assert_eq!(enc(Key::plain('€'), Mods::NONE), "€".as_bytes());
    }

    #[test]
    fn ctrl_folds_letters_to_control_bytes() {
        assert_eq!(enc(Key::plain('a'), Mods::CTRL), vec![0x01]);
        assert_eq!(enc(Key::plain('c'), Mods::CTRL), vec![0x03]);
        assert_eq!(enc(Key::plain('z'), Mods::CTRL), vec![0x1a]);
        // The symbol block @.._ and the two specials.
        assert_eq!(enc(Key::plain('@'), Mods::CTRL), vec![0x00]);
        assert_eq!(enc(Key::plain('['), Mods::CTRL), vec![0x1b]);
        assert_eq!(enc(Key::plain('\\'), Mods::CTRL), vec![0x1c]);
        assert_eq!(enc(Key::plain(']'), Mods::CTRL), vec![0x1d]);
        assert_eq!(enc(Key::plain(' '), Mods::CTRL), vec![0x00]);
        assert_eq!(enc(Key::plain('?'), Mods::CTRL), vec![0x7f]);
    }

    #[test]
    fn ctrl_without_a_mapping_sends_the_plain_char() {
        // Ctrl+1 has no control byte on this path; the digit goes through.
        assert_eq!(enc(Key::plain('1'), Mods::CTRL), b"1");
    }

    #[test]
    fn alt_prefixes_escape() {
        assert_eq!(enc(Key::plain('a'), Mods::ALT), vec![0x1b, b'a']);
        // Alt+Ctrl+A is ESC then the control byte.
        assert_eq!(
            enc(Key::plain('a'), Mods::ALT | Mods::CTRL),
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
    fn shift_enter_is_lf_in_the_legacy_encoding() {
        // The whole point: an application that looks can tell this from Enter, and one
        // that does not sees a newline, which is what Enter meant to it anyway. xterm
        // sends CR here and leaves nobody able to tell the difference.
        assert_eq!(enc(Key::Enter, Mods::NONE), vec![b'\r']);
        assert_eq!(enc(Key::Enter, Mods::SHIFT), vec![b'\n']);
        // It is exactly the byte Ctrl+J sends, which is the newline every CLI accepts.
        assert_eq!(enc(Key::plain('j'), Mods::CTRL), vec![b'\n']);
        // Alt still prefixes ESC, so Alt+Shift+Enter stays composable.
        assert_eq!(enc(Key::Enter, Mods::SHIFT | Mods::ALT), vec![0x1b, b'\n']);
        assert_eq!(enc(Key::Enter, Mods::ALT), vec![0x1b, b'\r']);
    }

    /// Modes with the kitty protocol's disambiguate flag pushed, as an application does.
    fn kitty() -> Modes {
        Modes {
            kitty: KittyFlags::DISAMBIGUATE,
            ..Modes::default()
        }
    }

    /// Modes with xterm's modifyOtherKeys at the given level.
    fn mok(level: ModifyOtherKeys) -> Modes {
        Modes {
            modify_other_keys: level,
            ..Modes::default()
        }
    }

    #[test]
    fn kitty_reports_modified_enter_as_csi_u() {
        // The sequence that makes Shift+Enter work in kitty, ghostty and wezterm.
        assert_eq!(encoded(Key::Enter, Mods::SHIFT, kitty()), b"\x1b[13;2u");
        assert_eq!(encoded(Key::Enter, Mods::CTRL, kitty()), b"\x1b[13;5u");
        assert_eq!(encoded(Key::Enter, Mods::ALT, kitty()), b"\x1b[13;3u");
        assert_eq!(
            encoded(Key::Enter, Mods::CTRL | Mods::SHIFT, kitty()),
            b"\x1b[13;6u"
        );
        // Unmodified Enter/Tab/Backspace keep their control bytes: a program reading a
        // line still wants a CR, and the spec carves out exactly these three.
        assert_eq!(encoded(Key::Enter, Mods::NONE, kitty()), vec![b'\r']);
        assert_eq!(encoded(Key::Tab, Mods::NONE, kitty()), vec![b'\t']);
        assert_eq!(encoded(Key::Backspace, Mods::NONE, kitty()), vec![0x7f]);
        // Modified, they take the CSI u form (Tab is 9, Backspace 127).
        assert_eq!(encoded(Key::Tab, Mods::CTRL, kitty()), b"\x1b[9;5u");
        assert_eq!(encoded(Key::Backspace, Mods::CTRL, kitty()), b"\x1b[127;5u");
    }

    #[test]
    fn kitty_reports_escape_even_unmodified() {
        // Esc is the one key reported bare, and it is the reason the flag exists: an
        // application can finally tell the Esc key from the ESC that opens a sequence.
        assert_eq!(encoded(Key::Escape, Mods::NONE, kitty()), b"\x1b[27u");
        assert_eq!(encoded(Key::Escape, Mods::SHIFT, kitty()), b"\x1b[27;2u");
    }

    #[test]
    fn kitty_sends_text_as_text_and_chords_as_csi_u() {
        // A key that produces text still produces text: shift is not a chord.
        assert_eq!(encoded(Key::plain('a'), Mods::NONE, kitty()), b"a");
        assert_eq!(encoded(Key::plain('A'), Mods::SHIFT, kitty()), b"A");
        assert_eq!(
            encoded(Key::plain('€'), Mods::NONE, kitty()),
            "€".as_bytes()
        );
        // Ctrl/Alt/Super produce no text, so they name the key instead of folding it
        // into a control byte: Ctrl+C is the key `c` (99) with the ctrl bit.
        assert_eq!(encoded(Key::plain('c'), Mods::CTRL, kitty()), b"\x1b[99;5u");
        assert_eq!(encoded(Key::plain('a'), Mods::ALT, kitty()), b"\x1b[97;3u");
        assert_eq!(
            encoded(Key::plain('a'), Mods::CTRL | Mods::ALT, kitty()),
            b"\x1b[97;7u"
        );
        // Ctrl+I is finally distinct from Tab, which is the whole promise of the flag.
        assert_eq!(
            encoded(Key::plain('i'), Mods::CTRL, kitty()),
            b"\x1b[105;5u"
        );
        assert_eq!(encoded(Key::Tab, Mods::NONE, kitty()), vec![b'\t']);
    }

    #[test]
    fn a_shifted_key_is_named_by_its_base() {
        // Ctrl+Shift+2 on a US layout types '@' but is the `2` key. The protocols report
        // the key, not the character, so an application can bind it without knowing the
        // user's layout. Getting this wrong by lowercasing the typed char would send 64.
        let shift_two = Key::Char {
            typed: '@',
            base: '2',
        };
        assert_eq!(
            encoded(shift_two, Mods::CTRL | Mods::SHIFT, kitty()),
            b"\x1b[50;6u" // 50 is '2', modifier 6 is ctrl+shift
        );
        // A letter's base is its lowercase, so Ctrl+Shift+A names `a` (97), not `A`.
        let shift_a = Key::Char {
            typed: 'A',
            base: 'a',
        };
        assert_eq!(
            encoded(shift_a, Mods::CTRL | Mods::SHIFT, kitty()),
            b"\x1b[97;6u"
        );
    }

    #[test]
    fn kitty_leaves_the_keys_that_were_never_ambiguous_alone() {
        // Arrows, editing and function keys already carry their modifiers in a CSI
        // parameter. Both specs leave them exactly as they were.
        assert_eq!(encoded(Key::Up, Mods::NONE, kitty()), b"\x1b[A");
        assert_eq!(encoded(Key::Up, Mods::CTRL, kitty()), b"\x1b[1;5A");
        assert_eq!(encoded(Key::Delete, Mods::SHIFT, kitty()), b"\x1b[3;2~");
        assert_eq!(encoded(Key::Function(5), Mods::NONE, kitty()), b"\x1b[15~");
    }

    #[test]
    fn modify_other_keys_level_2_is_the_csi_27_form() {
        let m = mok(ModifyOtherKeys::Level2);
        // The same disambiguation as kitty's, in xterm's older shape.
        assert_eq!(encoded(Key::Enter, Mods::SHIFT, m), b"\x1b[27;2;13~");
        assert_eq!(encoded(Key::Tab, Mods::CTRL, m), b"\x1b[27;5;9~");
        assert_eq!(encoded(Key::Escape, Mods::CTRL, m), b"\x1b[27;5;27~");
        // Level 2 gives up the well-known control bytes too: that is what makes Ctrl+I
        // distinguishable from Tab, and it is what an application asking for 2 wants.
        assert_eq!(encoded(Key::plain('c'), Mods::CTRL, m), b"\x1b[27;5;99~");
        // Unmodified keys are untouched, always.
        assert_eq!(encoded(Key::Enter, Mods::NONE, m), vec![b'\r']);
        assert_eq!(encoded(Key::plain('c'), Mods::NONE, m), b"c");
        assert_eq!(encoded(Key::plain('C'), Mods::SHIFT, m), b"C");
    }

    #[test]
    fn modify_other_keys_level_1_keeps_the_well_known_encodings() {
        let m = mok(ModifyOtherKeys::Level1);
        // Level 1 rescues only what legacy cannot express at all: Ctrl+1 has no control
        // byte, so it gets a sequence...
        assert_eq!(encoded(Key::plain('1'), Mods::CTRL, m), b"\x1b[27;5;49~");
        // ...while everything with a well-known encoding keeps it.
        assert_eq!(encoded(Key::plain('c'), Mods::CTRL, m), vec![0x03]);
        assert_eq!(encoded(Key::plain('a'), Mods::ALT, m), vec![0x1b, b'a']);
        assert_eq!(encoded(Key::Tab, Mods::CTRL, m), vec![b'\t']);
        assert_eq!(encoded(Key::Enter, Mods::SHIFT, m), vec![b'\n']);
    }

    #[test]
    fn kitty_outranks_modify_other_keys() {
        // An application that turns on both (Claude Code does, to cover terminals that
        // speak only one) gets the more precise of the two.
        let both = Modes {
            kitty: KittyFlags::DISAMBIGUATE,
            modify_other_keys: ModifyOtherKeys::Level2,
            ..Modes::default()
        };
        assert_eq!(encoded(Key::Enter, Mods::SHIFT, both), b"\x1b[13;2u");
    }

    #[test]
    fn an_empty_kitty_push_is_still_the_legacy_encoding() {
        // `CSI > u` pushes zero flags, which asks for nothing. The disambiguate bit is
        // what turns the protocol on, not the mere presence of a stack entry.
        let pushed_nothing = Modes {
            kitty: KittyFlags::NONE,
            ..Modes::default()
        };
        assert_eq!(
            encoded(Key::Enter, Mods::SHIFT, pushed_nothing),
            vec![b'\n']
        );
        assert_eq!(encoded(Key::Escape, Mods::NONE, pushed_nothing), vec![0x1b]);
    }

    #[test]
    fn a_program_negotiating_over_the_pty_changes_what_its_keys_encode_to() {
        // The whole loop, end to end and with nothing stubbed: bytes from the child go
        // through the real parser into the real screen, and the encoder reads its modes
        // back off that screen. Each piece is tested alone above; this is the only test
        // that proves they are actually wired to each other.
        let mut screen = Screen::new(80, 24);
        let mut parser = crate::vt::Parser::new();
        let mut feed = |screen: &mut Screen, bytes: &[u8]| parser.advance_bytes(screen, bytes);

        // A shell, having asked for nothing: Shift+Enter is the legacy LF.
        let modes = Modes::from_screen(&screen);
        assert_eq!(encoded(Key::Enter, Mods::SHIFT, modes), vec![b'\n']);

        // A full-screen program starts and pushes the kitty flags (this is byte-for-byte
        // what neovim and Claude Code send). Now the same key press is a CSI u sequence.
        feed(&mut screen, b"\x1b[>1u");
        let modes = Modes::from_screen(&screen);
        assert_eq!(encoded(Key::Enter, Mods::SHIFT, modes), b"\x1b[13;2u");
        assert_eq!(encoded(Key::Escape, Mods::NONE, modes), b"\x1b[27u");

        // It exits and pops. The shell underneath must not be left receiving CSI u for
        // keys it does not parse, which is exactly what the stack is for.
        feed(&mut screen, b"\x1b[<u");
        let modes = Modes::from_screen(&screen);
        assert_eq!(encoded(Key::Enter, Mods::SHIFT, modes), vec![b'\n']);
        assert_eq!(encoded(Key::Escape, Mods::NONE, modes), vec![0x1b]);

        // A program that speaks only xterm's older protocol gets the older shape.
        feed(&mut screen, b"\x1b[>4;2m");
        let modes = Modes::from_screen(&screen);
        assert_eq!(encoded(Key::Enter, Mods::SHIFT, modes), b"\x1b[27;2;13~");
    }

    #[test]
    fn the_handshake_neovim_actually_performs() {
        // Not an invented sequence: these are the bytes neovim sends, captured from it
        // running under a pty. It *queries* first and pushes nothing unless the terminal
        // answers, so a terminal that ignores `CSI ? u` never gets the protocol at all —
        // which is why the reply is load-bearing and not merely polite.
        let mut screen = Screen::new(80, 24);
        let mut parser = crate::vt::Parser::new();

        parser.advance_bytes(&mut screen, b"\x1b[?u");
        assert_eq!(
            screen.take_responses(),
            b"\x1b[?0u",
            "the answer nvim waits for"
        );

        // Having heard back, it pushes flags 3: disambiguate (1) plus report-event-types
        // (2). We implement the first and not the second, so we take the bit we honour
        // and leave the other off rather than promising key-release events we never send.
        parser.advance_bytes(&mut screen, b"\x1b[>3u");
        assert_eq!(screen.kitty_flags(), KittyFlags::DISAMBIGUATE);
        parser.advance_bytes(&mut screen, b"\x1b[?u");
        assert_eq!(
            screen.take_responses(),
            b"\x1b[?1u",
            "and we say so, truthfully"
        );

        // The payoff: nvim can now tell these apart, and every one of them was a plain
        // CR, TAB or ESC before.
        let modes = Modes::from_screen(&screen);
        assert_eq!(encoded(Key::Enter, Mods::SHIFT, modes), b"\x1b[13;2u");
        assert_eq!(encoded(Key::Enter, Mods::CTRL, modes), b"\x1b[13;5u");
        assert_eq!(encoded(Key::plain('i'), Mods::CTRL, modes), b"\x1b[105;5u");
        assert_eq!(encoded(Key::Escape, Mods::NONE, modes), b"\x1b[27u");
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
        encode(Key::plain('h'), Mods::NONE, Modes::default(), &mut out);
        encode(Key::plain('i'), Mods::NONE, Modes::default(), &mut out);
        encode(Key::Enter, Mods::NONE, Modes::default(), &mut out);
        assert_eq!(out, b"hi\r");
    }
}
