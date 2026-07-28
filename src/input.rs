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
    /// A keypad key in its *numeric* role, which under `DECKPAM` gets its own `SS3`
    /// form. See [`KeypadKey`].
    Keypad(KeypadKey),
}

/// A keypad key, named by what is printed on it.
///
/// These are only half the keypad's story, and the half NumLock is *on* for. With
/// NumLock off the layout resolves the very same physical keys to navigation keysyms
/// (`KP_Left`, `KP_Home`, `KP_Next`, …), and they are then genuinely the arrows and
/// editing keys — so they resolve to [`Key::Left`] and friends and never reach here.
/// Nothing in this module has to know about NumLock: xkb applies it before the keysym
/// is ever looked up, exactly as X does for xterm.
///
/// Under `DECKPNM` (the default) each of these sends the character printed on the key,
/// which is why an unmapped keypad has always *seemed* to work — until a program sets
/// `DECKPAM` and expects `SS3` back. Every ncurses program calling `keypad(true)` does:
/// `smkx` for `xterm-256color` is `\E[?1h\E=`, and that `\E=` is `DECKPAM`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeypadKey {
    /// `0`-`9`, held as the digit itself (always `0..=9`; nothing else constructs one).
    Digit(u8),
    /// `.` — or `,` on the layouts that print one there, which is a different keysym
    /// and a different application-mode final byte.
    Decimal,
    Separator,
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    /// The centre key (`5`) with NumLock off, which X names `KP_Begin`. It is the one
    /// keypad key whose NumLock-off role is not an editing key, so it stays here.
    Begin,
}

impl KeypadKey {
    /// The character printed on the key, sent verbatim under `DECKPNM`. `None` for
    /// [`KeypadKey::Begin`], which has no numeric role of its own.
    fn glyph(self) -> Option<char> {
        Some(match self {
            KeypadKey::Digit(n) => char::from(b'0'.saturating_add(n.min(9))),
            KeypadKey::Decimal => '.',
            KeypadKey::Separator => ',',
            KeypadKey::Add => '+',
            KeypadKey::Subtract => '-',
            KeypadKey::Multiply => '*',
            KeypadKey::Divide => '/',
            KeypadKey::Equal => '=',
            KeypadKey::Begin => return None,
        })
    }

    /// The final byte of this key's `SS3` form under `DECKPAM`, from xterm's keypad
    /// table. The digits run `0`→`p` … `9`→`y`, which is what makes terminfo's
    /// `ka1`/`ka3`/`kb2`/`kc1`/`kc3` (`\EOw`, `\EOy`, `\EOu`, `\EOq`, `\EOs`) name the
    /// keypad's four corners and its centre.
    fn application_final(self) -> u8 {
        match self {
            KeypadKey::Digit(n) => b'p'.saturating_add(n.min(9)),
            KeypadKey::Multiply => b'j',
            KeypadKey::Add => b'k',
            KeypadKey::Separator => b'l',
            KeypadKey::Subtract => b'm',
            KeypadKey::Decimal => b'n',
            KeypadKey::Divide => b'o',
            KeypadKey::Equal => b'X',
            KeypadKey::Begin => b'E',
        }
    }
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

/// What happened to a key. The kitty protocol reports all three when asked; the legacy
/// encoding has no way to say anything but "pressed", and a repeat is indistinguishable
/// from someone typing fast.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum KeyEvent {
    #[default]
    Press,
    /// The key is held and the keyboard is repeating it. Without the protocol this is a
    /// press, which is exactly what a repeat has always looked like to a terminal.
    Repeat,
    Release,
}

impl KeyEvent {
    /// The protocol's number for the event, for the `modifiers:event` sub-field.
    fn code(self) -> u32 {
        match self {
            KeyEvent::Press => 1,
            KeyEvent::Repeat => 2,
            KeyEvent::Release => 3,
        }
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

    /// `0b10`: report key *release* and *repeat*, not only press. A program tracking a
    /// held key (a game, a modal editor, anything with a chord) cannot do it otherwise:
    /// the legacy encoding has no way to say a key stopped being held.
    pub const REPORT_EVENT_TYPES: KittyFlags = KittyFlags(0b10);

    /// The flags bnkterm actually honours. An application's request is masked with
    /// this, so what we store is what we do.
    pub const SUPPORTED: KittyFlags = KittyFlags(0b11);

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

impl std::ops::BitOr for KittyFlags {
    type Output = KittyFlags;
    fn bitor(self, rhs: KittyFlags) -> KittyFlags {
        KittyFlags(self.0 | rhs.0)
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
    encode_event(key, mods, KeyEvent::Press, modes, out);
}

/// Encode a key *event*, which is a press unless the application asked to hear about the
/// others (kitty's `REPORT_EVENT_TYPES`).
///
/// Two rules keep a mis-behaving program from wedging the terminal, and both come
/// straight from the protocol:
///
/// - A **release** is reported only for keys that are reported as escape codes in the
///   first place. A key that produces text produces text; there is no text for "the `a`
///   key came back up", so nothing is sent.
/// - **`Enter`, `Tab` and `Backspace` never report a release**, even under the flag. The
///   spec spells out why, and it is a good reason: a program that turns this mode on and
///   dies without turning it off would otherwise leave the shell underneath receiving an
///   escape sequence every time you let go of `Enter` — and you could no longer type
///   `reset` to fix it.
///
/// A **repeat** with the flag off is a press, which is exactly what a repeat has always
/// looked like to a terminal.
pub fn encode_event(key: Key, mods: Mods, event: KeyEvent, modes: Modes, out: &mut Vec<u8>) {
    let kitty = modes.kitty.contains(KittyFlags::DISAMBIGUATE);
    let events = kitty && modes.kitty.contains(KittyFlags::REPORT_EVENT_TYPES);

    if !events {
        // Nobody is listening for anything but presses. A repeat is one; a release is
        // nothing at all.
        match event {
            KeyEvent::Release => return,
            KeyEvent::Press | KeyEvent::Repeat => {}
        }
        if kitty {
            return encode_kitty(key, mods, KeyEvent::Press, modes, out);
        } else if modes.modify_other_keys != ModifyOtherKeys::Off {
            return encode_modify_other_keys(key, mods, modes, out);
        }
        return encode_legacy(key, mods, modes, out);
    }

    if event == KeyEvent::Release && !reports_release(key, mods) {
        return;
    }
    encode_kitty(key, mods, event, modes, out);
}

/// Whether this key reports a release at all under `REPORT_EVENT_TYPES`. See
/// [`encode_event`] for the two rules; this is where they live.
fn reports_release(key: Key, mods: Mods) -> bool {
    match key {
        // A text key's release has no text to be, and no escape code either.
        Key::Char { .. } => mods.any(Mods::CTRL | Mods::ALT | Mods::SUPER),
        // The spec's carve-out, so `reset` stays typable after a program leaves the mode
        // on and exits.
        Key::Enter | Key::Tab | Key::Backspace => false,
        // The keypad's sequences are xterm's `SS3` forms, which have no parameters and
        // so nowhere to put an event type. Reporting a release would re-send the press
        // byte for byte, which is worse than saying nothing: a program counting keypad
        // presses would see two for every one.
        Key::KeypadEnter | Key::Keypad(_) => false,
        _ => true,
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
    encode_named_event(key, mods, KeyEvent::Press, modes, out)
}

/// The same, with an event type folded into the modifier parameter. Under the protocol a
/// functional key keeps its legacy shape and gains a `:event` sub-field — so an arrow
/// release is `CSI 1;1:3A`, still recognisably an arrow.
fn encode_named_event(key: Key, mods: Mods, event: KeyEvent, modes: Modes, out: &mut Vec<u8>) {
    match key {
        Key::Up => cursor_key(b'A', mods, event, modes, out),
        Key::Down => cursor_key(b'B', mods, event, modes, out),
        Key::Right => cursor_key(b'C', mods, event, modes, out),
        Key::Left => cursor_key(b'D', mods, event, modes, out),
        Key::Home => cursor_key(b'H', mods, event, modes, out),
        Key::End => cursor_key(b'F', mods, event, modes, out),
        Key::Insert => tilde_key(2, mods, event, out),
        Key::Delete => tilde_key(3, mods, event, out),
        Key::PageUp => tilde_key(5, mods, event, out),
        Key::PageDown => tilde_key(6, mods, event, out),
        Key::Function(n) => function_key(n, mods, event, out),
        Key::KeypadEnter => {
            if modes.app_keypad {
                out.extend_from_slice(b"\x1bOM");
            } else {
                alt_prefix(mods, out);
                out.push(b'\r');
            }
        }
        Key::Keypad(k) => keypad_key(k, mods, modes, out),
        // The text keys never reach here: each encoding handles them itself.
        Key::Char { .. } | Key::Enter | Key::Tab | Key::Backspace | Key::Escape => {}
    }
}

/// A keypad key under whichever keypad mode is in force.
///
/// `DECKPAM` (application) gives every key its own `SS3` form, which is the whole point
/// of the mode: an application can then tell keypad `7` from the `7` on the number row.
/// `DECKPNM` (numeric, the default) sends the character printed on the key, through the
/// ordinary text encoder so Ctrl and Alt fold exactly as they do everywhere else — a
/// keypad key in numeric mode really is just that character.
///
/// The modifier parameter has no place in either form. xterm's `SS3` keypad sequences
/// have no parameters at all, so a modified keypad key in application mode sends the
/// bare sequence; a program that wants modified keypad keys asks for a CSI-u protocol,
/// where this function is not reached.
fn keypad_key(key: KeypadKey, mods: Mods, modes: Modes, out: &mut Vec<u8>) {
    if modes.app_keypad {
        out.extend_from_slice(b"\x1bO");
        out.push(key.application_final());
        return;
    }
    match key.glyph() {
        Some(c) => encode_char(c, mods, out),
        // `Begin` is the centre key with NumLock *off*, so it has no character to be.
        // xterm sends the cursor-key form, the same `CSI E` the other four corners of
        // the cluster take in their navigation role.
        None => {
            alt_prefix(mods, out);
            out.extend_from_slice(b"\x1b[E");
        }
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
fn encode_kitty(key: Key, mods: Mods, event: KeyEvent, modes: Modes, out: &mut Vec<u8>) {
    match key {
        Key::Char { typed, base } => {
            if mods.any(Mods::CTRL | Mods::ALT | Mods::SUPER) {
                csi_u(base as u32, mods, event, out);
            } else if event != KeyEvent::Release {
                // Text on press, and on repeat: a held key typing again is what a repeat
                // has always meant. A release has no text to be, and `reports_release`
                // has already turned it away.
                push_utf8(typed, out);
            }
        }
        Key::Enter | Key::Tab | Key::Backspace => {
            let code = text_key_code(key);
            if mods.0 == 0 && event == KeyEvent::Press {
                out.push(legacy_text_byte(key));
            } else {
                csi_u(code, mods, event, out);
            }
        }
        Key::Escape => csi_u(KEY_ESCAPE, mods, event, out),
        _ => encode_named_event(key, mods, event, modes, out),
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
fn csi_u(code: u32, mods: Mods, event: KeyEvent, out: &mut Vec<u8>) {
    out.extend_from_slice(b"\x1b[");
    push_num(out, code);
    push_modifier(mods, event, out);
    out.push(b'u');
}

/// The modifier parameter, and the event type riding on it as a colon sub-field.
///
/// Omitted entirely when there is nothing to say — no modifiers, a plain press — because
/// the defaults are exactly that and `CSI 27u` is what the spec shows. But an event type
/// cannot be sent without the modifier in front of it, so a release with no modifiers
/// still has to spell out the default: `CSI 27;1:3u`.
fn push_modifier(mods: Mods, event: KeyEvent, out: &mut Vec<u8>) {
    if mods.0 == 0 && event == KeyEvent::Press {
        return;
    }
    out.push(b';');
    push_num(out, modifier_param(mods));
    if event != KeyEvent::Press {
        out.push(b':');
        push_num(out, event.code());
    }
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
fn cursor_key(final_byte: u8, mods: Mods, event: KeyEvent, modes: Modes, out: &mut Vec<u8>) {
    if mods.0 == 0 && event == KeyEvent::Press {
        out.push(0x1b);
        out.push(if modes.app_cursor { b'O' } else { b'[' });
        out.push(final_byte);
    } else {
        out.extend_from_slice(b"\x1b[1");
        push_modifier(mods, event, out);
        out.push(final_byte);
    }
}

/// An Insert/Delete/Page key: `CSI <n> ~`, or `CSI <n> ; <param> ~` when
/// modified. These do not vary with the cursor-key mode.
fn tilde_key(n: u32, mods: Mods, event: KeyEvent, out: &mut Vec<u8>) {
    out.extend_from_slice(b"\x1b[");
    push_num(out, n);
    push_modifier(mods, event, out);
    out.push(b'~');
}

/// A function key. F1-F4 are `SS3 P..S` (or `CSI 1 ; <param> P..S` when
/// modified); F5-F12 are tilde codes (`CSI 15 ~` .. `CSI 24 ~`). Keys outside
/// 1-12 produce nothing.
fn function_key(n: u8, mods: Mods, event: KeyEvent, out: &mut Vec<u8>) {
    match n {
        1..=4 => {
            let final_byte = b'P' + (n - 1);
            if mods.0 == 0 && event == KeyEvent::Press {
                out.push(0x1b);
                out.push(b'O');
                out.push(final_byte);
            } else {
                out.extend_from_slice(b"\x1b[1");
                push_modifier(mods, event, out);
                out.push(final_byte);
            }
        }
        5..=12 => {
            // The xterm tilde codes for F5..F12, with their two gaps.
            const CODES: [u32; 8] = [15, 17, 18, 19, 20, 21, 23, 24];
            let Some(&code) = CODES.get((n - 5) as usize) else {
                return;
            };
            tilde_key(code, mods, event, out);
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

/// Map an X keysym to a named [`Key`], for the keys a *keycode* cannot identify.
///
/// The keypad is the whole reason this exists, and NumLock is the reason it cannot be
/// done from the keycode. One physical key has two jobs — keypad `4` and Left arrow —
/// and which one it is depends on a modifier, so the physical code is not enough to
/// name it. The layout resolves that (`KP_4` vs `KP_Left`) before anything here runs,
/// which is exactly how xterm gets it right: X hands xterm a keysym, not a scancode.
///
/// So the NumLock-off half maps onto the ordinary named keys — keypad `4` genuinely
/// *is* Left, and sends what Left sends, `DECCKM` and all — while the NumLock-on half
/// becomes [`Key::Keypad`], which is where `DECKPAM` applies.
///
/// `None` for every keysym outside the keypad, including all ordinary text keys: those
/// resolve through the layout's character instead. `KP_Enter` is also absent, and
/// deliberately: it means the same thing under either NumLock state, so
/// [`key_from_keycode`] names it from the physical code and gets it right even before a
/// keymap has arrived.
pub fn key_from_keysym(sym: u32) -> Option<Key> {
    use keysym as k;
    Some(match sym {
        // NumLock off: the layout has already decided these are editing keys.
        k::KP_LEFT => Key::Left,
        k::KP_RIGHT => Key::Right,
        k::KP_UP => Key::Up,
        k::KP_DOWN => Key::Down,
        k::KP_HOME => Key::Home,
        k::KP_END => Key::End,
        k::KP_PRIOR => Key::PageUp,
        k::KP_NEXT => Key::PageDown,
        k::KP_INSERT => Key::Insert,
        k::KP_DELETE => Key::Delete,
        k::KP_BEGIN => Key::Keypad(KeypadKey::Begin),
        // NumLock on: the numeric keypad proper.
        k::KP_0..=k::KP_9 => {
            let n = u8::try_from(sym - k::KP_0).unwrap_or(0);
            Key::Keypad(KeypadKey::Digit(n))
        }
        k::KP_DECIMAL => Key::Keypad(KeypadKey::Decimal),
        k::KP_SEPARATOR => Key::Keypad(KeypadKey::Separator),
        k::KP_ADD => Key::Keypad(KeypadKey::Add),
        k::KP_SUBTRACT => Key::Keypad(KeypadKey::Subtract),
        k::KP_MULTIPLY => Key::Keypad(KeypadKey::Multiply),
        k::KP_DIVIDE => Key::Keypad(KeypadKey::Divide),
        k::KP_EQUAL => Key::Keypad(KeypadKey::Equal),
        _ => return None,
    })
}

/// The X keysyms this encoder names, from `X11/keysymdef.h`. Only the keypad: every
/// other key is identified by its evdev code (a named key) or by the character the
/// layout produces (text), and neither needs a keysym.
mod keysym {
    pub const KP_HOME: u32 = 0xff95;
    pub const KP_LEFT: u32 = 0xff96;
    pub const KP_UP: u32 = 0xff97;
    pub const KP_RIGHT: u32 = 0xff98;
    pub const KP_DOWN: u32 = 0xff99;
    pub const KP_PRIOR: u32 = 0xff9a;
    pub const KP_NEXT: u32 = 0xff9b;
    pub const KP_END: u32 = 0xff9c;
    pub const KP_BEGIN: u32 = 0xff9d;
    pub const KP_INSERT: u32 = 0xff9e;
    pub const KP_DELETE: u32 = 0xff9f;
    pub const KP_MULTIPLY: u32 = 0xffaa;
    pub const KP_ADD: u32 = 0xffab;
    pub const KP_SEPARATOR: u32 = 0xffac;
    pub const KP_SUBTRACT: u32 = 0xffad;
    pub const KP_DECIMAL: u32 = 0xffae;
    pub const KP_DIVIDE: u32 = 0xffaf;
    pub const KP_0: u32 = 0xffb0;
    pub const KP_9: u32 = 0xffb9;
    pub const KP_EQUAL: u32 = 0xffbd;
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

    /// Modes with both kitty flags: disambiguate, and report event types. This is what
    /// nvim pushes (`CSI > 3 u`) and what helix asks for.
    fn kitty_events() -> Modes {
        Modes {
            kitty: KittyFlags::DISAMBIGUATE | KittyFlags::REPORT_EVENT_TYPES,
            ..Modes::default()
        }
    }

    fn enc_event(key: Key, mods: Mods, event: KeyEvent, modes: Modes) -> Vec<u8> {
        let mut out = Vec::new();
        encode_event(key, mods, event, modes, &mut out);
        out
    }

    #[test]
    fn nobody_hears_a_key_come_up_unless_they_asked() {
        // A release is silent in every encoding but the one that asked for it. The legacy
        // encoding has no way to say a key stopped being held, and inventing one would put
        // bytes into every shell on the machine.
        for modes in [Modes::default(), kitty(), mok(ModifyOtherKeys::Level2)] {
            assert!(enc_event(Key::plain('a'), Mods::NONE, KeyEvent::Release, modes).is_empty());
            assert!(enc_event(Key::Up, Mods::NONE, KeyEvent::Release, modes).is_empty());
            assert!(enc_event(Key::Escape, Mods::NONE, KeyEvent::Release, modes).is_empty());
        }
        // And a repeat, to anyone not listening for one, is just a press — which is
        // exactly what a repeat has always looked like to a terminal.
        assert_eq!(
            enc_event(
                Key::plain('a'),
                Mods::NONE,
                KeyEvent::Repeat,
                Modes::default()
            ),
            b"a"
        );
        assert_eq!(
            enc_event(Key::Up, Mods::NONE, KeyEvent::Repeat, kitty()),
            b"\x1b[A"
        );
    }

    #[test]
    fn event_types_ride_on_the_modifier_as_a_sub_field() {
        let m = kitty_events();
        // Press is the default and says nothing extra.
        assert_eq!(
            enc_event(Key::Escape, Mods::NONE, KeyEvent::Press, m),
            b"\x1b[27u"
        );
        // A release cannot be sent without the modifier in front of it, so even with no
        // modifiers held it spells out the default: `1`.
        assert_eq!(
            enc_event(Key::Escape, Mods::NONE, KeyEvent::Release, m),
            b"\x1b[27;1:3u"
        );
        assert_eq!(
            enc_event(Key::Escape, Mods::NONE, KeyEvent::Repeat, m),
            b"\x1b[27;1:2u"
        );
        // With modifiers, the event rides alongside them.
        assert_eq!(
            enc_event(Key::plain('c'), Mods::CTRL, KeyEvent::Release, m),
            b"\x1b[99;5:3u"
        );
    }

    #[test]
    fn a_functional_key_keeps_its_shape_and_gains_an_event() {
        // An arrow release is still recognisably an arrow: the event is a sub-field of the
        // modifier parameter, not a different sequence. Claiming the flag and then
        // reporting releases for `Ctrl+C` but not for the arrow keys would be a half-truth
        // of exactly the kind this codebase keeps refusing to tell.
        let m = kitty_events();
        assert_eq!(
            enc_event(Key::Up, Mods::NONE, KeyEvent::Press, m),
            b"\x1b[A"
        );
        assert_eq!(
            enc_event(Key::Up, Mods::NONE, KeyEvent::Release, m),
            b"\x1b[1;1:3A"
        );
        assert_eq!(
            enc_event(Key::Up, Mods::CTRL, KeyEvent::Release, m),
            b"\x1b[1;5:3A"
        );
        assert_eq!(
            enc_event(Key::Delete, Mods::NONE, KeyEvent::Release, m),
            b"\x1b[3;1:3~"
        );
        assert_eq!(
            enc_event(Key::Function(5), Mods::NONE, KeyEvent::Release, m),
            b"\x1b[15;1:3~"
        );
        assert_eq!(
            enc_event(Key::Function(1), Mods::NONE, KeyEvent::Release, m),
            b"\x1b[1;1:3P"
        );
    }

    #[test]
    fn a_text_key_has_no_release_to_report() {
        // `a` produces text. There is no text for "the `a` key came back up", and no
        // escape code either, so nothing is sent — the key was never reported as a
        // sequence in the first place.
        let m = kitty_events();
        assert_eq!(
            enc_event(Key::plain('a'), Mods::NONE, KeyEvent::Press, m),
            b"a"
        );
        assert!(enc_event(Key::plain('a'), Mods::NONE, KeyEvent::Release, m).is_empty());
        // A held key typing again is a repeat, and a repeat of a text key is the text.
        assert_eq!(
            enc_event(Key::plain('a'), Mods::NONE, KeyEvent::Repeat, m),
            b"a"
        );
        // But hold Ctrl and it *is* reported as a sequence, so its release is too.
        assert_eq!(
            enc_event(Key::plain('a'), Mods::CTRL, KeyEvent::Release, m),
            b"\x1b[97;5:3u"
        );
    }

    #[test]
    fn enter_tab_and_backspace_never_report_a_release() {
        // The spec's carve-out, and it is a good one. A program that turns this mode on
        // and dies without turning it off would otherwise leave the shell underneath
        // receiving an escape sequence every time you let go of Enter — and you could no
        // longer type `reset` to fix it. The keys you need to repair a broken terminal are
        // the keys that stay boring.
        let m = kitty_events();
        for key in [Key::Enter, Key::Tab, Key::Backspace] {
            assert!(
                enc_event(key, Mods::NONE, KeyEvent::Release, m).is_empty(),
                "{key:?} must not report a release"
            );
            assert!(
                enc_event(key, Mods::CTRL, KeyEvent::Release, m).is_empty(),
                "{key:?} must not report a release even modified"
            );
        }
        // They still report presses, and still take their modifiers.
        assert_eq!(enc_event(Key::Enter, Mods::NONE, KeyEvent::Press, m), b"\r");
        assert_eq!(
            enc_event(Key::Enter, Mods::SHIFT, KeyEvent::Press, m),
            b"\x1b[13;2u"
        );
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
        // (2). We implement both, so it gets both — and the reply says 3, because the
        // reply has only ever been allowed to say what is true. The masking machinery is
        // still there and still honest; it simply has nothing left to take away.
        parser.advance_bytes(&mut screen, b"\x1b[>3u");
        assert_eq!(
            screen.kitty_flags(),
            KittyFlags::DISAMBIGUATE | KittyFlags::REPORT_EVENT_TYPES
        );
        parser.advance_bytes(&mut screen, b"\x1b[?u");
        assert_eq!(
            screen.take_responses(),
            b"\x1b[?3u",
            "and we say so, truthfully"
        );

        // The payoff: nvim can now tell these apart, and every one of them was a plain
        // CR, TAB or ESC before.
        let modes = Modes::from_screen(&screen);
        assert_eq!(encoded(Key::Enter, Mods::SHIFT, modes), b"\x1b[13;2u");
        assert_eq!(encoded(Key::Enter, Mods::CTRL, modes), b"\x1b[13;5u");
        assert_eq!(encoded(Key::plain('i'), Mods::CTRL, modes), b"\x1b[105;5u");
        // And it hears the key come back up, which it asked for and could not have had.
        assert_eq!(
            enc_event(Key::plain('i'), Mods::CTRL, KeyEvent::Release, modes),
            b"\x1b[105;5:3u"
        );
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

    /// The whole keypad, in both NumLock states and both keypad modes, against the byte
    /// sequences terminfo names.
    ///
    /// Exhaustive because it is a table, and because every single one of these used to
    /// be wrong in one of two ways: with NumLock off the cluster sent **nothing at all**
    /// (`xkb_keysym_to_utf32(XK_KP_Left)` is 0, so the key resolved to no key), and with
    /// NumLock on it sent the bare digit whatever the mode — so `DECKPAM` was parsed,
    /// stored, DECRQM-reported, reset correctly, and about 95% inert. It is not an exotic
    /// mode: `smkx` for `xterm-256color` is `\E[?1h\E=`, so every ncurses program that
    /// calls `keypad(true)` turns it on.
    #[test]
    fn the_keypad_encodes_in_both_numlock_states_and_both_modes() {
        let app = Modes {
            app_keypad: true,
            ..Modes::default()
        };

        // NumLock *off*. The layout has resolved these to editing keysyms, so they are
        // genuinely the editing keys and send exactly what those send — including under
        // DECKPAM, which says nothing about a key that is not a keypad key any more.
        for (sym, key, bytes) in [
            (0xff96u32, Key::Left, b"\x1b[D".as_slice()),
            (0xff98, Key::Right, b"\x1b[C"),
            (0xff97, Key::Up, b"\x1b[A"),
            (0xff99, Key::Down, b"\x1b[B"),
            (0xff95, Key::Home, b"\x1b[H"),
            (0xff9c, Key::End, b"\x1b[F"),
            (0xff9a, Key::PageUp, b"\x1b[5~"),
            (0xff9b, Key::PageDown, b"\x1b[6~"),
            (0xff9e, Key::Insert, b"\x1b[2~"),
            (0xff9f, Key::Delete, b"\x1b[3~"),
        ] {
            assert_eq!(key_from_keysym(sym), Some(key), "keysym {sym:#x}");
            assert_eq!(enc(key, Mods::NONE), bytes, "{key:?} with NumLock off");
            assert_eq!(
                encoded(key, Mods::NONE, app),
                bytes,
                "{key:?} under DECKPAM"
            );
        }

        // NumLock *on*: the numeric keypad proper. Numeric mode sends the character
        // printed on the key; application mode gives each one its own SS3 form.
        for (sym, key, numeric, application) in [
            (0xffb0u32, KeypadKey::Digit(0), "0", b"\x1bOp".as_slice()),
            (0xffb1, KeypadKey::Digit(1), "1", b"\x1bOq"),
            (0xffb2, KeypadKey::Digit(2), "2", b"\x1bOr"),
            (0xffb3, KeypadKey::Digit(3), "3", b"\x1bOs"),
            (0xffb4, KeypadKey::Digit(4), "4", b"\x1bOt"),
            (0xffb5, KeypadKey::Digit(5), "5", b"\x1bOu"),
            (0xffb6, KeypadKey::Digit(6), "6", b"\x1bOv"),
            (0xffb7, KeypadKey::Digit(7), "7", b"\x1bOw"),
            (0xffb8, KeypadKey::Digit(8), "8", b"\x1bOx"),
            (0xffb9, KeypadKey::Digit(9), "9", b"\x1bOy"),
            (0xffaa, KeypadKey::Multiply, "*", b"\x1bOj"),
            (0xffab, KeypadKey::Add, "+", b"\x1bOk"),
            (0xffac, KeypadKey::Separator, ",", b"\x1bOl"),
            (0xffad, KeypadKey::Subtract, "-", b"\x1bOm"),
            (0xffae, KeypadKey::Decimal, ".", b"\x1bOn"),
            (0xffaf, KeypadKey::Divide, "/", b"\x1bOo"),
            (0xffbd, KeypadKey::Equal, "=", b"\x1bOX"),
        ] {
            let k = Key::Keypad(key);
            assert_eq!(key_from_keysym(sym), Some(k), "keysym {sym:#x}");
            assert_eq!(enc(k, Mods::NONE), numeric.as_bytes(), "{key:?} numeric");
            assert_eq!(encoded(k, Mods::NONE, app), application, "{key:?} DECKPAM");
        }

        // The five terminfo names for the keypad, which is what an ncurses program is
        // actually matching against: the four corners and the centre.
        let ss3 = |k: KeypadKey| encoded(Key::Keypad(k), Mods::NONE, app);
        assert_eq!(ss3(KeypadKey::Digit(7)), b"\x1bOw", "ka1, upper left");
        assert_eq!(ss3(KeypadKey::Digit(9)), b"\x1bOy", "ka3, upper right");
        assert_eq!(ss3(KeypadKey::Digit(5)), b"\x1bOu", "kb2, centre");
        assert_eq!(ss3(KeypadKey::Digit(1)), b"\x1bOq", "kc1, lower left");
        assert_eq!(ss3(KeypadKey::Digit(3)), b"\x1bOs", "kc3, lower right");
        assert_eq!(
            encoded(Key::KeypadEnter, Mods::NONE, app),
            b"\x1bOM",
            "kent"
        );

        // The centre key with NumLock off is `KP_Begin`, the one keypad key whose
        // NumLock-off role is not an editing key.
        let begin = Key::Keypad(KeypadKey::Begin);
        assert_eq!(key_from_keysym(0xff9d), Some(begin));
        assert_eq!(enc(begin, Mods::NONE), b"\x1b[E");
        assert_eq!(encoded(begin, Mods::NONE, app), b"\x1bOE");

        // Not a keypad keysym: an ordinary key is text, and resolves through the layout.
        assert_eq!(key_from_keysym(0x0061), None, "XK_a");
        assert_eq!(key_from_keysym(0xff0d), None, "XK_Return");
    }

    #[test]
    fn a_keypad_key_in_numeric_mode_is_just_that_character() {
        // Numeric mode goes through the ordinary text encoder, so Ctrl and Alt fold the
        // way they do everywhere else rather than being silently dropped.
        let k = Key::Keypad(KeypadKey::Digit(4));
        assert_eq!(enc(k, Mods::ALT), b"\x1b4");
        assert_eq!(enc(Key::Keypad(KeypadKey::Divide), Mods::NONE), b"/");

        // Application mode has no parameter slot to put a modifier in, so a modified
        // keypad key sends the bare SS3 sequence. A program that wants modified keypad
        // keys asks for a CSI-u protocol instead.
        let app = Modes {
            app_keypad: true,
            ..Modes::default()
        };
        assert_eq!(encoded(k, Mods::CTRL, app), b"\x1bOt");
    }

    #[test]
    fn the_keypad_reports_no_release_because_it_has_nowhere_to_put_one() {
        // Under `REPORT_EVENT_TYPES` a key that reports a release must be able to *say*
        // it is a release. The keypad's SS3 forms have no parameters, so reporting one
        // would re-send the press byte for byte and a program counting keypad presses
        // would see two for every one.
        let mut screen = Screen::new(20, 4);
        let mut parser = crate::vt::Parser::new();
        parser.advance_bytes(&mut screen, b"\x1b[>3u"); // DISAMBIGUATE | REPORT_EVENT_TYPES
        let modes = Modes {
            app_keypad: true,
            ..Modes::from_screen(&screen)
        };
        let k = Key::Keypad(KeypadKey::Digit(7));
        assert_eq!(enc_event(k, Mods::NONE, KeyEvent::Press, modes), b"\x1bOw");
        assert!(enc_event(k, Mods::NONE, KeyEvent::Release, modes).is_empty());
        assert!(enc_event(Key::KeypadEnter, Mods::NONE, KeyEvent::Release, modes).is_empty());
        // An arrow, which *does* have a parameter to carry the event, still reports.
        assert_eq!(
            enc_event(Key::Up, Mods::NONE, KeyEvent::Release, modes),
            b"\x1b[1;1:3A"
        );
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
