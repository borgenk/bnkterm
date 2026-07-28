//! Keyboard translation backed by libxkbcommon (libxkbcommon.so.0).
//!
//! The compositor hands us an XKB keymap over wl_keyboard.keymap; we give it to
//! libxkbcommon and let it own layout and modifier state.
//!
//! What a key *means* belongs to the app that binds it, so this layer answers only
//! the questions the machine can answer about a keycode: the keysym it resolves to
//! under the current layout ([`Xkb::key_sym`]), the text it produces
//! ([`Xkb::key_text`], [`Xkb::key_char`]), whether it auto-repeats, and which
//! modifiers are held. The app turns those into its own actions, which is what keeps
//! one app's shortcuts out of the layer every app shares.
//!
//! Text runs through libxkbcommon's Compose machine (a compose table built from the
//! user's locale), so dead keys and compose sequences resolve: on a Nordic layout
//! `dead_grave` then Space yields `` ` ``, then `a` yields `à`, and on a French azerty
//! `´` then `e` yields `é`. A key mid-sequence produces no text until the sequence
//! completes; keys that are not part of one fall through to their own keysym.
//!
//! The machine is *state between two keystrokes*, which is what makes it awkward and
//! what the two rules here are about. It advances on exactly one call —
//! [`Xkb::key_text`], the typing path — so a caller that merely wants to know which key
//! this is asks [`Xkb::key_char`] and leaves any sequence in flight alone. And it is
//! reset on focus loss ([`Xkb::reset_compose`]), because the keystroke that would have
//! completed the sequence is now going to somebody else's window.

use core::ffi::{c_char, c_int, c_void};
use core::ptr::NonNull;

use crate::platform::error::{Error, Result};

/// Wayland passes raw evdev keycodes; XKB expects them offset by +8.
const EVDEV_OFFSET: u32 = 8;

/// Raw Linux evdev keycodes, as `wl_keyboard.key` reports them (before the
/// libxkbcommon `+8` offset). From `linux/input-event-codes.h`, a kernel ABI, which
/// is why they live down here: an app binds them, but it does not get to define them.
pub mod keycode {
    pub const ESC: u32 = 1;
    pub const BACKSPACE: u32 = 14;
    pub const TAB: u32 = 15;
    pub const ENTER: u32 = 28;
    pub const KP_ENTER: u32 = 96;
    pub const HOME: u32 = 102;
    pub const UP: u32 = 103;
    pub const PAGEUP: u32 = 104;
    pub const LEFT: u32 = 105;
    pub const RIGHT: u32 = 106;
    pub const END: u32 = 107;
    pub const DOWN: u32 = 108;
    pub const PAGEDOWN: u32 = 109;
    pub const DELETE: u32 = 111;
}

const XKB_KEYMAP_FORMAT_TEXT_V1: u32 = 1;
/// The *effective* modifier state: depressed (physically held), latched (sticky
/// keys: tapped, released, and applied to the next key), and locked (Caps Lock)
/// folded together. Position 3 in `enum xkb_state_component`, not 0 — that one is
/// `XKB_STATE_MODS_DEPRESSED`, and asking it would tell a sticky-keys user that
/// their Ctrl is not pressed. Effective is also what xterm reads out of an X
/// event's state mask, so it is what a terminal's chords should agree with.
const XKB_STATE_MODS_EFFECTIVE: u32 = 1 << 3;
/// libxkbcommon's canonical names for the Shift and Control modifiers.
const XKB_MOD_NAME_SHIFT: &[u8] = b"Shift\0";
const XKB_MOD_NAME_CTRL: &[u8] = b"Control\0";
/// The plain Alt key. AltGr is a different modifier (Level3), so this stays
/// false while typing AltGr characters.
const XKB_MOD_NAME_ALT: &[u8] = b"Mod1\0";

/// Compose-sequence status after feeding a keysym. The fourth state, "nothing"
/// (`0`, the keysym is part of no sequence), is the catch-all in the match below.
const XKB_COMPOSE_COMPOSING: c_int = 1;
const XKB_COMPOSE_COMPOSED: c_int = 2;
const XKB_COMPOSE_CANCELLED: c_int = 3;

#[link(name = "xkbcommon")]
#[allow(non_snake_case)]
extern "C" {
    fn xkb_context_new(flags: u32) -> *mut c_void;
    fn xkb_context_unref(ctx: *mut c_void);
    fn xkb_keymap_new_from_string(
        ctx: *mut c_void,
        string: *const c_char,
        format: u32,
        flags: u32,
    ) -> *mut c_void;
    fn xkb_keymap_unref(km: *mut c_void);
    fn xkb_state_new(km: *mut c_void) -> *mut c_void;
    fn xkb_state_unref(st: *mut c_void);
    fn xkb_state_update_mask(
        st: *mut c_void,
        depressed_mods: u32,
        latched_mods: u32,
        locked_mods: u32,
        depressed_layout: u32,
        latched_layout: u32,
        locked_layout: u32,
    ) -> u32;
    fn xkb_state_mod_name_is_active(st: *mut c_void, name: *const c_char, type_: u32) -> c_int;
    fn xkb_state_key_get_one_sym(st: *mut c_void, key: u32) -> u32;
    fn xkb_state_key_get_layout(st: *mut c_void, key: u32) -> u32;
    fn xkb_keymap_key_get_syms_by_level(
        keymap: *mut c_void,
        key: u32,
        layout: u32,
        level: u32,
        syms_out: *mut *const u32,
    ) -> c_int;
    fn xkb_keysym_to_utf32(keysym: u32) -> u32;
    fn xkb_keymap_key_repeats(keymap: *mut c_void, key: u32) -> c_int;
    fn xkb_compose_table_new_from_locale(
        ctx: *mut c_void,
        locale: *const c_char,
        flags: u32,
    ) -> *mut c_void;
    fn xkb_compose_table_unref(table: *mut c_void);
    fn xkb_compose_state_new(table: *mut c_void, flags: u32) -> *mut c_void;
    fn xkb_compose_state_unref(st: *mut c_void);
    fn xkb_compose_state_reset(st: *mut c_void);
    fn xkb_compose_state_feed(st: *mut c_void, keysym: u32) -> c_int;
    fn xkb_compose_state_get_status(st: *mut c_void) -> c_int;
    fn xkb_compose_state_get_utf8(st: *mut c_void, buf: *mut c_char, size: usize) -> c_int;
}

/// Owns the libxkbcommon context, and (once a keymap arrives) the keymap and
/// state. Until then character input returns `None`, which is the correct
/// startup state before wl_keyboard.keymap.
pub struct Xkb {
    ctx: NonNull<c_void>,
    keymap: Option<NonNull<c_void>>,
    state: Option<NonNull<c_void>>,
    /// The Compose state machine for dead keys and compose sequences, built from
    /// the user's locale. `None` if the locale has no compose table, in which
    /// case input falls back to each key's plain UTF-8 (no dead-key support).
    compose: Option<NonNull<c_void>>,
}

impl Xkb {
    pub fn new() -> Result<Self> {
        // SAFETY: xkb_context_new accepts any flags and returns null on failure.
        let ctx = unsafe { xkb_context_new(0) };
        let ctx = NonNull::new(ctx).ok_or_else(|| Error::msg("xkb_context_new returned null"))?;
        let compose = Self::new_compose(ctx);
        Ok(Self {
            ctx,
            keymap: None,
            state: None,
            compose,
        })
    }

    /// Build a Compose state from the user's locale (`$LC_ALL`, `$LC_CTYPE`, or
    /// `$LANG`, defaulting to `C`). Returns `None` if no compose table is found,
    /// which is not an error: input simply has no dead-key support then. The
    /// table is released once the state holds its own reference.
    fn new_compose(ctx: NonNull<c_void>) -> Option<NonNull<c_void>> {
        let locale = std::env::var("LC_ALL")
            .or_else(|_| std::env::var("LC_CTYPE"))
            .or_else(|_| std::env::var("LANG"))
            .unwrap_or_else(|_| "C".to_string());
        let locale = std::ffi::CString::new(locale)
            .or_else(|_| std::ffi::CString::new("C"))
            .ok()?;
        // SAFETY: ctx is valid; locale is a NUL-terminated string living across
        // the call. Returns null when the locale has no compose table.
        let table = unsafe { xkb_compose_table_new_from_locale(ctx.as_ptr(), locale.as_ptr(), 0) };
        let table = NonNull::new(table)?;
        // SAFETY: table is valid; state_new takes its own reference, so we drop
        // ours immediately afterwards regardless of whether state_new succeeded.
        let state = unsafe { xkb_compose_state_new(table.as_ptr(), 0) };
        // SAFETY: table came from compose_table_new and is released exactly once.
        unsafe { xkb_compose_table_unref(table.as_ptr()) };
        NonNull::new(state)
    }

    /// Install the keymap the compositor sent (its raw, NUL-terminated bytes).
    pub fn load_keymap(&mut self, bytes: &[u8], format: u32) -> Result<()> {
        if format != XKB_KEYMAP_FORMAT_TEXT_V1 {
            return Err(Error::msg("compositor sent an unsupported keymap format"));
        }
        // xkb_keymap_new_from_string wants a C string. The payload is
        // NUL-terminated; strip trailing NULs, reject an interior one, then add
        // exactly one terminator.
        let end = bytes.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        let body = &bytes[..end];
        if body.contains(&0) {
            return Err(Error::msg("keymap contains an embedded NUL"));
        }
        let mut cstr = Vec::with_capacity(body.len() + 1);
        cstr.extend_from_slice(body);
        cstr.push(0);

        // SAFETY: ctx is valid; cstr is NUL-terminated and lives across the call.
        let keymap = unsafe {
            xkb_keymap_new_from_string(self.ctx.as_ptr(), cstr.as_ptr() as *const c_char, format, 0)
        };
        let keymap =
            NonNull::new(keymap).ok_or_else(|| Error::msg("xkb_keymap_new_from_string failed"))?;

        // SAFETY: keymap is valid.
        let state = unsafe { xkb_state_new(keymap.as_ptr()) };
        let state = match NonNull::new(state) {
            Some(s) => s,
            None => {
                // SAFETY: keymap is valid and we are releasing the ref we hold.
                unsafe { xkb_keymap_unref(keymap.as_ptr()) };
                return Err(Error::msg("xkb_state_new failed"));
            }
        };

        self.replace(keymap, state);
        Ok(())
    }

    /// Forward the modifier mask from wl_keyboard.modifiers to xkb so that
    /// Shift, Caps Lock, and layout groups affect the next key lookup.
    pub fn update_modifiers(&self, depressed: u32, latched: u32, locked: u32, group: u32) {
        let Some(state) = self.state else { return };
        // SAFETY: state is valid (NonNull, owned).
        unsafe {
            xkb_state_update_mask(state.as_ptr(), depressed, latched, locked, 0, 0, group);
        }
    }

    /// Drop every modifier and layout group, which is what `wl_keyboard.leave`
    /// means: the protocol says the event "resets all values to their defaults",
    /// because once the surface is unfocused its releases are delivered elsewhere
    /// and we would never see the key come up. Skipping this leaves a modifier held
    /// down forever — Ctrl held while switching windows would still read as held on
    /// return, and the next plain click would be taken for a Ctrl+click.
    ///
    /// Safe against a compositor that ties modifiers to *pointer* focus and so keeps
    /// sending `wl_keyboard.modifiers` to us while we are unfocused: that event is
    /// allowed to arrive with no `enter` before it, and it lands after this reset and
    /// simply re-establishes the truth.
    pub fn clear_modifiers(&self) {
        self.update_modifiers(0, 0, 0, 0);
    }

    /// Abandon any compose sequence in flight, the other half of what
    /// `wl_keyboard.leave` means.
    ///
    /// A dead key is state held between two keystrokes, and the second one is now going
    /// somewhere else. Without this, pressing `´` and alt-tabbing away leaves the machine
    /// waiting: the next keystroke *after coming back*, whatever it is and however much
    /// later, gets composed against an accent the user typed into a different window and
    /// has long forgotten.
    pub fn reset_compose(&self) {
        let Some(compose) = self.compose else { return };
        // SAFETY: compose is valid for the lifetime of self; reset only clears state.
        unsafe { xkb_compose_state_reset(compose.as_ptr()) };
    }

    /// Whether Shift is currently active, so navigation keys extend a selection.
    pub fn shift_active(&self) -> bool {
        self.mod_active(XKB_MOD_NAME_SHIFT)
    }

    /// Whether Control is currently active, gating the editor shortcuts.
    pub fn ctrl_active(&self) -> bool {
        self.mod_active(XKB_MOD_NAME_CTRL)
    }

    /// Whether (plain) Alt is currently active; the find bar's mode toggles.
    pub fn alt_active(&self) -> bool {
        self.mod_active(XKB_MOD_NAME_ALT)
    }

    /// The base character a text key produces, from its keysym: Shift and the
    /// layout are applied (so `Shift+2` on a US layout is `'@'`), but Ctrl and Alt
    /// are not, because a terminal folds those itself (`Ctrl+A` → `0x01`, `Alt+x`
    /// → `ESC x`) through its own encoder. `None` for a key with no printable
    /// character (a modifier, a named key, an unresolved dead key), which the
    /// caller then treats as a named key or ignores. Control and DEL scalars are
    /// filtered out so only real graphic input flows through this path.
    pub fn key_char(&self, keycode: u32) -> Option<char> {
        let state = self.state?;
        // SAFETY: state is valid; the sym lookup only reads it. The keysym
        // reflects Shift and the layout but is Ctrl/Alt-independent.
        let sym = unsafe { xkb_state_key_get_one_sym(state.as_ptr(), keycode + EVDEV_OFFSET) };
        Self::graphic(sym)
    }

    /// The character `keycode` produces with *no* modifiers applied, under the layout
    /// group currently in effect: level 0 of the key. This is the key's identity, as
    /// distinct from what it typed — `Shift+a` types `'A'` but its base is `'a'`, and
    /// `Shift+2` on a US layout types `'@'` but its base is `'2'`.
    ///
    /// The CSI-u keyboard protocols (kitty's and xterm's `modifyOtherKeys`) report a key
    /// by its base codepoint plus a modifier bitmask, precisely so an application can tell
    /// `Ctrl+Shift+2` from `Ctrl+@` without knowing the user's layout. Deriving it by
    /// lowercasing the typed character would be right for letters and wrong for every
    /// shifted punctuation key, so we ask xkb instead of guessing.
    ///
    /// `None` before a keymap arrives, or for a key whose unshifted level produces no
    /// character (a modifier, a named key); the caller then falls back to the typed
    /// character.
    pub fn key_base_char(&self, keycode: u32) -> Option<char> {
        let (keymap, state) = (self.keymap?, self.state?);
        let key = keycode + EVDEV_OFFSET;
        // SAFETY: state and keymap are valid for the lifetime of self. `layout` is the
        // group xkb itself reports for this key, so it is in range for the keymap.
        let layout = unsafe { xkb_state_key_get_layout(state.as_ptr(), key) };
        let mut syms: *const u32 = std::ptr::null();
        // SAFETY: keymap is valid; the call writes a pointer to keymap-owned storage into
        // `syms` and returns how many keysyms it holds. The storage lives as long as the
        // keymap, and we only read from it while `self` (which owns the keymap) is alive.
        let n =
            unsafe { xkb_keymap_key_get_syms_by_level(keymap.as_ptr(), key, layout, 0, &mut syms) };
        if n <= 0 || syms.is_null() {
            return None;
        }
        // SAFETY: n > 0, so the first keysym exists.
        let sym = unsafe { *syms };
        // SAFETY: a pure value conversion; 0 means the keysym has no Unicode form.
        let cp = unsafe { xkb_keysym_to_utf32(sym) };
        let c = char::from_u32(cp)?;
        (cp >= 0x20 && c != '\u{7f}').then_some(c)
    }

    fn mod_active(&self, name: &[u8]) -> bool {
        let Some(state) = self.state else {
            return false;
        };
        // SAFETY: state is valid; the name is a static NUL-terminated string.
        let active = unsafe {
            xkb_state_mod_name_is_active(
                state.as_ptr(),
                name.as_ptr() as *const c_char,
                XKB_STATE_MODS_EFFECTIVE,
            )
        };
        active > 0
    }

    /// Whether the keymap marks `keycode` as auto-repeating (text and editing
    /// keys do; modifiers and the like do not).
    pub fn key_repeats(&self, keycode: u32) -> bool {
        let Some(keymap) = self.keymap else {
            return false;
        };
        // SAFETY: keymap is valid; the call only reads the per-key repeat flag.
        let repeats = unsafe { xkb_keymap_key_repeats(keymap.as_ptr(), keycode + EVDEV_OFFSET) };
        repeats != 0
    }

    /// The keysym `keycode` resolves to under the current layout and modifiers.
    /// `None` before a keymap arrives.
    ///
    /// An app binds its modifier shortcuts on this rather than on the physical
    /// keycode, so a shortcut follows the letter the user sees: Ctrl+W is the `w` key
    /// on QWERTY and on Dvorak alike.
    pub fn key_sym(&self, keycode: u32) -> Option<u32> {
        let state = self.state?;
        // SAFETY: state is valid; the call returns the effective keysym.
        Some(unsafe { xkb_state_key_get_one_sym(state.as_ptr(), keycode + EVDEV_OFFSET) })
    }

    /// The character `keycode` types, run through the Compose machine: `None` when it
    /// produces no text — a modifier, a named key, a key part-way through a dead-key
    /// sequence, or one that cancelled an invalid one.
    ///
    /// This is the typing path and the only caller of the Compose state, which is what
    /// makes `dead_acute` then `e` arrive as one `é` rather than as nothing and then an
    /// `e`. Only a single non-control scalar is returned; a sequence yielding several is
    /// dropped.
    ///
    /// **It advances the machine**, so it must be called exactly once per press and never
    /// for a lookup or a release. [`Self::key_char`] is the question to ask when the
    /// answer is only being looked at.
    ///
    /// Like `key_char`, the non-composing fallback goes through the *keysym*, not
    /// `xkb_state_key_get_utf8`. That is the load-bearing detail: `get_utf8` applies Ctrl,
    /// so `Ctrl+A` would come back as `0x01` — a control scalar this filters out, leaving
    /// the key sending nothing at all, and leaving the CSI-u protocols without the base
    /// codepoint they name a key by. Folding `Ctrl` is the encoder's job, not the layout's.
    pub fn key_text(&self, keycode: u32) -> Option<char> {
        let state = self.state?;
        let key = keycode + EVDEV_OFFSET;
        // SAFETY: state is valid; the sym lookup only reads it. The keysym reflects Shift
        // and the layout but is Ctrl/Alt-independent.
        let sym = unsafe { xkb_state_key_get_one_sym(state.as_ptr(), key) };
        let Some(compose) = self.compose else {
            // No compose table for this locale, which is not an error: input simply has
            // no dead-key support, and every key is its own keysym.
            return Self::graphic(sym);
        };
        // A key mid-sequence (or one that cancels an invalid sequence) produces nothing;
        // a completed sequence yields its composed text; a key that is part of no
        // sequence falls through to its own keysym (the common path for ordinary typing).
        //
        // SAFETY: compose is valid; the feed advances it.
        unsafe { xkb_compose_state_feed(compose.as_ptr(), sym) };
        // SAFETY: compose is valid.
        match unsafe { xkb_compose_state_get_status(compose.as_ptr()) } {
            XKB_COMPOSE_COMPOSING => None,
            XKB_COMPOSE_CANCELLED => {
                // SAFETY: compose is valid; clear the abandoned sequence.
                unsafe { xkb_compose_state_reset(compose.as_ptr()) };
                None
            }
            XKB_COMPOSE_COMPOSED => {
                let mut buf = [0u8; 16];
                // SAFETY: compose is valid; buf is writable and sized for one composed
                // result.
                let n = unsafe {
                    xkb_compose_state_get_utf8(
                        compose.as_ptr(),
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len(),
                    )
                };
                // SAFETY: compose is valid; ready it for the next sequence.
                unsafe { xkb_compose_state_reset(compose.as_ptr()) };
                if n <= 0 {
                    return None;
                }
                // Like snprintf, the return is the required length; clamp before slicing.
                let n = (n as usize).min(buf.len());
                let s = core::str::from_utf8(buf.get(..n)?).ok()?;
                let mut chars = s.chars();
                match (chars.next(), chars.next()) {
                    (Some(c), None) => Self::printable(c),
                    _ => None,
                }
            }
            // XKB_COMPOSE_NOTHING, or any value a newer library returns.
            _ => Self::graphic(sym),
        }
    }

    /// A keysym's printable character, or `None`. The one place a keysym becomes text,
    /// so `key_char` and `key_text` cannot drift about what counts as typing.
    fn graphic(sym: u32) -> Option<char> {
        // SAFETY: a pure value conversion; 0 means the keysym has no Unicode form.
        let cp = unsafe { xkb_keysym_to_utf32(sym) };
        Self::printable(char::from_u32(cp)?)
    }

    /// Control scalars and DEL are not typing: a terminal makes those itself, out of a
    /// key plus Ctrl, and a layout handing one over directly would bypass the encoder.
    fn printable(c: char) -> Option<char> {
        ((c as u32) >= 0x20 && c != '\u{7f}').then_some(c)
    }

    fn replace(&mut self, keymap: NonNull<c_void>, state: NonNull<c_void>) {
        if let Some(old) = self.state.replace(state) {
            // SAFETY: old came from xkb_state_new and is freed once.
            unsafe { xkb_state_unref(old.as_ptr()) };
        }
        if let Some(old) = self.keymap.replace(keymap) {
            // SAFETY: old came from xkb_keymap_new_* and is freed once.
            unsafe { xkb_keymap_unref(old.as_ptr()) };
        }
    }
}

impl Drop for Xkb {
    fn drop(&mut self) {
        // SAFETY: each pointer came from its matching xkb_*_new and is freed
        // exactly once; we hold the only references.
        unsafe {
            if let Some(c) = self.compose.take() {
                xkb_compose_state_unref(c.as_ptr());
            }
            if let Some(s) = self.state.take() {
                xkb_state_unref(s.as_ptr());
            }
            if let Some(k) = self.keymap.take() {
                xkb_keymap_unref(k.as_ptr());
            }
            xkb_context_unref(self.ctx.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_are_inert_without_a_keymap() {
        // Before wl_keyboard.keymap arrives there is no state, so every question
        // about a key answers "nothing" rather than guessing or panicking. Keycode
        // 30 is `a` on a Linux keyboard.
        let xkb = Xkb::new().expect("xkb context");
        assert_eq!(xkb.key_sym(30), None);
        assert_eq!(xkb.key_text(30), None);
        assert_eq!(xkb.key_char(30), None);
        assert!(!xkb.key_repeats(30));
    }

    #[test]
    fn modifiers_are_inactive_without_a_keymap() {
        let xkb = Xkb::new().expect("xkb context");
        assert!(!xkb.shift_active());
        assert!(!xkb.ctrl_active());
        assert!(!xkb.alt_active());
    }

    /// The smallest keymap that can answer "is Ctrl held": one key bound to
    /// `Control_L` and mapped into the real `Control` modifier, plus the two keys the
    /// compose tests type on. Written out here rather than read from the system's xkb
    /// data so the test depends on nothing outside the process.
    ///
    /// The keycodes carry the `+8` evdev offset, so `<AC01> = 38` is evdev 30 (the `a`
    /// key) and `<AD03> = 26` is evdev 18.
    const MINIMAL_KEYMAP: &str = r#"
xkb_keymap {
  xkb_keycodes "min" { minimum = 8; maximum = 255; <LCTL> = 37; <AC01> = 38; <AD03> = 26; };
  xkb_types "min" {
    type "ONE_LEVEL" { modifiers = none; map[none] = Level1; level_name[Level1] = "Any"; };
  };
  xkb_compatibility "min" {
    interpret Control_L { action = SetMods(modifiers = Control); };
  };
  xkb_symbols "min" {
    key <LCTL> { type = "ONE_LEVEL", [ Control_L ] };
    key <AC01> { type = "ONE_LEVEL", [ a ] };
    key <AD03> { type = "ONE_LEVEL", [ dead_acute ] };
    modifier_map Control { <LCTL> };
  };
};
"#;

    /// Evdev codes for the two keys `MINIMAL_KEYMAP` binds to text.
    const KEY_A: u32 = 30;
    const KEY_DEAD_ACUTE: u32 = 18;

    /// A compose table in the standard `Compose` file format, holding the one sequence
    /// these tests need.
    const COMPOSE_TABLE: &str = "<dead_acute> <a> : \"á\"\n";

    // Building a table from a *buffer* is the only reason this is declared: production
    // builds one from the user's locale, which means reading
    // `/usr/share/X11/locale/.../Compose` — a system file that may or may not exist on
    // the machine running the suite. A test that depends on it is a test that fails
    // somewhere else for reasons that have nothing to do with the code.
    #[link(name = "xkbcommon")]
    extern "C" {
        fn xkb_compose_table_new_from_buffer(
            ctx: *mut c_void,
            buffer: *const c_char,
            length: usize,
            locale: *const c_char,
            format: u32,
            flags: u32,
        ) -> *mut c_void;
    }

    /// An [`Xkb`] with `MINIMAL_KEYMAP` loaded and the real libxkbcommon Compose machine
    /// running `COMPOSE_TABLE`. Not a mock: it is the same machine production uses, fed
    /// a fixture table instead of the system's.
    fn composing_xkb() -> Xkb {
        let mut xkb = Xkb::new().expect("xkb context");
        xkb.load_keymap(MINIMAL_KEYMAP.as_bytes(), XKB_KEYMAP_FORMAT_TEXT_V1)
            .expect("the minimal keymap compiles");

        let locale = std::ffi::CString::new("C").expect("no NUL");
        // SAFETY: ctx is valid; the buffer and locale live across the call. Format 1 is
        // XKB_COMPOSE_FORMAT_TEXT_V1, the only format the library defines.
        let table = unsafe {
            xkb_compose_table_new_from_buffer(
                xkb.ctx.as_ptr(),
                COMPOSE_TABLE.as_ptr() as *const c_char,
                COMPOSE_TABLE.len(),
                locale.as_ptr(),
                1,
                0,
            )
        };
        let table = NonNull::new(table).expect("the fixture compose table compiles");
        // SAFETY: table is valid; the state takes its own reference, so ours is released
        // immediately afterwards.
        let state = unsafe { xkb_compose_state_new(table.as_ptr(), 0) };
        // SAFETY: table came from compose_table_new_from_buffer and is released once.
        unsafe { xkb_compose_table_unref(table.as_ptr()) };
        let state = NonNull::new(state).expect("compose state");
        if let Some(old) = xkb.compose.replace(state) {
            // SAFETY: the locale-built state this replaces is freed exactly once.
            unsafe { xkb_compose_state_unref(old.as_ptr()) };
        }
        xkb
    }

    #[test]
    fn a_dead_key_composes_with_the_key_after_it() {
        // The headline: on a layout with dead keys, `´` then `a` is one `á`. Before this
        // was wired up, `key_text` had zero production callers — the typing path used
        // `key_char`, which reads the keysym directly — so `´` sent nothing, buffered
        // nothing, and `a` then gave a bare `a`. Typing `á` was impossible, on every
        // French, Nordic and US-international layout.
        let xkb = composing_xkb();

        assert_eq!(
            xkb.key_text(KEY_DEAD_ACUTE),
            None,
            "a key mid-sequence types nothing yet"
        );
        assert_eq!(
            xkb.key_text(KEY_A),
            Some('á'),
            "and the next key completes it"
        );
        assert_eq!(
            xkb.key_text(KEY_A),
            Some('a'),
            "the machine is ready for a fresh sequence, not still holding the accent"
        );
    }

    #[test]
    fn a_half_typed_sequence_does_not_survive_losing_focus() {
        // The keystroke that would have completed the sequence is going to another
        // window now. Without the reset the accent waits — through an alt-tab, through a
        // coffee break — and lands on whatever is typed next, in a sequence the user
        // stopped composing minutes ago.
        let xkb = composing_xkb();
        assert_eq!(xkb.key_text(KEY_DEAD_ACUTE), None, "sequence in flight");

        xkb.reset_compose();
        assert_eq!(
            xkb.key_text(KEY_A),
            Some('a'),
            "the abandoned accent did not outlive the focus"
        );
    }

    #[test]
    fn holding_ctrl_does_not_turn_typing_into_a_control_byte() {
        // The trap in routing the typing path through Compose. The obvious way to read a
        // key's text is `xkb_state_key_get_utf8`, and it applies Ctrl: `Ctrl+A` comes
        // back as `0x01`. That is wrong twice over — the control scalar is filtered out
        // as non-typing, so the key would send *nothing at all*, and the CSI-u protocols
        // name a key by its base codepoint, which `0x01` is not. Folding Ctrl is the
        // encoder's job; the layer below hands over the character.
        let xkb = composing_xkb();
        // `Control` is real modifier index 2 in every keymap.
        xkb.update_modifiers(1 << 2, 0, 0, 0);
        assert!(xkb.ctrl_active());

        assert_eq!(xkb.key_text(KEY_A), Some('a'), "still the letter");
        assert_eq!(xkb.key_char(KEY_A), Some('a'), "and the lookup agrees");
    }

    #[test]
    fn a_lookup_leaves_a_sequence_in_flight_alone() {
        // `key_char` is the question to ask when the answer is only being looked at (a
        // shortcut table, a release). Advancing the machine there would consume a dead
        // key's partner before the user had typed it.
        let xkb = composing_xkb();
        assert_eq!(xkb.key_text(KEY_DEAD_ACUTE), None, "sequence in flight");

        assert_eq!(xkb.key_char(KEY_A), Some('a'), "the lookup reads the key");
        assert_eq!(
            xkb.key_text(KEY_A),
            Some('á'),
            "and the sequence is still waiting for it"
        );
    }

    #[test]
    fn losing_focus_clears_a_held_modifier() {
        // `wl_keyboard.leave` "resets all values to their defaults". It has to: once
        // the surface is unfocused, the *release* of a held key is delivered to
        // somebody else, so a Ctrl we do not drop here stays down for the life of the
        // window — and the next plain left-click would be read as a Ctrl+click and
        // follow a hyperlink the user never asked for.
        let mut xkb = Xkb::new().expect("xkb context");
        xkb.load_keymap(MINIMAL_KEYMAP.as_bytes(), XKB_KEYMAP_FORMAT_TEXT_V1)
            .expect("the minimal keymap compiles");

        // `Control` is real modifier index 2 in every keymap, so its mask is 1 << 2 —
        // the same bits a compositor puts in wl_keyboard.modifiers.
        xkb.update_modifiers(1 << 2, 0, 0, 0);
        assert!(xkb.ctrl_active(), "the depressed mask makes Ctrl active");

        xkb.clear_modifiers();
        assert!(!xkb.ctrl_active(), "and leaving the surface drops it");

        // And a modifiers event arriving with no `enter` before it (a compositor that
        // ties modifier state to pointer focus) still lands: that is what lets an
        // unfocused window offer the hand cursor over a Ctrl-clickable link.
        xkb.update_modifiers(1 << 2, 0, 0, 0);
        assert!(
            xkb.ctrl_active(),
            "an unfocused modifiers event is honoured"
        );
    }

    #[test]
    fn a_latched_or_locked_modifier_counts_as_active() {
        // Sticky keys do not hold a modifier down, they *latch* it: the user taps
        // Ctrl, releases it, and it applies to the next key. The compositor reports
        // that in the latched mask with nothing in the depressed one, so a terminal
        // that only asks about depressed modifiers answers "Ctrl is not pressed" and
        // Ctrl+C never reaches the shell. Locked modifiers (Caps Lock, or a layout
        // that locks Shift) have the same shape. Effective state is the union of the
        // three, and it is what xterm reads out of an X event's state mask.
        let mut xkb = Xkb::new().expect("xkb context");
        xkb.load_keymap(MINIMAL_KEYMAP.as_bytes(), XKB_KEYMAP_FORMAT_TEXT_V1)
            .expect("the minimal keymap compiles");

        xkb.update_modifiers(0, 1 << 2, 0, 0);
        assert!(xkb.ctrl_active(), "a latched Ctrl is active");

        xkb.update_modifiers(0, 0, 1 << 2, 0);
        assert!(xkb.ctrl_active(), "a locked Ctrl is active");

        xkb.clear_modifiers();
        assert!(!xkb.ctrl_active(), "and neither outlives a reset");
    }
}
