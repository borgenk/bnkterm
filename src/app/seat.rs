//! The seat: the keyboard and the pointer as the window receives them. Keys are resolved
//! through xkb, offered to the leader key-table and the scrollback chords, and what is
//! left goes to the active terminal, with auto-repeat synthesised from the compositor's
//! `repeat_info`. Pointer events are hit-tested against the tab strip, the scrollbar and
//! the grid, and a press in the grid carries its click count for word and line selection.
//!
//! ```text
//!   wl_keyboard ─▶ xkb ─▶ keymode ─┬─▶ window action (tabs, copy, paste)
//!                                  ├─▶ scrollback chord
//!                                  └─▶ active TerminalCore ─▶ PTY
//!   wl_pointer ─▶ tab strip / scrollbar / grid cell ─▶ TerminalCore::pointer
//! ```

use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use crate::app::clipboard::Transport;
use crate::app::message::{PointerEvent, Side};
use crate::app::State;
use crate::error::{Error, Result};
use crate::input;
use crate::keymode::{self, Disposition, KeyMode, WindowAction};
use crate::mouse::MouseButton;
use crate::platform::ffi;
use crate::platform::protocol::{
    self, wl_keyboard, wl_pointer, wl_seat, wp_cursor_shape_device_v1,
};
use crate::platform::verdict::Verdict;
use crate::platform::wire::{Arg, Reader};

/// Right mouse button (`BTN_RIGHT`) and middle (`BTN_MIDDLE`) from
/// `linux/input-event-codes.h`; `BTN_LEFT` is in `protocol`.
const BTN_RIGHT: u32 = 0x111;

const BTN_MIDDLE: u32 = 0x112;

/// The window, in milliseconds, within which successive left presses on the same
/// cell count as a double/triple click (a common desktop default).
const MULTI_CLICK_MS: u32 = 400;

/// Whether resolving a key advances the layout's Compose machine.
///
/// It has to advance exactly once per press, and for nothing else. Feeding a dead key
/// twice composes it with itself (`´´` is a literal acute in every compose table), and
/// feeding a *release* leaves the sequence one keystroke out of step with the user — so
/// the lookups that merely ask "which key is this?" hold the machine still, and only the
/// one call that actually types advances it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compose {
    /// The typing path: run the key through the Compose machine.
    Advance,
    /// A lookup: read the key's own character and leave any sequence in flight alone.
    Hold,
}

/// The pointer over the window: where it is, the cursor shape it shows, and the
/// press gestures in progress.
#[derive(Default)]
pub(super) struct PointerState {
    /// Latest position in surface pixels. `axis_accum` gathers fractional wheel
    /// deltas into whole notches (the button held for drag reporting lives on the
    /// core, with the mouse mode it reports under).
    x: f32,
    y: f32,
    /// The cursor shape currently applied (a `wp_cursor_shape_device_v1::SHAPE_*`),
    /// or `0` when none is set yet. Tracked so motion only re-sends `set_shape` when
    /// the pointer crosses between the grid (I-beam) and the tab strip (arrow).
    shape: u32,
    /// The serial of the latest `wl_pointer.enter`, which `set_shape` must cite to
    /// change the cursor (`0` before the pointer has entered).
    enter_serial: u32,
    axis_accum: f32,
    /// A button press that began in the tab bar. Its matching release is swallowed
    /// even if closing a tab made the bar disappear in between.
    bar_button: Option<MouseButton>,
    /// Whether a left press took the visible tab's scrollbar thumb and is scrubbing it.
    /// The drag owns the pointer wherever it wanders (off the lane, off the grid), and
    /// swallows its own release, so it can neither start a selection nor leave the thumb
    /// stuck to the pointer. Where inside the thumb it grabbed lives on the bar itself.
    drag_scroll: bool,
    /// Multi-click tracking for word/line selection: the wl time (ms) and cell of the
    /// last left press, and the running count (1 character, 2 word, 3 line, cycling).
    last_click_time: u32,
    last_click_cell: (usize, usize),
    click_count: usize,
}

impl State {
    /// The pointer in device pixels. Wayland delivers it in logical (surface-local)
    /// ones, but the grid and the chrome hanging off it are laid out in device pixels,
    /// so every hit-test against them has to scale up first.
    fn pointer_device(&self) -> (f32, f32) {
        let scale = self.scale.factor();
        (scale.pxf(self.pointer.x), scale.pxf(self.pointer.y))
    }

    /// One wl_keyboard event: install the keymap, track focus and modifiers, and
    /// turn a key press into bytes for the child.
    pub(super) fn on_keyboard(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_keyboard::EV_KEYMAP => {
                let format = r.u32()?;
                let size = r.u32()? as usize;
                let fd = self
                    .conn
                    .take_fd()
                    .ok_or_else(|| Error::msg("wl_keyboard.keymap arrived without its fd"))?;
                // A zero-length keymap is "no keymap", not a fatal error. `mmap` with
                // `len == 0` is `EINVAL`, which would propagate out of `run_until` and
                // exit the terminal over an event that is merely useless: `Xkb` already
                // handles having no keymap (every key query answers `None`), which is the
                // state it is in before this event arrives at all.
                if size == 0 {
                    return Ok(());
                }
                let bytes = ffi::read_mapped(fd.as_raw_fd(), size)?;
                self.xkb.load_keymap(&bytes, format)?;
            }
            wl_keyboard::EV_ENTER => {
                let _serial = r.u32()?;
                self.window_focused = true;
                self.tabs.active_mut().focus(true)?;
            }
            wl_keyboard::EV_LEAVE => {
                let _serial = r.u32()?;
                self.keyboard_left()?;
            }
            wl_keyboard::EV_MODIFIERS => {
                let _serial = r.u32()?;
                let depressed = r.u32()?;
                let latched = r.u32()?;
                let locked = r.u32()?;
                let group = r.u32()?;
                // Deliberately not gated on `window_focused`. This is the one keyboard
                // event the protocol lets a compositor send to an *unfocused* surface,
                // "to tie modifier information to pointer focus instead" — so a window
                // can know Ctrl is down and offer a Ctrl+click affordance before it is
                // focused. Compositors that do this send it with no `enter` before it;
                // those that do not simply never reach here while unfocused, and the
                // hand waits for focus. Either way, honouring it is correct.
                self.xkb.update_modifiers(depressed, latched, locked, group);
                // Ctrl is what turns a hovered link into a clickable one, so taking it
                // or letting it go changes the cursor, and shows or hides the link's
                // preview, with the pointer standing still.
                self.update_pointer_shape();
            }
            wl_keyboard::EV_KEY => {
                self.last_serial = r.u32()?;
                let _time = r.u32()?;
                let keycode = r.u32()?;
                let key_state = r.u32()?;
                if key_state == wl_keyboard::KEY_STATE_PRESSED {
                    self.on_key_press(keycode)?;
                } else {
                    if self.repeat_key == Some(keycode) {
                        // The held key was released: stop repeating it.
                        self.stop_repeat();
                    }
                    self.on_key_release(keycode)?;
                }
            }
            wl_keyboard::EV_REPEAT_INFO => {
                let rate = r.u32()?; // repeats per second (0 disables)
                let delay = r.u32()?; // ms before the first repeat
                self.repeat_interval =
                    (rate > 0).then(|| Duration::from_secs_f64(1.0 / rate as f64));
                self.repeat_delay = Duration::from_millis(u64::from(delay));
            }
            _ => {}
        }
        Ok(())
    }

    /// Keyboard focus is gone, by `leave` or because the keyboard itself went away.
    fn keyboard_left(&mut self) -> Result<()> {
        self.window_focused = false;
        self.tabs.active_mut().focus(false)?;
        // Drop any held-key repeat (window-side timer).
        self.stop_repeat();
        // Losing focus resets the modifier state (see `Xkb::clear_modifiers`); if Ctrl was
        // down, the hand cursor it earned goes with it.
        self.xkb.clear_modifiers();
        // And any half-typed compose sequence, for the same reason: its second keystroke
        // is going to another window now.
        self.xkb.reset_compose();
        self.update_pointer_shape();
        Ok(())
    }

    /// The seat lost its keyboard: release the inert object and forget it, remembering
    /// its id until the compositor confirms the deletion (see `released_keyboards`).
    pub(super) fn release_keyboard(&mut self) -> Result<()> {
        let keyboard = std::mem::take(&mut self.objects.keyboard);
        if self.seat_version >= wl_seat::VERSION_RELEASE {
            self.conn.request(keyboard, wl_keyboard::RELEASE, &[]);
        }
        self.released_keyboards.push(keyboard);
        self.keyboard_left()
    }

    /// The seat lost its pointer: destroy its cursor-shape device, release the inert
    /// object and forget it. Pointer events carry no fds, so nothing in flight for it
    /// needs taking off the fd queue.
    pub(super) fn release_pointer(&mut self) -> Result<()> {
        if self.objects.cursor_shape_device != 0 {
            self.conn.request(
                self.objects.cursor_shape_device,
                wp_cursor_shape_device_v1::DESTROY,
                &[],
            );
            self.objects.cursor_shape_device = 0;
        }
        let pointer = std::mem::take(&mut self.objects.pointer);
        if self.seat_version >= wl_seat::VERSION_RELEASE {
            self.conn.request(pointer, wl_pointer::RELEASE, &[]);
        }
        self.pointer.enter_serial = 0;
        // A pointer the seat took away sends no release, and maybe no leave first.
        self.end_pointer_gesture();
        self.pointer_left_grid()
    }

    /// Cancel any auto-repeat in flight.
    fn stop_repeat(&mut self) {
        self.repeat_key = None;
        self.repeat_at = None;
    }

    /// A key came back up.
    ///
    /// Almost nothing wants to know: the encoder drops a release unless a program has
    /// asked for release events (kitty's `REPORT_EVENT_TYPES`), so for every other child
    /// this resolves the key and then writes nothing. The modal leader tables never see
    /// it — a mode is armed by pressing a key, not by letting go of one.
    fn on_key_release(&mut self, keycode: u32) -> Result<()> {
        let mods = self.current_mods();
        // `Hold`: a release is not typing. Feeding it to the Compose machine would put
        // every dead key one keystroke out of step — press `´`, release it, and the
        // release would already have consumed the `a` the user had not typed yet.
        if let Some(key) = self.resolve_key(keycode, Compose::Hold) {
            self.tabs
                .active_mut()
                .key(key, mods, input::KeyEvent::Release)?;
        }
        Ok(())
    }

    /// Encode one key press and write it to the child. A named key (an arrow, a
    /// function key) is mapped by its keycode; anything else takes its layout
    /// character, and the [`crate::input`] encoder applies Ctrl/Alt. Modifiers
    /// come from xkb's current state.
    fn on_key_press(&mut self, keycode: u32) -> Result<()> {
        let mods = self.current_mods();
        // Resolved once for the whole press, and that is load-bearing rather than tidy:
        // this is the call that advances the Compose machine, so resolving again further
        // down would feed a dead key a second time and compose it with itself.
        let resolved = self.resolve_key(keycode, Compose::Advance);

        // The window's key bindings get first look at every resolved key. In Normal
        // mode only the leader and the direct chords are bound, so everything else
        // reports `Passthrough` and drops to the scrollback keys and the child below
        // unchanged; once a mode is armed the machine owns the keystroke until it
        // returns to Normal.
        if let Some(key) = resolved {
            let (next, disposition) = keymode::advance(self.key_mode, key, mods);
            self.set_key_mode(next);
            match disposition {
                Disposition::Consumed(action) => {
                    self.stop_repeat();
                    if let Some(action) = action {
                        self.apply_window_action(action)?;
                    }
                    return Ok(());
                }
                Disposition::SendLiteral(literal, literal_mods) => {
                    // The Ctrl+A Ctrl+A escape hatch: type a real Ctrl+A.
                    self.stop_repeat();
                    self.tabs
                        .active_mut()
                        .key(literal, literal_mods, input::KeyEvent::Press)?;
                    return Ok(());
                }
                Disposition::Passthrough => {}
            }
        }

        // Shift + Page/Home/End scrolls the scrollback view instead of reaching the
        // child (the alt screen has no history, so there it is a normal key). The
        // already-resolved key rather than the keycode, so the keypad's Page/Home/End
        // reach this too — with NumLock off they *are* those keys.
        if let Some(named) = resolved {
            if self.tabs.active_mut().handle_scroll_key(named, mods) {
                return Ok(());
            }
        }
        // Send the key, and if it produced bytes and the keymap marks it
        // repeatable, arm auto-repeat on it.
        if let Some(key) = resolved {
            if self
                .tabs
                .active_mut()
                .key(key, mods, input::KeyEvent::Press)?
            {
                self.arm_repeat(keycode);
            }
        }
        Ok(())
    }

    /// Switch the armed leader mode, repainting so the overlay appears, updates, or
    /// clears. A mode change touches neither the grid nor the bar, so it marks the
    /// active core dirty itself; the damage diff then repaints only the overlay
    /// region (the grid portion of the two frames is identical).
    fn set_key_mode(&mut self, mode: KeyMode) {
        if self.key_mode != mode {
            self.key_mode = mode;
            self.tabs.mark_dirty();
        }
    }

    /// Run an action a key resolved to. Opening and closing tabs are inert in the
    /// no-PTY demo; switching and reordering are naturally no-ops there, since the demo
    /// has a single tab.
    fn apply_window_action(&mut self, action: WindowAction) -> Result<()> {
        match action {
            WindowAction::NewTab => self.open_tab(),
            WindowAction::CloseTab => self.close_active_tab(),
            WindowAction::SelectTab(dir) => {
                self.tabs.cycle(dir, self.window_focused);
            }
            WindowAction::MoveTab(dir) => {
                self.tabs.move_active(dir);
            }
            // The core extracts the selection text and queues an `OfferSelection`; the
            // loop's outbox drain owns the clipboard.
            WindowAction::Copy => self.tabs.active_mut().copy_selection(),
            WindowAction::Paste => self.paste_from(Transport::Clipboard)?,
        }
        Ok(())
    }

    /// Open a fresh shell tab at the current geometry, reporting a spawn failure to
    /// stderr without disturbing the existing tabs. Inert in the no-PTY demo.
    fn open_tab(&mut self) {
        if self.tabs.active().is_demo() {
            return;
        }
        let geom = self.layout.terminal;
        if let Err(error) = self.tabs.open(geom, self.window_focused) {
            eprintln!("bnkterm: could not open tab: {error}");
        }
    }

    /// Close the visible tab, flagging window shutdown when it was the last one.
    /// Inert in the no-PTY demo.
    fn close_active_tab(&mut self) {
        if self.tabs.active().is_demo() {
            return;
        }
        if let Some(id) = self.tabs.active_id() {
            self.closed = self.tabs.close(id, self.window_focused);
        }
    }

    /// The window half of a key press: resolve `keycode` to an [`input::Key`] via
    /// the keymap (a named key by its keycode, a keypad key by its keysym, else its
    /// layout character), or `None` for a bare modifier / unresolved key. xkb belongs
    /// with the Wayland keyboard, so this stays window-side; the terminal half is
    /// [`TerminalCore::key`](crate::app::terminal::TerminalCore::key).
    fn resolve_key(&self, keycode: u32, compose: Compose) -> Option<input::Key> {
        if let Some(named) = input::key_from_keycode(keycode) {
            return Some(named);
        }
        // The keypad is named by its *keysym*, not its keycode, and that is not an
        // implementation detail: NumLock decides whether keypad 4 is a `4` or a Left
        // arrow, and xkb is what applies NumLock. Asking the physical code cannot tell
        // the two apart, which is why the whole cluster used to send nothing.
        if let Some(named) = self.xkb.key_sym(keycode).and_then(input::key_from_keysym) {
            return Some(named);
        }
        let typed = match compose {
            Compose::Advance => self.xkb.key_text(keycode)?,
            Compose::Hold => self.xkb.key_char(keycode)?,
        };
        // The character the key types, and the character it *is*. They differ on a
        // shifted key (`Shift+2` types `'@'` but is the `2` key), and the CSI-u keyboard
        // protocols report the latter. A key whose unshifted level produces no character
        // is its own base.
        let base = self.xkb.key_base_char(keycode).unwrap_or(typed);
        Some(input::Key::Char { typed, base })
    }

    /// Arm (or re-arm) auto-repeat on the just-pressed key. A key the keymap marks
    /// non-repeating (or repeat being disabled) instead stops any repeat in
    /// flight, so the last press always governs.
    fn arm_repeat(&mut self, keycode: u32) {
        if self.repeat_interval.is_some() && self.xkb.key_repeats(keycode) {
            self.repeat_key = Some(keycode);
            self.repeat_at = Some(Instant::now() + self.repeat_delay);
        } else {
            self.repeat_key = None;
            self.repeat_at = None;
        }
    }

    /// Fire one auto-repeat of the held key and schedule the next, or stop if the
    /// key no longer produces bytes (e.g. its modifiers changed).
    pub(super) fn fire_repeat(&mut self) -> Result<()> {
        let (Some(keycode), Some(interval)) = (self.repeat_key, self.repeat_interval) else {
            self.repeat_at = None;
            return Ok(());
        };
        let mods = self.current_mods();
        // A repeat is a press, Compose machine included: holding a key down after a dead
        // key composes the first one and then types the rest plainly, which is what every
        // other toolkit does with the same held key.
        let sent = match self.resolve_key(keycode, Compose::Advance) {
            // The keyboard is repeating a held key. A program that asked to hear about
            // repeats is told it is one; to everyone else it is another press, which is
            // exactly what a repeat has always looked like to a terminal.
            Some(key) => self
                .tabs
                .active_mut()
                .key(key, mods, input::KeyEvent::Repeat)?,
            None => false,
        };
        if sent {
            self.repeat_at = Some(Instant::now() + interval);
        } else {
            self.repeat_key = None;
            self.repeat_at = None;
        }
        Ok(())
    }

    /// The current modifier chord from xkb, for both key and mouse encoding.
    fn current_mods(&self) -> input::Mods {
        input::Mods::new(
            self.xkb.shift_active(),
            self.xkb.alt_active(),
            self.xkb.ctrl_active(),
            false,
        )
    }

    /// The cell under the pointer, clamped into the grid, plus which [`Side`] of
    /// that cell's midline the pointer falls on. Mouse reports, hovers, and
    /// multi-click identity use the cell alone; a character selection also reads
    /// the side to round each endpoint to the nearer cell edge.
    fn pointer_cell(&self) -> (usize, usize, Side) {
        let grid = self.layout.terminal;
        // Pointer coordinates are logical (surface-local); the grid is device
        // pixels, so scale up first. The grid is then inset by the (device) padding;
        // a pointer in the margin maps to the nearest edge cell (floored at zero).
        let (px, py) = self.pointer_device();
        let px = (px - grid.pad as f32).max(0.0);
        let py = (py - grid.origin_y as f32).max(0.0);
        let (col, side) = column_under(px, grid.metrics.w, grid.cols);
        let row = ((py / grid.metrics.h as f32) as usize).min(grid.rows.saturating_sub(1));
        (col, row, side)
    }

    /// Whether the latest pointer position is inside the device-pixel tab strip.
    /// The strip may sit at the top or the bottom, so this reads the resolved
    /// rectangle rather than assuming a one-cell bar under the padding.
    fn pointer_in_tab_bar(&self) -> bool {
        if !self.tabs.shows_bar() {
            return false;
        }
        let y = self.pointer_device().1;
        y >= self.layout.bar_y as f32 && y < (self.layout.bar_y + self.layout.bar_h) as f32
    }

    /// Stable tab identity under the pointer while it is in the bar.
    fn pointer_bar_tab(&self) -> Option<crate::app::tabs::TabId> {
        if !self.pointer_in_tab_bar() {
            return None;
        }
        let grid = self.layout.terminal;
        let x = self.pointer_device().0;
        if x < grid.pad as f32 {
            return None;
        }
        // Not `column_under`: that clamps to the last column, which would make the
        // padding past the final tab click it rather than nothing.
        let col = ((x - grid.pad as f32) / grid.metrics.w as f32) as usize;
        self.tabs.tab_at_bar_col(col)
    }

    /// Gather what the pointer is over and ask the compositor for the shape
    /// [`pointer_shape_for`] picks from it. Only re-sent when the shape changes, so a
    /// stream of motion events never spams `set_shape`. A no-op when the compositor
    /// lacks `wp_cursor_shape_manager_v1` (no device, so its default arrow stands) or
    /// before the pointer has entered (no serial to cite). The hovered link's preview
    /// follows the same facts ([`link_preview_shown`]), with or without the protocol, and
    /// the frame repaints when it comes or goes.
    ///
    /// Every shape here is a promise about what the next gesture does, which is why
    /// both of the modifier-gated facts are read live rather than cached: the hand
    /// needs Ctrl actually held (the underline says "this is a link", the hand says
    /// "and a click now follows it", and `not_allowed` over a refused link says "and a
    /// click does nothing"), and the I-beam needs Shift to have taken the grid back
    /// from a program reporting the mouse. Both can flip with the pointer parked, so
    /// the modifiers event re-runs this, and so does the main loop — a program
    /// grabbing the mouse arrives as output, not as a pointer event.
    ///
    /// Whether the Ctrl promise can be kept while the window is *unfocused* is the
    /// compositor's call, not ours: modifier state reaches an unfocused surface only
    /// if the compositor ties it to pointer focus (see the `EV_MODIFIERS` arm). Where
    /// it does, the hand appears over an unfocused window exactly as over a focused
    /// one. Where it does not, `ctrl_active` is false and the pointer keeps the I-beam
    /// until the window takes focus — the underline still marks the link, and the
    /// Ctrl+click still works, because the click brings focus (and the modifiers with
    /// it) before the button press is delivered.
    pub(super) fn update_pointer_shape(&mut self) {
        let (px, py) = self.pointer_device();
        let facts = PointerFacts {
            tab_dragging: self.tabs.dragging(),
            // A scrollbar is a control, not text, so over its grab band (and for as
            // long as its thumb is held) the I-beam gives way to the arrow.
            on_chrome: self.pointer_in_tab_bar()
                || self.pointer.drag_scroll
                || self.tabs.active().on_scrollbar(px, py),
            ctrl_link: self
                .tabs
                .active()
                .hovered_link()
                .filter(|_| self.xkb.ctrl_active()),
            reporting: self.tabs.active().mouse_reporting(),
            shift: self.xkb.shift_active(),
        };
        let preview = link_preview_shown(facts);
        if preview != self.link_preview {
            self.link_preview = preview;
            self.tabs.mark_dirty();
        }
        let shape = pointer_shape_for(facts);
        if shape == self.pointer.shape {
            return;
        }
        if self.objects.cursor_shape_device != 0 && self.pointer.enter_serial != 0 {
            self.conn.request(
                self.objects.cursor_shape_device,
                wp_cursor_shape_device_v1::SET_SHAPE,
                &[Arg::Uint(self.pointer.enter_serial), Arg::Uint(shape)],
            );
            self.pointer.shape = shape;
        }
    }

    /// Tell the terminal the pointer is no longer over its grid, because it left the
    /// surface or crossed into the tab strip. It drops the hovered hyperlink, so an
    /// underline never outlives the pointer that summoned it.
    fn pointer_left_grid(&mut self) -> Result<()> {
        // The scrollbar goes with it: a bar left expanded under a pointer that is no
        // longer there would never shrink back.
        self.tabs.active_mut().track_scrollbar(None);
        let mods = self.current_mods();
        self.tabs.active_mut().pointer(PointerEvent::Left, mods)?;
        Ok(())
    }

    /// End the press gesture the window owns, if one is live: a press in the tab strip
    /// with the tab drag it armed, or a scrollbar thumb drag. Without its release the
    /// gesture would follow a pointer with no button held, and the next release would
    /// be taken for its end.
    fn end_pointer_gesture(&mut self) {
        if self.pointer.bar_button.take().is_some() {
            self.tabs.end_drag();
        }
        if std::mem::take(&mut self.pointer.drag_scroll) {
            self.tabs.active_mut().release_scrollbar();
        }
    }

    /// The click multiplicity of a left press: 1, 2, or 3 for successive presses on
    /// the same cell within [`MULTI_CLICK_MS`], cycling back to 1 past a triple so a
    /// fourth click starts fresh. A press elsewhere or after the window restarts the
    /// count. `wrapping_sub` keeps the comparison correct across the wl clock's u32
    /// wrap.
    fn click_count(&mut self, time: u32, cell: (usize, usize)) -> usize {
        let quick = time.wrapping_sub(self.pointer.last_click_time) <= MULTI_CLICK_MS;
        self.pointer.click_count =
            if quick && self.pointer.last_click_cell == cell && self.pointer.click_count < 3 {
                self.pointer.click_count + 1
            } else {
                1
            };
        self.pointer.last_click_time = time;
        self.pointer.last_click_cell = cell;
        self.pointer.click_count
    }

    /// One wl_pointer event: track the position, and either report to the child or
    /// drive local selection/scroll.
    pub(super) fn on_pointer(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_pointer::EV_ENTER => {
                // The enter serial is the one `set_shape` must cite; a fresh enter is
                // also where the compositor resets the cursor, so forget the last
                // shape and re-apply whichever fits the entry position.
                self.pointer.enter_serial = r.u32()?;
                let _surface = r.u32()?;
                self.pointer.x = r.fixed()?;
                self.pointer.y = r.fixed()?;
                self.pointer.shape = 0;
                self.update_pointer_shape();
            }
            wl_pointer::EV_LEAVE => {
                let _serial = r.u32()?;
                let _surface = r.u32()?;
                // A held button keeps the pointer on the surface, so a leave mid-gesture
                // is a compositor grab that took the release with it.
                self.end_pointer_gesture();
                // The pointer is gone, so nothing is under it: drop any hovered link.
                self.pointer_left_grid()?;
            }
            wl_pointer::EV_MOTION => {
                let _time = r.u32()?;
                self.pointer.x = r.fixed()?;
                self.pointer.y = r.fixed()?;
                let (px, py) = self.pointer_device();
                if self.pointer.drag_scroll {
                    // A thumb the pointer is holding follows it anywhere, including out
                    // of the lane and off the grid. Nothing else sees the move.
                    self.tabs.active_mut().drag_scrollbar(py);
                } else if self.tabs.dragging() {
                    // A tab drag owns the motion: it reorders along x and ignores y, so
                    // dragging below the strip keeps reordering rather than dropping the
                    // tab (what every tabbed app does). The grid must not see this, or it
                    // would start a text selection or emit a mouse report mid-drag.
                    self.tabs.drag_to(self.pointer_device().0 as i32);
                } else if self.pointer_in_tab_bar() {
                    self.pointer_left_grid()?;
                } else {
                    // The bar only lights and grows here; the grid still gets the move,
                    // because the bar is an overlay and a live text drag runs under it.
                    self.tabs.active_mut().track_scrollbar(Some((px, py)));
                    let (col, row, side) = self.pointer_cell();
                    let mods = self.current_mods();
                    self.tabs
                        .active_mut()
                        .pointer(PointerEvent::Motion { col, row, side }, mods)?;
                }
                // After the terminal has seen the move, so the shape reflects the link
                // now under the pointer: the arrow over the strip, the hand over a
                // Ctrl+clickable link, the I-beam over plain text.
                self.update_pointer_shape();
            }
            wl_pointer::EV_BUTTON => {
                self.last_serial = r.u32()?;
                let time = r.u32()?;
                let button = r.u32()?;
                let pressed = r.u32()? == wl_pointer::BUTTON_STATE_PRESSED;
                let mapped = pointer_button(button);
                if !pressed && self.pointer.bar_button == mapped {
                    // The release that matches a bar press ends any drag it armed. This
                    // keys off the pressed button, not the pointer position, so a
                    // release dragged off the strip still settles the tab.
                    self.tabs.end_drag();
                    self.pointer.bar_button = None;
                    self.update_pointer_shape();
                    return Ok(());
                }
                // The release that ends a thumb drag is the scrollbar's, wherever the
                // pointer has wandered to by now: it must not reach the grid as the end
                // of a selection that never began.
                if !pressed && self.pointer.drag_scroll && mapped == Some(MouseButton::Left) {
                    self.pointer.drag_scroll = false;
                    self.tabs.active_mut().release_scrollbar();
                    let at = self.pointer_device();
                    self.tabs.active_mut().track_scrollbar(Some(at));
                    self.update_pointer_shape();
                    return Ok(());
                }
                if self.pointer_in_tab_bar() && pressed {
                    self.pointer.bar_button = mapped;
                    if let (Some(button), Some(id)) = (mapped, self.pointer_bar_tab()) {
                        match button {
                            MouseButton::Left => {
                                // Press selects immediately (unchanged) and arms a
                                // drag; the tab does not lift until the pointer travels
                                // past the threshold, so a plain click stays a click.
                                self.tabs.select(id, self.window_focused);
                                self.tabs.begin_drag(id, self.pointer_device().0 as i32);
                            }
                            MouseButton::Middle => {
                                self.closed = self.tabs.close(id, self.window_focused);
                            }
                            _ => {}
                        }
                    }
                    return Ok(());
                }
                // A left press in the lane takes the thumb before the grid beneath it
                // sees it: the gesture is a scrub, not a selection. Only a scrollable tab
                // has a lane, so this can never shadow a click on the alt screen.
                if pressed && mapped == Some(MouseButton::Left) {
                    let (px, py) = self.pointer_device();
                    if self.tabs.active_mut().press_scrollbar(px, py) {
                        self.pointer.drag_scroll = true;
                        return Ok(());
                    }
                }
                // A release whose press began in the grid must still finish that
                // gesture (selection or child mouse reporting), even if the pointer
                // has since crossed into the bar.
                // Map the raw Wayland button to ours (window-side); ignore unmapped.
                if let Some(button) = mapped {
                    let (col, row, side) = self.pointer_cell();
                    // A left press carries its click multiplicity (word/line select);
                    // any other event is a plain single.
                    let count = if button == MouseButton::Left && pressed {
                        self.click_count(time, (col, row))
                    } else {
                        1
                    };
                    let mods = self.current_mods();
                    self.tabs.active_mut().pointer(
                        PointerEvent::Button {
                            button,
                            pressed,
                            col,
                            row,
                            count,
                            side,
                        },
                        mods,
                    )?;
                }
            }
            wl_pointer::EV_AXIS => {
                let _time = r.u32()?;
                let axis = r.u32()?;
                let value = r.fixed()?;
                if axis == wl_pointer::AXIS_VERTICAL_SCROLL && !self.pointer_in_tab_bar() {
                    self.on_wheel(value)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// The mouse wheel: report it to a program that grabbed the mouse; else scroll
    /// the local scrollback view; else (the alt screen, no history) send arrow
    /// keys, the conventional fallback so wheel-scrolling `less`/`man` works.
    /// `value` is the wl_fixed vertical delta (positive = down).
    fn on_wheel(&mut self, value: f32) -> Result<()> {
        // Gather fractional deltas (touchpads send many small ones) into whole
        // notches, then hand them to the terminal (which reports them, scrolls the
        // view, or sends arrow keys; see `TerminalCore::pointer`).
        let notches = wheel_notches(&mut self.pointer.axis_accum, value);
        if notches == 0 {
            return Ok(());
        }
        let (col, row, _) = self.pointer_cell();
        let mods = self.current_mods();
        self.tabs.active_mut().pointer(
            PointerEvent::Wheel {
                down: notches > 0,
                notches: notches.unsigned_abs(),
                col,
                row,
            },
            mods,
        )?;
        Ok(())
    }
}

/// What the pointer is over and who owns it, gathered from the window, the tab strip,
/// and the active terminal so that [`pointer_shape_for`] stays a pure function of the
/// facts (and thus testable without a compositor to enter or a program to run).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PointerFacts {
    /// A tab is held and travelling along the strip.
    tab_dragging: bool,
    /// The pointer is over chrome rather than the grid: the tab strip, or the
    /// scrollbar's grab band (including a thumb held anywhere).
    on_chrome: bool,
    /// Ctrl is held over a hyperlink, with the verdict that decides whether a click
    /// follows it.
    ctrl_link: Option<Verdict>,
    /// A program has grabbed the mouse with `?1000`/`?1002`/`?1003`.
    reporting: bool,
    /// Shift is held, which forces the gesture back to local selection.
    shift: bool,
}

/// Pick the pointer shape the facts call for, most specific claim first: a held tab
/// owns the pointer outright, then chrome, then a program reporting the mouse, then a
/// link under Ctrl (the hand if it opens, `not_allowed` if it is refused), and failing
/// all of those the grid is plain selectable text.
///
/// The ordering is the point. Chrome outranks `reporting` because the strip and the
/// scrollbar are ours no matter what the child asked for. `reporting` outranks a link
/// because a program grabbing the mouse owns its clicks, Ctrl+click included, until
/// Shift takes the gesture back; a hover left over from before it grabbed the mouse
/// must not promise a click it will not get.
fn pointer_shape_for(f: PointerFacts) -> u32 {
    if f.tab_dragging {
        // The grabbing hand is the whole visual affordance of a tab drag, and it wins
        // over the strip's own arrow.
        wp_cursor_shape_device_v1::SHAPE_GRABBING
    } else if f.on_chrome {
        wp_cursor_shape_device_v1::SHAPE_DEFAULT
    } else if f.reporting && !f.shift {
        // The grid belongs to a program grabbing the mouse, so a drag is its event to
        // interpret, not a selection. Shift is the same override the terminal half
        // applies to the event itself, so the two never disagree about who the next
        // drag belongs to.
        wp_cursor_shape_device_v1::SHAPE_DEFAULT
    } else if f.ctrl_link == Some(Verdict::Open) {
        wp_cursor_shape_device_v1::SHAPE_POINTER
    } else if f.ctrl_link.is_some() {
        // Only the terminal sets the pointer, so a program cannot draw its way past
        // this the way it can fake an underline.
        wp_cursor_shape_device_v1::SHAPE_NOT_ALLOWED
    } else {
        wp_cursor_shape_device_v1::SHAPE_TEXT
    }
}

/// Whether the hovered link's preview shows: exactly when the pointer says a Ctrl+click
/// acts on a link, so the preview never names a target the click would not reach.
fn link_preview_shown(f: PointerFacts) -> bool {
    matches!(
        pointer_shape_for(f),
        wp_cursor_shape_device_v1::SHAPE_POINTER | wp_cursor_shape_device_v1::SHAPE_NOT_ALLOWED
    )
}

/// A wheel notch, in **logical pixels**. Not `wl_fixed` units: [`Reader::fixed`] has
/// already divided the wire's 24.8 fixed-point value by 256 by the time an axis delta
/// reaches here, so `value` is pixels and so is this. The constant was always right and
/// the comment beside it named the wrong unit, which is precisely the
/// measured-the-wrong-thing hazard the house rules are about.
const WHEEL_STEP: f32 = 15.0;

/// Fold one axis delta into the pending wheel travel, returning the whole notches to
/// emit (positive = down) and leaving the remainder behind.
///
/// The accumulator exists because a touchpad sends many small deltas where a wheel sends
/// one big one, and a terminal scrolls in whole lines either way.
///
/// **A direction change starts over**, and that is the part worth stating: the
/// accumulator holds *travel that has not yet become a notch*, and travel one way says
/// nothing about travel the other. Scroll down until it holds 14 — below the threshold,
/// so nothing has been emitted — then reverse, and an up-notch has to climb from +14 all
/// the way to −15. That is 29 units of upward travel for the first notch, so the first
/// two flicks of the reversal do nothing at all. And with no `axis_stop` handling the
/// leftover survives *between* gestures, so 14/15 of a notch from minutes ago makes the
/// next single flick overshoot.
fn wheel_notches(accum: &mut f32, value: f32) -> i32 {
    if value != 0.0 && *accum != 0.0 && (value < 0.0) != (*accum < 0.0) {
        *accum = 0.0;
    }
    *accum += value;
    let notches = (*accum / WHEEL_STEP) as i32;
    *accum -= notches as f32 * WHEEL_STEP;
    notches
}

/// Map a grid-local device x (already inset past the padding, floored at zero) to
/// the column under it, clamped into the grid, and the [`Side`] of that column's
/// midline it falls on. The side is measured against the *clamped* column, so a
/// pointer past the last column reads as its right half and a selection dragged
/// off the edge runs to the end of the row rather than stopping a cell short.
fn column_under(px: f32, cell_w: i32, cols: usize) -> (usize, Side) {
    let x = px / cell_w as f32;
    let col = (x as usize).min(cols.saturating_sub(1));
    let side = if x - col as f32 >= 0.5 {
        Side::Right
    } else {
        Side::Left
    };
    (col, side)
}

/// Map a Wayland pointer button (a `linux/input-event-codes.h` code) to the
/// logical button the mouse encoder speaks. Unknown buttons (side buttons) are
/// ignored.
fn pointer_button(code: u32) -> Option<MouseButton> {
    match code {
        protocol::wl_pointer::BTN_LEFT => Some(MouseButton::Left),
        BTN_MIDDLE => Some(MouseButton::Middle),
        BTN_RIGHT => Some(MouseButton::Right),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::app::seat::*;
    use crate::app::tests::{event, window};
    use crate::platform::verdict::Refusal;

    /// The pointer resting on plain, unclaimed grid text.
    const OVER_TEXT: PointerFacts = PointerFacts {
        tab_dragging: false,
        on_chrome: false,
        ctrl_link: None,
        reporting: false,
        shift: false,
    };

    #[test]
    fn the_wheel_accumulator_starts_over_when_the_direction_flips() {
        // The accumulator holds travel that has not yet become a notch, and travel one
        // way says nothing about travel the other.
        let mut accum = 0.0;

        // Almost a notch down: nothing emitted, 14 pending.
        assert_eq!(wheel_notches(&mut accum, 14.0), 0);
        assert_eq!(accum, 14.0);

        // Now reverse. Carrying the +14 forward, the first up-notch would need 29 units
        // of upward travel and the first two flicks of the reversal would do nothing —
        // the scroll wheel simply not responding, right after a scroll the other way.
        assert_eq!(
            wheel_notches(&mut accum, -15.0),
            -1,
            "one flick up is one notch up, whatever was pending downward"
        );

        // And a leftover cannot survive a reversal into the next gesture either. With no
        // `axis_stop` to clear it, 14/15 of a notch from minutes ago would make the next
        // single flick the other way overshoot.
        let mut accum = 0.0;
        assert_eq!(wheel_notches(&mut accum, 14.0), 0);
        assert_eq!(wheel_notches(&mut accum, -14.0), 0, "under the threshold");
        assert_eq!(accum, -14.0, "and the down travel is gone, not banked");
    }

    #[test]
    fn the_wheel_gathers_fractional_deltas_and_keeps_the_remainder() {
        // The reason the accumulator exists: a touchpad sends many small deltas where a
        // wheel sends one 15. Same direction throughout, so nothing is discarded.
        let mut accum = 0.0;
        for _ in 0..4 {
            assert_eq!(wheel_notches(&mut accum, 3.0), 0);
        }
        assert_eq!(wheel_notches(&mut accum, 3.0), 1, "12 + 3 is one notch");
        assert_eq!(accum, 0.0);

        // A big delta yields every notch in it and banks the rest.
        let mut accum = 0.0;
        assert_eq!(wheel_notches(&mut accum, 47.0), 3);
        assert!((accum - 2.0).abs() < 1e-4, "remainder kept: {accum}");

        // Upward is symmetric.
        let mut accum = 0.0;
        assert_eq!(wheel_notches(&mut accum, -47.0), -3);
        assert!((accum + 2.0).abs() < 1e-4, "remainder kept: {accum}");

        // A zero delta is not a direction, so it neither emits nor resets.
        let mut accum = 0.0;
        assert_eq!(wheel_notches(&mut accum, 7.0), 0);
        assert_eq!(wheel_notches(&mut accum, 0.0), 0);
        assert_eq!(accum, 7.0);
    }

    #[test]
    fn a_pointer_rounds_to_the_nearer_cell_edge() {
        // Cell width 8: x 0..4 is the left half of column 0, 4..8 the right, and
        // the dead-center pixel rounds forward like a text caret.
        assert_eq!(column_under(0.0, 8, 80), (0, Side::Left));
        assert_eq!(column_under(3.9, 8, 80), (0, Side::Left));
        assert_eq!(column_under(4.0, 8, 80), (0, Side::Right));
        assert_eq!(column_under(7.9, 8, 80), (0, Side::Right));
        assert_eq!(column_under(8.0, 8, 80), (1, Side::Left));
    }

    #[test]
    fn a_pointer_past_the_grid_clamps_to_the_last_column_right_half() {
        // Beyond the last column the clamp keeps the cell and the side reads
        // right, so a drag off the edge reaches the end of the row rather than
        // stopping a cell short.
        assert_eq!(column_under(640.0, 8, 80), (79, Side::Right));
        assert_eq!(column_under(9999.0, 8, 80), (79, Side::Right));
        // A degenerate grid clamps too, never panics.
        assert_eq!(column_under(5.0, 8, 0), (0, Side::Right));
    }

    #[test]
    fn a_plain_grid_offers_the_i_beam() {
        assert_eq!(
            pointer_shape_for(OVER_TEXT),
            wp_cursor_shape_device_v1::SHAPE_TEXT
        );
        // Shift alone changes nothing: there is no program to take the grid back from.
        assert_eq!(
            pointer_shape_for(PointerFacts {
                shift: true,
                ..OVER_TEXT
            }),
            wp_cursor_shape_device_v1::SHAPE_TEXT
        );
    }

    #[test]
    fn a_program_grabbing_the_mouse_drops_the_i_beam() {
        // A TUI (vim, tmux, hunk) reporting the mouse owns the drag, so the pointer
        // must stop promising a selection: the arrow, as every other terminal shows.
        assert_eq!(
            pointer_shape_for(PointerFacts {
                reporting: true,
                ..OVER_TEXT
            }),
            wp_cursor_shape_device_v1::SHAPE_DEFAULT
        );
        // Shift is the override that hands the grid back, and the I-beam returns with
        // it, matching the `reporting` test the terminal applies to the event itself.
        assert_eq!(
            pointer_shape_for(PointerFacts {
                reporting: true,
                shift: true,
                ..OVER_TEXT
            }),
            wp_cursor_shape_device_v1::SHAPE_TEXT
        );
    }

    #[test]
    fn chrome_and_a_reporting_program_outrank_a_link() {
        // The strip and the scrollbar are ours whatever the child asked for.
        assert_eq!(
            pointer_shape_for(PointerFacts {
                on_chrome: true,
                reporting: true,
                ..OVER_TEXT
            }),
            wp_cursor_shape_device_v1::SHAPE_DEFAULT
        );
        // A program grabbing the mouse owns its clicks, Ctrl+click included, so a hover
        // left over from before it grabbed the mouse does not get the hand.
        for verdict in [Verdict::Open, Verdict::Refused(Refusal::Scheme)] {
            assert_eq!(
                pointer_shape_for(PointerFacts {
                    ctrl_link: Some(verdict),
                    reporting: true,
                    ..OVER_TEXT
                }),
                wp_cursor_shape_device_v1::SHAPE_DEFAULT
            );
        }
        // Shift takes the gesture back, and the link with it.
        assert_eq!(
            pointer_shape_for(PointerFacts {
                ctrl_link: Some(Verdict::Open),
                reporting: true,
                shift: true,
                ..OVER_TEXT
            }),
            wp_cursor_shape_device_v1::SHAPE_POINTER
        );
        assert_eq!(
            pointer_shape_for(PointerFacts {
                ctrl_link: Some(Verdict::Refused(Refusal::Scheme)),
                reporting: true,
                shift: true,
                ..OVER_TEXT
            }),
            wp_cursor_shape_device_v1::SHAPE_NOT_ALLOWED
        );
        // A held tab beats everything, including the strip it is travelling over.
        assert_eq!(
            pointer_shape_for(PointerFacts {
                tab_dragging: true,
                on_chrome: true,
                ctrl_link: Some(Verdict::Open),
                reporting: true,
                shift: true,
            }),
            wp_cursor_shape_device_v1::SHAPE_GRABBING
        );
    }

    #[test]
    fn the_preview_shows_exactly_where_a_ctrl_click_acts_on_a_link() {
        let link = |verdict| PointerFacts {
            ctrl_link: Some(verdict),
            ..OVER_TEXT
        };
        assert!(link_preview_shown(link(Verdict::Open)));
        assert!(link_preview_shown(link(Verdict::Refused(Refusal::Scheme))));
        assert!(!link_preview_shown(OVER_TEXT), "no link under Ctrl");
        for (facts, why) in [
            (
                PointerFacts {
                    on_chrome: true,
                    ..link(Verdict::Open)
                },
                "over the scrollbar, which takes the click",
            ),
            (
                PointerFacts {
                    reporting: true,
                    ..link(Verdict::Open)
                },
                "a program grabbing the mouse takes the click",
            ),
            (
                PointerFacts {
                    tab_dragging: true,
                    ..link(Verdict::Open)
                },
                "a tab is held",
            ),
        ] {
            assert!(!link_preview_shown(facts), "{why}");
        }
        assert!(link_preview_shown(PointerFacts {
            reporting: true,
            shift: true,
            ..link(Verdict::Open)
        }));
    }

    #[test]
    fn the_shape_is_never_the_unset_zero() {
        // `PointerState::shape` 0 is the sentinel for "forget what was applied" on enter, so
        // no real choice may collide with it or the re-apply would be skipped.
        let links = [
            None,
            Some(Verdict::Open),
            Some(Verdict::Refused(Refusal::Scheme)),
        ];
        for bits in 0..16u8 {
            for ctrl_link in links {
                let facts = PointerFacts {
                    tab_dragging: bits & 1 != 0,
                    on_chrome: bits & 2 != 0,
                    ctrl_link,
                    reporting: bits & 4 != 0,
                    shift: bits & 8 != 0,
                };
                assert_ne!(pointer_shape_for(facts), 0, "{facts:?}");
            }
        }
    }

    /// Hand the window one `wl_pointer` event, its arguments encoded as the wire carries
    /// them: a `u32` each, a position in 24.8 fixed point.
    fn pointer_event(state: &mut State, opcode: u16, args: &[u32]) {
        let body: Vec<u8> = args.iter().flat_map(|a| a.to_ne_bytes()).collect();
        state
            .on_pointer(opcode, &mut Reader::new(&body))
            .expect("the pointer event");
    }

    fn fixed(v: f32) -> u32 {
        (v * 256.0) as i32 as u32
    }

    fn enter(state: &mut State, x: f32, y: f32) {
        pointer_event(state, wl_pointer::EV_ENTER, &[1, 0, fixed(x), fixed(y)]);
    }

    fn motion(state: &mut State, x: f32, y: f32) {
        pointer_event(state, wl_pointer::EV_MOTION, &[0, fixed(x), fixed(y)]);
    }

    fn press_left(state: &mut State) {
        let pressed = wl_pointer::BUTTON_STATE_PRESSED;
        pointer_event(
            state,
            wl_pointer::EV_BUTTON,
            &[2, 0, wl_pointer::BTN_LEFT, pressed],
        );
    }

    fn leave(state: &mut State) {
        pointer_event(state, wl_pointer::EV_LEAVE, &[3, 0]);
    }

    /// The seat loses its pointer and gets one back, as when the only mouse is unplugged
    /// and plugged in again. No leave comes first.
    fn replug_pointer(state: &mut State) {
        const SEAT: u32 = 90;
        state.objects.seat = Some(SEAT);
        if state.objects.pointer == 0 {
            event(
                state,
                SEAT,
                wl_seat::EV_CAPABILITIES,
                &[wl_seat::CAP_POINTER],
            );
        }
        event(state, SEAT, wl_seat::EV_CAPABILITIES, &[0]);
        assert_eq!(state.objects.pointer, 0, "the old pointer was released");
        event(
            state,
            SEAT,
            wl_seat::EV_CAPABILITIES,
            &[wl_seat::CAP_POINTER],
        );
    }

    type LosePointer = fn(&mut State);

    /// The ways a pointer goes away mid-gesture without sending the release: a compositor
    /// grab (an interactive move, a popup) sends `leave` and keeps the release for
    /// itself, and a pointer the seat loses sends nothing more at all.
    const POINTER_LOSSES: [(LosePointer, &str); 2] =
        [(leave, "a leave"), (replug_pointer, "losing the pointer")];

    /// The tab drag has to end with the pointer, or it keeps reordering under a pointer
    /// with no button held, and the next left release is swallowed as its end.
    #[test]
    fn losing_the_pointer_ends_a_tab_drag() {
        for (lose, how) in POINTER_LOSSES {
            let (mut state, _compositor) = window(&["one", "two", "three"]);
            let grid = state.layout.terminal;
            let bar_y = (state.layout.bar_y + state.layout.bar_h / 2) as f32;
            let first_x = (grid.pad + grid.metrics.w) as f32;
            enter(&mut state, first_x, bar_y);
            press_left(&mut state);
            assert!(state.tabs.dragging(), "the press on a tab armed a drag");
            let first = state.tabs.tab_at_bar_col(1);

            lose(&mut state);
            enter(&mut state, first_x, bar_y);
            motion(&mut state, (grid.width - 1) as f32, bar_y);
            assert_eq!(
                state.tabs.tab_at_bar_col(1),
                first,
                "after {how}, motion with no button held does not move a tab"
            );
            assert!(!state.tabs.dragging(), "{how} ended the drag");
            assert_eq!(
                state.pointer.bar_button, None,
                "after {how}, no release is owed to the strip"
            );
        }
    }

    /// The same cut-off for a scrollbar thumb drag: once the pointer has gone, moving one
    /// over the window again must not scroll the view.
    #[test]
    fn losing_the_pointer_ends_a_scrollbar_drag() {
        for (lose, how) in POINTER_LOSSES {
            let (mut state, _compositor) = window(&["one"]);
            let history: String = (0..300).map(|i| format!("line {i}\r\n")).collect();
            state.tabs.active_mut().feed_test_bytes(history.as_bytes());
            let grid = state.layout.terminal;
            let y = (grid.height / 2) as f32;
            let lane_x = (0..grid.width)
                .rev()
                .map(|x| x as f32)
                .find(|&x| state.tabs.active().on_scrollbar(x, y))
                .expect("a scrollbar lane");
            enter(&mut state, lane_x, y);
            press_left(&mut state);
            assert!(state.pointer.drag_scroll, "the press took the thumb");

            let resting = state.tabs.active().scroll_position();

            lose(&mut state);
            enter(&mut state, lane_x, y);
            motion(&mut state, lane_x, 1.0);
            assert_eq!(
                state.tabs.active().scroll_position(),
                resting,
                "after {how}, motion with no button held does not scroll"
            );
            assert!(!state.pointer.drag_scroll, "{how} let go of the thumb");
        }
    }
}
