//! The terminal's orchestration: it binds the compositor globals, builds the
//! xdg-shell surface stack, brings up the Vulkan/dmabuf presentation path,
//! spawns the shell on a PTY, and runs the event loop that ties them together.
//!
//! ```text
//!            ┌───────────── wl_keyboard ──▶ xkb ──▶ input::encode ──▶ PTY write
//!   poll ───▶│
//!            └── PTY master read ──▶ vt::Parser ──▶ grid::Screen ──▶ term_render ──▶ GPU
//! ```
//!
//! One [`State`] holds every protocol id and the render/PTY resources.
//! [`State::run_until`] is the drain-render-wait loop: it services Wayland events
//! and PTY output, paints when the grid changed (paced to the compositor's frame
//! callback), then blocks in a single `poll` over the Wayland socket *and* the
//! PTY master until either has more to say. Keyboard input is encoded by the
//! tested [`crate::input`] table and written back to the child; a resize divides
//! the new pixel size into cells, resizes the grid, and sends the child
//! `TIOCSWINSZ`. The GPU-facing half lives in the [`present`] submodule.
//!
//! `run_demo` keeps the phase-2 static grid (no PTY, no shell) for isolating a
//! render question from the live pipeline.

mod clipboard;
mod present;

use std::os::fd::AsRawFd;
use std::rc::Rc;
use std::time::{Duration, Instant};

use self::clipboard::ClipboardState;
use self::present::GpuPresentation;
use crate::color::Theme;
// The app orchestrates the platform/render layers (which carry their own error
// type) and the terminal core (which uses the crate-level one). It speaks the
// crate-level `Error`/`Result` throughout; a `?` on a platform call converts
// through the `From` bridge in `crate::error`, so there is one error type here.
use crate::error::{Error, Result};
use crate::grid::{CursorStyle, Screen};
use crate::input;
use crate::mouse::{self, MouseButton, MouseKind};
use crate::platform::conn::{Connection, Fill};
use crate::platform::ffi;
use crate::platform::freetype::Fonts;
use crate::platform::protocol::{
    self, wl_buffer, wl_callback, wl_compositor, wl_data_device_manager, wl_data_offer, wl_display,
    wl_keyboard, wl_pointer, wl_registry, wl_seat, wl_surface, xdg_surface, xdg_toplevel,
    xdg_wm_base,
};
use crate::platform::wire::{Arg, Message, Reader};
use crate::platform::xkb::Xkb;
use crate::pty::{self, Pty, ReadOutcome};
use crate::render::display::DisplayList;
use crate::term_render::{self, CellMetrics, CursorRender, CursorShape, Selection};
use crate::vt::Parser;

/// The default pixel size the terminal font is opened at, overridable with the
/// `BNKTERM_FONT_SIZE` env var (clamped to [`FONT_SIZE_RANGE`]).
const FONT_SIZE: u32 = 16;

/// The sane range a configured font size is clamped to.
const FONT_SIZE_RANGE: std::ops::RangeInclusive<u32> = 6..=72;

/// The cursor blink half-period: how long each of the on/off phases lasts.
const BLINK_INTERVAL: Duration = Duration::from_millis(530);

/// The grid the window opens at, before the compositor sends a size. Classic
/// 80x24; the surface then resizes to whatever the compositor grants.
const DEFAULT_COLS: usize = 80;
const DEFAULT_ROWS: usize = 24;

/// The PTY read chunk: large so a burst of output drains in few syscalls.
const PTY_READ_CHUNK: usize = 64 * 1024;

/// Lines the scrollback view moves per wheel notch, and arrows sent per notch
/// when the wheel falls back to arrow keys on the alt screen.
const WHEEL_LINES: usize = 3;

/// Right mouse button (`BTN_RIGHT`) and middle (`BTN_MIDDLE`) from
/// `linux/input-event-codes.h`; `BTN_LEFT` is in `protocol`.
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;

/// Upper bound on either surface dimension, applied to a compositor-supplied
/// configure size. Far beyond any real display, it just keeps a bogus or hostile
/// size from blowing up the buffer-size arithmetic.
const MAX_DIMENSION: u32 = 16384;

/// First object id in Wayland's server-allocated range. Client-created ids stay
/// below this; the free list only ever recycles client ids.
const SERVER_ID_BASE: u32 = 0xff00_0000;

/// Open the window, spawn `$SHELL`, and run the terminal until the shell exits or
/// the window is closed.
pub fn run() -> crate::error::Result<()> {
    let mut state = State::new(false)?;
    state.bring_up()?;
    Ok(())
}

/// `--demo`: open the window on a static styled grid, without a PTY. The phase-2
/// bring-up, kept for isolating a render question from the live shell.
pub fn run_demo() -> crate::error::Result<()> {
    let mut state = State::new(true)?;
    state.bring_up()?;
    Ok(())
}

/// `--gpu-probe`: report what the compositor and GPU offer the dmabuf
/// presentation path, without opening a window.
pub fn gpu_probe() -> crate::error::Result<()> {
    State::new(true)?.probe_dmabuf()?;
    Ok(())
}

struct State {
    conn: Connection,
    fonts: Fonts,
    xkb: Xkb,
    theme: Theme,
    /// The fixed cell box the whole grid is laid out on.
    metrics: CellMetrics,
    /// The grid being shown, always sized to the current window's `(cols, rows)`.
    screen: Screen,
    /// The VT state machine driving `screen` from the child's output bytes.
    parser: Parser,
    /// The child on the far side of the PTY; `None` in demo mode (and before the
    /// first configure, since the PTY is sized to the granted window).
    pty: Option<Pty>,
    /// Demo mode: a static grid, no shell.
    demo: bool,
    /// Reused PTY read buffer (allocated once, not per drain).
    pty_read_buf: Vec<u8>,
    /// Reused key-encoding buffer (allocated once, not per key press).
    key_buf: Vec<u8>,
    /// Whether the surface holds keyboard focus, so the cursor draws solid when
    /// focused and hollow when not.
    focused: bool,
    /// Cursor blink: the current on/off phase, and when it next toggles (`None`
    /// when not blinking, e.g. unfocused). Activity resets it to on.
    blink_on: bool,
    blink_at: Option<Instant>,
    /// Key auto-repeat: Wayland delivers no repeat events, so the client synthesises
    /// them from the compositor's `repeat_info`. `repeat_key`/`repeat_at` track the
    /// held key and when it next fires; `repeat_interval` is `None` when repeat is
    /// disabled. `repeat_delay` is the wait before the first repeat.
    repeat_delay: Duration,
    repeat_interval: Option<Duration>,
    repeat_key: Option<u32>,
    repeat_at: Option<Instant>,
    /// The toplevel title last sent, so it is only re-set when it changes.
    title: String,

    /// Current surface size in pixels.
    width: u32,
    height: u32,

    // Object ids. Globals are Option (discovered via the registry); ids we
    // create default to 0 (never valid) until assigned.
    registry: u32,
    compositor: Option<u32>,
    wm_base: Option<u32>,
    seat: Option<u32>,
    keyboard: u32,
    pointer: u32,
    /// Latest pointer position in surface pixels, and the button held for drag
    /// reporting (`None` when no button is down). `axis_accum` gathers fractional
    /// wheel deltas into whole notches.
    pointer_x: f32,
    pointer_y: f32,
    mouse_held: Option<MouseButton>,
    axis_accum: f32,
    /// The active text selection (a left-drag), or `None`. In display coords.
    selection: Option<Selection>,
    /// Whether a selection drag is in progress (the button is down).
    selecting: bool,
    surface: u32,
    xdg_surface: u32,
    toplevel: u32,
    /// The clipboard data device and manager (absent if the compositor has no
    /// `wl_data_device_manager`; copy/paste is then a no-op), plus our state.
    data_device_manager: Option<u32>,
    data_device: u32,
    clipboard: ClipboardState,
    /// The latest input-event serial, needed to claim the clipboard selection.
    last_serial: u32,
    /// GPU/dmabuf presentation resources.
    presentation: GpuPresentation,

    // Id allocation: a bump counter plus a free list fed by `delete_id`.
    next_id: u32,
    free_ids: Vec<u32>,

    // Loop state.
    pending_sync: Option<u32>,
    /// A compositor-requested size awaiting the next configure ack.
    pending_size: Option<(u32, u32)>,
    /// The serial of the latest xdg_surface.configure not yet acked; the ack is
    /// deferred to `render_frame` so it commits with the matching-size buffer.
    pending_configure: Option<u32>,
    configured: bool,
    closed: bool,
    /// The content changed and a frame should be drawn.
    dirty: bool,
    /// The in-flight frame callback's id, or 0 when none is pending. While it is
    /// nonzero the loop holds off redrawing, so bursts coalesce into at most one
    /// frame per refresh (frame pacing).
    frame_callback: u32,
    /// When set (the `BNKTERM_STATS` env var is present), each presented frame
    /// prints its render time and glyph-cache hit rate to stderr.
    stats: bool,
}

impl State {
    fn new(demo: bool) -> Result<Self> {
        let conn = Connection::connect()?;
        let font_size = config_font_size();
        let fonts = Fonts::new(&[font_size])?;
        let metrics = CellMetrics::from_fonts(&fonts, font_size);
        let (cols, rows) = (DEFAULT_COLS, DEFAULT_ROWS);
        let width = (cols as i32 * metrics.w).max(1) as u32;
        let height = (rows as i32 * metrics.h).max(1) as u32;
        // Demo mode shows a static grid; live mode starts blank and the shell
        // fills it once the PTY is spawned.
        let screen = if demo {
            demo_screen(cols, rows)
        } else {
            Screen::new(cols, rows)
        };

        Ok(Self {
            conn,
            fonts,
            xkb: Xkb::new()?,
            theme: Theme::default(),
            metrics,
            screen,
            parser: Parser::new(),
            pty: None,
            demo,
            pty_read_buf: vec![0u8; PTY_READ_CHUNK],
            key_buf: Vec::new(),
            focused: false,
            blink_on: true,
            blink_at: None,
            // Sensible defaults until the compositor sends repeat_info.
            repeat_delay: Duration::from_millis(400),
            repeat_interval: Some(Duration::from_millis(33)),
            repeat_key: None,
            repeat_at: None,
            title: String::new(),
            width,
            height,
            registry: 0,
            compositor: None,
            wm_base: None,
            seat: None,
            keyboard: 0,
            pointer: 0,
            pointer_x: 0.0,
            pointer_y: 0.0,
            mouse_held: None,
            axis_accum: 0.0,
            selection: None,
            selecting: false,
            surface: 0,
            xdg_surface: 0,
            toplevel: 0,
            data_device_manager: None,
            data_device: 0,
            clipboard: ClipboardState::new(),
            last_serial: 0,
            presentation: GpuPresentation::new(),
            next_id: 2, // 1 is wl_display
            free_ids: Vec::new(),
            pending_sync: None,
            pending_size: None,
            pending_configure: None,
            configured: false,
            closed: false,
            dirty: true,
            frame_callback: 0,
            stats: std::env::var_os("BNKTERM_STATS").is_some(),
        })
    }

    fn alloc_id(&mut self) -> u32 {
        self.free_ids.pop().unwrap_or_else(|| {
            let id = self.next_id;
            self.next_id += 1;
            id
        })
    }

    fn bring_up(&mut self) -> Result<()> {
        // Discover and bind globals.
        self.registry = self.alloc_id();
        self.conn.request(
            protocol::WL_DISPLAY,
            wl_display::GET_REGISTRY,
            &[Arg::NewId(self.registry)],
        );
        self.roundtrip()?;

        let compositor = self
            .compositor
            .ok_or_else(|| Error::msg("compositor has no wl_compositor"))?;
        let wm_base = self
            .wm_base
            .ok_or_else(|| Error::msg("compositor has no xdg_wm_base"))?;

        // The GPU (Vulkan + dmabuf) is mandatory: negotiate over dmabuf feedback
        // and bring Vulkan up, or exit with a clear message. The feedback must be
        // committed before init_gpu can allocate, so wait out the burst first.
        self.request_dmabuf_feedback();
        self.roundtrip()?;
        self.init_gpu()
            .map_err(|e| Error::msg(format!("requires a Vulkan device; none available: {e}")))?;

        // Surface stack: wl_surface -> xdg_surface -> xdg_toplevel.
        self.surface = self.alloc_id();
        self.conn.request(
            compositor,
            wl_compositor::CREATE_SURFACE,
            &[Arg::NewId(self.surface)],
        );
        self.xdg_surface = self.alloc_id();
        self.conn.request(
            wm_base,
            xdg_wm_base::GET_XDG_SURFACE,
            &[Arg::NewId(self.xdg_surface), Arg::Object(self.surface)],
        );
        self.toplevel = self.alloc_id();
        self.conn.request(
            self.xdg_surface,
            xdg_surface::GET_TOPLEVEL,
            &[Arg::NewId(self.toplevel)],
        );
        self.conn.request(
            self.toplevel,
            xdg_toplevel::SET_APP_ID,
            &[Arg::Str("bnkterm")],
        );
        self.refresh_title();

        // The data device drives the clipboard; skip it when the compositor has no
        // manager (copy/paste is then simply unavailable).
        if let (Some(manager), Some(seat)) = (self.data_device_manager, self.seat) {
            let device = self.alloc_id();
            self.conn.request(
                manager,
                wl_data_device_manager::GET_DATA_DEVICE,
                &[Arg::NewId(device), Arg::Object(seat)],
            );
            self.data_device = device;
        }

        self.create_buffers()?;
        self.init_explicit_sync();

        // Empty commit to get the initial configure; the loop draws once
        // `configured` is set.
        self.conn.request(self.surface, wl_surface::COMMIT, &[]);
        self.run_until(|s| s.configured)?;

        // Now that the window has its granted size, spawn the shell on a PTY
        // sized to the grid. Demo mode skips this and shows its static screen.
        if !self.demo {
            let (cols, rows) = self.screen.dimensions();
            self.pty = Some(Pty::spawn(cols, rows)?);
            eprintln!("bnkterm: shell on a {cols}x{rows} grid. Close the window to exit.");
        } else {
            eprintln!("bnkterm: phase-2 static demo. Close the window to exit.");
        }
        if let Some(gpu) = &self.presentation.backend {
            eprintln!("bnkterm: renderer: gpu ({})", gpu.name());
        }

        self.run_until(|s| s.closed)?;
        Ok(())
    }

    fn roundtrip(&mut self) -> Result<()> {
        let callback = self.alloc_id();
        self.conn.request(
            protocol::WL_DISPLAY,
            wl_display::SYNC,
            &[Arg::NewId(callback)],
        );
        self.pending_sync = Some(callback);
        self.run_until(|s| s.pending_sync.is_none())
    }

    /// Drain Wayland events and PTY output, service the timers, draw a frame if
    /// one is due, then block in one `poll` over both fds until either is ready or
    /// the soonest timer (cursor blink, key repeat) comes due. An idle, unfocused
    /// terminal with nothing held waits open-ended.
    fn run_until(&mut self, done: impl Fn(&State) -> bool) -> Result<()> {
        loop {
            while let Some(msg) = self.conn.next_message()? {
                self.handle(msg)?;
            }
            // Read a chunk of the child's output into the grid (a no-op with no
            // PTY, or when nothing is ready).
            self.pump_pty()?;
            // Fire any blink toggle or key repeat that has come due.
            self.service_timers()?;
            // Pace to the compositor: only draw when no frame callback is
            // outstanding, so a burst collapses into a single repaint.
            if self.configured && self.dirty && self.frame_callback == 0 && self.render_frame()? {
                self.dirty = false;
            }
            if done(self) {
                return Ok(());
            }
            self.conn.flush()?;
            let ready = pty::wait_readable(
                self.conn.fd(),
                self.pty.as_ref().map(Pty::fd),
                self.next_wake(),
            )?;
            if ready.wayland {
                // poll said the socket has data (or hung up); this recv returns
                // immediately, its short timeout only a safety net.
                if let Fill::Bytes(0) = self.conn.fill(Some(Duration::from_millis(50)))? {
                    return Err(Error::msg("compositor closed the connection"));
                }
            }
            // Due timers are serviced at the top of the next turn.
        }
    }

    /// Fire the cursor blink and key repeat if their deadlines have passed.
    fn service_timers(&mut self) -> Result<()> {
        let now = Instant::now();
        if self.cursor_blinking() && self.blink_at.is_some_and(|at| at <= now) {
            self.tick_blink();
        }
        if self.repeat_at.is_some_and(|at| at <= now) {
            self.fire_repeat()?;
        }
        Ok(())
    }

    /// How long to block for input: the soonest of the pending cursor-blink and
    /// key-repeat deadlines, or `None` (block indefinitely) when neither is armed.
    fn next_wake(&self) -> Option<Duration> {
        let now = Instant::now();
        let due = |at: Instant| {
            at.saturating_duration_since(now)
                .max(Duration::from_millis(1))
        };
        let blink = self.cursor_blinking().then_some(self.blink_at).flatten();
        [blink, self.repeat_at].into_iter().flatten().map(due).min()
    }

    /// Whether the cursor should be blinking right now: focused, visible, and the
    /// child asked for a blinking style.
    fn cursor_blinking(&self) -> bool {
        self.focused && self.screen.cursor_visible() && self.screen.cursor_blinks()
    }

    /// Flip the blink phase and schedule the next toggle.
    fn tick_blink(&mut self) {
        self.blink_on = !self.blink_on;
        self.blink_at = Some(Instant::now() + BLINK_INTERVAL);
        self.dirty = true;
    }

    /// Reset the cursor to its lit phase and restart the blink timer, so it shows
    /// solid immediately after activity (a keystroke, output) and blinks only when
    /// idle. A no-op's timer stays `None` while unfocused.
    fn bump_cursor(&mut self) {
        self.blink_on = true;
        self.blink_at = self.focused.then(|| Instant::now() + BLINK_INTERVAL);
    }

    /// Read one chunk of the child's output through the parser into the grid. A
    /// closed PTY (the shell exited) ends the session.
    fn pump_pty(&mut self) -> Result<()> {
        // Borrow the PTY and its buffer as distinct fields, so the read does not
        // conflict with the parser/screen borrows below.
        let outcome = match &self.pty {
            Some(pty) => pty.read(&mut self.pty_read_buf)?,
            None => return Ok(()),
        };
        match outcome {
            ReadOutcome::Data(n) => {
                self.parser
                    .advance_bytes(&mut self.screen, &self.pty_read_buf[..n]);
                // Answer any query the child made (DA/DSR): the grid queued the
                // reply bytes; write them back through the PTY.
                let responses = self.screen.take_responses();
                if !responses.is_empty() {
                    if let Some(pty) = &self.pty {
                        pty.write_all(&responses)?;
                    }
                }
                // New output snaps the view to the live bottom (xterm behavior),
                // so a stream of output always shows its latest line.
                self.screen.scroll_view_to_bottom();
                // The cells under any selection just changed meaning; drop it
                // rather than leave a highlight over stale content.
                self.selection = None;
                self.selecting = false;
                self.bump_cursor(); // output shows the cursor solid, then blinks
                self.dirty = true;
                // The child may have set its title via OSC 0/2.
                self.refresh_title();
            }
            ReadOutcome::WouldBlock => {}
            ReadOutcome::Eof => self.closed = true,
        }
        Ok(())
    }

    /// Compose the window's display list: the visible grid painted at the current
    /// size, cursor on top (solid when focused, hollow when not).
    fn build_frame_list(&mut self) -> Rc<DisplayList> {
        // The child chose the shape (DECSCUSR); blink hides it on the off phase
        // while focused, and DECTCEM hides it entirely.
        let blinked_off = self.cursor_blinking() && !self.blink_on;
        let cursor = CursorRender {
            shape: cursor_shape(self.screen.cursor_style()),
            visible: self.screen.cursor_visible() && !blinked_off,
            focused: self.focused,
        };
        let list = term_render::build_display_list(
            &self.screen,
            &self.theme,
            self.metrics,
            (self.width as i32, self.height as i32),
            cursor,
            self.selection,
        );
        Rc::new(list)
    }

    /// Record a new surface size, resize the grid to the cells that now fit, and
    /// tell the child (via `TIOCSWINSZ`, so it gets SIGWINCH and repaints). The
    /// GPU buffers are reallocated lazily in `render_frame`. A no-op if unchanged.
    fn resize_to(&mut self, w: u32, h: u32) {
        if (w, h) == (self.width, self.height) {
            return;
        }
        self.width = w;
        self.height = h;
        let (cols, rows) = self.metrics.columns_rows(w as i32, h as i32);
        if self.demo {
            self.screen = demo_screen(cols, rows);
        } else {
            self.screen.resize(cols, rows);
            if let Some(pty) = &self.pty {
                // Best-effort: a resize on a dead child errors, which the next
                // read surfaces as EOF and shuts the session down cleanly.
                let _ = pty.resize(cols, rows);
            }
        }
        self.dirty = true;
    }

    fn handle(&mut self, msg: Message) -> Result<()> {
        let mut r = Reader::new(&msg.body);

        if msg.object == protocol::WL_DISPLAY {
            match msg.opcode {
                wl_display::EV_ERROR => {
                    let object = r.u32()?;
                    let code = r.u32()?;
                    let message = r.string()?;
                    return Err(Error::msg(format!(
                        "wayland protocol error: object {object} code {code}: {message}"
                    )));
                }
                wl_display::EV_DELETE_ID => {
                    let id = r.u32()?;
                    if id < SERVER_ID_BASE {
                        self.free_ids.push(id);
                    }
                }
                _ => {}
            }
            return Ok(());
        }

        if msg.object == self.registry && msg.opcode == wl_registry::EV_GLOBAL {
            return self.on_global(&mut r);
        }

        if Some(msg.object) == self.pending_sync && msg.opcode == wl_callback::EV_DONE {
            self.pending_sync = None;
            return Ok(());
        }

        // The frame callback fired: the compositor is ready for the next frame.
        if self.frame_callback != 0
            && msg.object == self.frame_callback
            && msg.opcode == wl_callback::EV_DONE
        {
            self.frame_callback = 0;
            return Ok(());
        }

        if msg.object == self.xdg_surface && msg.opcode == xdg_surface::EV_CONFIGURE {
            self.pending_configure = Some(r.u32()?);
            if let Some((w, h)) = self.pending_size.take() {
                self.resize_to(w, h);
            }
            self.configured = true;
            self.dirty = true;
            return Ok(());
        }

        if msg.object == self.toplevel {
            match msg.opcode {
                xdg_toplevel::EV_CONFIGURE => {
                    let w = r.u32()?;
                    let h = r.u32()?;
                    let _states = r.array()?;
                    if w > 0 && h > 0 {
                        self.pending_size = Some((w.min(MAX_DIMENSION), h.min(MAX_DIMENSION)));
                    }
                }
                xdg_toplevel::EV_CLOSE => self.closed = true,
                _ => {}
            }
            return Ok(());
        }

        if Some(msg.object) == self.wm_base && msg.opcode == xdg_wm_base::EV_PING {
            let serial = r.u32()?;
            self.conn
                .request(msg.object, xdg_wm_base::PONG, &[Arg::Uint(serial)]);
            return Ok(());
        }

        if Some(msg.object) == self.seat && msg.opcode == wl_seat::EV_CAPABILITIES {
            let caps = r.u32()?;
            if caps & wl_seat::CAP_KEYBOARD != 0 && self.keyboard == 0 {
                let keyboard = self.alloc_id();
                self.conn
                    .request(msg.object, wl_seat::GET_KEYBOARD, &[Arg::NewId(keyboard)]);
                self.keyboard = keyboard;
            }
            if caps & wl_seat::CAP_POINTER != 0 && self.pointer == 0 {
                let pointer = self.alloc_id();
                self.conn
                    .request(msg.object, wl_seat::GET_POINTER, &[Arg::NewId(pointer)]);
                self.pointer = pointer;
            }
            return Ok(());
        }

        if self.keyboard != 0 && msg.object == self.keyboard {
            return self.on_keyboard(msg.opcode, &mut r);
        }

        if self.pointer != 0 && msg.object == self.pointer {
            return self.on_pointer(msg.opcode, &mut r);
        }

        if self.data_device != 0 && msg.object == self.data_device {
            return self.on_data_device(msg.opcode, &mut r);
        }

        if self.clipboard.source != 0 && msg.object == self.clipboard.source {
            return self.on_data_source(msg.opcode, &mut r);
        }

        if self.clipboard.incoming_offer != 0
            && msg.object == self.clipboard.incoming_offer
            && msg.opcode == wl_data_offer::EV_OFFER
        {
            return self.on_offer_mime(&mut r);
        }

        if self.presentation.feedback_id != 0 && msg.object == self.presentation.feedback_id {
            return self.on_dmabuf_feedback(msg.opcode, &mut r);
        }

        if msg.opcode == wl_buffer::EV_RELEASE {
            if msg.object == self.presentation.buffers[0] {
                self.presentation.busy[0] = false;
            } else if msg.object == self.presentation.buffers[1] {
                self.presentation.busy[1] = false;
            }
        }
        Ok(())
    }

    /// One wl_keyboard event: install the keymap, track focus and modifiers, and
    /// turn a key press into bytes for the child.
    fn on_keyboard(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_keyboard::EV_KEYMAP => {
                let format = r.u32()?;
                let size = r.u32()? as usize;
                let fd = self
                    .conn
                    .take_fd()
                    .ok_or_else(|| Error::msg("wl_keyboard.keymap arrived without its fd"))?;
                let bytes = ffi::read_mapped(fd.as_raw_fd(), size)?;
                self.xkb.load_keymap(&bytes, format)?;
            }
            wl_keyboard::EV_ENTER => {
                let _serial = r.u32()?;
                self.focused = true;
                self.bump_cursor(); // start blinking from a lit cursor
                self.dirty = true;
            }
            wl_keyboard::EV_LEAVE => {
                let _serial = r.u32()?;
                self.focused = false;
                self.blink_at = None; // stop the blink timer while unfocused
                self.stop_repeat(); // and drop any held-key repeat
                self.dirty = true;
            }
            wl_keyboard::EV_MODIFIERS => {
                let _serial = r.u32()?;
                let depressed = r.u32()?;
                let latched = r.u32()?;
                let locked = r.u32()?;
                let group = r.u32()?;
                self.xkb.update_modifiers(depressed, latched, locked, group);
            }
            wl_keyboard::EV_KEY => {
                self.last_serial = r.u32()?;
                let _time = r.u32()?;
                let keycode = r.u32()?;
                let key_state = r.u32()?;
                if key_state == wl_keyboard::KEY_STATE_PRESSED {
                    self.on_key_press(keycode)?;
                } else if self.repeat_key == Some(keycode) {
                    // The held key was released: stop repeating it.
                    self.stop_repeat();
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

    /// Cancel any auto-repeat in flight.
    fn stop_repeat(&mut self) {
        self.repeat_key = None;
        self.repeat_at = None;
    }

    /// Encode one key press and write it to the child. A named key (an arrow, a
    /// function key) is mapped by its keycode; anything else takes its layout
    /// character, and the [`crate::input`] encoder applies Ctrl/Alt. Modifiers
    /// come from xkb's current state.
    fn on_key_press(&mut self, keycode: u32) -> Result<()> {
        let mods = self.current_mods();
        // Ctrl+Shift+C/V copy and paste (the terminal convention, since bare
        // Ctrl+C/V are the interrupt and a control byte the child needs).
        if mods.contains(input::Mods::CTRL) && mods.contains(input::Mods::SHIFT) {
            if let Some(c) = self.xkb.key_char(keycode) {
                match c.to_ascii_lowercase() {
                    'c' => {
                        self.copy_selection();
                        return Ok(());
                    }
                    'v' => {
                        self.paste()?;
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }
        // Shift + Page/Home/End scrolls the scrollback view instead of reaching
        // the child (the alt screen has no history, so there it is a normal key).
        if self.handle_scroll_key(keycode, mods) {
            return Ok(());
        }
        // Send the key, and if it produced bytes and the keymap marks it
        // repeatable, arm auto-repeat on it.
        if self.send_key(keycode)? {
            self.arm_repeat(keycode);
        }
        Ok(())
    }

    /// Encode `keycode` under the current modifiers and write it to the child,
    /// returning whether any bytes were sent. Shared by the first press and each
    /// auto-repeat, so a held arrow repeats exactly as it first fired.
    fn send_key(&mut self, keycode: u32) -> Result<bool> {
        let mods = self.current_mods();
        let key = match input::key_from_keycode(keycode) {
            Some(named) => named,
            None => match self.xkb.key_char(keycode) {
                Some(c) => input::Key::Char(c),
                None => return Ok(false), // a modifier or unresolved key: nothing to send
            },
        };
        let modes = input::Modes::from_screen(&self.screen);
        self.key_buf.clear();
        input::encode(key, mods, modes, &mut self.key_buf);
        if self.key_buf.is_empty() {
            return Ok(false);
        }
        // Typing snaps the view back to the live bottom before the bytes go out,
        // so a keystroke never lands "blind" while reading history.
        if self.screen.is_scrolled() {
            self.screen.scroll_view_to_bottom();
            self.dirty = true;
        }
        self.bump_cursor(); // keep the cursor solid while typing
        if let Some(pty) = &self.pty {
            pty.write_all(&self.key_buf)?;
        }
        Ok(true)
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
    fn fire_repeat(&mut self) -> Result<()> {
        let (Some(keycode), Some(interval)) = (self.repeat_key, self.repeat_interval) else {
            self.repeat_at = None;
            return Ok(());
        };
        if self.send_key(keycode)? {
            self.repeat_at = Some(Instant::now() + interval);
        } else {
            self.repeat_key = None;
            self.repeat_at = None;
        }
        Ok(())
    }

    /// Intercept the scrollback-navigation chords (Shift + PageUp/PageDown/Home/
    /// End) on the primary screen, returning whether the key was consumed. A page
    /// is a screenful less one line, so a line of context carries across.
    fn handle_scroll_key(&mut self, keycode: u32, mods: input::Mods) -> bool {
        if !mods.contains(input::Mods::SHIFT) || self.screen.is_alt() {
            return false;
        }
        let (_, rows) = self.screen.dimensions();
        let page = rows.saturating_sub(1).max(1);
        match input::key_from_keycode(keycode) {
            Some(input::Key::PageUp) => self.screen.scroll_view_up(page),
            Some(input::Key::PageDown) => self.screen.scroll_view_down(page),
            Some(input::Key::Home) => self.screen.scroll_view_to_top(),
            Some(input::Key::End) => self.screen.scroll_view_to_bottom(),
            _ => return false,
        }
        self.dirty = true;
        true
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

    /// The cell under the pointer, clamped into the grid. Used for mouse reports.
    fn pointer_cell(&self) -> (usize, usize) {
        let (cols, rows) = self.screen.dimensions();
        let col = (self.pointer_x / self.metrics.w as f32) as usize;
        let row = (self.pointer_y / self.metrics.h as f32) as usize;
        (
            col.min(cols.saturating_sub(1)),
            row.min(rows.saturating_sub(1)),
        )
    }

    /// One wl_pointer event: track the position, and either report to the child or
    /// drive local selection/scroll.
    fn on_pointer(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_pointer::EV_ENTER => {
                let _serial = r.u32()?;
                let _surface = r.u32()?;
                self.pointer_x = r.fixed()?;
                self.pointer_y = r.fixed()?;
            }
            wl_pointer::EV_MOTION => {
                let _time = r.u32()?;
                self.pointer_x = r.fixed()?;
                self.pointer_y = r.fixed()?;
                if self.selecting {
                    self.extend_selection();
                } else {
                    self.report_mouse_motion()?;
                }
            }
            wl_pointer::EV_BUTTON => {
                self.last_serial = r.u32()?;
                let _time = r.u32()?;
                let button = r.u32()?;
                let pressed = r.u32()? == wl_pointer::BUTTON_STATE_PRESSED;
                self.on_pointer_button(button, pressed)?;
            }
            wl_pointer::EV_AXIS => {
                let _time = r.u32()?;
                let axis = r.u32()?;
                let value = r.fixed()?;
                if axis == wl_pointer::AXIS_VERTICAL_SCROLL {
                    self.on_wheel(value)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// A pointer button press/release. While a program is reporting the mouse (and
    /// Shift is not held to force local use), the event is sent to the child;
    /// otherwise a left button drives a local text selection.
    fn on_pointer_button(&mut self, button: u32, pressed: bool) -> Result<()> {
        let Some(btn) = pointer_button(button) else {
            return Ok(());
        };
        let mode = self.screen.mouse_mode();
        if mode.reports() && !self.xkb.shift_active() {
            self.mouse_held = pressed.then_some(btn);
            let kind = if pressed {
                MouseKind::Press
            } else {
                MouseKind::Release
            };
            let (col, row) = self.pointer_cell();
            self.key_buf.clear();
            if mouse::encode(
                mode,
                btn,
                kind,
                col,
                row,
                self.current_mods(),
                &mut self.key_buf,
            ) {
                if let Some(pty) = &self.pty {
                    pty.write_all(&self.key_buf)?;
                }
            }
        } else if btn == MouseButton::Left {
            // Local selection: press begins a fresh one, release ends the drag.
            if pressed {
                let (col, row) = self.pointer_cell();
                self.selection = Some(Selection {
                    anchor: (row, col),
                    head: (row, col),
                });
                self.selecting = true;
            } else {
                self.selecting = false;
            }
            self.dirty = true;
        }
        Ok(())
    }

    /// Extend the in-progress selection to the pointer's current cell.
    fn extend_selection(&mut self) {
        let (col, row) = self.pointer_cell();
        if let Some(sel) = self.selection.as_mut() {
            if sel.head != (row, col) {
                sel.head = (row, col);
                self.dirty = true;
            }
        }
    }

    /// Report pointer motion to a program that asked for it (drag under `?1002`,
    /// any move under `?1003`). Silent otherwise.
    fn report_mouse_motion(&mut self) -> Result<()> {
        let mode = self.screen.mouse_mode();
        if !mode.reports() || self.xkb.shift_active() {
            return Ok(());
        }
        let button = self.mouse_held.unwrap_or(MouseButton::None);
        let (col, row) = self.pointer_cell();
        self.key_buf.clear();
        if mouse::encode(
            mode,
            button,
            MouseKind::Motion,
            col,
            row,
            self.current_mods(),
            &mut self.key_buf,
        ) {
            if let Some(pty) = &self.pty {
                pty.write_all(&self.key_buf)?;
            }
        }
        Ok(())
    }

    /// The mouse wheel: report it to a program that grabbed the mouse; else scroll
    /// the local scrollback view; else (the alt screen, no history) send arrow
    /// keys, the conventional fallback so wheel-scrolling `less`/`man` works.
    /// `value` is the wl_fixed vertical delta (positive = down).
    fn on_wheel(&mut self, value: f32) -> Result<()> {
        // Gather fractional deltas (touchpads send many small ones) into notches.
        self.axis_accum += value;
        let step = 15.0; // a typical wheel notch in wl_fixed units
        let mut notches = (self.axis_accum / step) as i32;
        if notches == 0 {
            return Ok(());
        }
        self.axis_accum -= notches as f32 * step;
        let down = notches > 0;
        notches = notches.abs();

        let mode = self.screen.mouse_mode();
        if mode.reports() && !self.xkb.shift_active() {
            let button = if down {
                MouseButton::WheelDown
            } else {
                MouseButton::WheelUp
            };
            let (col, row) = self.pointer_cell();
            for _ in 0..notches {
                self.key_buf.clear();
                if mouse::encode(
                    mode,
                    button,
                    MouseKind::Press,
                    col,
                    row,
                    self.current_mods(),
                    &mut self.key_buf,
                ) {
                    if let Some(pty) = &self.pty {
                        pty.write_all(&self.key_buf)?;
                    }
                }
            }
        } else if !self.screen.is_alt() {
            let lines = notches as usize * WHEEL_LINES;
            if down {
                self.screen.scroll_view_down(lines);
            } else {
                self.screen.scroll_view_up(lines);
            }
            self.dirty = true;
        } else {
            // Alt screen without mouse reporting: wheel becomes arrow keys.
            let key = if down {
                input::Key::Down
            } else {
                input::Key::Up
            };
            let modes = input::Modes::from_screen(&self.screen);
            for _ in 0..(notches as usize * WHEEL_LINES) {
                self.key_buf.clear();
                input::encode(key, input::Mods::NONE, modes, &mut self.key_buf);
                if let Some(pty) = &self.pty {
                    pty.write_all(&self.key_buf)?;
                }
            }
        }
        Ok(())
    }

    fn on_global(&mut self, r: &mut Reader) -> Result<()> {
        let name = r.u32()?;
        let interface = r.string()?;
        let version = r.u32()?;
        match interface {
            protocol::IFACE_COMPOSITOR => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_COMPOSITOR);
                self.compositor = Some(id);
            }
            protocol::IFACE_WM_BASE => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_WM_BASE);
                self.wm_base = Some(id);
            }
            protocol::IFACE_SEAT => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_SEAT);
                self.seat = Some(id);
            }
            protocol::IFACE_DATA_DEVICE_MANAGER => {
                let id = self.bind_capped(
                    name,
                    interface,
                    version,
                    protocol::VERSION_DATA_DEVICE_MANAGER,
                );
                self.data_device_manager = Some(id);
            }
            protocol::IFACE_DMABUF if version >= protocol::VERSION_DMABUF => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_DMABUF);
                self.presentation.dmabuf = Some(id);
                self.presentation.dmabuf_version = version.min(protocol::VERSION_DMABUF);
            }
            protocol::IFACE_DRM_SYNCOBJ => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_DRM_SYNCOBJ);
                self.presentation.syncobj_manager = Some(id);
            }
            _ => {}
        }
        Ok(())
    }

    /// Send the toplevel title only if it changed. The child sets it via OSC 0/2
    /// (tracked on the grid); an empty title falls back to the app name. The
    /// change-guard dedupes the request on the common unchanged path.
    fn refresh_title(&mut self) {
        let desired = match self.screen.title() {
            "" => "bnkterm".to_string(),
            t => t.to_string(),
        };
        if desired != self.title {
            self.conn.request(
                self.toplevel,
                xdg_toplevel::SET_TITLE,
                &[Arg::Str(&desired)],
            );
            self.title = desired;
        }
    }

    /// Bind the just-advertised global at the version we negotiate: the
    /// compositor's advertised `version` capped at `max`.
    fn bind_capped(&mut self, name: u32, interface: &str, version: u32, max: u32) -> u32 {
        let id = self.alloc_id();
        self.conn.request(
            self.registry,
            wl_registry::BIND,
            &[
                Arg::Uint(name),
                Arg::Bind {
                    interface,
                    version: version.min(max),
                    new_id: id,
                },
            ],
        );
        id
    }
}

/// The font size to open at: `BNKTERM_FONT_SIZE` if it parses, clamped to a sane
/// range, else the default. The one config knob for now; theme and font family
/// follow when config grows into a file.
fn config_font_size() -> u32 {
    std::env::var("BNKTERM_FONT_SIZE")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .map(|s| s.clamp(*FONT_SIZE_RANGE.start(), *FONT_SIZE_RANGE.end()))
        .unwrap_or(FONT_SIZE)
}

/// Map the grid's cursor style (from DECSCUSR) to how the renderer paints it.
fn cursor_shape(style: CursorStyle) -> CursorShape {
    match style {
        CursorStyle::Block => CursorShape::Block,
        CursorStyle::Underline => CursorShape::Underline,
        CursorStyle::Bar => CursorShape::Bar,
    }
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

/// Fill a fresh grid with a static demo that exercises the phase-2 acceptance
/// list: the 16 ANSI colors, the text styles, DEC box drawing, a CJK wide char,
/// an emoji cluster, a combining mark, and truecolor. Driven through the real
/// `Parser` -> `Screen`, so what the window shows is exactly what the VT pipeline
/// produces, not a bespoke fixture.
fn demo_screen(cols: usize, rows: usize) -> Screen {
    let mut s = Screen::new(cols.max(1), rows.max(1));
    let mut p = Parser::new();
    let mut out: Vec<u8> = Vec::new();

    // Move the cursor to 1-based (row, col) and reset the pen.
    let at = |out: &mut Vec<u8>, row: usize, col: usize| {
        out.extend_from_slice(format!("\x1b[{row};{col}H\x1b[0m").as_bytes());
    };

    at(&mut out, 1, 1);
    out.extend_from_slice(
        b"\x1b[1;36mbnkterm\x1b[0m \x1b[2m-- Wayland + Vulkan terminal, phase 2 static demo\x1b[0m",
    );

    at(&mut out, 3, 1);
    out.extend_from_slice(b"ANSI: ");
    for c in 0..8 {
        out.extend_from_slice(format!("\x1b[4{c}m  ").as_bytes());
    }
    out.extend_from_slice(b"\x1b[0m ");
    for c in 0..8 {
        out.extend_from_slice(format!("\x1b[10{c}m  ").as_bytes());
    }

    at(&mut out, 5, 1);
    out.extend_from_slice(
        b"styles: normal \x1b[1mbold\x1b[0m \x1b[3mitalic\x1b[0m \x1b[4munderline\x1b[0m \
          \x1b[9mstrike\x1b[0m \x1b[7mreverse\x1b[0m \x1b[2mdim\x1b[0m",
    );

    // A box via DEC Special Graphics (ESC ( 0 designates G0), then back to ASCII.
    let box_w = 22.min(cols.saturating_sub(2)).max(2);
    let mid = "q".repeat(box_w.saturating_sub(2));
    at(&mut out, 7, 1);
    out.extend_from_slice(format!("\x1b(0l{mid}k\x1b(B").as_bytes());
    at(&mut out, 8, 1);
    out.extend_from_slice(b"\x1b(0x\x1b(B");
    out.extend_from_slice(b" box drawing (DEC) ");
    at(&mut out, 8, box_w);
    out.extend_from_slice(b"\x1b(0x\x1b(B");
    at(&mut out, 9, 1);
    out.extend_from_slice(format!("\x1b(0m{mid}j\x1b(B").as_bytes());

    at(&mut out, 11, 1);
    out.extend_from_slice(
        "unicode: CJK \u{6f22}\u{5b57}  emoji \u{1F600}\u{1F389}  accent cafe\u{0301}".as_bytes(),
    );

    at(&mut out, 13, 1);
    out.extend_from_slice(b"256-color: ");
    for i in (16..=231).step_by(18) {
        out.extend_from_slice(format!("\x1b[48;5;{i}m  ").as_bytes());
    }
    out.extend_from_slice(b"\x1b[0m");

    at(&mut out, 15, 1);
    out.extend_from_slice(b"truecolor: ");
    for step in 0..24 {
        let r = 255 - step * 10;
        let g = step * 10;
        let b = 128;
        out.extend_from_slice(format!("\x1b[48;2;{r};{g};{b}m ").as_bytes());
    }
    out.extend_from_slice(b"\x1b[0m");

    // Park the cursor somewhere visible for the block-cursor demo.
    at(&mut out, 17, 1);
    out.extend_from_slice(b"prompt$ ");

    p.advance_bytes(&mut s, &out);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_screen_fills_a_grid_without_panicking() {
        // The demo drives the real parser into the grid; it must stay in bounds
        // for a range of sizes (including tiny ones the box math must clamp for).
        // On a small grid the content wraps and scrolls, so only its dimensions
        // are asserted; a full grid keeps its title (checked below).
        for &(cols, rows) in &[(80, 24), (40, 12), (10, 4), (2, 2), (1, 1)] {
            let s = demo_screen(cols, rows);
            assert_eq!(s.dimensions(), (cols, rows));
        }
        // At a comfortable size the title sits untouched on the top-left.
        assert_eq!(demo_screen(80, 24).cell(0, 0).rune, 'b');
    }

    #[test]
    fn demo_screen_renders_to_a_nonempty_display_list() {
        // The bring-up seam: a demo grid produces real draw commands (a base
        // fill plus glyph runs), so the window would show content.
        let s = demo_screen(80, 24);
        let metrics = CellMetrics {
            size: FONT_SIZE,
            w: 8,
            h: 16,
            ascent: 12,
            descent: 4,
        };
        let list = term_render::build_display_list(
            &s,
            &Theme::default(),
            metrics,
            (80 * 8, 24 * 16),
            CursorRender::default(),
            None,
        );
        assert!(list.len() > 1, "more than just the background fill");
    }
}
