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
    wl_keyboard, wl_output, wl_pointer, wl_registry, wl_seat, wl_surface,
    wp_fractional_scale_manager_v1, wp_fractional_scale_v1, wp_viewporter, xdg_surface,
    xdg_toplevel, xdg_wm_base,
};
use crate::platform::wire::{Arg, Message, Reader};
use crate::platform::xkb::Xkb;
use crate::pty::{self, Pty, ReadOutcome};
use crate::render::display::DisplayList;
use crate::term_render::{self, CellMetrics, CursorRender, CursorShape, Selection};
use crate::vt::Parser;

/// The default font size in points. Converted to device pixels at the display's
/// scale factor (see [`points_to_px`]), so the physical size tracks DPI the way
/// mainstream terminal configs do. Overridable with `BNKTERM_FONT_POINTS`, or
/// pinned to explicit pixels with `BNKTERM_FONT_SIZE`.
const FONT_POINTS: f32 = 10.0;

/// The sane range a resolved device-pixel font size is clamped to.
const FONT_SIZE_RANGE: std::ops::RangeInclusive<u32> = 6..=72;

/// A compositor scale of 1.0, in the fractional-scale protocol's 120ths unit. The
/// scale is unknown until the compositor reports it, so the window opens at unity.
const SCALE_120_UNITY: u32 = 120;

/// The default glyph mask gamma: the power the fragment shader raises coverage to.
/// Linear-light compositing renders light-on-dark text heavier than the gamma-space
/// stacks most GPU terminals use, so a value > 1 thins the anti-aliased edges back
/// to a matching weight. 2.0 is the default; `BNKTERM_TEXT_GAMMA`
/// tunes it (1.0 disables the correction, the old heavier look).
const TEXT_GAMMA: f32 = 2.0;

/// The cursor blink half-period: how long each of the on/off phases lasts.
const BLINK_INTERVAL: Duration = Duration::from_millis(530);

/// The grid the window opens at, before the compositor sends a size. Classic
/// 80x24; the surface then resizes to whatever the compositor grants.
const DEFAULT_COLS: usize = 80;
const DEFAULT_ROWS: usize = 24;

/// Blank margin, in pixels, between the window edge and the grid on every side
/// (a common terminal default of `padding = 5`). The grid is
/// inset by this and drawn from `(WINDOW_PADDING, WINDOW_PADDING)`; the surface
/// background fills behind it, so the inset reads as a border of background.
const WINDOW_PADDING: i32 = 5;

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

/// Compositor scale tracking and the objects that deliver it.
///
/// Two mechanisms, in priority order. The *fractional* path is primary: a
/// `wp_fractional_scale_v1` reports `preferred_scale` in 120ths and a `wp_viewport`
/// maps the device-pixel buffer down to the logical window size, so a 1.25 or 1.5
/// display renders pixel-exact. The *integer* path is the fallback for compositors
/// with no fractional-scale global: each `wl_output`'s `scale` event, the surface's
/// scale being the max over the outputs it spans (`wl_surface.enter`/`leave`), and
/// `wl_surface.set_buffer_scale`. `viewport != 0` means the fractional path is live.
struct Scaling {
    /// Effective scale as 120ths (120 = 1.0); drives font pixels and buffer size.
    factor_120: u32,
    /// Logical window size (surface-local px) from the last xdg configure. The
    /// device buffer is this scaled up; the viewport (or buffer scale) maps back.
    logical: (u32, u32),
    /// Whether the surface's geometry + viewport/buffer-scale state has been resent
    /// since `factor_120` or `logical` last changed. Cleared on change, set on sync.
    synced: bool,
    fractional_manager: Option<u32>,
    viewporter: Option<u32>,
    /// Per-surface objects the managers mint (0 when unavailable).
    fractional_scale: u32,
    viewport: u32,
    /// Integer fallback: each bound `wl_output`'s scale, and the ids the surface
    /// currently spans. Unused once the fractional path is live.
    outputs: Vec<(u32, i32)>,
    entered: Vec<u32>,
}

impl Scaling {
    fn new(logical: (u32, u32)) -> Self {
        Scaling {
            factor_120: SCALE_120_UNITY,
            logical,
            synced: false,
            fractional_manager: None,
            viewporter: None,
            fractional_scale: 0,
            viewport: 0,
            outputs: Vec::new(),
            entered: Vec::new(),
        }
    }

    /// Whether the fractional path is live (a viewport was created for the surface).
    fn is_fractional(&self) -> bool {
        self.viewport != 0
    }

    /// The integer scale the surface spans: the max scale over the outputs it is
    /// on, at least 1. Used only on the fallback path.
    fn integer_scale(&self) -> i32 {
        self.entered
            .iter()
            .filter_map(|id| self.outputs.iter().find(|(oid, _)| oid == id))
            .map(|(_, scale)| *scale)
            .max()
            .unwrap_or(1)
            .max(1)
    }
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

    /// Current surface size in *device* pixels (the buffer resolution the grid is
    /// laid out in). The logical size lives in `scale.logical`.
    width: u32,
    height: u32,
    /// Compositor scale factor and the objects that report it.
    scale: Scaling,

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
        // Open at unity scale; the compositor's real scale arrives after bring-up
        // and reopens the fonts (see `apply_scale`).
        let font_size = config_font_px(SCALE_120_UNITY);
        let fonts = Fonts::new(&[font_size])?;
        let metrics = CellMetrics::from_fonts(&fonts, font_size);
        let (cols, rows) = (DEFAULT_COLS, DEFAULT_ROWS);
        let width = (cols as i32 * metrics.w + 2 * WINDOW_PADDING).max(1) as u32;
        let height = (rows as i32 * metrics.h + 2 * WINDOW_PADDING).max(1) as u32;
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
            scale: Scaling::new((width, height)),
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

        // Fractional scaling: a per-surface fractional-scale object delivers the
        // preferred scale, and a viewport maps the device-pixel buffer onto the
        // logical window size. They come as a pair; without both, the integer
        // fallback (wl_output scale + set_buffer_scale) is used instead.
        if let (Some(fmgr), Some(vp)) = (self.scale.fractional_manager, self.scale.viewporter) {
            let fractional = self.alloc_id();
            self.conn.request(
                fmgr,
                wp_fractional_scale_manager_v1::GET_FRACTIONAL_SCALE,
                &[Arg::NewId(fractional), Arg::Object(self.surface)],
            );
            self.scale.fractional_scale = fractional;
            let viewport = self.alloc_id();
            self.conn.request(
                vp,
                wp_viewporter::GET_VIEWPORT,
                &[Arg::NewId(viewport), Arg::Object(self.surface)],
            );
            self.scale.viewport = viewport;
        }

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
        let pad = self.device_pad();
        let list = term_render::build_display_list(
            &self.screen,
            &self.theme,
            self.metrics,
            (self.width as i32, self.height as i32),
            (pad, pad),
            cursor,
            self.selection,
        );
        Rc::new(list)
    }

    /// A logical (surface-local) length in device pixels at the current scale,
    /// rounded to nearest. Device pixels are what the buffer and grid are sized in.
    fn to_device(&self, logical: u32) -> u32 {
        logical_to_device(logical, self.scale.factor_120)
    }

    /// The logical window size scaled to the device buffer size, clamped so a bogus
    /// configure cannot blow up the buffer arithmetic.
    fn device_size(&self, logical_w: u32, logical_h: u32) -> (u32, u32) {
        (
            self.to_device(logical_w).clamp(1, MAX_DIMENSION),
            self.to_device(logical_h).clamp(1, MAX_DIMENSION),
        )
    }

    /// The window padding in device pixels: the logical [`WINDOW_PADDING`] scaled,
    /// so the margin looks the same physical size at any DPI.
    fn device_pad(&self) -> i32 {
        self.to_device(WINDOW_PADDING as u32) as i32
    }

    /// Record a new *device* surface size, resize the grid to the cells that now fit
    /// (inside the scaled padding), and tell the child (via `TIOCSWINSZ`, so it gets
    /// SIGWINCH and repaints). GPU buffers are reallocated lazily in `render_frame`.
    /// A no-op only when neither the device size nor the resulting grid changed (the
    /// grid can change from a scale-driven metrics change at an unchanged size).
    fn resize_to(&mut self, w: u32, h: u32) {
        // Reserve the padding on all sides, so the grid fits inside the margins.
        let pad = self.device_pad();
        let usable_w = (w as i32 - 2 * pad).max(0);
        let usable_h = (h as i32 - 2 * pad).max(0);
        let (cols, rows) = self.metrics.columns_rows(usable_w, usable_h);
        if (w, h) == (self.width, self.height) && (cols, rows) == self.screen.dimensions() {
            return;
        }
        self.width = w;
        self.height = h;
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

    /// Adopt a new compositor scale (in 120ths): reopen the fonts at the size it
    /// calls for, re-derive the device buffer size from the logical window, and
    /// resize the grid. A no-op if the scale is unchanged.
    fn set_scale_120(&mut self, factor_120: u32) {
        let factor_120 = factor_120.max(1);
        if factor_120 == self.scale.factor_120 {
            return;
        }
        self.scale.factor_120 = factor_120;
        self.scale.synced = false; // geometry + viewport/buffer-scale must be resent
        let px = config_font_px(factor_120);
        self.apply_font_size(px);
        let (lw, lh) = self.scale.logical;
        let (dw, dh) = self.device_size(lw, lh);
        self.resize_to(dw, dh);
        // Force the next frame even if the grid dimensions happened to land the
        // same, so the resent scale state (and rescaled glyphs) reach the screen.
        self.dirty = true;
    }

    /// Reopen the fonts at device-pixel `size` and recompute the cell metrics. On a
    /// font-open failure the working fonts are kept (no panic, no blank window).
    fn apply_font_size(&mut self, size: u32) {
        if size == self.metrics.size {
            return;
        }
        // On a font-open failure, keep the working fonts (no panic, no blank
        // window); the next scale event may recover.
        if let Ok(fonts) = Fonts::new(&[size]) {
            self.metrics = CellMetrics::from_fonts(&fonts, size);
            self.fonts = fonts;
        }
    }

    /// On the integer-scale fallback, recompute the surface scale as the max over
    /// the outputs it currently spans and adopt it. A no-op on the fractional path.
    fn refresh_integer_scale(&mut self) {
        if self.scale.is_fractional() {
            return;
        }
        let factor_120 = self.scale.integer_scale() as u32 * 120;
        self.set_scale_120(factor_120);
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
            if let Some((lw, lh)) = self.pending_size.take() {
                // The configure size is logical (surface-local); the buffer and
                // grid are device pixels. Record the logical size (for the viewport
                // destination) and resize the grid to the scaled device size.
                self.scale.logical = (lw, lh);
                self.scale.synced = false;
                let (dw, dh) = self.device_size(lw, lh);
                self.resize_to(dw, dh);
            }
            self.configured = true;
            self.dirty = true;
            return Ok(());
        }

        // wp_fractional_scale_v1.preferred_scale: the compositor's scale as 120ths.
        // This is the primary scale source; adopting it reopens the fonts and
        // resizes the buffer to exact device resolution.
        if self.scale.fractional_scale != 0
            && msg.object == self.scale.fractional_scale
            && msg.opcode == wp_fractional_scale_v1::EV_PREFERRED_SCALE
        {
            let scale_120 = r.u32()?;
            self.set_scale_120(scale_120);
            return Ok(());
        }

        // wl_surface.enter/leave and wl_output.scale drive the integer fallback
        // (used only when the fractional path is absent).
        if msg.object == self.surface
            && (msg.opcode == wl_surface::EV_ENTER || msg.opcode == wl_surface::EV_LEAVE)
        {
            let output = r.u32()?;
            if msg.opcode == wl_surface::EV_ENTER {
                if !self.scale.entered.contains(&output) {
                    self.scale.entered.push(output);
                }
            } else {
                self.scale.entered.retain(|&o| o != output);
            }
            self.refresh_integer_scale();
            return Ok(());
        }
        if msg.opcode == wl_output::EV_SCALE
            && self.scale.outputs.iter().any(|(id, _)| *id == msg.object)
        {
            let factor = r.u32()? as i32;
            for out in &mut self.scale.outputs {
                if out.0 == msg.object {
                    out.1 = factor.max(1);
                }
            }
            self.refresh_integer_scale();
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
        // Pointer coordinates are logical (surface-local); the grid is device
        // pixels, so scale up first. The grid is then inset by the (device) padding;
        // a pointer in the margin maps to the nearest edge cell (floored at zero).
        let scale = self.scale.factor_120 as f32 / 120.0;
        let pad = self.device_pad() as f32;
        let px = (self.pointer_x * scale - pad).max(0.0);
        let py = (self.pointer_y * scale - pad).max(0.0);
        let col = (px / self.metrics.w as f32) as usize;
        let row = (py / self.metrics.h as f32) as usize;
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
            protocol::IFACE_OUTPUT => {
                // One global per monitor; bind each and default its scale to 1
                // until a `scale` event refines it (integer fallback only).
                let id = self.bind_capped(name, interface, version, protocol::VERSION_OUTPUT);
                self.scale.outputs.push((id, 1));
            }
            protocol::IFACE_FRACTIONAL_SCALE_MANAGER => {
                let id = self.bind_capped(
                    name,
                    interface,
                    version,
                    protocol::VERSION_FRACTIONAL_SCALE_MANAGER,
                );
                self.scale.fractional_manager = Some(id);
            }
            protocol::IFACE_VIEWPORTER => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_VIEWPORTER);
                self.scale.viewporter = Some(id);
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

/// Round a logical (surface-local) length to device pixels at scale `factor_120`
/// (120ths; 120 = 1.0). The `+ 60` is round-to-nearest (half of 120). `u64` math
/// so a large window times a large scale cannot overflow before the divide.
fn logical_to_device(logical: u32, factor_120: u32) -> u32 {
    (((logical as u64) * (factor_120 as u64) + 60) / 120) as u32
}

/// Convert a point size to device pixels at compositor scale `scale_120` (120ths;
/// 120 = 1.0), clamped to [`FONT_SIZE_RANGE`]. The `96/72` factor is the reference
/// 96 DPI over 72 points per inch, the same basis the sibling terminals use, so a
/// given point size renders at the same physical height here as there.
fn points_to_px(points: f32, scale_120: u32) -> u32 {
    let px = points * (96.0 / 72.0) * (scale_120 as f32 / 120.0);
    // `max(0)` guards a nonsense (negative/NaN) env value; NaN compares false, so
    // it lands on the range start rather than a panic.
    let px = if px.is_finite() {
        px.round().max(0.0) as u32
    } else {
        0
    };
    px.clamp(*FONT_SIZE_RANGE.start(), *FONT_SIZE_RANGE.end())
}

/// The device-pixel font size at compositor scale `scale_120`. `BNKTERM_FONT_SIZE`
/// pins an explicit pixel height and opts out of scaling (the escape hatch);
/// otherwise `BNKTERM_FONT_POINTS` (or [`FONT_POINTS`]) is scaled by the display.
fn config_font_px(scale_120: u32) -> u32 {
    if let Some(px) = std::env::var("BNKTERM_FONT_SIZE")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
    {
        return px.clamp(*FONT_SIZE_RANGE.start(), *FONT_SIZE_RANGE.end());
    }
    let points = std::env::var("BNKTERM_FONT_POINTS")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|p| p.is_finite() && *p > 0.0)
        .unwrap_or(FONT_POINTS);
    points_to_px(points, scale_120)
}

/// The glyph coverage gamma (see [`TEXT_GAMMA`]), overridable with
/// `BNKTERM_TEXT_GAMMA` and clamped to a sane range so a bad value cannot make
/// text vanish.
fn config_text_gamma() -> f32 {
    std::env::var("BNKTERM_TEXT_GAMMA")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|g| g.is_finite() && *g > 0.0)
        .unwrap_or(TEXT_GAMMA)
        .clamp(0.5, 4.0)
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
            size: 16,
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
            (0, 0),
            CursorRender::default(),
            None,
        );
        assert!(list.len() > 1, "more than just the background fill");
    }

    #[test]
    fn points_to_px_scales_with_the_display() {
        // 10pt at 96/72 is 13.33px; scale multiplies it and rounds to nearest.
        assert_eq!(points_to_px(10.0, 120), 13); // 1.0x  -> 13.33 -> 13
        assert_eq!(points_to_px(10.0, 180), 20); // 1.5x  -> 20.0  -> 20
        assert_eq!(points_to_px(10.0, 240), 27); // 2.0x  -> 26.67 -> 27
        assert_eq!(points_to_px(9.0, 120), 12); // 9pt at 1.0x -> 12
                                                // Same physical size two ways: 10pt at 2x equals 20pt at 1x.
        assert_eq!(points_to_px(10.0, 240), points_to_px(20.0, 120));
    }

    #[test]
    fn points_to_px_clamps_and_survives_bad_input() {
        let (lo, hi) = (*FONT_SIZE_RANGE.start(), *FONT_SIZE_RANGE.end());
        assert_eq!(points_to_px(1000.0, 120), hi); // absurdly large clamps down
        assert_eq!(points_to_px(1.0, 120), lo); // tiny clamps up
        assert_eq!(points_to_px(f32::NAN, 120), lo); // NaN -> range start, no panic
        assert_eq!(points_to_px(-5.0, 120), lo); // negative -> range start
        assert_eq!(points_to_px(0.0, 120), lo);
    }

    #[test]
    fn logical_to_device_rounds_to_nearest() {
        assert_eq!(logical_to_device(100, 120), 100); // 1.0x is identity
        assert_eq!(logical_to_device(100, 240), 200); // 2.0x
        assert_eq!(logical_to_device(100, 180), 150); // 1.5x
        assert_eq!(logical_to_device(100, 150), 125); // 1.25x
        assert_eq!(logical_to_device(101, 150), 126); // 126.25 -> 126 (nearest)
                                                      // No overflow at the extremes (u64 math, then narrowed).
        assert_eq!(logical_to_device(16384, 240), 32768);
    }

    #[test]
    fn integer_scale_is_the_max_over_entered_outputs() {
        let mut s = Scaling::new((800, 600));
        s.outputs = vec![(10, 1), (11, 2), (12, 3)];
        assert_eq!(s.integer_scale(), 1, "no outputs entered -> 1");
        s.entered = vec![10];
        assert_eq!(s.integer_scale(), 1);
        s.entered = vec![10, 11];
        assert_eq!(
            s.integer_scale(),
            2,
            "spanning two outputs takes the larger"
        );
        s.entered = vec![11, 12];
        assert_eq!(s.integer_scale(), 3);
        // An entered id we never bound is ignored, and the floor is 1.
        s.entered = vec![99];
        assert_eq!(s.integer_scale(), 1);
    }

    #[test]
    fn a_fresh_scaling_is_unity_and_not_fractional() {
        let s = Scaling::new((800, 600));
        assert_eq!(s.factor_120, SCALE_120_UNITY);
        assert!(!s.is_fractional(), "no viewport yet");
    }
}
