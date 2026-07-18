//! The terminal's orchestration: it binds the compositor globals, builds the
//! xdg-shell surface stack, brings up the Vulkan/dmabuf presentation path,
//! spawns the shell on a PTY, and runs the event loop that ties them together.
//!
//! ```text
//!   poll [Wayland, ready(0)..ready(n)]
//!            │
//!            ├─ wl_keyboard ─▶ xkb ─▶ Tabs ─▶ active TerminalCore ─▶ PTY write
//!            └─ gather wakes ─────────▶ Tabs ─┬▶ core 0 parser/grid
//!                                              ├▶ core 1 parser/grid
//!                                              └▶ core n parser/grid
//!   active grid + cached tab bar ─▶ display list ─▶ damage ─▶ GPU
//! ```
//!
//! One [`State`] holds every protocol id, the render resources, and [`Tabs`],
//! which owns the per-tab PTY/core fan-out.
//! [`State::run_until`] is the drain-render-wait loop: it services Wayland events
//! and PTY output, paints when the grid changed (paced to the compositor's frame
//! callback), then blocks in a single `poll` over the Wayland socket and every
//! tab's gather wake fd until any has more to say. Keyboard input is encoded by the
//! tested [`crate::input`] table and written back to the child; a resize divides
//! the new pixel size into cells, resizes the grid, and sends the child
//! `TIOCSWINSZ`. The GPU-facing half lives in the [`present`] submodule.
//!
//! `run_demo` keeps the phase-2 static grid (no PTY, no shell) for isolating a
//! render question from the live pipeline.

mod clipboard;
mod message;
mod present;
mod tabs;
mod terminal;

use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use self::clipboard::SelectionState;
use self::message::{PointerEvent, Side, ToTerminal, ToWindow};
use self::present::GpuPresentation;
use self::tabs::{Reorder, Tabs};
use self::terminal::TerminalCore;
use crate::config::{TabBarConfig, TabBarPosition};
// The app orchestrates the platform/render layers (which carry their own error
// type) and the terminal core (which uses the crate-level one). It speaks the
// crate-level `Error`/`Result` throughout; a `?` on a platform call converts
// through the `From` bridge in `crate::error`, so there is one error type here.
use crate::error::{Error, Result};
use crate::input;
use crate::keymode::{self, Disposition, KeyMode, TabAction};
use crate::mouse::MouseButton;
use crate::platform::conn::{Connection, Fill};
use crate::platform::ffi;
use crate::platform::freetype::Fonts;
use crate::platform::geom::{logical_to_device, Scale};
use crate::platform::protocol::{
    self, wl_buffer, wl_callback, wl_compositor, wl_data_device_manager, wl_data_offer, wl_display,
    wl_keyboard, wl_output, wl_pointer, wl_registry, wl_seat, wl_surface,
    wp_cursor_shape_device_v1, wp_cursor_shape_manager_v1, wp_fractional_scale_manager_v1,
    wp_fractional_scale_v1, wp_viewporter, xdg_surface, xdg_toplevel, xdg_wm_base,
    zwp_primary_selection_device_manager_v1, zwp_primary_selection_offer_v1,
};
use crate::platform::wire::{Arg, Message, Reader};
use crate::platform::xkb::Xkb;
use crate::pty;
use crate::render::gpu::TextGamma;
use crate::term_render::CellMetrics;

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

/// The default glyph coverage gammas: how hard to correct the two *opposite*
/// artifacts linear-light compositing inflicts on anti-aliased text. They are
/// separate dials because they correct opposite errors, and nothing says the two
/// need the same magnitude; folding them into one number is a trap, because it
/// makes "lighter correction" mean *bolder* in one direction and *thinner* in the
/// other. Both reach `1.0` for "no correction at all".
///
/// - `light_on_dark` thins **light-on-dark** text (the usual terminal case), which
///   linear-light compositing renders heavier than the gamma-space stacks most
///   terminals use. `> 1` thins; `BNKTERM_TEXT_GAMMA` tunes it.
/// - `dark_on_light` thickens **dark-on-light** text (a reverse-video highlight, a
///   light theme, the active tab's dark label on its light block), which the same
///   compositing washes out instead. Its strength scales with the run's contrast, so
///   it is a `> 1` base raised to a negative power;
///   `BNKTERM_TEXT_GAMMA_DARK_ON_LIGHT` tunes it.
///
/// These are the shipping values, before any environment override; the perf gate
/// uses them directly so its numbers never depend on the environment.
pub(crate) const TEXT_GAMMA: TextGamma = TextGamma {
    light_on_dark: 1.5,
    dark_on_light: 2.0,
};

/// The grid the window opens at, before the compositor sends a size. Classic
/// 80x24; the surface then resizes to whatever the compositor grants.
const DEFAULT_COLS: usize = 80;
const DEFAULT_ROWS: usize = 24;

/// Blank margin, in pixels, between the window edge and the grid on every side
/// (a common terminal default of `padding = 5`). The grid is
/// inset by this and drawn from `(WINDOW_PADDING, WINDOW_PADDING)`; the surface
/// background fills behind it, so the inset reads as a border of background.
const WINDOW_PADDING: i32 = 5;

/// Right mouse button (`BTN_RIGHT`) and middle (`BTN_MIDDLE`) from
/// `linux/input-event-codes.h`; `BTN_LEFT` is in `protocol`.
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;

/// The window, in milliseconds, within which successive left presses on the same
/// cell count as a double/triple click (a common desktop default).
const MULTI_CLICK_MS: u32 = 400;

/// Upper bound on either surface dimension, applied to a compositor-supplied
/// configure size. Far beyond any real display, it just keeps a bogus or hostile
/// size from blowing up the buffer-size arithmetic.
const MAX_DIMENSION: u32 = 16384;

/// First object id in Wayland's server-allocated range. Client-created ids stay
/// below this; the free list only ever recycles client ids.
const SERVER_ID_BASE: u32 = 0xff00_0000;

/// How much bnkterm says about itself on stderr. Quiet by default: a terminal
/// that narrates its own bring-up prints into the very scrollback the user
/// opened it to read, and the shell's first line should be the shell's. Only
/// real errors survive `Quiet`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verbosity {
    /// Errors only. The default, and what a user ever sees unless they ask.
    #[default]
    Quiet,
    /// `--verbose`: the bring-up lines (grid size, renderer, presentation path).
    Verbose,
    /// `--stats`: the bring-up lines plus a timing line on every presented frame.
    Stats,
}

impl Verbosity {
    /// Read the verbosity out of the command line. `BNKTERM_STATS` in the
    /// environment is the same knob as `--stats`, kept because the perf lab and
    /// desktop launchers already set it that way.
    pub fn from_args(args: &[String]) -> Self {
        Self::parse(args, std::env::var_os("BNKTERM_STATS").is_some())
    }

    /// The parse itself, with the environment passed in so it stays pure (and so
    /// the test does not have to mutate a process-wide variable to run).
    fn parse(args: &[String], stats_env: bool) -> Self {
        if stats_env || args.iter().any(|a| a == "--stats") {
            Self::Stats
        } else if args.iter().any(|a| a == "--verbose" || a == "-v") {
            Self::Verbose
        } else {
            Self::Quiet
        }
    }

    /// Whether bring-up narrates itself: the grid it opened on, the renderer it
    /// chose, the sync path it negotiated.
    fn verbose(self) -> bool {
        self >= Self::Verbose
    }

    /// Whether every presented frame prints its build/submit timing and glyph
    /// cache hit rate.
    fn frame_stats(self) -> bool {
        self == Self::Stats
    }
}

/// Open the window, spawn `$SHELL`, and run the terminal until the shell exits or
/// the window is closed.
pub fn run(verbosity: Verbosity) -> crate::error::Result<()> {
    // Export terminal capabilities once, before State construction and before any
    // gather thread can exist. Every shell opened by the process inherits them.
    std::env::set_var("TERM", "xterm-256color");
    std::env::set_var("COLORTERM", "truecolor");
    // Who we are. Nothing consumes this yet — the CLIs that sniff `TERM_PROGRAM` all
    // match it against a hardcoded list of terminals they know, and we are on nobody's
    // list — but it is what a terminal is supposed to say, and it is how anything ever
    // *could* recognise us.
    std::env::set_var("TERM_PROGRAM", "bnkterm");
    std::env::set_var("TERM_PROGRAM_VERSION", env!("CARGO_PKG_VERSION"));
    // And the capability that list is standing in for. The `supports-hyperlinks`
    // family (Node's, and so most JS CLIs) has no capability query for OSC 8: it
    // decides by *name*, from a fixed allowlist of iTerm/WezTerm/vscode/ghostty/VTE,
    // and everyone else is told no forever. `FORCE_HYPERLINK` is the one door out, so
    // we assert what is now simply true — bnkterm renders OSC 8 (see
    // `grid::Screen::set_hyperlink`) — rather than impersonating a terminal on the
    // list to get the same answer.
    std::env::set_var("FORCE_HYPERLINK", "1");
    let mut state = State::new(false, verbosity)?;
    state.bring_up()?;
    Ok(())
}

/// `--demo`: open the window on a static styled grid, without a PTY. The phase-2
/// bring-up, kept for isolating a render question from the live shell.
pub fn run_demo(verbosity: Verbosity) -> crate::error::Result<()> {
    let mut state = State::new(true, verbosity)?;
    state.bring_up()?;
    Ok(())
}

/// `--gpu-probe`: report what the compositor and GPU offer the dmabuf
/// presentation path, without opening a window. Its report *is* its output, so it
/// prints at any verbosity.
pub fn gpu_probe() -> crate::error::Result<()> {
    State::new(true, Verbosity::Quiet)?.probe_dmabuf()?;
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
    /// The tab manager behind the window seam. It owns the terminal core (and in
    /// Phase 1, every core), routes input/timers, supplies the visible display list,
    /// and drains terminal facts for the window to act on.
    tabs: Tabs,
    /// Reused `poll(2)` descriptor storage: Wayland first, followed by gatherer
    /// wake fds. Clearing it retains capacity, so idle waits do not allocate.
    poll_set: pty::PollSet,
    /// The fixed cell box the whole grid is laid out on. Window-authoritative (it
    /// owns the fonts and scale); the core holds a copy, shipped over on a resize.
    metrics: CellMetrics,
    /// The cell box for the (smaller) tab-bar label font, recomputed with `metrics`
    /// on every scale/zoom change and handed to the tabs layer on a resize. Equal to
    /// `metrics` when the label scale is 100%.
    label_metrics: CellMetrics,
    /// Key auto-repeat: Wayland delivers no repeat events, so the client synthesises
    /// them from the compositor's `repeat_info`. `repeat_key`/`repeat_at` track the
    /// held key and when it next fires; `repeat_interval` is `None` when repeat is
    /// disabled. `repeat_delay` is the wait before the first repeat.
    repeat_delay: Duration,
    repeat_interval: Option<Duration>,
    repeat_key: Option<u32>,
    repeat_at: Option<Instant>,
    /// Whether the Wayland keyboard currently focuses this window. Tab switches
    /// use it to focus only the newly active core and disarm every hidden cursor.
    window_focused: bool,
    /// The armed leader key-table, wezterm-style. `Normal` (the default) sends
    /// every key to the child; the others intercept keys for tab control and drive
    /// the centered overlay. See [`crate::keymode`].
    key_mode: KeyMode,

    /// Current surface size in *device* pixels (the buffer resolution the grid is
    /// laid out in). The logical size lives in `scale.logical`.
    width: u32,
    height: u32,
    /// The grid size in cells the window last shipped to the core, mirrored here so
    /// the window can clamp a pointer to the grid and skip a no-op resize without
    /// reaching into the core's screen.
    grid_dims: (usize, usize),
    /// Device-pixel y coordinate of grid row zero. It moves down by the strip
    /// height while a top-anchored tab bar is visible.
    grid_origin_y: i32,
    /// The tab strip's device-pixel top and height (both `0`-height when hidden),
    /// mirrored here so pointer hit testing and the resize math share one source.
    bar_y: i32,
    bar_h: i32,
    /// Tab strip appearance and placement (top/bottom, height, widths, colors).
    tab_bar_config: TabBarConfig,
    /// Compositor scale factor and the objects that report it.
    scale: Scaling,

    // Object ids. Globals are Option (discovered via the registry); ids we
    // create default to 0 (never valid) until assigned.
    registry: u32,
    compositor: Option<u32>,
    wm_base: Option<u32>,
    seat: Option<u32>,
    /// The cursor-shape manager, absent when the compositor lacks
    /// `wp_cursor_shape_manager_v1`; the pointer then keeps the compositor's default
    /// shape (an arrow) rather than the I-beam we would ask for over the grid.
    cursor_shape_manager: Option<u32>,
    keyboard: u32,
    pointer: u32,
    /// The per-pointer shape device the manager hands out, `0` until the pointer
    /// exists (and forever if there is no manager).
    cursor_shape_device: u32,
    /// The cursor shape currently applied (a `wp_cursor_shape_device_v1::SHAPE_*`),
    /// or `0` when none is set yet. Tracked so motion only re-sends `set_shape` when
    /// the pointer crosses between the grid (I-beam) and the tab strip (arrow).
    pointer_shape: u32,
    /// The serial of the latest `wl_pointer.enter`, which `set_shape` must cite to
    /// change the cursor (`0` before the pointer has entered).
    pointer_enter_serial: u32,
    /// Latest pointer position in surface pixels. `axis_accum` gathers fractional
    /// wheel deltas into whole notches (the button held for drag reporting lives
    /// on the core, with the mouse mode it reports under).
    pointer_x: f32,
    pointer_y: f32,
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
    surface: u32,
    xdg_surface: u32,
    toplevel: u32,
    /// The clipboard data device and manager (absent if the compositor has no
    /// `wl_data_device_manager`; copy/paste is then a no-op), plus our state.
    data_device_manager: Option<u32>,
    data_device: u32,
    clipboard: SelectionState,
    /// The primary-selection device and manager (absent if the compositor has no
    /// `zwp_primary_selection_device_manager_v1`; select-to-copy and middle-click
    /// paste are then a no-op), plus our state. Same machinery as the clipboard.
    primary_manager: Option<u32>,
    primary_device: u32,
    primary: SelectionState,
    /// The latest input-event serial, needed to claim a selection.
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
    /// The in-flight frame callback's id, or 0 when none is pending. While it is
    /// nonzero the loop holds off redrawing, so bursts coalesce into at most one
    /// frame per refresh (frame pacing).
    frame_callback: u32,
    /// How chatty stderr is: quiet unless the user asked for the bring-up lines
    /// (`--verbose`) or the per-frame stats line (`--stats`).
    verbosity: Verbosity,
}

impl State {
    fn new(demo: bool, verbosity: Verbosity) -> Result<Self> {
        let conn = Connection::connect()?;
        // Open at unity scale; the compositor's real scale arrives after bring-up
        // and reopens the fonts (see `apply_scale`).
        let font_size = config_font_px(SCALE_120_UNITY);
        let tab_bar_config = TabBarConfig::default();
        let label_size = label_font_px(font_size, tab_bar_config.label_scale_pct);
        let fonts = Fonts::new(&font_sizes(font_size, label_size))?;
        let metrics = CellMetrics::from_fonts(&fonts, font_size);
        let label_metrics = CellMetrics::from_ui(&fonts, label_size);
        let (cols, rows) = (DEFAULT_COLS, DEFAULT_ROWS);
        let width = (cols as i32 * metrics.w + 2 * WINDOW_PADDING).max(1) as u32;
        let height = (rows as i32 * metrics.h + 2 * WINDOW_PADDING).max(1) as u32;
        // The terminal core owns the grid/parser/PTY and its own geometry copies.
        // At unity scale the device padding is just `WINDOW_PADDING`; the first
        // configure ships the real geometry over on a `Resize`.
        let core = TerminalCore::new(demo, cols, rows, metrics, width, height, WINDOW_PADDING);

        Ok(Self {
            conn,
            fonts,
            xkb: Xkb::new()?,
            tabs: Tabs::new(core, tab_bar_config.clone()),
            poll_set: pty::PollSet::new(),
            metrics,
            label_metrics,
            // Sensible defaults until the compositor sends repeat_info.
            repeat_delay: Duration::from_millis(400),
            repeat_interval: Some(Duration::from_millis(33)),
            repeat_key: None,
            repeat_at: None,
            window_focused: false,
            key_mode: KeyMode::Normal,
            width,
            height,
            grid_dims: (cols, rows),
            grid_origin_y: WINDOW_PADDING,
            bar_y: WINDOW_PADDING,
            bar_h: 0,
            tab_bar_config,
            scale: Scaling::new((width, height)),
            registry: 0,
            compositor: None,
            wm_base: None,
            seat: None,
            cursor_shape_manager: None,
            keyboard: 0,
            pointer: 0,
            cursor_shape_device: 0,
            pointer_shape: 0,
            pointer_enter_serial: 0,
            pointer_x: 0.0,
            pointer_y: 0.0,
            axis_accum: 0.0,
            bar_button: None,
            drag_scroll: false,
            last_click_time: 0,
            last_click_cell: (0, 0),
            click_count: 0,
            surface: 0,
            xdg_surface: 0,
            toplevel: 0,
            data_device_manager: None,
            data_device: 0,
            clipboard: SelectionState::new(),
            primary_manager: None,
            primary_device: 0,
            primary: SelectionState::new(),
            last_serial: 0,
            presentation: GpuPresentation::new(),
            next_id: 2, // 1 is wl_display
            free_ids: Vec::new(),
            pending_sync: None,
            pending_size: None,
            pending_configure: None,
            configured: false,
            closed: false,
            frame_callback: 0,
            verbosity,
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
        // The initial title is the app name; the core emits a `Title` later if the
        // child sets one via OSC 0/2.
        self.set_toplevel_title("bnkterm");

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

        // The primary-selection device drives select-to-copy / middle-click paste,
        // the same way; skip it when the compositor has no manager.
        if let (Some(manager), Some(seat)) = (self.primary_manager, self.seat) {
            let device = self.alloc_id();
            self.conn.request(
                manager,
                zwp_primary_selection_device_manager_v1::GET_DEVICE,
                &[Arg::NewId(device), Arg::Object(seat)],
            );
            self.primary_device = device;
        }

        self.create_buffers()?;
        self.init_explicit_sync();

        // Empty commit to get the initial configure; the loop draws once
        // `configured` is set.
        self.conn.request(self.surface, wl_surface::COMMIT, &[]);
        self.run_until(|s| s.configured)?;

        // Now that the window has its granted size, spawn the shell on a PTY
        // sized to the grid. Demo mode skips this and shows its static screen.
        // Under `--verbose` bring-up says what it landed on; quiet by default, so
        // the first thing on stderr is the shell's, not ours.
        if !self.tabs.active().is_demo() {
            let (cols, rows) = self.tabs.active_mut().spawn_shell()?;
            if self.verbosity.verbose() {
                eprintln!("bnkterm: shell on a {cols}x{rows} grid.");
            }
        } else if self.verbosity.verbose() {
            eprintln!("bnkterm: static demo grid, no shell.");
        }
        if self.verbosity.verbose() {
            match &self.presentation.backend {
                Some(gpu) => eprintln!("bnkterm: renderer: gpu ({})", gpu.name()),
                None => eprintln!("bnkterm: renderer: shm"),
            }
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
    /// one is due, then block in one `poll` over Wayland and every tab wake fd until
    /// any is ready or
    /// the soonest timer (cursor blink, key repeat) comes due. An idle, unfocused
    /// terminal with nothing held waits open-ended.
    fn run_until(&mut self, done: impl Fn(&State) -> bool) -> Result<()> {
        // One reusable decode buffer for the whole loop: the main loop is a single
        // app-lifetime `run_until`, so a frame's `wl_callback.done` and every other
        // event decode into this same body buffer without a per-message allocation.
        let mut msg = Message::default();
        loop {
            while self.conn.next_message_into(&mut msg)? {
                self.handle(&msg)?;
            }
            // Drain the child's output into the grid (a no-op with no PTY, or when
            // nothing is ready), then act on what it produced (a title change, a
            // copied selection to own, the child exiting). `more_pty` is true when a
            // fairness-budgeted gather pump left batches queued, so the wait below
            // must not block.
            let more_pty = self.tabs.pump_all(self.window_focused)?;
            // The pump reaps a child that has exited, so the last shell exiting empties
            // the tab list right here. The window is over, and every step below sizes,
            // shapes, or paints around a visible terminal that no longer exists, so the
            // turn ends now — the same invariant `handle` keeps for Wayland messages
            // that were already queued when the last tab went away.
            if self.tabs.is_empty() {
                self.closed = true;
                return Ok(());
            }
            // Opening/closing (including a child exiting during the pump) can
            // make the bar appear or disappear without a compositor configure.
            self.resize_to(self.width, self.height);
            self.drain_outbox()?;
            // A program grabbing or releasing the mouse changes what a drag means, and
            // it arrives as output rather than as a pointer event: without this the
            // shape would be stale under a parked pointer until the user moved it,
            // which is exactly when starting a TUI would leave the I-beam behind. The
            // early-out on an unchanged shape makes the steady-state cost a compare.
            self.update_pointer_shape();
            // Fire any blink toggle or key repeat that has come due.
            self.service_timers()?;
            // Pace to the compositor: only draw when no frame callback is
            // outstanding, so a burst collapses into a single repaint.
            if self.configured
                && self.tabs.needs_frame()
                && self.frame_callback == 0
                && self.render_frame()?
            {
                self.tabs.clear_dirty();
            }
            if done(self) {
                return Ok(());
            }
            self.conn.flush()?;
            // While the gather pump has more batches queued (it stopped on its
            // fairness budget), take the next turn immediately rather than blocking,
            // so a continuous producer drains without stalling Wayland input.
            let wait = if more_pty {
                Some(Duration::ZERO)
            } else {
                self.next_wake()
            };
            self.poll_set.clear();
            let wayland_slot = self.poll_set.add(self.conn.fd());
            for fd in self.tabs.gather_fds() {
                self.poll_set.add(fd);
            }
            self.poll_set.wait(wait)?;
            if self.poll_set.readable(wayland_slot) {
                // poll said the socket has data (or hung up); this recv returns
                // immediately, its short timeout only a safety net.
                if let Fill::Bytes(0) = self.conn.fill(Some(Duration::from_millis(50)))? {
                    return Err(Error::msg("compositor closed the connection"));
                }
            }
            // Due timers are serviced at the top of the next turn.
        }
    }

    /// Fire the cursor blink, the scrollbar's fade, and key repeat if their deadlines
    /// have passed. The blink and the fade are the core's (it builds the frame); key
    /// repeat is the window's (it holds the compositor's `repeat_info` and the held key).
    fn service_timers(&mut self) -> Result<()> {
        self.tabs.tick_blink_if_due();
        self.tabs.tick_bell_if_due();
        self.tabs.tick_scrollbar();
        if self.repeat_at.is_some_and(|at| at <= Instant::now()) {
            self.fire_repeat()?;
        }
        Ok(())
    }

    /// How long to block for input: the soonest of the pending cursor-blink, scrollbar
    /// fade, and key-repeat deadlines, or `None` (block indefinitely) when none is armed.
    /// The blink and fade deadlines are the core's; the key-repeat deadline is the
    /// window's.
    fn next_wake(&self) -> Option<Duration> {
        let now = Instant::now();
        let due = |at: Instant| {
            at.saturating_duration_since(now)
                .max(Duration::from_millis(1))
        };
        // While a frame callback is outstanding the compositor is driving the fade, so
        // the loop does not also need to wake for it (see the callback handler).
        let scrollbar = (self.frame_callback == 0)
            .then(|| self.tabs.scrollbar_retry_at())
            .flatten();
        [
            self.tabs.blink_deadline(),
            self.repeat_at,
            scrollbar,
            self.tabs.sync_deadline(),
            self.tabs.bell_deadline(),
        ]
        .into_iter()
        .flatten()
        .map(due)
        .min()
    }

    /// A logical (surface-local) length in device pixels at the current scale,
    /// rounded to nearest. Device pixels are what the buffer and grid are sized in.
    fn to_device(&self, logical: u32) -> u32 {
        logical_to_device(logical, self.scale.factor_120)
    }

    /// The current display scale, as the painter and the chrome hit-tests take it.
    fn ui_scale(&self) -> Scale {
        Scale::from_120(self.scale.factor_120)
    }

    /// The pointer in device pixels. Wayland delivers it in logical (surface-local)
    /// ones, but the grid and the chrome hanging off it are laid out in device pixels,
    /// so every hit-test against them has to scale up first.
    fn pointer_device(&self) -> (f32, f32) {
        let scale = self.scale.factor_120 as f32 / 120.0;
        (self.pointer_x * scale, self.pointer_y * scale)
    }

    /// The pointer's x in device pixels, the coordinate space the bar lays out in and
    /// the one a tab drag tracks. `Tabs` needs only this scalar from the window.
    fn pointer_device_x(&self) -> i32 {
        self.pointer_device().0 as i32
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
        let cfg = &self.tab_bar_config;
        let (bar_h, gap) = if self.tabs.shows_bar() {
            // The configured logical height, DPI-scaled, floored at one text row so
            // the label can never clip on a small height or a large font, plus the
            // configured breathing room between the strip and the grid.
            (
                (self.to_device(cfg.height_px) as i32).max(self.metrics.h),
                self.to_device(cfg.gap_px) as i32,
            )
        } else {
            (0, 0)
        };
        let usable_w = (w as i32 - 2 * pad).max(0);
        let usable_h = (h as i32 - 2 * pad - bar_h - gap).max(0);
        let (cols, rows) = self.metrics.columns_rows(usable_w, usable_h);
        // The strip steals its height (and the gap) from the side it sits on: a top
        // bar tucks under the padding and pushes the grid down past the gap; a bottom
        // bar sits flush above the bottom padding, the gap reserved above it.
        let (origin_y, bar_y) = match (bar_h, cfg.position) {
            (0, _) => (pad, pad),
            (_, TabBarPosition::Top) => (pad + bar_h + gap, pad),
            (_, TabBarPosition::Bottom) => (pad, h as i32 - pad - bar_h),
        };
        if (w, h) == (self.width, self.height)
            && (cols, rows) == self.grid_dims
            && origin_y == self.grid_origin_y
            && (bar_y, bar_h) == (self.bar_y, self.bar_h)
        {
            return;
        }
        self.width = w;
        self.height = h;
        self.grid_dims = (cols, rows);
        self.grid_origin_y = origin_y;
        self.bar_y = bar_y;
        self.bar_h = bar_h;
        // Ship the fresh grid size and geometry to the core: it resizes the grid and
        // the PTY winsize (best-effort, so this cannot fail from here — see `apply`),
        // and keeps the geometry copies `fill_frame_list` lays out with.
        let _ = self.tabs.resize_all(
            cols,
            rows,
            w,
            h,
            self.metrics,
            self.label_metrics,
            pad,
            origin_y,
            bar_y,
            bar_h,
            self.ui_scale(),
        );
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
        self.tabs.mark_dirty();
    }

    /// Reopen the fonts at device-pixel `size` and recompute the cell metrics. On a
    /// font-open failure the working fonts are kept (no panic, no blank window).
    fn apply_font_size(&mut self, size: u32) {
        if size == self.metrics.size {
            return;
        }
        // On a font-open failure, keep the working fonts (no panic, no blank
        // window); the next scale event may recover.
        let label_size = label_font_px(size, self.tab_bar_config.label_scale_pct);
        if let Ok(fonts) = Fonts::new(&font_sizes(size, label_size)) {
            self.metrics = CellMetrics::from_fonts(&fonts, size);
            self.label_metrics = CellMetrics::from_ui(&fonts, label_size);
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

    fn handle(&mut self, msg: &Message) -> Result<()> {
        // Closing the last tab can happen while more Wayland messages are already
        // queued. The process is leaving; ignore those messages so none can route
        // input through an intentionally empty tab list.
        if self.closed {
            return Ok(());
        }
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
            // Keep a fade going: a scrollbar mid-transition wants the frame the
            // compositor has just offered. A step whose quantised colours did not move
            // presents nothing and arms no new callback, which is what `retry_at` and the
            // wake deadline are for; this is the fast path while it is visibly changing.
            if self.tabs.scrollbar_animating() {
                self.tabs.mark_dirty();
            }
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
            self.tabs.mark_dirty();
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
                // Pair the pointer with a cursor-shape device so we can ask for the
                // I-beam over the grid. The manager is bound in the same global burst
                // as the seat, so it is known by the time capabilities arrive; without
                // it the pointer keeps the compositor default shape.
                if let Some(manager) = self.cursor_shape_manager {
                    let device = self.alloc_id();
                    self.conn.request(
                        manager,
                        wp_cursor_shape_manager_v1::GET_POINTER,
                        &[Arg::NewId(device), Arg::Object(pointer)],
                    );
                    self.cursor_shape_device = device;
                }
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

        if self.primary_device != 0 && msg.object == self.primary_device {
            return self.on_primary_device(msg.opcode, &mut r);
        }

        if self.primary.source != 0 && msg.object == self.primary.source {
            return self.on_primary_source(msg.opcode, &mut r);
        }

        if self.primary.incoming_offer != 0
            && msg.object == self.primary.incoming_offer
            && msg.opcode == zwp_primary_selection_offer_v1::EV_OFFER
        {
            return self.on_primary_offer_mime(&mut r);
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
                self.window_focused = true;
                self.tabs.active_mut().apply(ToTerminal::Focus(true))?;
            }
            wl_keyboard::EV_LEAVE => {
                let _serial = r.u32()?;
                self.window_focused = false;
                self.tabs.active_mut().apply(ToTerminal::Focus(false))?;
                self.stop_repeat(); // drop any held-key repeat (window-side timer)
                                    // Losing focus resets the modifier state (see `Xkb::clear_modifiers`);
                                    // if Ctrl was down, the hand cursor it earned goes with it.
                self.xkb.clear_modifiers();
                self.update_pointer_shape();
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
                // or letting it go changes the cursor with the pointer standing still.
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

    /// Cancel any auto-repeat in flight.
    fn stop_repeat(&mut self) {
        self.repeat_key = None;
        self.repeat_at = None;
    }

    /// Encode one key press and write it to the child. A named key (an arrow, a
    /// function key) is mapped by its keycode; anything else takes its layout
    /// character, and the [`crate::input`] encoder applies Ctrl/Alt. Modifiers
    /// come from xkb's current state.
    /// A key came back up.
    ///
    /// Almost nothing wants to know: the encoder drops a release unless a program has
    /// asked for release events (kitty's `REPORT_EVENT_TYPES`), so for every other child
    /// this resolves the key and then writes nothing. The modal leader tables never see
    /// it — a mode is armed by pressing a key, not by letting go of one.
    fn on_key_release(&mut self, keycode: u32) -> Result<()> {
        let mods = self.current_mods();
        if let Some(key) = self.resolve_key(keycode) {
            self.tabs.active_mut().apply(ToTerminal::Key {
                key,
                mods,
                event: input::KeyEvent::Release,
            })?;
        }
        Ok(())
    }

    fn on_key_press(&mut self, keycode: u32) -> Result<()> {
        let mods = self.current_mods();

        // The modal leader tables (wezterm-style) get first look at every resolved
        // key. In Normal mode only Ctrl+A is a mode key, so everything else reports
        // `Passthrough` and drops to the chords and the child below unchanged; once
        // a mode is armed the machine owns the keystroke until it returns to Normal.
        if let Some(key) = self.resolve_key(keycode) {
            let (next, disposition) = keymode::advance(self.key_mode, key, mods);
            self.set_key_mode(next);
            match disposition {
                Disposition::Consumed(action) => {
                    self.stop_repeat();
                    if let Some(action) = action {
                        self.apply_tab_action(action)?;
                    }
                    return Ok(());
                }
                Disposition::SendLiteral(literal, literal_mods) => {
                    // The Ctrl+A Ctrl+A escape hatch: type a real Ctrl+A.
                    self.stop_repeat();
                    self.tabs.active_mut().apply(ToTerminal::Key {
                        key: literal,
                        mods: literal_mods,
                        event: input::KeyEvent::Press,
                    })?;
                    return Ok(());
                }
                Disposition::Passthrough => {}
            }
        }

        // Ctrl+Tab / Ctrl+Shift+Tab cycle to the next / previous tab, a common
        // accelerator alongside the leader table. Intercepted here because the
        // child would otherwise receive a literal tab / back-tab.
        if mods.contains(input::Mods::CTRL) && !mods.contains(input::Mods::ALT) {
            if let Some(input::Key::Tab) = input::key_from_keycode(keycode) {
                self.stop_repeat();
                if !self.tabs.active().is_demo() {
                    if mods.contains(input::Mods::SHIFT) {
                        self.tabs.prev(self.window_focused);
                    } else {
                        self.tabs.next(self.window_focused);
                    }
                }
                return Ok(());
            }
        }

        // Ctrl+Shift+T/W open and close tabs; C/V retain the terminal copy/paste
        // convention. Demo mode consumes tab chords without creating a PTY.
        if mods.contains(input::Mods::CTRL) && mods.contains(input::Mods::SHIFT) {
            if let Some(c) = self.xkb.key_char(keycode) {
                match c.to_ascii_lowercase() {
                    't' => {
                        self.stop_repeat();
                        self.open_tab();
                        return Ok(());
                    }
                    'w' => {
                        self.stop_repeat();
                        self.close_active_tab();
                        return Ok(());
                    }
                    'c' => {
                        // The core extracts the selection text and queues an
                        // `OfferSelection`; the loop's outbox drain owns the clipboard.
                        self.tabs.active_mut().copy_selection();
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
        // Ctrl+PageUp/PageDown switches tabs. Shift remains reserved for the
        // scrollback chords below, matching the established terminal convention.
        if mods == input::Mods::CTRL {
            match input::key_from_keycode(keycode) {
                Some(input::Key::PageUp) => {
                    self.stop_repeat();
                    if !self.tabs.active().is_demo() {
                        self.tabs.prev(self.window_focused);
                    }
                    return Ok(());
                }
                Some(input::Key::PageDown) => {
                    self.stop_repeat();
                    if !self.tabs.active().is_demo() {
                        self.tabs.next(self.window_focused);
                    }
                    return Ok(());
                }
                _ => {}
            }
        }
        // Shift + Page/Home/End scrolls the scrollback view instead of reaching the
        // child (the alt screen has no history, so there it is a normal key). The
        // window resolves the keycode to a named key; the core does the scrolling.
        if let Some(named) = input::key_from_keycode(keycode) {
            if self.tabs.active_mut().handle_scroll_key(named, mods) {
                return Ok(());
            }
        }
        // Send the key, and if it produced bytes and the keymap marks it
        // repeatable, arm auto-repeat on it.
        if let Some(key) = self.resolve_key(keycode) {
            if self.tabs.active_mut().apply(ToTerminal::Key {
                key,
                mods,
                event: input::KeyEvent::Press,
            })? {
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

    /// Run a leader-resolved tab action against the manager. Tab creation is inert
    /// in the no-PTY demo (as with the Ctrl+Shift chords); switching and reordering
    /// are naturally no-ops there, since the demo has a single tab.
    fn apply_tab_action(&mut self, action: TabAction) -> Result<()> {
        match action {
            TabAction::New => self.open_tab(),
            TabAction::Close => self.close_active_tab(),
            TabAction::Prev => {
                self.tabs.prev(self.window_focused);
            }
            TabAction::Next => {
                self.tabs.next(self.window_focused);
            }
            TabAction::MovePrev => {
                self.tabs.move_active(Reorder::Prev);
            }
            TabAction::MoveNext => {
                self.tabs.move_active(Reorder::Next);
            }
        }
        Ok(())
    }

    /// Open a fresh shell tab at the current geometry, reporting a spawn failure to
    /// stderr without disturbing the existing tabs. Inert in the no-PTY demo.
    fn open_tab(&mut self) {
        if self.tabs.active().is_demo() {
            return;
        }
        let (cols, rows) = self.grid_dims;
        let pad = self.device_pad();
        if let Err(error) = self.tabs.open(
            cols,
            rows,
            self.metrics,
            self.width,
            self.height,
            pad,
            self.window_focused,
        ) {
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
    /// the keymap (a named key by its keycode, else its layout character), or `None`
    /// for a bare modifier / unresolved key. xkb belongs with the Wayland keyboard,
    /// so this stays window-side; the terminal half is [`apply`](Self::apply).
    fn resolve_key(&self, keycode: u32) -> Option<input::Key> {
        if let Some(named) = input::key_from_keycode(keycode) {
            return Some(named);
        }
        let typed = self.xkb.key_char(keycode)?;
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
    fn fire_repeat(&mut self) -> Result<()> {
        let (Some(keycode), Some(interval)) = (self.repeat_key, self.repeat_interval) else {
            self.repeat_at = None;
            return Ok(());
        };
        let mods = self.current_mods();
        let sent = match self.resolve_key(keycode) {
            // The keyboard is repeating a held key. A program that asked to hear about
            // repeats is told it is one; to everyone else it is another press, which is
            // exactly what a repeat has always looked like to a terminal.
            Some(key) => self.tabs.active_mut().apply(ToTerminal::Key {
                key,
                mods,
                event: input::KeyEvent::Repeat,
            })?,
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
        let (cols, rows) = self.grid_dims;
        // Pointer coordinates are logical (surface-local); the grid is device
        // pixels, so scale up first. The grid is then inset by the (device) padding;
        // a pointer in the margin maps to the nearest edge cell (floored at zero).
        let scale = self.scale.factor_120 as f32 / 120.0;
        let pad = self.device_pad() as f32;
        let px = (self.pointer_x * scale - pad).max(0.0);
        let py = (self.pointer_y * scale - self.grid_origin_y as f32).max(0.0);
        let (col, side) = column_under(px, self.metrics.w, cols);
        let row = ((py / self.metrics.h as f32) as usize).min(rows.saturating_sub(1));
        (col, row, side)
    }

    /// Whether the latest pointer position is inside the device-pixel tab strip.
    /// The strip may sit at the top or the bottom, so this reads the resolved
    /// rectangle rather than assuming a one-cell bar under the padding.
    fn pointer_in_tab_bar(&self) -> bool {
        if !self.tabs.shows_bar() {
            return false;
        }
        let scale = self.scale.factor_120 as f32 / 120.0;
        let y = self.pointer_y * scale;
        y >= self.bar_y as f32 && y < (self.bar_y + self.bar_h) as f32
    }

    /// Stable tab identity under the pointer while it is in the bar.
    fn pointer_bar_tab(&self) -> Option<self::tabs::TabId> {
        if !self.pointer_in_tab_bar() {
            return None;
        }
        let scale = self.scale.factor_120 as f32 / 120.0;
        let x = self.pointer_x * scale;
        let pad = self.device_pad() as f32;
        if x < pad {
            return None;
        }
        let col = ((x - pad) / self.metrics.w as f32) as usize;
        self.tabs.tab_at_bar_col(col)
    }

    /// Gather what the pointer is over and ask the compositor for the shape
    /// [`pointer_shape_for`] picks from it. Only re-sent when the shape changes, so a
    /// stream of motion events never spams `set_shape`. A no-op when the compositor
    /// lacks `wp_cursor_shape_manager_v1` (no device, so its default arrow stands) or
    /// before the pointer has entered (no serial to cite).
    ///
    /// Every shape here is a promise about what the next gesture does, which is why
    /// both of the modifier-gated facts are read live rather than cached: the hand
    /// needs Ctrl actually held (the underline says "this is a link", the hand says
    /// "and a click now follows it"), and the I-beam needs Shift to have taken the grid
    /// back from a program reporting the mouse. Both can flip with the pointer parked,
    /// so the modifiers event re-runs this, and so does the main loop — a program
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
    fn update_pointer_shape(&mut self) {
        let (px, py) = self.pointer_device();
        let shape = pointer_shape_for(PointerFacts {
            tab_dragging: self.tabs.dragging(),
            // A scrollbar is a control, not text, so over its grab band (and for as
            // long as its thumb is held) the I-beam gives way to the arrow.
            on_chrome: self.pointer_in_tab_bar()
                || self.drag_scroll
                || self.tabs.active().on_scrollbar(px, py),
            ctrl_link: self.xkb.ctrl_active() && self.tabs.hovering_link(),
            reporting: self.tabs.mouse_reporting(),
            shift: self.xkb.shift_active(),
        });
        if shape == self.pointer_shape {
            return;
        }
        if self.cursor_shape_device != 0 && self.pointer_enter_serial != 0 {
            self.conn.request(
                self.cursor_shape_device,
                wp_cursor_shape_device_v1::SET_SHAPE,
                &[Arg::Uint(self.pointer_enter_serial), Arg::Uint(shape)],
            );
            self.pointer_shape = shape;
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
        self.tabs.active_mut().apply(ToTerminal::Pointer {
            event: PointerEvent::Left,
            mods,
        })?;
        Ok(())
    }

    /// The click multiplicity of a left press: 1, 2, or 3 for successive presses on
    /// the same cell within [`MULTI_CLICK_MS`], cycling back to 1 past a triple so a
    /// fourth click starts fresh. A press elsewhere or after the window restarts the
    /// count. `wrapping_sub` keeps the comparison correct across the wl clock's u32
    /// wrap.
    fn click_count(&mut self, time: u32, cell: (usize, usize)) -> usize {
        let quick = time.wrapping_sub(self.last_click_time) <= MULTI_CLICK_MS;
        self.click_count = if quick && self.last_click_cell == cell && self.click_count < 3 {
            self.click_count + 1
        } else {
            1
        };
        self.last_click_time = time;
        self.last_click_cell = cell;
        self.click_count
    }

    /// One wl_pointer event: track the position, and either report to the child or
    /// drive local selection/scroll.
    fn on_pointer(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_pointer::EV_ENTER => {
                // The enter serial is the one `set_shape` must cite; a fresh enter is
                // also where the compositor resets the cursor, so forget the last
                // shape and re-apply whichever fits the entry position.
                self.pointer_enter_serial = r.u32()?;
                let _surface = r.u32()?;
                self.pointer_x = r.fixed()?;
                self.pointer_y = r.fixed()?;
                self.pointer_shape = 0;
                self.update_pointer_shape();
            }
            wl_pointer::EV_LEAVE => {
                let _serial = r.u32()?;
                let _surface = r.u32()?;
                // The pointer is gone, so nothing is under it: drop any hovered link.
                self.pointer_left_grid()?;
            }
            wl_pointer::EV_MOTION => {
                let _time = r.u32()?;
                self.pointer_x = r.fixed()?;
                self.pointer_y = r.fixed()?;
                let (px, py) = self.pointer_device();
                if self.drag_scroll {
                    // A thumb the pointer is holding follows it anywhere, including out
                    // of the lane and off the grid. Nothing else sees the move.
                    self.tabs.active_mut().drag_scrollbar(py);
                } else if self.tabs.dragging() {
                    // A tab drag owns the motion: it reorders along x and ignores y, so
                    // dragging below the strip keeps reordering rather than dropping the
                    // tab (what every tabbed app does). The grid must not see this, or it
                    // would start a text selection or emit a mouse report mid-drag.
                    self.tabs.drag_to(self.pointer_device_x());
                } else if self.pointer_in_tab_bar() {
                    self.pointer_left_grid()?;
                } else {
                    // The bar only lights and grows here; the grid still gets the move,
                    // because the bar is an overlay and a live text drag runs under it.
                    self.tabs.active_mut().track_scrollbar(Some((px, py)));
                    let (col, row, side) = self.pointer_cell();
                    let mods = self.current_mods();
                    self.tabs.active_mut().apply(ToTerminal::Pointer {
                        event: PointerEvent::Motion { col, row, side },
                        mods,
                    })?;
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
                if !pressed && self.bar_button == mapped {
                    // The release that matches a bar press ends any drag it armed. This
                    // keys off the pressed button, not the pointer position, so a
                    // release dragged off the strip still settles the tab.
                    self.tabs.end_drag();
                    self.bar_button = None;
                    self.update_pointer_shape();
                    return Ok(());
                }
                // The release that ends a thumb drag is the scrollbar's, wherever the
                // pointer has wandered to by now: it must not reach the grid as the end
                // of a selection that never began.
                if !pressed && self.drag_scroll && mapped == Some(MouseButton::Left) {
                    self.drag_scroll = false;
                    self.tabs.active_mut().release_scrollbar();
                    let at = self.pointer_device();
                    self.tabs.active_mut().track_scrollbar(Some(at));
                    self.update_pointer_shape();
                    return Ok(());
                }
                if self.pointer_in_tab_bar() && pressed {
                    self.bar_button = mapped;
                    if let (Some(button), Some(id)) = (mapped, self.pointer_bar_tab()) {
                        match button {
                            MouseButton::Left => {
                                // Press selects immediately (unchanged) and arms a
                                // drag; the tab does not lift until the pointer travels
                                // past the threshold, so a plain click stays a click.
                                self.tabs.select(id, self.window_focused);
                                self.tabs.begin_drag(id, self.pointer_device_x());
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
                        self.drag_scroll = true;
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
                    self.tabs.active_mut().apply(ToTerminal::Pointer {
                        event: PointerEvent::Button {
                            button,
                            pressed,
                            col,
                            row,
                            count,
                            side,
                        },
                        mods,
                    })?;
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
        // view, or sends arrow keys — see `apply_pointer`).
        self.axis_accum += value;
        let step = 15.0; // a typical wheel notch in wl_fixed units
        let notches = (self.axis_accum / step) as i32;
        if notches == 0 {
            return Ok(());
        }
        self.axis_accum -= notches as f32 * step;
        let (col, row, _) = self.pointer_cell();
        let mods = self.current_mods();
        self.tabs.active_mut().apply(ToTerminal::Pointer {
            event: PointerEvent::Wheel {
                down: notches > 0,
                notches: notches.unsigned_abs(),
                col,
                row,
            },
            mods,
        })?;
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
            protocol::IFACE_CURSOR_SHAPE_MANAGER => {
                let id = self.bind_capped(
                    name,
                    interface,
                    version,
                    protocol::VERSION_CURSOR_SHAPE_MANAGER,
                );
                self.cursor_shape_manager = Some(id);
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
            protocol::IFACE_PRIMARY_SELECTION => {
                let id = self.bind_capped(
                    name,
                    interface,
                    version,
                    protocol::VERSION_PRIMARY_SELECTION,
                );
                self.primary_manager = Some(id);
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

    /// Set the toplevel title. The core deduplicates on the sending side (it only
    /// emits a `Title` when the child's title actually changes), so this just makes
    /// the request; bring-up calls it once for the initial app name.
    fn set_toplevel_title(&mut self, title: &str) {
        self.conn
            .request(self.toplevel, xdg_toplevel::SET_TITLE, &[Arg::Str(title)]);
    }

    /// Act on the tabs layer's routed outbound messages after a pump (or a copy):
    /// each is a Wayland request the terminal cannot make itself. Window-side by
    /// necessity; Stage 2 turns this outbox into the terminal→window channel.
    fn drain_outbox(&mut self) -> Result<()> {
        // Swap the queue out so the match arms can take `&mut self` freely, drain it
        // (which keeps the local buffer's capacity), then hand the emptied buffer back
        // for the next pump rather than leaving a zero-cap vec the next message regrows.
        let mut outbox = self.tabs.take_outbox();
        for msg in outbox.drain(..) {
            match msg {
                ToWindow::Title(title) => self.set_toplevel_title(&title),
                ToWindow::OfferSelection(bytes) => self.set_clipboard(bytes),
                ToWindow::OfferPrimary(bytes) => self.set_primary(bytes),
                ToWindow::PastePrimary => self.paste_primary()?,
                ToWindow::OpenUrl(url) => open_url(&url),
                ToWindow::Closed => self.closed = true,
            }
        }
        self.tabs.reclaim_outbox(outbox);
        Ok(())
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
    /// Ctrl is held over a hyperlink, so a click would follow it.
    ctrl_link: bool,
    /// A program has grabbed the mouse with `?1000`/`?1002`/`?1003`.
    reporting: bool,
    /// Shift is held, which forces the gesture back to local selection.
    shift: bool,
}

/// Pick the pointer shape the facts call for, most specific claim first: a held tab
/// owns the pointer outright, then chrome, then a link Ctrl would follow, then a
/// program reporting the mouse, and failing all of those the grid is plain selectable
/// text.
///
/// The ordering is the point. Chrome outranks `reporting` because the strip and the
/// scrollbar are ours no matter what the child asked for, and `ctrl_link` outranks it
/// because Ctrl+click follows a link whether or not a program is watching the mouse.
fn pointer_shape_for(f: PointerFacts) -> u32 {
    if f.tab_dragging {
        // The grabbing hand is the whole visual affordance of a tab drag, and it wins
        // over the strip's own arrow.
        wp_cursor_shape_device_v1::SHAPE_GRABBING
    } else if f.on_chrome {
        wp_cursor_shape_device_v1::SHAPE_DEFAULT
    } else if f.ctrl_link {
        wp_cursor_shape_device_v1::SHAPE_POINTER
    } else if f.reporting && !f.shift {
        // The grid belongs to a program grabbing the mouse, so a drag is its event to
        // interpret, not a selection. Shift is the same override the terminal half
        // applies to the event itself, so the two never disagree about who the next
        // drag belongs to.
        wp_cursor_shape_device_v1::SHAPE_DEFAULT
    } else {
        wp_cursor_shape_device_v1::SHAPE_TEXT
    }
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

/// The tab-label font size in device pixels: the body `size` scaled by the config
/// percentage, floored so a small font or a low percentage cannot shrink the label
/// to nothing, and never larger than the body font.
fn label_font_px(size: u32, scale_pct: u16) -> u32 {
    const LABEL_FLOOR_PX: u32 = 8;
    (size * scale_pct as u32 / 100).clamp(LABEL_FLOOR_PX.min(size), size)
}

/// The distinct font sizes to open for a body/label pair: one entry when the label
/// lands on the body size (100% scale, or a floor that meets it), two otherwise, so
/// `Fonts` never rasterizes the same size twice.
fn font_sizes(body: u32, label: u32) -> Vec<u32> {
    if label == body {
        vec![body]
    } else {
        vec![body, label]
    }
}

/// The glyph coverage gammas (see [`TEXT_GAMMA`]), each overridable by its own
/// environment variable and clamped to a sane range so a bad value cannot make text
/// vanish.
fn config_text_gamma() -> TextGamma {
    fn dial(var: &str, default: f32) -> f32 {
        std::env::var(var)
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|g| g.is_finite() && *g > 0.0)
            .unwrap_or(default)
            .clamp(0.5, 4.0)
    }
    TextGamma {
        light_on_dark: dial("BNKTERM_TEXT_GAMMA", TEXT_GAMMA.light_on_dark),
        dark_on_light: dial("BNKTERM_TEXT_GAMMA_DARK_ON_LIGHT", TEXT_GAMMA.dark_on_light),
    }
}

/// Launch a Ctrl+clicked hyperlink in whatever the desktop has set as its handler.
/// The core already vetted the scheme, so a refusal here means the desktop could not
/// start a handler at all: report it and carry on, because a link that will not open
/// is an annoyance and must never be allowed to take the terminal down with it.
fn open_url(url: &str) {
    if let Err(err) = crate::platform::browser::open_url(url) {
        eprintln!("bnkterm: could not open {url}: {err}");
    }
}

/// Map a Wayland pointer button (a `linux/input-event-codes.h` code) to the
/// logical button the mouse encoder speaks. Unknown buttons (side buttons) are
/// ignored.
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
    use super::*;

    /// The pointer resting on plain, unclaimed grid text.
    const OVER_TEXT: PointerFacts = PointerFacts {
        tab_dragging: false,
        on_chrome: false,
        ctrl_link: false,
        reporting: false,
        shift: false,
    };

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
    fn chrome_and_links_outrank_a_reporting_program() {
        // The strip and the scrollbar are ours whatever the child asked for.
        assert_eq!(
            pointer_shape_for(PointerFacts {
                on_chrome: true,
                reporting: true,
                ..OVER_TEXT
            }),
            wp_cursor_shape_device_v1::SHAPE_DEFAULT
        );
        // Ctrl+click follows a link whether or not a program watches the mouse, so the
        // hand must survive reporting rather than collapse to the arrow.
        assert_eq!(
            pointer_shape_for(PointerFacts {
                ctrl_link: true,
                reporting: true,
                ..OVER_TEXT
            }),
            wp_cursor_shape_device_v1::SHAPE_POINTER
        );
        // A held tab beats everything, including the strip it is travelling over.
        assert_eq!(
            pointer_shape_for(PointerFacts {
                tab_dragging: true,
                on_chrome: true,
                ctrl_link: true,
                reporting: true,
                shift: true,
            }),
            wp_cursor_shape_device_v1::SHAPE_GRABBING
        );
    }

    #[test]
    fn the_shape_is_never_the_unset_zero() {
        // `pointer_shape: 0` is the sentinel for "forget what was applied" on enter, so
        // no real choice may collide with it or the re-apply would be skipped.
        for bits in 0..32u8 {
            let facts = PointerFacts {
                tab_dragging: bits & 1 != 0,
                on_chrome: bits & 2 != 0,
                ctrl_link: bits & 4 != 0,
                reporting: bits & 8 != 0,
                shift: bits & 16 != 0,
            };
            assert_ne!(pointer_shape_for(facts), 0, "{facts:?}");
        }
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

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn verbosity_is_quiet_unless_asked_for() {
        // The plain launch, and a launch whose other flags say nothing about noise.
        assert_eq!(Verbosity::parse(&args(&[]), false), Verbosity::Quiet);
        assert_eq!(
            Verbosity::parse(&args(&["--demo"]), false),
            Verbosity::Quiet
        );
        assert!(!Verbosity::Quiet.verbose());
        assert!(!Verbosity::Quiet.frame_stats());
    }

    #[test]
    fn verbose_and_stats_flags_climb_the_ladder() {
        assert_eq!(Verbosity::parse(&args(&["-v"]), false), Verbosity::Verbose);
        assert_eq!(
            Verbosity::parse(&args(&["--verbose"]), false),
            Verbosity::Verbose
        );
        assert_eq!(
            Verbosity::parse(&args(&["--stats"]), false),
            Verbosity::Stats
        );
        // The env var is the same knob as `--stats`, and `--stats` implies verbose:
        // the frame lines are useless without knowing which renderer produced them.
        assert_eq!(Verbosity::parse(&args(&[]), true), Verbosity::Stats);
        assert_eq!(
            Verbosity::parse(&args(&["--verbose"]), true),
            Verbosity::Stats,
            "the louder of the two wins"
        );
        assert!(Verbosity::Verbose.verbose());
        assert!(!Verbosity::Verbose.frame_stats());
        assert!(Verbosity::Stats.verbose());
        assert!(Verbosity::Stats.frame_stats());
    }
}
