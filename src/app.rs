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
//! `TIOCSWINSZ`. The GPU-facing half lives in the [`present`] submodule, the keyboard
//! and pointer in [`seat`], and the protocol dispatch in [`wayland`].
//!
//! `run_demo` keeps the phase-2 static grid (no PTY, no shell) for isolating a
//! render question from the live pipeline.

mod activation;
mod clipboard;
mod host_check;
mod layout;
mod message;
mod present;
mod seat;
mod tabs;
mod terminal;
mod wayland;

use std::rc::Rc;
use std::time::{Duration, Instant};

use self::activation::{Activation, ClickedLink};
use self::clipboard::{PendingSend, SelectionTransport, Transport};
use self::host_check::HostChecks;
use self::layout::{TerminalGeometry, WindowLayout, WINDOW_PADDING};
use self::message::ToWindow;
use self::present::GpuPresentation;
use self::seat::PointerState;
use self::tabs::Tabs;
use self::terminal::TerminalCore;
use crate::config::{self, Config, TabBarConfig, FONT_SIZE_RANGE};
// The app orchestrates the platform/render layers (which carry their own error
// type) and the terminal core (which uses the crate-level one). It speaks the
// crate-level `Error`/`Result` throughout; a `?` on a platform call converts
// through the `From` bridge in `crate::error`, so there is one error type here.
use crate::error::{Error, Result};
use crate::keymode::KeyMode;
use crate::platform::conn::{Connection, Fill};
use crate::platform::freetype::{FontConfig, FontSelection, Fonts};
use crate::platform::geom::Scale;
use crate::platform::host::Hostname;
use crate::platform::protocol::{
    self, wl_compositor, wl_display, wl_surface, wp_fractional_scale_manager_v1, wp_viewporter,
    xdg_surface, xdg_toplevel, xdg_wm_base,
};
use crate::platform::wire::{Arg, Message};
use crate::platform::xkb::Xkb;
use crate::pty;
use crate::pty::{EnvVar, Launch, Target};
use crate::render::gpu::TextGamma;
use crate::term_render::CellMetrics;

/// The default font size in points. Converted to device pixels at the display's
/// scale factor (see [`points_to_px`]), so the physical size tracks DPI the way
/// mainstream terminal configs do. Overridable with `BNKTERM_FONT_POINTS`, or
/// pinned to explicit pixels with `BNKTERM_FONT_SIZE`.
const FONT_POINTS: f32 = 10.0;

/// A compositor scale of 1.0, in the fractional-scale protocol's 120ths unit. The
/// scale is unknown until the compositor reports it, so the window opens at unity.
const SCALE_120_UNITY: u32 = 120;

/// The grid the window opens at, before the compositor sends a size. Classic
/// 80x24; the surface then resizes to whatever the compositor grants.
const DEFAULT_COLS: usize = 80;

const DEFAULT_ROWS: usize = 24;

/// Upper bound on either surface dimension, applied to a compositor-supplied
/// configure size. Far beyond any real display, it just keeps a bogus or hostile
/// size from blowing up the buffer-size arithmetic.
const MAX_DIMENSION: u32 = 16384;

/// How long past the latest frame a pending configure waits for the frame callback before
/// it draws without it. A few refreshes at any common rate, and short of the 100 ms labwc
/// waits for an ack before it carries on without one.
const FRAME_CALLBACK_OVERDUE: Duration = Duration::from_millis(50);

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

/// Open the window, spawn the user's shell, and run the terminal until the shell exits or
/// the window is closed.
pub fn run(verbosity: Verbosity) -> crate::error::Result<()> {
    // First, while this is the only thread: every child spawned from here on, a local
    // shell, `flatpak-spawn` or a link's `xdg-open`, starts without the token.
    let launch_token = activation::take_launch_token();
    let host = flatpak_host();
    let shell = match &host {
        Some(host) => host.var("SHELL"),
        None => std::env::var_os("SHELL"),
    }
    .and_then(|shell| shell.into_string().ok())
    .unwrap_or_else(|| "/bin/sh".to_string());
    // Shell integration (OSC 133 prompt marks), so a resize reflow does not fight the
    // shell's prompt redraw. Dropping it removes the generated directory, so it is declared
    // before the state that spawns shells against it and outlives it.
    let shell_integration = match &host {
        Some(host) => crate::flatpak::shared_dir().and_then(|base| {
            crate::shell_integration::install(&shell, &|name| host.var(name), &[base])
        }),
        None => crate::shell_integration::install(
            &shell,
            &|name| std::env::var_os(name),
            &crate::shell_integration::local_bases(),
        ),
    };
    let launch = shell_launch(shell, shell_integration.as_ref(), host.is_some());
    let mut state = State::new(false, verbosity, launch, launch_token)?;
    state.bring_up()?;
    Ok(())
}

/// The host side of the Flatpak sandbox this process runs in, or `None` outside one. A
/// sandbox that cannot reach its host keeps the shell inside, which still works, and says
/// why.
fn flatpak_host() -> Option<crate::flatpak::Host> {
    if !crate::flatpak::sandboxed() {
        return None;
    }
    crate::flatpak::Host::query()
        .map_err(|error| eprintln!("bnkterm: {error}; the shell runs inside the sandbox"))
        .ok()
}

/// How every tab starts `shell`: with the shell integration's arguments, and with the
/// terminal's variables and the integration's in the shell's environment alone. On the
/// Flatpak host they travel as `flatpak-spawn` arguments; for a local shell they are
/// handed to the spawn, so this process's environment, which a link's opener inherits,
/// keeps the user's own values.
fn shell_launch(
    shell: String,
    integration: Option<&crate::shell_integration::Session>,
    on_host: bool,
) -> Launch {
    let mut env = terminal_env();
    let mut argv = vec![shell];
    if let Some(session) = integration {
        argv.extend_from_slice(session.shell_args());
        env.extend_from_slice(session.env());
    }
    if on_host {
        Launch::new(crate::flatpak::host_argv(&argv, &env), Target::FlatpakHost)
    } else {
        Launch::new(argv, Target::Local).with_env(env)
    }
}

/// The terminal's side of every shell's environment: what it can do and what it is.
fn terminal_env() -> Vec<EnvVar> {
    let set = |name: &'static str, value: &str| EnvVar {
        name,
        value: Some(value.to_string()),
    };
    vec![
        set("TERM", "xterm-256color"),
        set("COLORTERM", "truecolor"),
        // Who we are. Nothing consumes this yet: the CLIs that sniff `TERM_PROGRAM` match it
        // against a hardcoded list, and bnkterm is on nobody's. It is still what a terminal
        // is supposed to say.
        set("TERM_PROGRAM", "bnkterm"),
        set("TERM_PROGRAM_VERSION", env!("CARGO_PKG_VERSION")),
        // Hyperlink support in most JS CLIs is decided by terminal name, from a fixed
        // allowlist, with no capability query for OSC 8. `FORCE_HYPERLINK` is the one way
        // past it, and bnkterm does render OSC 8 (see `grid::Screen::set_hyperlink`).
        set("FORCE_HYPERLINK", "1"),
    ]
}

/// `--demo`: open the window on the static styled grid, without a PTY. It is the same grid
/// the screenshot asset is captured from (`make screenshot`), which is why the flag stays,
/// and it isolates a render question from a live shell.
pub fn run_demo(verbosity: Verbosity) -> crate::error::Result<()> {
    let launch_token = activation::take_launch_token();
    let mut state = State::new(
        true,
        verbosity,
        Launch::new(Vec::new(), Target::Local),
        launch_token,
    )?;
    state.bring_up()?;
    Ok(())
}

/// The window the screenshot is taken in.
#[cfg(test)]
const CAPTURE_COLS: usize = 120;

/// The scale the screenshot renders at, in 120ths. Twice unity, so the asset is a 2x
/// raster: a hidpi display draws it sharply and everything else scales it down. Scale
/// rather than a larger font, so the padding and the tab strip grow with the text.
#[cfg(test)]
const CAPTURE_SCALE_120: u32 = 240;

/// Open the demo window, write its first frame to `path` as a PNG, and close.
/// The demo grid rather than a live shell, so the image is the same every run.
#[cfg(test)]
pub(crate) fn capture_demo_frame(path: &std::path::Path) -> crate::error::Result<()> {
    let mut state = State::new(
        true,
        Verbosity::Quiet,
        Launch::new(Vec::new(), Target::Local),
        None,
    )?;
    // Derive the font from the point size rather than this machine's configured pixels,
    // so the asset is the same everywhere, and take the scale before pinning the grid:
    // `set_scale_120` ignores later changes once `capture_cells` is set.
    state.configured_font_size = None;
    state.set_scale_120(CAPTURE_SCALE_120);
    // A terminal's shape comes from its columns, not from how far the text reaches.
    // Pinned before bring-up so the first configure already lands on that size.
    state.capture_cells = Some((CAPTURE_COLS, terminal::DEMO_ROWS + 1));
    state.bring_up_surface()?;
    // The strip needs two tabs or more (`Tabs::shows_bar`).
    state.stage_demo_tabs(&["~/projects/bnkterm", "nvim", "~"]);
    state.capture_to = Some(path.to_path_buf());
    state.request_capture();
    // Only a successful write clears the destination; closing early is a failure.
    state.run_until(|s| s.closed || s.capture_to.is_none())?;
    if state.capture_to.is_some() {
        return Err(crate::error::Error::msg(
            "window closed before the capture was written",
        ));
    }
    Ok(())
}

/// `--gpu-probe`: report what the compositor and GPU offer the dmabuf
/// presentation path, without opening a window. Its report *is* its output, so it
/// prints at any verbosity.
pub fn gpu_probe() -> crate::error::Result<()> {
    State::new(
        true,
        Verbosity::Quiet,
        Launch::new(Vec::new(), Target::Local),
        None,
    )?
    .probe_dmabuf()?;
    Ok(())
}

/// The device buffer a capture pinned to `wanted` uses at `factor_120`, and the logical
/// geometry to declare with it. Rounded up to the integer scale: the fallback path
/// declares that with `set_buffer_scale`, which rejects a buffer not divisible by it.
#[cfg(test)]
fn capture_surface(wanted: (u32, u32), factor_120: u32) -> ((u32, u32), (u32, u32)) {
    let factor = factor_120.max(1);
    let n = (factor / 120).max(1);
    let device = (
        wanted.0.max(1).next_multiple_of(n),
        wanted.1.max(1).next_multiple_of(n),
    );
    let logical = (
        (device.0 * 120 / factor).max(1),
        (device.1 * 120 / factor).max(1),
    );
    (device, logical)
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
    /// The current scale, as every conversion takes it: one type, one rounding rule,
    /// so the path that measures a lane and the path that draws it cannot disagree.
    fn factor(&self) -> Scale {
        Scale::from_120(self.factor_120)
    }

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

/// The window's Wayland object ids. Globals are `Option` (discovered via the
/// registry); ids we create are 0 (never valid) until assigned.
#[derive(Default)]
struct Objects {
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
    surface: u32,
    xdg_surface: u32,
    toplevel: u32,
}

/// A `wl_surface.frame` callback the compositor has not fired yet.
#[derive(Clone, Copy)]
struct FrameCallback {
    id: u32,
    /// When the latest frame was committed. A hidden window's callback never fires, so a
    /// pending configure waits for it only so long past this (see [`State::may_draw`]).
    last_frame: Instant,
}

struct State {
    conn: Connection,
    fonts: Fonts,
    /// An explicit pixel size, unaffected by compositor scale changes.
    configured_font_size: Option<u32>,
    /// Resolved font files reused when a scale change reopens the faces.
    font_selection: FontSelection,
    xkb: Xkb,
    /// The tab manager behind the window seam. It owns the terminal core (and in
    /// Phase 1, every core), routes input/timers, supplies the visible display list,
    /// and drains terminal facts for the window to act on.
    tabs: Tabs,
    /// Reused `poll(2)` descriptor storage: Wayland first, followed by gatherer
    /// wake fds. Clearing it retains capacity, so idle waits do not allocate.
    poll_set: pty::PollSet,
    /// Where everything sits: the grid's device-pixel geometry (the core holds a copy,
    /// shipped over on a resize) and the tab strip's. Window-authoritative, because the
    /// window owns the fonts and the scale. [`Self::resize_to`] recomputes it whole, so
    /// no two parts of a frame can come from different passes.
    layout: WindowLayout,
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

    /// Tab strip appearance and placement (top/bottom, height, widths, colors).
    tab_bar_config: TabBarConfig,
    /// Compositor scale factor and the objects that report it.
    scale: Scaling,

    /// The window's Wayland objects.
    objects: Objects,
    /// The version [`Objects::seat`] is bound at, which decides whether its keyboard and
    /// pointer can be released.
    seat_version: u32,
    /// Keyboards released after the seat lost its keyboard, until the compositor confirms
    /// the deletion with `wl_display.delete_id` (for good on a seat below version 3, which
    /// cannot release them). A `keymap` it queued for one before reading the release still
    /// arrives with an fd, which must be taken off the connection's fd queue or every later
    /// fd-carrying event receives the one before it.
    released_keyboards: Vec<u32>,
    /// Where the pointer is, the cursor it shows, and the press gestures in progress.
    pointer: PointerState,
    /// Whether the frame shows the hovered link's preview: the pointer offers a Ctrl+click
    /// on a link (`link_preview_shown`).
    link_preview: bool,
    /// The clipboard (Ctrl+Shift+C/V) and the primary selection (select-to-copy,
    /// middle-click paste), indexed by [`Transport`]. One state machine over two object
    /// families; a transport whose manager the compositor never advertised is a no-op.
    selections: [SelectionTransport; 2],
    /// Selection transfers still owed bytes to whoever asked for them, shared by both
    /// transports. Serving is asynchronous because the receiver decides when to read and
    /// a pipe holds only 64 KiB (see `State::begin_selection_send`).
    pending_sends: Vec<PendingSend>,
    /// The latest input-event serial, needed to claim a selection and cited by a clicked
    /// link's activation token.
    last_serial: u32,
    /// The launcher's activation token and the links waiting on theirs.
    activation: Activation,
    /// Host shells' `file:` links being judged on the Flatpak host, off the loop. `None`
    /// for a local shell, or when the wake could not be made, which refuses them all.
    host_checks: Option<HostChecks>,
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
    /// Where the screenshot harness wants the next captured frame written. Cleared
    /// only after a successful write, which is how the harness knows it is done.
    #[cfg(test)]
    capture_to: Option<std::path::PathBuf>,
    /// The grid a capture holds the window at, in place of the configured size. In
    /// cells, so a scale change re-derives the pixels instead of clipping the demo.
    #[cfg(test)]
    capture_cells: Option<(usize, usize)>,
    closed: bool,
    /// The frame callback in flight, if any. While there is one the loop holds off
    /// redrawing, so bursts coalesce into at most one frame per refresh (frame pacing).
    frame_callback: Option<FrameCallback>,
    /// A grid geometry the window has worked out but not yet handed to the tabs, set by
    /// [`Self::resize_to`] and delivered by [`Self::flush_layout`] at the next paint. See
    /// `flush_layout` for why the expensive half of a resize is paced with the frames.
    pending_layout: bool,
    /// How chatty stderr is: quiet unless the user asked for the bring-up lines
    /// (`--verbose`) or the per-frame stats line (`--stats`).
    verbosity: Verbosity,
}

impl State {
    /// `launch` is how every tab this window opens starts its shell; the paths that never
    /// spawn one (`--demo`, `--gpu-probe`) pass an empty one. `launch_token` is the
    /// activation token the launcher passed in, which the first frame spends.
    fn new(
        demo: bool,
        verbosity: Verbosity,
        launch: Launch,
        launch_token: Option<String>,
    ) -> Result<Self> {
        let (config, problems) = config::load::load();
        for problem in &problems {
            eprintln!("bnkterm: {problem}");
        }
        let conn = Connection::connect()?;
        Self::with_config(conn, config, demo, verbosity, launch, launch_token)
    }

    /// The window over `conn`, set up from `config` and not yet bound to anything: all
    /// of [`Self::new`] past reading the config file and reaching the compositor, so a
    /// test can drive the window's handlers over a socketpair.
    fn with_config(
        conn: Connection,
        config: Config,
        demo: bool,
        verbosity: Verbosity,
        launch: Launch,
        launch_token: Option<String>,
    ) -> Result<Self> {
        let (font_selection, used_default_prose) = resolve_configured_fonts(&config.fonts)?;
        if used_default_prose {
            eprintln!(
                "bnkterm: config: none of the grid fonts resolved: {:?}; using monospace",
                config.fonts.families
            );
        }
        // Open at unity scale; the compositor's real scale arrives after bring-up
        // and reopens the fonts (see `apply_scale`).
        let font_size = config_font_px(SCALE_120_UNITY, config.font_size);
        let tab_bar_config = config.tab_bar.clone();
        let label_size = label_font_px(font_size, tab_bar_config.label_scale_pct);
        let fonts = Fonts::with_selection(&font_selection, &font_sizes(font_size, label_size))?;
        let metrics = CellMetrics::from_fonts(&fonts, font_size);
        let label_metrics = CellMetrics::from_ui(&fonts, label_size);
        let (cols, rows) = (DEFAULT_COLS, DEFAULT_ROWS);
        let width = (cols as i32 * metrics.w + 2 * WINDOW_PADDING).max(1) as u32;
        let height = (rows as i32 * metrics.h + 2 * WINDOW_PADDING).max(1) as u32;
        // The terminal core owns the grid/parser/PTY and its own geometry copy. At
        // unity scale the device padding is just `WINDOW_PADDING`; the first configure
        // ships the real geometry over on a `Resize`.
        let geom = TerminalGeometry {
            cols,
            rows,
            width,
            height,
            metrics,
            pad: WINDOW_PADDING,
            origin_y: WINDOW_PADDING,
            scale: Scale::ONE,
        };
        let mut core = TerminalCore::new(demo, geom);
        let theme = Rc::new(config.theme);
        core.set_theme(Rc::clone(&theme));
        let activation = Activation::new(launch_token, launch.target());
        let host_checks = (launch.target() == Target::FlatpakHost)
            .then(HostChecks::new)
            .and_then(|checks| {
                checks
                    .map_err(|error| eprintln!("bnkterm: host file links refused: {error}"))
                    .ok()
            });

        Ok(Self {
            conn,
            fonts,
            configured_font_size: config.font_size,
            font_selection,
            xkb: Xkb::new()?,
            tabs: Tabs::new(
                core,
                tab_bar_config.clone(),
                theme,
                Hostname::read(),
                launch,
                config.shell_startup,
            ),
            poll_set: pty::PollSet::new(),
            layout: WindowLayout {
                terminal: geom,
                label_metrics,
                bar_y: WINDOW_PADDING,
                bar_h: 0,
            },
            // Sensible defaults until the compositor sends repeat_info.
            repeat_delay: Duration::from_millis(400),
            repeat_interval: Some(Duration::from_millis(33)),
            repeat_key: None,
            repeat_at: None,
            window_focused: false,
            key_mode: KeyMode::Normal,
            tab_bar_config,
            scale: Scaling::new((width, height)),
            objects: Objects::default(),
            seat_version: 0,
            released_keyboards: Vec::new(),
            pointer: PointerState::default(),
            link_preview: false,
            selections: SelectionTransport::pair(),
            pending_sends: Vec::new(),
            last_serial: 0,
            activation,
            host_checks,
            presentation: GpuPresentation::with_gamma(config_text_gamma()),
            next_id: 2, // 1 is wl_display
            free_ids: Vec::new(),
            pending_sync: None,
            pending_size: None,
            pending_configure: None,
            configured: false,
            #[cfg(test)]
            capture_to: None,
            #[cfg(test)]
            capture_cells: None,
            closed: false,
            frame_callback: None,
            pending_layout: false,
            verbosity,
        })
    }

    fn bring_up(&mut self) -> Result<()> {
        self.bring_up_surface()?;
        self.run_until(|s| s.closed)?;
        Ok(())
    }

    /// Everything [`bring_up`](Self::bring_up) does before it starts running: bind the
    /// globals, bring Vulkan up, build the surface stack, and wait out the first
    /// configure. Split out so a caller can run the window to a different finish (the
    /// screenshot harness runs it until the capture is written).
    fn bring_up_surface(&mut self) -> Result<()> {
        // Discover and bind globals.
        self.objects.registry = self.create(protocol::WL_DISPLAY, wl_display::GET_REGISTRY);
        self.roundtrip()?;

        let compositor = self.objects.compositor.ok_or_else(|| {
            Error::msg(format!(
                "compositor has no wl_compositor at version {} or newer",
                protocol::VERSION_COMPOSITOR
            ))
        })?;
        let wm_base = self
            .objects
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
        self.objects.surface = self.create(compositor, wl_compositor::CREATE_SURFACE);
        self.objects.xdg_surface =
            self.create_for(wm_base, xdg_wm_base::GET_XDG_SURFACE, self.objects.surface);
        self.objects.toplevel = self.create(self.objects.xdg_surface, xdg_surface::GET_TOPLEVEL);
        self.conn.request(
            self.objects.toplevel,
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
            self.scale.fractional_scale = self.create_for(
                fmgr,
                wp_fractional_scale_manager_v1::GET_FRACTIONAL_SCALE,
                self.objects.surface,
            );
            self.scale.viewport =
                self.create_for(vp, wp_viewporter::GET_VIEWPORT, self.objects.surface);
        }

        // The devices drive copy and paste; each is skipped when the compositor has no
        // manager for it, leaving that transport unavailable rather than broken.
        if let Some(seat) = self.objects.seat {
            self.create_selection_devices(seat);
        }

        self.create_buffers()?;
        self.init_explicit_sync();

        // Empty commit to get the initial configure; the loop draws once
        // `configured` is set.
        self.conn
            .request(self.objects.surface, wl_surface::COMMIT, &[]);
        self.run_until(|s| s.configured)?;

        // Now that the window has its granted size, spawn the shell on a PTY
        // sized to the grid. Demo mode skips this and shows its static screen.
        // Under `--verbose` bring-up says what it landed on; quiet by default, so
        // the first thing on stderr is the shell's, not ours.
        if !self.tabs.active().is_demo() {
            let (cols, rows) = self.tabs.spawn_active_shell()?;
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

        Ok(())
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
            self.resize_to(self.layout.terminal.width, self.layout.terminal.height);
            self.drain_outbox()?;
            // A program grabbing or releasing the mouse changes what a drag means, and
            // it arrives as output rather than as a pointer event: without this the
            // shape would be stale under a parked pointer until the user moved it,
            // which is exactly when starting a TUI would leave the I-beam behind. The
            // early-out on an unchanged shape makes the steady-state cost a compare.
            self.update_pointer_shape();
            // Fire any blink toggle or key repeat that has come due, all against one
            // clock read: a timer that fires must disarm against the same instant its
            // neighbours were judged by.
            let now = Instant::now();
            self.service_timers(now)?;
            // Pace to the compositor, so a burst collapses into a single repaint. A
            // resize the configures have piled up rides the same pacing, and lands
            // before the frame that has to show it.
            if self.configured && self.may_draw(now) {
                self.flush_layout();
                if self.tabs.needs_frame() && self.render_frame()? {
                    self.tabs.clear_dirty();
                }
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
                // Read the clock again rather than reusing the turn's `now`: painting
                // the frame above can take milliseconds, and a wait measured from before
                // it would overshoot every deadline by however long it took.
                self.next_wake(now, Instant::now())
            };
            self.poll_set.clear();
            let wayland_slot = self.poll_set.add(self.conn.fd());
            for fd in self.tabs.gather_fds() {
                self.poll_set.add(fd);
            }
            // A child that was not reading has to be able to end this wait, or bytes we
            // owe it sit in the queue until something unrelated wakes us. Registers
            // nothing in the steady state, where every write is taken in full. The same
            // goes for another application still reading a selection off us.
            for fd in self.tabs.write_fds() {
                self.poll_set.add_writable(fd);
            }
            for send in &self.pending_sends {
                self.poll_set.add_writable(send.fd());
            }
            // Registered only while a host check owes a verdict.
            let host_check_slot = self
                .host_checks
                .as_ref()
                .and_then(HostChecks::wake_fd)
                .map(|fd| self.poll_set.add(fd));
            // Under explicit sync, a frame held back because the compositor holds both
            // buffers waits here for one of them to be released. It stays quiet otherwise.
            let release_slot = self.release_wake_fd().map(|fd| self.poll_set.add(fd));
            self.poll_set.wait(wait)?;
            // Before anything else: whatever the far side will now take. Cheap when
            // nothing is owed, and it is what keeps a queued paste or copy moving.
            self.tabs.pump_writes()?;
            self.pump_selection_sends();
            if host_check_slot.is_some_and(|slot| self.poll_set.readable(slot)) {
                self.finish_host_checks();
            }
            if release_slot.is_some_and(|slot| self.poll_set.readable(slot)) {
                self.on_release_wake();
            }
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

    /// Fire every deadline that has come due, against the turn's `now`. Each one here
    /// has a counterpart in [`Self::next_wake`], and the pairing is load-bearing: a
    /// deadline that can be waited on but never disarmed clamps the wait to 1 ms and
    /// spins the loop forever.
    ///
    /// Ownership splits the obvious way. The blink, the fade, the bell, the
    /// synchronized-output hold and the debounced resize are the tabs' (they build the
    /// frame and own the children); key repeat is the window's, which holds the
    /// compositor's `repeat_info` and the key being held down.
    fn service_timers(&mut self, now: Instant) -> Result<()> {
        self.tabs.service_due(now)?;
        if self.repeat_at.is_some_and(|at| at <= now) {
            self.fire_repeat()?;
        }
        Ok(())
    }

    /// Whether the loop may draw now. Not while a frame callback is in flight, so a burst
    /// coalesces into one frame per refresh, unless a configure is waiting and the
    /// callback is overdue. A hidden window gets no callbacks, and the configure would
    /// otherwise stay unacked until the window is shown again. Its ack still rides a
    /// frame drawn at its size.
    fn may_draw(&self, now: Instant) -> bool {
        self.frame_callback.is_none() || self.configure_overdue_at().is_some_and(|at| at <= now)
    }

    /// When a pending configure stops waiting for the frame callback in flight.
    fn configure_overdue_at(&self) -> Option<Instant> {
        self.pending_configure?;
        self.frame_callback
            .map(|callback| callback.last_frame + FRAME_CALLBACK_OVERDUE)
    }

    /// Ask for a frame callback with the frame about to be committed. One still in flight
    /// means a configure is drawing past it. It stays, since it already asks for the next
    /// frame, and the next configure waits for it another [`FRAME_CALLBACK_OVERDUE`], so
    /// a stalled compositor gets a frame per period rather than one per configure.
    fn arm_frame_callback(&mut self, now: Instant) {
        match &mut self.frame_callback {
            Some(callback) => callback.last_frame = now,
            None => {
                let id = self.create(self.objects.surface, wl_surface::FRAME);
                self.frame_callback = Some(FrameCallback {
                    id,
                    last_frame: now,
                });
            }
        }
    }

    /// How long to block for input: the soonest of the pending cursor-blink, scrollbar
    /// fade, corner-notice, and key-repeat deadlines and the moment a configure stops
    /// waiting for the frame callback, or `None` (block indefinitely) when none is armed.
    /// The blink and fade deadlines are the core's; the others are the window's.
    ///
    /// `turn` is the instant the turn judged whether to draw, and `now` the clock as the
    /// wait begins.
    fn next_wake(&self, turn: Instant, now: Instant) -> Option<Duration> {
        let due = |at: Instant| {
            at.saturating_duration_since(now)
                .max(Duration::from_millis(1))
        };
        // While a frame callback is outstanding the compositor is driving the fade, so
        // the loop does not also need to wake for it (see the callback handler).
        [
            self.tabs.next_deadline(now, self.frame_callback.is_some()),
            self.repeat_at,
            // Only while it was ahead when the turn judged the draw. One that had passed
            // by then was drawn, or is waiting on a buffer release, and waiting on it
            // again would spin. One that passed since is still owed its draw.
            self.configure_overdue_at().filter(|&at| at > turn),
        ]
        .into_iter()
        .flatten()
        .map(due)
        .min()
    }

    /// The logical window size scaled to the device buffer size, clamped so a bogus
    /// configure cannot blow up the buffer arithmetic.
    fn device_size(&self, logical_w: u32, logical_h: u32) -> (u32, u32) {
        let scale = self.scale.factor();
        let device = |logical: u32| (scale.px(logical as i32) as u32).clamp(1, MAX_DIMENSION);
        (device(logical_w), device(logical_h))
    }

    /// Hold the surface at a `cols` x `rows` grid, padding and strip included, and
    /// declare the matching logical geometry with it.
    #[cfg(test)]
    fn pin_surface_to_cells(&mut self, cols: usize, rows: usize) -> (u32, u32) {
        let wanted = layout::surface_for_cells(
            cols,
            rows,
            self.scale.factor(),
            self.layout.terminal.metrics,
            self.strip_config(),
        );
        let (device, logical) = capture_surface(wanted, self.scale.factor_120);
        if self.scale.logical != logical {
            self.scale.logical = logical;
            self.scale.synced = false;
        }
        device
    }

    /// Record a new *device* surface size and the grid geometry that now fits inside the
    /// scaled padding, leaving [`Self::flush_layout`] to hand it to the tabs at the next
    /// paint. GPU buffers are reallocated lazily in `render_frame` for the same reason.
    /// A no-op only when the whole layout lands identical, so a scale-driven metrics
    /// change still reaches the cores at an unchanged window size.
    fn resize_to(&mut self, w: u32, h: u32) {
        #[cfg(test)]
        let (w, h) = match self.capture_cells {
            Some((cols, rows)) => self.pin_surface_to_cells(cols, rows),
            None => (w, h),
        };
        let layout = layout::window(
            (w, h),
            self.scale.factor(),
            self.layout.terminal.metrics,
            self.layout.label_metrics,
            self.strip_config(),
        );
        if layout == self.layout {
            return;
        }
        self.layout = layout;
        self.pending_layout = true;
    }

    /// The strip's configuration while it is showing, and `None` while it is hidden,
    /// which is how the layout is told to give the grid the whole window.
    fn strip_config(&self) -> Option<&TabBarConfig> {
        self.tabs.shows_bar().then_some(&self.tab_bar_config)
    }

    /// Hand the geometry [`Self::resize_to`] worked out to the tabs: each one resizes its
    /// grid (a width change rewraps it) and its PTY winsize (best-effort, so this cannot
    /// fail from here — see `apply`), and the bar takes its strip. A no-op when nothing has
    /// moved since the last paint.
    ///
    /// This is the expensive half of a resize, and it is paced with the frames rather than
    /// run per configure. A compositor delivers a configure per pointer motion during an
    /// interactive drag — several per refresh — and rewrapping the scrollback for each one
    /// throws most of that work away unpainted: measured over one drag, 600 configures
    /// against 346 frames. The window's own geometry above stays immediate, because it is
    /// arithmetic and the pointer hit-tests read it.
    fn flush_layout(&mut self) {
        if !self.pending_layout {
            return;
        }
        self.pending_layout = false;
        self.tabs.resize_all(self.layout);
    }

    /// Adopt a new compositor scale (in 120ths): reopen the fonts at the size it
    /// calls for, re-derive the device buffer size from the logical window, and
    /// resize the grid. A no-op if the scale is unchanged.
    fn set_scale_120(&mut self, factor_120: u32) {
        // A capture renders at its own scale, whatever this display reports.
        #[cfg(test)]
        if self.capture_cells.is_some() {
            return;
        }
        let factor_120 = factor_120.max(1);
        if factor_120 == self.scale.factor_120 {
            return;
        }
        self.scale.factor_120 = factor_120;
        self.scale.synced = false; // geometry + viewport/buffer-scale must be resent
        let px = config_font_px(factor_120, self.configured_font_size);
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
        if size == self.layout.terminal.metrics.size {
            return;
        }
        // On a font-open failure, keep the working fonts (no panic, no blank
        // window); the next scale event may recover.
        let label_size = label_font_px(size, self.tab_bar_config.label_scale_pct);
        let sizes = font_sizes(size, label_size);
        if let Ok(fonts) = Fonts::with_selection(&self.font_selection, &sizes) {
            self.layout.terminal.metrics = CellMetrics::from_fonts(&fonts, size);
            self.layout.label_metrics = CellMetrics::from_ui(&fonts, label_size);
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

    /// Name the active demo tab and open the rest beside it. The first title goes
    /// to the tab already open, so it is the active one.
    #[cfg(test)]
    fn stage_demo_tabs(&mut self, titles: &[&str]) {
        let Some((active, rest)) = titles.split_first() else {
            return;
        };
        self.tabs
            .active_mut()
            .feed_test_bytes(format!("\x1b]2;{active}\x07").as_bytes());
        let geom = self.layout.terminal;
        for title in rest {
            self.tabs.push_demo_tab(geom, title);
        }
        // The strip appears at two tabs and takes its height from the grid; only a
        // resize re-derives that split.
        let (w, h) = (self.layout.terminal.width, self.layout.terminal.height);
        self.resize_to(w, h);
    }

    /// Arm a capture of the next frame.
    ///
    /// The pool is reset and the tabs marked dirty so the next frame repaints whole:
    /// an idle terminal presents nothing, and a capture needs a frame to ride.
    #[cfg(test)]
    fn request_capture(&mut self) {
        self.presentation.capture = true;
        self.presentation.lists.reset();
        self.tabs.mark_dirty();
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
                ToWindow::OfferSelection(bytes) => self.set_selection(Transport::Clipboard, bytes),
                ToWindow::OfferPrimary(bytes) => self.set_selection(Transport::Primary, bytes),
                ToWindow::PastePrimary => self.paste_from(Transport::Primary)?,
                ToWindow::OpenUrl(url) => self.launch_link(ClickedLink::Url(url)),
                ToWindow::OpenHostFile(url) => self.launch_link(ClickedLink::HostFile(url)),
                // The tabs raise the corner notice for it; nothing reaches here.
                ToWindow::LinkRefused(_) => {}
                ToWindow::Closed => self.closed = true,
            }
        }
        self.tabs.reclaim_outbox(outbox);
        Ok(())
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

/// The device-pixel font size at compositor scale `scale_120`.
///
/// `BNKTERM_FONT_SIZE` overrides the configured pixel size. Without either, the point
/// size from `BNKTERM_FONT_POINTS` or [`FONT_POINTS`] follows the display scale.
fn config_font_px(scale_120: u32, configured: Option<u32>) -> u32 {
    if let Some(px) = std::env::var("BNKTERM_FONT_SIZE")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
    {
        return px.clamp(*FONT_SIZE_RANGE.start(), *FONT_SIZE_RANGE.end());
    }
    if let Some(px) = configured {
        return px.clamp(*FONT_SIZE_RANGE.start(), *FONT_SIZE_RANGE.end());
    }
    let points = std::env::var("BNKTERM_FONT_POINTS")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|p| p.is_finite() && *p > 0.0)
        .unwrap_or(FONT_POINTS);
    points_to_px(points, scale_120)
}

/// Resolve configured fonts, using the default prose family when every configured
/// grid family misses. Other configured font roles remain unchanged.
fn resolve_configured_fonts(config: &FontConfig) -> Result<(FontSelection, bool)> {
    match FontSelection::resolve(config) {
        Ok(selection) => Ok((selection, false)),
        Err(_) => {
            let mut fallback = config.clone();
            fallback.families = FontConfig::default().families;
            Ok((FontSelection::resolve(&fallback)?, true))
        }
    }
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

/// The glyph coverage gammas (see [`TextGamma`]), each overridable by its own
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
    let shipping = TextGamma::default();
    TextGamma {
        light_on_dark: dial("BNKTERM_TEXT_GAMMA", shipping.light_on_dark),
        dark_on_light: dial("BNKTERM_TEXT_GAMMA_DARK_ON_LIGHT", shipping.dark_on_light),
    }
}

#[cfg(test)]
mod tests {
    use crate::app::*;
    use crate::platform::protocol::wl_callback;

    /// A pinned capture hands the fallback path a buffer that divides by the integer
    /// scale, or the compositor answers `invalid_size` and bring-up dies.
    #[test]
    fn a_pinned_capture_surface_divides_by_the_buffer_scale() {
        assert_eq!(
            capture_surface((395, 321), 120),
            ((395, 321), (395, 321)),
            "unity takes the demo's size as it is"
        );
        assert_eq!(
            capture_surface((395, 321), 240),
            ((396, 322), (198, 161)),
            "an odd buffer rounds up at scale 2, and logical is half of it"
        );
        assert_eq!(
            capture_surface((396, 322), 180),
            ((396, 322), (264, 214)),
            "the fractional path has no divisibility rule to satisfy"
        );
        let (device, logical) = capture_surface((0, 0), 0);
        assert_eq!(device, (1, 1), "a degenerate size still yields a buffer");
        assert!(
            logical.0 >= 1 && logical.1 >= 1,
            "and a geometry the compositor can accept, with no divide by zero"
        );
    }

    /// The environment `launch` gives a child: `env` run under a PTY, read to its end.
    fn spawned_environment(launch: &Launch) -> std::collections::HashMap<String, String> {
        use crate::pty::{Pty, ReadOutcome};
        let pty = Pty::spawn(80, 24, launch).expect("spawn env");
        let mut out = Vec::new();
        let mut buf = [0u8; 4096];
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            match pty.read(&mut buf) {
                Ok(ReadOutcome::Data(n)) => out.extend_from_slice(&buf[..n]),
                Ok(ReadOutcome::WouldBlock) => {
                    std::thread::sleep(std::time::Duration::from_millis(5))
                }
                Ok(ReadOutcome::Eof) | Err(_) => break,
            }
        }
        String::from_utf8_lossy(&out)
            .lines()
            .filter_map(|line| line.trim_end_matches('\r').split_once('='))
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn a_local_shell_gets_its_environment_at_spawn_and_the_process_keeps_its_own() {
        // Changing this process's environment races every other test in the binary, so
        // the checks run in a child test process started with the environment they need.
        const PROBE: &str = "BNKTERM_ENV_PROBE";
        const NAME: &str =
            "app::tests::a_local_shell_gets_its_environment_at_spawn_and_the_process_keeps_its_own";
        if std::env::var_os(PROBE).is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([NAME, "--exact", "--nocapture", "--test-threads=1"])
                .env(PROBE, "1")
                .env("TERM", "foo")
                .env("BNKTERM_KEEP", "kept")
                .env("BNKTERM_ORIG_ZDOTDIR", "/stale/from/a/parent/bnkterm")
                .env_remove("ZDOTDIR")
                .env_remove("BNKTERM_SHELL_INTEGRATION")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            // A probe that matched no test would pass having checked nothing.
            assert!(
                out.status.success() && stdout.contains("1 passed"),
                "the probe failed: {stdout}{}",
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }

        let base = std::env::temp_dir();
        let session =
            crate::shell_integration::install("zsh", &|name| std::env::var_os(name), &[base])
                .expect("zsh integration");
        let zdotdir = session
            .env()
            .iter()
            .find(|var| var.name == "ZDOTDIR")
            .and_then(|var| var.value.clone())
            .unwrap();
        // `env` in the shell's place, so the launch prints what the shell would start with.
        let launch = shell_launch("env".to_string(), Some(&session), false);
        for tab in ["first", "second"] {
            let env = spawned_environment(&launch);
            let get = |name: &str| env.get(name).map(String::as_str);
            assert_eq!(get("BNKTERM_KEEP"), Some("kept"), "{tab} tab: inherited");
            assert_eq!(get("TERM"), Some("xterm-256color"), "{tab} tab: overridden");
            assert_eq!(get("ZDOTDIR"), Some(zdotdir.as_str()), "{tab} tab: set");
            assert_eq!(get("BNKTERM_ORIG_ZDOTDIR"), None, "{tab} tab: removed");
        }
        // What a link's opener inherits is this process's environment, untouched.
        assert_eq!(std::env::var("TERM").as_deref(), Ok("foo"));
        assert_eq!(
            std::env::var("BNKTERM_ORIG_ZDOTDIR").as_deref(),
            Ok("/stale/from/a/parent/bnkterm")
        );
        assert_eq!(std::env::var_os("ZDOTDIR"), None);
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
    fn missing_configured_grid_fonts_fall_back_to_the_default_prose() {
        let defaults = FontConfig::default();
        let (default, used_default) = match resolve_configured_fonts(&defaults) {
            Ok(resolved) => resolved,
            Err(_) => return,
        };
        assert!(!used_default);

        let config = FontConfig {
            families: vec!["NoSuchFamilyExistsAnywhere".into()],
            ..defaults
        };
        let (selection, used_default) =
            resolve_configured_fonts(&config).expect("the default prose family resolves");

        assert!(used_default);
        assert_eq!(selection.prose.regular, default.prose.regular);
    }

    #[test]
    fn default_prose_fallback_keeps_other_configured_font_roles() {
        let Some(serif) = crate::platform::freetype::FontFamily::resolve("serif") else {
            return;
        };
        let config = FontConfig {
            families: vec!["NoSuchFamilyExistsAnywhere".into()],
            ui: vec!["serif".into()],
            fallback: vec!["serif".into()],
            ..FontConfig::default()
        };

        let (selection, used_default) =
            resolve_configured_fonts(&config).expect("the default prose family resolves");

        assert!(used_default);
        assert_eq!(selection.ui.regular, serif.regular);
        assert_eq!(selection.fallback, [serif.regular]);
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

    #[test]
    fn each_verbosity_prints_what_its_level_promises() {
        // Quiet prints neither; `--stats` implies the bring-up lines too, since the frame
        // lines are useless without knowing which renderer produced them.
        assert!(!Verbosity::Quiet.verbose());
        assert!(!Verbosity::Quiet.frame_stats());
        assert!(Verbosity::Verbose.verbose());
        assert!(!Verbosity::Verbose.frame_stats());
        assert!(Verbosity::Stats.verbose());
        assert!(Verbosity::Stats.frame_stats());
    }

    /// A window over a socketpair, on the default config and the demo grid with a tab per
    /// title, and no compositor behind it, so its handlers can be driven directly. The
    /// other end of the socket comes back with it, so its requests have somewhere to go.
    pub(super) fn window(titles: &[&str]) -> (State, std::os::unix::net::UnixStream) {
        let (ours, compositor) = std::os::unix::net::UnixStream::pair().expect("socketpair");
        let mut state = State::with_config(
            Connection::from_stream(ours),
            Config::default(),
            true,
            Verbosity::Quiet,
            Launch::new(Vec::new(), Target::Local),
            None,
        )
        .expect("the window state");
        state.stage_demo_tabs(titles);
        state.flush_layout();
        (state, compositor)
    }

    /// Hand the window one event on `object`, a `u32` per argument.
    pub(super) fn event(state: &mut State, object: u32, opcode: u16, args: &[u32]) {
        let body = args.iter().flat_map(|a| a.to_ne_bytes()).collect();
        state
            .handle(&Message {
                object,
                opcode,
                body,
            })
            .expect("the event");
    }

    /// A hidden window gets no frame callbacks. A configure that waited for the one in
    /// flight stayed unacked until the window was shown again, and labwc gave up on it
    /// after 100 ms. Once the callback is overdue the configure draws without it, and the
    /// loop wakes for that moment. A callback a refresh old still paces it, and so does
    /// the frame drawn past it, so the next configure waits a period from that frame.
    #[test]
    fn a_configure_does_not_wait_for_an_overdue_frame_callback() {
        let (mut state, _compositor) = window(&["one"]);
        state.configured = true;
        state.objects.xdg_surface = 30;
        let drawn = Instant::now();
        state.frame_callback = Some(FrameCallback {
            id: 40,
            last_frame: drawn,
        });
        let overdue = drawn + FRAME_CALLBACK_OVERDUE;
        let soon = drawn + Duration::from_millis(5);
        assert!(
            !state.may_draw(overdue + Duration::from_secs(60)),
            "output alone waits for the callback, however long"
        );
        assert_eq!(state.next_wake(soon, soon), None, "nothing to wake for yet");

        event(&mut state, 30, xdg_surface::EV_CONFIGURE, &[7]);
        assert!(
            !state.may_draw(soon),
            "a fresh callback paces the configure"
        );
        assert_eq!(
            state.next_wake(soon, soon),
            Some(overdue - soon),
            "the loop wakes when the callback is overdue"
        );
        let after = overdue + Duration::from_micros(10);
        assert_eq!(
            state.next_wake(soon, after),
            Some(Duration::from_millis(1)),
            "a deadline that passed after the turn judged the draw is still owed one"
        );
        assert!(
            state.may_draw(overdue),
            "an overdue callback holds nothing back"
        );
        assert_eq!(
            state.next_wake(overdue, after),
            None,
            "a turn that tried the draw does not wait on the deadline again"
        );

        state.arm_frame_callback(overdue);
        event(&mut state, 30, xdg_surface::EV_CONFIGURE, &[8]);
        assert_eq!(
            state.frame_callback.map(|callback| callback.id),
            Some(40),
            "the frame drawn past the callback keeps it in flight"
        );
        let next = overdue + Duration::from_millis(10);
        assert!(
            !state.may_draw(next),
            "the next configure waits a period from the frame drawn past it"
        );
        assert_eq!(
            state.next_wake(next, next),
            Some(FRAME_CALLBACK_OVERDUE - Duration::from_millis(10)),
            "and the loop wakes when that period is up"
        );

        event(&mut state, 40, wl_callback::EV_DONE, &[0]);
        assert!(state.may_draw(soon), "a fired callback paces nothing");
    }
}
