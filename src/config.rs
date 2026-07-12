//! App configuration: the concrete font choices that were once hardcoded in the
//! shared `platform/freetype.rs`, gathered here so the portable font layer stays
//! free of per-project paths (it takes a [`FontConfig`]; this app supplies one).
//!
//! This module owns the [`Default`] for [`FontConfig`]: a terminal is monospace
//! end to end, so the prose family is a monospace (Hack first), the code arm
//! carries a distinct monospace for anything that asks for it, and the fallback
//! chain fills the private-use icon ranges a prompt or statusline emits (Nerd
//! Font / Powerline glyphs) that no text family carries. A future on-disk config
//! loader replaces these built-ins at startup.

use crate::color::Rgb;
use crate::platform::freetype::{FontConfig, FontFamily};
use std::path::PathBuf;

/// A font file in the XDG user font directory (`$XDG_DATA_HOME/fonts`, else
/// `$HOME/.local/share/fonts`). Used for fonts a user installs themselves, e.g.
/// Consolas, a Microsoft font no distro packages under `/usr/share/fonts`.
/// Derived from the environment rather than a literal `/home/<user>/…` so it
/// stays portable; when neither variable is set it yields a path that will not
/// exist, and the family simply falls through to the next candidate.
fn user_font(name: &str) -> String {
    let dir = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_default();
    dir.join("fonts").join(name).to_string_lossy().into_owned()
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            // Prose is the terminal grid's one family: a monospace, first
            // installed wins. Paths are hardcoded by design; a short list keeps
            // the terminal runnable across machines without a font-discovery
            // crate.
            families: vec![
                // Consolas leads: a common primary monospace face. User-installed,
                // so it sits in the XDG user font dir; absent, the list falls
                // through to Hack.
                FontFamily::new(
                    &user_font("consola.ttf"),
                    &user_font("consolab.ttf"),
                    &user_font("consolai.ttf"),
                    &user_font("consolaz.ttf"),
                ),
                FontFamily::new(
                    "/usr/share/fonts/TTF/Hack-Regular.ttf",
                    "/usr/share/fonts/TTF/Hack-Bold.ttf",
                    "/usr/share/fonts/TTF/Hack-Italic.ttf",
                    "/usr/share/fonts/TTF/Hack-BoldItalic.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
                    "/usr/share/fonts/noto/NotoSansMono-Bold.ttf",
                    // Noto Sans Mono ships no italic; these paths simply will not
                    // exist, and the lookup falls back to the regular face.
                    "/usr/share/fonts/noto/NotoSansMono-Italic.ttf",
                    "/usr/share/fonts/noto/NotoSansMono-BoldItalic.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf",
                    "/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf",
                    "/usr/share/fonts/Adwaita/AdwaitaMono-Italic.ttf",
                    "/usr/share/fonts/Adwaita/AdwaitaMono-BoldItalic.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
                    "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
                    // DejaVu calls its slanted styles Oblique rather than Italic.
                    "/usr/share/fonts/TTF/DejaVuSansMono-Oblique.ttf",
                    "/usr/share/fonts/TTF/DejaVuSansMono-BoldOblique.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/TTF/Roboto-Regular.ttf",
                    "/usr/share/fonts/TTF/Roboto-Bold.ttf",
                    "/usr/share/fonts/TTF/Roboto-Italic.ttf",
                    "/usr/share/fonts/TTF/Roboto-BoldItalic.ttf",
                ),
            ],
            // Interface faces: a proportional sans for chrome (the tab bar), first
            // installed wins. A real UI sans reads cleaner and carries a heavier,
            // clearly distinct Bold at small sizes than the monospace body family
            // does, so the active tab's weight actually shows. These are the
            // desktop's own UI-sans candidates (Ghostty's GTK tabs use whatever the
            // system font is; on GNOME that is Adwaita Sans, a variable font this
            // static-face loader cannot pull a Bold from, so the static Noto Sans /
            // Roboto pairs lead). Only regular + bold are consulted; italics are
            // listed for shape but unused. Absent all of these, UI text falls back
            // to the prose family.
            ui: vec![
                FontFamily::new(
                    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
                    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
                    "/usr/share/fonts/noto/NotoSans-Italic.ttf",
                    "/usr/share/fonts/noto/NotoSans-BoldItalic.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/TTF/Roboto-Regular.ttf",
                    "/usr/share/fonts/TTF/Roboto-Bold.ttf",
                    "/usr/share/fonts/TTF/Roboto-Italic.ttf",
                    "/usr/share/fonts/TTF/Roboto-BoldItalic.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
                    "/usr/share/fonts/liberation/LiberationSans-Italic.ttf",
                    "/usr/share/fonts/liberation/LiberationSans-BoldItalic.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/TTF/DejaVuSans.ttf",
                    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                    "/usr/share/fonts/TTF/DejaVuSans-Oblique.ttf",
                    "/usr/share/fonts/TTF/DejaVuSans-BoldOblique.ttf",
                ),
            ],
            // The medium-weight interface face for the tab-bar label: heavier than
            // regular, lighter than bold. One file per candidate, matched in the same
            // order as `ui` above so the weight tracks the chosen family. Only true
            // medium files are listed (Liberation Sans and DejaVu Sans ship none); if
            // none is installed the label stays at the UI regular weight.
            ui_medium: vec![
                "/usr/share/fonts/noto/NotoSans-Medium.ttf".into(),
                "/usr/share/fonts/TTF/Roboto-Medium.ttf".into(),
            ],
            // Code faces: a monospace deliberately distinct from the prose family
            // for anything drawn through the code arm. Code never renders bold or
            // italic, so only the regular face is listed.
            code: vec![
                "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf".into(),
                "/usr/share/fonts/noto/NotoSansMono-Regular.ttf".into(),
                "/usr/share/fonts/liberation/LiberationMono-Regular.ttf".into(),
                "/usr/share/fonts/TTF/DejaVuSansMono.ttf".into(),
                "/usr/share/fonts/gnu-free/FreeMono.otf".into(),
            ],
            // Fallback chain for private-use icons and stray symbols the prose
            // family lacks. Symbols Nerd Font is the icons-only companion built
            // to fill the Nerd Font / Powerline ranges; its Mono cut sizes every
            // icon to one cell, so it leads when installed. When only the non-Mono
            // cut is present it renders icons larger than one cell, on purpose, see
            // the natural-size policy in `platform/freetype.rs` (fallback faces are
            // not scaled to the cell). The Noto symbol faces mop up other Unicode
            // symbols and dingbats.
            fallback: vec![
                "/usr/share/fonts/TTF/SymbolsNerdFontMono-Regular.ttf".into(),
                "/usr/share/fonts/TTF/SymbolsNerdFont-Regular.ttf".into(),
                "/usr/share/fonts/noto/NotoSansSymbols-Regular.ttf".into(),
                "/usr/share/fonts/noto/NotoSansSymbols2-Regular.ttf".into(),
            ],
        }
    }
}

/// Where the tab strip sits relative to the grid. The strip steals its height
/// from the grid on the side it lives; `Bottom` is the wezterm/ghostty default
/// this config mirrors (`tab_bar_at_bottom = true`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TabBarPosition {
    Top,
    #[default]
    Bottom,
}

/// The foreground/background pair painting one tab state. Absolute colors (not
/// theme references) so they match the wezterm reference exactly and stay a
/// self-contained, `Copy` config value; a future theme-aware config can make
/// individual channels follow the palette.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TabColors {
    pub fg: Rgb,
    pub bg: Rgb,
}

/// Tab-strip appearance and layout. These were the hardcoded constants the tab
/// bar once carried inline; gathered here as the single source of truth the app
/// threads through, exactly as [`FontConfig`] gathers the font choices. There is
/// no on-disk config loader yet, so these are
/// the compile-time defaults a future loader will overwrite at startup.
///
/// The values mirror the sibling `wezterm.lua` tab palette: a muted-turquoise
/// active block on dark teal, faint inactive tabs that share the bar (terminal)
/// background, and a divider between two inactive tabs so equal-width blocks stay
/// separable when they share that background.
///
/// Not `Copy` (it owns [`path_prefix_programs`](Self::path_prefix_programs)); it is
/// built once and threaded by reference, never copied on a hot path.
#[derive(Clone, PartialEq, Debug)]
pub struct TabBarConfig {
    /// Top or bottom of the window (default `Bottom`).
    pub position: TabBarPosition,
    /// Strip height in *logical* pixels. It is DPI-scaled at use and floored at
    /// one text row, so a value below the row height simply yields a one-row bar
    /// rather than clipping the label.
    pub height_px: u32,
    /// Breathing room between the strip and the grid, in *logical* pixels (the same
    /// unit as [`height_px`], DPI-scaled at use). Reserved from the grid on the
    /// side the strip lives, so the chrome never butts against the terminal text.
    /// Only applied while the strip is visible.
    pub gap_px: u32,
    /// Per-tab width bounds in cells. Every tab renders at one equal width clamped
    /// into `[min_width, max_width]`; the cap keeps two tabs from each stretching
    /// to half the window (the wezterm left-aligned-blocks look), the floor keeps
    /// them legible until the bar is too crowded to honor it.
    pub min_width: usize,
    pub max_width: usize,
    /// The active (focused) tab's colors: a muted-turquoise block.
    pub active: TabColors,
    /// Every inactive tab's colors. `bg` doubles as the whole strip background, so
    /// it should match the terminal background for a seamless bar.
    pub inactive: TabColors,
    /// The one-pixel rule drawn between two adjacent inactive tabs (never beside
    /// the active block), so equal-width tabs sharing the strip background do not
    /// blur together.
    pub divider: Rgb,
    /// Tab-label font size as a percentage of the terminal font size, so labels
    /// read as chrome rather than body text. Applied window-side to pick the label
    /// point size and clamped to a sane floor there (see `app`). A value of `100`
    /// keeps the label at the terminal size.
    pub label_scale_pct: u16,
    /// Programs (matched by the foreground process group's `comm`) whose tab label
    /// is prefixed with the working directory. A program that sets its own title
    /// (e.g. `claude`, whose title is the session name) otherwise hides which
    /// directory it runs in; the prefix keeps that visible. The path stays pinned
    /// because labels truncate from the end (see `tab_bar::fit_end`), so the volatile
    /// title is what clips, not the directory.
    pub path_prefix_programs: Vec<String>,
}

impl Default for TabBarConfig {
    fn default() -> Self {
        Self {
            position: TabBarPosition::Bottom,
            height_px: 20,
            gap_px: 6,
            min_width: 10,
            max_width: 24,
            active: TabColors {
                fg: Rgb::new(0x12, 0x30, 0x28), // dark teal, strong on the turquoise
                bg: Rgb::new(0x8a, 0xbe, 0xb7), // muted turquoise
            },
            inactive: TabColors {
                fg: Rgb::new(0x9a, 0xa3, 0xb0),
                bg: Rgb::new(0x28, 0x2c, 0x34), // == the default terminal background
            },
            divider: Rgb::new(0x3f, 0x46, 0x53),
            label_scale_pct: 85,
            path_prefix_programs: vec!["claude".to_string()],
        }
    }
}
