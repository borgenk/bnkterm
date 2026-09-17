//! The settings the terminal starts with, and the [`Config`] value that carries
//! them through the program.
//!
//! Fonts are named by family. Fontconfig resolves a family against every font
//! installed on the system, including the user's own in `~/.local/share/fonts`.

pub mod load;

use crate::color::{Rgb, Theme};
use crate::platform::freetype::FontConfig;
use std::time::Duration;

/// Valid device-pixel font sizes for configuration and display scaling.
pub const FONT_SIZE_RANGE: std::ops::RangeInclusive<u32> = 6..=72;

/// The compiled defaults with the on-disk config layered over them.
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub fonts: FontConfig,
    /// The grid font size in device pixels, when the user pinned one. `None` leaves the
    /// size derived from the point size and the compositor's scale, which is what makes a
    /// window look the same on a 1x and a 2x display.
    pub font_size: Option<u32>,
    pub theme: Theme,
    pub tab_bar: TabBarConfig,
    pub shell_startup: ShellStartupConfig,
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            families: vec!["monospace".into()],
            ui: vec!["sans-serif".into()],
            code: Vec::new(),
            emoji: "emoji".into(),
            // Prefer cell-fitted symbols before the wider cut.
            fallback: vec!["Symbols Nerd Font Mono".into(), "Symbols Nerd Font".into()],
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

/// When a shell takes long enough reaching its first prompt to be worth saying so,
/// and how long the notice then stands.
///
/// A terminal cannot see *why* a shell is slow — that is the shell's own startup, and
/// diagnosing it means knowing what `compinit` or a plugin manager is doing, which is
/// knowledge a terminal has no business holding. What it can do is separate its own
/// cost from the shell's and say which one the wait belonged to, because the user
/// staring at an empty tab cannot tell them apart. The clock is exact rather than
/// guessed: `fork` to the first `OSC 133;A` the injected integration emits (see
/// [`crate::shell_integration`]), so a shell with no marks is never measured and never
/// warned about.
///
/// [`Self::warn_after`] is a plain absolute threshold, deliberately: a *relative* one
/// would mean the terminal keeping a history of what "normal" is for this machine, and
/// a slow shell that is slow every time is exactly the case that history would learn to
/// call normal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ShellStartupConfig {
    /// How long a shell may take to reach its first prompt before the notice appears.
    /// `None` disables it outright, which is what a `0` in a future config file means:
    /// a startup cost paid on purpose (a plugin manager, a version-manager hook) is a
    /// choice, and a terminal that keeps second-guessing it is a terminal you turn off.
    pub warn_after: Option<Duration>,
    /// How long the notice stands at full strength before it starts fading.
    pub hold: Duration,
    /// How long it then takes to fade out. The display list carries no alpha, so this
    /// is the panel and its text mixing toward the background (see
    /// [`crate::notice::paint`]), which is also how the overlay scrollbar fades.
    pub fade: Duration,
}

impl Default for ShellStartupConfig {
    fn default() -> Self {
        Self {
            // 150ms. A zsh with an empty rc reaches its prompt in ~5ms and a healthy one
            // with completion cached in ~40ms, so this is "not snappy" rather than
            // "broken", and it is meant to fire while the cause is still fresh. The whole
            // point is catching a config that quietly got expensive, which a threshold
            // set at the pain point (half a second) would sit under forever. It sits high
            // enough above a healthy shell to leave room for an ordinary rc — a couple of
            // `eval`ed hooks, a version manager — without calling that a fault.
            warn_after: Some(Duration::from_millis(150)),
            hold: Duration::from_millis(1_800),
            fade: Duration::from_millis(400),
        }
    }
}

/// Tab-strip appearance and layout.
#[derive(Clone, PartialEq, Debug)]
pub struct TabBarConfig {
    /// Top or bottom of the window (default `Bottom`).
    pub position: TabBarPosition,
    /// Strip height in *logical* pixels. It is DPI-scaled at use and floored at
    /// one text row, so a value below the row height simply yields a one-row bar
    /// rather than clipping the label.
    pub height_px: u32,
    /// Breathing room between the strip and the grid, in *logical* pixels (the same
    /// unit as [`Self::height_px`], DPI-scaled at use). Reserved from the grid on the
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
    /// is prefixed with the working directory. An application title describing a
    /// session can hide which directory it runs in; the prefix keeps that visible.
    /// The path stays pinned because labels truncate from the end (see
    /// `tab_bar::fit_end`), so the volatile title clips before the directory.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::freetype::FontSelection;

    #[test]
    fn the_shipped_defaults_resolve_on_this_machine() {
        let selection = FontSelection::resolve(&FontConfig::default())
            .expect("the default families resolve to something installed");
        assert!(selection.prose.regular.path.exists());
        assert!(selection.ui.regular.path.exists());
    }

    #[test]
    fn the_prose_and_chrome_defaults_are_generic_aliases() {
        let config = FontConfig::default();
        for (role, names) in [("families", &config.families), ("ui", &config.ui)] {
            let last = names.last().expect("{role} has candidates");
            assert!(
                last == "monospace" || last == "sans-serif",
                "{role} ends in {last:?}, which is a font that may not be installed"
            );
        }
        assert!(config.code.is_empty(), "the code arm is the user's to name");
    }
}
