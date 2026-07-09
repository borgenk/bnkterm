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
