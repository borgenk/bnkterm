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

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            // Prose is the terminal grid's one family: a monospace, first
            // installed wins. Paths are hardcoded by design; a short list keeps
            // the terminal runnable across machines without a font-discovery
            // crate.
            families: vec![
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
            // icon to one cell, so it leads when installed. The Noto symbol faces
            // mop up other Unicode symbols and dingbats.
            fallback: vec![
                "/usr/share/fonts/TTF/SymbolsNerdFontMono-Regular.ttf".into(),
                "/usr/share/fonts/TTF/SymbolsNerdFont-Regular.ttf".into(),
                "/usr/share/fonts/noto/NotoSansSymbols-Regular.ttf".into(),
                "/usr/share/fonts/noto/NotoSansSymbols2-Regular.ttf".into(),
            ],
        }
    }
}
