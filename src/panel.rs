//! The plate that chrome floating over the grid stands on: the corner notice, the leader
//! overlay and the link preview.
//!
//! Each of those decides where its panel goes; a [`Panel`] sizes the plate around the
//! text, pushes it, and writes the text on top in fixed-pitch cells, so the three look
//! like one family. Only the spacing differs, between a panel that floats free
//! ([`Panel::roomy`]) and one that sits against the text it names ([`Panel::snug`]).
//!
//! ```text
//!   ┌──────────────────────────────┐ ← plate: the background darkened halfway to black,
//!   │         pad_y                │   corners rounded by `radius`
//!   │ pad_x ┌──────────────┐       │
//!   │       │ line 0       │       │ ← text in cells, `col` cells in from the padding
//!   │       │ line 1       │       │
//!   │       └──────────────┘       │
//!   └──────────────────────────────┘
//! ```

use crate::color::{Rgb, Theme};
use crate::platform::freetype::{FaceKey, FontStyle};
use crate::platform::geom::Rect;
use crate::render::display::{DisplayList, DrawCmd};
use crate::term_render::{self, CellMetrics};

/// A plate's spacing at one cell size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Panel {
    metrics: CellMetrics,
    /// Pixels between the plate's left and right edges and its text.
    pad_x: i32,
    /// Pixels between the plate's top and bottom edges and its text.
    pad_y: i32,
    radius: i32,
}

impl Panel {
    /// A cell across, half a cell down, corners a third of a cell round: the notice and the
    /// leader overlay, which float clear of the grid under them.
    pub(crate) fn roomy(metrics: CellMetrics) -> Self {
        Self {
            metrics,
            pad_x: metrics.w,
            pad_y: (metrics.h / 2).max(1),
            radius: (metrics.h / 3).max(2),
        }
    }

    /// Half a cell across, a sixth down, corners a quarter round: the link preview, which
    /// sits tight against the link it names.
    pub(crate) fn snug(metrics: CellMetrics) -> Self {
        Self {
            metrics,
            pad_x: (metrics.w / 2).max(1),
            pad_y: (metrics.h / 6).max(1),
            radius: (metrics.h / 4).max(2),
        }
    }

    /// The plate's colour under `theme`: the background darkened halfway to black.
    pub(crate) fn plate(theme: &Theme) -> Rgb {
        theme.bg.mix(Rgb::new(0, 0, 0), 1, 2)
    }

    /// The outer width of a plate whose text is `cols` cells wide.
    pub(crate) fn width(&self, cols: i32) -> i32 {
        cols * self.metrics.w + 2 * self.pad_x
    }

    /// How many cells of text fit across a plate `outer` pixels wide: the inverse of
    /// [`Self::width`].
    pub(crate) fn cols_within(&self, outer: i32) -> i32 {
        (outer - 2 * self.pad_x)
            .checked_div(self.metrics.w)
            .unwrap_or(0)
    }

    /// The outer height of a plate holding `lines` lines of text.
    pub(crate) fn height(&self, lines: i32) -> i32 {
        lines * self.metrics.h + 2 * self.pad_y
    }

    /// Push the plate over `rect` in `plate` and return the pen that writes on it in `fg`.
    pub(crate) fn open<'a>(
        &self,
        out: &'a mut DisplayList,
        strings: &'a mut Vec<String>,
        rect: Rect,
        fg: Rgb,
        plate: Rgb,
    ) -> Pen<'a> {
        out.push(DrawCmd::RoundRect {
            rect,
            color: plate.to_u32(),
            radius: self.radius,
        });
        Pen {
            out,
            strings,
            metrics: self.metrics,
            x: rect.x + self.pad_x,
            y: rect.y + self.pad_y,
            fg: fg.to_u32(),
            plate: plate.to_u32(),
        }
    }
}

/// Writes text on an open [`Panel`]: its runs are pushed after the plate, so they land on it.
pub(crate) struct Pen<'a> {
    out: &'a mut DisplayList,
    strings: &'a mut Vec<String>,
    metrics: CellMetrics,
    /// The top-left of the text area, inside the padding.
    x: i32,
    y: i32,
    fg: u32,
    plate: u32,
}

impl Pen<'_> {
    /// Write `text` on `line`, starting `col` cells in from the padding.
    pub(crate) fn write(&mut self, line: i32, col: i32, text: &str) {
        let m = self.metrics;
        term_render::push_cell_text(
            self.out,
            self.strings,
            text,
            self.x + col * m.w,
            self.y + line * m.h + m.baseline,
            m,
            FaceKey::Prose {
                size: m.size,
                style: FontStyle::Regular,
            },
            self.fg,
            self.plate,
        );
    }
}
