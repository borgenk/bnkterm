//! Window-pixel geometry primitives shared by every layer: the app lays lines
//! out in them, the painter clips to them, and the GPU backend scissors to them,
//! plus the display scale that turns a logical length into the device pixels they
//! are all measured in.

/// An axis-aligned rectangle in window pixels.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    /// Whether the window point `(px, py)` falls inside this rectangle, left/top
    /// inclusive and right/bottom exclusive, as the pointer hit-tests want.
    pub fn contains(&self, px: f32, py: f32) -> bool {
        let (x, y) = (px as i32, py as i32);
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}

/// The display scale the compositor reports, in 120ths.
///
/// Chrome constants (the scrollbar's width, the window padding) are written once in
/// logical pixels and pass through [`Scale::px`], so they come out the same physical
/// size on a 1x display and a 2x one. The grid needs no such treatment: its metrics
/// come from the font, which is reopened at the scaled pixel size.
///
/// It exists as a type rather than a bare `u32` so that the path which *measures* the
/// scrollbar's lane (the pointer hit-test) and the path which *draws* it (the painter)
/// cannot scale it differently. They take the same `Scale` and call the same function.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Scale(u32);

impl Scale {
    /// Unity: one device pixel per logical pixel. What tests and a headless build run at.
    pub const ONE: Scale = Scale(120);

    /// The scale a compositor reported. A zero factor would collapse every length to
    /// nothing, so it floors at 1.
    pub fn from_120(factor_120: u32) -> Self {
        Scale(factor_120.max(1))
    }

    /// A logical length in device pixels, unrounded. The pointer arrives as a float and
    /// is hit-tested against cell and lane edges, where rounding to whole device pixels
    /// would move the boundary it is being compared to.
    pub fn pxf(self, logical: f32) -> f32 {
        logical * self.0 as f32 / 120.0
    }

    /// A logical length in device pixels, rounded to nearest — the `+ 60` is half a
    /// step. Device pixels are what the buffer, the grid, and every display-list
    /// rectangle are measured in. A negative length has no meaning in chrome geometry,
    /// so it scales to zero rather than wrapping.
    pub fn px(self, logical: i32) -> i32 {
        let scaled = (u64::from(logical.max(0) as u32) * u64::from(self.0) + 60) / 120;
        scaled.min(i32::MAX as u64) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pxf_keeps_the_fraction_px_rounds_away() {
        // The hit-test path measures in fractions of a device pixel; the layout path
        // rounds. Same scale, same direction, different precision.
        assert_eq!(Scale::ONE.pxf(10.5), 10.5);
        assert_eq!(Scale::from_120(240).pxf(10.5), 21.0);
        assert_eq!(Scale::from_120(150).pxf(1.0), 1.25);
        assert_eq!(Scale::from_120(150).px(1), 1);
    }

    #[test]
    fn scale_px_rounds_to_nearest() {
        assert_eq!(Scale::ONE.px(10), 10);
        assert_eq!(Scale::from_120(240).px(10), 20);
        assert_eq!(Scale::from_120(180).px(10), 15);
        assert_eq!(Scale::from_120(150).px(100), 125);
        // 126.25 rounds down to 126.
        assert_eq!(Scale::from_120(150).px(101), 126);
        // No overflow at the extremes (u64 math, then narrowed).
        assert_eq!(Scale::from_120(240).px(16384), 32768);
        assert_eq!(Scale::from_120(240).px(i32::MAX), i32::MAX);
        // A nonsense scale floors at 1/120th rather than annihilating the chrome.
        assert_eq!(Scale::from_120(0).px(1200), 10);
        // Negative lengths are not geometry; they scale to nothing, never wrap.
        assert_eq!(Scale::ONE.px(-5), 0);
    }
}
