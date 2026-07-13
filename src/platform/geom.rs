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

/// A logical (surface-local) length in device pixels at scale `factor_120` (120ths,
/// 120 = 1.0), rounded to nearest — the `+ 60` is half a step. Device pixels are what
/// the buffer, the grid, and every display-list rectangle are measured in.
pub fn logical_to_device(logical: u32, factor_120: u32) -> u32 {
    (((logical as u64) * (factor_120 as u64) + 60) / 120) as u32
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

    /// A logical length in device pixels. A negative length has no meaning in chrome
    /// geometry, so it scales to zero rather than wrapping.
    pub fn px(self, logical: i32) -> i32 {
        let scaled = logical_to_device(logical.max(0) as u32, self.0);
        scaled.min(i32::MAX as u32) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn scale_px_matches_the_logical_conversion() {
        assert_eq!(Scale::ONE.px(10), 10);
        assert_eq!(Scale::from_120(240).px(10), 20);
        assert_eq!(Scale::from_120(180).px(10), 15);
        // A nonsense scale floors at 1/120th rather than annihilating the chrome.
        assert_eq!(Scale::from_120(0).px(1200), 10);
        // Negative lengths are not geometry; they scale to nothing, never wrap.
        assert_eq!(Scale::ONE.px(-5), 0);
    }
}
