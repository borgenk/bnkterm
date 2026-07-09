//! Pure viewport-scroll math, kept separate from the Wayland plumbing so it can
//! be tested on its own.
//!
//! The document is laid out in content space (y grows downward from 0); the
//! window shows the band `[scroll_y, scroll_y + viewport_height)`. `scroll_y` is
//! how far down the document the top of the window sits, always in
//! `0..=max_scroll`.

/// The largest valid downward scroll: 0 when the whole document fits.
pub fn max_scroll(content_height: i32, viewport_height: i32) -> i32 {
    (content_height - viewport_height).max(0)
}

/// Clamp `scroll_y` into the valid range for this content and viewport.
pub fn clamp(scroll_y: i32, content_height: i32, viewport_height: i32) -> i32 {
    scroll_y.clamp(0, max_scroll(content_height, viewport_height))
}

/// The scroll offset that brings a caret spanning content y `[top, top + height)`
/// into view, keeping `pad` pixels of breathing room at the edge it enters from,
/// while moving as little as possible from `scroll_y`. The result is clamped to
/// the document, so a caret in an already-visible position leaves `scroll_y`
/// unchanged.
pub fn reveal(
    scroll_y: i32,
    top: i32,
    height: i32,
    viewport_height: i32,
    content_height: i32,
    pad: i32,
) -> i32 {
    let sy = if top - pad < scroll_y {
        // Caret is above the visible band: scroll up to it.
        top - pad
    } else if top + height + pad > scroll_y + viewport_height {
        // Caret is below the visible band: scroll down to it.
        top + height + pad - viewport_height
    } else {
        scroll_y
    };
    clamp(sy, content_height, viewport_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_scroll_is_zero_when_content_fits() {
        assert_eq!(max_scroll(300, 600), 0);
        assert_eq!(max_scroll(600, 600), 0);
        assert_eq!(max_scroll(900, 600), 300);
    }

    #[test]
    fn clamp_bounds_into_valid_range() {
        // content 900, viewport 600 -> max 300.
        assert_eq!(clamp(-50, 900, 600), 0);
        assert_eq!(clamp(150, 900, 600), 150);
        assert_eq!(clamp(500, 900, 600), 300);
        // Everything fits: only 0 is valid.
        assert_eq!(clamp(100, 400, 600), 0);
    }

    #[test]
    fn reveal_leaves_an_in_view_caret_untouched() {
        // viewport [0, 600); caret at 100..130 with pad 40 is comfortably inside.
        assert_eq!(reveal(0, 100, 30, 600, 2000, 40), 0);
        // Same caret with the window scrolled so it sits mid-band.
        assert_eq!(reveal(50, 100, 30, 600, 2000, 40), 50);
    }

    #[test]
    fn reveal_scrolls_up_to_a_caret_above_the_band() {
        // Window at 500..1100; caret at content 480 is above it.
        let sy = reveal(500, 480, 30, 600, 2000, 40);
        assert_eq!(sy, 440, "aligns the caret top minus the pad");
    }

    #[test]
    fn reveal_scrolls_down_to_a_caret_below_the_band() {
        // Window at 0..600; caret bottom at 700+30 is below it.
        let sy = reveal(0, 700, 30, 600, 2000, 40);
        assert_eq!(sy, 700 + 30 + 40 - 600, "aligns the caret bottom plus pad");
    }

    #[test]
    fn reveal_never_exceeds_the_document() {
        // A caret on the last line cannot scroll past max_scroll.
        let content = 1000;
        let sy = reveal(0, 980, 30, 600, content, 40);
        assert_eq!(sy, max_scroll(content, 600));
    }
}
