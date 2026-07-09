//! Pure ARGB pixel math: premultiplied source-over compositing, straight vs.
//! premultiplied conversion, and area/bilinear resampling. No FreeType or
//! Vulkan dependency, so the color arithmetic behind emoji rasterization is
//! unit-testable on plain buffers and reusable from the perf harness.

/// Premultiplied source-over composite of `src` on `dst`, both premultiplied
/// `0xAARRGGBB`: every channel, alpha included, is `s + d * (255 - sa) / 255`.
/// Used only when a shaped cluster places more than one glyph; the common
/// single-glyph case writes onto transparent pixels where this is a plain copy.
pub(crate) fn premultiplied_over(src: u32, dst: u32) -> u32 {
    let sa = src >> 24;
    if sa == 255 || dst == 0 {
        return src;
    }
    if sa == 0 {
        return dst;
    }
    let inv = 255 - sa;
    let ch = |shift: u32| ((src >> shift) & 0xff) + ((dst >> shift) & 0xff) * inv / 255;
    (ch(24) << 24) | (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

/// Straight-alpha `0xAARRGGBB` from a premultiplied pixel, for the blitter,
/// which blends straight colors by coverage. Fully transparent maps to zero.
pub(crate) fn straight_from_premultiplied(px: u32) -> u32 {
    let a = px >> 24;
    if a == 0 {
        return 0;
    }
    let un = |shift: u32| (((px >> shift) & 0xff) * 255 / a).min(255) << shift;
    (a << 24) | un(16) | un(8) | un(0)
}

/// Resample ARGB `src` (`src_w` x `src_h`) into a fresh `dst_w` x `dst_h`
/// buffer. A downscale uses an area (box) filter, so every source pixel
/// contributes its coverage-weighted share — a bilinear tap would read only the
/// four nearest texels and alias badly past a ~2x reduction (misshapen emoji
/// eyes, stair-stepped edges). Upscales and 1:1 use bilinear, which is exact at
/// 1:1 and smooth going up. A color emoji glyph is resampled to its display size
/// once and the result cached, so this runs per (glyph, size), not per frame.
/// Returns an empty buffer for degenerate or malformed input.
pub(crate) fn resample_argb(
    src: &[u32],
    src_w: u32,
    src_h: u32,
    dst_w: i32,
    dst_h: i32,
) -> Vec<u32> {
    let (sw, sh) = (src_w as i32, src_h as i32);
    if dst_w <= 0 || dst_h <= 0 || sw <= 0 || sh <= 0 || src.len() < (sw * sh) as usize {
        return Vec::new();
    }
    let mut out = Vec::with_capacity((dst_w * dst_h) as usize);
    if dst_w < sw || dst_h < sh {
        // Downscale: average each destination pixel's source footprint.
        let (fx, fy) = (sw as f32 / dst_w as f32, sh as f32 / dst_h as f32);
        for dy in 0..dst_h {
            let (y0, y1) = (dy as f32 * fy, (dy + 1) as f32 * fy);
            for dx in 0..dst_w {
                let (x0, x1) = (dx as f32 * fx, (dx + 1) as f32 * fx);
                out.push(sample_area(src, sw, sh, x0, x1, y0, y1));
            }
        }
        return out;
    }
    for dy in 0..dst_h {
        // Map each destination pixel centre back into source space; the -0.5
        // offsets sample at source pixel centres, so a 1:1 copy is exact.
        let sy = (dy as f32 + 0.5) * sh as f32 / dst_h as f32 - 0.5;
        for dx in 0..dst_w {
            let sx = (dx as f32 + 0.5) * sw as f32 / dst_w as f32 - 0.5;
            out.push(sample_bilinear(src, sw, sh, sx, sy));
        }
    }
    out
}

/// Average the ARGB pixels of `src` (`w` x `h`, row-major) over the fractional
/// box `[x0, x1) x [y0, y1)`, weighting edge rows and columns by how much of
/// them the box covers. Indices are clamped into the surface, so a box edge
/// that lands a rounding error past the last row or column stays in bounds.
/// Each channel, alpha included, is averaged independently; the caller keeps
/// color meaningful under a varying alpha by passing premultiplied pixels
/// where that matters (the emoji path does).
fn sample_area(src: &[u32], w: i32, h: i32, x0: f32, x1: f32, y0: f32, y1: f32) -> u32 {
    let mut acc = [0.0f32; 4];
    let mut total = 0.0f32;
    let mut y = y0.floor();
    while y < y1 {
        let weight_y = (y + 1.0).min(y1) - y.max(y0);
        let row = (y as i32).clamp(0, h - 1) * w;
        let mut x = x0.floor();
        while x < x1 {
            let weight = weight_y * ((x + 1.0).min(x1) - x.max(x0));
            let px = src[(row + (x as i32).clamp(0, w - 1)) as usize];
            for (i, a) in acc.iter_mut().enumerate() {
                *a += ((px >> (i * 8)) & 0xff) as f32 * weight;
            }
            total += weight;
            x += 1.0;
        }
        y += 1.0;
    }
    if total <= 0.0 {
        return 0;
    }
    let mut out = 0u32;
    for (i, a) in acc.iter().enumerate() {
        out |= ((a / total).round().clamp(0.0, 255.0) as u32) << (i * 8);
    }
    out
}

/// Bilinearly sample the ARGB pixels `src` (`w` x `h`, row-major) at fractional
/// `(fx, fy)`, clamping to the edges. Each channel, alpha included, is
/// interpolated independently across the four nearest texels. Used for upscales
/// and 1:1 only; downscales go through [`sample_area`], which averages instead
/// of skipping source pixels.
fn sample_bilinear(src: &[u32], w: i32, h: i32, fx: f32, fy: f32) -> u32 {
    let (x0f, y0f) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0f, fy - y0f);
    let (xi, yi) = (x0f as i32, y0f as i32);
    // Clamp each corner independently, so a coordinate past an edge folds both
    // texels onto the edge pixel (no wrap, no out-of-range index).
    let x0 = xi.clamp(0, w - 1);
    let x1 = (xi + 1).clamp(0, w - 1);
    let y0 = yi.clamp(0, h - 1);
    let y1 = (yi + 1).clamp(0, h - 1);
    let at = |x: i32, y: i32| src[(y * w + x) as usize];
    let (c00, c10, c01, c11) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    let mut out = 0u32;
    for shift in [0, 8, 16, 24] {
        let ch = |c: u32| ((c >> shift) & 0xff) as f32;
        let top = ch(c00) + (ch(c10) - ch(c00)) * tx;
        let bot = ch(c01) + (ch(c11) - ch(c01)) * tx;
        let v = (top + (bot - top) * ty).round().clamp(0.0, 255.0) as u32;
        out |= v << shift;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn premultiplied_edges_resample_without_dark_fringe() {
        // Half yellow, half transparent, averaged to one pixel in premultiplied
        // space and converted to straight alpha afterwards: the color must stay
        // pure yellow at half coverage. Filtering straight alpha instead would
        // mix the transparent texel's black into the color and darken the edge,
        // which is exactly the fringe this pipeline order exists to avoid.
        let src = [0xff_ff_ff_00, 0x0000_0000];
        let avg = resample_argb(&src, 2, 1, 1, 1)[0];
        let straight = straight_from_premultiplied(avg);
        assert_eq!(straight >> 24, 0x80, "half coverage");
        assert_eq!(straight & 0x00ff_ffff, 0x00ff_ff00, "still pure yellow");
    }

    #[test]
    fn resample_upscales_a_single_pixel_to_a_solid_block() {
        // 1x1 red, scaled to 3x3, is nine reds (the cache stores this, blitted later).
        let scaled = resample_argb(&[0xff_ff_00_00], 1, 1, 3, 3);
        assert_eq!(scaled.len(), 9);
        assert!(scaled.iter().all(|&p| p == 0xff_ff_00_00), "all red");
    }

    #[test]
    fn resample_downscale_averages_rather_than_skips() {
        // 4x1 [white, black, black, black] to 1x1: the area filter yields the
        // 25% average; a bilinear tap between the middle pixels would give
        // plain black, silently dropping the white pixel.
        let src = [0xff_ff_ff_ff, 0xff_00_00_00, 0xff_00_00_00, 0xff_00_00_00];
        let out = resample_argb(&src, 4, 1, 1, 1);
        let gray = out[0] & 0xff;
        assert!(
            (60..=68).contains(&gray),
            "expected the 25% average, got {:08x}",
            out[0]
        );
    }

    #[test]
    fn resample_downscale_keeps_every_source_pixel_contributing() {
        // A lone red corner pixel in an otherwise black 3x3 must tint the 1x1
        // result; a center-only sample would drop it entirely.
        let mut src = [0xff_00_00_00; 9];
        src[0] = 0xff_ff_00_00;
        let out = resample_argb(&src, 3, 3, 1, 1);
        let red = (out[0] >> 16) & 0xff;
        assert!(
            red > 0,
            "the corner pixel must contribute, got {:08x}",
            out[0]
        );
    }

    #[test]
    fn resample_is_exact_at_one_to_one() {
        let src = [0xff_11_22_33, 0xff_44_55_66, 0xff_77_88_99, 0xff_aa_bb_cc];
        let out = resample_argb(&src, 2, 2, 2, 2);
        assert_eq!(out, src, "a 1:1 resample copies the pixels exactly");
    }

    #[test]
    fn resample_rejects_malformed_input() {
        // Claims 2x2 but carries one pixel: empty result rather than reading OOB.
        assert!(resample_argb(&[0xff_ff_00_00], 2, 2, 4, 4).is_empty());
        assert!(
            resample_argb(&[0xff_ff_00_00], 1, 1, 0, 4).is_empty(),
            "zero dst"
        );
    }

    #[test]
    fn premultiplied_over_keeps_translucent_sources_undarkened() {
        let half_yellow = 0x80_80_80_00; // premultiplied: a=128, r=g=128, b=0
        assert_eq!(
            premultiplied_over(half_yellow, 0),
            half_yellow,
            "compositing onto transparent is a plain copy"
        );
        let opaque = 0xff_10_20_30;
        assert_eq!(premultiplied_over(opaque, half_yellow), opaque);
        assert_eq!(premultiplied_over(0, half_yellow), half_yellow);
        // A half-alpha source over an opaque background accumulates both.
        let out = premultiplied_over(half_yellow, 0xff_00_00_ff);
        assert_eq!(out >> 24, 0xff, "alpha saturates over an opaque ground");
        assert_eq!((out >> 16) & 0xff, 0x80, "red comes from the source");
        assert_eq!(out & 0xff, 0x7f, "blue is the ground's remainder");
    }

    #[test]
    fn straight_from_premultiplied_recovers_color_and_maps_clear_to_zero() {
        assert_eq!(straight_from_premultiplied(0), 0);
        assert_eq!(straight_from_premultiplied(0x80_80_00_40), 0x80_ff_00_7f);
        let opaque = 0xff_12_34_56;
        assert_eq!(
            straight_from_premultiplied(opaque),
            opaque,
            "opaque is identity"
        );
    }
}
