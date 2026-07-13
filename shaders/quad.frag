// The one fragment shader. Modes (src/render/gpu.rs): 0 solid fill, 1 glyph
// coverage from the R8 atlas, 2 emoji from the BGRA atlas, 4 rounded-rect
// coverage. (Mode 3, the decoded-image texture, is excised: a terminal draws no
// images.) Glyph and emoji reads are texelFetch at integer coords so a 1:1 atlas
// quad reproduces its source texels exactly.
#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec4 v_extra;
layout(location = 3) flat in uint v_mode;

layout(set = 0, binding = 0) uniform sampler2D glyph_atlas;
layout(set = 0, binding = 1) uniform sampler2D emoji_atlas;

layout(location = 0) out vec4 out_color;

// Shared with the vertex stage, which is the only stage that reads it.
layout(push_constant) uniform Push {
    vec2 viewport;
} push;

// sRGB to linear. Vertex colours and the emoji atlas carry sRGB-encoded bytes
// and are decoded here. The colour attachment is an sRGB view, so blending runs
// in linear space and the hardware re-encodes on store, leaving opaque fills
// byte-exact while anti-aliased edges blend gamma-correct.
vec3 srgb_to_linear(vec3 c) {
    bvec3 hi = greaterThan(c, vec3(0.04045));
    vec3 lo = c / 12.92;
    vec3 hs = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hs, hi);
}

// A text run may ramp its ink away across a span of screen x (v_extra.yz, from
// full ink to none; see display.rs Fade) so a truncated tab label dissolves in
// place of an ellipsis. Scaling alpha, rather than mixing the colour toward the
// background, is what lets the tail reach *exactly* the background on any tab
// colour, and it leaves v_extra.x (the contrast the coverage gamma is weighted
// against) intact, so a label's stroke weight does not drift along its own tail.
//
// smoothstep leaves the ramp flat at both ends, so the fade neither begins on a
// visible seam nor stops short of zero. An empty span (z <= y) means "no fade".
float ink_ramp() {
    if (v_extra.z <= v_extra.y) {
        return 1.0;
    }
    return 1.0 - smoothstep(v_extra.y, v_extra.z, gl_FragCoord.x);
}

// Replicates render.rs corner_coverage: full coverage everywhere except
// inside an active corner cell, where the pixel centre's distance to the
// corner circle fades coverage across the one-pixel band at the radius.
// v_uv already sits at the fragment (pixel) centre, the CPU's dx + 0.5.
float round_coverage(vec2 p, vec2 size, float r, uint mask) {
    bool top = (mask & 1u) != 0u;
    bool bottom = (mask & 2u) != 0u;
    vec2 c;
    if (top && p.x < r && p.y < r) {
        c = vec2(r, r);
    } else if (top && p.x >= size.x - r && p.y < r) {
        c = vec2(size.x - r, r);
    } else if (bottom && p.x < r && p.y >= size.y - r) {
        c = vec2(r, size.y - r);
    } else if (bottom && p.x >= size.x - r && p.y >= size.y - r) {
        c = vec2(size.x - r, size.y - r);
    } else {
        return 1.0;
    }
    return clamp(r - length(p - c) + 0.5, 0.0, 1.0);
}

void main() {
    if (v_mode == 0u) {
        out_color = vec4(srgb_to_linear(v_color.rgb), v_color.a);
    } else if (v_mode == 1u) {
        // v_extra.x is the run's coverage exponent, computed once per run on the CPU
        // (gpu.rs coverage_exponent) rather than per pixel here: linear-light
        // compositing renders light-on-dark text too heavy and dark-on-light text too
        // washed out, so a power > 1 thins the anti-aliased edges and a power < 1
        // thickens them. Solid pixels (coverage 1) are unchanged either way, so
        // colour fills keep full strength.
        float cov = pow(texelFetch(glyph_atlas, ivec2(v_uv), 0).r, v_extra.x);
        out_color = vec4(srgb_to_linear(v_color.rgb), v_color.a * cov * ink_ramp());
    } else if (v_mode == 2u) {
        vec4 e = texelFetch(emoji_atlas, ivec2(v_uv), 0);
        out_color = vec4(srgb_to_linear(e.rgb), e.a * ink_ramp());
    } else {
        float cov = round_coverage(v_uv, v_extra.xy, v_extra.z, uint(v_extra.w));
        out_color = vec4(srgb_to_linear(v_color.rgb), v_color.a * cov);
    }
}
