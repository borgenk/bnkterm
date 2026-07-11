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

// Shared with the vertex stage (which reads viewport). The fragment stage reads
// only the glyph mask gamma; both members sit at the offsets the CPU packs.
layout(push_constant) uniform Push {
    vec2 viewport;
    float glyph_coverage_gamma;
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
        // Mask gamma thins the anti-aliased edges: linear-light compositing is
        // physically correct but renders light-on-dark text heavier than the
        // gamma-space stacks beside it, so raising coverage to a power > 1 lowers
        // partial-coverage alpha back to a matching weight. v_extra.x steers that
        // power per run by contrast direction (gpu.rs contrast_factor): the exponent
        // is G^factor, so factor 1 keeps the tuned light-on-dark thinning, while dark
        // text on a light background (factor < 1, down to G^-1) is thickened back
        // from the washout the same compositing gives it. Solid pixels (coverage 1)
        // are unchanged, so colour fills keep full strength.
        float exponent = pow(push.glyph_coverage_gamma, v_extra.x);
        float cov = pow(texelFetch(glyph_atlas, ivec2(v_uv), 0).r, exponent);
        out_color = vec4(srgb_to_linear(v_color.rgb), v_color.a * cov);
    } else if (v_mode == 2u) {
        vec4 e = texelFetch(emoji_atlas, ivec2(v_uv), 0);
        out_color = vec4(srgb_to_linear(e.rgb), e.a);
    } else {
        float cov = round_coverage(v_uv, v_extra.xy, v_extra.z, uint(v_extra.w));
        out_color = vec4(srgb_to_linear(v_color.rgb), v_color.a * cov);
    }
}
