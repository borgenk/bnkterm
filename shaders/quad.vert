// The one vertex shader: every display-list primitive is an axis-aligned quad
// in surface pixels, mapped to clip space by the viewport push constant.
// Attribute meanings per mode are documented in src/gpu.rs (the batcher).
#version 450

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec4 in_extra;
layout(location = 4) in uint in_mode;

layout(push_constant) uniform Push {
    vec2 viewport;
} push;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;
layout(location = 2) out vec4 v_extra;
layout(location = 3) flat out uint v_mode;

void main() {
    // Pixel coordinates (origin top-left) to Vulkan NDC (y down): 0 maps to
    // -1, the full extent to +1.
    gl_Position = vec4(in_pos / push.viewport * 2.0 - 1.0, 0.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
    v_extra = in_extra;
    v_mode = in_mode;
}
