use core::ffi::{c_char, c_void, CStr};

use crate::platform::error::{Error, Result};

// ---------------------------------------------------------------------------
// Handles, scalars, and constants.
// ---------------------------------------------------------------------------

/// Dispatchable handles are pointers.
pub(super) type VkInstance = *mut c_void;
pub(super) type VkPhysicalDevice = *mut c_void;
pub(super) type VkDevice = *mut c_void;
pub(super) type VkQueue = *mut c_void;
pub(super) type VkCommandBuffer = *mut c_void;

/// Non-dispatchable handles are 64-bit opaque values.
pub(super) type VkImage = u64;
pub(super) type VkDeviceMemory = u64;
pub(super) type VkSemaphore = u64;
pub(super) type VkFence = u64;
pub(super) type VkCommandPool = u64;
pub(super) type VkBuffer = u64;
pub(super) type VkImageView = u64;
pub(super) type VkFramebuffer = u64;
pub(super) type VkRenderPass = u64;
pub(super) type VkPipeline = u64;
pub(super) type VkPipelineLayout = u64;
pub(super) type VkShaderModule = u64;
pub(super) type VkDescriptorSetLayout = u64;
pub(super) type VkDescriptorPool = u64;
pub(super) type VkDescriptorSet = u64;
pub(super) type VkSampler = u64;

pub(super) type VkResult = i32;
pub(super) const VK_SUCCESS: VkResult = 0;
/// The enumerate-style calls return this when the caller's array was smaller
/// than the full set; the two-call pattern below retries.
pub(super) const VK_INCOMPLETE: VkResult = 5;

/// `VK_API_VERSION_1_2`: high enough that every dependency of the extensions
/// this backend enables is core (bind_memory2, image_format_list, external
/// memory/semaphore capabilities), low enough for any 2020+ driver.
pub(super) const API_VERSION_1_2: u32 = (1 << 22) | (2 << 12);

pub(super) const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
pub(super) const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
pub(super) const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
pub(super) const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
pub(super) const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
pub(super) const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
pub(super) const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
pub(super) const VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO: u32 = 9;
pub(super) const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
pub(super) const VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO: u32 = 14;
pub(super) const VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO: u32 = 15;
pub(super) const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
pub(super) const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
pub(super) const VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO: u32 = 19;
pub(super) const VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO: u32 = 20;
pub(super) const VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO: u32 = 22;
pub(super) const VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO: u32 = 23;
pub(super) const VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO: u32 = 24;
pub(super) const VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO: u32 = 26;
pub(super) const VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO: u32 = 27;
pub(super) const VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO: u32 = 28;
pub(super) const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
pub(super) const VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO: u32 = 31;
pub(super) const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
pub(super) const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
pub(super) const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
pub(super) const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
pub(super) const VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO: u32 = 37;
pub(super) const VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO: u32 = 38;
pub(super) const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
pub(super) const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
pub(super) const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
pub(super) const VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO: u32 = 43;
pub(super) const VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER: u32 = 45;
pub(super) const VK_STRUCTURE_TYPE_MEMORY_BARRIER: u32 = 46;
pub(super) const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2: u32 = 1000059001;
pub(super) const VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2: u32 = 1000059002;
pub(super) const VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO: u32 = 1000072001;
pub(super) const VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO: u32 = 1000072002;
pub(super) const VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR: u32 = 1000074002;
pub(super) const VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO: u32 = 1000077000;
pub(super) const VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR: u32 = 1000079000;
pub(super) const VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR: u32 = 1000079001;
pub(super) const VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO: u32 = 1000127001;
pub(super) const VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO: u32 = 1000147000;
pub(super) const VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT: u32 = 1000158000;
pub(super) const VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT: u32 = 1000158003;
pub(super) const VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT: u32 = 1000158005;
pub(super) const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT: u32 = 1000353000;

/// `VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT`.
pub(super) const VK_QUEUE_GRAPHICS_BIT: u32 = 0x1;

/// `VK_FORMAT_B8G8R8A8_UNORM`: bytes B,G,R,A in memory, the Vulkan twin of
/// little-endian DRM XRGB8888 (and of the software canvas's pixel layout).
pub(super) const VK_FORMAT_B8G8R8A8_UNORM: u32 = 44;
/// `VK_FORMAT_B8G8R8A8_SRGB`: the same bytes as the UNORM twin, but sampled and
/// stored through the sRGB transfer function. The presentable image is created
/// mutable and rendered through a view in this format, so blending runs in
/// linear space while the exported bytes stay UNORM/XRGB8888 for the compositor.
pub(super) const VK_FORMAT_B8G8R8A8_SRGB: u32 = 50;
/// `VK_FORMAT_R8_UNORM`: the glyph coverage atlas.
pub(super) const VK_FORMAT_R8_UNORM: u32 = 9;
/// Vertex attribute formats.
pub(super) const VK_FORMAT_R32_UINT: u32 = 98;
pub(super) const VK_FORMAT_R32G32_SFLOAT: u32 = 103;
pub(super) const VK_FORMAT_R32G32B32A32_SFLOAT: u32 = 109;

/// `VkImageTiling::VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT`.
pub(super) const VK_IMAGE_TILING_DRM_FORMAT_MODIFIER: u32 = 1000158000;
pub(super) const VK_IMAGE_TILING_OPTIMAL: u32 = 0;
pub(super) const VK_IMAGE_TYPE_2D: u32 = 1;
pub(super) const VK_IMAGE_VIEW_TYPE_2D: u32 = 1;
pub(super) const VK_SAMPLE_COUNT_1_BIT: u32 = 1;
pub(super) const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;

/// `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT`: lets the presentable image carry both
/// its UNORM store format and an sRGB view for gamma-correct blending.
pub(super) const VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT: u32 = 0x8;

pub(super) const VK_IMAGE_USAGE_TRANSFER_SRC_BIT: u32 = 0x1;
pub(super) const VK_IMAGE_USAGE_TRANSFER_DST_BIT: u32 = 0x2;
pub(super) const VK_IMAGE_USAGE_SAMPLED_BIT: u32 = 0x4;
pub(super) const VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT: u32 = 0x10;
/// The format features a presentable image's modifier must support: rendering
/// plus both transfer directions (the screenshot readback uses the source
/// direction; clears use the destination).
pub(super) const VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT: u32 = 0x80;
pub(super) const VK_FORMAT_FEATURE_TRANSFER_SRC_BIT: u32 = 0x4000;
pub(super) const VK_FORMAT_FEATURE_TRANSFER_DST_BIT: u32 = 0x8000;

pub(super) const VK_IMAGE_LAYOUT_UNDEFINED: u32 = 0;
/// The layout the image is handed to the compositor in: the only layout
/// defined for foreign (non-Vulkan) access.
pub(super) const VK_IMAGE_LAYOUT_GENERAL: u32 = 1;
pub(super) const VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: u32 = 2;
pub(super) const VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: u32 = 5;
pub(super) const VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: u32 = 6;
pub(super) const VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: u32 = 7;

pub(super) const VK_ACCESS_SHADER_READ_BIT: u32 = 0x20;
pub(super) const VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT: u32 = 0x100;
pub(super) const VK_ACCESS_TRANSFER_READ_BIT: u32 = 0x800;
pub(super) const VK_ACCESS_TRANSFER_WRITE_BIT: u32 = 0x1000;
pub(super) const VK_ACCESS_HOST_READ_BIT: u32 = 0x2000;
pub(super) const VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT: u32 = 0x1;
pub(super) const VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT: u32 = 0x80;
pub(super) const VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT: u32 = 0x400;
pub(super) const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x1000;
pub(super) const VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT: u32 = 0x2000;
pub(super) const VK_PIPELINE_STAGE_HOST_BIT: u32 = 0x4000;

pub(super) const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x1;
pub(super) const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x2;
pub(super) const VK_BUFFER_USAGE_VERTEX_BUFFER_BIT: u32 = 0x80;
pub(super) const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x2;
pub(super) const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x4;

pub(super) const VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: u32 = 1;
pub(super) const VK_SHADER_STAGE_VERTEX_BIT: u32 = 0x1;
pub(super) const VK_SHADER_STAGE_FRAGMENT_BIT: u32 = 0x10;
pub(super) const VK_PIPELINE_BIND_POINT_GRAPHICS: u32 = 0;
pub(super) const VK_ATTACHMENT_LOAD_OP_CLEAR: u32 = 1;
pub(super) const VK_ATTACHMENT_LOAD_OP_DONT_CARE: u32 = 2;
pub(super) const VK_ATTACHMENT_STORE_OP_STORE: u32 = 0;
pub(super) const VK_ATTACHMENT_STORE_OP_DONT_CARE: u32 = 1;
pub(super) const VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: u32 = 3;
pub(super) const VK_POLYGON_MODE_FILL: u32 = 0;
pub(super) const VK_DYNAMIC_STATE_VIEWPORT: u32 = 0;
pub(super) const VK_DYNAMIC_STATE_SCISSOR: u32 = 1;
pub(super) const VK_BLEND_FACTOR_ZERO: u32 = 0;
pub(super) const VK_BLEND_FACTOR_ONE: u32 = 1;
pub(super) const VK_BLEND_FACTOR_SRC_ALPHA: u32 = 6;
pub(super) const VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: u32 = 7;
pub(super) const VK_BLEND_OP_ADD: u32 = 0;
pub(super) const VK_COLOR_COMPONENT_RGBA: u32 = 0xf;
pub(super) const VK_FILTER_LINEAR: u32 = 1;
pub(super) const VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: u32 = 2;
pub(super) const VK_VERTEX_INPUT_RATE_VERTEX: u32 = 0;
pub(super) const VK_SUBPASS_CONTENTS_INLINE: u32 = 0;
pub(super) const VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT: u32 = 0x1;
/// Room for the atlas set, the dummy set, and a generous number of distinct
/// visible images; a notes document with more is out of scope.
pub(super) const DESCRIPTOR_MAX_SETS: u32 = 256;

pub(super) const VK_IMAGE_ASPECT_COLOR_BIT: u32 = 0x1;
/// The dmabuf plane aspect for layout queries under modifier tiling.
pub(super) const VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT: u32 = 0x80;

pub(super) const VK_QUEUE_FAMILY_IGNORED: u32 = u32::MAX;
/// `VK_QUEUE_FAMILY_FOREIGN_EXT`: the compositor's side of the ownership
/// transfer barriers around each frame.
pub(super) const VK_QUEUE_FAMILY_FOREIGN: u32 = u32::MAX - 2;

pub(super) const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x1;
pub(super) const VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT: u32 = 0x200;
pub(super) const VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT: u32 = 0x10;
/// `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT`: a sync-fd import replaces the payload
/// for one wait, then the semaphore reverts, exactly the per-frame acquire.
pub(super) const VK_SEMAPHORE_IMPORT_TEMPORARY_BIT: u32 = 0x1;

pub(super) const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x2;
pub(super) const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
pub(super) const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x1;

/// Every fence wait in this backend bounds at one second: a GPU that takes
/// longer to clear or paint a window is broken, and a clean error (falling
/// back to software) beats a frozen editor.
pub(super) const FENCE_TIMEOUT_NS: u64 = 1_000_000_000;

/// `DRM_FORMAT_MOD_INVALID`: "the layout is implicit"; never allocate with it.
pub(super) const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

pub(super) const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
pub(super) const VK_UUID_SIZE: usize = 16;
pub(super) const VK_MAX_EXTENSION_NAME_SIZE: usize = 256;

/// The device extensions the dmabuf presentation path enables: image layouts
/// by DRM modifier, dmabuf memory export, fd-based semaphore export/import for
/// the implicit-sync bridge, and foreign-queue ownership transfer so the
/// compositor's reads are well-defined.
pub(super) const REQUIRED_DEVICE_EXTENSIONS: [&CStr; 5] = [
    c"VK_EXT_image_drm_format_modifier",
    c"VK_EXT_external_memory_dma_buf",
    c"VK_KHR_external_memory_fd",
    c"VK_KHR_external_semaphore_fd",
    c"VK_EXT_queue_family_foreign",
];
/// Needed only to *identify* the device (its DRM dev_t); physical-device-level
/// properties may be queried when supported without enabling anything.
pub(super) const EXT_PHYSICAL_DEVICE_DRM: &CStr = c"VK_EXT_physical_device_drm";

// ---------------------------------------------------------------------------
// Struct layouts (x86_64/arm64 C ABI).
// ---------------------------------------------------------------------------

#[repr(C)]
pub(super) struct VkApplicationInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) p_application_name: *const c_char,
    pub(super) application_version: u32,
    pub(super) p_engine_name: *const c_char,
    pub(super) engine_version: u32,
    pub(super) api_version: u32,
}

#[repr(C)]
pub(super) struct VkInstanceCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) p_application_info: *const VkApplicationInfo,
    pub(super) enabled_layer_count: u32,
    pub(super) pp_enabled_layer_names: *const *const c_char,
    pub(super) enabled_extension_count: u32,
    pub(super) pp_enabled_extension_names: *const *const c_char,
}

#[repr(C)]
pub(super) struct VkDeviceQueueCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) queue_family_index: u32,
    pub(super) queue_count: u32,
    pub(super) p_queue_priorities: *const f32,
}

#[repr(C)]
pub(super) struct VkDeviceCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) queue_create_info_count: u32,
    pub(super) p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    pub(super) enabled_layer_count: u32,
    pub(super) pp_enabled_layer_names: *const *const c_char,
    pub(super) enabled_extension_count: u32,
    pub(super) pp_enabled_extension_names: *const *const c_char,
    pub(super) p_enabled_features: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkExtensionProperties {
    pub(super) extension_name: [c_char; VK_MAX_EXTENSION_NAME_SIZE],
    pub(super) spec_version: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkExtent3D {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkQueueFamilyProperties {
    pub(super) queue_flags: u32,
    pub(super) queue_count: u32,
    pub(super) timestamp_valid_bits: u32,
    pub(super) min_image_transfer_granularity: VkExtent3D,
}

/// `VkPhysicalDeviceLimits`, transcribed in full: the properties struct embeds
/// it by value, so its exact size is load-bearing for everything after it on
/// the stack. The size assertion in the tests guards the transcription.
#[repr(C)]
pub(super) struct VkPhysicalDeviceLimits {
    pub(super) max_image_dimension_1d: u32,
    pub(super) max_image_dimension_2d: u32,
    pub(super) max_image_dimension_3d: u32,
    pub(super) max_image_dimension_cube: u32,
    pub(super) max_image_array_layers: u32,
    pub(super) max_texel_buffer_elements: u32,
    pub(super) max_uniform_buffer_range: u32,
    pub(super) max_storage_buffer_range: u32,
    pub(super) max_push_constants_size: u32,
    pub(super) max_memory_allocation_count: u32,
    pub(super) max_sampler_allocation_count: u32,
    pub(super) buffer_image_granularity: u64,
    pub(super) sparse_address_space_size: u64,
    pub(super) max_bound_descriptor_sets: u32,
    pub(super) max_per_stage_descriptor_samplers: u32,
    pub(super) max_per_stage_descriptor_uniform_buffers: u32,
    pub(super) max_per_stage_descriptor_storage_buffers: u32,
    pub(super) max_per_stage_descriptor_sampled_images: u32,
    pub(super) max_per_stage_descriptor_storage_images: u32,
    pub(super) max_per_stage_descriptor_input_attachments: u32,
    pub(super) max_per_stage_resources: u32,
    pub(super) max_descriptor_set_samplers: u32,
    pub(super) max_descriptor_set_uniform_buffers: u32,
    pub(super) max_descriptor_set_uniform_buffers_dynamic: u32,
    pub(super) max_descriptor_set_storage_buffers: u32,
    pub(super) max_descriptor_set_storage_buffers_dynamic: u32,
    pub(super) max_descriptor_set_sampled_images: u32,
    pub(super) max_descriptor_set_storage_images: u32,
    pub(super) max_descriptor_set_input_attachments: u32,
    pub(super) max_vertex_input_attributes: u32,
    pub(super) max_vertex_input_bindings: u32,
    pub(super) max_vertex_input_attribute_offset: u32,
    pub(super) max_vertex_input_binding_stride: u32,
    pub(super) max_vertex_output_components: u32,
    pub(super) max_tessellation_generation_level: u32,
    pub(super) max_tessellation_patch_size: u32,
    pub(super) max_tessellation_control_per_vertex_input_components: u32,
    pub(super) max_tessellation_control_per_vertex_output_components: u32,
    pub(super) max_tessellation_control_per_patch_output_components: u32,
    pub(super) max_tessellation_control_total_output_components: u32,
    pub(super) max_tessellation_evaluation_input_components: u32,
    pub(super) max_tessellation_evaluation_output_components: u32,
    pub(super) max_geometry_shader_invocations: u32,
    pub(super) max_geometry_input_components: u32,
    pub(super) max_geometry_output_components: u32,
    pub(super) max_geometry_output_vertices: u32,
    pub(super) max_geometry_total_output_components: u32,
    pub(super) max_fragment_input_components: u32,
    pub(super) max_fragment_output_attachments: u32,
    pub(super) max_fragment_dual_src_attachments: u32,
    pub(super) max_fragment_combined_output_resources: u32,
    pub(super) max_compute_shared_memory_size: u32,
    pub(super) max_compute_work_group_count: [u32; 3],
    pub(super) max_compute_work_group_invocations: u32,
    pub(super) max_compute_work_group_size: [u32; 3],
    pub(super) sub_pixel_precision_bits: u32,
    pub(super) sub_texel_precision_bits: u32,
    pub(super) mipmap_precision_bits: u32,
    pub(super) max_draw_indexed_index_value: u32,
    pub(super) max_draw_indirect_count: u32,
    pub(super) max_sampler_lod_bias: f32,
    pub(super) max_sampler_anisotropy: f32,
    pub(super) max_viewports: u32,
    pub(super) max_viewport_dimensions: [u32; 2],
    pub(super) viewport_bounds_range: [f32; 2],
    pub(super) viewport_sub_pixel_bits: u32,
    pub(super) min_memory_map_alignment: usize,
    pub(super) min_texel_buffer_offset_alignment: u64,
    pub(super) min_uniform_buffer_offset_alignment: u64,
    pub(super) min_storage_buffer_offset_alignment: u64,
    pub(super) min_texel_offset: i32,
    pub(super) max_texel_offset: u32,
    pub(super) min_texel_gather_offset: i32,
    pub(super) max_texel_gather_offset: u32,
    pub(super) min_interpolation_offset: f32,
    pub(super) max_interpolation_offset: f32,
    pub(super) sub_pixel_interpolation_offset_bits: u32,
    pub(super) max_framebuffer_width: u32,
    pub(super) max_framebuffer_height: u32,
    pub(super) max_framebuffer_layers: u32,
    pub(super) framebuffer_color_sample_counts: u32,
    pub(super) framebuffer_depth_sample_counts: u32,
    pub(super) framebuffer_stencil_sample_counts: u32,
    pub(super) framebuffer_no_attachments_sample_counts: u32,
    pub(super) max_color_attachments: u32,
    pub(super) sampled_image_color_sample_counts: u32,
    pub(super) sampled_image_integer_sample_counts: u32,
    pub(super) sampled_image_depth_sample_counts: u32,
    pub(super) sampled_image_stencil_sample_counts: u32,
    pub(super) storage_image_sample_counts: u32,
    pub(super) max_sample_mask_words: u32,
    pub(super) timestamp_compute_and_graphics: u32,
    pub(super) timestamp_period: f32,
    pub(super) max_clip_distances: u32,
    pub(super) max_cull_distances: u32,
    pub(super) max_combined_clip_and_cull_distances: u32,
    pub(super) discrete_queue_priorities: u32,
    pub(super) point_size_range: [f32; 2],
    pub(super) line_width_range: [f32; 2],
    pub(super) point_size_granularity: f32,
    pub(super) line_width_granularity: f32,
    pub(super) strict_lines: u32,
    pub(super) standard_sample_locations: u32,
    pub(super) optimal_buffer_copy_offset_alignment: u64,
    pub(super) optimal_buffer_copy_row_pitch_alignment: u64,
    pub(super) non_coherent_atom_size: u64,
}

#[repr(C)]
pub(super) struct VkPhysicalDeviceSparseProperties {
    pub(super) residency_standard_2d_block_shape: u32,
    pub(super) residency_standard_2d_multisample_block_shape: u32,
    pub(super) residency_standard_3d_block_shape: u32,
    pub(super) residency_aligned_mip_size: u32,
    pub(super) residency_non_resident_strict: u32,
}

#[repr(C)]
pub(super) struct VkPhysicalDeviceProperties {
    pub(super) api_version: u32,
    pub(super) driver_version: u32,
    pub(super) vendor_id: u32,
    pub(super) device_id: u32,
    pub(super) device_type: u32,
    pub(super) device_name: [c_char; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE],
    pub(super) pipeline_cache_uuid: [u8; VK_UUID_SIZE],
    pub(super) limits: VkPhysicalDeviceLimits,
    pub(super) sparse_properties: VkPhysicalDeviceSparseProperties,
}

#[repr(C)]
pub(super) struct VkPhysicalDeviceProperties2 {
    pub(super) s_type: u32,
    pub(super) p_next: *mut c_void,
    pub(super) properties: VkPhysicalDeviceProperties,
}

/// `VK_EXT_physical_device_drm`: which DRM primary/render node a physical
/// device is, matched against the dev_t in the compositor's dmabuf feedback.
#[repr(C)]
pub(super) struct VkPhysicalDeviceDrmPropertiesEXT {
    pub(super) s_type: u32,
    pub(super) p_next: *mut c_void,
    pub(super) has_primary: u32,
    pub(super) has_render: u32,
    pub(super) primary_major: i64,
    pub(super) primary_minor: i64,
    pub(super) render_major: i64,
    pub(super) render_minor: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkMemoryType {
    pub(super) property_flags: u32,
    pub(super) heap_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkMemoryHeap {
    pub(super) size: u64,
    pub(super) flags: u32,
}

#[repr(C)]
pub(super) struct VkPhysicalDeviceMemoryProperties {
    pub(super) memory_type_count: u32,
    pub(super) memory_types: [VkMemoryType; 32],
    pub(super) memory_heap_count: u32,
    pub(super) memory_heaps: [VkMemoryHeap; 16],
}

#[repr(C)]
pub(super) struct VkFormatProperties {
    pub(super) linear_tiling_features: u32,
    pub(super) optimal_tiling_features: u32,
    pub(super) buffer_features: u32,
}

#[repr(C)]
pub(super) struct VkFormatProperties2 {
    pub(super) s_type: u32,
    pub(super) p_next: *mut c_void,
    pub(super) format_properties: VkFormatProperties,
}

/// One modifier the device supports for a format: its layout, how many memory
/// planes it uses, and what the format can do under it.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkDrmFormatModifierPropertiesEXT {
    pub(super) drm_format_modifier: u64,
    pub(super) drm_format_modifier_plane_count: u32,
    pub(super) drm_format_modifier_tiling_features: u32,
}

#[repr(C)]
pub(super) struct VkDrmFormatModifierPropertiesListEXT {
    pub(super) s_type: u32,
    pub(super) p_next: *mut c_void,
    pub(super) drm_format_modifier_count: u32,
    pub(super) p_drm_format_modifier_properties: *mut VkDrmFormatModifierPropertiesEXT,
}

#[repr(C)]
pub(super) struct VkImageCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) image_type: u32,
    pub(super) format: u32,
    pub(super) extent: VkExtent3D,
    pub(super) mip_levels: u32,
    pub(super) array_layers: u32,
    pub(super) samples: u32,
    pub(super) tiling: u32,
    pub(super) usage: u32,
    pub(super) sharing_mode: u32,
    pub(super) queue_family_index_count: u32,
    pub(super) p_queue_family_indices: *const u32,
    pub(super) initial_layout: u32,
}

#[repr(C)]
pub(super) struct VkExternalMemoryImageCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) handle_types: u32,
}

#[repr(C)]
pub(super) struct VkImageFormatListCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) view_format_count: u32,
    pub(super) p_view_formats: *const u32,
}

#[repr(C)]
pub(super) struct VkImageDrmFormatModifierListCreateInfoEXT {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) drm_format_modifier_count: u32,
    pub(super) p_drm_format_modifiers: *const u64,
}

#[repr(C)]
pub(super) struct VkImageDrmFormatModifierPropertiesEXT {
    pub(super) s_type: u32,
    pub(super) p_next: *mut c_void,
    pub(super) drm_format_modifier: u64,
}

#[repr(C)]
pub(super) struct VkMemoryRequirements {
    pub(super) size: u64,
    pub(super) alignment: u64,
    pub(super) memory_type_bits: u32,
}

#[repr(C)]
pub(super) struct VkMemoryAllocateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) allocation_size: u64,
    pub(super) memory_type_index: u32,
}

#[repr(C)]
pub(super) struct VkExportMemoryAllocateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) handle_types: u32,
}

/// Exportable images get a dedicated allocation: it is what external-memory
/// consumers expect, and it makes the exported fd describe exactly one image.
#[repr(C)]
pub(super) struct VkMemoryDedicatedAllocateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) image: VkImage,
    pub(super) buffer: u64,
}

#[repr(C)]
pub(super) struct VkMemoryGetFdInfoKHR {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) memory: VkDeviceMemory,
    pub(super) handle_type: u32,
}

#[repr(C)]
pub(super) struct VkImageSubresource {
    pub(super) aspect_mask: u32,
    pub(super) mip_level: u32,
    pub(super) array_layer: u32,
}

#[repr(C)]
pub(super) struct VkSubresourceLayout {
    pub(super) offset: u64,
    pub(super) size: u64,
    pub(super) row_pitch: u64,
    pub(super) array_pitch: u64,
    pub(super) depth_pitch: u64,
}

#[repr(C)]
pub(super) struct VkCommandPoolCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) queue_family_index: u32,
}

#[repr(C)]
pub(super) struct VkCommandBufferAllocateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) command_pool: VkCommandPool,
    pub(super) level: u32,
    pub(super) command_buffer_count: u32,
}

#[repr(C)]
pub(super) struct VkCommandBufferBeginInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) p_inheritance_info: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkImageSubresourceRange {
    pub(super) aspect_mask: u32,
    pub(super) base_mip_level: u32,
    pub(super) level_count: u32,
    pub(super) base_array_layer: u32,
    pub(super) layer_count: u32,
}

#[repr(C)]
pub(super) struct VkMemoryBarrier {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) src_access_mask: u32,
    pub(super) dst_access_mask: u32,
}

#[repr(C)]
pub(super) struct VkImageMemoryBarrier {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) src_access_mask: u32,
    pub(super) dst_access_mask: u32,
    pub(super) old_layout: u32,
    pub(super) new_layout: u32,
    pub(super) src_queue_family_index: u32,
    pub(super) dst_queue_family_index: u32,
    pub(super) image: VkImage,
    pub(super) subresource_range: VkImageSubresourceRange,
}

#[repr(C)]
pub(super) struct VkSubmitInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) wait_semaphore_count: u32,
    pub(super) p_wait_semaphores: *const VkSemaphore,
    pub(super) p_wait_dst_stage_mask: *const u32,
    pub(super) command_buffer_count: u32,
    pub(super) p_command_buffers: *const VkCommandBuffer,
    pub(super) signal_semaphore_count: u32,
    pub(super) p_signal_semaphores: *const VkSemaphore,
}

#[repr(C)]
pub(super) struct VkFenceCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
}

#[repr(C)]
pub(super) struct VkSemaphoreCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
}

#[repr(C)]
pub(super) struct VkExportSemaphoreCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) handle_types: u32,
}

#[repr(C)]
pub(super) struct VkSemaphoreGetFdInfoKHR {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) semaphore: VkSemaphore,
    pub(super) handle_type: u32,
}

#[repr(C)]
pub(super) struct VkImportSemaphoreFdInfoKHR {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) semaphore: VkSemaphore,
    pub(super) flags: u32,
    pub(super) handle_type: u32,
    pub(super) fd: i32,
}

/// `VkClearColorValue`, always used as float32 here.
#[repr(C)]
pub(super) struct VkClearColorValue {
    pub(super) float32: [f32; 4],
}

#[repr(C)]
pub(super) struct VkBufferCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) size: u64,
    pub(super) usage: u32,
    pub(super) sharing_mode: u32,
    pub(super) queue_family_index_count: u32,
    pub(super) p_queue_family_indices: *const u32,
}

#[repr(C)]
pub(super) struct VkComponentMapping {
    pub(super) r: u32,
    pub(super) g: u32,
    pub(super) b: u32,
    pub(super) a: u32,
}

#[repr(C)]
pub(super) struct VkImageViewCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) image: VkImage,
    pub(super) view_type: u32,
    pub(super) format: u32,
    pub(super) components: VkComponentMapping,
    pub(super) subresource_range: VkImageSubresourceRange,
}

#[repr(C)]
pub(super) struct VkShaderModuleCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) code_size: usize,
    pub(super) p_code: *const u32,
}

#[repr(C)]
pub(super) struct VkPipelineShaderStageCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) stage: u32,
    pub(super) module: VkShaderModule,
    pub(super) p_name: *const c_char,
    pub(super) p_specialization_info: *const c_void,
}

#[repr(C)]
pub(super) struct VkVertexInputBindingDescription {
    pub(super) binding: u32,
    pub(super) stride: u32,
    pub(super) input_rate: u32,
}

#[repr(C)]
pub(super) struct VkVertexInputAttributeDescription {
    pub(super) location: u32,
    pub(super) binding: u32,
    pub(super) format: u32,
    pub(super) offset: u32,
}

#[repr(C)]
pub(super) struct VkPipelineVertexInputStateCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) vertex_binding_description_count: u32,
    pub(super) p_vertex_binding_descriptions: *const VkVertexInputBindingDescription,
    pub(super) vertex_attribute_description_count: u32,
    pub(super) p_vertex_attribute_descriptions: *const VkVertexInputAttributeDescription,
}

#[repr(C)]
pub(super) struct VkPipelineInputAssemblyStateCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) topology: u32,
    pub(super) primitive_restart_enable: u32,
}

#[repr(C)]
pub(super) struct VkViewport {
    pub(super) x: f32,
    pub(super) y: f32,
    pub(super) width: f32,
    pub(super) height: f32,
    pub(super) min_depth: f32,
    pub(super) max_depth: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkOffset2D {
    pub(super) x: i32,
    pub(super) y: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkExtent2D {
    pub(super) width: u32,
    pub(super) height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct VkRect2D {
    pub(super) offset: VkOffset2D,
    pub(super) extent: VkExtent2D,
}

#[repr(C)]
pub(super) struct VkPipelineViewportStateCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) viewport_count: u32,
    pub(super) p_viewports: *const VkViewport,
    pub(super) scissor_count: u32,
    pub(super) p_scissors: *const VkRect2D,
}

#[repr(C)]
pub(super) struct VkPipelineRasterizationStateCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) depth_clamp_enable: u32,
    pub(super) rasterizer_discard_enable: u32,
    pub(super) polygon_mode: u32,
    pub(super) cull_mode: u32,
    pub(super) front_face: u32,
    pub(super) depth_bias_enable: u32,
    pub(super) depth_bias_constant_factor: f32,
    pub(super) depth_bias_clamp: f32,
    pub(super) depth_bias_slope_factor: f32,
    pub(super) line_width: f32,
}

#[repr(C)]
pub(super) struct VkPipelineMultisampleStateCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) rasterization_samples: u32,
    pub(super) sample_shading_enable: u32,
    pub(super) min_sample_shading: f32,
    pub(super) p_sample_mask: *const u32,
    pub(super) alpha_to_coverage_enable: u32,
    pub(super) alpha_to_one_enable: u32,
}

#[repr(C)]
pub(super) struct VkPipelineColorBlendAttachmentState {
    pub(super) blend_enable: u32,
    pub(super) src_color_blend_factor: u32,
    pub(super) dst_color_blend_factor: u32,
    pub(super) color_blend_op: u32,
    pub(super) src_alpha_blend_factor: u32,
    pub(super) dst_alpha_blend_factor: u32,
    pub(super) alpha_blend_op: u32,
    pub(super) color_write_mask: u32,
}

#[repr(C)]
pub(super) struct VkPipelineColorBlendStateCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) logic_op_enable: u32,
    pub(super) logic_op: u32,
    pub(super) attachment_count: u32,
    pub(super) p_attachments: *const VkPipelineColorBlendAttachmentState,
    pub(super) blend_constants: [f32; 4],
}

#[repr(C)]
pub(super) struct VkPipelineDynamicStateCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) dynamic_state_count: u32,
    pub(super) p_dynamic_states: *const u32,
}

#[repr(C)]
pub(super) struct VkPushConstantRange {
    pub(super) stage_flags: u32,
    pub(super) offset: u32,
    pub(super) size: u32,
}

#[repr(C)]
pub(super) struct VkPipelineLayoutCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) set_layout_count: u32,
    pub(super) p_set_layouts: *const VkDescriptorSetLayout,
    pub(super) push_constant_range_count: u32,
    pub(super) p_push_constant_ranges: *const VkPushConstantRange,
}

#[repr(C)]
pub(super) struct VkGraphicsPipelineCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) stage_count: u32,
    pub(super) p_stages: *const VkPipelineShaderStageCreateInfo,
    pub(super) p_vertex_input_state: *const VkPipelineVertexInputStateCreateInfo,
    pub(super) p_input_assembly_state: *const VkPipelineInputAssemblyStateCreateInfo,
    pub(super) p_tessellation_state: *const c_void,
    pub(super) p_viewport_state: *const VkPipelineViewportStateCreateInfo,
    pub(super) p_rasterization_state: *const VkPipelineRasterizationStateCreateInfo,
    pub(super) p_multisample_state: *const VkPipelineMultisampleStateCreateInfo,
    pub(super) p_depth_stencil_state: *const c_void,
    pub(super) p_color_blend_state: *const VkPipelineColorBlendStateCreateInfo,
    pub(super) p_dynamic_state: *const VkPipelineDynamicStateCreateInfo,
    pub(super) layout: VkPipelineLayout,
    pub(super) render_pass: VkRenderPass,
    pub(super) subpass: u32,
    pub(super) base_pipeline_handle: VkPipeline,
    pub(super) base_pipeline_index: i32,
}

#[repr(C)]
pub(super) struct VkSamplerCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) mag_filter: u32,
    pub(super) min_filter: u32,
    pub(super) mipmap_mode: u32,
    pub(super) address_mode_u: u32,
    pub(super) address_mode_v: u32,
    pub(super) address_mode_w: u32,
    pub(super) mip_lod_bias: f32,
    pub(super) anisotropy_enable: u32,
    pub(super) max_anisotropy: f32,
    pub(super) compare_enable: u32,
    pub(super) compare_op: u32,
    pub(super) min_lod: f32,
    pub(super) max_lod: f32,
    pub(super) border_color: u32,
    pub(super) unnormalized_coordinates: u32,
}

#[repr(C)]
pub(super) struct VkDescriptorSetLayoutBinding {
    pub(super) binding: u32,
    pub(super) descriptor_type: u32,
    pub(super) descriptor_count: u32,
    pub(super) stage_flags: u32,
    pub(super) p_immutable_samplers: *const VkSampler,
}

#[repr(C)]
pub(super) struct VkDescriptorSetLayoutCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) binding_count: u32,
    pub(super) p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
pub(super) struct VkDescriptorPoolSize {
    pub(super) descriptor_type: u32,
    pub(super) descriptor_count: u32,
}

#[repr(C)]
pub(super) struct VkDescriptorPoolCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) max_sets: u32,
    pub(super) pool_size_count: u32,
    pub(super) p_pool_sizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
pub(super) struct VkDescriptorSetAllocateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) descriptor_pool: VkDescriptorPool,
    pub(super) descriptor_set_count: u32,
    pub(super) p_set_layouts: *const VkDescriptorSetLayout,
}

#[repr(C)]
pub(super) struct VkDescriptorImageInfo {
    pub(super) sampler: VkSampler,
    pub(super) image_view: VkImageView,
    pub(super) image_layout: u32,
}

#[repr(C)]
pub(super) struct VkWriteDescriptorSet {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) dst_set: VkDescriptorSet,
    pub(super) dst_binding: u32,
    pub(super) dst_array_element: u32,
    pub(super) descriptor_count: u32,
    pub(super) descriptor_type: u32,
    pub(super) p_image_info: *const VkDescriptorImageInfo,
    pub(super) p_buffer_info: *const c_void,
    pub(super) p_texel_buffer_view: *const c_void,
}

#[repr(C)]
pub(super) struct VkAttachmentDescription {
    pub(super) flags: u32,
    pub(super) format: u32,
    pub(super) samples: u32,
    pub(super) load_op: u32,
    pub(super) store_op: u32,
    pub(super) stencil_load_op: u32,
    pub(super) stencil_store_op: u32,
    pub(super) initial_layout: u32,
    pub(super) final_layout: u32,
}

#[repr(C)]
pub(super) struct VkAttachmentReference {
    pub(super) attachment: u32,
    pub(super) layout: u32,
}

#[repr(C)]
pub(super) struct VkSubpassDescription {
    pub(super) flags: u32,
    pub(super) pipeline_bind_point: u32,
    pub(super) input_attachment_count: u32,
    pub(super) p_input_attachments: *const VkAttachmentReference,
    pub(super) color_attachment_count: u32,
    pub(super) p_color_attachments: *const VkAttachmentReference,
    pub(super) p_resolve_attachments: *const VkAttachmentReference,
    pub(super) p_depth_stencil_attachment: *const VkAttachmentReference,
    pub(super) preserve_attachment_count: u32,
    pub(super) p_preserve_attachments: *const u32,
}

#[repr(C)]
pub(super) struct VkRenderPassCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) attachment_count: u32,
    pub(super) p_attachments: *const VkAttachmentDescription,
    pub(super) subpass_count: u32,
    pub(super) p_subpasses: *const VkSubpassDescription,
    pub(super) dependency_count: u32,
    pub(super) p_dependencies: *const c_void,
}

#[repr(C)]
pub(super) struct VkFramebufferCreateInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) flags: u32,
    pub(super) render_pass: VkRenderPass,
    pub(super) attachment_count: u32,
    pub(super) p_attachments: *const VkImageView,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) layers: u32,
}

#[repr(C)]
pub(super) struct VkRenderPassBeginInfo {
    pub(super) s_type: u32,
    pub(super) p_next: *const c_void,
    pub(super) render_pass: VkRenderPass,
    pub(super) framebuffer: VkFramebuffer,
    pub(super) render_area: VkRect2D,
    pub(super) clear_value_count: u32,
    pub(super) p_clear_values: *const VkClearColorValue,
}

#[repr(C)]
pub(super) struct VkImageSubresourceLayers {
    pub(super) aspect_mask: u32,
    pub(super) mip_level: u32,
    pub(super) base_array_layer: u32,
    pub(super) layer_count: u32,
}

#[repr(C)]
pub(super) struct VkOffset3D {
    pub(super) x: i32,
    pub(super) y: i32,
    pub(super) z: i32,
}

#[repr(C)]
pub(super) struct VkBufferImageCopy {
    pub(super) buffer_offset: u64,
    pub(super) buffer_row_length: u32,
    pub(super) buffer_image_height: u32,
    pub(super) image_subresource: VkImageSubresourceLayers,
    pub(super) image_offset: VkOffset3D,
    pub(super) image_extent: VkExtent3D,
}

// ---------------------------------------------------------------------------
// Function pointer types.
// ---------------------------------------------------------------------------

/// A yet-untyped Vulkan command, as `vkGet*ProcAddr` returns it.
pub(super) type PfnVoid = unsafe extern "C" fn();

pub(super) type PfnGetInstanceProcAddr =
    unsafe extern "C" fn(VkInstance, *const c_char) -> Option<PfnVoid>;
pub(super) type PfnCreateInstance =
    unsafe extern "C" fn(*const VkInstanceCreateInfo, *const c_void, *mut VkInstance) -> VkResult;
pub(super) type PfnDestroyInstance = unsafe extern "C" fn(VkInstance, *const c_void);
pub(super) type PfnEnumeratePhysicalDevices =
    unsafe extern "C" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> VkResult;
pub(super) type PfnGetPhysicalDeviceProperties2 =
    unsafe extern "C" fn(VkPhysicalDevice, *mut VkPhysicalDeviceProperties2);
pub(super) type PfnEnumerateDeviceExtensionProperties = unsafe extern "C" fn(
    VkPhysicalDevice,
    *const c_char,
    *mut u32,
    *mut VkExtensionProperties,
) -> VkResult;
pub(super) type PfnGetPhysicalDeviceQueueFamilyProperties =
    unsafe extern "C" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties);
pub(super) type PfnCreateDevice = unsafe extern "C" fn(
    VkPhysicalDevice,
    *const VkDeviceCreateInfo,
    *const c_void,
    *mut VkDevice,
) -> VkResult;
pub(super) type PfnGetDeviceProcAddr =
    unsafe extern "C" fn(VkDevice, *const c_char) -> Option<PfnVoid>;
pub(super) type PfnDestroyDevice = unsafe extern "C" fn(VkDevice, *const c_void);
pub(super) type PfnGetDeviceQueue = unsafe extern "C" fn(VkDevice, u32, u32, *mut VkQueue);
pub(super) type PfnGetPhysicalDeviceMemoryProperties =
    unsafe extern "C" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties);
pub(super) type PfnGetPhysicalDeviceFormatProperties2 =
    unsafe extern "C" fn(VkPhysicalDevice, u32, *mut VkFormatProperties2);
pub(super) type PfnCreateImage = unsafe extern "C" fn(
    VkDevice,
    *const VkImageCreateInfo,
    *const c_void,
    *mut VkImage,
) -> VkResult;
pub(super) type PfnDestroyImage = unsafe extern "C" fn(VkDevice, VkImage, *const c_void);
pub(super) type PfnGetImageMemoryRequirements =
    unsafe extern "C" fn(VkDevice, VkImage, *mut VkMemoryRequirements);
pub(super) type PfnAllocateMemory = unsafe extern "C" fn(
    VkDevice,
    *const VkMemoryAllocateInfo,
    *const c_void,
    *mut VkDeviceMemory,
) -> VkResult;
pub(super) type PfnFreeMemory = unsafe extern "C" fn(VkDevice, VkDeviceMemory, *const c_void);
pub(super) type PfnBindImageMemory =
    unsafe extern "C" fn(VkDevice, VkImage, VkDeviceMemory, u64) -> VkResult;
pub(super) type PfnGetImageDrmFormatModifierPropertiesEXT =
    unsafe extern "C" fn(VkDevice, VkImage, *mut VkImageDrmFormatModifierPropertiesEXT) -> VkResult;
pub(super) type PfnGetImageSubresourceLayout =
    unsafe extern "C" fn(VkDevice, VkImage, *const VkImageSubresource, *mut VkSubresourceLayout);
pub(super) type PfnGetMemoryFdKHR =
    unsafe extern "C" fn(VkDevice, *const VkMemoryGetFdInfoKHR, *mut i32) -> VkResult;
pub(super) type PfnCreateCommandPool = unsafe extern "C" fn(
    VkDevice,
    *const VkCommandPoolCreateInfo,
    *const c_void,
    *mut VkCommandPool,
) -> VkResult;
pub(super) type PfnDestroyCommandPool =
    unsafe extern "C" fn(VkDevice, VkCommandPool, *const c_void);
pub(super) type PfnAllocateCommandBuffers = unsafe extern "C" fn(
    VkDevice,
    *const VkCommandBufferAllocateInfo,
    *mut VkCommandBuffer,
) -> VkResult;
pub(super) type PfnFreeCommandBuffers =
    unsafe extern "C" fn(VkDevice, VkCommandPool, u32, *const VkCommandBuffer);
pub(super) type PfnBeginCommandBuffer =
    unsafe extern "C" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> VkResult;
pub(super) type PfnEndCommandBuffer = unsafe extern "C" fn(VkCommandBuffer) -> VkResult;
pub(super) type PfnCmdPipelineBarrier = unsafe extern "C" fn(
    VkCommandBuffer,
    u32,
    u32,
    u32,
    u32,
    *const VkMemoryBarrier,
    u32,
    *const c_void,
    u32,
    *const VkImageMemoryBarrier,
);
pub(super) type PfnQueueSubmit =
    unsafe extern "C" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> VkResult;
pub(super) type PfnCreateFence = unsafe extern "C" fn(
    VkDevice,
    *const VkFenceCreateInfo,
    *const c_void,
    *mut VkFence,
) -> VkResult;
pub(super) type PfnDestroyFence = unsafe extern "C" fn(VkDevice, VkFence, *const c_void);
pub(super) type PfnWaitForFences =
    unsafe extern "C" fn(VkDevice, u32, *const VkFence, u32, u64) -> VkResult;
pub(super) type PfnResetFences = unsafe extern "C" fn(VkDevice, u32, *const VkFence) -> VkResult;
pub(super) type PfnCreateSemaphore = unsafe extern "C" fn(
    VkDevice,
    *const VkSemaphoreCreateInfo,
    *const c_void,
    *mut VkSemaphore,
) -> VkResult;
pub(super) type PfnDestroySemaphore = unsafe extern "C" fn(VkDevice, VkSemaphore, *const c_void);
pub(super) type PfnGetSemaphoreFdKHR =
    unsafe extern "C" fn(VkDevice, *const VkSemaphoreGetFdInfoKHR, *mut i32) -> VkResult;
pub(super) type PfnImportSemaphoreFdKHR =
    unsafe extern "C" fn(VkDevice, *const VkImportSemaphoreFdInfoKHR) -> VkResult;
pub(super) type PfnDeviceWaitIdle = unsafe extern "C" fn(VkDevice) -> VkResult;
pub(super) type PfnCreateBuffer = unsafe extern "C" fn(
    VkDevice,
    *const VkBufferCreateInfo,
    *const c_void,
    *mut VkBuffer,
) -> VkResult;
pub(super) type PfnDestroyBuffer = unsafe extern "C" fn(VkDevice, VkBuffer, *const c_void);
pub(super) type PfnGetBufferMemoryRequirements =
    unsafe extern "C" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements);
pub(super) type PfnBindBufferMemory =
    unsafe extern "C" fn(VkDevice, VkBuffer, VkDeviceMemory, u64) -> VkResult;
pub(super) type PfnMapMemory =
    unsafe extern "C" fn(VkDevice, VkDeviceMemory, u64, u64, u32, *mut *mut c_void) -> VkResult;
pub(super) type PfnCreateShaderModule = unsafe extern "C" fn(
    VkDevice,
    *const VkShaderModuleCreateInfo,
    *const c_void,
    *mut VkShaderModule,
) -> VkResult;
pub(super) type PfnDestroyShaderModule =
    unsafe extern "C" fn(VkDevice, VkShaderModule, *const c_void);
pub(super) type PfnCreatePipelineLayout = unsafe extern "C" fn(
    VkDevice,
    *const VkPipelineLayoutCreateInfo,
    *const c_void,
    *mut VkPipelineLayout,
) -> VkResult;
pub(super) type PfnDestroyPipelineLayout =
    unsafe extern "C" fn(VkDevice, VkPipelineLayout, *const c_void);
pub(super) type PfnCreateGraphicsPipelines = unsafe extern "C" fn(
    VkDevice,
    u64,
    u32,
    *const VkGraphicsPipelineCreateInfo,
    *const c_void,
    *mut VkPipeline,
) -> VkResult;
pub(super) type PfnDestroyPipeline = unsafe extern "C" fn(VkDevice, VkPipeline, *const c_void);
pub(super) type PfnCreateDescriptorSetLayout = unsafe extern "C" fn(
    VkDevice,
    *const VkDescriptorSetLayoutCreateInfo,
    *const c_void,
    *mut VkDescriptorSetLayout,
) -> VkResult;
pub(super) type PfnDestroyDescriptorSetLayout =
    unsafe extern "C" fn(VkDevice, VkDescriptorSetLayout, *const c_void);
pub(super) type PfnCreateDescriptorPool = unsafe extern "C" fn(
    VkDevice,
    *const VkDescriptorPoolCreateInfo,
    *const c_void,
    *mut VkDescriptorPool,
) -> VkResult;
pub(super) type PfnDestroyDescriptorPool =
    unsafe extern "C" fn(VkDevice, VkDescriptorPool, *const c_void);
pub(super) type PfnAllocateDescriptorSets = unsafe extern "C" fn(
    VkDevice,
    *const VkDescriptorSetAllocateInfo,
    *mut VkDescriptorSet,
) -> VkResult;
pub(super) type PfnFreeDescriptorSets =
    unsafe extern "C" fn(VkDevice, VkDescriptorPool, u32, *const VkDescriptorSet) -> VkResult;
pub(super) type PfnUpdateDescriptorSets =
    unsafe extern "C" fn(VkDevice, u32, *const VkWriteDescriptorSet, u32, *const c_void);
pub(super) type PfnCreateSampler = unsafe extern "C" fn(
    VkDevice,
    *const VkSamplerCreateInfo,
    *const c_void,
    *mut VkSampler,
) -> VkResult;
pub(super) type PfnDestroySampler = unsafe extern "C" fn(VkDevice, VkSampler, *const c_void);
pub(super) type PfnCreateRenderPass = unsafe extern "C" fn(
    VkDevice,
    *const VkRenderPassCreateInfo,
    *const c_void,
    *mut VkRenderPass,
) -> VkResult;
pub(super) type PfnDestroyRenderPass = unsafe extern "C" fn(VkDevice, VkRenderPass, *const c_void);
pub(super) type PfnCreateFramebuffer = unsafe extern "C" fn(
    VkDevice,
    *const VkFramebufferCreateInfo,
    *const c_void,
    *mut VkFramebuffer,
) -> VkResult;
pub(super) type PfnDestroyFramebuffer =
    unsafe extern "C" fn(VkDevice, VkFramebuffer, *const c_void);
pub(super) type PfnCreateImageView = unsafe extern "C" fn(
    VkDevice,
    *const VkImageViewCreateInfo,
    *const c_void,
    *mut VkImageView,
) -> VkResult;
pub(super) type PfnDestroyImageView = unsafe extern "C" fn(VkDevice, VkImageView, *const c_void);
pub(super) type PfnCmdBeginRenderPass =
    unsafe extern "C" fn(VkCommandBuffer, *const VkRenderPassBeginInfo, u32);
pub(super) type PfnCmdEndRenderPass = unsafe extern "C" fn(VkCommandBuffer);
pub(super) type PfnCmdBindPipeline = unsafe extern "C" fn(VkCommandBuffer, u32, VkPipeline);
pub(super) type PfnCmdBindVertexBuffers =
    unsafe extern "C" fn(VkCommandBuffer, u32, u32, *const VkBuffer, *const u64);
pub(super) type PfnCmdBindDescriptorSets = unsafe extern "C" fn(
    VkCommandBuffer,
    u32,
    VkPipelineLayout,
    u32,
    u32,
    *const VkDescriptorSet,
    u32,
    *const u32,
);
pub(super) type PfnCmdPushConstants =
    unsafe extern "C" fn(VkCommandBuffer, VkPipelineLayout, u32, u32, u32, *const c_void);
pub(super) type PfnCmdSetViewport =
    unsafe extern "C" fn(VkCommandBuffer, u32, u32, *const VkViewport);
pub(super) type PfnCmdSetScissor = unsafe extern "C" fn(VkCommandBuffer, u32, u32, *const VkRect2D);
pub(super) type PfnCmdDraw = unsafe extern "C" fn(VkCommandBuffer, u32, u32, u32, u32);
pub(super) type PfnCmdCopyBufferToImage =
    unsafe extern "C" fn(VkCommandBuffer, VkBuffer, VkImage, u32, u32, *const VkBufferImageCopy);
pub(super) type PfnCmdCopyImageToBuffer =
    unsafe extern "C" fn(VkCommandBuffer, VkImage, u32, VkBuffer, u32, *const VkBufferImageCopy);

/// Fetch an instance-level command by name, as a specific pointer type.
///
/// # Safety
/// `T` must be the exact PFN type of `name` per the Vulkan spec.
pub(super) unsafe fn load<T>(
    gipa: PfnGetInstanceProcAddr,
    instance: VkInstance,
    name: &CStr,
) -> Result<T> {
    // SAFETY: gipa is the loader's vkGetInstanceProcAddr; name is a command name.
    let p = unsafe { gipa(instance, name.as_ptr()) }
        .ok_or_else(|| Error::msg(format!("vulkan loader lacks {}", name.to_string_lossy())))?;
    debug_assert_eq!(size_of::<T>(), size_of::<PfnVoid>());
    // SAFETY: caller guarantees T is the command's true signature; both are
    // fn-pointer sized (asserted above).
    Ok(unsafe { core::mem::transmute_copy::<PfnVoid, T>(&p) })
}

/// Map a VkResult to our error type with the failing call named.
pub(super) fn check(result: VkResult, what: &str) -> Result<()> {
    if result == VK_SUCCESS {
        Ok(())
    } else {
        Err(Error::msg(format!("{what} failed: VkResult {result}")))
    }
}

// ---------------------------------------------------------------------------
// Device function table.
// ---------------------------------------------------------------------------

/// Every device-level command the render path uses, loaded once at bring-up so
/// a driver missing one fails there instead of mid-frame.
pub(super) struct DeviceFns {
    pub(super) create_image: PfnCreateImage,
    pub(super) destroy_image: PfnDestroyImage,
    pub(super) get_image_memory_requirements: PfnGetImageMemoryRequirements,
    pub(super) allocate_memory: PfnAllocateMemory,
    pub(super) free_memory: PfnFreeMemory,
    pub(super) bind_image_memory: PfnBindImageMemory,
    pub(super) get_image_drm_format_modifier_properties: PfnGetImageDrmFormatModifierPropertiesEXT,
    pub(super) get_image_subresource_layout: PfnGetImageSubresourceLayout,
    pub(super) get_memory_fd: PfnGetMemoryFdKHR,
    pub(super) create_command_pool: PfnCreateCommandPool,
    pub(super) destroy_command_pool: PfnDestroyCommandPool,
    pub(super) allocate_command_buffers: PfnAllocateCommandBuffers,
    pub(super) free_command_buffers: PfnFreeCommandBuffers,
    pub(super) begin_command_buffer: PfnBeginCommandBuffer,
    pub(super) end_command_buffer: PfnEndCommandBuffer,
    pub(super) cmd_pipeline_barrier: PfnCmdPipelineBarrier,
    pub(super) queue_submit: PfnQueueSubmit,
    pub(super) create_fence: PfnCreateFence,
    pub(super) destroy_fence: PfnDestroyFence,
    pub(super) wait_for_fences: PfnWaitForFences,
    pub(super) reset_fences: PfnResetFences,
    pub(super) create_semaphore: PfnCreateSemaphore,
    pub(super) destroy_semaphore: PfnDestroySemaphore,
    pub(super) get_semaphore_fd: PfnGetSemaphoreFdKHR,
    pub(super) import_semaphore_fd: PfnImportSemaphoreFdKHR,
    pub(super) device_wait_idle: PfnDeviceWaitIdle,
    pub(super) create_buffer: PfnCreateBuffer,
    pub(super) destroy_buffer: PfnDestroyBuffer,
    pub(super) get_buffer_memory_requirements: PfnGetBufferMemoryRequirements,
    pub(super) bind_buffer_memory: PfnBindBufferMemory,
    pub(super) map_memory: PfnMapMemory,
    pub(super) create_shader_module: PfnCreateShaderModule,
    pub(super) destroy_shader_module: PfnDestroyShaderModule,
    pub(super) create_pipeline_layout: PfnCreatePipelineLayout,
    pub(super) destroy_pipeline_layout: PfnDestroyPipelineLayout,
    pub(super) create_graphics_pipelines: PfnCreateGraphicsPipelines,
    pub(super) destroy_pipeline: PfnDestroyPipeline,
    pub(super) create_descriptor_set_layout: PfnCreateDescriptorSetLayout,
    pub(super) destroy_descriptor_set_layout: PfnDestroyDescriptorSetLayout,
    pub(super) create_descriptor_pool: PfnCreateDescriptorPool,
    pub(super) destroy_descriptor_pool: PfnDestroyDescriptorPool,
    pub(super) allocate_descriptor_sets: PfnAllocateDescriptorSets,
    pub(super) free_descriptor_sets: PfnFreeDescriptorSets,
    pub(super) update_descriptor_sets: PfnUpdateDescriptorSets,
    pub(super) create_sampler: PfnCreateSampler,
    pub(super) destroy_sampler: PfnDestroySampler,
    pub(super) create_render_pass: PfnCreateRenderPass,
    pub(super) destroy_render_pass: PfnDestroyRenderPass,
    pub(super) create_framebuffer: PfnCreateFramebuffer,
    pub(super) destroy_framebuffer: PfnDestroyFramebuffer,
    pub(super) create_image_view: PfnCreateImageView,
    pub(super) destroy_image_view: PfnDestroyImageView,
    pub(super) cmd_begin_render_pass: PfnCmdBeginRenderPass,
    pub(super) cmd_end_render_pass: PfnCmdEndRenderPass,
    pub(super) cmd_bind_pipeline: PfnCmdBindPipeline,
    pub(super) cmd_bind_vertex_buffers: PfnCmdBindVertexBuffers,
    pub(super) cmd_bind_descriptor_sets: PfnCmdBindDescriptorSets,
    pub(super) cmd_push_constants: PfnCmdPushConstants,
    pub(super) cmd_set_viewport: PfnCmdSetViewport,
    pub(super) cmd_set_scissor: PfnCmdSetScissor,
    pub(super) cmd_draw: PfnCmdDraw,
    pub(super) cmd_copy_buffer_to_image: PfnCmdCopyBufferToImage,
    pub(super) cmd_copy_image_to_buffer: PfnCmdCopyImageToBuffer,
}

impl DeviceFns {
    pub(super) fn load(gdpa: PfnGetDeviceProcAddr, device: VkDevice) -> Result<Self> {
        // SAFETY (each load): the name matches the transcribed PFN type.
        unsafe {
            Ok(Self {
                create_image: load_device(gdpa, device, c"vkCreateImage")?,
                destroy_image: load_device(gdpa, device, c"vkDestroyImage")?,
                get_image_memory_requirements: load_device(
                    gdpa,
                    device,
                    c"vkGetImageMemoryRequirements",
                )?,
                allocate_memory: load_device(gdpa, device, c"vkAllocateMemory")?,
                free_memory: load_device(gdpa, device, c"vkFreeMemory")?,
                bind_image_memory: load_device(gdpa, device, c"vkBindImageMemory")?,
                get_image_drm_format_modifier_properties: load_device(
                    gdpa,
                    device,
                    c"vkGetImageDrmFormatModifierPropertiesEXT",
                )?,
                get_image_subresource_layout: load_device(
                    gdpa,
                    device,
                    c"vkGetImageSubresourceLayout",
                )?,
                get_memory_fd: load_device(gdpa, device, c"vkGetMemoryFdKHR")?,
                create_command_pool: load_device(gdpa, device, c"vkCreateCommandPool")?,
                destroy_command_pool: load_device(gdpa, device, c"vkDestroyCommandPool")?,
                allocate_command_buffers: load_device(gdpa, device, c"vkAllocateCommandBuffers")?,
                free_command_buffers: load_device(gdpa, device, c"vkFreeCommandBuffers")?,
                begin_command_buffer: load_device(gdpa, device, c"vkBeginCommandBuffer")?,
                end_command_buffer: load_device(gdpa, device, c"vkEndCommandBuffer")?,
                cmd_pipeline_barrier: load_device(gdpa, device, c"vkCmdPipelineBarrier")?,
                queue_submit: load_device(gdpa, device, c"vkQueueSubmit")?,
                create_fence: load_device(gdpa, device, c"vkCreateFence")?,
                destroy_fence: load_device(gdpa, device, c"vkDestroyFence")?,
                wait_for_fences: load_device(gdpa, device, c"vkWaitForFences")?,
                reset_fences: load_device(gdpa, device, c"vkResetFences")?,
                create_semaphore: load_device(gdpa, device, c"vkCreateSemaphore")?,
                destroy_semaphore: load_device(gdpa, device, c"vkDestroySemaphore")?,
                get_semaphore_fd: load_device(gdpa, device, c"vkGetSemaphoreFdKHR")?,
                import_semaphore_fd: load_device(gdpa, device, c"vkImportSemaphoreFdKHR")?,
                device_wait_idle: load_device(gdpa, device, c"vkDeviceWaitIdle")?,
                create_buffer: load_device(gdpa, device, c"vkCreateBuffer")?,
                destroy_buffer: load_device(gdpa, device, c"vkDestroyBuffer")?,
                get_buffer_memory_requirements: load_device(
                    gdpa,
                    device,
                    c"vkGetBufferMemoryRequirements",
                )?,
                bind_buffer_memory: load_device(gdpa, device, c"vkBindBufferMemory")?,
                map_memory: load_device(gdpa, device, c"vkMapMemory")?,
                create_shader_module: load_device(gdpa, device, c"vkCreateShaderModule")?,
                destroy_shader_module: load_device(gdpa, device, c"vkDestroyShaderModule")?,
                create_pipeline_layout: load_device(gdpa, device, c"vkCreatePipelineLayout")?,
                destroy_pipeline_layout: load_device(gdpa, device, c"vkDestroyPipelineLayout")?,
                create_graphics_pipelines: load_device(gdpa, device, c"vkCreateGraphicsPipelines")?,
                destroy_pipeline: load_device(gdpa, device, c"vkDestroyPipeline")?,
                create_descriptor_set_layout: load_device(
                    gdpa,
                    device,
                    c"vkCreateDescriptorSetLayout",
                )?,
                destroy_descriptor_set_layout: load_device(
                    gdpa,
                    device,
                    c"vkDestroyDescriptorSetLayout",
                )?,
                create_descriptor_pool: load_device(gdpa, device, c"vkCreateDescriptorPool")?,
                destroy_descriptor_pool: load_device(gdpa, device, c"vkDestroyDescriptorPool")?,
                allocate_descriptor_sets: load_device(gdpa, device, c"vkAllocateDescriptorSets")?,
                free_descriptor_sets: load_device(gdpa, device, c"vkFreeDescriptorSets")?,
                update_descriptor_sets: load_device(gdpa, device, c"vkUpdateDescriptorSets")?,
                create_sampler: load_device(gdpa, device, c"vkCreateSampler")?,
                destroy_sampler: load_device(gdpa, device, c"vkDestroySampler")?,
                create_render_pass: load_device(gdpa, device, c"vkCreateRenderPass")?,
                destroy_render_pass: load_device(gdpa, device, c"vkDestroyRenderPass")?,
                create_framebuffer: load_device(gdpa, device, c"vkCreateFramebuffer")?,
                destroy_framebuffer: load_device(gdpa, device, c"vkDestroyFramebuffer")?,
                create_image_view: load_device(gdpa, device, c"vkCreateImageView")?,
                destroy_image_view: load_device(gdpa, device, c"vkDestroyImageView")?,
                cmd_begin_render_pass: load_device(gdpa, device, c"vkCmdBeginRenderPass")?,
                cmd_end_render_pass: load_device(gdpa, device, c"vkCmdEndRenderPass")?,
                cmd_bind_pipeline: load_device(gdpa, device, c"vkCmdBindPipeline")?,
                cmd_bind_vertex_buffers: load_device(gdpa, device, c"vkCmdBindVertexBuffers")?,
                cmd_bind_descriptor_sets: load_device(gdpa, device, c"vkCmdBindDescriptorSets")?,
                cmd_push_constants: load_device(gdpa, device, c"vkCmdPushConstants")?,
                cmd_set_viewport: load_device(gdpa, device, c"vkCmdSetViewport")?,
                cmd_set_scissor: load_device(gdpa, device, c"vkCmdSetScissor")?,
                cmd_draw: load_device(gdpa, device, c"vkCmdDraw")?,
                cmd_copy_buffer_to_image: load_device(gdpa, device, c"vkCmdCopyBufferToImage")?,
                cmd_copy_image_to_buffer: load_device(gdpa, device, c"vkCmdCopyImageToBuffer")?,
            })
        }
    }
}

/// Fetch a device-level command by name (see [`load`]).
///
/// # Safety
/// `T` must be the exact PFN type of `name` per the Vulkan spec.
pub(super) unsafe fn load_device<T>(
    gdpa: PfnGetDeviceProcAddr,
    device: VkDevice,
    name: &CStr,
) -> Result<T> {
    // SAFETY: gdpa is vkGetDeviceProcAddr for this device.
    let p = unsafe { gdpa(device, name.as_ptr()) }
        .ok_or_else(|| Error::msg(format!("vulkan device lacks {}", name.to_string_lossy())))?;
    debug_assert_eq!(size_of::<T>(), size_of::<PfnVoid>());
    // SAFETY: caller guarantees T is the command's true signature.
    Ok(unsafe { core::mem::transmute_copy::<PfnVoid, T>(&p) })
}

#[cfg(test)]
mod tests {
    use crate::render::vulkan::abi::*;

    /// The transcription's one real hazard is a struct size drifting from the
    /// C ABI: the driver writes past what we allocated. These sizes are from
    /// vulkan_core.h on x86_64/arm64 (LP64).
    #[test]
    fn struct_sizes_match_the_c_abi() {
        assert_eq!(size_of::<VkPhysicalDeviceLimits>(), 504);
        assert_eq!(size_of::<VkPhysicalDeviceSparseProperties>(), 20);
        assert_eq!(size_of::<VkPhysicalDeviceProperties>(), 824);
        assert_eq!(size_of::<VkPhysicalDeviceProperties2>(), 840);
        assert_eq!(size_of::<VkPhysicalDeviceDrmPropertiesEXT>(), 56);
        assert_eq!(size_of::<VkQueueFamilyProperties>(), 24);
        assert_eq!(size_of::<VkExtensionProperties>(), 260);
        assert_eq!(size_of::<VkApplicationInfo>(), 48);
        assert_eq!(size_of::<VkInstanceCreateInfo>(), 64);
        assert_eq!(size_of::<VkDeviceQueueCreateInfo>(), 40);
        assert_eq!(size_of::<VkDeviceCreateInfo>(), 72);
        assert_eq!(size_of::<VkPhysicalDeviceMemoryProperties>(), 520);
        assert_eq!(size_of::<VkFormatProperties2>(), 32);
        assert_eq!(size_of::<VkDrmFormatModifierPropertiesEXT>(), 16);
        assert_eq!(size_of::<VkDrmFormatModifierPropertiesListEXT>(), 32);
        assert_eq!(size_of::<VkImageCreateInfo>(), 88);
        assert_eq!(size_of::<VkExternalMemoryImageCreateInfo>(), 24);
        assert_eq!(size_of::<VkImageFormatListCreateInfo>(), 32);
        assert_eq!(size_of::<VkImageDrmFormatModifierListCreateInfoEXT>(), 32);
        assert_eq!(size_of::<VkImageDrmFormatModifierPropertiesEXT>(), 24);
        assert_eq!(size_of::<VkMemoryRequirements>(), 24);
        assert_eq!(size_of::<VkMemoryAllocateInfo>(), 32);
        assert_eq!(size_of::<VkMemoryDedicatedAllocateInfo>(), 32);
        assert_eq!(size_of::<VkMemoryGetFdInfoKHR>(), 32);
        assert_eq!(size_of::<VkSubresourceLayout>(), 40);
        assert_eq!(size_of::<VkCommandBufferAllocateInfo>(), 32);
        assert_eq!(size_of::<VkMemoryBarrier>(), 24);
        assert_eq!(size_of::<VkImageMemoryBarrier>(), 72);
        assert_eq!(size_of::<VkSubmitInfo>(), 72);
        assert_eq!(size_of::<VkSemaphoreGetFdInfoKHR>(), 32);
        assert_eq!(size_of::<VkImportSemaphoreFdInfoKHR>(), 40);
    }

    /// Offsets and enum values emitted by a C compiler against vulkan_core.h (LP64).
    #[test]
    fn host_readback_barrier_matches_the_c_abi() {
        assert_eq!(std::mem::offset_of!(VkMemoryBarrier, s_type), 0);
        assert_eq!(std::mem::offset_of!(VkMemoryBarrier, p_next), 8);
        assert_eq!(std::mem::offset_of!(VkMemoryBarrier, src_access_mask), 16);
        assert_eq!(std::mem::offset_of!(VkMemoryBarrier, dst_access_mask), 20);
        assert_eq!(VK_STRUCTURE_TYPE_MEMORY_BARRIER, 46);
        assert_eq!(VK_ACCESS_TRANSFER_WRITE_BIT, 0x1000);
        assert_eq!(VK_ACCESS_HOST_READ_BIT, 0x2000);
        assert_eq!(VK_PIPELINE_STAGE_TRANSFER_BIT, 0x1000);
        assert_eq!(VK_PIPELINE_STAGE_HOST_BIT, 0x4000);
    }

    #[test]
    fn api_version_is_1_2() {
        assert_eq!(API_VERSION_1_2 >> 22, 1);
        assert_eq!((API_VERSION_1_2 >> 12) & 0x3ff, 2);
    }
}
