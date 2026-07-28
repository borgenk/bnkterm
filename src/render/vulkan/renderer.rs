use crate::platform::error::{Error, Result};
use crate::render::gpu::Vertex;
use crate::render::vulkan::abi::*;

/// One sampled texture (an atlas or the 1x1 dummy) and whether its contents have
/// ever been defined (governs the upload barrier's old layout).
#[derive(Default)]
pub(super) struct Texture {
    pub(super) image: VkImage,
    pub(super) memory: VkDeviceMemory,
    pub(super) view: VkImageView,
    pub(super) initialized: bool,
}

/// A host-visible, persistently mapped buffer (vertices or staging), grown by
/// recreation when a frame needs more.
#[derive(Default)]
pub(super) struct HostBuffer {
    pub(super) buf: VkBuffer,
    pub(super) memory: VkDeviceMemory,
    pub(super) ptr: *mut u8,
    pub(super) capacity: usize,
}

/// The draw machinery shared by every frame: the one pipeline over the one
/// render pass, the descriptor plumbing, and the sampled textures (the two
/// atlases, and the 1x1 dummy that stands in for an atlas not yet created).
#[derive(Default)]
pub(super) struct Renderer {
    pub(super) render_pass: VkRenderPass,
    pub(super) set_layout_atlas: VkDescriptorSetLayout,
    pub(super) pipeline_layout: VkPipelineLayout,
    pub(super) pipeline: VkPipeline,
    pub(super) sampler: VkSampler,
    pub(super) descriptor_pool: VkDescriptorPool,
    pub(super) set_atlas: VkDescriptorSet,
    pub(super) dummy: Texture,
    /// The R8 glyph atlas and BGRA emoji atlas textures, each tagged with the
    /// CPU mirror generation they were created for.
    pub(super) glyph_tex: Option<(Texture, u32)>,
    pub(super) emoji_tex: Option<(Texture, u32)>,
}

/// The full-image color subresource range every barrier and view here uses.
pub(super) fn color_range() -> VkImageSubresourceRange {
    VkImageSubresourceRange {
        aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Point `binding` of `set` at `view` through `sampler`.
pub(super) fn write_image_set(
    fns: &DeviceFns,
    device: VkDevice,
    set: VkDescriptorSet,
    binding: u32,
    sampler: VkSampler,
    view: VkImageView,
) {
    let info = VkDescriptorImageInfo {
        sampler,
        image_view: view,
        image_layout: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    let write = VkWriteDescriptorSet {
        s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        p_next: core::ptr::null(),
        dst_set: set,
        dst_binding: binding,
        dst_array_element: 0,
        descriptor_count: 1,
        descriptor_type: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        p_image_info: &info,
        p_buffer_info: core::ptr::null(),
        p_texel_buffer_view: core::ptr::null(),
    };
    // SAFETY: set, sampler, and view are live; write outlives the call.
    unsafe { (fns.update_descriptor_sets)(device, 1, &write, 0, core::ptr::null()) };
}

/// Destroy a texture's view, memory, and image (a zeroed default is a no-op).
pub(super) fn destroy_texture(fns: &DeviceFns, device: VkDevice, t: Texture) {
    if t.image == 0 {
        return;
    }
    // SAFETY: all three were created together on this device and nothing in
    // flight references them (callers drain the queue where needed).
    unsafe {
        (fns.destroy_image_view)(device, t.view, core::ptr::null());
        (fns.free_memory)(device, t.memory, core::ptr::null());
        (fns.destroy_image)(device, t.image, core::ptr::null());
    }
}

impl Renderer {
    /// Destroy everything this renderer owns, in reverse creation order.
    /// Zeroed handles (a partial build) are skipped. The caller drained the
    /// queue.
    pub(super) fn destroy(self, fns: &DeviceFns, device: VkDevice) {
        if let Some((tex, _)) = self.glyph_tex {
            destroy_texture(fns, device, tex);
        }
        if let Some((tex, _)) = self.emoji_tex {
            destroy_texture(fns, device, tex);
        }
        destroy_texture(fns, device, self.dummy);
        // SAFETY: every non-zero handle below came from this device and is
        // destroyed exactly once; destroying the pool frees its sets.
        unsafe {
            if self.descriptor_pool != 0 {
                (fns.destroy_descriptor_pool)(device, self.descriptor_pool, core::ptr::null());
            }
            if self.pipeline != 0 {
                (fns.destroy_pipeline)(device, self.pipeline, core::ptr::null());
            }
            if self.pipeline_layout != 0 {
                (fns.destroy_pipeline_layout)(device, self.pipeline_layout, core::ptr::null());
            }
            if self.sampler != 0 {
                (fns.destroy_sampler)(device, self.sampler, core::ptr::null());
            }
            if self.render_pass != 0 {
                (fns.destroy_render_pass)(device, self.render_pass, core::ptr::null());
            }
            if self.set_layout_atlas != 0 {
                (fns.destroy_descriptor_set_layout)(
                    device,
                    self.set_layout_atlas,
                    core::ptr::null(),
                );
            }
        }
    }
}

/// SPIR-V bytes (the committed blobs) as the u32 words Vulkan wants.
fn shader_words(bytes: &[u8]) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) || bytes.is_empty() {
        return Err(Error::msg("malformed SPIR-V blob"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Compile one SPIR-V stage into a shader module. `what` names the stage for the
/// error message.
fn create_shader_module(
    fns: &DeviceFns,
    device: VkDevice,
    words: &[u32],
    what: &str,
) -> Result<VkShaderModule> {
    let info = VkShaderModuleCreateInfo {
        s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        code_size: words.len() * 4,
        p_code: words.as_ptr(),
    };
    let mut module: VkShaderModule = 0;
    // SAFETY: the words outlive the call.
    check(
        unsafe { (fns.create_shader_module)(device, &info, core::ptr::null(), &mut module) },
        what,
    )?;
    Ok(module)
}

/// A descriptor-set layout of `binding_count` combined image samplers at bindings
/// `0..binding_count`, all visible to the fragment stage (the atlas set uses two).
fn sampler_set_layout(
    fns: &DeviceFns,
    device: VkDevice,
    binding_count: u32,
    what: &str,
    out: &mut VkDescriptorSetLayout,
) -> Result<()> {
    let bindings: Vec<VkDescriptorSetLayoutBinding> = (0..binding_count)
        .map(|binding| VkDescriptorSetLayoutBinding {
            binding,
            descriptor_type: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: VK_SHADER_STAGE_FRAGMENT_BIT,
            p_immutable_samplers: core::ptr::null(),
        })
        .collect();
    let info = VkDescriptorSetLayoutCreateInfo {
        s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        binding_count,
        p_bindings: bindings.as_ptr(),
    };
    // SAFETY: info and bindings outlive the call.
    check(
        unsafe { (fns.create_descriptor_set_layout)(device, &info, core::ptr::null(), out) },
        what,
    )
}

/// Allocate one descriptor set of `layout` from `pool`. Argument order matters:
/// `pool` and `layout` are both `u64` handle aliases, so a transposition would
/// compile — pool first, then layout.
pub(super) fn allocate_set(
    fns: &DeviceFns,
    device: VkDevice,
    pool: VkDescriptorPool,
    layout: VkDescriptorSetLayout,
    what: &str,
) -> Result<VkDescriptorSet> {
    let info = VkDescriptorSetAllocateInfo {
        s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        p_next: core::ptr::null(),
        descriptor_pool: pool,
        descriptor_set_count: 1,
        p_set_layouts: &layout,
    };
    let mut set: VkDescriptorSet = 0;
    // SAFETY: the pool and layout are live; one handle is written.
    check(
        unsafe { (fns.allocate_descriptor_sets)(device, &info, &mut set) },
        what,
    )?;
    Ok(set)
}

/// Build the frame renderer: the render pass, descriptor layouts, pipeline
/// (from the committed SPIR-V), sampler, and descriptor pool with the two
/// standing sets. On error, everything built so far is destroyed.
pub(super) fn create_renderer(fns: &DeviceFns, device: VkDevice) -> Result<Renderer> {
    let mut r = Renderer::default();
    match fill_renderer(&mut r, fns, device) {
        Ok(()) => Ok(r),
        Err(e) => {
            r.destroy(fns, device);
            Err(e)
        }
    }
}

fn fill_renderer(r: &mut Renderer, fns: &DeviceFns, device: VkDevice) -> Result<()> {
    // The render pass: one B8G8R8A8 color attachment, cleared on load, kept
    // in COLOR_ATTACHMENT_OPTIMAL (the ownership barriers live outside). The
    // sRGB format matches the image's sRGB view, so blending is gamma-correct
    // and the store re-encodes to the UNORM bytes the compositor reads.
    let attachment = VkAttachmentDescription {
        flags: 0,
        format: VK_FORMAT_B8G8R8A8_SRGB,
        samples: VK_SAMPLE_COUNT_1_BIT,
        load_op: VK_ATTACHMENT_LOAD_OP_CLEAR,
        store_op: VK_ATTACHMENT_STORE_OP_STORE,
        stencil_load_op: VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        stencil_store_op: VK_ATTACHMENT_STORE_OP_DONT_CARE,
        initial_layout: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        final_layout: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    let color_ref = VkAttachmentReference {
        attachment: 0,
        layout: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    let subpass = VkSubpassDescription {
        flags: 0,
        pipeline_bind_point: VK_PIPELINE_BIND_POINT_GRAPHICS,
        input_attachment_count: 0,
        p_input_attachments: core::ptr::null(),
        color_attachment_count: 1,
        p_color_attachments: &color_ref,
        p_resolve_attachments: core::ptr::null(),
        p_depth_stencil_attachment: core::ptr::null(),
        preserve_attachment_count: 0,
        p_preserve_attachments: core::ptr::null(),
    };
    let pass_info = VkRenderPassCreateInfo {
        s_type: VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        attachment_count: 1,
        p_attachments: &attachment,
        subpass_count: 1,
        p_subpasses: &subpass,
        dependency_count: 0,
        p_dependencies: core::ptr::null(),
    };
    // SAFETY: pass_info and its pointees outlive the call.
    check(
        unsafe {
            (fns.create_render_pass)(device, &pass_info, core::ptr::null(), &mut r.render_pass)
        },
        "vkCreateRenderPass",
    )?;

    // Descriptor layout: set 0 the two atlases (the image set is excised).
    sampler_set_layout(
        fns,
        device,
        2,
        "vkCreateDescriptorSetLayout (atlas)",
        &mut r.set_layout_atlas,
    )?;

    // Pipeline layout: the atlas set plus the push constant. **Eight bytes, one
    // member**: the viewport `vec2` at offset 0, which is what both shaders declare
    // (`layout(push_constant) uniform Push { vec2 viewport; }`) and what `target.rs`
    // pushes. The description here used to name a second member — a gamma float at
    // offset 8, 12 bytes total — left over from before the coverage exponent moved to a
    // per-run value. The code was right end to end and only the comment was stale, which
    // in a codebase where comments are the spec is the dangerous half: someone adding a
    // fragment constant at offset 8 would have trusted it and got a shader/layout
    // mismatch. `size` below is the single source of truth; the shaders are pinned
    // against it by `struct_sizes_match_the_c_abi`'s neighbours in `abi.rs`.
    let set_layouts = [r.set_layout_atlas];
    let push_range = VkPushConstantRange {
        stage_flags: VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        offset: 0,
        size: 8,
    };
    let layout_info = VkPipelineLayoutCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        set_layout_count: set_layouts.len() as u32,
        p_set_layouts: set_layouts.as_ptr(),
        push_constant_range_count: 1,
        p_push_constant_ranges: &push_range,
    };
    // SAFETY: layout_info and its pointees outlive the call.
    check(
        unsafe {
            (fns.create_pipeline_layout)(
                device,
                &layout_info,
                core::ptr::null(),
                &mut r.pipeline_layout,
            )
        },
        "vkCreatePipelineLayout",
    )?;

    // The pipeline, from the committed SPIR-V blobs.
    let vert_words = shader_words(include_bytes!("../../../shaders/quad.vert.spv"))?;
    let frag_words = shader_words(include_bytes!("../../../shaders/quad.frag.spv"))?;
    let vert_module =
        create_shader_module(fns, device, &vert_words, "vkCreateShaderModule (vert)")?;
    // The vert module is already live, so a frag failure must destroy it before
    // returning (the one non-guard unwind left in this function).
    let frag_module =
        match create_shader_module(fns, device, &frag_words, "vkCreateShaderModule (frag)") {
            Ok(module) => module,
            Err(e) => {
                // SAFETY: vert_module is live and unused.
                unsafe { (fns.destroy_shader_module)(device, vert_module, core::ptr::null()) };
                return Err(e);
            }
        };

    let pipeline_result = create_pipeline(fns, device, r, vert_module, frag_module);
    // The modules are compiled into the pipeline; they are not needed after.
    // SAFETY: both modules are live; the pipeline (if any) keeps its own copy.
    unsafe {
        (fns.destroy_shader_module)(device, vert_module, core::ptr::null());
        (fns.destroy_shader_module)(device, frag_module, core::ptr::null());
    }
    pipeline_result?;

    // The glyph and emoji paths sample with texelFetch, which ignores the
    // filter, so LINEAR here is inert; it is kept as the neutral default now
    // that the image path (the only texture() sampler) is gone.
    let sampler_info = VkSamplerCreateInfo {
        s_type: VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        mag_filter: VK_FILTER_LINEAR,
        min_filter: VK_FILTER_LINEAR,
        mipmap_mode: 0,
        address_mode_u: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        address_mode_v: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        address_mode_w: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        mip_lod_bias: 0.0,
        anisotropy_enable: 0,
        max_anisotropy: 1.0,
        compare_enable: 0,
        compare_op: 0,
        min_lod: 0.0,
        max_lod: 0.0,
        border_color: 0,
        unnormalized_coordinates: 0,
    };
    // SAFETY: sampler_info outlives the call.
    check(
        unsafe { (fns.create_sampler)(device, &sampler_info, core::ptr::null(), &mut r.sampler) },
        "vkCreateSampler",
    )?;

    // The descriptor pool and the two standing sets.
    let pool_size = VkDescriptorPoolSize {
        descriptor_type: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        descriptor_count: DESCRIPTOR_MAX_SETS * 2,
    };
    let pool_info = VkDescriptorPoolCreateInfo {
        s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        max_sets: DESCRIPTOR_MAX_SETS,
        pool_size_count: 1,
        p_pool_sizes: &pool_size,
    };
    // SAFETY: pool_info outlives the call.
    check(
        unsafe {
            (fns.create_descriptor_pool)(
                device,
                &pool_info,
                core::ptr::null(),
                &mut r.descriptor_pool,
            )
        },
        "vkCreateDescriptorPool",
    )?;
    r.set_atlas = allocate_set(
        fns,
        device,
        r.descriptor_pool,
        r.set_layout_atlas,
        "vkAllocateDescriptorSets (atlas)",
    )?;
    Ok(())
}

/// The one graphics pipeline: the quad vertex layout, alpha blending matching
/// the software blitter, dynamic viewport/scissor.
fn create_pipeline(
    fns: &DeviceFns,
    device: VkDevice,
    r: &mut Renderer,
    vert: VkShaderModule,
    frag: VkShaderModule,
) -> Result<()> {
    let stage = |stage: u32, module: VkShaderModule| VkPipelineShaderStageCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        stage,
        module,
        p_name: c"main".as_ptr(),
        p_specialization_info: core::ptr::null(),
    };
    let stages = [
        stage(VK_SHADER_STAGE_VERTEX_BIT, vert),
        stage(VK_SHADER_STAGE_FRAGMENT_BIT, frag),
    ];
    let binding = VkVertexInputBindingDescription {
        binding: 0,
        stride: size_of::<Vertex>() as u32,
        input_rate: VK_VERTEX_INPUT_RATE_VERTEX,
    };
    // Locations mirror shaders/quad.vert; offsets mirror gpu::Vertex.
    let attrs = [
        VkVertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: VK_FORMAT_R32G32_SFLOAT,
            offset: 0,
        },
        VkVertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: VK_FORMAT_R32G32_SFLOAT,
            offset: 8,
        },
        VkVertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: VK_FORMAT_R32G32B32A32_SFLOAT,
            offset: 16,
        },
        VkVertexInputAttributeDescription {
            location: 3,
            binding: 0,
            format: VK_FORMAT_R32G32B32A32_SFLOAT,
            offset: 32,
        },
        VkVertexInputAttributeDescription {
            location: 4,
            binding: 0,
            format: VK_FORMAT_R32_UINT,
            offset: 48,
        },
    ];
    let vertex_input = VkPipelineVertexInputStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        vertex_binding_description_count: 1,
        p_vertex_binding_descriptions: &binding,
        vertex_attribute_description_count: attrs.len() as u32,
        p_vertex_attribute_descriptions: attrs.as_ptr(),
    };
    let input_assembly = VkPipelineInputAssemblyStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        topology: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        primitive_restart_enable: 0,
    };
    // Viewport and scissor are dynamic (set per frame for the window size).
    let viewport_state = VkPipelineViewportStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        viewport_count: 1,
        p_viewports: core::ptr::null(),
        scissor_count: 1,
        p_scissors: core::ptr::null(),
    };
    let raster = VkPipelineRasterizationStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        depth_clamp_enable: 0,
        rasterizer_discard_enable: 0,
        polygon_mode: VK_POLYGON_MODE_FILL,
        cull_mode: 0,
        front_face: 0,
        depth_bias_enable: 0,
        depth_bias_constant_factor: 0.0,
        depth_bias_clamp: 0.0,
        depth_bias_slope_factor: 0.0,
        line_width: 1.0,
    };
    let multisample = VkPipelineMultisampleStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        rasterization_samples: VK_SAMPLE_COUNT_1_BIT,
        sample_shading_enable: 0,
        min_sample_shading: 0.0,
        p_sample_mask: core::ptr::null(),
        alpha_to_coverage_enable: 0,
        alpha_to_one_enable: 0,
    };
    // Straight-alpha source-over blending: the fragment colour, weighted by its
    // own alpha, over one-minus-alpha of the destination. Solid quads carry
    // alpha 1 and come out exact.
    let blend_attachment = VkPipelineColorBlendAttachmentState {
        blend_enable: 1,
        src_color_blend_factor: VK_BLEND_FACTOR_SRC_ALPHA,
        dst_color_blend_factor: VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        color_blend_op: VK_BLEND_OP_ADD,
        src_alpha_blend_factor: VK_BLEND_FACTOR_ONE,
        dst_alpha_blend_factor: VK_BLEND_FACTOR_ZERO,
        alpha_blend_op: VK_BLEND_OP_ADD,
        color_write_mask: VK_COLOR_COMPONENT_RGBA,
    };
    let blend = VkPipelineColorBlendStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        logic_op_enable: 0,
        logic_op: 0,
        attachment_count: 1,
        p_attachments: &blend_attachment,
        blend_constants: [0.0; 4],
    };
    let dynamic_states = [VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR];
    let dynamic = VkPipelineDynamicStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        dynamic_state_count: dynamic_states.len() as u32,
        p_dynamic_states: dynamic_states.as_ptr(),
    };
    let info = VkGraphicsPipelineCreateInfo {
        s_type: VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        stage_count: stages.len() as u32,
        p_stages: stages.as_ptr(),
        p_vertex_input_state: &vertex_input,
        p_input_assembly_state: &input_assembly,
        p_tessellation_state: core::ptr::null(),
        p_viewport_state: &viewport_state,
        p_rasterization_state: &raster,
        p_multisample_state: &multisample,
        p_depth_stencil_state: core::ptr::null(),
        p_color_blend_state: &blend,
        p_dynamic_state: &dynamic,
        layout: r.pipeline_layout,
        render_pass: r.render_pass,
        subpass: 0,
        base_pipeline_handle: 0,
        base_pipeline_index: -1,
    };
    // SAFETY: info and every state struct it points at outlive the call.
    check(
        unsafe {
            (fns.create_graphics_pipelines)(device, 0, 1, &info, core::ptr::null(), &mut r.pipeline)
        },
        "vkCreateGraphicsPipelines",
    )
}
