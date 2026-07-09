use core::ffi::c_void;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

use crate::platform::error::{Error, Result};
use crate::render::gpu::{AtlasInfo, AtlasUpload, FrameData};
use crate::render::vulkan::abi::*;
use crate::render::vulkan::renderer::{
    color_range, destroy_texture, write_image_set, HostBuffer, Texture,
};
use crate::render::vulkan::Gpu;

// ---------------------------------------------------------------------------
// Render targets.
// ---------------------------------------------------------------------------

/// Which of the two font atlases a sync/upload targets. The glyph atlas is a
/// single-channel coverage mask; the emoji atlas is full-colour, so they differ
/// only in their texture format (and which `Renderer` slot holds them).
#[derive(Clone, Copy)]
enum AtlasSlot {
    Glyph,
    Emoji,
}

impl AtlasSlot {
    fn format(self) -> u32 {
        match self {
            AtlasSlot::Glyph => VK_FORMAT_R8_UNORM,
            AtlasSlot::Emoji => VK_FORMAT_B8G8R8A8_UNORM,
        }
    }
}

/// One presentable render target: a device-local image whose memory is
/// exported as the dmabuf behind a `wl_buffer`, plus the per-image submission
/// state (command buffer, completion fence, and the two semaphores of the
/// implicit-sync bridge). Created and destroyed through [`Gpu`]; it holds raw
/// handles, so dropping it without [`Gpu::destroy_image`] leaks GPU memory
/// until the device goes down.
pub struct GpuImage {
    image: VkImage,
    memory: VkDeviceMemory,
    /// The exported dmabuf, kept for the sync-file ioctls; the compositor
    /// holds its own reference once the wl_buffer imports it.
    fd: OwnedFd,
    cmd: VkCommandBuffer,
    /// Signaled when the last submission touching this image completed.
    fence: VkFence,
    /// Carries the compositor's read fences into the next submission (a
    /// temporary sync-fd import per frame).
    acquire: VkSemaphore,
    /// Signaled by each submission; exported as a sync file and attached to
    /// the dmabuf so the compositor's reads wait for the render.
    release: VkSemaphore,
    /// The render-pass view of the image and its framebuffer.
    view: VkImageView,
    framebuffer: VkFramebuffer,
    /// This frame slot's vertex buffer and upload staging, safe to rewrite
    /// once the slot's fence has signaled.
    vertices: HostBuffer,
    staging: HostBuffer,
    width: u32,
    height: u32,
    stride: u32,
    offset: u32,
    modifier: u64,
    /// False until the first submission: the first frame enters from
    /// UNDEFINED, later frames re-acquire from the foreign queue family.
    used: bool,
}

impl GpuImage {
    /// The dmabuf to hand to `zwp_linux_buffer_params_v1.add`.
    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    pub fn stride(&self) -> u32 {
        self.stride
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    /// The modifier the driver actually chose from the allowed list.
    pub fn modifier(&self) -> u64 {
        self.modifier
    }
}

/// Partially constructed image state, so an error mid-`create_image` destroys
/// exactly what exists so far (handle 0 = never created).
struct PendingImage<'a> {
    gpu: &'a Gpu,
    image: VkImage,
    memory: VkDeviceMemory,
    fence: VkFence,
    acquire: VkSemaphore,
    release: VkSemaphore,
    cmd: VkCommandBuffer,
    view: VkImageView,
    framebuffer: VkFramebuffer,
    vertices: HostBuffer,
    staging: HostBuffer,
}

impl Drop for PendingImage<'_> {
    fn drop(&mut self) {
        let d = self.gpu.device.raw;
        let f = &self.gpu.fns;
        // SAFETY: every non-zero handle below was created on this device and
        // is destroyed exactly once; nothing was ever submitted with them.
        unsafe {
            if self.framebuffer != 0 {
                (f.destroy_framebuffer)(d, self.framebuffer, core::ptr::null());
            }
            if self.view != 0 {
                (f.destroy_image_view)(d, self.view, core::ptr::null());
            }
        }
        self.gpu
            .destroy_host_buffer(std::mem::take(&mut self.vertices));
        self.gpu
            .destroy_host_buffer(std::mem::take(&mut self.staging));
        // SAFETY: as above; the fd (if exported) closes with its OwnedFd owner.
        unsafe {
            if !self.cmd.is_null() {
                (f.free_command_buffers)(d, self.gpu.pool, 1, &self.cmd);
            }
            if self.release != 0 {
                (f.destroy_semaphore)(d, self.release, core::ptr::null());
            }
            if self.acquire != 0 {
                (f.destroy_semaphore)(d, self.acquire, core::ptr::null());
            }
            if self.fence != 0 {
                (f.destroy_fence)(d, self.fence, core::ptr::null());
            }
            if self.memory != 0 {
                (f.free_memory)(d, self.memory, core::ptr::null());
            }
            if self.image != 0 {
                (f.destroy_image)(d, self.image, core::ptr::null());
            }
        }
    }
}

/// Partially constructed sampled texture, so an error mid-`create_texture`
/// destroys exactly what exists so far (handle 0 = never created). Disarmed with
/// [`core::mem::forget`] once the finished [`Texture`] takes the handles.
struct PendingTexture<'a> {
    gpu: &'a Gpu,
    image: VkImage,
    memory: VkDeviceMemory,
    view: VkImageView,
}

impl Drop for PendingTexture<'_> {
    fn drop(&mut self) {
        let d = self.gpu.device.raw;
        let f = &self.gpu.fns;
        // SAFETY: every non-zero handle was created on this device and is
        // destroyed exactly once; nothing was ever submitted with them.
        unsafe {
            if self.view != 0 {
                (f.destroy_image_view)(d, self.view, core::ptr::null());
            }
            if self.memory != 0 {
                (f.free_memory)(d, self.memory, core::ptr::null());
            }
            if self.image != 0 {
                (f.destroy_image)(d, self.image, core::ptr::null());
            }
        }
    }
}

/// Partially constructed host buffer, mirroring [`PendingTexture`] for
/// `create_host_buffer` (handle 0 = never created).
struct PendingBuffer<'a> {
    gpu: &'a Gpu,
    buf: VkBuffer,
    memory: VkDeviceMemory,
}

impl Drop for PendingBuffer<'_> {
    fn drop(&mut self) {
        let d = self.gpu.device.raw;
        let f = &self.gpu.fns;
        // SAFETY: as for PendingTexture; freeing the memory unmaps it.
        unsafe {
            if self.memory != 0 {
                (f.free_memory)(d, self.memory, core::ptr::null());
            }
            if self.buf != 0 {
                (f.destroy_buffer)(d, self.buf, core::ptr::null());
            }
        }
    }
}

impl Gpu {
    /// Create a `width` x `height` render target, letting the driver pick a
    /// layout from `modifiers` (the compositor∩device intersection), and
    /// export its memory as a dmabuf. Returns the image with the geometry the
    /// wl_buffer needs (chosen modifier, plane stride and offset).
    pub fn create_image(&self, width: u32, height: u32, modifiers: &[u64]) -> Result<GpuImage> {
        if width == 0 || height == 0 {
            return Err(Error::msg("gpu image with a zero dimension"));
        }
        if modifiers.is_empty() {
            return Err(Error::msg("gpu image with no allowed modifiers"));
        }
        let d = self.device.raw;
        let f = &self.fns;
        let mut pending = PendingImage {
            gpu: self,
            image: 0,
            memory: 0,
            fence: 0,
            acquire: 0,
            release: 0,
            cmd: core::ptr::null_mut(),
            view: 0,
            framebuffer: 0,
            vertices: HostBuffer::default(),
            staging: HostBuffer::default(),
        };

        // The image: renderable, clearable, exportable, laid out by one of the
        // allowed modifiers (driver's choice, queried back below). It stores
        // UNORM bytes for the compositor but is mutable so the render pass can
        // drive it through an sRGB view; the format list names both.
        let modifier_list = VkImageDrmFormatModifierListCreateInfoEXT {
            s_type: VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
            p_next: core::ptr::null(),
            drm_format_modifier_count: modifiers.len() as u32,
            p_drm_format_modifiers: modifiers.as_ptr(),
        };
        let external = VkExternalMemoryImageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            p_next: (&modifier_list as *const VkImageDrmFormatModifierListCreateInfoEXT).cast(),
            handle_types: VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT,
        };
        let view_formats = [VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB];
        let format_list = VkImageFormatListCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
            p_next: (&external as *const VkExternalMemoryImageCreateInfo).cast(),
            view_format_count: view_formats.len() as u32,
            p_view_formats: view_formats.as_ptr(),
        };
        let image_info = VkImageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            p_next: (&format_list as *const VkImageFormatListCreateInfo).cast(),
            flags: VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
            image_type: VK_IMAGE_TYPE_2D,
            format: VK_FORMAT_B8G8R8A8_UNORM,
            extent: VkExtent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: VK_SAMPLE_COUNT_1_BIT,
            tiling: VK_IMAGE_TILING_DRM_FORMAT_MODIFIER,
            // Rendered into by the pipeline; read back by the parity harness.
            usage: VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
            initial_layout: VK_IMAGE_LAYOUT_UNDEFINED,
        };
        // SAFETY: d is live; image_info and its chain outlive the call.
        check(
            unsafe { (f.create_image)(d, &image_info, core::ptr::null(), &mut pending.image) },
            "vkCreateImage",
        )?;

        // Dedicated, exportable, device-local memory.
        let mut reqs = VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        // SAFETY: image is live; reqs is a valid out-parameter.
        unsafe { (f.get_image_memory_requirements)(d, pending.image, &mut reqs) };
        let memory_type = self
            .pick_memory_type(reqs.memory_type_bits, 0)
            .ok_or_else(|| Error::msg("no memory type fits the exported image"))?;
        let dedicated = VkMemoryDedicatedAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            image: pending.image,
            buffer: 0,
        };
        let export = VkExportMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            p_next: (&dedicated as *const VkMemoryDedicatedAllocateInfo).cast(),
            handle_types: VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT,
        };
        let alloc = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: (&export as *const VkExportMemoryAllocateInfo).cast(),
            allocation_size: reqs.size,
            memory_type_index: memory_type,
        };
        // SAFETY: alloc and its chain outlive the call.
        check(
            unsafe { (f.allocate_memory)(d, &alloc, core::ptr::null(), &mut pending.memory) },
            "vkAllocateMemory",
        )?;
        // SAFETY: image and memory are live and compatible (requirements above).
        check(
            unsafe { (f.bind_image_memory)(d, pending.image, pending.memory, 0) },
            "vkBindImageMemory",
        )?;

        // What layout did the driver pick? The wl_buffer must describe it.
        let mut chosen = VkImageDrmFormatModifierPropertiesEXT {
            s_type: VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT,
            p_next: core::ptr::null_mut(),
            drm_format_modifier: 0,
        };
        // SAFETY: image is live; chosen is a valid out-parameter.
        check(
            unsafe { (f.get_image_drm_format_modifier_properties)(d, pending.image, &mut chosen) },
            "vkGetImageDrmFormatModifierPropertiesEXT",
        )?;
        let subresource = VkImageSubresource {
            aspect_mask: VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
            mip_level: 0,
            array_layer: 0,
        };
        let mut layout = VkSubresourceLayout {
            offset: 0,
            size: 0,
            row_pitch: 0,
            array_pitch: 0,
            depth_pitch: 0,
        };
        // SAFETY: image is live; the plane-0 aspect is valid for modifier tiling.
        unsafe { (f.get_image_subresource_layout)(d, pending.image, &subresource, &mut layout) };
        let stride = u32::try_from(layout.row_pitch)
            .map_err(|_| Error::msg("image row pitch exceeds the wl_buffer stride range"))?;
        let offset = u32::try_from(layout.offset)
            .map_err(|_| Error::msg("image plane offset exceeds the wl_buffer range"))?;

        // Export the dmabuf.
        let get_fd = VkMemoryGetFdInfoKHR {
            s_type: VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
            p_next: core::ptr::null(),
            memory: pending.memory,
            handle_type: VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT,
        };
        let mut raw_fd: i32 = -1;
        // SAFETY: memory is live; raw_fd receives a fresh fd we own.
        check(
            unsafe { (f.get_memory_fd)(d, &get_fd, &mut raw_fd) },
            "vkGetMemoryFdKHR",
        )?;
        if raw_fd < 0 {
            return Err(Error::msg("vkGetMemoryFdKHR returned an invalid fd"));
        }
        // SAFETY: the driver just handed us ownership of this fd.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        // Per-image submission state. The fence starts signaled so the first
        // frame's wait passes.
        let fence_info = VkFenceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0x1, // VK_FENCE_CREATE_SIGNALED_BIT
        };
        // SAFETY: fence_info outlives the call.
        check(
            unsafe { (f.create_fence)(d, &fence_info, core::ptr::null(), &mut pending.fence) },
            "vkCreateFence",
        )?;
        let plain_sem = VkSemaphoreCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
        };
        // SAFETY: plain_sem outlives the call.
        check(
            unsafe { (f.create_semaphore)(d, &plain_sem, core::ptr::null(), &mut pending.acquire) },
            "vkCreateSemaphore (acquire)",
        )?;
        let export_sem = VkExportSemaphoreCreateInfo {
            s_type: VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
            p_next: core::ptr::null(),
            handle_types: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
        };
        let export_sem_info = VkSemaphoreCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            p_next: (&export_sem as *const VkExportSemaphoreCreateInfo).cast(),
            flags: 0,
        };
        // SAFETY: export_sem_info and its chain outlive the call.
        check(
            unsafe {
                (f.create_semaphore)(d, &export_sem_info, core::ptr::null(), &mut pending.release)
            },
            "vkCreateSemaphore (release)",
        )?;

        let cmd_info = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            command_pool: self.pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        // SAFETY: the pool is live; one handle is written.
        check(
            unsafe { (f.allocate_command_buffers)(d, &cmd_info, &mut pending.cmd) },
            "vkAllocateCommandBuffers",
        )?;

        // The render-pass plumbing: an sRGB view of the image and its
        // framebuffer, so fixed-function blending runs in linear space.
        pending.view = self.create_view(pending.image, VK_FORMAT_B8G8R8A8_SRGB)?;
        let fb_info = VkFramebufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            render_pass: self.renderer.render_pass,
            attachment_count: 1,
            p_attachments: &pending.view,
            width,
            height,
            layers: 1,
        };
        // SAFETY: the render pass and view are live; fb_info outlives the call.
        check(
            unsafe {
                (f.create_framebuffer)(d, &fb_info, core::ptr::null(), &mut pending.framebuffer)
            },
            "vkCreateFramebuffer",
        )?;

        // Host-visible per-slot buffers; both grow on demand once a frame
        // needs more (the fence has been waited by then, so recreation is safe).
        pending.vertices = self.create_host_buffer(64 * 1024, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)?;
        pending.staging = self.create_host_buffer(
            256 * 1024,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        )?;

        // Success: move everything out of the guard.
        let img = GpuImage {
            image: pending.image,
            memory: pending.memory,
            fd,
            cmd: pending.cmd,
            fence: pending.fence,
            acquire: pending.acquire,
            release: pending.release,
            view: pending.view,
            framebuffer: pending.framebuffer,
            vertices: std::mem::take(&mut pending.vertices),
            staging: std::mem::take(&mut pending.staging),
            width,
            height,
            stride,
            offset,
            modifier: chosen.drm_format_modifier,
            used: false,
        };
        core::mem::forget(pending);
        Ok(img)
    }

    /// Tear an image down. The caller ensures no submission is in flight
    /// (waits for release + fence, or [`Gpu::wait_idle`]).
    ///
    /// The finished image holds exactly the handles a [`PendingImage`] guards, so
    /// rebuild one from them and let its `Drop` run the single teardown sequence
    /// (destroying each handle once). The dmabuf `fd` closes when it drops at the
    /// end of this scope.
    pub fn destroy_image(&self, img: GpuImage) {
        let GpuImage {
            image,
            memory,
            fd: _fd,
            cmd,
            fence,
            acquire,
            release,
            view,
            framebuffer,
            vertices,
            staging,
            ..
        } = img;
        drop(PendingImage {
            gpu: self,
            image,
            memory,
            fence,
            acquire,
            release,
            cmd,
            view,
            framebuffer,
            vertices,
            staging,
        });
    }

    /// Upload the frame's textures and vertices and record its command buffer,
    /// after waiting out the image's previous submission. The sync strategy
    /// (implicit bridge or explicit points) is layered on by the caller, which
    /// then submits; on return the command buffer is recorded but not yet in
    /// flight.
    fn prepare_frame(
        &mut self,
        img: &mut GpuImage,
        frame: &FrameData,
        background: [f32; 4],
    ) -> Result<()> {
        let d = self.device.raw;
        let f = &self.fns;

        // The image's previous submission must be complete before its command
        // buffer and buffers are rewritten (the wl_buffer release already
        // gates reuse of the *pixels*; this gates the CPU-visible state).
        // SAFETY: fence is live; the timeout bounds a broken driver.
        check(
            unsafe { (f.wait_for_fences)(d, 1, &img.fence, 1, FENCE_TIMEOUT_NS) },
            "vkWaitForFences",
        )?;
        // SAFETY: the fence is signaled and nothing else references it.
        check(
            unsafe { (f.reset_fences)(d, 1, &img.fence) },
            "vkResetFences",
        )?;

        // Textures first: (re)create atlases and image textures, assemble the
        // staging bytes and the copy list.
        let (staging_bytes, uploads) = self.sync_frame_textures(frame)?;
        if !staging_bytes.is_empty() {
            self.grow_host_buffer(
                &mut img.staging,
                staging_bytes.len(),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            )?;
            // SAFETY: staging is mapped with at least staging_bytes.len()
            // capacity (grown above) and the GPU is done with it (fence).
            unsafe {
                core::ptr::copy_nonoverlapping(
                    staging_bytes.as_ptr(),
                    img.staging.ptr,
                    staging_bytes.len(),
                );
            }
        }

        // Vertices.
        let vbytes = std::mem::size_of_val(frame.vertices.as_slice());
        if vbytes > 0 {
            self.grow_host_buffer(&mut img.vertices, vbytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)?;
            // SAFETY: the vertex buffer is mapped with capacity >= vbytes and
            // idle (fence); Vertex is repr(C) plain data.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    frame.vertices.as_ptr().cast::<u8>(),
                    img.vertices.ptr,
                    vbytes,
                );
            }
        }

        self.record_frame(img, frame, &uploads, background)?;
        Ok(())
    }

    /// Render `img` and present it under the milestone-3 implicit-sync bridge:
    /// acquire from the dmabuf's read fences, release the render fence back into
    /// the dmabuf (CPU wait if the bridge is unavailable). The fallback path for
    /// compositors that do not offer explicit sync. Returns with the frame in
    /// flight; the caller may attach/commit immediately.
    pub fn render_list(
        &mut self,
        img: &mut GpuImage,
        frame: &FrameData,
        background: [f32; 4],
    ) -> Result<()> {
        self.prepare_frame(img, frame, background)?;
        // Acquire: the compositor's unfinished reads become a wait semaphore.
        let wait_acquire = self.import_read_fences(img)?;
        self.submit_frame(img, wait_acquire)
    }

    /// Render `img` and present it under explicit sync: wait on `release_wait`
    /// (the compositor's release point for this buffer's previous frame, `None`
    /// on first use) instead of the dmabuf's implicit fences, and return the
    /// render-done sync file for the caller to publish as this frame's acquire
    /// point. `Ok(None)` means the driver could not export a fence, so the frame
    /// was CPU-waited to completion and the caller must set no points on the
    /// commit (the buffer is already safe to read).
    pub fn render_list_explicit(
        &mut self,
        img: &mut GpuImage,
        frame: &FrameData,
        background: [f32; 4],
        release_wait: Option<OwnedFd>,
    ) -> Result<Option<OwnedFd>> {
        self.prepare_frame(img, frame, background)?;
        let wait_acquire = self.import_acquire(img, release_wait)?;
        self.submit_frame_explicit(img, wait_acquire)
    }

    /// The acquire side of the per-frame ownership dance: where the image is
    /// coming from (undefined on first use, the compositor after).
    fn acquire_params(&self, img: &GpuImage) -> (u32, u32, u32) {
        if img.used {
            (
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_FOREIGN,
                self.queue_family,
            )
        } else {
            (
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
            )
        }
    }

    /// Submit `img`'s recorded command buffer: wait on its acquire semaphore
    /// when `wait_acquire`, signal its release semaphore and fence. Sync out
    /// (bridge or explicit) is layered on by the callers below.
    fn submit_only(&self, img: &mut GpuImage, wait_acquire: bool) -> Result<()> {
        let f = &self.fns;
        let wait_stage: u32 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        let submit = VkSubmitInfo {
            s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: core::ptr::null(),
            wait_semaphore_count: u32::from(wait_acquire),
            p_wait_semaphores: &img.acquire,
            p_wait_dst_stage_mask: &wait_stage,
            command_buffer_count: 1,
            p_command_buffers: &img.cmd,
            signal_semaphore_count: 1,
            p_signal_semaphores: &img.release,
        };
        // SAFETY: queue, cmd, semaphores, and fence are live; submit and its
        // pointees outlive the call.
        check(
            unsafe { (f.queue_submit)(self.queue, 1, &submit, img.fence) },
            "vkQueueSubmit",
        )?;
        img.used = true;
        Ok(())
    }

    /// CPU-wait for `img`'s render fence to signal, the fallback both submit
    /// paths take when the driver cannot hand the fence to the compositor. `what`
    /// names the calling path for the error. A slow correct frame beats a torn one.
    fn wait_fence(&self, img: &GpuImage, what: &str) -> Result<()> {
        // SAFETY: fence is live; the timeout bounds a broken driver.
        check(
            unsafe {
                (self.fns.wait_for_fences)(self.device.raw, 1, &img.fence, 1, FENCE_TIMEOUT_NS)
            },
            what,
        )
    }

    /// Submit under the implicit-sync bridge: after the submit, the render-done
    /// fence rides the dmabuf so the compositor waits for it. If the bridge is
    /// unavailable, wait on the CPU instead.
    fn submit_frame(&self, img: &mut GpuImage, wait_acquire: bool) -> Result<()> {
        self.submit_only(img, wait_acquire)?;
        if !self.export_render_fence(img)? {
            self.wait_fence(img, "vkWaitForFences (bridge fallback)")?;
        }
        Ok(())
    }

    /// Submit under explicit sync: after the submit, hand back the render-done
    /// sync file so the caller can publish it as this frame's acquire point. If
    /// the driver cannot export one, CPU-wait to completion and return `None` so
    /// the caller commits with no sync points.
    fn submit_frame_explicit(
        &self,
        img: &mut GpuImage,
        wait_acquire: bool,
    ) -> Result<Option<OwnedFd>> {
        self.submit_only(img, wait_acquire)?;
        match self.render_done_fence(img)? {
            Some(fd) => Ok(Some(fd)),
            None => {
                self.wait_fence(img, "vkWaitForFences (explicit fallback)")?;
                Ok(None)
            }
        }
    }

    /// Import `sync` (a sync file) as a temporary payload of the acquire
    /// semaphore so the submission waits on it; `None` arms no wait. Returns
    /// whether a wait was armed. Consumes `sync` (Vulkan owns it on success).
    fn import_acquire(&self, img: &GpuImage, sync: Option<OwnedFd>) -> Result<bool> {
        let Some(sync) = sync else {
            return Ok(false);
        };
        let raw = sync.as_raw_fd();
        let info = VkImportSemaphoreFdInfoKHR {
            s_type: VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
            p_next: core::ptr::null(),
            semaphore: img.acquire,
            flags: VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
            handle_type: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
            fd: raw,
        };
        // SAFETY: info outlives the call; on success Vulkan owns the fd.
        let r = unsafe { (self.fns.import_semaphore_fd)(self.device.raw, &info) };
        if r == VK_SUCCESS {
            // Ownership moved to Vulkan; do not let OwnedFd close it.
            core::mem::forget(sync);
            Ok(true)
        } else {
            // Failed import: we keep ownership (sync drops); skip the wait.
            Ok(false)
        }
    }

    /// The implicit-sync bridge's acquire: the dmabuf's current read/write
    /// fences become the acquire wait.
    fn import_read_fences(&self, img: &GpuImage) -> Result<bool> {
        self.import_acquire(
            img,
            crate::platform::ffi::dmabuf_export_sync_file(img.fd())?,
        )
    }

    /// Export the just-submitted release semaphore as a sync file. `None` when
    /// the driver cannot, in which case the caller must CPU-wait.
    fn render_done_fence(&self, img: &GpuImage) -> Result<Option<OwnedFd>> {
        let info = VkSemaphoreGetFdInfoKHR {
            s_type: VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
            p_next: core::ptr::null(),
            semaphore: img.release,
            handle_type: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
        };
        let mut raw: i32 = -1;
        // SAFETY: the semaphore has a pending signal from the submit above,
        // which is what a sync-fd export requires.
        let r = unsafe { (self.fns.get_semaphore_fd)(self.device.raw, &info, &mut raw) };
        if r != VK_SUCCESS || raw < 0 {
            return Ok(None);
        }
        // SAFETY: the driver handed us ownership of this sync-file fd.
        Ok(Some(unsafe { OwnedFd::from_raw_fd(raw) }))
    }

    /// The implicit-sync bridge's release: attach the render-done fence to the
    /// dmabuf as its write fence so the compositor waits for it. Returns false
    /// when the kernel or driver cannot bridge.
    fn export_render_fence(&self, img: &GpuImage) -> Result<bool> {
        let Some(sync) = self.render_done_fence(img)? else {
            return Ok(false);
        };
        crate::platform::ffi::dmabuf_import_sync_file(img.fd(), sync.as_raw_fd())
    }

    /// Record `img`'s command buffer for one frame: texture uploads, the
    /// acquire barrier, the render pass (clear plus batched quads), and the
    /// release back to the compositor in GENERAL layout.
    fn record_frame(
        &self,
        img: &GpuImage,
        frame: &FrameData,
        uploads: &[PendingUpload],
        background: [f32; 4],
    ) -> Result<()> {
        let f = &self.fns;
        let begin = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        // SAFETY: the pool allows reset-on-begin; the previous submission
        // completed (fence wait in render_list).
        check(
            unsafe { (f.begin_command_buffer)(img.cmd, &begin) },
            "vkBeginCommandBuffer",
        )?;

        for up in uploads {
            self.record_upload(img, up);
        }

        let range = color_range();
        let (old_layout, src_qf, dst_qf) = self.acquire_params(img);
        // SAFETY (all cmd_* to the end): cmd is in the recording state; every
        // struct outlives its call; handles are live.
        unsafe {
            let acquire = VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: 0,
                dst_access_mask: VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                old_layout,
                new_layout: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                src_queue_family_index: src_qf,
                dst_queue_family_index: dst_qf,
                image: img.image,
                subresource_range: range,
            };
            (f.cmd_pipeline_barrier)(
                img.cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &acquire,
            );

            let clear = VkClearColorValue {
                float32: background,
            };
            let pass = VkRenderPassBeginInfo {
                s_type: VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                p_next: core::ptr::null(),
                render_pass: self.renderer.render_pass,
                framebuffer: img.framebuffer,
                render_area: VkRect2D {
                    offset: VkOffset2D { x: 0, y: 0 },
                    extent: VkExtent2D {
                        width: img.width,
                        height: img.height,
                    },
                },
                clear_value_count: 1,
                p_clear_values: &clear,
            };
            (f.cmd_begin_render_pass)(img.cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);

            if !frame.vertices.is_empty() {
                (f.cmd_bind_pipeline)(
                    img.cmd,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    self.renderer.pipeline,
                );
                let viewport = VkViewport {
                    x: 0.0,
                    y: 0.0,
                    width: img.width as f32,
                    height: img.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                (f.cmd_set_viewport)(img.cmd, 0, 1, &viewport);
                let scissor = VkRect2D {
                    offset: VkOffset2D { x: 0, y: 0 },
                    extent: VkExtent2D {
                        width: img.width,
                        height: img.height,
                    },
                };
                (f.cmd_set_scissor)(img.cmd, 0, 1, &scissor);
                let offset = 0u64;
                (f.cmd_bind_vertex_buffers)(img.cmd, 0, 1, &img.vertices.buf, &offset);
                // Viewport (vertex) then glyph mask gamma (fragment), packed at the
                // offsets the shaders' shared push block names.
                let push: [f32; 3] = [
                    img.width as f32,
                    img.height as f32,
                    self.renderer.glyph_coverage_gamma,
                ];
                (f.cmd_push_constants)(
                    img.cmd,
                    self.renderer.pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    12,
                    push.as_ptr().cast(),
                );
                // The atlas set (set 0) is the only descriptor set now; bind it
                // once, then draw every batch (a frame is a single batch, but the
                // loop is kept so a future clipped-repaint path stays a drop-in).
                (f.cmd_bind_descriptor_sets)(
                    img.cmd,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    self.renderer.pipeline_layout,
                    0,
                    1,
                    &self.renderer.set_atlas,
                    0,
                    core::ptr::null(),
                );
                for batch in &frame.batches {
                    (f.cmd_draw)(img.cmd, batch.count, 1, batch.start, 0);
                }
            }

            (f.cmd_end_render_pass)(img.cmd);

            let release = VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dst_access_mask: 0,
                old_layout: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                new_layout: VK_IMAGE_LAYOUT_GENERAL,
                src_queue_family_index: self.queue_family,
                dst_queue_family_index: VK_QUEUE_FAMILY_FOREIGN,
                image: img.image,
                subresource_range: range,
            };
            (f.cmd_pipeline_barrier)(
                img.cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &release,
            );
        }
        // SAFETY: cmd is in the recording state.
        check(
            unsafe { (f.end_command_buffer)(img.cmd) },
            "vkEndCommandBuffer",
        )
    }

    /// Record one texture upload (or a bare initialization barrier when the
    /// upload carries no bytes, the dummy texture's case).
    fn record_upload(&self, img: &GpuImage, up: &PendingUpload) {
        let f = &self.fns;
        let range = color_range();
        let (old_layout, src_stage, src_access) = if up.was_initialized {
            (
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT,
            )
        } else {
            (
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                0,
            )
        };
        // SAFETY (whole fn): cmd is recording; structs outlive their calls.
        unsafe {
            let to_dst = VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: src_access,
                dst_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT,
                old_layout,
                new_layout: VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                image: up.image,
                subresource_range: range,
            };
            (f.cmd_pipeline_barrier)(
                img.cmd,
                src_stage,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &to_dst,
            );
            if up.w > 0 && up.h > 0 {
                let region = VkBufferImageCopy {
                    buffer_offset: up.staging_offset as u64,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: VkImageSubresourceLayers {
                        aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: VkOffset3D {
                        x: up.x as i32,
                        y: up.y as i32,
                        z: 0,
                    },
                    image_extent: VkExtent3D {
                        width: up.w,
                        height: up.h,
                        depth: 1,
                    },
                };
                (f.cmd_copy_buffer_to_image)(
                    img.cmd,
                    img.staging.buf,
                    up.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &region,
                );
            }
            let to_read = VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_SHADER_READ_BIT,
                old_layout: VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                new_layout: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                image: up.image,
                subresource_range: range,
            };
            (f.cmd_pipeline_barrier)(
                img.cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &to_read,
            );
        }
    }

    /// Reconcile the renderer's textures with this frame's data: (re)create
    /// atlas textures on generation changes, create textures and descriptor
    /// sets for new images, and assemble the staging bytes plus copy list.
    fn sync_frame_textures(&mut self, frame: &FrameData) -> Result<(Vec<u8>, Vec<PendingUpload>)> {
        let mut staging = Vec::new();
        let mut uploads = Vec::new();

        // The dummy texture: 1x1, never written, stands in for an atlas binding
        // in set 0 before that atlas texture exists (see `update_atlas_set`).
        // Contents are irrelevant, only the layout must be right, so its
        // "upload" is a bare barrier.
        if self.renderer.dummy.image == 0 {
            let dummy = self.create_texture(VK_FORMAT_B8G8R8A8_UNORM, 1, 1)?;
            uploads.push(PendingUpload {
                image: dummy.image,
                was_initialized: false,
                staging_offset: 0,
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            });
            self.renderer.dummy = dummy;
        }

        // Atlas textures follow the CPU mirrors' generations, then take any
        // dirty-region upload for the frame.
        self.sync_atlas(AtlasSlot::Glyph, &frame.glyph_atlas)?;
        self.sync_atlas(AtlasSlot::Emoji, &frame.emoji_atlas)?;
        self.stage_atlas_upload(
            AtlasSlot::Glyph,
            &frame.glyph_upload,
            &mut staging,
            &mut uploads,
        );
        self.stage_atlas_upload(
            AtlasSlot::Emoji,
            &frame.emoji_upload,
            &mut staging,
            &mut uploads,
        );

        // Uploads whose staging_offset was pushed before later bytes landed
        // stay valid: offsets only ever append.
        Ok((staging, uploads))
    }

    /// Rebuild an atlas texture when its CPU mirror's generation moved (or it
    /// does not exist yet). A replacement may still be sampled by the other
    /// slot's in-flight frame, so the queue is drained before the old texture is
    /// destroyed, and the shared atlas descriptor set is repointed after the new
    /// one exists. Shared by the glyph and emoji slots so that ordering lives in
    /// one place.
    fn sync_atlas(&mut self, slot: AtlasSlot, info: &AtlasInfo) -> Result<()> {
        let tex = match slot {
            AtlasSlot::Glyph => &mut self.renderer.glyph_tex,
            AtlasSlot::Emoji => &mut self.renderer.emoji_tex,
        };
        if tex.as_ref().is_some_and(|(_, g)| *g == info.generation) {
            return Ok(());
        }
        if let Some((old, _)) = tex.take() {
            self.wait_idle();
            destroy_texture(&self.fns, self.device.raw, old);
        }
        let tex = self.create_texture(slot.format(), info.width, info.height)?;
        match slot {
            AtlasSlot::Glyph => self.renderer.glyph_tex = Some((tex, info.generation)),
            AtlasSlot::Emoji => self.renderer.emoji_tex = Some((tex, info.generation)),
        }
        self.update_atlas_set()
    }

    /// Queue an atlas's dirty region for upload into its live texture, appending
    /// the bytes to `staging`. A no-op when the frame carried no change for that
    /// slot or the texture does not exist yet.
    fn stage_atlas_upload(
        &mut self,
        slot: AtlasSlot,
        upload: &Option<AtlasUpload>,
        staging: &mut Vec<u8>,
        uploads: &mut Vec<PendingUpload>,
    ) {
        let Some(up) = upload else {
            return;
        };
        let tex = match slot {
            AtlasSlot::Glyph => self.renderer.glyph_tex.as_mut(),
            AtlasSlot::Emoji => self.renderer.emoji_tex.as_mut(),
        };
        if let Some((tex, _)) = tex {
            uploads.push(PendingUpload {
                image: tex.image,
                was_initialized: tex.initialized,
                staging_offset: staging.len(),
                x: up.x,
                y: up.y,
                w: up.w,
                h: up.h,
            });
            staging.extend_from_slice(&up.bytes);
            tex.initialized = true;
        }
    }

    /// Point the shared atlas descriptor set at the current atlas textures
    /// (both bindings, using the dummy for one that does not exist yet).
    fn update_atlas_set(&self) -> Result<()> {
        let glyph_view = self
            .renderer
            .glyph_tex
            .as_ref()
            .map(|(t, _)| t.view)
            .unwrap_or(self.renderer.dummy.view);
        let emoji_view = self
            .renderer
            .emoji_tex
            .as_ref()
            .map(|(t, _)| t.view)
            .unwrap_or(self.renderer.dummy.view);
        if glyph_view == 0 || emoji_view == 0 {
            return Err(Error::msg("atlas descriptor update before any texture"));
        }
        write_image_set(
            &self.fns,
            self.device.raw,
            self.renderer.set_atlas,
            0,
            self.renderer.sampler,
            glyph_view,
        );
        write_image_set(
            &self.fns,
            self.device.raw,
            self.renderer.set_atlas,
            1,
            self.renderer.sampler,
            emoji_view,
        );
        Ok(())
    }

    /// A sampled, transfer-destination 2D texture (an atlas or the dummy).
    fn create_texture(&self, format: u32, width: u32, height: u32) -> Result<Texture> {
        let d = self.device.raw;
        let f = &self.fns;
        let info = VkImageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            image_type: VK_IMAGE_TYPE_2D,
            format,
            extent: VkExtent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: VK_SAMPLE_COUNT_1_BIT,
            tiling: VK_IMAGE_TILING_OPTIMAL,
            usage: VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
            initial_layout: VK_IMAGE_LAYOUT_UNDEFINED,
        };
        // The guard destroys whatever exists so far if any step below fails.
        let mut pending = PendingTexture {
            gpu: self,
            image: 0,
            memory: 0,
            view: 0,
        };
        // SAFETY: info outlives the call.
        check(
            unsafe { (f.create_image)(d, &info, core::ptr::null(), &mut pending.image) },
            "vkCreateImage (texture)",
        )?;
        let mut reqs = VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        // SAFETY: image is live.
        unsafe { (f.get_image_memory_requirements)(d, pending.image, &mut reqs) };
        let memory_type = self
            .pick_memory_type(reqs.memory_type_bits, 0)
            .ok_or_else(|| Error::msg("no memory type fits a texture"))?;
        let alloc = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            allocation_size: reqs.size,
            memory_type_index: memory_type,
        };
        // SAFETY: alloc outlives the call.
        check(
            unsafe { (f.allocate_memory)(d, &alloc, core::ptr::null(), &mut pending.memory) },
            "vkAllocateMemory (texture)",
        )?;
        // SAFETY: image and memory are live and compatible.
        check(
            unsafe { (f.bind_image_memory)(d, pending.image, pending.memory, 0) },
            "vkBindImageMemory (texture)",
        )?;
        pending.view = self.create_view(pending.image, format)?;
        // Success: hand the handles to the Texture and disarm the guard.
        let texture = Texture {
            image: pending.image,
            memory: pending.memory,
            view: pending.view,
            initialized: false,
        };
        core::mem::forget(pending);
        Ok(texture)
    }

    /// A plain 2D color view of `image`.
    fn create_view(&self, image: VkImage, format: u32) -> Result<VkImageView> {
        let info = VkImageViewCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            image,
            view_type: VK_IMAGE_VIEW_TYPE_2D,
            format,
            components: VkComponentMapping {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            },
            subresource_range: color_range(),
        };
        let mut view: VkImageView = 0;
        // SAFETY: image is live; info outlives the call.
        check(
            unsafe {
                (self.fns.create_image_view)(self.device.raw, &info, core::ptr::null(), &mut view)
            },
            "vkCreateImageView",
        )?;
        Ok(view)
    }

    /// A host-visible, host-coherent, persistently mapped buffer.
    fn create_host_buffer(&self, size: usize, usage: u32) -> Result<HostBuffer> {
        let d = self.device.raw;
        let f = &self.fns;
        let info = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            size: size as u64,
            usage,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
        };
        // The guard destroys whatever exists so far if any step below fails.
        let mut pending = PendingBuffer {
            gpu: self,
            buf: 0,
            memory: 0,
        };
        // SAFETY: info outlives the call.
        check(
            unsafe { (f.create_buffer)(d, &info, core::ptr::null(), &mut pending.buf) },
            "vkCreateBuffer",
        )?;
        let mut reqs = VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        // SAFETY: buf is live.
        unsafe { (f.get_buffer_memory_requirements)(d, pending.buf, &mut reqs) };
        let required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        let memory_type = self
            .pick_memory_type(reqs.memory_type_bits, required)
            .ok_or_else(|| Error::msg("no host-visible memory type for a buffer"))?;
        let alloc = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            allocation_size: reqs.size,
            memory_type_index: memory_type,
        };
        // SAFETY: alloc outlives the call.
        check(
            unsafe { (f.allocate_memory)(d, &alloc, core::ptr::null(), &mut pending.memory) },
            "vkAllocateMemory (buffer)",
        )?;
        // SAFETY: buf and memory are live and compatible.
        let r = unsafe { (f.bind_buffer_memory)(d, pending.buf, pending.memory, 0) };
        let mut ptr: *mut c_void = core::ptr::null_mut();
        // SAFETY: memory is live, host-visible, and bound (checked below).
        let r2 = if r == VK_SUCCESS {
            unsafe { (f.map_memory)(d, pending.memory, 0, u64::MAX, 0, &mut ptr) }
        } else {
            r
        };
        if r2 != VK_SUCCESS || ptr.is_null() {
            return Err(Error::msg(format!("buffer bind/map failed: VkResult {r2}")));
        }
        // Success: hand the handles to the HostBuffer and disarm the guard.
        let buffer = HostBuffer {
            buf: pending.buf,
            memory: pending.memory,
            ptr: ptr.cast(),
            capacity: size,
        };
        core::mem::forget(pending);
        Ok(buffer)
    }

    /// Free a host buffer (a zeroed default is a no-op).
    fn destroy_host_buffer(&self, b: HostBuffer) {
        if b.buf == 0 {
            return;
        }
        // SAFETY: buf/memory are live and no submission references them (the
        // caller waited the slot's fence or the queue). Freeing unmaps.
        unsafe {
            (self.fns.destroy_buffer)(self.device.raw, b.buf, core::ptr::null());
            (self.fns.free_memory)(self.device.raw, b.memory, core::ptr::null());
        }
    }

    /// Grow `b` to at least `needed` bytes (doubling), preserving nothing:
    /// callers rewrite the whole buffer each frame.
    fn grow_host_buffer(&self, b: &mut HostBuffer, needed: usize, usage: u32) -> Result<()> {
        if needed <= b.capacity {
            return Ok(());
        }
        let new_cap = needed.max(b.capacity * 2);
        let fresh = self.create_host_buffer(new_cap, usage)?;
        let old = std::mem::replace(b, fresh);
        self.destroy_host_buffer(old);
        Ok(())
    }

    /// The first allowed memory type carrying every `required` property flag;
    /// with no requirement, prefer device-local and fall back to any allowed.
    fn pick_memory_type(&self, allowed_bits: u32, required: u32) -> Option<u32> {
        let allowed = |i: usize| allowed_bits & (1u32 << i) != 0;
        if required != 0 {
            return self
                .memory_type_flags
                .iter()
                .enumerate()
                .position(|(i, &flags)| allowed(i) && flags & required == required)
                .map(|i| i as u32);
        }
        self.memory_type_flags
            .iter()
            .enumerate()
            .position(|(i, &flags)| allowed(i) && flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0)
            .or_else(|| (0..self.memory_type_flags.len()).find(|&i| allowed(i)))
            .map(|i| i as u32)
    }
}

/// One texture copy scheduled for this frame's command buffer: which image,
/// where its bytes sit in the staging buffer, and the destination region.
/// A zero-sized region is a bare layout-initialization barrier.
struct PendingUpload {
    image: VkImage,
    was_initialized: bool,
    staging_offset: usize,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}
