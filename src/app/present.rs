//! The GPU/dmabuf presentation half of the app: negotiating the buffers, sync,
//! and per-frame submit that turn a [`DisplayList`] into pixels on the
//! compositor's surface. This is bnkterm's image-free render path (a terminal
//! decodes no images, so the atlas, batch, and Vulkan pipeline carry glyphs and
//! solid quads only).
//!
//! ```text
//!   DisplayList ─gpu::build_frame─▶ FrameData ─Gpu::render_list─▶ dmabuf ─attach─▶ compositor
//!                                              (+ explicit sync when offered)
//! ```
//!
//! The two exported swap buffers double-buffer the surface so the compositor
//! never reads a frame we are still painting; explicit sync (`linux-drm-syncobj`)
//! is used when the compositor advertises it, else the implicit dmabuf-fence
//! bridge in the Vulkan backend carries ordering.

use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::rc::Rc;
use std::time::Instant;

use super::State;
// Crate-level error type throughout (see the note in `app.rs`); `?` on a
// platform/render call converts via the `From` bridge in `crate::error`.
use crate::error::{Error, Result};
use crate::platform::conn::Connection;
use crate::platform::dmabuf;
use crate::platform::ffi;
use crate::platform::protocol::{
    self, wl_buffer, wl_display, wl_surface, wp_linux_drm_syncobj_manager_v1,
    wp_linux_drm_syncobj_surface_v1, wp_viewport, xdg_surface, zwp_linux_buffer_params_v1,
    zwp_linux_dmabuf_feedback_v1, zwp_linux_dmabuf_v1,
};
use crate::platform::wire::{Arg, Reader};
use crate::render::display::{self, DisplayList};
use crate::render::gpu;
use crate::render::vulkan;

/// Client-owned explicit-sync state (`linux-drm-syncobj-v1`), negotiated over a
/// live GPU backend when the compositor advertises the manager. Two DRM syncobj
/// timelines the compositor shares: we signal `acquire` with each frame's
/// render-done fence, and it signals `release` when done reading a buffer. A
/// scratch binary syncobj carries the sync-file transfers. `point` increments
/// per present; `buffer_release[i]` is the release point last published for
/// buffer slot `i`, so that slot's next reuse waits on it. Dropping this closes
/// the render-node fd, which frees every syncobj created on it.
pub(super) struct ExplicitSync {
    drm: OwnedFd,
    acquire_syncobj: u32,
    release_syncobj: u32,
    scratch: u32,
    acquire_timeline: u32,
    release_timeline: u32,
    surface: u32,
    point: u64,
    buffer_release: [Option<u64>; 2],
}

impl ExplicitSync {
    /// The sync file to wait on before rendering into buffer `idx`: the
    /// compositor's release of that buffer's previous frame, or `None` the first
    /// time the slot is used (nothing has read it yet).
    fn release_fence(&self, idx: usize) -> Result<Option<OwnedFd>> {
        match self.buffer_release[idx] {
            Some(p) => Ok(Some(ffi::drm_syncobj_point_to_sync_file(
                self.drm.as_raw_fd(),
                self.release_syncobj,
                p,
                self.scratch,
            )?)),
            None => Ok(None),
        }
    }

    /// Publish this frame's render-done `fence` as buffer `idx`'s acquire point
    /// and claim the matching release point on a fresh timeline value, so the
    /// compositor waits for the GPU before reading and signals the point when it
    /// is done (the buffer's next reuse waits on that via [`Self::release_fence`]).
    fn publish(&mut self, conn: &mut Connection, fence: &OwnedFd, idx: usize) -> Result<()> {
        self.point += 1;
        let p = self.point;
        ffi::drm_syncobj_sync_file_to_point(
            self.drm.as_raw_fd(),
            fence.as_raw_fd(),
            self.acquire_syncobj,
            p,
            self.scratch,
        )?;
        self.buffer_release[idx] = Some(p);
        let (hi, lo) = ((p >> 32) as u32, p as u32);
        conn.request(
            self.surface,
            wp_linux_drm_syncobj_surface_v1::SET_ACQUIRE_POINT,
            &[
                Arg::Object(self.acquire_timeline),
                Arg::Uint(hi),
                Arg::Uint(lo),
            ],
        );
        conn.request(
            self.surface,
            wp_linux_drm_syncobj_surface_v1::SET_RELEASE_POINT,
            &[
                Arg::Object(self.release_timeline),
                Arg::Uint(hi),
                Arg::Uint(lo),
            ],
        );
        Ok(())
    }

    /// Forget every buffer's published release point (the buffers they named are
    /// gone, e.g. after a resize), so the next use of each slot waits on nothing.
    fn forget_buffers(&mut self) {
        self.buffer_release = [None, None];
    }
}

/// GPU/dmabuf presentation resources owned by the app: protocol globals and
/// feedback, the Vulkan backend, the two exported swap buffers, explicit-sync
/// state, and the small counters the stats line reports.
pub(super) struct GpuPresentation {
    pub(super) dmabuf: Option<u32>,
    pub(super) dmabuf_version: u32,
    pub(super) feedback_id: u32,
    pub(super) feedback: dmabuf::FeedbackState,
    pub(super) backend: Option<vulkan::Gpu>,
    pub(super) images: Vec<vulkan::GpuImage>,
    pub(super) modifiers: Vec<u64>,
    pub(super) glyphs: gpu::GlyphCache,
    pub(super) syncobj_manager: Option<u32>,
    pub(super) explicit_sync: Option<ExplicitSync>,
    pub(super) buffers: [u32; 2],
    pub(super) busy: [bool; 2],
    pub(super) buffer_size: (u32, u32),
    pub(super) present_list: Rc<DisplayList>,
    pub(super) frame_count: u64,
    pub(super) last_present: Option<Instant>,
    pub(super) explicit_fence_frames: u64,
    pub(super) explicit_cpu_wait_frames: u64,
}

impl GpuPresentation {
    pub(super) fn new() -> Self {
        Self {
            dmabuf: None,
            dmabuf_version: 0,
            feedback_id: 0,
            feedback: dmabuf::FeedbackState::default(),
            backend: None,
            images: Vec::new(),
            modifiers: Vec::new(),
            glyphs: gpu::GlyphCache::new(),
            syncobj_manager: None,
            explicit_sync: None,
            buffers: [0, 0],
            busy: [false, false],
            buffer_size: (0, 0),
            present_list: Rc::default(),
            frame_count: 0,
            last_present: None,
            explicit_fence_frames: 0,
            explicit_cpu_wait_frames: 0,
        }
    }
}

impl State {
    /// Bring the Vulkan backend up from the committed dmabuf feedback: a device
    /// matching the compositor's, and a modifier list that is their common
    /// ground. Called before `create_buffers`, which allocates the exportable GPU
    /// images. A failure here is fatal (the terminal requires a GPU).
    pub(super) fn init_gpu(&mut self) -> Result<()> {
        if self.presentation.dmabuf.is_none() {
            return Err(Error::msg("compositor lacks zwp_linux_dmabuf_v1 v4+"));
        }
        let fb = self
            .presentation
            .feedback
            .ready()
            .ok_or_else(|| Error::msg("dmabuf feedback never committed"))?;
        let main_device = fb.main_device;
        let compositor_mods = fb.modifiers_for(dmabuf::DRM_FORMAT_XRGB8888);
        if compositor_mods.is_empty() {
            return Err(Error::msg("compositor does not accept XRGB8888 dmabufs"));
        }
        let gpu = vulkan::Gpu::new(main_device, crate::app::config_text_gamma())?;
        let mods = gpu.image_modifiers(&compositor_mods);
        if mods.is_empty() {
            return Err(Error::msg(
                "no common DRM modifier between compositor and device",
            ));
        }
        self.presentation.modifiers = mods;
        self.presentation.backend = Some(gpu);
        Ok(())
    }

    /// Negotiate explicit sync (`linux-drm-syncobj-v1`) when a GPU backend is
    /// live and the compositor advertises the manager. Best-effort: on any
    /// failure the GPU path keeps the implicit-sync bridge, which is why this
    /// returns nothing and only logs under stats.
    pub(super) fn init_explicit_sync(&mut self) {
        let Some(manager) = self.presentation.syncobj_manager else {
            return;
        };
        let Some(minor) = self
            .presentation
            .backend
            .as_ref()
            .and_then(vulkan::Gpu::render_node)
        else {
            return; // No GPU backend, or the device exposes no render node.
        };
        match self.negotiate_explicit_sync(manager, minor) {
            Ok(es) => self.presentation.explicit_sync = Some(es),
            Err(e) => {
                if self.stats {
                    eprintln!("bnkterm: explicit sync unavailable, using the sync bridge: {e}");
                }
            }
        }
    }

    /// Create the two client-owned timelines plus a scratch syncobj, share the
    /// timelines with the compositor, and attach explicit sync to the surface.
    pub(super) fn negotiate_explicit_sync(
        &mut self,
        manager: u32,
        minor: u32,
    ) -> Result<ExplicitSync> {
        let drm = ffi::open_drm_render_node(minor)?;
        let fd = drm.as_raw_fd();
        let acquire_syncobj = ffi::drm_syncobj_create(fd, false)?;
        let release_syncobj = ffi::drm_syncobj_create(fd, false)?;
        let scratch = ffi::drm_syncobj_create(fd, false)?;
        let acquire_timeline = self.import_timeline(manager, fd, acquire_syncobj)?;
        let release_timeline = self.import_timeline(manager, fd, release_syncobj)?;
        let surface = self.alloc_id();
        self.conn.request(
            manager,
            wp_linux_drm_syncobj_manager_v1::GET_SURFACE,
            &[Arg::NewId(surface), Arg::Object(self.surface)],
        );
        Ok(ExplicitSync {
            drm,
            acquire_syncobj,
            release_syncobj,
            scratch,
            acquire_timeline,
            release_timeline,
            surface,
            point: 0,
            buffer_release: [None, None],
        })
    }

    /// Export `syncobj` as a DRM syncobj fd and import it into the compositor as
    /// a timeline object; returns the protocol object id. The fd is sent inline,
    /// so the local `OwnedFd` can drop on return.
    pub(super) fn import_timeline(
        &mut self,
        manager: u32,
        drm: RawFd,
        syncobj: u32,
    ) -> Result<u32> {
        let fd = ffi::drm_syncobj_export_fd(drm, syncobj)?;
        let id = self.alloc_id();
        self.conn.request_with_fd(
            manager,
            wp_linux_drm_syncobj_manager_v1::IMPORT_TIMELINE,
            &[Arg::NewId(id)],
            fd.as_raw_fd(),
        )?;
        Ok(id)
    }

    /// Allocate the two presentable buffers at the current size: one exported GPU
    /// image per swap slot, each wrapped in a wl_buffer via
    /// `zwp_linux_buffer_params_v1` (`create_immed` is sound because every
    /// parameter comes from the compositor's own feedback). A failure (a resize
    /// on a dying device) propagates and ends the process; there is no software
    /// path to fall back to.
    pub(super) fn create_buffers(&mut self) -> Result<()> {
        let dmabuf_global = self
            .presentation
            .dmabuf
            .ok_or_else(|| Error::msg("gpu buffers without zwp_linux_dmabuf_v1"))?;
        let (w, h) = (self.width, self.height);
        let mut images = Vec::with_capacity(2);
        {
            let Some(gpu) = self.presentation.backend.as_ref() else {
                return Err(Error::msg("gpu buffers without a gpu backend"));
            };
            for _ in 0..2 {
                images.push(gpu.create_image(w, h, &self.presentation.modifiers)?);
            }
        }
        for (i, img) in images.iter().enumerate() {
            let params = self.alloc_id();
            self.conn.request(
                dmabuf_global,
                zwp_linux_dmabuf_v1::CREATE_PARAMS,
                &[Arg::NewId(params)],
            );
            let modifier = img.modifier();
            self.conn.request_with_fd(
                params,
                zwp_linux_buffer_params_v1::ADD,
                &[
                    Arg::Uint(0), // plane index
                    Arg::Uint(img.offset()),
                    Arg::Uint(img.stride()),
                    Arg::Uint((modifier >> 32) as u32),
                    Arg::Uint(modifier as u32),
                ],
                img.fd(),
            )?;
            let buffer = self.alloc_id();
            self.conn.request(
                params,
                zwp_linux_buffer_params_v1::CREATE_IMMED,
                &[
                    Arg::NewId(buffer),
                    Arg::Int(w as i32),
                    Arg::Int(h as i32),
                    Arg::Uint(dmabuf::DRM_FORMAT_XRGB8888),
                    Arg::Uint(0),
                ],
            );
            self.conn
                .request(params, zwp_linux_buffer_params_v1::DESTROY, &[]);
            self.presentation.buffers[i] = buffer;
        }
        self.presentation.images = images;
        self.presentation.buffer_size = (self.width, self.height);
        self.presentation.busy = [false, false];
        // Fresh images never carried a release point, so a resize must forget the
        // old buffers' points; their next use starts with no wait.
        if let Some(es) = self.presentation.explicit_sync.as_mut() {
            es.forget_buffers();
        }
        // Fresh images hold no known frame, so forget what was on screen: the
        // screen diff gates whether the next frame presents at all.
        self.presentation.present_list = Rc::default();
        Ok(())
    }

    /// Destroy the current buffers and their backing GPU images. The server
    /// recycles the wl_buffer ids via `delete_id` into our free list.
    pub(super) fn destroy_buffers(&mut self) {
        let buffers = self.presentation.buffers;
        for b in buffers {
            if b != 0 {
                self.conn.request(b, wl_buffer::DESTROY, &[]);
            }
        }
        if let Some(gpu) = &self.presentation.backend {
            // Nothing may be in flight while images are destroyed; resize is rare
            // enough that a full drain is the simple correct answer.
            gpu.wait_idle();
            for img in self.presentation.images.drain(..) {
                gpu.destroy_image(img);
            }
        }
        self.presentation.buffers = [0, 0];
        self.presentation.busy = [false, false];
    }

    /// Declare the window geometry, then ack configure `serial`. The geometry is
    /// the whole surface (no client-side decorations), and being double-buffered
    /// surface state it is what the compositor anchors a resize's fixed edge to;
    /// sending it in the same commit as the ack and the matching buffer gives the
    /// compositor one atomic, consistent size for the frame, so the anchored edge
    /// does not drift. Caller commits after this.
    pub(super) fn ack_configure(&mut self, serial: u32) {
        self.declare_surface_scale();
        self.conn.request(
            self.xdg_surface,
            xdg_surface::ACK_CONFIGURE,
            &[Arg::Uint(serial)],
        );
    }

    /// (Re)declare the surface's logical geometry and how the device-pixel buffer
    /// maps onto it: a viewport destination on the fractional path, or an integer
    /// buffer scale on the fallback. The geometry is the whole logical surface (no
    /// client-side decorations); sending it with the ack and the matching buffer
    /// gives the compositor one atomic size for the frame, so an anchored resize
    /// edge does not drift. Idempotent surface state; marks the scale synced.
    pub(super) fn declare_surface_scale(&mut self) {
        let (lw, lh) = (self.scale.logical.0 as i32, self.scale.logical.1 as i32);
        self.conn.request(
            self.xdg_surface,
            xdg_surface::SET_WINDOW_GEOMETRY,
            &[Arg::Int(0), Arg::Int(0), Arg::Int(lw), Arg::Int(lh)],
        );
        if self.scale.is_fractional() {
            self.conn.request(
                self.scale.viewport,
                wp_viewport::SET_DESTINATION,
                &[Arg::Int(lw), Arg::Int(lh)],
            );
        } else {
            let n = (self.scale.factor_120 / 120).max(1) as i32;
            self.conn
                .request(self.surface, wl_surface::SET_BUFFER_SCALE, &[Arg::Int(n)]);
        }
        self.scale.synced = true;
    }

    /// Build the current frame and present it on the GPU. Returns false if both
    /// buffers are still held by the compositor, leaving `dirty` set so the next
    /// buffer release retries.
    ///
    /// The display list is built and diffed against `present_list` (what is on
    /// screen): an identical frame early-outs (the common idle case), otherwise
    /// the changed screen regions are reported to the compositor as damage. The
    /// GPU repaints the full frame every present; the *screen* damage is still
    /// just the diff.
    pub(super) fn render_frame(&mut self) -> Result<bool> {
        // Reallocate the buffers if the surface was resized since they were made;
        // deferring from every configure to the first frame after them collapses
        // an interactive resize's dmabuf churn to one reallocation per painted
        // frame (frame-callback paced) instead of one per configure.
        if self.presentation.buffer_size != (self.width, self.height) {
            self.destroy_buffers();
            self.create_buffers()?;
        }
        let Some(idx) = (0..2).find(|&i| !self.presentation.busy[i]) else {
            return Ok(false);
        };
        let started = self.stats.then(Instant::now);
        let (sw, sh) = (self.width as i32, self.height as i32);
        let new_list = self.core.build_frame_list();

        // Nothing changed since the on-screen frame: nothing to present (idle).
        let screen_dmg = display::damage(&self.presentation.present_list, &new_list, sw, sh);
        // Ack the latest configure paired with this commit, so the buffer the
        // compositor sees is always the one sized to the configure it just acked;
        // that is what keeps an anchored resize edge from jumping. Taken here so
        // it rides the same flush as the commit below.
        let ack = self.pending_configure.take();
        if screen_dmg.is_empty() {
            // Nothing new to draw, but a pending configure must still be acked so
            // the compositor can finalize a state-only change; a bare commit
            // applies it against the current, already correctly sized buffer.
            if let Some(serial) = ack {
                self.ack_configure(serial);
                self.conn.request(self.surface, wl_surface::COMMIT, &[]);
            }
            return Ok(true);
        }
        if let Some(serial) = ack {
            self.ack_configure(serial);
        }
        // A scale change can arrive without a configure (e.g. a monitor move), so
        // make sure the geometry + viewport/buffer-scale ride this commit too.
        if !self.scale.synced {
            self.declare_surface_scale();
        }

        let background = color_f32(self.core.clear_color());
        // Render, returning the render-done fence under explicit sync (`None` on
        // the bridge path, or when the driver could not export one).
        let render_done = {
            let Some(gpu) = self.presentation.backend.as_mut() else {
                return Err(Error::msg("gpu frame without a gpu backend"));
            };
            let Some(img) = self.presentation.images.get_mut(idx) else {
                return Err(Error::msg("gpu frame without gpu buffers"));
            };
            let frame = gpu::build_frame(&self.fonts, &new_list, &mut self.presentation.glyphs);
            match self.presentation.explicit_sync.as_ref() {
                Some(es) => {
                    let release_wait = es.release_fence(idx)?;
                    gpu.render_list_explicit(img, &frame, background, release_wait)?
                }
                None => {
                    gpu.render_list(img, &frame, background)?;
                    None
                }
            }
        };
        self.presentation.present_list = new_list;

        let buffer = self.presentation.buffers[idx];
        self.conn.request(
            self.surface,
            wl_surface::ATTACH,
            &[Arg::Object(buffer), Arg::Int(0), Arg::Int(0)],
        );
        // Explicit sync: publish this frame's render-done fence as the acquire
        // point and claim a release point, so the compositor waits before reading
        // and signals when done. With no exportable fence the frame was
        // CPU-waited: set no points.
        match (render_done, self.presentation.explicit_sync.as_mut()) {
            (Some(fence), Some(es)) => {
                es.publish(&mut self.conn, &fence, idx)?;
                self.presentation.explicit_fence_frames += 1;
            }
            (None, Some(es)) => {
                es.buffer_release[idx] = None;
                self.presentation.explicit_cpu_wait_frames += 1;
            }
            _ => {}
        }
        for r in &screen_dmg {
            self.conn.request(
                self.surface,
                wl_surface::DAMAGE,
                &[Arg::Int(r.x), Arg::Int(r.y), Arg::Int(r.w), Arg::Int(r.h)],
            );
        }
        let callback = self.alloc_id();
        self.conn
            .request(self.surface, wl_surface::FRAME, &[Arg::NewId(callback)]);
        self.frame_callback = callback;
        self.conn.request(self.surface, wl_surface::COMMIT, &[]);
        self.presentation.busy[idx] = true;

        if let Some(started) = started {
            self.log_frame_stats(started);
            self.presentation.last_present = Some(started);
        }
        self.presentation.frame_count += 1;
        Ok(true)
    }

    /// Print the per-frame stats line under `BNKTERM_STATS`: the build+submit
    /// time, the gap since the last present, the glyph-cache hit rate, and which
    /// sync strategy carried the frame.
    fn log_frame_stats(&self, started: Instant) {
        let cache = self.fonts.cache_stats();
        let gap = self
            .presentation
            .last_present
            .map(|t| started.duration_since(t).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let sync = if self.presentation.explicit_sync.is_some() {
            format!(
                "explicit-sync {} fence/{} cpu-wait",
                self.presentation.explicit_fence_frames, self.presentation.explicit_cpu_wait_frames
            )
        } else {
            "implicit-sync bridge".to_string()
        };
        eprintln!(
            "[stats] gpu frame {} {:.2}ms build+submit, {:.1}ms since last | glyphs hit {} miss {} ({:.1}% of {}) | {}",
            self.presentation.frame_count,
            started.elapsed().as_secs_f64() * 1000.0,
            gap,
            cache.hits,
            cache.misses,
            cache.hit_rate() * 100.0,
            cache.lookups(),
            sync,
        );
    }

    /// Ask the dmabuf global for its default feedback. The events accumulate in
    /// the feedback state as they arrive; nothing waits on them.
    pub(super) fn request_dmabuf_feedback(&mut self) {
        let Some(dmabuf) = self.presentation.dmabuf else {
            return;
        };
        let id = self.alloc_id();
        self.conn.request(
            dmabuf,
            zwp_linux_dmabuf_v1::GET_DEFAULT_FEEDBACK,
            &[Arg::NewId(id)],
        );
        self.presentation.feedback_id = id;
    }

    /// One event of a feedback burst; the feedback state accumulates and commits.
    pub(super) fn on_dmabuf_feedback(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            zwp_linux_dmabuf_feedback_v1::EV_FORMAT_TABLE => {
                let size = r.u32()? as usize;
                let fd = self.conn.take_fd().ok_or_else(|| {
                    Error::msg("dmabuf feedback format_table arrived without its fd")
                })?;
                if size == 0 {
                    self.presentation.feedback.set_table(&[]);
                    return Ok(());
                }
                let bytes = ffi::read_mapped(fd.as_raw_fd(), size)?;
                self.presentation.feedback.set_table(&bytes);
            }
            zwp_linux_dmabuf_feedback_v1::EV_MAIN_DEVICE => {
                self.presentation.feedback.main_device(r.array()?)?;
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_TARGET_DEVICE => {
                self.presentation.feedback.tranche_device(r.array()?)?;
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_FORMATS => {
                self.presentation.feedback.tranche_formats(r.array()?);
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_FLAGS => {
                self.presentation.feedback.tranche_flags(r.u32()?);
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_DONE => {
                self.presentation.feedback.tranche_done()
            }
            zwp_linux_dmabuf_feedback_v1::EV_DONE => self.presentation.feedback.done(),
            _ => {}
        }
        Ok(())
    }

    /// The `--gpu-probe` body: bind the registry, fetch dmabuf feedback, and
    /// print what the GPU presentation path has to work with, without opening a
    /// window. A successful probe means the terminal's GPU prerequisites hold.
    pub(super) fn probe_dmabuf(&mut self) -> Result<()> {
        self.registry = self.alloc_id();
        self.conn.request(
            protocol::WL_DISPLAY,
            wl_display::GET_REGISTRY,
            &[Arg::NewId(self.registry)],
        );
        self.roundtrip()?;
        if self.presentation.dmabuf.is_none() {
            println!(
                "gpu-probe: compositor lacks zwp_linux_dmabuf_v1 v{}+; \
                 bnkterm cannot run here (it requires the GPU path)",
                protocol::VERSION_DMABUF
            );
            return Ok(());
        }
        println!(
            "gpu-probe: zwp_linux_dmabuf_v1 bound at v{}",
            self.presentation.dmabuf_version
        );
        match self.presentation.syncobj_manager {
            Some(_) => println!(
                "gpu-probe: {} present: explicit sync available",
                protocol::IFACE_DRM_SYNCOBJ
            ),
            None => println!(
                "gpu-probe: {} absent: GPU path uses the implicit-sync bridge",
                protocol::IFACE_DRM_SYNCOBJ
            ),
        }
        self.request_dmabuf_feedback();
        self.roundtrip()?;
        let Some(fb) = self.presentation.feedback.ready() else {
            return Err(Error::msg(
                "compositor never committed dmabuf feedback (no done event)",
            ));
        };
        println!(
            "gpu-probe: main device: {}:{}",
            dmabuf::dev_major(fb.main_device),
            dmabuf::dev_minor(fb.main_device)
        );
        let mods = fb.modifiers_for(dmabuf::DRM_FORMAT_XRGB8888);
        if mods.is_empty() {
            println!("gpu-probe: XRGB8888 is not offered; the GPU path needs it");
            return Ok(());
        }
        let main_device = fb.main_device;
        let gpu = match vulkan::Gpu::new(main_device, crate::app::config_text_gamma()) {
            Ok(gpu) => gpu,
            Err(e) => {
                println!("gpu-probe: vulkan unavailable: {e}");
                return Ok(());
            }
        };
        println!(
            "gpu-probe: vulkan device: {} (graphics queue family {})",
            gpu.name(),
            gpu.queue_family()
        );
        let common = gpu.image_modifiers(&mods);
        if common.is_empty() {
            println!("gpu-probe: no common DRM modifier; the GPU path is unavailable");
            return Ok(());
        }
        match gpu.create_image(64, 64, &common) {
            Ok(img) => {
                println!(
                    "gpu-probe: test image exported: modifier {:#x}, stride {}, offset {}",
                    img.modifier(),
                    img.stride(),
                    img.offset()
                );
                gpu.destroy_image(img);
                println!("gpu-probe: ready (bnkterm would use the GPU path)");
            }
            Err(e) => println!("gpu-probe: image export failed: {e}"),
        }
        Ok(())
    }
}

/// A 0x00RRGGBB color as the linear-float RGBA a Vulkan clear expects. The
/// authored bytes are sRGB-encoded; the color attachment is an sRGB view, so the
/// clear value is taken as linear and re-encoded on store. Converting here makes
/// the cleared background land on the same bytes as the authored color.
fn color_f32(color: u32) -> [f32; 4] {
    let srgb_to_linear = |c: u32| {
        let c = c as f32 / 255.0;
        if c > 0.04045 {
            ((c + 0.055) / 1.055).powf(2.4)
        } else {
            c / 12.92
        }
    };
    [
        srgb_to_linear((color >> 16) & 0xff),
        srgb_to_linear((color >> 8) & 0xff),
        srgb_to_linear(color & 0xff),
        1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The store-side sRGB encode the attachment applies, inverse of the decode
    /// in color_f32.
    fn linear_to_srgb(c: f32) -> f32 {
        if c > 0.003_130_8 {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        } else {
            12.92 * c
        }
    }

    #[test]
    fn clear_color_decode_then_store_encode_reproduces_the_authored_bytes() {
        for color in [
            0x0000_0000,
            0x00ff_ffff,
            0x0080_8080,
            0x0012_3456,
            0x001c_2127,
        ] {
            let lin = color_f32(color);
            let enc = |c: f32| (linear_to_srgb(c) * 255.0).round() as u32;
            let got = (enc(lin[0]) << 16) | (enc(lin[1]) << 8) | enc(lin[2]);
            assert_eq!(
                got, color,
                "0x{color:08x} must survive decode then re-encode"
            );
            assert_eq!(lin[3], 1.0, "the clear is opaque");
        }
    }
}
