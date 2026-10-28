//! The GPU/dmabuf presentation half of the app: negotiating the buffers, sync,
//! and per-frame submit that turn a [`crate::render::display::DisplayList`] into pixels on the
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
use std::time::Instant;

use crate::app::State;
// Crate-level error type throughout (see the note in `app.rs`); `?` on a
// platform/render call converts via the `From` bridge in `crate::error`.
use crate::error::{Error, Result};
use crate::platform::conn::Connection;
use crate::platform::dmabuf;
use crate::platform::ffi;
use crate::platform::geom::Rect;
use crate::platform::protocol::{
    self, wl_buffer, wl_display, wl_surface, wp_linux_drm_syncobj_manager_v1,
    wp_linux_drm_syncobj_surface_v1, wp_viewport, xdg_surface, zwp_linux_buffer_params_v1,
    zwp_linux_dmabuf_feedback_v1, zwp_linux_dmabuf_v1,
};
use crate::platform::wire::{Arg, Reader};
use crate::render::display;
use crate::render::gpu;
use crate::render::vulkan;
use crate::term_render;

/// How long [`State::destroy_buffers`] waits for the compositor to release a buffer before
/// giving up on freeing its image *now*. Normally the release has long since signalled and
/// the wait returns at once; when it does not, the image is not freed anyway (that would be
/// the use-after-free) but retired to be reaped once the release actually arrives, so this is
/// only how long the fast inline free is worth waiting for. Two frames at 60 Hz is ~33 ms;
/// this is comfortable slack over that.
const BUFFER_RELEASE_WAIT_MS: core::ffi::c_int = 32;

/// The retired-image list is only ever fed two images per resize and drained as their
/// releases arrive, so it stays tiny. This cap is the backstop against a compositor that
/// never signals a release at all: past it, the oldest is freed regardless (the app is
/// already wedged, and bounded memory beats an unbounded leak). ~8 resizes' worth.
const MAX_RETIRED_IMAGES: usize = 16;

/// A swap image retired by a resize while the compositor may still be reading its wl_buffer.
/// Freeing it then is a GPU use-after-free — the intermittent crash a rapid resize raced
/// into — so it waits here until the release is observed and is destroyed only then. See
/// [`State::destroy_buffers`], [`State::reap_retired`], and [`State::release_retired`].
struct RetiredImage {
    image: vulkan::GpuImage,
    /// The wl_buffer this image backed, kept alive so the compositor can still signal its
    /// `wl_buffer.release`; destroyed together with the image when reaped.
    buffer: u32,
    /// Explicit-sync release fence to poll (signalled means released), or `None` on the
    /// implicit path, where the `wl_buffer.release` event is the only release signal.
    fence: Option<OwnedFd>,
}

/// Client-owned explicit-sync state (`linux-drm-syncobj-v1`), negotiated over a
/// live GPU backend when the compositor advertises the manager. DRM syncobj
/// timelines the compositor shares: we signal `acquire` with each frame's
/// render-done fence, and it signals a buffer slot's `release` timeline when done
/// reading that buffer. A scratch binary syncobj carries the sync-file transfers,
/// and a binary syncobj created signalled is the acquire fence of a frame with no
/// render-done fence. `point` increments per present and numbers both of a frame's
/// points. Dropping this closes the render-node fd, which frees every syncobj
/// created on it.
///
/// A buffer is free for the next frame once its release point has a fence, and
/// `wl_buffer.release` plays no part: the protocol leaves its delivery undefined
/// under explicit sync. When both buffers are held, `wake` is registered on their
/// release points and the event loop polls it, so the release itself ends the wait.
pub(super) struct ExplicitSync {
    drm: OwnedFd,
    acquire_syncobj: u32,
    scratch: u32,
    signalled: u32,
    acquire_timeline: u32,
    release: [ReleaseTimeline; 2],
    surface: u32,
    point: u64,
    wake: OwnedFd,
}

/// Where a buffer slot stands for the next frame rendered into it.
enum SlotRelease {
    /// No frame has gone into the slot's buffer: render into it at once.
    Unused,
    /// The compositor still holds the buffer: its release point has no fence yet.
    Held,
    /// The compositor released the buffer: a fence sits at its release point, which
    /// the next render into it waits on. It may not have signalled yet.
    Released,
}

/// One buffer slot's release timeline. Each slot has its own because signalling a
/// timeline point signals every earlier point too, and a compositor may release
/// buffers in a different order than they were committed: one buffer's release
/// must never signal a point another buffer is waiting on.
#[derive(Default)]
struct ReleaseTimeline {
    syncobj: u32,
    /// The protocol object the compositor imported `syncobj` as.
    timeline: u32,
    /// The point last published for the slot's buffer, which its next reuse waits
    /// on. `None` before the slot's first frame and after a resize replaced it.
    point: Option<u64>,
    /// The point the release wake was last registered on. A registration fires as
    /// soon as its point has a fence, so while the slot is held at this point one is
    /// still pending, and the kernel keeps each until it fires.
    waking: Option<u64>,
}

impl ExplicitSync {
    /// Create the client's syncobjs on `drm`: the acquire timeline, a release
    /// timeline per buffer slot, and the scratch and signalled binaries. Their
    /// protocol objects are zero until [`State::negotiate_explicit_sync`] shares
    /// them with the compositor.
    ///
    /// `None` where the kernel or driver has no syncobj eventfds (Linux 6.6), which
    /// the release wake needs. The caller then keeps the implicit bridge.
    fn new(drm: OwnedFd) -> Result<Option<Self>> {
        let fd = drm.as_raw_fd();
        let mut release = [ReleaseTimeline::default(), ReleaseTimeline::default()];
        for slot in &mut release {
            slot.syncobj = ffi::drm_syncobj_create(fd, false)?;
        }
        let signalled = ffi::drm_syncobj_create(fd, true)?;
        let wake = ffi::make_eventfd()?;
        if !ffi::drm_syncobj_eventfd_supported(fd, signalled, wake.as_raw_fd())? {
            return Ok(None);
        }
        ffi::efd_clear(wake.as_raw_fd());
        Ok(Some(Self {
            acquire_syncobj: ffi::drm_syncobj_create(fd, false)?,
            scratch: ffi::drm_syncobj_create(fd, false)?,
            signalled,
            drm,
            acquire_timeline: 0,
            release,
            surface: 0,
            point: 0,
            wake,
        }))
    }

    /// Where buffer slot `idx` stands: unused, held by the compositor, or released.
    fn slot_release(&self, idx: usize) -> Result<SlotRelease> {
        let slot = self
            .release
            .get(idx)
            .ok_or_else(|| Error::msg(format!("no buffer slot {idx}")))?;
        let Some(p) = slot.point else {
            return Ok(SlotRelease::Unused);
        };
        if ffi::drm_syncobj_point_available(self.drm.as_raw_fd(), slot.syncobj, p)? {
            Ok(SlotRelease::Released)
        } else {
            Ok(SlotRelease::Held)
        }
    }

    /// The fence a render into buffer `idx` waits on: the compositor's release of
    /// the slot's last frame, or `None` before its first. Only for a slot that is
    /// not [`SlotRelease::Held`]: a held slot's point has no fence to export, and
    /// asking for one is an error.
    fn release_fence(&self, idx: usize) -> Result<Option<OwnedFd>> {
        let slot = self
            .release
            .get(idx)
            .ok_or_else(|| Error::msg(format!("no buffer slot {idx}")))?;
        match slot.point {
            Some(p) => Ok(Some(ffi::drm_syncobj_point_to_sync_file(
                self.drm.as_raw_fd(),
                slot.syncobj,
                p,
                self.scratch,
            )?)),
            None => Ok(None),
        }
    }

    /// The slot the next frame goes into, or `None` while the compositor holds both
    /// buffers. Then [`Self::wake_fd`] becomes readable once either is released: each
    /// held point carries one registration, made by the first wait that finds it held.
    fn free_slot(&mut self) -> Result<Option<usize>> {
        for idx in 0..self.release.len() {
            if !matches!(self.slot_release(idx)?, SlotRelease::Held) {
                return Ok(Some(idx));
            }
        }
        let drm = self.drm.as_raw_fd();
        for slot in &mut self.release {
            if let Some(p) = slot.point.filter(|&p| slot.waking != Some(p)) {
                ffi::drm_syncobj_eventfd_when_available(
                    drm,
                    slot.syncobj,
                    p,
                    self.wake.as_raw_fd(),
                )?;
                slot.waking = Some(p);
            }
        }
        Ok(None)
    }

    /// The eventfd the event loop polls for a held buffer's release.
    fn wake_fd(&self) -> RawFd {
        self.wake.as_raw_fd()
    }

    /// The wake fired: clear it, so the next registration to fire wakes the loop
    /// again.
    fn woke(&mut self) {
        ffi::efd_clear(self.wake.as_raw_fd());
    }

    /// Publish this frame's render-done `fence` as buffer `idx`'s acquire point
    /// and claim the matching release point on that buffer's own timeline, so the
    /// compositor waits for the GPU before reading and signals the point when it
    /// is done (the buffer's next reuse waits on that via [`Self::release_fence`]).
    ///
    /// `None` is a frame the CPU already waited out: its acquire point is published
    /// already signalled. Both points are set either way, because a commit that
    /// attaches a buffer without them is the `no_acquire_point` protocol error,
    /// which closes the connection.
    fn publish(
        &mut self,
        conn: &mut Connection,
        fence: Option<&OwnedFd>,
        idx: usize,
    ) -> Result<()> {
        let slot = self
            .release
            .get_mut(idx)
            .ok_or_else(|| Error::msg(format!("no buffer slot {idx}")))?;
        self.point += 1;
        let p = self.point;
        let drm = self.drm.as_raw_fd();
        match fence {
            Some(fence) => ffi::drm_syncobj_sync_file_to_point(
                drm,
                fence.as_raw_fd(),
                self.acquire_syncobj,
                p,
                self.scratch,
            )?,
            None => ffi::drm_syncobj_signal_point(drm, self.acquire_syncobj, p, self.signalled)?,
        }
        slot.point = Some(p);
        let release_timeline = slot.timeline;
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
            &[Arg::Object(release_timeline), Arg::Uint(hi), Arg::Uint(lo)],
        );
        Ok(())
    }

    /// Forget every buffer's published release point (the buffers they named are
    /// gone, e.g. after a resize), so the next use of each slot waits on nothing.
    /// A registration still pending on a forgotten point stays: the slot's next
    /// buffer publishes later points on the same timeline, and their release fires
    /// it too.
    fn forget_buffers(&mut self) {
        for slot in &mut self.release {
            slot.point = None;
        }
    }
}

/// GPU/dmabuf presentation resources owned by the app: protocol globals and
/// feedback, the Vulkan backend, the two exported swap buffers, explicit-sync
/// state, and the small counters the stats line reports.
#[derive(Default)]
pub(super) struct GpuPresentation {
    pub(super) dmabuf: Option<u32>,
    pub(super) dmabuf_version: u32,
    pub(super) feedback_id: u32,
    pub(super) feedback: dmabuf::FeedbackState,
    pub(super) backend: Option<vulkan::Gpu>,
    pub(super) images: Vec<vulkan::GpuImage>,
    pub(super) modifiers: Vec<u64>,
    pub(super) glyphs: gpu::GlyphCache,
    /// The coverage-gamma dials, resolved from the environment once at startup: the
    /// batcher turns them into each run's exponent, so they are read here rather than
    /// per frame.
    pub(super) text_gamma: gpu::TextGamma,
    pub(super) syncobj_manager: Option<u32>,
    pub(super) explicit_sync: Option<ExplicitSync>,
    pub(super) buffers: [u32; 2],
    /// Whether the compositor holds each buffer, as `wl_buffer.release` reports it.
    /// It picks the slot on the implicit path only; under explicit sync the release
    /// points do, and this is consulted only when a resize frees a buffer whose
    /// release point has no fence yet.
    pub(super) busy: [bool; 2],
    /// Images a resize retired that the compositor may still be reading, awaiting their
    /// release before they are freed (see [`RetiredImage`]).
    retired: Vec<RetiredImage>,
    pub(super) buffer_size: (u32, u32),
    /// The recycled display-list double buffer: the on-screen frame the next one is
    /// diffed against, plus a pool of run strings, so a steady frame builds the list
    /// without allocating.
    pub(super) lists: term_render::DisplayListPool,
    /// The reused GPU frame data (vertices/batches), refilled in place each frame.
    pub(super) frame_scratch: gpu::FrameData,
    /// The reused damage-diff output, so computing the changed regions each frame
    /// refills one rectangle buffer instead of allocating a fresh one.
    pub(super) damage: Vec<Rect>,
    /// Record the capture copy into the next frame. Cleared once it is written.
    pub(super) capture: bool,
    pub(super) frame_count: u64,
    pub(super) last_present: Option<Instant>,
    pub(super) explicit_fence_frames: u64,
    pub(super) explicit_cpu_wait_frames: u64,
}

impl GpuPresentation {
    /// Everything at rest, but for the text gamma the window resolved from the
    /// environment at startup (see `app::config_text_gamma`).
    pub(super) fn with_gamma(text_gamma: gpu::TextGamma) -> Self {
        let mut presentation = Self::default();
        presentation.text_gamma = text_gamma;
        presentation
    }
}

impl Drop for GpuPresentation {
    /// The images, live and retired, are the device's children and must go before it
    /// (VUID-vkDestroyDevice-device-05137). The backend is an earlier field, so it would
    /// otherwise drop first.
    fn drop(&mut self) {
        let Some(gpu) = &self.backend else {
            return;
        };
        gpu.wait_idle();
        let retired = self.retired.drain(..).map(|retired| retired.image);
        for image in self.images.drain(..).chain(retired) {
            gpu.destroy_image(image);
        }
    }
}

/// The sync mode reported on a `[stats]` line, written straight into the log via
/// [`Display`](std::fmt::Display) rather than a heap `String`: `--stats` exists to
/// profile a live session, so its own reporting must not allocate per frame.
enum SyncStatus {
    Explicit {
        fence_frames: u64,
        cpu_wait_frames: u64,
    },
    Implicit,
}

impl std::fmt::Display for SyncStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SyncStatus::Explicit {
                fence_frames,
                cpu_wait_frames,
            } => write!(
                f,
                "explicit-sync {fence_frames} fence/{cpu_wait_frames} cpu-wait"
            ),
            SyncStatus::Implicit => f.write_str("implicit-sync bridge"),
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
        let gpu = vulkan::Gpu::new(main_device)?;
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
    /// returns nothing and only says so under `--verbose`.
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
                if self.verbosity.verbose() {
                    eprintln!("bnkterm: explicit sync unavailable, using the sync bridge: {e}");
                }
            }
        }
    }

    /// Create the client-owned syncobjs, share the timelines with the compositor,
    /// and attach explicit sync to the surface.
    pub(super) fn negotiate_explicit_sync(
        &mut self,
        manager: u32,
        minor: u32,
    ) -> Result<ExplicitSync> {
        let Some(mut es) = ExplicitSync::new(ffi::open_drm_render_node(minor)?)? else {
            return Err(Error::msg("the kernel has no syncobj eventfds (Linux 6.6)"));
        };
        let fd = es.drm.as_raw_fd();
        es.acquire_timeline = self.import_timeline(manager, fd, es.acquire_syncobj)?;
        for slot in &mut es.release {
            slot.timeline = self.import_timeline(manager, fd, slot.syncobj)?;
        }
        es.surface = self.create_for(
            manager,
            wp_linux_drm_syncobj_manager_v1::GET_SURFACE,
            self.objects.surface,
        );
        Ok(es)
    }

    /// The fd the event loop polls for a held buffer's release, while explicit sync
    /// is on.
    pub(super) fn release_wake_fd(&self) -> Option<RawFd> {
        self.presentation
            .explicit_sync
            .as_ref()
            .map(ExplicitSync::wake_fd)
    }

    /// A held buffer's release point got its fence: clear the wake, so the next
    /// turn's frame can take that buffer.
    pub(super) fn on_release_wake(&mut self) {
        if let Some(es) = self.presentation.explicit_sync.as_mut() {
            es.woke();
        }
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
        let (w, h) = (self.layout.terminal.width, self.layout.terminal.height);
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
            let params = self.create(dmabuf_global, zwp_linux_dmabuf_v1::CREATE_PARAMS);
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
        self.presentation.buffer_size = (self.layout.terminal.width, self.layout.terminal.height);
        self.presentation.busy = [false, false];
        // Fresh images never carried a release point, so a resize must forget the
        // old buffers' points; their next use starts with no wait.
        if let Some(es) = self.presentation.explicit_sync.as_mut() {
            es.forget_buffers();
        }
        // Fresh images hold no known frame, so forget what was on screen; the next
        // frame into them repaints whole rather than trusting the diff.
        self.presentation.lists.reset();
        Ok(())
    }

    /// Destroy the current buffers and their backing GPU images, or, for any the compositor
    /// may still be reading, retire them to be freed once their release arrives.
    ///
    /// `wait_idle` drains only *our* device; the compositor reading a buffer for scan-out is a
    /// separate lifetime, and freeing an image while it does is a GPU use-after-free — the
    /// intermittent crash a rapid resize raced into. So each image is freed now only when its
    /// buffer is known released: an explicit-sync release fence that signals within a short
    /// bounded wait, or, on the implicit path, a slot the compositor is not marked as holding.
    /// One that is still held is moved to the retired list (its wl_buffer kept alive so the
    /// release can still be signalled) and reaped later by [`Self::reap_retired`] or
    /// [`Self::release_retired`]. A timeout therefore never frees a buffer out from under the
    /// compositor; it only defers the free.
    pub(super) fn destroy_buffers(&mut self) {
        // First reap anything an earlier resize retired that has since been released, so a
        // burst of resizes cannot pile the list up.
        self.reap_retired();

        let buffers = self.presentation.buffers;
        let busy = self.presentation.busy;
        let images: Vec<vulkan::GpuImage> = std::mem::take(&mut self.presentation.images);
        // Nothing may be in flight on *our* device while its images are touched, whether they
        // are freed now or held for later; resize is rare enough that a full drain is the
        // simple correct answer.
        if let Some(gpu) = &self.presentation.backend {
            gpu.wait_idle();
        }
        for (idx, image) in images.into_iter().enumerate() {
            let buffer = buffers.get(idx).copied().unwrap_or(0);
            // The compositor's "done reading" signal: the explicit-sync release fence when we
            // have one, else the busy flag the `wl_buffer.release` event clears.
            let fence = self.presentation.explicit_sync.as_ref().and_then(|es| {
                match es.slot_release(idx) {
                    Ok(SlotRelease::Released) => es.release_fence(idx).ok().flatten(),
                    _ => None,
                }
            });
            let released = match &fence {
                Some(f) => ffi::wait_sync_file(f.as_raw_fd(), BUFFER_RELEASE_WAIT_MS),
                None => buffer == 0 || !busy.get(idx).copied().unwrap_or(false),
            };
            if released {
                if buffer != 0 {
                    self.conn.request(buffer, wl_buffer::DESTROY, &[]);
                }
                if let Some(gpu) = &self.presentation.backend {
                    gpu.destroy_image(image);
                }
            } else {
                self.presentation.retired.push(RetiredImage {
                    image,
                    buffer,
                    fence,
                });
            }
        }
        self.enforce_retired_cap();
        self.presentation.buffers = [0, 0];
        self.presentation.busy = [false, false];
    }

    /// Free every retired image whose buffer the compositor has finished with: an explicit
    /// release fence that now polls signalled (a zero timeout — a poll, never a wait). Implicit
    /// entries carry no fence and are reaped by [`Self::release_retired`] on their
    /// `wl_buffer.release` instead. Cheap and a no-op when the list is empty, so the frame path
    /// can call it every frame.
    pub(super) fn reap_retired(&mut self) {
        let mut idx = 0;
        while idx < self.presentation.retired.len() {
            let released = match &self.presentation.retired[idx].fence {
                Some(f) => ffi::wait_sync_file(f.as_raw_fd(), 0),
                None => false,
            };
            if released {
                let r = self.presentation.retired.swap_remove(idx);
                self.free_retired(r);
            } else {
                idx += 1;
            }
        }
    }

    /// A `wl_buffer.release` arrived for a retired buffer (one a resize dropped from the active
    /// slots): the compositor is definitively done with it, so free its image now. Returns
    /// whether a retired entry matched. This is the only release signal on the implicit path,
    /// and a belt-and-suspenders one under explicit sync.
    pub(super) fn release_retired(&mut self, buffer: u32) {
        if let Some(pos) = self
            .presentation
            .retired
            .iter()
            .position(|r| r.buffer == buffer)
        {
            let r = self.presentation.retired.swap_remove(pos);
            self.free_retired(r);
        }
    }

    /// Destroy a retired image and its wl_buffer. The device was drained when it was retired
    /// and countless frames have passed since, so no in-flight work of ours references it.
    fn free_retired(&mut self, r: RetiredImage) {
        if r.buffer != 0 {
            self.conn.request(r.buffer, wl_buffer::DESTROY, &[]);
        }
        if let Some(gpu) = &self.presentation.backend {
            gpu.destroy_image(r.image);
        }
    }

    /// Backstop for a compositor that never signals a release: keep the retired list bounded by
    /// force-freeing the oldest past the cap. Reaching it means releases are not arriving at
    /// all (the app is already wedged), where bounded memory beats an unbounded image leak.
    fn enforce_retired_cap(&mut self) {
        while self.presentation.retired.len() > MAX_RETIRED_IMAGES {
            if self.verbosity.verbose() {
                eprintln!(
                    "bnkterm: retired-image cap reached ({} held); force-freeing the oldest",
                    self.presentation.retired.len()
                );
            }
            let r = self.presentation.retired.remove(0);
            self.free_retired(r);
        }
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
            self.objects.xdg_surface,
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
            self.objects.xdg_surface,
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
            self.conn.request(
                self.objects.surface,
                wl_surface::SET_BUFFER_SCALE,
                &[Arg::Int(n)],
            );
        }
        self.scale.synced = true;
    }

    /// Build the current frame and present it on the GPU. Returns false if both
    /// buffers are still held by the compositor, leaving `dirty` set so the next
    /// buffer release retries: a `wl_buffer.release` on the implicit path, the
    /// release wake under explicit sync.
    ///
    /// The display list is built and diffed against `present_list` (what is on
    /// screen): an identical frame early-outs (the common idle case), otherwise
    /// the changed screen regions are reported to the compositor as damage. The
    /// GPU repaints the full frame every present; the *screen* damage is still
    /// just the diff.
    pub(super) fn render_frame(&mut self) -> Result<bool> {
        // Free any image a prior resize retired that the compositor has now released. Cheap
        // and a no-op once the list drains, which it does within a frame or two of a resize.
        self.reap_retired();
        // Reallocate the buffers if the surface was resized since they were made;
        // deferring from every configure to the first frame after them collapses
        // an interactive resize's dmabuf churn to one reallocation per painted
        // frame (frame-callback paced) instead of one per configure.
        if self.presentation.buffer_size
            != (self.layout.terminal.width, self.layout.terminal.height)
        {
            self.destroy_buffers();
            self.create_buffers()?;
        }
        let slot = match self.presentation.explicit_sync.as_mut() {
            Some(es) => es.free_slot()?,
            None => (0..2).find(|&i| !self.presentation.busy[i]),
        };
        let Some(idx) = slot else {
            return Ok(false);
        };
        let started = self.verbosity.frame_stats().then(Instant::now);
        let (sw, sh) = (
            self.layout.terminal.width as i32,
            self.layout.terminal.height as i32,
        );
        // Rebuild the display list into the recycled back buffer (salvaging the
        // previous occupant's run strings into the pool first), so a steady frame
        // builds it without allocating. `front` still holds the on-screen frame.
        {
            let (out, strings) = self.presentation.lists.begin();
            self.tabs
                .fill_frame_list(out, strings, &self.fonts, self.link_preview);
            // The leader-mode indicator draws over the grid and bar (last wins in
            // list order); it is nothing at all in Normal mode.
            crate::keymode::paint_overlay(
                self.key_mode,
                out,
                strings,
                self.layout.terminal.metrics,
                self.tabs.active().theme(),
                sw,
                sh,
            );
            // The corner notice sits above both, and above the leader overlay it can
            // never collide with (that one is centered). Nothing at all when none stands.
            crate::notice::paint(
                self.tabs.notice(),
                out,
                strings,
                self.layout.terminal.metrics,
                self.tabs.active().theme(),
                sw,
                sh,
            );
        }

        // Nothing changed since the on-screen frame: nothing to present (idle). The
        // rectangles land in the reused scratch and are read from there at each use
        // point below, so the per-frame diff allocates nothing.
        if self.presentation.lists.front_presented() {
            display::damage_into(
                self.presentation.lists.front(),
                self.presentation.lists.back(),
                sw,
                sh,
                &mut self.presentation.damage,
            );
        } else {
            // Fresh buffers (startup, or a resize that reallocated them) hold no known
            // image, so there is nothing to diff against and an empty diff would read
            // as "already on screen". The first frame into them repaints whole.
            self.presentation.damage.clear();
            self.presentation.damage.push(Rect {
                x: 0,
                y: 0,
                w: sw.max(0),
                h: sh.max(0),
            });
        }
        // Ack the latest configure paired with this commit, so the buffer the
        // compositor sees is always the one sized to the configure it just acked;
        // that is what keeps an anchored resize edge from jumping. Taken here so
        // it rides the same flush as the commit below.
        let ack = self.pending_configure.take();
        if self.presentation.damage.is_empty() {
            // Nothing new to draw, but a pending configure must still be acked so
            // the compositor can finalize a state-only change; a bare commit
            // applies it against the current, already correctly sized buffer.
            if let Some(serial) = ack {
                self.ack_configure(serial);
                self.conn
                    .request(self.objects.surface, wl_surface::COMMIT, &[]);
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

        let background = clear_color_linear(self.tabs.active().clear_color());
        // Batch the freshly built list into the reused frame data (vertices/batches
        // refilled in place, no allocation in steady state).
        gpu::build_frame_into(
            &self.fonts,
            self.presentation.lists.back(),
            &mut self.presentation.glyphs,
            self.presentation.text_gamma,
            &mut self.presentation.frame_scratch,
        );
        // Read before the borrow below, which takes the backend and the image
        // mutably for the whole render.
        let capture = self.presentation.capture;
        // Render, returning the render-done fence under explicit sync (`None` on
        // the bridge path, or when the driver could not export one).
        let render_done = {
            let Some(gpu) = self.presentation.backend.as_mut() else {
                return Err(Error::msg("gpu frame without a gpu backend"));
            };
            let Some(img) = self.presentation.images.get_mut(idx) else {
                return Err(Error::msg("gpu frame without gpu buffers"));
            };
            let frame = &self.presentation.frame_scratch;
            match self.presentation.explicit_sync.as_ref() {
                Some(es) => {
                    let release_wait = es.release_fence(idx)?;
                    gpu.render_list_explicit(img, frame, background, release_wait, capture)?
                }
                None => {
                    gpu.render_list(img, frame, background, capture)?;
                    None
                }
            }
        };
        // The freshly built back buffer is now on screen; it becomes next frame's
        // `front` (the diff's old side).
        self.presentation.lists.commit();

        let buffer = self.presentation.buffers[idx];
        self.conn.request(
            self.objects.surface,
            wl_surface::ATTACH,
            &[Arg::Object(buffer), Arg::Int(0), Arg::Int(0)],
        );
        // Explicit sync: publish this frame's acquire point and claim a release
        // point, so the compositor waits before reading and signals when done. A
        // frame with no exportable fence was CPU-waited, and its acquire point goes
        // out already signalled.
        if let Some(es) = self.presentation.explicit_sync.as_mut() {
            es.publish(&mut self.conn, render_done.as_ref(), idx)?;
            if render_done.is_some() {
                self.presentation.explicit_fence_frames += 1;
            } else {
                self.presentation.explicit_cpu_wait_frames += 1;
            }
        }
        // `damage_buffer`, not `damage`: these rectangles come out of the display-list
        // diff, which works in the same device pixels the buffer is drawn in, and
        // `wl_surface.damage` would read them as surface-local (logical) coordinates.
        // The two spaces coincide only at scale 1.0.
        for r in &self.presentation.damage {
            self.conn.request(
                self.objects.surface,
                wl_surface::DAMAGE_BUFFER,
                &[Arg::Int(r.x), Arg::Int(r.y), Arg::Int(r.w), Arg::Int(r.h)],
            );
        }
        self.arm_frame_callback(Instant::now());
        self.conn
            .request(self.objects.surface, wl_surface::COMMIT, &[]);
        self.presentation.busy[idx] = true;
        // The first commit with a buffer maps the window, which activation needs to act on.
        self.spend_launch_token();

        if let Some(started) = started {
            self.log_frame_stats(started);
            self.presentation.last_present = Some(started);
        }
        // The capture is complete only when the GPU readback and file write succeed.
        #[cfg(test)]
        if capture {
            self.save_capture(idx)?;
            self.presentation.capture = false;
        }
        self.presentation.frame_count += 1;
        Ok(true)
    }

    /// Write the frame captured into image `idx` out as a PNG.
    #[cfg(test)]
    fn save_capture(&mut self, idx: usize) -> Result<()> {
        let Some(path) = self.capture_to.as_deref() else {
            return Err(Error::msg("a frame was captured with nowhere to put it"));
        };
        let Some(gpu) = self.presentation.backend.as_ref() else {
            return Err(Error::msg("capture without a gpu backend"));
        };
        let Some(img) = self.presentation.images.get(idx) else {
            return Err(Error::msg("capture without gpu buffers"));
        };
        let pixels = gpu.read_pixels(img)?;
        let (width, height) = self.presentation.buffer_size;
        crate::dev::screenshot::write(path, &pixels, width, height)?;
        eprintln!("bnkterm: screenshot: wrote {}", path.display());
        self.capture_to = None;
        Ok(())
    }

    /// Print the per-frame stats line under `--stats`: the build+submit
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
            SyncStatus::Explicit {
                fence_frames: self.presentation.explicit_fence_frames,
                cpu_wait_frames: self.presentation.explicit_cpu_wait_frames,
            }
        } else {
            SyncStatus::Implicit
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
        self.presentation.feedback_id =
            self.create(dmabuf, zwp_linux_dmabuf_v1::GET_DEFAULT_FEEDBACK);
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
        self.objects.registry = self.create(protocol::WL_DISPLAY, wl_display::GET_REGISTRY);
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
        let gpu = match vulkan::Gpu::new(main_device) {
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
fn clear_color_linear(color: u32) -> [f32; 4] {
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
    use std::io::Read;
    use std::os::unix::net::UnixStream;
    use std::time::Duration;

    use crate::app::present::*;
    use crate::platform::wire::Message;

    /// Explicit-sync state on this machine's first render node, or `None`, after
    /// saying why, where `test` cannot run: no render node, or a kernel without
    /// syncobj eventfds. Any other failure fails the test.
    fn explicit_sync(test: &str) -> Option<ExplicitSync> {
        let Some(drm) = ffi::open_any_render_node() else {
            eprintln!("no DRM render node; skipping {test}");
            return None;
        };
        let es = ExplicitSync::new(drm).expect("create the syncobjs");
        if es.is_none() {
            eprintln!("no syncobj eventfds (Linux 6.6); skipping {test}");
        }
        es
    }

    /// Frames go into buffer 0 and then buffer 1, and the compositor finishes with
    /// buffer 1 first, which the protocol allows. Buffer 0 is still being read, so
    /// it must not read as released until its own point signals. The compositor
    /// here acts only on what the requests on the socket name.
    #[test]
    fn a_buffer_released_out_of_order_does_not_release_the_other() {
        let Some(mut es) = explicit_sync("the release-order test") else {
            return;
        };
        // The ids the compositor knows the release timelines by after negotiation.
        es.release[0].timeline = 10;
        es.release[1].timeline = 11;
        let (stream, peer) = UnixStream::pair().expect("socketpair");
        let mut conn = Connection::from_stream(stream);
        es.publish(&mut conn, None, 0).expect("frame into buffer 0");
        es.publish(&mut conn, None, 1).expect("frame into buffer 1");
        conn.flush().expect("send the sync requests");

        // Each commit's release point, as the timeline and point it named.
        let mut compositor = Connection::from_stream(peer);
        compositor
            .fill(Some(Duration::from_secs(1)))
            .expect("read the sync requests");
        let mut msg = Message::default();
        let mut commits = Vec::new();
        while compositor.next_message_into(&mut msg).expect("decode") {
            if msg.opcode == wp_linux_drm_syncobj_surface_v1::SET_RELEASE_POINT {
                let mut r = Reader::new(&msg.body);
                let timeline = r.u32().unwrap();
                let point = (u64::from(r.u32().unwrap()) << 32) | u64::from(r.u32().unwrap());
                commits.push((timeline, point));
            }
        }
        assert_eq!(commits.len(), 2, "one release point per commit");

        // What the compositor does when it is done with a commit's buffer: signal
        // the point it named on the syncobj behind the timeline it named.
        let release = |es: &ExplicitSync, (timeline, point): (u32, u64)| {
            let slot = es.release.iter().find(|s| s.timeline == timeline).unwrap();
            ffi::drm_syncobj_signal_point(es.drm.as_raw_fd(), slot.syncobj, point, es.signalled)
                .expect("signal the release point");
        };
        let released = |es: &ExplicitSync, idx: usize| {
            matches!(es.slot_release(idx), Ok(SlotRelease::Released))
                && matches!(es.release_fence(idx), Ok(Some(f)) if ffi::wait_sync_file(f.as_raw_fd(), 0))
        };

        release(&es, commits[1]);
        assert!(released(&es, 1), "buffer 1 reads as released");
        assert!(
            matches!(es.slot_release(0), Ok(SlotRelease::Held)),
            "buffer 0 is still being read"
        );

        release(&es, commits[0]);
        assert!(
            released(&es, 0),
            "buffer 0 reads as released once its own point signals"
        );
    }

    /// The compositor holds both buffers and never sends `wl_buffer.release`, which
    /// the protocol allows under explicit sync. The frame waits, and the release of
    /// one buffer both wakes the event loop and frees that buffer for the frame.
    #[test]
    fn a_release_point_alone_frees_a_held_buffer_and_wakes_the_loop() {
        let Some(mut es) = explicit_sync("the release-wake test") else {
            return;
        };
        let (stream, _compositor) = UnixStream::pair().expect("socketpair");
        let mut conn = Connection::from_stream(stream);
        // The wake is an eventfd, readable while its counter is nonzero.
        let woken = |es: &ExplicitSync| ffi::wait_sync_file(es.wake_fd(), 0);

        assert!(matches!(es.free_slot(), Ok(Some(0))), "unused buffer 0");
        es.publish(&mut conn, None, 0).expect("frame into buffer 0");
        assert!(matches!(es.free_slot(), Ok(Some(1))), "unused buffer 1");
        es.publish(&mut conn, None, 1).expect("frame into buffer 1");

        assert!(matches!(es.free_slot(), Ok(None)), "both buffers are held");
        assert!(!woken(&es), "nothing has been released");
        assert!(
            matches!(es.free_slot(), Ok(None)),
            "still held, already armed"
        );

        // The compositor is done with buffer 1 and signals its release point.
        let slot = &es.release[1];
        let point = slot.point.expect("a published release point");
        ffi::drm_syncobj_signal_point(es.drm.as_raw_fd(), slot.syncobj, point, es.signalled)
            .expect("signal the release point");

        assert!(woken(&es), "the release wakes the loop");
        es.woke();
        assert!(!woken(&es), "the wake is cleared");
        assert!(matches!(es.free_slot(), Ok(Some(1))), "buffer 1 is free");
        let Ok(Some(fence)) = es.release_fence(1) else {
            panic!("buffer 1's render waits on the compositor's release");
        };
        assert!(
            ffi::wait_sync_file(fence.as_raw_fd(), 0),
            "the render waits on the release the compositor signalled"
        );
    }

    /// The compositor keeps buffer 0 while buffer 1 goes round a hundred times, each
    /// round a wait with both held. Every registration is a kernel allocation that
    /// lives until it fires, so buffer 0's one point must carry one registration, not
    /// one per wait.
    #[test]
    fn a_held_buffer_keeps_one_wake_registration_while_the_other_cycles() {
        let Some(mut es) = explicit_sync("the registration test") else {
            return;
        };
        let (stream, _compositor) = UnixStream::pair().expect("socketpair");
        let mut conn = Connection::from_stream(stream);
        // Reading the eventfd returns how many registrations fired since the last read.
        let mut wake = std::fs::File::from(ffi::dup_cloexec(es.wake_fd()).unwrap());
        let mut fired = || {
            let mut count = [0u8; 8];
            wake.read_exact(&mut count).expect("a registration fired");
            u64::from_ne_bytes(count)
        };
        let release = |es: &ExplicitSync, idx: usize| {
            let slot = &es.release[idx];
            let point = slot.point.expect("a published release point");
            ffi::drm_syncobj_signal_point(es.drm.as_raw_fd(), slot.syncobj, point, es.signalled)
                .expect("signal the release point");
        };

        es.publish(&mut conn, None, 0).expect("frame into buffer 0");
        es.publish(&mut conn, None, 1).expect("frame into buffer 1");
        for _ in 0..100 {
            assert!(matches!(es.free_slot(), Ok(None)), "both buffers are held");
            release(&es, 1);
            assert_eq!(fired(), 1, "buffer 1's release");
            es.woke();
            assert!(matches!(es.free_slot(), Ok(Some(1))), "buffer 1 is free");
            es.publish(&mut conn, None, 1).expect("frame into buffer 1");
        }

        release(&es, 0);
        assert_eq!(fired(), 1, "buffer 0's release fires one registration");
    }

    /// A resize while both old buffers are held and the wake is armed: when both new
    /// buffers are held too, a new buffer's release must still wake the loop.
    #[test]
    fn a_wake_armed_before_a_resize_still_wakes_on_a_new_buffers_release() {
        let Some(mut es) = explicit_sync("the resize wake test") else {
            return;
        };
        let (stream, _compositor) = UnixStream::pair().expect("socketpair");
        let mut conn = Connection::from_stream(stream);
        let woken = |es: &ExplicitSync| ffi::wait_sync_file(es.wake_fd(), 0);

        es.publish(&mut conn, None, 0)
            .expect("frame into old buffer 0");
        es.publish(&mut conn, None, 1)
            .expect("frame into old buffer 1");
        assert!(matches!(es.free_slot(), Ok(None)), "both old buffers held");

        es.forget_buffers();
        es.publish(&mut conn, None, 0)
            .expect("frame into new buffer 0");
        es.publish(&mut conn, None, 1)
            .expect("frame into new buffer 1");
        assert!(matches!(es.free_slot(), Ok(None)), "both new buffers held");

        let slot = &es.release[0];
        let point = slot.point.expect("a published release point");
        ffi::drm_syncobj_signal_point(es.drm.as_raw_fd(), slot.syncobj, point, es.signalled)
            .expect("signal the release point");
        assert!(woken(&es), "the new buffer's release wakes the loop");
    }

    #[test]
    fn sync_status_formats_without_allocating_a_string() {
        // The `[stats]` sync tail renders through Display straight into the log; pin
        // both shapes so the format cannot drift silently.
        let explicit = SyncStatus::Explicit {
            fence_frames: 12,
            cpu_wait_frames: 3,
        };
        assert_eq!(explicit.to_string(), "explicit-sync 12 fence/3 cpu-wait");
        assert_eq!(SyncStatus::Implicit.to_string(), "implicit-sync bridge");
    }

    /// The store-side sRGB encode the attachment applies, inverse of the decode in
    /// `clear_color_linear`.
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
            let lin = clear_color_linear(color);
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
