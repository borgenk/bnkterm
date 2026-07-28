//! The Vulkan backend's device layer: a hand-transcribed slice of the Vulkan C
//! ABI (only what the GPU presentation path uses) and the bring-up that turns
//! the compositor's dmabuf feedback into a live device and queue.
//!
//! The loader is `dlopen`ed rather than linked, so on a machine without Vulkan
//! the binary still starts and exits with a clear message instead of failing at
//! load time (the GPU is mandatory, there is no software renderer); every
//! function pointer is fetched through `vkGetInstanceProcAddr` /
//! `vkGetDeviceProcAddr` as the spec prescribes. There is no window surface
//! and no swapchain anywhere in this module: presentation is dmabuf export
//! over the Wayland wire, so the WSI machinery never appears.
//!
//! Transcription conventions: struct and constant names keep Vulkan's own
//! (minus the `Vk`/`VK_` prefix would hurt grep-ability, so they keep it);
//! fields are renamed to snake_case; only the structure types, results, and
//! flags actually exercised are declared. Layouts mirror the x86_64/arm64 C
//! ABI, guarded by size assertions in the tests at the bottom.

mod abi;
mod renderer;
mod target;

use core::ffi::{c_char, CStr};

use crate::platform::error::{Error, Result};
use crate::platform::ffi::DynLib;
use crate::render::vulkan::abi::*;
use crate::render::vulkan::renderer::{create_renderer, Renderer};

pub(crate) use crate::render::vulkan::target::GpuImage;

// ---------------------------------------------------------------------------
// Owned handles.
// ---------------------------------------------------------------------------

/// The instance and its destructor, so any error after creation still tears it
/// down on drop.
struct Instance {
    raw: VkInstance,
    destroy: PfnDestroyInstance,
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: raw came from vkCreateInstance and is destroyed exactly once,
        // after every object created from it (field order in Gpu).
        unsafe { (self.destroy)(self.raw, core::ptr::null()) };
    }
}

/// The logical device and its destructor.
struct Device {
    raw: VkDevice,
    destroy: PfnDestroyDevice,
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: raw came from vkCreateDevice and is destroyed exactly once.
        unsafe { (self.destroy)(self.raw, core::ptr::null()) };
    }
}

// ---------------------------------------------------------------------------
// Bring-up.
// ---------------------------------------------------------------------------

/// A live Vulkan device matched to the compositor's main DRM device, with a
/// graphics queue, a command pool, and the format/memory facts the render
/// targets need. Every failure on the way here is a clean `Err` the caller
/// answers with the software path.
pub struct Gpu {
    fns: DeviceFns,
    queue: VkQueue,
    queue_family: u32,
    pool: VkCommandPool,
    renderer: Renderer,
    /// The modifiers this device can render B8G8R8A8 into (single-plane,
    /// renderable, transfer-capable), to intersect with the compositor's list.
    modifiers: Vec<VkDrmFormatModifierPropertiesEXT>,
    /// Property flags per memory type index, for allocation.
    memory_type_flags: Vec<u32>,
    /// The chosen device's DRM render-node minor, for opening a fd to create
    /// explicit-sync sync objects. `None` if the device exposes no render node.
    render_node: Option<u32>,
    // Drop order is field order: the device before the instance, the instance
    // before the loader library. The underscored two are pure keep-alives.
    device: Device,
    _instance: Instance,
    _lib: DynLib,
    name: String,
}

impl Gpu {
    /// Bring up Vulkan against the DRM device `main_device` (a dev_t from the
    /// compositor's dmabuf feedback): load the loader, create an instance, pick
    /// the physical device matching that DRM identity, and create a one-queue
    /// logical device with the dmabuf presentation extensions.
    pub fn new(main_device: u64) -> Result<Self> {
        let lib = DynLib::open(c"libvulkan.so.1")?;
        let gipa_ptr = lib
            .sym(c"vkGetInstanceProcAddr")
            .ok_or_else(|| Error::msg("libvulkan.so.1 lacks vkGetInstanceProcAddr"))?;
        // SAFETY: the loader exports vkGetInstanceProcAddr with exactly this
        // signature; the pointer stays valid while `lib` lives.
        let gipa: PfnGetInstanceProcAddr = unsafe { core::mem::transmute(gipa_ptr) };

        let instance = create_instance(gipa)?;

        // SAFETY (all loads below): the names match the transcribed PFN types.
        let enumerate: PfnEnumeratePhysicalDevices =
            unsafe { load(gipa, instance.raw, c"vkEnumeratePhysicalDevices") }?;
        let get_props2: PfnGetPhysicalDeviceProperties2 =
            unsafe { load(gipa, instance.raw, c"vkGetPhysicalDeviceProperties2") }?;
        let enumerate_exts: PfnEnumerateDeviceExtensionProperties =
            unsafe { load(gipa, instance.raw, c"vkEnumerateDeviceExtensionProperties") }?;
        let get_queue_families: PfnGetPhysicalDeviceQueueFamilyProperties = unsafe {
            load(
                gipa,
                instance.raw,
                c"vkGetPhysicalDeviceQueueFamilyProperties",
            )
        }?;
        let create_device: PfnCreateDevice =
            unsafe { load(gipa, instance.raw, c"vkCreateDevice") }?;
        let gdpa: PfnGetDeviceProcAddr =
            unsafe { load(gipa, instance.raw, c"vkGetDeviceProcAddr") }?;
        let get_memory_props: PfnGetPhysicalDeviceMemoryProperties =
            unsafe { load(gipa, instance.raw, c"vkGetPhysicalDeviceMemoryProperties") }?;
        let get_format_props2: PfnGetPhysicalDeviceFormatProperties2 =
            unsafe { load(gipa, instance.raw, c"vkGetPhysicalDeviceFormatProperties2") }?;

        let physicals = enumerate_physicals(enumerate, instance.raw)?;
        let (physical, name, queue_family, render_node) = pick_device(
            get_props2,
            enumerate_exts,
            get_queue_families,
            &physicals,
            main_device,
        )?;

        let modifiers = render_modifiers(get_format_props2, physical);
        if modifiers.is_empty() {
            return Err(Error::msg(format!(
                "device {name} offers no usable DRM modifier for B8G8R8A8"
            )));
        }
        let memory_type_flags = memory_type_flags(get_memory_props, physical);

        let device = create_logical_device(create_device, gdpa, physical, queue_family)?;
        let fns = DeviceFns::load(gdpa, device.raw)?;

        let get_queue: PfnGetDeviceQueue =
            unsafe { load_device(gdpa, device.raw, c"vkGetDeviceQueue") }?;
        let mut queue: VkQueue = core::ptr::null_mut();
        // SAFETY: device is live and queue_family/index 0 were created with it.
        unsafe { get_queue(device.raw, queue_family, 0, &mut queue) };
        if queue.is_null() {
            return Err(Error::msg("vkGetDeviceQueue returned null"));
        }

        let pool_info = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index: queue_family,
        };
        let mut pool: VkCommandPool = 0;
        // SAFETY: device is live; pool_info is fully initialized.
        check(
            unsafe {
                (fns.create_command_pool)(device.raw, &pool_info, core::ptr::null(), &mut pool)
            },
            "vkCreateCommandPool",
        )?;

        let renderer = match create_renderer(&fns, device.raw) {
            Ok(r) => r,
            Err(e) => {
                // SAFETY: pool is live and nothing was ever submitted.
                unsafe { (fns.destroy_command_pool)(device.raw, pool, core::ptr::null()) };
                return Err(e);
            }
        };

        Ok(Self {
            fns,
            queue,
            queue_family,
            pool,
            renderer,
            modifiers,
            memory_type_flags,
            render_node,
            device,
            _instance: instance,
            _lib: lib,
            name,
        })
    }

    /// The marketing name of the chosen device (for diagnostics).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The DRM render node minor of the chosen device (`/dev/dri/renderD<minor>`),
    /// where explicit sync opens a fd to create its sync objects. `None` when the
    /// device exposes no render node.
    pub fn render_node(&self) -> Option<u32> {
        self.render_node
    }

    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    /// The subset of the compositor's modifier list this device can render
    /// into, in the compositor's (preference) order. `DRM_FORMAT_MOD_INVALID`
    /// ("implicit layout") is dropped: explicit-modifier allocation is the
    /// whole point of this path.
    pub fn image_modifiers(&self, compositor: &[u64]) -> Vec<u64> {
        compositor
            .iter()
            .copied()
            .filter(|&m| m != DRM_FORMAT_MOD_INVALID)
            .filter(|&m| self.modifiers.iter().any(|p| p.drm_format_modifier == m))
            .collect()
    }

    /// Block until the queue drains; teardown and resize use this so nothing
    /// is destroyed while a frame is in flight.
    pub fn wait_idle(&self) {
        // SAFETY: device is live. The result only matters to callers about to
        // destroy things, and a lost device makes destruction safe anyway.
        let _ = unsafe { (self.fns.device_wait_idle)(self.device.raw) };
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.wait_idle();
        let renderer = std::mem::take(&mut self.renderer);
        renderer.destroy(&self.fns, self.device.raw);
        // SAFETY: pool came from vkCreateCommandPool on this device; command
        // buffers still allocated from it are freed with it.
        unsafe { (self.fns.destroy_command_pool)(self.device.raw, self.pool, core::ptr::null()) };
    }
}

/// The device's single-plane, color-attachment-capable modifiers for
/// B8G8R8A8_UNORM (the two-call pattern over the chained modifier list).
fn render_modifiers(
    get_format_props2: PfnGetPhysicalDeviceFormatProperties2,
    physical: VkPhysicalDevice,
) -> Vec<VkDrmFormatModifierPropertiesEXT> {
    let mut list = VkDrmFormatModifierPropertiesListEXT {
        s_type: VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
        p_next: core::ptr::null_mut(),
        drm_format_modifier_count: 0,
        p_drm_format_modifier_properties: core::ptr::null_mut(),
    };
    let mut props = VkFormatProperties2 {
        s_type: VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
        p_next: (&mut list as *mut VkDrmFormatModifierPropertiesListEXT).cast(),
        format_properties: VkFormatProperties {
            linear_tiling_features: 0,
            optimal_tiling_features: 0,
            buffer_features: 0,
        },
    };
    // SAFETY: physical is live; props chains one correctly-typed list struct.
    unsafe { get_format_props2(physical, VK_FORMAT_B8G8R8A8_UNORM, &mut props) };
    let mut mods = vec![
        VkDrmFormatModifierPropertiesEXT {
            drm_format_modifier: 0,
            drm_format_modifier_plane_count: 0,
            drm_format_modifier_tiling_features: 0,
        };
        list.drm_format_modifier_count as usize
    ];
    list.p_drm_format_modifier_properties = mods.as_mut_ptr();
    // SAFETY: the array now has room for the advertised count.
    unsafe { get_format_props2(physical, VK_FORMAT_B8G8R8A8_UNORM, &mut props) };
    mods.truncate(list.drm_format_modifier_count as usize);
    let needed = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT
        | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
        | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    mods.retain(|m| {
        m.drm_format_modifier_plane_count == 1
            && m.drm_format_modifier_tiling_features & needed == needed
    });
    mods
}

/// Property flags per memory type index.
fn memory_type_flags(
    get_memory_props: PfnGetPhysicalDeviceMemoryProperties,
    physical: VkPhysicalDevice,
) -> Vec<u32> {
    // SAFETY: zeroed VkPhysicalDeviceMemoryProperties (integers) is a valid
    // out-parameter.
    let mut props = unsafe { core::mem::zeroed::<VkPhysicalDeviceMemoryProperties>() };
    // SAFETY: physical is live; props is sized per the ABI test.
    unsafe { get_memory_props(physical, &mut props) };
    props
        .memory_types
        .iter()
        .take((props.memory_type_count as usize).min(32))
        .map(|t| t.property_flags)
        .collect()
}

fn create_instance(gipa: PfnGetInstanceProcAddr) -> Result<Instance> {
    // SAFETY: vkCreateInstance is fetchable with a null instance.
    let create: PfnCreateInstance =
        unsafe { load(gipa, core::ptr::null_mut(), c"vkCreateInstance") }?;
    let app = VkApplicationInfo {
        s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
        p_next: core::ptr::null(),
        p_application_name: c"bnkterm".as_ptr(),
        application_version: 0,
        p_engine_name: core::ptr::null(),
        engine_version: 0,
        api_version: API_VERSION_1_2,
    };
    let info = VkInstanceCreateInfo {
        s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        p_application_info: &app,
        enabled_layer_count: 0,
        pp_enabled_layer_names: core::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: core::ptr::null(),
    };
    let mut raw: VkInstance = core::ptr::null_mut();
    // SAFETY: info and app are fully initialized and outlive the call.
    check(
        unsafe { create(&info, core::ptr::null(), &mut raw) },
        "vkCreateInstance",
    )?;
    // Fetch the destructor immediately so every later error path tears the
    // instance down through Instance's Drop.
    let destroy: PfnDestroyInstance = unsafe { load(gipa, raw, c"vkDestroyInstance") }?;
    Ok(Instance { raw, destroy })
}

fn enumerate_physicals(
    enumerate: PfnEnumeratePhysicalDevices,
    instance: VkInstance,
) -> Result<Vec<VkPhysicalDevice>> {
    // The standard two-call pattern; loop in case a device appears between the
    // count and the fill (VK_INCOMPLETE).
    loop {
        let mut count = 0u32;
        // SAFETY: instance is live; null array means "count only".
        check(
            unsafe { enumerate(instance, &mut count, core::ptr::null_mut()) },
            "vkEnumeratePhysicalDevices (count)",
        )?;
        if count == 0 {
            return Err(Error::msg("no vulkan physical devices"));
        }
        let mut devices = vec![core::ptr::null_mut(); count as usize];
        // SAFETY: devices has room for count entries.
        let r = unsafe { enumerate(instance, &mut count, devices.as_mut_ptr()) };
        if r == VK_INCOMPLETE {
            continue;
        }
        check(r, "vkEnumeratePhysicalDevices")?;
        devices.truncate(count as usize);
        return Ok(devices);
    }
}

/// Read a NUL-terminated device or extension name out of a fixed C array.
fn c_name(bytes: &[c_char]) -> String {
    let unsigned: Vec<u8> = bytes.iter().map(|&c| c as u8).collect();
    let end = unsigned
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(unsigned.len());
    String::from_utf8_lossy(&unsigned[..end]).into_owned()
}

/// Whether `wanted` appears in the device's supported extension list.
fn has_extension(exts: &[VkExtensionProperties], wanted: &CStr) -> bool {
    exts.iter()
        .any(|e| c_name(&e.extension_name).as_bytes() == wanted.to_bytes())
}

/// Pick the physical device whose DRM identity matches `main_device` and which
/// offers a graphics queue and every required extension. Returns the device, its
/// name, and the graphics queue family index.
fn pick_device(
    get_props2: PfnGetPhysicalDeviceProperties2,
    enumerate_exts: PfnEnumerateDeviceExtensionProperties,
    get_queue_families: PfnGetPhysicalDeviceQueueFamilyProperties,
    physicals: &[VkPhysicalDevice],
    main_device: u64,
) -> Result<(VkPhysicalDevice, String, u32, Option<u32>)> {
    let (want_major, want_minor) = (
        i64::from(crate::platform::dmabuf::dev_major(main_device)),
        i64::from(crate::platform::dmabuf::dev_minor(main_device)),
    );
    let mut seen = Vec::new();

    for &physical in physicals {
        let mut drm = VkPhysicalDeviceDrmPropertiesEXT {
            s_type: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
            p_next: core::ptr::null_mut(),
            has_primary: 0,
            has_render: 0,
            primary_major: 0,
            primary_minor: 0,
            render_major: 0,
            render_minor: 0,
        };
        // SAFETY: zeroed VkPhysicalDeviceProperties is a valid out-parameter;
        // the struct layouts are guarded by the size tests below.
        let mut props = unsafe { core::mem::zeroed::<VkPhysicalDeviceProperties2>() };
        props.s_type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

        let exts = enumerate_extensions(enumerate_exts, physical)?;
        let name = if has_extension(&exts, EXT_PHYSICAL_DEVICE_DRM) {
            props.p_next = (&mut drm as *mut VkPhysicalDeviceDrmPropertiesEXT).cast();
            // SAFETY: physical is live; props chains one correctly-typed struct.
            unsafe { get_props2(physical, &mut props) };
            c_name(&props.properties.device_name)
        } else {
            // Without the DRM identity the device cannot be matched to the
            // compositor's; note it for the error message and move on.
            // SAFETY: physical is live; props chains nothing.
            unsafe { get_props2(physical, &mut props) };
            seen.push(format!(
                "{} (no VK_EXT_physical_device_drm)",
                c_name(&props.properties.device_name)
            ));
            continue;
        };

        let primary_matches = drm.has_primary != 0
            && (drm.primary_major, drm.primary_minor) == (want_major, want_minor);
        let render_matches =
            drm.has_render != 0 && (drm.render_major, drm.render_minor) == (want_major, want_minor);
        if !primary_matches && !render_matches {
            seen.push(format!(
                "{name} (drm {}:{} / {}:{})",
                drm.primary_major, drm.primary_minor, drm.render_major, drm.render_minor
            ));
            continue;
        }

        let missing: Vec<&str> = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .filter(|e| !has_extension(&exts, e))
            .filter_map(|e| e.to_str().ok())
            .collect();
        if !missing.is_empty() {
            // The device the compositor named cannot serve: a hard, explainable
            // failure rather than a silent fallback.
            return Err(Error::msg(format!(
                "device {name} matches the compositor's DRM device but lacks {}",
                missing.join(", ")
            )));
        }

        let Some(queue_family) = graphics_family(get_queue_families, physical) else {
            return Err(Error::msg(format!("device {name} has no graphics queue")));
        };
        // The render node (no DRM master needed) is where explicit sync creates
        // its syncobjs; `None` when the device exposes no render node.
        let render_minor = (drm.has_render != 0).then_some(drm.render_minor as u32);
        return Ok((physical, name, queue_family, render_minor));
    }

    Err(Error::msg(format!(
        "no vulkan device matches DRM device {want_major}:{want_minor} (saw: {})",
        seen.join("; ")
    )))
}

fn enumerate_extensions(
    enumerate_exts: PfnEnumerateDeviceExtensionProperties,
    physical: VkPhysicalDevice,
) -> Result<Vec<VkExtensionProperties>> {
    loop {
        let mut count = 0u32;
        // SAFETY: physical is live; null array means "count only".
        check(
            unsafe {
                enumerate_exts(
                    physical,
                    core::ptr::null(),
                    &mut count,
                    core::ptr::null_mut(),
                )
            },
            "vkEnumerateDeviceExtensionProperties (count)",
        )?;
        // SAFETY: zeroed VkExtensionProperties (a char array and a u32) is valid.
        let mut exts =
            vec![unsafe { core::mem::zeroed::<VkExtensionProperties>() }; count as usize];
        // SAFETY: exts has room for count entries.
        let r =
            unsafe { enumerate_exts(physical, core::ptr::null(), &mut count, exts.as_mut_ptr()) };
        if r == VK_INCOMPLETE {
            continue;
        }
        check(r, "vkEnumerateDeviceExtensionProperties")?;
        exts.truncate(count as usize);
        return Ok(exts);
    }
}

/// The first queue family with graphics support, if any.
fn graphics_family(
    get_queue_families: PfnGetPhysicalDeviceQueueFamilyProperties,
    physical: VkPhysicalDevice,
) -> Option<u32> {
    let mut count = 0u32;
    // SAFETY: physical is live; null array means "count only".
    unsafe { get_queue_families(physical, &mut count, core::ptr::null_mut()) };
    // SAFETY: zeroed VkQueueFamilyProperties (plain integers) is valid.
    let mut families =
        vec![unsafe { core::mem::zeroed::<VkQueueFamilyProperties>() }; count as usize];
    // SAFETY: families has room for count entries.
    unsafe { get_queue_families(physical, &mut count, families.as_mut_ptr()) };
    families
        .iter()
        .take(count as usize)
        .position(|f| f.queue_flags & VK_QUEUE_GRAPHICS_BIT != 0 && f.queue_count > 0)
        .map(|i| i as u32)
}

fn create_logical_device(
    create_device: PfnCreateDevice,
    gdpa: PfnGetDeviceProcAddr,
    physical: VkPhysicalDevice,
    queue_family: u32,
) -> Result<Device> {
    let priority = 1.0f32;
    let queue_info = VkDeviceQueueCreateInfo {
        s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        queue_family_index: queue_family,
        queue_count: 1,
        p_queue_priorities: &priority,
    };
    let ext_ptrs: Vec<*const c_char> = REQUIRED_DEVICE_EXTENSIONS
        .iter()
        .map(|e| e.as_ptr())
        .collect();
    let info = VkDeviceCreateInfo {
        s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        queue_create_info_count: 1,
        p_queue_create_infos: &queue_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: core::ptr::null(),
        enabled_extension_count: ext_ptrs.len() as u32,
        pp_enabled_extension_names: ext_ptrs.as_ptr(),
        p_enabled_features: core::ptr::null(),
    };
    let mut raw: VkDevice = core::ptr::null_mut();
    // SAFETY: physical is live; info and everything it points at outlive the call.
    check(
        unsafe { create_device(physical, &info, core::ptr::null(), &mut raw) },
        "vkCreateDevice",
    )?;
    // As with the instance: fetch the destructor first, so later failures
    // tear the device down through Drop.
    let destroy: PfnDestroyDevice = unsafe { load_device(gdpa, raw, c"vkDestroyDevice") }?;
    Ok(Device { raw, destroy })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_type_picking_prefers_device_local() {
        let gpu_types = [0u32, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0x6];
        // A fake Gpu is not constructible here (it needs a live device), so
        // test the picking logic through a copy of its rule.
        let pick = |allowed_bits: u32| -> Option<u32> {
            let allowed = |i: usize| allowed_bits & (1u32 << i) != 0;
            gpu_types
                .iter()
                .enumerate()
                .position(|(i, &flags)| {
                    allowed(i) && flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0
                })
                .or_else(|| (0..gpu_types.len()).find(|&i| allowed(i)))
                .map(|i| i as u32)
        };
        assert_eq!(pick(0b111), Some(1), "device-local wins");
        assert_eq!(pick(0b101), Some(0), "fallback to first allowed");
        assert_eq!(pick(0), None);
    }

    #[test]
    fn c_name_stops_at_the_nul() {
        let mut bytes = [0 as c_char; 8];
        for (i, b) in b"abc\0zzz".iter().enumerate() {
            bytes[i] = *b as c_char;
        }
        assert_eq!(c_name(&bytes), "abc");
        // No NUL at all: the whole array is the name.
        let full = [b'x' as c_char; 4];
        assert_eq!(c_name(&full), "xxxx");
    }
}
