//! The GPU presentation layer: everything about turning a frame's worth of
//! drawing commands into pixels on the compositor's surface.
//!
//! - `display`: the drawing vocabulary — `DrawCmd`, the `DisplayList` the app
//!   builds each frame, and the damage diff between frames. These are concrete,
//!   fixed types (no generics): the app builds a `DisplayList` however it likes
//!   and the rest of this layer consumes it identically, so the renderer stays
//!   agnostic to whatever produced the list.
//! - `gpu`: batching a `DisplayList` into vertices against the glyph/image
//!   atlases (the CPU half of a frame).
//! - `vulkan`: the device backend — bring-up, the exported dmabuf render
//!   targets, explicit sync, and the pipeline (the GPU half).
//!
//! (There is no image module: a terminal decodes no images, so the
//! `DrawCmd::Image` primitive, its GPU batch path, and the Vulkan image
//! descriptor set are all absent, keeping bnkterm zero-crate.)
//!
//! Like `platform`, this is a clean leaf: it depends only on `platform` and
//! itself, never on the app above it (the test at the bottom enforces that). An
//! app sits on top and hands it a `DisplayList`.

pub(crate) mod boxdraw;
pub(crate) mod display;
pub(crate) mod gpu;
pub(crate) mod vulkan;

#[cfg(test)]
mod tests {
    /// The render layer must depend only on `platform` and itself so it stays a
    /// liftable leaf: a module here may name `crate::render::*`
    /// or `crate::platform::*` and `std`, never the app above it. Rust checks no
    /// such direction within a crate, so this walks the sources (including the
    /// nested `vulkan/`) and does.
    #[test]
    fn render_only_depends_on_platform_and_itself() {
        let mut stack = vec![std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/render"
        ))];
        let mut offenders = Vec::new();
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("read src/render") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let src = std::fs::read_to_string(&path).expect("read render source");
                let name = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                for (line_no, line) in src.lines().enumerate() {
                    if line.trim_start().starts_with("//") {
                        continue;
                    }
                    for (idx, _) in line.match_indices("crate::") {
                        let after = &line[idx + "crate::".len()..];
                        let module: String = after
                            .chars()
                            .take_while(|c| c.is_alphanumeric() || *c == '_')
                            .collect();
                        if !module.is_empty() && module != "render" && module != "platform" {
                            offenders.push(format!("{name}:{}: crate::{module}", line_no + 1));
                        }
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "render must depend only on itself and platform:\n{}",
            offenders.join("\n")
        );
    }
}
