//! The platform layer: the platform-specific parts — talking to the compositor,
//! the input and font stacks, and the syscalls underneath.
//!
//! - **Wayland**: the wire codec (`wire`), the connection (`conn`), the
//!   hand-transcribed protocol constants (`protocol`), and dmabuf feedback
//!   (`dmabuf`).
//! - **Input**: keyboard translation via libxkbcommon (`xkb`).
//! - **Font**: FreeType rasterization (`freetype`), the colour-emoji face
//!   (`emoji`), HarfBuzz shaping/measurement (`shape`), UAX #29 grapheme
//!   segmentation (`grapheme`), and ARGB pixel math (`pixel`).
//! - **Foundation**: the syscall FFI surface (`ffi`), the minimal error type
//!   (`error`), branchless byte scanning (`bytes`), scroll math (`scroll`), and
//!   finding a URL in text (`link`) so it can be opened in the browser (`browser`).

pub(crate) mod browser;
pub(crate) mod bytes;
pub(crate) mod conn;
pub(crate) mod dmabuf;
pub(crate) mod emoji;
pub(crate) mod error;
pub(crate) mod ffi;
pub(crate) mod freetype;
pub(crate) mod geom;
pub(crate) mod grapheme;
pub(crate) mod link;
pub(crate) mod pixel;
pub(crate) mod protocol;
pub(crate) mod scroll;
pub(crate) mod shape;
pub(crate) mod wire;
pub(crate) mod xkb;

#[cfg(test)]
mod tests {
    /// The platform layer must stay a clean leaf: a module here may name only
    /// `crate::platform::*` and `std`, never `crate::app`, `crate::render::gpu`,
    /// or anything else above it. Rust checks no such direction inside one crate,
    /// so this walks the sources and does.
    #[test]
    fn platform_only_depends_on_itself() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/platform");
        let mut offenders = Vec::new();
        for entry in std::fs::read_dir(dir).expect("read src/platform") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("read platform source");
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            for (line_no, line) in src.lines().enumerate() {
                // Doc and line comments may cross-reference any module freely.
                if line.trim_start().starts_with("//") {
                    continue;
                }
                for (idx, _) in line.match_indices("crate::") {
                    let after = &line[idx + "crate::".len()..];
                    let module: String = after
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !module.is_empty() && module != "platform" {
                        offenders.push(format!("{name}:{}: crate::{module}", line_no + 1));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "platform modules must not reach outside the layer:\n{}",
            offenders.join("\n")
        );
    }
}
