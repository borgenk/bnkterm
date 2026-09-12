//! One frame of the terminal, saved as a PNG.
//!
//! `make screenshot` opens the demo window, captures the frame off the GPU, and
//! writes it to `assets/screenshot.png` for the README and the AppStream metainfo.
//! The demo grid rather than a live shell, so the image does not depend on whatever
//! the user's shell happened to print. The window is held at the demo's size
//! (`State::capture_cells`), so the asset is the same shape on any desktop.
//!
//! There is no runtime path to this. Capturing is reached only from the ignored test
//! below, so nothing a child prints can make the terminal write a file.

use std::io;
use std::path::{Path, PathBuf};

use crate::dev::png;

/// The radius carved off the corners. The frame the GPU draws is square.
const CORNER_RADIUS: f32 = 4.0;

/// Encode `pixels` as a PNG at `path`, creating the directory above it if needed.
///
/// The surface is `XRGB8888`, so the top byte is not colour: it is forced opaque,
/// then the corners are carved, which is the only alpha the file carries.
pub(crate) fn write(path: &Path, pixels: &[u32], width: u32, height: u32) -> io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut out: Vec<u32> = pixels.iter().map(|p| p | 0xff00_0000).collect();
    round_corners(&mut out, width, height, CORNER_RADIUS);
    std::fs::write(path, png::encode(&out, width, height))
}

/// Fade the alpha outside a rounded rectangle, over the distance to the corner's
/// circle so the curve stays smooth rather than stepping.
fn round_corners(pixels: &mut [u32], width: u32, height: u32, radius: f32) {
    if radius < 1.0 {
        return;
    }
    let (w, h) = (width as f32, height as f32);
    let span = radius.ceil() as u32;
    for y in 0..height {
        if y >= span && y < height - span {
            continue;
        }
        for x in 0..width {
            if x >= span && x < width - span {
                continue;
            }
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            // How far past the corner circle the pixel sits.
            let dx = (radius - px).max(px - (w - radius)).max(0.0);
            let dy = (radius - py).max(py - (h - radius)).max(0.0);
            let cover = (0.5 - (dx.hypot(dy) - radius)).clamp(0.0, 1.0);
            let Some(slot) = pixels.get_mut((y * width + x) as usize) else {
                continue;
            };
            let alpha = ((*slot >> 24) as f32 * cover).round() as u32;
            *slot = (alpha << 24) | (*slot & 0x00ff_ffff);
        }
    }
}

/// Where a screenshot lands: the committed asset, or wherever `BNKTERM_SCREENSHOT`
/// points when someone wants one somewhere else.
fn destination() -> PathBuf {
    path_for(std::env::var_os("BNKTERM_SCREENSHOT").map(PathBuf::from))
}

/// The destination rule, with the override passed in so it is testable without
/// touching the process environment.
fn path_for(explicit: Option<PathBuf>) -> PathBuf {
    explicit.unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join("screenshot.png")
    })
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use crate::dev::screenshot::*;

    #[test]
    fn an_explicit_path_wins_over_the_committed_asset() {
        let got = path_for(Some(PathBuf::from("/tmp/elsewhere.png")));
        assert_eq!(got, PathBuf::from("/tmp/elsewhere.png"));
    }

    #[test]
    fn the_default_is_the_committed_asset() {
        assert!(path_for(None).ends_with("assets/screenshot.png"));
    }

    /// The corners lose their alpha and the middle keeps it, which is what makes a
    /// square frame read as a rounded window.
    #[test]
    fn the_corners_are_rounded_and_the_middle_is_left_alone() {
        let (w, h) = (32u32, 16u32);
        let mut pixels = vec![0xff33_4455; (w * h) as usize];
        round_corners(&mut pixels, w, h, 4.0);

        let alpha = |x: u32, y: u32| pixels[(y * w + x) as usize] >> 24;
        assert_eq!(alpha(0, 0), 0, "the very corner is cut away");
        assert_eq!(alpha(w - 1, 0), 0, "every corner, not just the first");
        assert_eq!(alpha(0, h - 1), 0);
        assert_eq!(alpha(w - 1, h - 1), 0);
        assert_eq!(alpha(w / 2, h / 2), 0xff, "the middle is untouched");
        assert_eq!(alpha(w / 2, 0), 0xff, "a straight edge is not a corner");
        assert_eq!(
            pixels[(h / 2 * w + w / 2) as usize] & 0x00ff_ffff,
            0x0033_4455,
            "colour is never touched, only alpha"
        );
    }

    /// A radius under a pixel is a square frame, and asking for one changes nothing.
    #[test]
    fn a_radius_below_one_pixel_leaves_the_frame_square() {
        let mut pixels = vec![0xff33_4455; 64];
        let before = pixels.clone();
        round_corners(&mut pixels, 8, 8, 0.0);
        assert_eq!(pixels, before);
    }

    /// Pixels in, a PNG on disk. The encoder's own round-trip lives in `png`; this
    /// is the file-handling half, including creating the directory above it.
    #[test]
    fn a_capture_lands_on_disk_as_a_png() {
        let dir = std::env::temp_dir().join("bnkterm_screenshot_test");
        let path = dir.join("nested").join("shot.png");
        let _ = std::fs::remove_dir_all(&dir);
        write(&path, &[0xff00_ff00; 4], 2, 2).expect("write");
        let bytes = std::fs::read(&path).expect("read back");
        assert_eq!(&bytes[1..4], b"PNG");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn a_directory_destination_reports_a_write_error() {
        let dir = std::env::temp_dir().join(format!(
            "bnkterm_screenshot_directory_{}",
            std::process::id()
        ));
        std::fs::create_dir(&dir).expect("create destination directory");
        let result = write(&dir, &[0xff00_ff00; 4], 2, 2);
        std::fs::remove_dir(&dir).expect("cleanup");
        assert_eq!(
            result.expect_err("writing a directory must fail").kind(),
            io::ErrorKind::IsADirectory
        );
    }

    /// The whole capture must report the write failure, even with a readable PNG
    /// already at the destination. Checking the I/O error also rules out a failure
    /// to bring up the GPU or connect to the compositor passing this regression.
    #[test]
    #[ignore = "needs Wayland, Vulkan, and an unprivileged user for file permissions"]
    fn a_failed_capture_does_not_accept_an_old_png() {
        let dir = std::env::temp_dir().join(format!(
            "bnkterm_screenshot_readonly_{}",
            std::process::id()
        ));
        std::fs::create_dir(&dir).expect("create capture directory");
        let path = dir.join("shot.png");
        let original = png::encode(&[0xff00_ff00; 4], 2, 2);
        std::fs::write(&path, &original).expect("write the old PNG");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444))
            .expect("make the old PNG read-only");

        let result = crate::app::capture_demo_frame(&path);
        let after = std::fs::read(&path).expect("read the old PNG");
        std::fs::remove_dir_all(&dir).expect("cleanup");

        let error = result.expect_err("the old PNG cannot stand in for a failed capture");
        assert!(
            error.to_string().contains("io error: Permission denied"),
            "{error}"
        );
        assert_eq!(after, original, "the read-only PNG is still the old image");
    }

    /// Capture the real window, which is what `make screenshot` runs. Ignored
    /// otherwise: it needs a Wayland session and a Vulkan device, and the glyphs it
    /// draws depend on the fonts installed here.
    #[test]
    #[ignore]
    fn write_screenshot() {
        let path = destination();
        crate::app::capture_demo_frame(&path).expect("capture the demo frame");
        let bytes = std::fs::read(&path).expect("read the capture back");
        assert_eq!(&bytes[1..4], b"PNG", "what landed is a PNG");
        println!("wrote {} ({} bytes)", path.display(), bytes.len());
    }
}
