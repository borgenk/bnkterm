//! The on-disk config: one field per line, read once at startup.
//!
//! ```text
//!   ~/.config/bnkterm/config ─▶ parse ─┬─▶ Config   (defaults, overridden line by line)
//!                                      └─▶ Problems (what was ignored, and why)
//! ```
//!
//! Each non-comment line contains a field and value separated by whitespace. A missing
//! file uses the compiled defaults. Invalid lines are reported as [`Problem`]s without
//! preventing valid lines from loading.
//!
//! Colors use the same X11 syntax as OSC color commands: `#rgb`, `#rrggbb`,
//! `#rrrrggggbbbb`, or `rgb:aa/bb/cc`.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::color::{parse_x11_color, Rgb};
use crate::config::{Config, TabBarPosition, FONT_SIZE_RANGE};

/// A config line the loader could not use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// 1-based line number, or 0 when the problem is with the file as a whole.
    pub line: usize,
    pub message: String,
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            0 => write!(f, "config: {}", self.message),
            line => write!(f, "config:{line}: {}", self.message),
        }
    }
}

/// The config path under an absolute `$XDG_CONFIG_HOME` or `$HOME`. Inside a Flatpak
/// sandbox `$XDG_CONFIG_HOME` is the app's private directory, so the host's own, when it
/// set one, comes from `$HOST_XDG_CONFIG_HOME` instead.
pub fn path() -> Option<PathBuf> {
    let config_home = if crate::flatpak::sandboxed() {
        "HOST_XDG_CONFIG_HOME"
    } else {
        "XDG_CONFIG_HOME"
    };
    path_from(std::env::var_os(config_home), std::env::var_os("HOME"))
}

/// [`path`] from the two variables it reads.
fn path_from(
    config_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    let base = config_home
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute())
        .or_else(|| {
            home.map(PathBuf::from)
                .filter(|dir| dir.is_absolute())
                .map(|home| home.join(".config"))
        })?;
    Some(base.join("bnkterm").join("config"))
}

/// The user's config, or the defaults when there is none.
///
/// An unreadable existing file returns one file-level problem.
pub fn load() -> (Config, Vec<Problem>) {
    let Some(path) = path() else {
        return (Config::default(), Vec::new());
    };
    load_from(&path)
}

fn load_from(path: &Path) -> (Config, Vec<Problem>) {
    match std::fs::read_to_string(path) {
        Ok(text) => parse(&text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            (Config::default(), Vec::new())
        }
        Err(error) => (
            Config::default(),
            vec![Problem {
                line: 0,
                message: format!("cannot read {}: {error}", path.display()),
            }],
        ),
    }
}

/// Parse config text over the compiled defaults.
pub fn parse(text: &str) -> (Config, Vec<Problem>) {
    let mut config = Config::default();
    let mut problems = Vec::new();
    // The first list entry replaces defaults; later entries append in file order.
    let mut replaced: Vec<&str> = Vec::new();

    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let content = strip_trailing_comment(trimmed).trim_end();
        if content.is_empty() {
            continue;
        }
        let (field, value) = match content.split_once(char::is_whitespace) {
            Some((field, value)) => (field, value.trim()),
            None => {
                problems.push(Problem {
                    line,
                    message: format!("{content:?} has no value"),
                });
                continue;
            }
        };
        if value.is_empty() {
            problems.push(Problem {
                line,
                message: format!("{field:?} has no value"),
            });
            continue;
        }
        if let Err(message) = apply(&mut config, &mut replaced, field, value) {
            problems.push(Problem { line, message });
        }
    }
    (config, problems)
}

/// Drop a trailing comment without treating a `#rgb` value as one.
///
/// A comment marker follows whitespace and precedes whitespace or the end of the line.
/// The caller handles markers at the start of a line.
fn strip_trailing_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    for (i, byte) in bytes.iter().enumerate().skip(1) {
        if *byte != b'#' {
            continue;
        }
        let after_space = bytes[i - 1].is_ascii_whitespace();
        let before_space = bytes.get(i + 1).is_none_or(u8::is_ascii_whitespace);
        if after_space && before_space {
            return &line[..i];
        }
    }
    line
}

fn apply(
    config: &mut Config,
    replaced: &mut Vec<&'static str>,
    field: &str,
    value: &str,
) -> Result<(), String> {
    let list = match field {
        "font" => Some(&mut config.fonts.families),
        "ui_font" => Some(&mut config.fonts.ui),
        "code_font" => Some(&mut config.fonts.code),
        "fallback_font" => Some(&mut config.fonts.fallback),
        _ => None,
    };
    if let Some(list) = list {
        let key = match field {
            "font" => "font",
            "ui_font" => "ui_font",
            "code_font" => "code_font",
            _ => "fallback_font",
        };
        if !replaced.contains(&key) {
            replaced.push(key);
            list.clear();
        }
        list.push(value.to_string());
        return Ok(());
    }

    match field {
        "emoji_font" => config.fonts.emoji = value.to_string(),
        "font_size" => {
            let px: u32 = value.parse().map_err(|_| number(value))?;
            if !FONT_SIZE_RANGE.contains(&px) {
                return Err(format!(
                    "font_size {px} is outside {}..={}",
                    FONT_SIZE_RANGE.start(),
                    FONT_SIZE_RANGE.end()
                ));
            }
            config.font_size = Some(px);
        }

        "foreground" => config.theme.fg = color(value)?,
        "background" => config.theme.bg = color(value)?,
        "cursor" => config.theme.cursor = color(value)?,

        "tab_bar" => {
            config.tab_bar.position = match value {
                "top" => TabBarPosition::Top,
                "bottom" => TabBarPosition::Bottom,
                other => return Err(format!("tab_bar {other:?} is not top or bottom")),
            }
        }
        "tab_bar_height" => config.tab_bar.height_px = value.parse().map_err(|_| number(value))?,
        "tab_bar_label_scale" => {
            config.tab_bar.label_scale_pct = value.parse().map_err(|_| number(value))?;
        }
        "tab_active_fg" => config.tab_bar.active.fg = color(value)?,
        "tab_active_bg" => config.tab_bar.active.bg = color(value)?,
        "tab_inactive_fg" => config.tab_bar.inactive.fg = color(value)?,
        "tab_inactive_bg" => config.tab_bar.inactive.bg = color(value)?,
        "tab_divider" => config.tab_bar.divider = color(value)?,

        // `off` rather than 0: a threshold of zero milliseconds reads like "warn
        // immediately", which is the opposite of what it would do.
        "shell_notice_after" => {
            config.shell_startup.warn_after = match value {
                "off" => None,
                ms => Some(Duration::from_millis(ms.parse().map_err(|_| number(ms))?)),
            }
        }
        "shell_notice_hold" => {
            config.shell_startup.hold =
                Duration::from_millis(value.parse().map_err(|_| number(value))?);
        }

        _ => match field.strip_prefix("color") {
            Some(rest) => {
                let index: u8 = rest
                    .parse()
                    .map_err(|_| format!("color{rest} is not one of color0..color255"))?;
                config.theme.set_indexed(index, color(value)?);
            }
            // Unknown fields are nonfatal for forward compatibility but still reported.
            None => return Err(format!("unknown field {field:?}, skipped")),
        },
    }
    Ok(())
}

/// A color from any spec the terminal already understands.
fn color(value: &str) -> Result<Rgb, String> {
    parse_x11_color(value.as_bytes())
        .ok_or_else(|| format!("{value:?} is not a color like #1e1e2e or rgb:1e/1e/2e"))
}

fn number(value: &str) -> String {
    format!("{value:?} is not a number")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean(text: &str) -> Config {
        let (config, problems) = parse(text);
        assert!(problems.is_empty(), "unexpected problems: {problems:?}");
        config
    }

    #[test]
    fn an_empty_config_is_the_defaults() {
        let config = clean("");
        assert_eq!(config.fonts.families, Config::default().fonts.families);
        assert_eq!(config.font_size, None);
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let config = clean(
            "
            # the grid font
            font Iosevka   # trailing comments too

            ",
        );
        assert_eq!(config.fonts.families, vec!["Iosevka".to_string()]);
    }

    #[test]
    fn a_hash_is_a_comment_or_a_colour_by_position() {
        let config = clean("background #1e1e2e  # the window itself\nfont Iosevka # the grid");
        assert_eq!(config.theme.bg, Rgb::new(0x1e, 0x1e, 0x2e));
        assert_eq!(config.fonts.families, ["Iosevka"]);

        let (config, problems) = parse("font   # nothing here");
        assert_eq!(
            config.fonts.families,
            Config::default().fonts.families,
            "a value that is entirely a comment is no value"
        );
        assert_eq!(problems.len(), 1);
    }

    #[test]
    fn the_first_list_line_replaces_the_defaults_and_the_rest_append() {
        let config = clean("font Iosevka\nfont Hack\nfont monospace");
        assert_eq!(config.fonts.families, ["Iosevka", "Hack", "monospace"]);
        assert_eq!(
            config.fonts.ui,
            Config::default().fonts.ui,
            "a list nobody set keeps its defaults"
        );
    }

    #[test]
    fn a_value_may_hold_spaces() {
        let config = clean("font  Symbols Nerd Font Mono ");
        assert_eq!(config.fonts.families, ["Symbols Nerd Font Mono"]);
    }

    #[test]
    fn every_font_role_can_be_set() {
        let config =
            clean("font A\nui_font B\ncode_font C\nemoji_font D\nfallback_font E\nfont_size 18");
        assert_eq!(config.fonts.families, ["A"]);
        assert_eq!(config.fonts.ui, ["B"]);
        assert_eq!(config.fonts.code, ["C"]);
        assert_eq!(config.fonts.emoji, "D");
        assert_eq!(config.fonts.fallback, ["E"]);
        assert_eq!(config.font_size, Some(18));
    }

    #[test]
    fn colors_take_every_spec_the_terminal_takes() {
        let config = clean(
            "foreground #cdd6f4\nbackground #1e1e2e\ncursor rgb:f5/e0/dc\ncolor1 #f38ba8\ncolor255 #abc",
        );
        assert_eq!(config.theme.fg, Rgb::new(0xcd, 0xd6, 0xf4));
        assert_eq!(config.theme.bg, Rgb::new(0x1e, 0x1e, 0x2e));
        assert_eq!(config.theme.cursor, Rgb::new(0xf5, 0xe0, 0xdc));
        assert_eq!(config.theme.indexed(1), Rgb::new(0xf3, 0x8b, 0xa8));
        // X11 short components are left-justified, unlike CSS shorthand.
        assert_eq!(config.theme.indexed(255), Rgb::new(0xa0, 0xb0, 0xc0));
    }

    #[test]
    fn the_tab_bar_and_shell_notice_are_configurable() {
        let config = clean(
            "tab_bar top\ntab_bar_height 30\ntab_bar_label_scale 90\ntab_active_bg #313244\nshell_notice_after 2500\nshell_notice_hold 4000",
        );
        assert_eq!(config.tab_bar.position, TabBarPosition::Top);
        assert_eq!(config.tab_bar.height_px, 30);
        assert_eq!(config.tab_bar.label_scale_pct, 90);
        assert_eq!(config.tab_bar.active.bg, Rgb::new(0x31, 0x32, 0x44));
        assert_eq!(
            config.shell_startup.warn_after,
            Some(Duration::from_millis(2500))
        );
        assert_eq!(config.shell_startup.hold, Duration::from_millis(4000));
    }

    #[test]
    fn the_shell_notice_can_be_turned_off() {
        assert_eq!(
            clean("shell_notice_after off").shell_startup.warn_after,
            None
        );
    }

    #[test]
    fn a_bad_value_leaves_the_default_standing_and_is_reported() {
        let (config, problems) = parse("font_size enormous\nfont Iosevka");
        assert_eq!(config.font_size, None, "the default survives");
        assert_eq!(
            config.fonts.families,
            ["Iosevka"],
            "and the line after it still applies"
        );
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].line, 1);
        assert!(problems[0].message.contains("enormous"), "{problems:?}");
    }

    #[test]
    fn an_out_of_range_font_size_is_refused_rather_than_clamped() {
        let (config, problems) = parse("font_size 400");
        assert_eq!(config.font_size, None);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].message.contains("400"), "{problems:?}");
    }

    #[test]
    fn an_unknown_field_is_skipped_but_still_reported() {
        let (config, problems) = parse("fnt Iosevka\nfont Hack");
        assert_eq!(config.fonts.families, ["Hack"], "the file still loads");
        assert_eq!(problems.len(), 1);
        assert!(problems[0].message.contains("fnt"), "{problems:?}");
        assert!(problems[0].message.contains("skipped"), "{problems:?}");
    }

    #[test]
    fn a_field_with_no_value_is_reported() {
        let (_, problems) = parse("font\nbackground   ");
        assert_eq!(problems.len(), 2);
        assert_eq!(problems[0].line, 1);
        assert_eq!(problems[1].line, 2);
    }

    #[test]
    fn a_problem_prints_with_its_line() {
        let problem = Problem {
            line: 12,
            message: "unknown field \"fnt\", skipped".into(),
        };
        assert_eq!(
            problem.to_string(),
            "config:12: unknown field \"fnt\", skipped"
        );
        let whole_file = Problem {
            line: 0,
            message: "cannot read /x: nope".into(),
        };
        assert_eq!(whole_file.to_string(), "config: cannot read /x: nope");
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bnkterm-config-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a writable temp dir");
        dir
    }

    #[test]
    fn a_missing_file_is_the_defaults_and_no_complaint() {
        let dir = scratch_dir("missing");
        let (config, problems) = load_from(&dir.join("nothing-here"));
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(config.fonts.families, Config::default().fonts.families);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_real_file_is_read_and_applied() {
        let dir = scratch_dir("read");
        let path = dir.join("config");
        std::fs::write(
            &path,
            "# mine\nfont Iosevka\nfont_size 20\nbackground #101010\n",
        )
        .expect("write the config");
        let (config, problems) = load_from(&path);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(config.fonts.families, ["Iosevka"]);
        assert_eq!(config.font_size, Some(20));
        assert_eq!(config.theme.bg, Rgb::new(0x10, 0x10, 0x10));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_file_that_cannot_be_read_is_reported_and_survived() {
        // A directory in place of a file produces a portable read error.
        let dir = scratch_dir("unreadable");
        let (config, problems) = load_from(&dir);
        assert_eq!(config.fonts.families, Config::default().fonts.families);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert_eq!(problems[0].line, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_config_path_is_absolute_and_has_the_expected_suffix() {
        let path = path().expect("a home directory in the test environment");
        assert!(path.is_absolute(), "{path:?}");
        assert!(path.ends_with("bnkterm/config"), "{path:?}");
    }

    #[test]
    fn the_path_follows_xdg_then_home() {
        let os = |s: &str| Some(std::ffi::OsString::from(s));
        let config = |p: &str| Some(PathBuf::from(p));
        assert_eq!(
            path_from(os("/cfg"), os("/home/u")),
            config("/cfg/bnkterm/config")
        );
        assert_eq!(
            path_from(None, os("/home/u")),
            config("/home/u/.config/bnkterm/config")
        );
        // A relative directory would resolve against wherever the process started.
        assert_eq!(
            path_from(os("cfg"), os("/home/u")),
            config("/home/u/.config/bnkterm/config")
        );
        assert_eq!(path_from(os("cfg"), os("home")), None);
        assert_eq!(path_from(None, None), None);
    }
}
