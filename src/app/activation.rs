//! Window activation through `xdg_activation_v1`: the window takes focus with the token
//! its launcher passed in, and a clicked link hands a fresh token to the program it
//! starts.
//!
//! ```text
//!   launcher ─ XDG_ACTIVATION_TOKEN ─▶ take_launch_token ─▶ activate(token, surface)
//!                                       removed from the       after the first commit
//!                                       environment
//!
//!   Ctrl+click ─▶ get_activation_token(serial, seat, surface) ─▶ done(token)
//!                                        ─▶ xdg-open with XDG_ACTIVATION_TOKEN=token,
//!                                           beside the shell (the Flatpak host's for a
//!                                           host shell)
//! ```
//!
//! A token is addressed to the window it was issued for. Inherited, a shell and everything
//! started from it would present this window's token as their own launch token instead of
//! asking for one, which is why the protocol asks a launched client to unset the variable
//! once it has read it. Without the global, the window opens without asking for focus and
//! a link opens without a token.

use crate::app::State;
use crate::error::Result;
use crate::platform::browser::Opener;
use crate::platform::protocol::{xdg_activation_token_v1, xdg_activation_v1};
use crate::platform::wire::{Arg, Message, Reader};
use crate::pty::Target;

/// Where a launcher passes its token: the Wayland variable first, then the X11 startup id
/// some launchers still set beside it.
const LAUNCH_TOKEN_VARS: [&str; 2] = ["XDG_ACTIVATION_TOKEN", "DESKTOP_STARTUP_ID"];

/// The window's `xdg_activation_v1` state.
pub(super) struct Activation {
    /// The global, absent when the compositor does not offer one.
    pub(super) global: Option<u32>,
    /// The launcher's token, until the first frame spends it.
    launch_token: Option<String>,
    /// Which `xdg-open` a clicked link goes to: the one beside the shell.
    opener: Opener,
    /// Token requests in flight for clicked links: the request object and the link that
    /// is followed when its token arrives.
    pending_links: Vec<(u32, ClickedLink)>,
}

/// A Ctrl+clicked link, waiting on its activation token.
pub(super) enum ClickedLink {
    /// Judged in full by the core: it opens as it is.
    Url(String),
    /// A `file:` link from a shell on the Flatpak host, judged there before it opens
    /// ([`crate::app::host_check`]).
    HostFile(String),
}

impl Activation {
    pub(super) fn new(launch_token: Option<String>, shell: Target) -> Self {
        Self {
            global: None,
            launch_token,
            opener: match shell {
                Target::Local => Opener::Local,
                Target::FlatpakHost => Opener::FlatpakHost,
            },
            pending_links: Vec::new(),
        }
    }
}

/// The token this process was launched with, removed from the environment with its X11
/// counterpart so no child inherits either. Sound only before any other thread exists.
pub(super) fn take_launch_token() -> Option<String> {
    let token = LAUNCH_TOKEN_VARS
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|token| !token.is_empty()));
    for name in LAUNCH_TOKEN_VARS {
        std::env::remove_var(name);
    }
    token
}

impl State {
    /// Ask the compositor to activate the window with the launcher's token, once. Called
    /// right after a buffer commit, so the window is mapped when the request arrives.
    pub(super) fn spend_launch_token(&mut self) {
        let Some(token) = self.activation.launch_token.take() else {
            return;
        };
        if let Some(global) = self.activation.global {
            self.conn.request(
                global,
                xdg_activation_v1::ACTIVATE,
                &[Arg::Str(&token), Arg::Object(self.surface)],
            );
        }
    }

    /// Follow a clicked link. With the global, a token is requested for the click first,
    /// citing its serial, and the link is followed when the token arrives.
    pub(super) fn launch_link(&mut self, link: ClickedLink) {
        let (Some(global), Some(seat)) = (self.activation.global, self.seat) else {
            self.follow(link, None);
            return;
        };
        let request = self.create(global, xdg_activation_v1::GET_ACTIVATION_TOKEN);
        self.conn.request(
            request,
            xdg_activation_token_v1::SET_SERIAL,
            &[Arg::Uint(self.last_serial), Arg::Object(seat)],
        );
        self.conn.request(
            request,
            xdg_activation_token_v1::SET_SURFACE,
            &[Arg::Object(self.surface)],
        );
        self.conn
            .request(request, xdg_activation_token_v1::COMMIT, &[]);
        self.activation.pending_links.push((request, link));
    }

    /// Open `link` with `token`, or start judging it on the host first.
    fn follow(&mut self, link: ClickedLink, token: Option<String>) {
        match link {
            ClickedLink::Url(url) => self.open_url(&url, token.as_deref()),
            ClickedLink::HostFile(url) => self.check_on_host(url, token),
        }
    }

    /// Launch a link in whatever the desktop has set as its handler, beside the shell. The
    /// link was already judged, so a refusal here means the desktop could not start a
    /// handler at all: report it and carry on, because a link that will not open is an
    /// annoyance and must never be allowed to take the terminal down with it.
    pub(super) fn open_url(&self, url: &str, activation_token: Option<&str>) {
        let opener = self.activation.opener;
        if let Err(err) = crate::platform::browser::open_url(url, activation_token, opener) {
            eprintln!("bnkterm: could not open {url}: {err}");
        }
    }

    /// A `done` for a link's token request: destroy the request and open its link with the
    /// token. `None` when `msg` is not addressed to one.
    pub(super) fn on_activation_message(
        &mut self,
        msg: &Message,
        r: &mut Reader,
    ) -> Option<Result<()>> {
        if msg.opcode != xdg_activation_token_v1::EV_DONE {
            return None;
        }
        let at = self
            .activation
            .pending_links
            .iter()
            .position(|(request, _)| *request == msg.object)?;
        let (request, link) = self.activation.pending_links.swap_remove(at);
        self.conn
            .request(request, xdg_activation_token_v1::DESTROY, &[]);
        // An undecodable token is dropped, not fatal: the link opens without one.
        let token = r.string().ok().filter(|token| !token.is_empty());
        self.follow(link, token.map(str::to_string));
        Some(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use crate::app::activation::{take_launch_token, Activation};
    use crate::platform::browser::Opener;
    use crate::pty::Target;

    #[test]
    fn a_link_opens_beside_the_shell() {
        // Opened inside the sandbox, a host shell's link loses its token to the portal.
        let host = Activation::new(None, Target::FlatpakHost);
        assert_eq!(host.opener, Opener::FlatpakHost);
        let local = Activation::new(None, Target::Local);
        assert_eq!(local.opener, Opener::Local);
    }

    /// Run in a process of its own by `the_launch_token_is_taken_and_never_inherited`:
    /// changing the test process's environment would race every other test.
    #[test]
    #[ignore = "helper: run in a child process by the_launch_token_is_taken_and_never_inherited"]
    fn take_the_launch_token_then_report_what_a_child_inherits() {
        let token = take_launch_token();
        let inherited = Command::new("env").output().unwrap();
        println!("token={token:?}");
        print!("{}", String::from_utf8_lossy(&inherited.stdout));
    }

    #[test]
    fn the_launch_token_is_taken_and_never_inherited() {
        let run = |vars: &[(&str, &str)]| {
            let out = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "app::activation::tests::take_the_launch_token_then_report_what_a_child_inherits",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env_remove("XDG_ACTIVATION_TOKEN")
                .env_remove("DESKTOP_STARTUP_ID")
                .envs(vars.iter().copied())
                .output()
                .unwrap();
            assert!(out.status.success(), "{out:?}");
            String::from_utf8(out.stdout).unwrap()
        };

        let both = run(&[
            ("XDG_ACTIVATION_TOKEN", "wayland-token"),
            ("DESKTOP_STARTUP_ID", "x11-id"),
        ]);
        assert!(both.contains("token=Some(\"wayland-token\")"), "{both}");
        assert!(!both.contains("XDG_ACTIVATION_TOKEN="), "{both}");
        assert!(!both.contains("DESKTOP_STARTUP_ID="), "{both}");

        // An empty Wayland token is no token, so the X11 one is used, and neither is passed on.
        let fallback = run(&[
            ("XDG_ACTIVATION_TOKEN", ""),
            ("DESKTOP_STARTUP_ID", "x11-id"),
        ]);
        assert!(fallback.contains("token=Some(\"x11-id\")"), "{fallback}");
        assert!(!fallback.contains("XDG_ACTIVATION_TOKEN="), "{fallback}");
        assert!(!fallback.contains("DESKTOP_STARTUP_ID="), "{fallback}");
    }
}
