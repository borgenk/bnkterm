//! Judging a `file:` link on the Flatpak host without stalling the event loop.
//!
//! A shell on the host prints links to host files, which the sandbox cannot see, so the
//! host resolves the path and reads its mode ([`verdict::judge_host_file`]). That is two
//! `flatpak-spawn` round trips, far too slow for the thread that renders and reads keys,
//! so each check runs on a thread of its own and the verdict comes back to the loop.
//!
//! ```text
//!   click ─▶ token ─▶ HostChecks::start ─▶ thread: judge_host_file
//!                                                 │ channel, eventfd wake
//!   loop: open the checked URL, or refuse it ◀────┘
//! ```
//!
//! The loop does the opening, so the link starts with its activation token from the same
//! place every other link does, and the thread only judges.

use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

use crate::app::State;
use crate::platform::ffi;
use crate::platform::verdict::{self, Refusal};

/// What a finished check leaves the loop to do.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Finished {
    /// Open the URL of the path the host checked, with the click's token.
    Open { url: String, token: Option<String> },
    /// Refuse the link as it was clicked.
    Refused { url: String, refusal: Refusal },
}

/// The checks in flight and the way their verdicts come back.
pub(super) struct HostChecks {
    sender: Sender<Finished>,
    results: Receiver<Finished>,
    /// Signalled by a check's thread once its verdict is sent. Shared with every thread,
    /// so it stays open until the last one is done with it.
    wake: Arc<OwnedFd>,
    /// Checks started whose verdict has not been taken. The wake fd is polled only while
    /// there are some.
    in_flight: usize,
    judge: fn(&str) -> Result<String, Refusal>,
}

impl HostChecks {
    pub(super) fn new() -> crate::error::Result<Self> {
        Self::judged_by(verdict::judge_host_file)
    }

    fn judged_by(judge: fn(&str) -> Result<String, Refusal>) -> crate::error::Result<Self> {
        let (sender, results) = mpsc::channel();
        Ok(Self {
            sender,
            results,
            wake: Arc::new(ffi::make_eventfd()?),
            in_flight: 0,
            judge,
        })
    }

    /// Judge `url` on a thread of its own. A thread that cannot be started is a check that
    /// cannot run, so its verdict is a refusal, delivered the same way.
    fn start(&mut self, url: String, token: Option<String>) {
        let (sender, wake, judge) = (self.sender.clone(), Arc::clone(&self.wake), self.judge);
        let clicked = url.clone();
        let spawned = std::thread::Builder::new()
            .name("bnkterm-host-check".to_string())
            .spawn(move || {
                let finished = match judge(&url) {
                    Ok(checked) => Finished::Open {
                        url: checked,
                        token,
                    },
                    Err(refusal) => Finished::Refused { url, refusal },
                };
                if sender.send(finished).is_ok() {
                    ffi::efd_signal(wake.as_raw_fd());
                }
            });
        if let Err(error) = spawned {
            eprintln!("bnkterm: cannot check a link on the host: {error}");
            let refused = Finished::Refused {
                url: clicked,
                refusal: Refusal::Unreachable,
            };
            if self.sender.send(refused).is_ok() {
                ffi::efd_signal(self.wake.as_raw_fd());
            }
        }
        self.in_flight += 1;
    }

    /// The fd to poll while a verdict is still owed, `None` when none is.
    pub(super) fn wake_fd(&self) -> Option<RawFd> {
        (self.in_flight > 0).then(|| self.wake.as_raw_fd())
    }

    /// Every check that has come back. The wake is cleared before the channel is read, so
    /// a check finishing after the read signals again rather than being missed.
    fn finished(&mut self) -> Vec<Finished> {
        ffi::efd_clear(self.wake.as_raw_fd());
        let done: Vec<Finished> = self.results.try_iter().collect();
        self.in_flight = self.in_flight.saturating_sub(done.len());
        done
    }
}

impl State {
    /// Judge a host shell's `file:` link on the host, and open it with `token` once that
    /// lets it. Without the machinery to check, it is refused: a check that cannot run
    /// must not open.
    pub(super) fn check_on_host(&mut self, url: String, token: Option<String>) {
        match &mut self.host_checks {
            Some(checks) => checks.start(url, token),
            None => self.tabs.host_link_refused(&url, Refusal::Unreachable),
        }
    }

    /// Act on every host check that has come back: open what was let through, as the URL
    /// of the path the host checked, and hand every refusal to the tabs.
    pub(super) fn finish_host_checks(&mut self) {
        let Some(checks) = &mut self.host_checks else {
            return;
        };
        for finished in checks.finished() {
            match finished {
                Finished::Open { url, token } => self.open_url(&url, token.as_deref()),
                Finished::Refused { url, refusal } => self.tabs.host_link_refused(&url, refusal),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::app::host_check::*;
    use crate::platform::file_url;
    use crate::pty::PollSet;

    /// Wait for `checks` to signal, then take what came back.
    fn wait_for(checks: &mut HostChecks) -> Vec<Finished> {
        let fd = checks.wake_fd().expect("a verdict is owed");
        let mut poll = PollSet::new();
        let slot = poll.add(fd);
        poll.wait(Some(Duration::from_secs(10))).unwrap();
        assert!(poll.readable(slot), "the check woke the loop");
        checks.finished()
    }

    #[test]
    fn a_check_reports_back_what_to_open_or_refuse_and_wakes_the_loop() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let dir = std::env::temp_dir().join(format!(
            "bnkterm-host-check-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let dir = std::fs::canonicalize(dir).unwrap();
        let script = dir.join("build.sh");
        std::fs::write(&script, b"x").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let notes = dir.join("notes.txt");
        std::fs::write(&notes, b"x").unwrap();
        let alias = dir.join("readme");
        symlink(&notes, &alias).unwrap();
        let url_of = |path: &std::path::Path| file_url::encode(path.to_str().unwrap());

        // The local judge stands in for the host's, which needs a sandbox.
        let mut checks = HostChecks::judged_by(verdict::judge_file).unwrap();
        assert!(checks.wake_fd().is_none(), "nothing polled while idle");
        checks.start(url_of(&script), Some("token".to_string()));
        assert_eq!(
            wait_for(&mut checks),
            [Finished::Refused {
                url: url_of(&script),
                refusal: Refusal::Executable
            }]
        );
        assert!(checks.wake_fd().is_none(), "nothing owed once taken");

        // What opens is the path the check resolved, not the link as clicked.
        checks.start(url_of(&alias), Some("token".to_string()));
        assert_eq!(
            wait_for(&mut checks),
            [Finished::Open {
                url: url_of(&notes),
                token: Some("token".to_string())
            }]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
