//! The command line: the handful of flags bnkterm takes, read in one place so that
//! anything else is refused rather than silently ignored. A launcher passing `-e htop`
//! or a user typing `--help` would otherwise get a plain shell and no word why.

use crate::app::Verbosity;

/// The one-line usage printed for an argument bnkterm does not take.
pub const USAGE: &str = "usage: bnkterm [--verbose | -v] [--stats] [--demo | --gpu-probe]";

/// What the command line asked bnkterm to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Open the window on the user's shell.
    Run,
    /// `--demo`: open the window on the static styled grid the screenshot asset is
    /// captured from, with no shell behind it.
    Demo,
    /// `--gpu-probe`: report what the compositor and GPU offer the render path, and exit.
    /// It wins over `--demo` when both are given.
    GpuProbe,
}

/// A parsed command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Invocation {
    pub mode: Mode,
    pub verbosity: Verbosity,
}

impl Invocation {
    /// Read `args`, the command line without the program name. `stats_env` is whether
    /// `BNKTERM_STATS` is set, the same knob as `--stats` for a launcher that cannot pass
    /// arguments; it is passed in so the parse stays pure. The error is the first
    /// argument that is not one of bnkterm's flags.
    pub fn parse(args: &[String], stats_env: bool) -> Result<Self, String> {
        let mut mode = Mode::Run;
        let (mut verbose, mut stats) = (false, stats_env);
        for arg in args {
            match arg.as_str() {
                "--verbose" | "-v" => verbose = true,
                "--stats" => stats = true,
                "--demo" if mode == Mode::Run => mode = Mode::Demo,
                "--demo" => {}
                "--gpu-probe" => mode = Mode::GpuProbe,
                _ => return Err(arg.clone()),
            }
        }
        // `--stats` implies verbose: the frame lines are useless without knowing which
        // renderer produced them, so the louder of the two wins.
        let verbosity = if stats {
            Verbosity::Stats
        } else if verbose {
            Verbosity::Verbose
        } else {
            Verbosity::Quiet
        };
        Ok(Self { mode, verbosity })
    }
}

#[cfg(test)]
mod tests {
    use crate::cli::*;

    fn parse(list: &[&str]) -> Result<Invocation, String> {
        let args: Vec<String> = list.iter().map(|a| a.to_string()).collect();
        Invocation::parse(&args, false)
    }

    fn mode(list: &[&str]) -> Mode {
        parse(list).expect("bnkterm's own flags").mode
    }

    fn verbosity(list: &[&str]) -> Verbosity {
        parse(list).expect("bnkterm's own flags").verbosity
    }

    #[test]
    fn a_bare_launch_runs_the_shell_quietly() {
        assert_eq!(
            parse(&[]),
            Ok(Invocation {
                mode: Mode::Run,
                verbosity: Verbosity::Quiet
            })
        );
    }

    #[test]
    fn each_flag_picks_its_mode() {
        assert_eq!(mode(&["--demo"]), Mode::Demo);
        assert_eq!(mode(&["--gpu-probe"]), Mode::GpuProbe);
        // The probe never opens a window, so it wins over the demo in either order.
        assert_eq!(mode(&["--demo", "--gpu-probe"]), Mode::GpuProbe);
        assert_eq!(mode(&["--gpu-probe", "--demo"]), Mode::GpuProbe);
        // A flag that says nothing about noise leaves the launch quiet.
        assert_eq!(verbosity(&["--demo"]), Verbosity::Quiet);
    }

    #[test]
    fn verbose_and_stats_flags_climb_the_ladder() {
        assert_eq!(verbosity(&["-v"]), Verbosity::Verbose);
        assert_eq!(verbosity(&["--verbose"]), Verbosity::Verbose);
        assert_eq!(verbosity(&["--stats"]), Verbosity::Stats);
        assert_eq!(
            verbosity(&["--stats", "--verbose"]),
            Verbosity::Stats,
            "the louder of the two wins"
        );
        // The environment variable is the same knob as `--stats`.
        let env = Invocation::parse(&["--verbose".to_string()], true);
        assert_eq!(env.map(|i| i.verbosity), Ok(Verbosity::Stats));
    }

    #[test]
    fn anything_else_is_refused_and_named() {
        // A shell must not open for a request bnkterm cannot carry out: `-e htop` would
        // otherwise drop the user at a prompt, with `htop` never run.
        assert_eq!(parse(&["--help"]), Err("--help".to_string()));
        assert_eq!(parse(&["-e", "htop"]), Err("-e".to_string()));
        assert_eq!(parse(&["--verbose", "htop"]), Err("htop".to_string()));
        assert_eq!(parse(&[""]), Err(String::new()), "an empty argument too");
    }
}
