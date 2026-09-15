//! Command line.
//!
//! Running `cookie-backend` with no arguments serves. Everything else exists
//! because there is a real moment when you need it: pairing a laptop,
//! checking why nothing answers, or seeing what is resident.

use clap::{Parser, Subcommand};

/// Cookie's backend: routing, planning, tools and memory.
#[derive(Debug, Parser)]
#[command(name = "cookie-backend", version, about, long_about = None)]
pub struct Cli {
    /// Configuration file to use.
    #[arg(long, value_name = "PATH", global = true)]
    pub config: Option<std::path::PathBuf>,

    /// Override the configured port.
    #[arg(long, value_name = "PORT", global = true)]
    pub port: Option<u16>,

    /// Log level: error, warn, info, debug, trace. `RUST_LOG` wins if set.
    #[arg(long, value_name = "LEVEL", default_value = "info", global = true)]
    pub log_level: String,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Issue a pairing code for a new frontend.
    Pair,
    /// List paired frontends.
    Devices,
    /// Remove a paired frontend.
    Revoke {
        /// The device's name, as shown by `devices`.
        name: String,
    },
    /// Check that everything is working, in plain English.
    Doctor,
    /// Write the default configuration and stop.
    Init,
    /// Print the configuration and where it lives.
    Show,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn bare_invocation_serves() {
        let cli = Cli::parse_from(["cookie-backend"]);
        assert!(cli.command.is_none());
    }

    #[test]
    fn subcommands_and_global_flags_parse_together() {
        let cli = Cli::parse_from(["cookie-backend", "--port", "9090", "doctor"]);
        assert_eq!(cli.port, Some(9090));
        assert!(matches!(cli.command, Some(Command::Doctor)));

        let cli = Cli::parse_from(["cookie-backend", "revoke", "laptop"]);
        match cli.command {
            Some(Command::Revoke { name }) => assert_eq!(name, "laptop"),
            other => panic!("{other:?}"),
        }
    }
}
