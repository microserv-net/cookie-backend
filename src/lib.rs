//! # cookie-backend
//!
//! Cookie's mind. The frontend (`cookie-interface`) owns the microphone, the
//! speaker and the orb; everything here is about deciding what to say and
//! what to do — and, because the first machine this runs on holds roughly one
//! large model in memory, about staying interruptible while doing it.
//!
//! ```text
//!   frontend ──POST /v1/chat──▶ router ─▶ architect ─▶ worker ─▶ validator
//!            ◀──NDJSON stream──    │          │          │
//!                                  └──────────┴──────────┴─▶ tools
//!                                                            (here, or back
//!                                                             on your laptop)
//! ```

pub mod api;
pub mod auth;
pub mod cli;
pub mod config;
pub mod error;
pub mod ollama;
pub mod orchestrator;
pub mod parsing;
pub mod prompts;
pub mod tasks;
pub mod tools;

pub use config::Config;
pub use error::{Error, Result};

/// Crate version, reported by `/v1/health` and `--version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The wire protocol this backend speaks. See `docs/protocol.md`.
pub const PROTOCOL: &str = "cookie-interface/1";
