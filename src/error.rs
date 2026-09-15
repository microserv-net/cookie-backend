//! One error type, with a code for machines and a hint for people.

use std::path::{Path, PathBuf};

/// Everything that can go wrong, in the vocabulary of the person who has to
/// fix it rather than the layer that noticed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("configuration problem: {0}")]
    Config(String),

    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{0}")]
    Model(String),

    #[error("the model {name} is not installed")]
    ModelMissing { name: String },

    #[error("could not reach Ollama at {endpoint}")]
    OllamaUnreachable { endpoint: String },

    #[error("not paired")]
    NotPaired,

    #[error("{0}")]
    Pairing(String),

    #[error("{0}")]
    Network(String),

    #[error("port {port} is already in use")]
    PortInUse { port: u16 },

    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }

    /// Stable identifier for API responses.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Config(_) => "config_error",
            Error::Io { .. } => "io_error",
            Error::Model(_) => "model_error",
            Error::ModelMissing { .. } => "model_missing",
            Error::OllamaUnreachable { .. } => "ollama_unreachable",
            Error::NotPaired => "not_paired",
            Error::Pairing(_) => "pairing_failed",
            Error::Network(_) => "network_error",
            Error::PortInUse { .. } => "port_in_use",
            Error::Other(_) => "error",
        }
    }

    /// What to do about it, when there is something obvious to do.
    pub fn hint(&self) -> Option<String> {
        match self {
            Error::ModelMissing { name } => Some(format!("run: ollama pull {name}")),
            Error::OllamaUnreachable { .. } => Some("start it with: ollama serve".into()),
            Error::NotPaired => Some("run `cookie-backend pair` on the backend machine".into()),
            Error::PortInUse { .. } => Some("try --port with a different number".into()),
            _ => None,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<toml::de::Error> for Error {
    fn from(e: toml::de::Error) -> Self {
        Error::Config(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Other(format!("JSON: {e}"))
    }
}
