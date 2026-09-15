//! Configuration.
//!
//! Model names never appear in source. Moving to a larger machine should mean
//! editing this file and nothing else — the orchestrator talks about *roles*
//! (router, worker, architect) and only this module knows what fills them.

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Written on first run, with the reasoning inline so the file explains
/// itself to whoever opens it next.
pub const DEFAULT_CONFIG: &str = r#"# cookie-backend

[server]
# 0.0.0.0 so your laptop and Tailscale can both reach it. Pairing is what
# keeps that safe: until a device pairs the API is open, and after that
# everything but /v1/health needs a token.
bind = "0.0.0.0"
port = 8080
base_path = "/api"

[ollama]
endpoint = "http://127.0.0.1:11434"
request_timeout_secs = 600

[models.router]
model = "qwen3:1.7b"
# Resident: it is on the path of every request and reloading it would be felt
# on all of them.
keep_alive = "30m"

[models.worker]
model = "qwen3:4b"
keep_alive = "10m"

[models.architect]
model = "qwen3:8b"
# Unloaded promptly: holding this alongside the worker is what pushes a 16 GB
# machine into swap.
keep_alive = "2m"

[limits]
# Roughly how much model weight may be resident at once, in gigabytes.
model_memory_gb = 9.0
max_replans = 2
max_tool_calls = 6
"#;

/// One named role and the model currently filling it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelRole {
    pub model: String,
    /// Passed straight to Ollama. On a small machine this is not a tuning
    /// detail — see `docs/scheduling.md`.
    pub keep_alive: String,
    /// Provider-specific generation options, passed through untouched.
    pub options: BTreeMap<String, toml::Value>,
}

impl Default for ModelRole {
    fn default() -> Self {
        Self {
            model: "qwen3:4b".into(),
            keep_alive: "10m".into(),
            options: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub bind: IpAddr,
    pub port: u16,
    /// Prefix every route sits under, so the backend can live behind a proxy
    /// at `/cookie` without the frontend caring.
    pub base_path: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            port: 8080,
            base_path: "/api".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OllamaConfig {
    pub endpoint: String,
    pub request_timeout_secs: u64,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:11434".into(),
            request_timeout_secs: 600,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    pub model_memory_gb: f32,
    /// How many times the architect may replan before we stop and explain.
    pub max_replans: u32,
    /// Tool calls allowed within one step, before the worker must report.
    pub max_tool_calls: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            model_memory_gb: 9.0,
            max_replans: 2,
            max_tool_calls: 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    pub ollama: OllamaConfig,
    pub limits: Limits,
    /// Role name to model. Unknown roles fall back to `worker`.
    pub models: BTreeMap<String, ModelRole>,
    /// Where tokens, memory and state live. Not written to the file by
    /// default; overridden in tests and by `COOKIE_BACKEND_HOME`.
    #[serde(skip)]
    pub data_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        let mut models = BTreeMap::new();
        models.insert(
            "router".into(),
            ModelRole {
                model: "qwen3:1.7b".into(),
                keep_alive: "30m".into(),
                options: BTreeMap::new(),
            },
        );
        models.insert("worker".into(), ModelRole::default());
        models.insert(
            "architect".into(),
            ModelRole {
                model: "qwen3:8b".into(),
                keep_alive: "2m".into(),
                options: BTreeMap::new(),
            },
        );
        Self {
            server: ServerConfig::default(),
            ollama: OllamaConfig::default(),
            limits: Limits::default(),
            models,
            data_dir: data_home(),
        }
    }
}

impl Config {
    /// The model filling `name`, falling back to the worker.
    ///
    /// Falling back rather than failing matters: a config written before a
    /// role existed should degrade, not refuse to start.
    pub fn role(&self, name: &str) -> ModelRole {
        self.models
            .get(name)
            .or_else(|| self.models.get("worker"))
            .cloned()
            .unwrap_or_default()
    }

    /// Absolute path for a route.
    pub fn route(&self, suffix: &str) -> String {
        format!(
            "{}/{}",
            self.server.base_path.trim_end_matches('/'),
            suffix.trim_start_matches('/')
        )
    }

    pub fn chat_url(&self) -> String {
        self.route("v1/chat")
    }

    pub fn validate(&self) -> Result<()> {
        if self.server.base_path.contains(' ') {
            return Err(Error::Config(
                "server.base_path must not contain spaces".into(),
            ));
        }
        if self.models.is_empty() {
            return Err(Error::Config("no models are configured".into()));
        }
        if self.limits.model_memory_gb <= 0.0 {
            return Err(Error::Config(
                "limits.model_memory_gb must be positive".into(),
            ));
        }
        Ok(())
    }

    /// Load, writing the default file on first run.
    ///
    /// Returns whether the file had to be created, so the CLI can say where
    /// it went exactly once rather than on every start.
    pub fn load_or_init(path: &Path) -> Result<(Self, bool)> {
        if !path.exists() {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
            }
            std::fs::write(path, DEFAULT_CONFIG).map_err(|e| Error::io(path, e))?;
            return Ok((Self::read(path)?, true));
        }
        Ok((Self::read(path)?, false))
    }

    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        let mut config: Config = toml::from_str(&text)?;
        config.data_dir = data_home();
        config.validate()?;
        Ok(config)
    }

    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| Error::Config(e.to_string()))
    }

    pub fn devices_file(&self) -> PathBuf {
        self.data_dir.join("devices.json")
    }

    pub fn memory_file(&self) -> PathBuf {
        self.data_dir.join("memory.json")
    }
}

/// Platform-independent config directory, honouring an override.
pub fn config_home() -> PathBuf {
    if let Some(root) = std::env::var_os("COOKIE_BACKEND_HOME") {
        return PathBuf::from(root).join("config");
    }
    directories::ProjectDirs::from("net", "microserv", "cookie-backend")
        .map(|dirs| dirs.config_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".cookie-backend/config"))
}

/// Where tokens, memory and state live.
pub fn data_home() -> PathBuf {
    if let Some(root) = std::env::var_os("COOKIE_BACKEND_HOME") {
        return PathBuf::from(root).join("data");
    }
    directories::ProjectDirs::from("net", "microserv", "cookie-backend")
        .map(|dirs| dirs.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".cookie-backend/data"))
}

pub fn config_file() -> PathBuf {
    config_home().join("config.toml")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_file_parses_into_the_default_config() {
        let parsed: Config = toml::from_str(DEFAULT_CONFIG).unwrap();
        assert_eq!(parsed.role("architect").model, "qwen3:8b");
        assert_eq!(parsed.role("router").keep_alive, "30m");
        assert!(parsed.validate().is_ok());
    }

    #[test]
    fn an_unknown_role_falls_back_to_the_worker() {
        let config = Config::default();
        assert_eq!(config.role("summariser").model, config.role("worker").model);
    }

    #[test]
    fn routes_join_cleanly_whatever_the_base_path() {
        let mut config = Config::default();
        assert_eq!(config.chat_url(), "/api/v1/chat");
        config.server.base_path = "/cookie/".into();
        assert_eq!(config.chat_url(), "/cookie/v1/chat");
    }

    #[test]
    fn first_run_writes_a_file_and_the_second_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let (_, created) = Config::load_or_init(&path).unwrap();
        assert!(created && path.exists());
        let (_, created_again) = Config::load_or_init(&path).unwrap();
        assert!(!created_again);
    }

    #[test]
    fn a_typo_is_refused_rather_than_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[server]\nprot = 8080\n").unwrap();
        assert!(Config::read(&path).is_err());
    }

    #[test]
    fn nonsense_limits_are_caught_by_validation() {
        let mut config = Config::default();
        config.limits.model_memory_gb = 0.0;
        assert!(config.validate().is_err());
    }
}
