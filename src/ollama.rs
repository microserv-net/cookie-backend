//! The model provider.
//!
//! Only this module knows the models are hosted by Ollama, and only the
//! configuration knows which models they are. Everything above talks about
//! *roles*, so replacing a model — or eventually the host — is a
//! configuration change rather than a rewrite.
//!
//! The non-obvious responsibility here is residency. The first machine holds
//! roughly one large model in RAM, so `keep_alive` is not a tuning detail: it
//! is the difference between the architect being available and the worker
//! being evicted to make room for it. Ollama's own lifecycle is used rather
//! than reimplemented, because it already knows what is loaded and we do not.

use std::time::Duration;

use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::config::{Config, ModelRole};
use crate::error::{Error, Result};

/// One message in a conversation sent to a model.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: &'static str,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system",
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user",
            content: content.into(),
        }
    }
}

/// A model resident in memory right now.
#[derive(Debug, Clone, Deserialize)]
pub struct LoadedModel {
    #[serde(alias = "model")]
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub expires_at: Option<String>,
}

impl LoadedModel {
    pub fn size_gb(&self) -> f32 {
        self.size as f32 / 1_000_000_000.0
    }
}

/// What generation looks like to a caller: fragments, in order, as they come.
pub struct Completion {
    rx: mpsc::Receiver<Result<String>>,
}

impl Completion {
    pub async fn next_fragment(&mut self) -> Option<Result<String>> {
        self.rx.recv().await
    }

    /// Collect the whole reply. Used for the structured calls — routing,
    /// planning, validation — where partial output has no meaning.
    pub async fn collect(mut self) -> Result<String> {
        let mut out = String::new();
        while let Some(fragment) = self.next_fragment().await {
            out.push_str(&fragment?);
        }
        Ok(out)
    }
}

/// Talks to Ollama.
#[derive(Debug, Clone)]
pub struct OllamaProvider {
    client: reqwest::Client,
    endpoint: String,
    memory_budget_gb: f32,
}

impl OllamaProvider {
    pub fn new(config: &Config) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(
                config.ollama.request_timeout_secs.max(5),
            ))
            .build()
            .map_err(|e| Error::Network(format!("could not build an HTTP client: {e}")))?;
        Ok(Self {
            client,
            endpoint: config.ollama.endpoint.trim_end_matches('/').to_string(),
            memory_budget_gb: config.limits.model_memory_gb,
        })
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    // --- introspection ---------------------------------------------------

    /// Is Ollama up? Used by health and by `doctor`.
    pub async fn available(&self) -> bool {
        matches!(
            tokio::time::timeout(
                Duration::from_secs(3),
                self.client.get(format!("{}/api/version", self.endpoint)).send(),
            )
            .await,
            Ok(Ok(response)) if response.status().is_success()
        )
    }

    pub async fn installed_models(&self) -> Result<Vec<String>> {
        let response = self
            .client
            .get(format!("{}/api/tags", self.endpoint))
            .send()
            .await
            .map_err(|_| Error::OllamaUnreachable {
                endpoint: self.endpoint.clone(),
            })?;
        let body: Value = response
            .json()
            .await
            .map_err(|e| Error::Model(e.to_string()))?;
        Ok(body
            .get("models")
            .and_then(Value::as_array)
            .map(|models| {
                models
                    .iter()
                    .filter_map(|m| m.get("name").and_then(Value::as_str))
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default())
    }

    /// What is resident right now, and how much memory it is using.
    pub async fn loaded_models(&self) -> Result<Vec<LoadedModel>> {
        let response = self
            .client
            .get(format!("{}/api/ps", self.endpoint))
            .send()
            .await
            .map_err(|_| Error::OllamaUnreachable {
                endpoint: self.endpoint.clone(),
            })?;
        let body: Value = response
            .json()
            .await
            .map_err(|e| Error::Model(e.to_string()))?;
        Ok(
            serde_json::from_value(body.get("models").cloned().unwrap_or_else(|| json!([])))
                .unwrap_or_default(),
        )
    }

    // --- residency -------------------------------------------------------

    /// Evict a model now.
    ///
    /// `keep_alive: 0` is Ollama's own idiom, so we are not fighting its
    /// lifecycle — we are using it at the one point where we know a model is
    /// no longer needed, which Ollama cannot know.
    pub async fn unload(&self, model: &str) {
        let _ = self
            .client
            .post(format!("{}/api/generate", self.endpoint))
            .json(&json!({"model": model, "keep_alive": 0, "prompt": ""}))
            .timeout(Duration::from_secs(30))
            .send()
            .await;
    }

    /// Evict whatever has to go before `role` can load.
    ///
    /// Returns what was evicted so the caller can say so in the activity log
    /// rather than leaving a ten-second pause unexplained.
    pub async fn make_room_for(&self, role: &ModelRole) -> Vec<String> {
        let Ok(loaded) = self.loaded_models().await else {
            return Vec::new();
        };
        let mut resident: f32 = loaded.iter().map(LoadedModel::size_gb).sum();
        if resident < self.memory_budget_gb * 0.75 {
            return Vec::new();
        }

        let mut candidates = loaded;
        // Evict the largest thing that is not the model we are about to use;
        // on this machine that is nearly always the architect finishing up.
        candidates.sort_by(|a, b| {
            b.size_gb()
                .partial_cmp(&a.size_gb())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut evicted = Vec::new();
        for model in candidates {
            if model.name == role.model {
                continue;
            }
            self.unload(&model.name).await;
            resident -= model.size_gb();
            evicted.push(model.name);
            if resident < self.memory_budget_gb * 0.6 {
                break;
            }
        }
        evicted
    }

    // --- generation ------------------------------------------------------

    /// Start generating. Fragments arrive on the returned [`Completion`].
    ///
    /// Dropping it stops the generation: the response body is dropped, the
    /// connection closes, and Ollama stops. That is how cancellation reaches
    /// the model.
    pub fn chat(&self, role: &ModelRole, messages: Vec<Message>) -> Completion {
        let (tx, rx) = mpsc::channel(32);
        let client = self.client.clone();
        let endpoint = self.endpoint.clone();
        let model = role.model.clone();
        let keep_alive = role.keep_alive.clone();
        let options: serde_json::Map<String, Value> = role
            .options
            .iter()
            .filter_map(|(k, v)| serde_json::to_value(v).ok().map(|v| (k.clone(), v)))
            .collect();

        tokio::spawn(async move {
            let payload = json!({
                "model": model,
                "messages": messages
                    .iter()
                    .map(|m| json!({"role": m.role, "content": m.content}))
                    .collect::<Vec<_>>(),
                "stream": true,
                "keep_alive": keep_alive,
                "options": Value::Object(options),
            });

            let response = match client
                .post(format!("{endpoint}/api/chat"))
                .json(&payload)
                .send()
                .await
            {
                Ok(response) => response,
                Err(e) if e.is_connect() => {
                    let _ = tx.send(Err(Error::OllamaUnreachable { endpoint })).await;
                    return;
                }
                Err(e) => {
                    let _ = tx
                        .send(Err(Error::Model(format!("the model call failed: {e}"))))
                        .await;
                    return;
                }
            };

            if response.status() == reqwest::StatusCode::NOT_FOUND {
                let _ = tx.send(Err(Error::ModelMissing { name: model })).await;
                return;
            }
            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                let _ = tx
                    .send(Err(Error::Model(format!(
                        "Ollama returned {status}: {}",
                        body.chars().take(200).collect::<String>()
                    ))))
                    .await;
                return;
            }

            // Ollama streams newline-delimited JSON; a chunk may split a line.
            let mut buffer = String::new();
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(e) => {
                        let _ = tx
                            .send(Err(Error::Network(format!("model stream: {e}"))))
                            .await;
                        return;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(newline) = buffer.find('\n') {
                    let line: String = buffer.drain(..=newline).collect();
                    let line = line.trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    let Ok(value) = serde_json::from_str::<Value>(&line) else {
                        continue;
                    };
                    if let Some(error) = value.get("error").and_then(Value::as_str) {
                        let _ = tx.send(Err(Error::Model(error.to_string()))).await;
                        return;
                    }
                    if let Some(fragment) = value
                        .pointer("/message/content")
                        .and_then(Value::as_str)
                        .filter(|f| !f.is_empty())
                    {
                        // A closed receiver means whoever asked has gone away.
                        if tx.send(Ok(fragment.to_string())).await.is_err() {
                            return;
                        }
                    }
                    if value.get("done").and_then(Value::as_bool).unwrap_or(false) {
                        return;
                    }
                }
            }
        });

        Completion { rx }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> OllamaProvider {
        let mut config = Config::default();
        // Port 1 is reliably closed, so nothing here can accidentally talk to
        // a real Ollama if one happens to be running.
        config.ollama.endpoint = "http://127.0.0.1:1".into();
        OllamaProvider::new(&config).unwrap()
    }

    #[tokio::test]
    async fn an_absent_ollama_is_reported_not_hidden() {
        let provider = provider();
        assert!(!provider.available().await);
        let error = provider.installed_models().await.unwrap_err();
        assert_eq!(error.code(), "ollama_unreachable");
        assert!(error.hint().unwrap().contains("ollama serve"));
    }

    #[tokio::test]
    async fn a_failed_generation_surfaces_as_an_error_fragment() {
        let provider = provider();
        let role = ModelRole::default();
        let completion = provider.chat(&role, vec![Message::user("hello")]);
        let error = completion.collect().await.unwrap_err();
        assert_eq!(error.code(), "ollama_unreachable");
    }

    #[tokio::test]
    async fn making_room_with_no_ollama_is_a_no_op_rather_than_a_failure() {
        // Residency management is an optimisation; it must never be the
        // reason a request fails.
        assert!(provider()
            .make_room_for(&ModelRole::default())
            .await
            .is_empty());
    }

    #[test]
    fn sizes_are_reported_in_gigabytes() {
        let model = LoadedModel {
            name: "qwen3:8b".into(),
            size: 5_600_000_000,
            expires_at: None,
        };
        assert!((model.size_gb() - 5.6).abs() < 0.01);
    }
}
