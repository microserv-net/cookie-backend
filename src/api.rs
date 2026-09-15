//! The HTTP surface the frontend talks to.
//!
//! Implements `docs/protocol.md`. Everything streams: a turn is answered with
//! newline-delimited JSON as the model produces it, so Cookie starts speaking
//! the first sentence while the rest is still being generated.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};

use crate::auth::DeviceStore;
use crate::config::Config;
use crate::error::Error;
use crate::ollama::OllamaProvider;
use crate::orchestrator::{Emitter, Orchestrator};
use crate::tasks::{TaskManager, TaskState, Weight};
use crate::tools::builtin::Memory;
use crate::tools::frontend::FrontendBridge;
use crate::tools::{ToolRegistry, ToolResult};
use crate::{PROTOCOL, VERSION};

/// A turn longer than this is treated as work rather than conversation.
const HEAVY_WORD_COUNT: usize = 12;

/// Everything the handlers share.
#[derive(Clone)]
pub struct ApiState {
    pub config: Arc<Config>,
    pub provider: Arc<OllamaProvider>,
    pub devices: Arc<Mutex<DeviceStore>>,
    pub tasks: Arc<TaskManager>,
    pub registry: Arc<ToolRegistry>,
    pub bridge: Arc<FrontendBridge>,
    pub started: Instant,
}

impl std::fmt::Debug for ApiState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiState").finish()
    }
}

impl ApiState {
    pub fn new(config: Arc<Config>) -> crate::Result<Self> {
        let memory = Arc::new(Memory::open(config.memory_file()));
        Ok(Self {
            provider: Arc::new(OllamaProvider::new(&config)?),
            devices: Arc::new(Mutex::new(DeviceStore::open(config.devices_file()))),
            tasks: Arc::new(TaskManager::new()),
            registry: Arc::new(ToolRegistry::standard(memory)),
            bridge: Arc::new(FrontendBridge::new()),
            started: Instant::now(),
            config,
        })
    }
}

/// Build the router. Exposed so tests can drive it without a socket.
pub fn router(state: ApiState) -> Router {
    let base = state
        .config
        .server
        .base_path
        .trim_end_matches('/')
        .to_string();
    Router::new()
        .route(&format!("{base}/v1/health"), get(health))
        .route(&format!("{base}/v1/models"), get(models))
        .route(&format!("{base}/v1/tools"), get(tools))
        .route(&format!("{base}/v1/tasks"), get(tasks))
        .route(&format!("{base}/v1/pair"), post(pair))
        .route(&format!("{base}/v1/chat"), post(chat))
        .route(&format!("{base}/v1/cancel"), post(cancel))
        .route(&format!("{base}/v1/tool-result"), post(tool_result))
        .with_state(state)
}

/// Bind and serve until `shutdown` resolves.
pub async fn serve(
    state: ApiState,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> crate::Result<()> {
    let addr = SocketAddr::new(state.config.server.bind, state.config.server.port);
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::AddrInUse {
            Error::PortInUse { port: addr.port() }
        } else {
            Error::io(addr.to_string(), e)
        }
    })?;
    tracing::info!(%addr, "listening");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| Error::Other(format!("the server stopped: {e}")))
}

// ---------------------------------------------------------------------------
// Errors and authentication
// ---------------------------------------------------------------------------

struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    hint: Option<String>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut body = json!({"error": self.message, "code": self.code});
        if let Some(hint) = self.hint {
            body["hint"] = json!(hint);
        }
        (self.status, Json(body)).into_response()
    }
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "bad_request",
            message: message.into(),
            hint: None,
        }
    }
}

impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        let status = match &error {
            Error::NotPaired => StatusCode::UNAUTHORIZED,
            Error::Pairing(_) => StatusCode::FORBIDDEN,
            Error::ModelMissing { .. } | Error::OllamaUnreachable { .. } => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            code: error.code(),
            message: error.to_string(),
            hint: error.hint(),
        }
    }
}

/// Every endpoint but health and pairing needs a paired device.
///
/// When nothing has been paired the backend is open, because a locked-out
/// machine with no way in is worse than one on your own network that has not
/// been paired yet. The moment the first device pairs, this closes.
async fn require_device(state: &ApiState, headers: &HeaderMap) -> Result<(), ApiError> {
    let mut devices = state.devices.lock().await;
    if devices.is_empty() {
        return Ok(());
    }
    let token = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim);
    if devices.authenticate(token).is_some() {
        return Ok(());
    }
    Err(Error::NotPaired.into())
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async fn health(State(state): State<ApiState>) -> Json<Value> {
    let ollama_up = state.provider.available().await;
    let paired = state.devices.lock().await.devices().len();
    Json(json!({
        "status": if ollama_up { "ok" } else { "degraded" },
        "backend": "cookie-backend",
        "version": VERSION,
        "protocol": PROTOCOL,
        "ollama": if ollama_up { "up" } else { "unreachable" },
        "paired_devices": paired,
        "active_tasks": state.tasks.active().len(),
        "uptime_seconds": state.started.elapsed().as_secs(),
    }))
}

/// What is configured, installed and resident.
///
/// Residency is the thing worth watching on a small machine, so it is
/// reported rather than left to `ollama ps` in another terminal.
async fn models(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    require_device(&state, &headers).await?;
    let installed = state.provider.installed_models().await.unwrap_or_default();
    let loaded = state.provider.loaded_models().await.unwrap_or_default();
    let resident: f32 = loaded.iter().map(|m| m.size_gb()).sum();

    let roles: serde_json::Map<String, Value> = state
        .config
        .models
        .iter()
        .map(|(name, role)| {
            (
                name.clone(),
                json!({
                    "model": role.model,
                    "keep_alive": role.keep_alive,
                    "installed": installed.contains(&role.model),
                }),
            )
        })
        .collect();

    Ok(Json(json!({
        "roles": Value::Object(roles),
        "loaded": loaded.iter().map(|m| json!({
            "model": m.name,
            "size_gb": (m.size_gb() * 100.0).round() / 100.0,
            "expires_at": m.expires_at,
        })).collect::<Vec<_>>(),
        "resident_gb": (resident * 100.0).round() / 100.0,
        "budget_gb": state.config.limits.model_memory_gb,
    })))
}

/// What this backend can do, and where each tool runs.
async fn tools(State(state): State<ApiState>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    require_device(&state, &headers).await?;
    Ok(Json(json!({
        "capabilities": state.registry.capabilities(),
        "tools": state.registry.all().iter().map(|tool| json!({
            "name": tool.name,
            "version": tool.version,
            "summary": tool.summary,
            "capabilities": tool.capabilities,
            "runs": tool.runs,
            "risk": tool.risk.as_str(),
            "parameters": tool.parameters.iter()
                .map(|(k, v)| (k.to_string(), json!(v)))
                .collect::<serde_json::Map<_, _>>(),
            "required": tool.required,
        })).collect::<Vec<_>>(),
    })))
}

async fn tasks(State(state): State<ApiState>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    require_device(&state, &headers).await?;
    Ok(Json(json!({
        "active": state.tasks.active(),
        "summary": state.tasks.spoken_summary(),
    })))
}

// ---------------------------------------------------------------------------
// Pairing
// ---------------------------------------------------------------------------

async fn pair(
    State(state): State<ApiState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let code = body.get("code").and_then(Value::as_str).unwrap_or_default();
    let name = body
        .get("device_name")
        .and_then(Value::as_str)
        .unwrap_or("frontend");
    let token = state.devices.lock().await.complete_pairing(code, name)?;
    Ok(Json(json!({"token": token, "protocol": PROTOCOL})))
}

// ---------------------------------------------------------------------------
// Work
// ---------------------------------------------------------------------------

async fn cancel(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> Result<Json<Value>, ApiError> {
    require_device(&state, &headers).await?;
    let task_id = body
        .as_ref()
        .and_then(|Json(body)| body.get("task_id"))
        .and_then(Value::as_str);
    let cancelled: Vec<String> = state
        .tasks
        .cancel(task_id)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    Ok(Json(json!({"cancelled": cancelled})))
}

/// The frontend answering a `tool.request` it received on the stream.
///
/// Correlated by `id`. An unknown id is not worth failing on: it means the
/// turn moved on, usually because the call timed out, and the frontend should
/// not be made to care.
async fn tool_result(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    require_device(&state, &headers).await?;
    let id = body.get("id").and_then(Value::as_str).unwrap_or_default();
    if id.is_empty() {
        return Err(ApiError::bad_request("a tool result needs an id"));
    }
    let result = ToolResult {
        ok: body.get("ok").and_then(Value::as_bool).unwrap_or(false),
        summary: body
            .get("summary")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        evidence: body
            .get("evidence")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        data: body.get("data").cloned(),
        error: body.get("error").and_then(Value::as_str).map(str::to_owned),
    };
    Ok(Json(json!({"accepted": state.bridge.deliver(id, result)})))
}

/// One conversational turn, streamed as newline-delimited JSON.
async fn chat(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    require_device(&state, &headers).await?;

    let text = body
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let preempt_requested = body
        .pointer("/scheduling/preempt")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // The frontend tells us what it can do when it opens a turn; anything it
    // does not implement is never offered to the model.
    state.bridge.advertise(body.get("tools"));

    let heavy = text.split_whitespace().count() > HEAVY_WORD_COUNT;
    let task = state.tasks.create(
        if heavy {
            "working on that"
        } else {
            "thinking about that"
        },
        if heavy { Weight::Heavy } else { Weight::Light },
    );

    let (tx, rx) = mpsc::channel::<Value>(32);
    let emit: Emitter = {
        let tx = tx.clone();
        Arc::new(move |message: Value| {
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send(message).await;
            })
        })
    };

    let orchestrator = Orchestrator::new(
        state.config.clone(),
        state.provider.clone(),
        state.tasks.clone(),
        state.registry.clone(),
        state.bridge.clone(),
    );
    let tasks = state.tasks.clone();
    let task_for_run = task.clone();

    tokio::spawn(async move {
        // Ask heavier work to stand aside *before* touching a model: the gap
        // between turns is itself a checkpoint, and the whole point is not to
        // have two large models resident at once.
        let mut stood_aside = Vec::new();
        if tasks.should_preempt(preempt_requested) {
            for (id, message) in tasks.stand_aside("paused while I deal with this") {
                stood_aside.push(id);
                let _ = tx.send(message).await;
            }
        }

        let spoken = if text.is_empty() {
            "(the user said nothing audible)".to_string()
        } else {
            text
        };
        let outcome = orchestrator.run(&spoken, &task_for_run, emit).await;

        if !task_for_run.cancelled() {
            let state = if outcome.succeeded {
                TaskState::Completed
            } else {
                TaskState::Failed
            };
            if let Some(message) = tasks.set_state(&task_for_run, state, None) {
                let _ = tx.send(message).await;
            }
        }
        for message in tasks.resume(&stood_aside) {
            let _ = tx.send(message).await;
        }
        let _ = tx.send(json!({"type": "end"})).await;
        tasks.prune(std::time::Duration::from_secs(300));
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
        .map(|message| Ok::<_, std::io::Error>(format!("{message}\n").into_bytes()));

    Ok(Response::builder()
        .header("content-type", "application/x-ndjson")
        .header("cache-control", "no-store")
        .body(axum::body::Body::from_stream(stream))
        .expect("valid response"))
}

use tokio_stream::StreamExt as _;
