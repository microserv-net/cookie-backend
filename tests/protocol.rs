//! Protocol-level tests, driven through the real router with no Ollama.
//!
//! These replace the Python conformance checker: same coverage, same
//! language as the rest of the system, and they run in `cargo test` rather
//! than needing a server to be started first.
//!
//! No Ollama, no model, no network. If these ever need one, something has
//! leaked out of the provider abstraction and that is the bug.

use std::sync::Arc;
use std::time::Instant;

use axum_test::TestServer;
use cookie_backend::api::{router, ApiState};
use cookie_backend::auth::DeviceStore;
use cookie_backend::config::Config;
use cookie_backend::ollama::OllamaProvider;
use cookie_backend::tasks::TaskManager;
use cookie_backend::tools::builtin::Memory;
use cookie_backend::tools::frontend::FrontendBridge;
use cookie_backend::tools::{ToolRegistry, ToolResult};
use serde_json::{json, Value};
use tokio::sync::Mutex;

struct Harness {
    server: TestServer,
    state: ApiState,
    _dir: tempfile::TempDir,
}

fn harness() -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::default();
    // Port 1 is reliably closed, so nothing here can reach a real Ollama
    // even if one happens to be running on this machine.
    config.ollama.endpoint = "http://127.0.0.1:1".into();
    config.data_dir = dir.path().to_path_buf();
    let config = Arc::new(config);

    let memory = Arc::new(Memory::open(config.memory_file()));
    let state = ApiState {
        provider: Arc::new(OllamaProvider::new(&config).unwrap()),
        devices: Arc::new(Mutex::new(DeviceStore::open(config.devices_file()))),
        tasks: Arc::new(TaskManager::new()),
        registry: Arc::new(ToolRegistry::standard(memory)),
        bridge: Arc::new(FrontendBridge::new()),
        started: Instant::now(),
        config,
    };
    Harness {
        server: TestServer::new(router(state.clone())).unwrap(),
        state,
        _dir: dir,
    }
}

fn turn_body(text: &str) -> Value {
    json!({
        "protocol": "cookie-interface/1",
        "session_id": "s",
        "utterance_id": "u",
        "text": text,
        "final": true,
        "interface": {"speech": true, "listening": true, "visual": "orb"},
        "scheduling": {"priority": "normal", "preempt": false},
        "active_tasks": [],
    })
}

/// Every NDJSON message from one turn.
fn messages(body: &str) -> Vec<Value> {
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("every line must be one JSON object"))
        .collect()
}

#[tokio::test]
async fn health_is_open_and_names_the_protocol() {
    let h = harness();
    let body: Value = h.server.get("/api/v1/health").await.json();
    assert_eq!(body["protocol"], "cookie-interface/1");
    // Ollama is absent here, and the honest answer is "degraded", not "ok".
    assert_eq!(body["status"], "degraded");
    assert_eq!(body["ollama"], "unreachable");
}

#[tokio::test]
async fn a_turn_streams_ndjson_and_terminates() {
    let h = harness();
    let response = h
        .server
        .post("/api/v1/chat")
        .json(&turn_body("hello"))
        .await;
    response.assert_status_ok();
    assert!(response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("ndjson"));

    let parsed = messages(&response.text());
    // Every line parses, and the stream is terminated — without an `end` the
    // frontend waits for the connection to close.
    assert_eq!(parsed.last().unwrap()["type"], "end");
    assert!(parsed.iter().any(|m| m["type"] == "task"));
}

#[tokio::test]
async fn a_turn_reports_its_task_lifecycle_with_documented_states() {
    let h = harness();
    let response = h
        .server
        .post("/api/v1/chat")
        .json(&turn_body("hello"))
        .await;
    let parsed = messages(&response.text());

    let tasks: Vec<&Value> = parsed.iter().filter(|m| m["type"] == "task").collect();
    assert!(!tasks.is_empty());
    for task in &tasks {
        assert!(task["id"].as_str().is_some_and(|id| !id.is_empty()));
        assert!(task["title"].as_str().is_some_and(|t| !t.is_empty()));
        let state = task["state"].as_str().unwrap();
        assert!(
            [
                "queued",
                "running",
                "suspended",
                "completed",
                "failed",
                "cancelled"
            ]
            .contains(&state),
            "undocumented state {state}"
        );
    }
}

#[tokio::test]
async fn a_long_request_is_heavy_and_a_short_one_is_not() {
    let h = harness();
    let short = messages(
        &h.server
            .post("/api/v1/chat")
            .json(&turn_body("what time is it"))
            .await
            .text(),
    );
    let long_text = "please go through the project and fix every failing test you find in it";
    let long = messages(
        &h.server
            .post("/api/v1/chat")
            .json(&turn_body(long_text))
            .await
            .text(),
    );

    let weight = |parsed: &[Value]| {
        parsed
            .iter()
            .find(|m| m["type"] == "task")
            .and_then(|m| m["weight"].as_str())
            .unwrap()
            .to_string()
    };
    assert_eq!(weight(&short), "light");
    assert_eq!(weight(&long), "heavy");
}

#[tokio::test]
async fn an_absent_ollama_fails_the_turn_rather_than_hanging() {
    let h = harness();
    let response = h
        .server
        .post("/api/v1/chat")
        .json(&turn_body("hello"))
        .await;
    let parsed = messages(&response.text());
    // The turn still terminates, and the task ends in a state the frontend
    // can show. Silence would be the worst outcome here.
    assert_eq!(parsed.last().unwrap()["type"], "end");
    let last_task = parsed.iter().rev().find(|m| m["type"] == "task").unwrap();
    assert_eq!(last_task["state"], "failed");

    // And the reason is *spoken*, because the user asked a question and
    // deserves to know why there is no answer.
    let spoken: String = parsed
        .iter()
        .filter(|m| m["type"] == "delta")
        .filter_map(|m| m["text"].as_str())
        .collect();
    assert!(spoken.to_lowercase().contains("ollama"), "said: {spoken:?}");
}

#[tokio::test]
async fn pairing_closes_the_door_behind_it() {
    let h = harness();
    // Nothing paired yet: open, so you can always get in to pair.
    h.server.get("/api/v1/tasks").await.assert_status_ok();

    let code = h.state.devices.lock().await.begin_pairing();
    let paired: Value = h
        .server
        .post("/api/v1/pair")
        .json(&json!({"code": code, "device_name": "laptop"}))
        .await
        .json();
    let token = paired["token"].as_str().unwrap().to_string();

    // Now it is shut.
    h.server
        .get("/api/v1/tasks")
        .await
        .assert_status(axum::http::StatusCode::UNAUTHORIZED);
    h.server
        .get("/api/v1/tasks")
        .add_header("authorization", format!("Bearer {token}"))
        .await
        .assert_status_ok();
    // Health stays open so a supervisor can probe it without a secret.
    h.server.get("/api/v1/health").await.assert_status_ok();
}

#[tokio::test]
async fn a_bad_pairing_code_is_refused() {
    let h = harness();
    h.server
        .post("/api/v1/pair")
        .json(&json!({"code": "not-a-code", "device_name": "x"}))
        .await
        .assert_status(axum::http::StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn cancelling_with_nothing_running_is_a_no_op_not_an_error() {
    let h = harness();
    let body: Value = h
        .server
        .post("/api/v1/cancel")
        .json(&json!({}))
        .await
        .json();
    assert_eq!(body["cancelled"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn the_tool_catalogue_says_where_each_tool_runs() {
    let h = harness();
    let body: Value = h.server.get("/api/v1/tools").await.json();
    let tools = body["tools"].as_array().unwrap();
    let find = |name: &str| {
        tools
            .iter()
            .find(|t| t["name"] == name)
            .unwrap_or_else(|| panic!("{name} is missing"))
            .clone()
    };
    assert_eq!(find("shell.run")["runs"], "frontend");
    assert_eq!(find("web.fetch")["runs"], "backend");
    assert_eq!(find("git.push")["risk"], "critical");
    assert!(body["capabilities"]
        .as_array()
        .unwrap()
        .contains(&json!("files")));
}

#[tokio::test]
async fn tool_results_are_correlated_by_id() {
    let h = harness();
    // Nobody is waiting for this one, which is not an error: the turn has
    // moved on, and the frontend should not be made to care.
    let body: Value = h
        .server
        .post("/api/v1/tool-result")
        .json(&json!({"id": "call-nobody-wants", "ok": true, "summary": "done"}))
        .await
        .json();
    assert_eq!(body["accepted"], false);

    h.server
        .post("/api/v1/tool-result")
        .json(&json!({"ok": true}))
        .await
        .assert_status(axum::http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn a_frontend_is_only_offered_what_it_implements() {
    let h = harness();
    let mut body = turn_body("hello");
    body["tools"] = json!(["filesystem.search", "shell.run"]);
    h.server
        .post("/api/v1/chat")
        .json(&body)
        .await
        .assert_status_ok();

    assert!(h
        .state
        .bridge
        .offers(h.state.registry.get("shell.run").unwrap()));
    assert!(!h
        .state
        .bridge
        .offers(h.state.registry.get("vscode.workspace").unwrap()));
    // Backend tools are unaffected by what the frontend can do.
    assert!(h
        .state
        .bridge
        .offers(h.state.registry.get("web.fetch").unwrap()));
}

#[tokio::test]
async fn models_endpoint_reports_configuration_and_residency() {
    let h = harness();
    let body: Value = h.server.get("/api/v1/models").await.json();
    assert_eq!(body["roles"]["architect"]["model"], "qwen3:8b");
    assert_eq!(body["roles"]["router"]["keep_alive"], "30m");
    // Nothing is resident because there is no Ollama; the budget still reads.
    assert_eq!(body["resident_gb"], 0.0);
    assert!(body["budget_gb"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn a_delivered_tool_result_reaches_whoever_is_waiting() {
    let h = harness();
    let tool = h.state.registry.get("filesystem.search").unwrap().clone();
    let bridge = h.state.bridge.clone();

    let waiting = {
        let bridge = bridge.clone();
        tokio::spawn(async move {
            bridge
                .call("call-1", |_| async {}, &tool, json!({"pattern": "*.rs"}))
                .await
        })
    };
    // Give the call a moment to register before answering it.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let body: Value = h
        .server
        .post("/api/v1/tool-result")
        .json(&json!({"id": "call-1", "ok": true, "summary": "4 matches",
                      "evidence": "found 4 files"}))
        .await
        .json();
    assert_eq!(body["accepted"], true);

    let result: ToolResult = waiting.await.unwrap();
    assert!(result.ok);
    assert_eq!(result.evidence, "found 4 files");
}
