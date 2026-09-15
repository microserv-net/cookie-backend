//! The tests that need a real model.
//!
//! Everything else in this repository runs against a scripted provider, which
//! is what keeps `cargo test` honest on a machine with no GPU. But a fake
//! provider answers in whatever shape the test author imagined, and the
//! failures that actually happen are the ones where Qwen3 answers in a shape
//! nobody imagined: prose instead of JSON, a fenced block, a `<think>` that
//! runs to the end of the reply.
//!
//! So these run in CI, against a real `qwen3:1.7b`, and are `#[ignore]` so
//! they never block anybody's local `cargo test`:
//!
//! ```bash
//! ollama pull qwen3:1.7b
//! COOKIE_OLLAMA_TESTS=1 cargo test --test ollama -- --ignored --nocapture
//! ```
//!
//! Only the router model is exercised. The 8b would not finish on a CPU
//! runner, and the failures worth catching are about prompting and parsing,
//! which a small model exhibits more readily than a large one.

use std::sync::Arc;

use cookie_backend::config::{Config, ModelRole};
use cookie_backend::ollama::{Message, OllamaProvider};
use cookie_backend::orchestrator::Orchestrator;
use cookie_backend::parsing::{find_json_object, spoken_text, strip_reasoning};
use cookie_backend::prompts;
use cookie_backend::tasks::{TaskManager, Weight};
use cookie_backend::tools::builtin::Memory;
use cookie_backend::tools::frontend::FrontendBridge;
use cookie_backend::tools::ToolRegistry;
use serde_json::Value;

/// Skip unless CI has set the model up. Returns the model to use.
fn model() -> Option<String> {
    if std::env::var("COOKIE_OLLAMA_TESTS").is_err() {
        eprintln!("skipping: set COOKIE_OLLAMA_TESTS=1 with Ollama running");
        return None;
    }
    Some(std::env::var("COOKIE_TEST_MODEL").unwrap_or_else(|_| "qwen3:1.7b".into()))
}

fn setup() -> Option<(Arc<Config>, Arc<OllamaProvider>, String)> {
    let name = model()?;
    let mut config = Config::default();
    // Every role is the small model: this suite is about shapes, not quality.
    for role in ["router", "worker", "architect"] {
        config.models.insert(
            role.into(),
            ModelRole {
                model: name.clone(),
                keep_alive: "5m".into(),
                options: Default::default(),
            },
        );
    }
    let config = Arc::new(config);
    let provider = Arc::new(OllamaProvider::new(&config).unwrap());
    Some((config, provider, name))
}

fn orchestrator(
    config: Arc<Config>,
    provider: Arc<OllamaProvider>,
) -> (Orchestrator, Arc<TaskManager>) {
    let dir = tempfile::tempdir().unwrap();
    let memory = Arc::new(Memory::open(dir.path().join("memory.json")));
    std::mem::forget(dir);
    let tasks = Arc::new(TaskManager::new());
    let orchestrator = Orchestrator::new(
        config,
        provider,
        tasks.clone(),
        Arc::new(ToolRegistry::standard(memory)),
        Arc::new(FrontendBridge::new()),
    );
    (orchestrator, tasks)
}

/// Collect everything the orchestrator would say.
fn collector() -> (
    Arc<std::sync::Mutex<Vec<Value>>>,
    cookie_backend::orchestrator::Emitter,
) {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let emitter = {
        let seen = seen.clone();
        Arc::new(move |message: Value| {
            seen.lock().unwrap().push(message);
            Box::pin(async {}) as futures_util::future::BoxFuture<'static, ()>
        }) as cookie_backend::orchestrator::Emitter
    };
    (seen, emitter)
}

#[tokio::test]
#[ignore = "needs a real Ollama"]
async fn the_model_is_reachable_and_answers() {
    let Some((_, provider, name)) = setup() else {
        return;
    };
    assert!(provider.available().await, "Ollama is not answering");
    assert!(
        provider
            .installed_models()
            .await
            .unwrap()
            .iter()
            .any(|m| m == &name),
        "{name} is not installed"
    );

    let role = ModelRole {
        model: name,
        ..Default::default()
    };
    let reply = provider
        .chat(
            &role,
            vec![Message::user("Say the word ready and nothing else.")],
        )
        .collect()
        .await
        .unwrap();
    assert!(!reply.trim().is_empty(), "the model said nothing");
}

#[tokio::test]
#[ignore = "needs a real Ollama"]
async fn the_router_answers_in_json_we_can_actually_parse() {
    // The single most likely thing to break in the field: a small model
    // decides to explain itself instead of answering with an object.
    let Some((config, provider, _)) = setup() else {
        return;
    };
    let system = format!("{}{}", prompts::ROUTER, "files, web, memory, code, shell");

    for utterance in [
        "what time is it",
        "open the project I was working on and fix the failing tests",
        "hello",
    ] {
        let raw = provider
            .chat(
                &config.role("router"),
                vec![Message::system(&system), Message::user(utterance)],
            )
            .collect()
            .await
            .unwrap();
        let parsed = find_json_object(&raw)
            .unwrap_or_else(|| panic!("no JSON object in router reply for {utterance:?}:\n{raw}"));
        let kind = parsed["kind"].as_str().unwrap_or_default();
        assert!(
            kind == "chat" || kind == "task",
            "router said {kind:?} for {utterance:?}"
        );
    }
}

#[tokio::test]
#[ignore = "needs a real Ollama"]
async fn reasoning_blocks_are_stripped_from_real_output() {
    // Qwen3 emits <think> constantly. This asserts the stripping works against
    // what the model actually produces, not against a handcrafted example.
    let Some((config, provider, _)) = setup() else {
        return;
    };
    let raw = provider
        .chat(
            &config.role("worker"),
            vec![
                Message::system(prompts::SPOKEN),
                Message::user("Think carefully, then tell me what two plus two is."),
            ],
        )
        .collect()
        .await
        .unwrap();

    let spoken = spoken_text(&raw);
    assert!(
        !spoken.to_lowercase().contains("<think>"),
        "leaked: {spoken}"
    );
    assert!(
        !spoken.to_lowercase().contains("</think>"),
        "leaked: {spoken}"
    );
    assert!(
        !spoken.is_empty(),
        "stripping removed everything from:\n{raw}"
    );
    assert!(!strip_reasoning(&raw).is_empty());
}

#[tokio::test]
#[ignore = "needs a real Ollama"]
async fn the_architect_produces_steps_with_checkable_conditions() {
    let Some((config, provider, _)) = setup() else {
        return;
    };
    let raw = provider
        .chat(
            &config.role("architect"),
            vec![
                Message::system(prompts::ARCHITECT),
                Message::user("Objective: find the config file for this project and read it"),
            ],
        )
        .collect()
        .await
        .unwrap();

    let parsed = find_json_object(&raw).unwrap_or_else(|| panic!("no plan in:\n{raw}"));
    let steps = parsed["steps"]
        .as_array()
        .unwrap_or_else(|| panic!("no steps in:\n{raw}"));
    assert!(!steps.is_empty());
    // Every step must state how it will be recognised as done — the property
    // the whole validation loop rests on.
    for step in steps {
        assert!(
            step["what"].as_str().is_some_and(|w| !w.trim().is_empty()),
            "a step with no action: {step}"
        );
    }
}

#[tokio::test]
#[ignore = "needs a real Ollama"]
async fn a_conversational_turn_reaches_the_speaker() {
    let Some((config, provider, _)) = setup() else {
        return;
    };
    let (orchestrator, tasks) = orchestrator(config, provider);
    let task = tasks.create("answering", Weight::Light);
    let (seen, emit) = collector();

    let outcome = orchestrator
        .run("what is the capital of France?", &task, emit)
        .await;
    assert!(outcome.succeeded, "{outcome:?}");

    let spoken: String = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|m| m["type"] == "delta")
        .filter_map(|m| m["text"].as_str().map(str::to_owned))
        .collect();
    assert!(!spoken.trim().is_empty(), "nothing was said");
    assert!(
        !spoken.contains("<think>"),
        "reasoning was spoken: {spoken}"
    );
    // The spoken answer must be speech, not a document.
    assert!(!spoken.contains("```"), "code fence in speech: {spoken}");
    assert!(spoken.to_lowercase().contains("paris"), "said: {spoken}");
}

#[tokio::test]
#[ignore = "needs a real Ollama"]
async fn the_validator_can_say_no_to_a_real_model() {
    // A worker claiming success with contradicting evidence must not pass.
    let Some((config, provider, _)) = setup() else {
        return;
    };
    let raw = provider
        .chat(
            &config.role("worker"),
            vec![
                Message::system(prompts::VALIDATOR),
                Message::user(
                    "Step: run the test suite\nSucceeds when: every test passes\n\
                     Worker reported: I fixed it and everything works\n\
                     Evidence: exit code 1; 3 tests failed",
                ),
            ],
        )
        .collect()
        .await
        .unwrap();
    let parsed = find_json_object(&raw).unwrap_or_else(|| panic!("no verdict in:\n{raw}"));
    assert_eq!(
        parsed["passed"], false,
        "the validator believed a claim its evidence contradicts:\n{raw}"
    );
}
