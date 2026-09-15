//! Router, architect, worker, and the loop that keeps them honest.
//!
//! ```text
//! utterance ─▶ router (1.7b)   what kind of thing is this?
//!                  │
//!         ┌────────┴────────┐
//!         ▼                 ▼
//!    chat: worker      task: architect (8b) plans
//!    answers and            │
//!    that is all            ▼
//!                      worker (4b) executes a step, using tools
//!                           │
//!                           ▼
//!                      validator checks it against the step's own
//!                      success condition — not against how confident
//!                      the worker sounded
//!                           │
//!                     ┌─────┴─────┐
//!                  passed       failed
//!                     │             │
//!                 next step    architect replans with the evidence
//! ```
//!
//! Three properties this is built around:
//!
//! **The architect does not mark its own homework.** It must state in advance
//! how each step will be recognised as done; the worker executes and a
//! separate validator judges the report against that condition. An architect
//! that believes its own output is how you get an assistant which cheerfully
//! reports success while the tests are still red.
//!
//! **Replanning is bounded and remembers.** Failed approaches are
//! fingerprinted by their steps, so rewording a plan cannot disguise a
//! repeat; after a few attempts the assistant explains itself instead of
//! trying forever.
//!
//! **Every model call is a checkpoint.** On a machine holding one large
//! model, the gaps between calls are where something more urgent gets to go
//! first.

use std::sync::Arc;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::error::Error;
use crate::ollama::{Message, OllamaProvider};
use crate::parsing::{find_json_object, spoken_text, string_field};
use crate::prompts;
use crate::tasks::{Cancelled, Task, TaskManager, TaskState};
use crate::tools::frontend::FrontendBridge;
use crate::tools::{Runs, Tool, ToolCall, ToolRegistry, ToolResult};

/// Cap on steps per plan, whatever the architect thinks.
const MAX_STEPS: usize = 5;
/// How many tools the model is shown at once.
const TOOL_LIMIT: usize = 8;

/// Anything the orchestrator wants said or shown, as a protocol message.
pub type Emitter = Arc<dyn Fn(Value) -> futures_util::future::BoxFuture<'static, ()> + Send + Sync>;

/// One step of a plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub what: String,
    /// How the step will be recognised as done. Stated in advance, which is
    /// what makes the validator more than a second opinion.
    pub done_when: String,
}

/// What the architect proposes.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub steps: Vec<Step>,
    pub say: String,
}

impl Plan {
    /// Identity of an *approach*, for detecting repetition.
    ///
    /// Deliberately the steps and not `say`: rewording the preamble is not a
    /// new idea, and letting it count as one is how a loop hides.
    pub fn fingerprint(&self) -> String {
        let joined = self
            .steps
            .iter()
            .map(|step| format!("{}→{}", step.what, step.done_when))
            .collect::<Vec<_>>()
            .join("|")
            .to_lowercase();
        let digest = Sha256::digest(joined.as_bytes());
        digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
    }
}

/// What was tried and what came of it. Fed back to the architect.
#[derive(Debug, Clone, PartialEq)]
pub struct Attempt {
    pub step: Step,
    pub result: String,
    pub evidence: String,
    pub passed: bool,
}

/// How a turn ended.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Outcome {
    pub succeeded: bool,
    pub attempts: Vec<Attempt>,
    pub replans: u32,
    pub gave_up_because: Option<String>,
}

/// Runs one turn, start to finish.
pub struct Orchestrator {
    config: Arc<Config>,
    provider: Arc<OllamaProvider>,
    tasks: Arc<TaskManager>,
    registry: Arc<ToolRegistry>,
    bridge: Arc<FrontendBridge>,
}

impl std::fmt::Debug for Orchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Orchestrator").finish()
    }
}

impl Orchestrator {
    pub fn new(
        config: Arc<Config>,
        provider: Arc<OllamaProvider>,
        tasks: Arc<TaskManager>,
        registry: Arc<ToolRegistry>,
        bridge: Arc<FrontendBridge>,
    ) -> Self {
        Self {
            config,
            provider,
            tasks,
            registry,
            bridge,
        }
    }

    // --- model access ----------------------------------------------------

    /// One structured model call, with a checkpoint in front of it.
    async fn complete(
        &self,
        role_name: &str,
        system: &str,
        user: &str,
        task: &Task,
        emit: &Emitter,
    ) -> Result<String, Error> {
        self.tasks
            .checkpoint(task)
            .await
            .map_err(|Cancelled| Error::Other("cancelled".into()))?;

        let role = self.config.role(role_name);
        let evicted = self.provider.make_room_for(&role).await;
        if !evicted.is_empty() {
            // A ten-second pause with no explanation is indistinguishable
            // from a hang, so say what is happening.
            self.report(
                task,
                TaskState::Running,
                Some(&format!("unloading {} to make room", evicted.join(", "))),
                emit,
            )
            .await;
        }

        self.provider
            .chat(&role, vec![Message::system(system), Message::user(user)])
            .collect()
            .await
    }

    /// A model call whose output is spoken as it arrives.
    async fn speak(
        &self,
        role_name: &str,
        system: &str,
        user: &str,
        task: &Task,
        emit: &Emitter,
    ) -> Result<String, Error> {
        self.tasks
            .checkpoint(task)
            .await
            .map_err(|Cancelled| Error::Other("cancelled".into()))?;

        let role = self.config.role(role_name);
        let mut completion = self
            .provider
            .chat(&role, vec![Message::system(system), Message::user(user)]);

        let mut whole = String::new();
        let mut held_back = false;
        while let Some(fragment) = completion.next_fragment().await {
            if task.cancelled() {
                break;
            }
            let fragment = fragment?;
            whole.push_str(&fragment);
            // Reasoning must never be spoken, so once an opening `<think>` is
            // seen everything is held until the reply is complete.
            if whole.to_lowercase().contains("<think>") {
                held_back = true;
                continue;
            }
            if !held_back {
                emit(json!({"type": "delta", "text": fragment})).await;
            }
        }

        if held_back {
            let cleaned = spoken_text(&whole);
            if !cleaned.is_empty() {
                emit(json!({"type": "delta", "text": cleaned.clone()})).await;
            }
            return Ok(cleaned);
        }
        Ok(whole)
    }

    async fn report(&self, task: &Task, state: TaskState, detail: Option<&str>, emit: &Emitter) {
        if let Some(message) = self.tasks.set_state(task, state, detail) {
            emit(message).await;
        }
    }

    // --- the pipeline ----------------------------------------------------

    /// What kind of thing this is, and what it will need.
    ///
    /// Falls back to conversation, which is always safe: an assistant that
    /// answers when it should have acted is a disappointment; one that acts
    /// when it should have answered is a hazard.
    pub async fn route(
        &self,
        text: &str,
        task: &Task,
        emit: &Emitter,
    ) -> (bool, bool, Vec<String>) {
        let system = format!(
            "{}{}",
            prompts::ROUTER,
            self.registry.capabilities().join(", ")
        );
        let Ok(raw) = self.complete("router", &system, text, task, emit).await else {
            return (false, false, Vec::new());
        };
        let Some(parsed) = find_json_object(&raw) else {
            return (false, false, Vec::new());
        };
        let is_task = string_field(&parsed, "kind") == "task";
        let heavy = string_field(&parsed, "weight") == "heavy";
        let capabilities = parsed
            .get("capabilities")
            .and_then(Value::as_array)
            .map(|list| {
                list.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        (is_task, heavy, capabilities)
    }

    /// The tools worth showing the model, filtered by what is reachable.
    pub fn tools_for(&self, capabilities: &[String]) -> Vec<Tool> {
        self.registry
            .discover(capabilities, TOOL_LIMIT)
            .into_iter()
            .filter(|tool| self.bridge.offers(tool))
            .cloned()
            .collect()
    }

    /// Ask the architect for a plan, or a *different* plan.
    async fn plan(
        &self,
        objective: &str,
        task: &Task,
        history: &[Attempt],
        avoid: &[String],
        emit: &Emitter,
    ) -> Option<Plan> {
        let (system, user) = if history.is_empty() {
            (
                prompts::ARCHITECT.to_string(),
                format!("Objective: {objective}"),
            )
        } else {
            let evidence = history
                .iter()
                .map(|attempt| {
                    format!(
                        "- tried: {}\n  result: {}\n  evidence: {}\n  verdict: {}",
                        attempt.step.what,
                        attempt.result,
                        attempt.evidence,
                        if attempt.passed { "passed" } else { "FAILED" }
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            (
                prompts::ARCHITECT_REPLAN.to_string(),
                format!("Objective: {objective}\n\nWhat has been tried:\n{evidence}"),
            )
        };

        let raw = self
            .complete("architect", &system, &user, task, emit)
            .await
            .ok()?;
        let parsed = find_json_object(&raw)?;
        let steps: Vec<Step> = parsed
            .get("steps")
            .and_then(Value::as_array)?
            .iter()
            .filter_map(|step| {
                let what = string_field(step, "what");
                (!what.is_empty()).then(|| Step {
                    what,
                    done_when: string_field(step, "done_when"),
                })
            })
            .take(MAX_STEPS)
            .collect();
        if steps.is_empty() {
            return None;
        }
        let plan = Plan {
            steps,
            say: spoken_text(&string_field(&parsed, "say")),
        };
        if avoid.contains(&plan.fingerprint()) {
            // Already known not to work. Treating this as "no plan" sends us
            // to the give-up path, which explains itself, rather than round
            // the loop again.
            return None;
        }
        Some(plan)
    }

    /// Run one tool call, wherever it lives.
    async fn use_tool(&self, call: &ToolCall, task: &Task, emit: &Emitter) -> ToolResult {
        let Some(tool) = self.registry.get(&call.name) else {
            return ToolResult::failed(format!("there is no tool called {}", call.name));
        };
        if let Some(complaint) = tool.validate(&call.arguments) {
            return ToolResult::failed(complaint);
        }

        self.report(
            task,
            TaskState::Running,
            Some(&format!("using {}", tool.name)),
            emit,
        )
        .await;
        // A tool call is a checkpoint: we are about to wait on something
        // anyway, so it is free to let a more urgent turn go first.
        if self.tasks.checkpoint(task).await.is_err() {
            return ToolResult::failed("cancelled");
        }

        match tool.runs {
            Runs::Frontend => {
                let emit = Arc::clone(emit);
                self.bridge
                    .call(
                        &call.id,
                        move |message| {
                            let emit = Arc::clone(&emit);
                            async move { emit(message).await }
                        },
                        tool,
                        call.arguments.clone(),
                    )
                    .await
            }
            Runs::Backend => match &tool.handler {
                Some(handler) => handler(call.arguments.clone()).await,
                None => ToolResult::failed(format!(
                    "{} is declared but not implemented here",
                    tool.name
                )),
            },
        }
    }

    /// Worker does the step, using tools; validator decides whether it counts.
    async fn execute(
        &self,
        objective: &str,
        step: &Step,
        task: &Task,
        tools: &[Tool],
        emit: &Emitter,
    ) -> Attempt {
        let refs: Vec<&Tool> = tools.iter().collect();
        let system = if tools.is_empty() {
            prompts::WORKER.to_string()
        } else {
            format!(
                "{}\n\nTools available:\n{}",
                prompts::WORKER,
                ToolRegistry::describe(&refs)
            )
        };

        let mut transcript = format!(
            "Objective: {objective}\nStep: {}\nSucceeds when: {}",
            step.what, step.done_when
        );
        let mut tool_evidence: Vec<String> = Vec::new();
        let mut parsed = Value::Null;
        let mut raw = String::new();

        for call_number in 0..=self.config.limits.max_tool_calls {
            raw = match self
                .complete("worker", &system, &transcript, task, emit)
                .await
            {
                Ok(raw) => raw,
                Err(e) => {
                    return Attempt {
                        step: step.clone(),
                        result: e.to_string(),
                        evidence: e.to_string(),
                        passed: false,
                    }
                }
            };
            parsed = find_json_object(&raw).unwrap_or(Value::Null);

            let call = if tools.is_empty() {
                None
            } else {
                ToolCall::from_value(&parsed, format!("{}-call-{call_number}", task.id))
            };
            let Some(call) = call else { break };

            if call_number == self.config.limits.max_tool_calls {
                tool_evidence.push("stopped after too many tool calls".into());
                break;
            }

            let result = self.use_tool(&call, task, emit).await;
            tool_evidence.push(format!(
                "{}: {}",
                call.name,
                if result.evidence.is_empty() {
                    &result.summary
                } else {
                    &result.evidence
                }
            ));
            // The model sees exactly what happened, including failures, and
            // decides what to do about it.
            transcript.push_str(&format!(
                "\n\nYou used {} with {}.\nResult: {} — {}",
                call.name,
                call.arguments,
                if result.ok { "ok" } else { "FAILED" },
                result.summary
            ));
            if let Some(data) = &result.data {
                let rendered: String = data.to_string().chars().take(1500).collect();
                transcript.push_str(&format!("\nData: {rendered}"));
            }
        }

        let mut result = spoken_text(&string_field(&parsed, "result"));
        if result.is_empty() {
            result = spoken_text(&raw).chars().take(400).collect();
        }
        let mut evidence = string_field(&parsed, "evidence");
        if !tool_evidence.is_empty() {
            // Tool output is *external* evidence, which is the entire reason
            // the validator is worth more than a second opinion.
            let joined = tool_evidence.join("; ");
            evidence = if evidence.is_empty() {
                joined
            } else {
                format!("{joined}; {evidence}")
            };
        }
        let claimed = parsed.get("ok").and_then(Value::as_bool).unwrap_or(false);

        // The worker's own verdict is an input, not the answer.
        let verdict = self
            .complete(
                "worker",
                prompts::VALIDATOR,
                &format!(
                    "Step: {}\nSucceeds when: {}\nWorker reported: {result}\nEvidence: {}",
                    step.what,
                    step.done_when,
                    if evidence.is_empty() {
                        "(none given)"
                    } else {
                        &evidence
                    }
                ),
                task,
                emit,
            )
            .await
            .unwrap_or_default();
        let checked = find_json_object(&verdict).unwrap_or(Value::Null);
        let passed = checked
            .get("passed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let reason = string_field(&checked, "reason");

        Attempt {
            step: step.clone(),
            result,
            evidence: if evidence.is_empty() {
                reason
            } else {
                evidence
            },
            // No evidence and no independent pass means it did not happen,
            // however confident the worker was.
            passed: passed && (claimed || !tool_evidence.is_empty()),
        }
    }

    /// Handle one utterance. Speech and progress go out through `emit`.
    pub async fn run(&self, text: &str, task: &Task, emit: Emitter) -> Outcome {
        let (is_task, heavy, capabilities) = self.route(text, task, &emit).await;
        let tools = self.tools_for(&capabilities);
        self.report(
            task,
            TaskState::Running,
            Some(if is_task { "planning" } else { "answering" }),
            &emit,
        )
        .await;

        if !is_task {
            // A model failure here is spoken, not swallowed. Reporting a turn
            // as completed while saying nothing is the worst of both: the
            // user hears silence and the frontend shows success.
            return match self
                .speak("worker", prompts::SPOKEN, text, task, &emit)
                .await
            {
                Ok(_) => Outcome {
                    succeeded: true,
                    ..Default::default()
                },
                Err(e) => {
                    emit(json!({"type": "delta", "text": e.to_string()})).await;
                    Outcome {
                        succeeded: false,
                        gave_up_because: Some(e.to_string()),
                        ..Default::default()
                    }
                }
            };
        }
        let _ = heavy;

        let mut outcome = Outcome::default();
        let mut avoid: Vec<String> = Vec::new();

        for attempt_number in 0..=self.config.limits.max_replans {
            let Some(plan) = self
                .plan(text, task, &outcome.attempts, &avoid, &emit)
                .await
            else {
                outcome.gave_up_because = Some(if attempt_number > 0 {
                    "I could not come up with an approach I have not already tried.".into()
                } else {
                    "I could not work out how to do that.".into()
                });
                break;
            };
            avoid.push(plan.fingerprint());
            outcome.replans = attempt_number;

            if !plan.say.is_empty() {
                emit(json!({"type": "delta", "text": format!("{} ", plan.say)})).await;
            }

            let mut failed = None;
            for (index, step) in plan.steps.iter().enumerate() {
                self.report(
                    task,
                    TaskState::Running,
                    Some(&format!(
                        "step {} of {}: {}",
                        index + 1,
                        plan.steps.len(),
                        step.what
                    )),
                    &emit,
                )
                .await;
                let attempt = self.execute(text, step, task, &tools, &emit).await;
                let passed = attempt.passed;
                outcome.attempts.push(attempt.clone());
                if !passed {
                    failed = Some(attempt);
                    break;
                }
            }

            match failed {
                None => {
                    outcome.succeeded = true;
                    break;
                }
                Some(attempt) => {
                    self.report(
                        task,
                        TaskState::Running,
                        Some(&format!(
                            "that did not work: {}",
                            if attempt.evidence.is_empty() {
                                "no evidence"
                            } else {
                                &attempt.evidence
                            }
                        )),
                        &emit,
                    )
                    .await;
                }
            }
        }

        self.summarise(text, &outcome, task, &emit).await;
        outcome
    }

    /// Say what happened. The user watched none of it.
    async fn summarise(&self, objective: &str, outcome: &Outcome, task: &Task, emit: &Emitter) {
        if outcome.attempts.is_empty() {
            if let Some(reason) = &outcome.gave_up_because {
                emit(json!({"type": "delta", "text": reason})).await;
            }
            return;
        }
        let transcript = outcome
            .attempts
            .iter()
            .map(|attempt| {
                format!(
                    "- {}: {} ({})",
                    attempt.step.what,
                    attempt.result,
                    if attempt.passed {
                        "worked"
                    } else {
                        "did not work"
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut status = if outcome.succeeded {
            "It worked.".to_string()
        } else {
            "It did not work in the end.".to_string()
        };
        if let Some(reason) = &outcome.gave_up_because {
            status.push_str(&format!(" {reason}"));
        }

        if self
            .speak(
                "worker",
                prompts::SUMMARISE,
                &format!("Request: {objective}\n\nWhat happened:\n{transcript}\n\n{status}"),
                task,
                emit,
            )
            .await
            .is_err()
        {
            // Losing the model at the last step must not lose the news.
            emit(json!({"type": "delta", "text": status})).await;
        }
    }
}
