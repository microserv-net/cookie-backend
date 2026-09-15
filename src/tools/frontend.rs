//! Tools that run on the user's machine.
//!
//! The backend never touches the user's filesystem, shell, applications or
//! screen. It asks. A model that has been talked into something can reach at
//! most a *request* for `filesystem.delete`, which the frontend can refuse,
//! confirm or classify — because the frontend is the machine being acted upon
//! and the only one that knows whether the user is standing there.
//!
//! The mechanism is small. A call goes out on the turn's existing reply
//! stream as `{"type":"tool.request", "id":…}`, and the frontend posts the
//! answer back to `/v1/tool-result`. Correlation is by id, and every call has
//! a deadline: a frontend that has gone away must not leave a turn hanging,
//! so a timeout is reported to the model as a failed call — something it can
//! plan around — rather than as an error nobody sees.
//!
//! The catalogue lives here rather than being discovered from the frontend,
//! because the model needs the schemas before the first call. What the
//! frontend actually implements it advertises when it opens a turn; anything
//! it does not implement is never offered.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::oneshot;

use super::{Risk, Runs, Tool, ToolResult};

/// How long to wait for the frontend to answer.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

macro_rules! frontend_tool {
    ($name:literal, $version:literal, $summary:literal, [$($cap:literal),*],
     [$(($arg:literal, $kind:literal)),*], [$($req:literal),*], $risk:expr) => {
        Tool {
            name: $name,
            version: $version,
            summary: $summary,
            capabilities: &[$($cap),*],
            parameters: &[$(($arg, $kind)),*],
            required: &[$($req),*],
            runs: Runs::Frontend,
            risk: $risk,
            handler: None,
        }
    };
}

/// The catalogue. Risk levels are advisory here and authoritative on the
/// frontend, which owns the confirmation policy.
pub fn tools() -> Vec<Tool> {
    vec![
        frontend_tool!(
            "filesystem.search",
            1,
            "find files by name or pattern",
            ["files", "projects"],
            [("pattern", "string"), ("root", "string")],
            ["pattern"],
            Risk::Safe
        ),
        frontend_tool!(
            "filesystem.read",
            1,
            "read a file, or a range of its lines",
            ["files", "code"],
            [
                ("path", "string"),
                ("from_line", "integer"),
                ("to_line", "integer")
            ],
            ["path"],
            Risk::Safe
        ),
        frontend_tool!(
            "filesystem.write",
            1,
            "write or replace a file's contents",
            ["files", "code"],
            [("path", "string"), ("content", "string")],
            ["path", "content"],
            Risk::Normal
        ),
        frontend_tool!(
            "filesystem.delete",
            1,
            "delete a file or directory",
            ["files"],
            [("path", "string")],
            ["path"],
            Risk::Dangerous
        ),
        frontend_tool!(
            "shell.which",
            1,
            "check whether a command exists and what it is",
            ["shell", "system"],
            [("command", "string")],
            ["command"],
            Risk::Safe
        ),
        frontend_tool!(
            "shell.run",
            1,
            "run a command and return its output and exit code",
            ["shell", "code", "system"],
            [
                ("command", "string"),
                ("arguments", "list of strings"),
                ("cwd", "string"),
                ("timeout_seconds", "integer")
            ],
            ["command"],
            Risk::Normal
        ),
        frontend_tool!(
            "app.open",
            1,
            "open an application",
            ["applications"],
            [("name", "string")],
            ["name"],
            Risk::Normal
        ),
        frontend_tool!(
            "app.close",
            1,
            "close an application or window",
            ["applications"],
            [("name", "string")],
            ["name"],
            Risk::Normal
        ),
        frontend_tool!(
            "browser.open",
            1,
            "open a URL in the default browser",
            ["web", "applications"],
            [("url", "string")],
            ["url"],
            Risk::Safe
        ),
        frontend_tool!(
            "vscode.workspace",
            1,
            "the open workspace, file, selection and diagnostics",
            ["code", "projects"],
            [],
            [],
            Risk::Safe
        ),
        frontend_tool!(
            "vscode.open",
            1,
            "open a file in the editor, optionally at a line",
            ["code"],
            [("path", "string"), ("line", "integer")],
            ["path"],
            Risk::Safe
        ),
        frontend_tool!(
            "git.status",
            1,
            "branch, staged and unstaged changes",
            ["git", "code"],
            [("repository", "string")],
            [],
            Risk::Safe
        ),
        frontend_tool!(
            "git.diff",
            1,
            "the current diff, optionally for one path",
            ["git", "code"],
            [("repository", "string"), ("path", "string")],
            [],
            Risk::Safe
        ),
        frontend_tool!(
            "git.commit",
            1,
            "stage and commit with a message",
            ["git", "code"],
            [("repository", "string"), ("message", "string")],
            ["message"],
            Risk::Normal
        ),
        frontend_tool!(
            "git.push",
            1,
            "push the current branch",
            ["git", "code"],
            [("repository", "string"), ("force", "boolean")],
            [],
            Risk::Critical
        ),
        frontend_tool!(
            "screen.capture",
            1,
            "a screenshot, described or returned",
            ["screen"],
            [("display", "integer")],
            [],
            Risk::Normal
        ),
    ]
}

/// Dispatches tool calls to the frontend and waits for the answers.
#[derive(Debug, Default)]
pub struct FrontendBridge {
    pending: Mutex<BTreeMap<String, oneshot::Sender<ToolResult>>>,
    /// What this particular frontend says it implements. Empty means "we have
    /// not been told", treated as everything — an older frontend that does
    /// not advertise still works.
    supported: Mutex<BTreeSet<String>>,
    timeout: Duration,
}

impl FrontendBridge {
    pub fn new() -> Self {
        Self::with_timeout(DEFAULT_TIMEOUT)
    }

    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            pending: Mutex::new(BTreeMap::new()),
            supported: Mutex::new(BTreeSet::new()),
            timeout,
        }
    }

    /// Record what the frontend can do, from the turn it just opened.
    pub fn advertise(&self, names: Option<&Value>) {
        let mut supported = self.supported.lock().expect("bridge");
        supported.clear();
        if let Some(list) = names.and_then(Value::as_array) {
            for name in list.iter().filter_map(Value::as_str) {
                supported.insert(name.to_string());
            }
        }
    }

    /// Whether a tool can be offered to the model at all.
    pub fn offers(&self, tool: &Tool) -> bool {
        if tool.runs != Runs::Frontend {
            return true;
        }
        let supported = self.supported.lock().expect("bridge");
        supported.is_empty() || supported.contains(tool.name)
    }

    /// The frontend answering a call. Returns whether anyone was waiting.
    ///
    /// An unknown id is not worth failing on: it means the turn moved on,
    /// usually because the call timed out, and the frontend should not care.
    pub fn deliver(&self, call_id: &str, result: ToolResult) -> bool {
        let sender = self.pending.lock().expect("bridge").remove(call_id);
        match sender {
            Some(sender) => sender.send(result).is_ok(),
            None => false,
        }
    }

    /// Send the request and wait for the frontend, or time out.
    pub async fn call<F, Fut>(
        &self,
        call_id: &str,
        emit: F,
        tool: &Tool,
        arguments: Value,
    ) -> ToolResult
    where
        F: FnOnce(Value) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .expect("bridge")
            .insert(call_id.to_string(), tx);

        emit(json!({
            "type": "tool.request",
            "id": call_id,
            "tool": tool.name,
            "version": tool.version,
            "arguments": arguments,
            "risk": tool.risk.as_str(),
        }))
        .await;

        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                self.pending.lock().expect("bridge").remove(call_id);
                ToolResult::failed(format!("{} was never answered", tool.name))
            }
            Err(_) => {
                self.pending.lock().expect("bridge").remove(call_id);
                ToolResult::failed(format!(
                    "{} did not answer within {} seconds",
                    tool.name,
                    self.timeout.as_secs()
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolRegistry;

    fn tool(name: &str) -> Tool {
        tools().into_iter().find(|t| t.name == name).unwrap()
    }

    #[tokio::test]
    async fn a_call_goes_out_and_the_answer_comes_back() {
        let bridge = std::sync::Arc::new(FrontendBridge::with_timeout(Duration::from_secs(5)));
        let sent = std::sync::Arc::new(Mutex::new(Vec::new()));

        let result = {
            let bridge_for_emit = bridge.clone();
            let sent = sent.clone();
            bridge
                .call(
                    "call-1",
                    move |message| {
                        sent.lock().unwrap().push(message.clone());
                        // The frontend answers.
                        bridge_for_emit.deliver(
                            message["id"].as_str().unwrap(),
                            ToolResult::ok("4 matches", "found 4 files"),
                        );
                        async {}
                    },
                    &tool("filesystem.search"),
                    json!({"pattern": "*.rs"}),
                )
                .await
        };

        assert!(result.ok);
        assert_eq!(result.summary, "4 matches");
        let sent = sent.lock().unwrap();
        assert_eq!(sent[0]["type"], "tool.request");
        assert_eq!(sent[0]["tool"], "filesystem.search");
        assert_eq!(sent[0]["risk"], "safe");
    }

    #[tokio::test]
    async fn a_frontend_that_never_answers_times_out_as_a_failed_call() {
        let bridge = FrontendBridge::with_timeout(Duration::from_millis(150));
        let result = bridge
            .call(
                "call-2",
                |_| async {},
                &tool("shell.run"),
                json!({"command": "ls"}),
            )
            .await;
        assert!(!result.ok);
        assert!(
            result.summary.contains("did not answer"),
            "{}",
            result.summary
        );
    }

    #[test]
    fn a_frontend_only_gets_offered_what_it_implements() {
        let bridge = FrontendBridge::new();
        let dir = tempfile::tempdir().unwrap();
        let memory = std::sync::Arc::new(crate::tools::builtin::Memory::open(
            dir.path().join("memory.json"),
        ));
        let registry = ToolRegistry::standard(memory);

        // Unknown means all, so an older frontend keeps working.
        assert!(bridge.offers(registry.get("vscode.workspace").unwrap()));

        bridge.advertise(Some(&json!(["filesystem.search", "shell.run"])));
        assert!(bridge.offers(registry.get("filesystem.search").unwrap()));
        assert!(!bridge.offers(registry.get("vscode.workspace").unwrap()));
        // Backend tools are unaffected by what the frontend can do.
        assert!(bridge.offers(registry.get("web.fetch").unwrap()));
    }

    #[test]
    fn late_and_duplicate_results_are_ignored() {
        let bridge = FrontendBridge::new();
        assert!(!bridge.deliver("never-asked", ToolResult::ok("?", "?")));
    }
}
