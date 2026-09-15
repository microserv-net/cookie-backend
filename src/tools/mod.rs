//! The tool system.
//!
//! Three properties matter more than the tools themselves:
//!
//! **Versioned contracts.** A tool is `name@vN` with a stated parameter
//! schema. Adding or replacing one must never require touching the
//! orchestrator, and a frontend running last month's build must not break
//! because a tool grew an argument.
//!
//! **Discovery, not enumeration.** There will be dozens of tools. Sending
//! every schema on every request is the obvious approach and it is wrong on a
//! machine where context costs seconds: the router names a capability, and
//! only the tools tagged with it reach the model.
//!
//! **A tool declares where it runs.** Filesystem and shell work belongs on
//! the user's laptop; web and memory belong here. That boundary is a security
//! property: the backend asks for `filesystem.search` with arguments, it does
//! not ship a shell string. Nothing in this module decides whether an action
//! is *allowed* — that is the frontend's job, because it is the machine being
//! acted upon.

pub mod builtin;
pub mod frontend;

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;

/// Where a tool executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Runs {
    Backend,
    Frontend,
}

/// How much damage a tool can do if the model is wrong about wanting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Risk {
    /// Read-only: listing, searching, fetching.
    Safe,
    /// Recoverable writes: editing a file, committing.
    Normal,
    /// Deletes, installs, credentials.
    Dangerous,
    /// Irreversible: disk operations, force pushes.
    Critical,
}

impl Risk {
    pub fn as_str(self) -> &'static str {
        match self {
            Risk::Safe => "safe",
            Risk::Normal => "normal",
            Risk::Dangerous => "dangerous",
            Risk::Critical => "critical",
        }
    }
}

/// What a tool hands back.
///
/// `summary` is for the model and for speech; `data` is for tools feeding
/// other tools; `evidence` is what the validator judges against, and is the
/// field that turns "the worker says it worked" into something checkable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub ok: bool,
    pub summary: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub evidence: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolResult {
    pub fn ok(summary: impl Into<String>, evidence: impl Into<String>) -> Self {
        Self {
            ok: true,
            summary: summary.into(),
            evidence: evidence.into(),
            data: None,
            error: None,
        }
    }

    pub fn failed(error: impl Into<String>) -> Self {
        let error = error.into();
        Self {
            ok: false,
            summary: error.clone(),
            evidence: error.clone(),
            data: None,
            error: Some(error),
        }
    }

    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
}

/// A backend tool's implementation.
pub type Handler = Arc<
    dyn Fn(Value) -> futures_util::future::BoxFuture<'static, ToolResult> + Send + Sync + 'static,
>;

/// One versioned capability.
#[derive(Clone)]
pub struct Tool {
    pub name: &'static str,
    pub version: u32,
    pub summary: &'static str,
    /// Capability tags the router can ask for: "files", "web", "memory"…
    pub capabilities: &'static [&'static str],
    /// Argument name to type, kept minimal because it goes into a prompt.
    pub parameters: &'static [(&'static str, &'static str)],
    pub required: &'static [&'static str],
    pub runs: Runs,
    pub risk: Risk,
    pub handler: Option<Handler>,
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("version", &self.version)
            .field("runs", &self.runs)
            .field("risk", &self.risk)
            .finish()
    }
}

impl Tool {
    pub fn id(&self) -> String {
        format!("{}@v{}", self.name, self.version)
    }

    /// One line. Every extra word is paid for on every request.
    pub fn describe_for_model(&self) -> String {
        let arguments: Vec<String> = self
            .parameters
            .iter()
            .map(|(key, kind)| {
                let optional = if self.required.contains(key) { "" } else { "?" };
                format!("{key}{optional}: {kind}")
            })
            .collect();
        format!("{}({}) — {}", self.name, arguments.join(", "), self.summary)
    }

    /// A complaint, or `None` if the arguments are usable.
    ///
    /// Checked before anything runs, so a hallucinated argument name gets the
    /// model a sentence it can act on rather than a panic.
    pub fn validate(&self, arguments: &Value) -> Option<String> {
        let missing: Vec<&str> = self
            .required
            .iter()
            .copied()
            .filter(|key| {
                arguments
                    .get(*key)
                    .map(|v| v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false))
                    .unwrap_or(true)
            })
            .collect();
        if !missing.is_empty() {
            return Some(format!("{} needs {}", self.name, missing.join(", ")));
        }
        if let Some(object) = arguments.as_object() {
            let unknown: Vec<&str> = object
                .keys()
                .map(String::as_str)
                .filter(|key| !self.parameters.iter().any(|(name, _)| name == key))
                .collect();
            if !unknown.is_empty() {
                return Some(format!(
                    "{} has no argument {}",
                    self.name,
                    unknown.join(", ")
                ));
            }
        }
        None
    }
}

/// Everything available, and the means to find the few that matter.
#[derive(Debug, Default, Clone)]
pub struct ToolRegistry {
    tools: BTreeMap<&'static str, Tool>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// The standard set: backend tools plus the frontend catalogue.
    pub fn standard(memory: Arc<builtin::Memory>) -> Self {
        let mut registry = Self::new();
        for tool in builtin::tools(memory) {
            registry.register(tool);
        }
        for tool in frontend::tools() {
            registry.register(tool);
        }
        registry
    }

    /// Register a tool. A newer version supersedes an older one by name.
    pub fn register(&mut self, tool: Tool) {
        if let Some(existing) = self.tools.get(tool.name) {
            if existing.version > tool.version {
                return;
            }
        }
        self.tools.insert(tool.name, tool);
    }

    /// Models copy the id back at us, so `name@v1` resolves too.
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name.split('@').next().unwrap_or(name))
    }

    pub fn all(&self) -> Vec<&Tool> {
        self.tools.values().collect()
    }

    /// Every capability tag, for the router to choose from.
    pub fn capabilities(&self) -> Vec<String> {
        let mut tags: Vec<String> = self
            .tools
            .values()
            .flat_map(|tool| tool.capabilities.iter().map(|c| c.to_string()))
            .collect();
        tags.sort();
        tags.dedup();
        tags
    }

    /// The tools worth showing the model for this request.
    ///
    /// Ordered by how many of the requested capabilities each covers, so a
    /// tool matching two beats one matching one. Capped, because an unbounded
    /// list defeats the point.
    pub fn discover(&self, capabilities: &[String], limit: usize) -> Vec<&Tool> {
        let wanted: Vec<String> = capabilities
            .iter()
            .map(|c| c.trim().to_lowercase())
            .filter(|c| !c.is_empty())
            .collect();
        if wanted.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, &Tool)> = self
            .tools
            .values()
            .filter_map(|tool| {
                let overlap = tool
                    .capabilities
                    .iter()
                    .filter(|tag| wanted.iter().any(|w| w == &tag.to_lowercase()))
                    .count();
                (overlap > 0).then_some((overlap, tool))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(b.1.name)));
        scored
            .into_iter()
            .take(limit)
            .map(|(_, tool)| tool)
            .collect()
    }

    /// The block that goes into the worker's prompt.
    pub fn describe(tools: &[&Tool]) -> String {
        tools
            .iter()
            .map(|tool| format!("- {}", tool.describe_for_model()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A model's request to use a tool.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

impl ToolCall {
    /// Read a call out of a model's reply, if that is what it is.
    pub fn from_value(value: &Value, id: impl Into<String>) -> Option<Self> {
        let name = value.get("tool")?.as_str()?.trim();
        if name.is_empty() {
            return None;
        }
        let arguments = value
            .get("arguments")
            .cloned()
            .filter(Value::is_object)
            .unwrap_or_else(|| serde_json::json!({}));
        Some(Self {
            id: id.into(),
            name: name.to_string(),
            arguments,
        })
    }
}

/// Anchors the crate's `Result` for tool authors.
#[allow(dead_code)]
fn _result_anchor() -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn registry() -> ToolRegistry {
        let dir = tempfile::tempdir().unwrap();
        let memory = Arc::new(builtin::Memory::open(dir.path().join("memory.json")));
        std::mem::forget(dir);
        ToolRegistry::standard(memory)
    }

    #[test]
    fn discovery_returns_only_what_was_asked_for() {
        let registry = registry();
        let found: Vec<&str> = registry
            .discover(&["memory".into()], 8)
            .iter()
            .map(|tool| tool.name)
            .collect();
        assert!(found.contains(&"memory.remember"));
        assert!(!found.contains(&"shell.run"));
    }

    #[test]
    fn discovery_prefers_tools_covering_more_of_the_request() {
        let registry = registry();
        let found: Vec<&str> = registry
            .discover(&["git".into(), "code".into()], 3)
            .iter()
            .map(|tool| tool.name)
            .collect();
        // git.* carries both tags; filesystem.read carries only "code".
        assert!(
            found.iter().all(|name| name.starts_with("git.")),
            "{found:?}"
        );
    }

    #[test]
    fn nothing_is_offered_when_nothing_is_asked_for() {
        assert!(registry().discover(&[], 8).is_empty());
    }

    #[test]
    fn a_newer_version_supersedes_an_older_one() {
        let mut registry = ToolRegistry::new();
        let mut tool = frontend::tools()[0].clone();
        tool.version = 1;
        registry.register(tool.clone());
        tool.version = 2;
        registry.register(tool.clone());
        assert_eq!(registry.get(tool.name).unwrap().version, 2);
        tool.version = 1;
        registry.register(tool.clone());
        assert_eq!(
            registry.get(tool.name).unwrap().version,
            2,
            "older must not clobber newer"
        );
    }

    #[test]
    fn tools_are_addressable_by_id_as_well_as_name() {
        let registry = registry();
        assert!(registry.get("web.fetch@v1").is_some());
    }

    #[test]
    fn arguments_are_checked_before_anything_runs() {
        let registry = registry();
        let write = registry.get("filesystem.write").unwrap();
        assert_eq!(
            write.validate(&json!({"path": "/tmp/x"})).unwrap(),
            "filesystem.write needs content"
        );
        assert!(write
            .validate(&json!({"path": "/tmp/x", "content": "y", "mode": "0644"}))
            .unwrap()
            .contains("no argument"));
        assert!(write
            .validate(&json!({"path": "/tmp/x", "content": "y"}))
            .is_none());
    }

    #[test]
    fn descriptions_are_one_line_and_mark_optional_arguments() {
        let registry = registry();
        let line = registry
            .get("filesystem.read")
            .unwrap()
            .describe_for_model();
        assert!(!line.contains('\n'));
        assert!(line.contains("path: string"));
        assert!(line.contains("from_line?: integer"));
    }

    #[test]
    fn risk_is_declared_so_the_frontend_can_decide() {
        let registry = registry();
        assert_eq!(registry.get("filesystem.read").unwrap().risk, Risk::Safe);
        assert_eq!(
            registry.get("filesystem.delete").unwrap().risk,
            Risk::Dangerous
        );
        assert_eq!(registry.get("git.push").unwrap().risk, Risk::Critical);
    }

    #[test]
    fn a_tool_call_is_recognised_and_anything_else_is_not() {
        let call = ToolCall::from_value(
            &json!({"tool": "shell.run", "arguments": {"command": "ls"}}),
            "c1",
        )
        .unwrap();
        assert_eq!(call.name, "shell.run");
        assert_eq!(call.arguments["command"], "ls");
        assert!(ToolCall::from_value(&json!({"result": "done", "ok": true}), "c2").is_none());
        // A call with no arguments is still a call.
        assert!(ToolCall::from_value(&json!({"tool": "time.now"}), "c3").is_some());
    }
}
