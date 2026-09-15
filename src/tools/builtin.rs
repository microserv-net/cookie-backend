//! Tools that run here.
//!
//! Deliberately few. Anything touching the user's files, applications, screen
//! or shell belongs on their machine and is declared in [`super::frontend`]
//! instead — this process should not be able to read your home directory even
//! if a model asks it to nicely.
//!
//! What is left is what genuinely lives here: the web, and memory.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{Handler, Risk, Runs, Tool, ToolResult};

/// Pages are truncated before they reach a model: a 400 KB article costs more
/// context than it is worth, and the useful part is near the top.
const MAX_PAGE_CHARS: usize = 6_000;

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// One thing Cookie knows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub created_at: u64,
    pub updated_at: u64,
    /// When this last *mattered*, as opposed to when it was learned. The
    /// distinction is what makes "forget what I have not asked about in two
    /// months" answerable.
    pub last_referenced_at: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct StoredMemory {
    #[serde(default)]
    entries: BTreeMap<String, MemoryEntry>,
}

/// Persistent memory.
///
/// Nothing expires on its own. Backend memory persisting by default is the
/// opposite of the frontend's generated-audio policy, and deliberately so:
/// audio is a by-product, this is what Cookie knows.
#[derive(Debug)]
pub struct Memory {
    path: PathBuf,
    entries: Mutex<BTreeMap<String, MemoryEntry>>,
}

impl Memory {
    pub fn open(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let entries = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<StoredMemory>(&text).ok())
            .map(|stored| stored.entries)
            .unwrap_or_default();
        Self {
            path,
            entries: Mutex::new(entries),
        }
    }

    fn save(&self, entries: &BTreeMap<String, MemoryEntry>) {
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let stored = StoredMemory {
            entries: entries.clone(),
        };
        if let Ok(text) = serde_json::to_string_pretty(&stored) {
            let tmp = self.path.with_extension("tmp");
            if std::fs::write(&tmp, text).is_ok() {
                let _ = std::fs::rename(&tmp, &self.path);
            }
        }
    }

    pub fn remember(&self, key: &str, value: &str, tags: Vec<String>) -> MemoryEntry {
        let mut entries = self.entries.lock().expect("memory");
        let now = now();
        let entry = entries
            .get(key)
            .map(|existing| MemoryEntry {
                value: value.to_string(),
                tags: if tags.is_empty() {
                    existing.tags.clone()
                } else {
                    tags.clone()
                },
                updated_at: now,
                last_referenced_at: now,
                ..existing.clone()
            })
            .unwrap_or_else(|| MemoryEntry {
                key: key.to_string(),
                value: value.to_string(),
                tags: tags.clone(),
                created_at: now,
                updated_at: now,
                last_referenced_at: now,
            });
        entries.insert(key.to_string(), entry.clone());
        self.save(&entries);
        entry
    }

    /// Substring match over keys, values and tags.
    ///
    /// Not embeddings: an embedding model is another thing resident in RAM on
    /// a machine that has none spare, and for a few hundred facts about one
    /// person's setup this is not the bottleneck. Worth revisiting when it is.
    pub fn recall(&self, query: &str, limit: usize) -> Vec<MemoryEntry> {
        let mut entries = self.entries.lock().expect("memory");
        let needle = query.trim().to_lowercase();
        let mut hits: Vec<MemoryEntry> = entries
            .values()
            .filter(|entry| {
                needle.is_empty()
                    || format!("{} {} {}", entry.key, entry.value, entry.tags.join(" "))
                        .to_lowercase()
                        .contains(&needle)
            })
            .cloned()
            .collect();
        hits.sort_by_key(|entry| std::cmp::Reverse(entry.last_referenced_at));
        hits.truncate(limit);

        let now = now();
        for hit in &hits {
            if let Some(entry) = entries.get_mut(&hit.key) {
                entry.last_referenced_at = now;
            }
        }
        if !hits.is_empty() {
            self.save(&entries);
        }
        hits
    }

    pub fn forget(&self, key: &str) -> bool {
        let mut entries = self.entries.lock().expect("memory");
        if entries.remove(key).is_none() {
            return false;
        }
        self.save(&entries);
        true
    }

    /// Entries nobody has asked about in a while.
    ///
    /// The safeguard against "delete some of your memory" wiping everything:
    /// this returns *candidates*, and deleting them is a separate act.
    pub fn stale(&self, older_than_days: f64) -> Vec<MemoryEntry> {
        // Signed arithmetic on purpose: a negative window is how a caller (or
        // a test) asks "what would go if I said *everything* was stale", and
        // clamping it to zero would quietly answer a different question.
        let seconds = (older_than_days * 86_400.0) as i64;
        let cutoff = (now() as i64 - seconds).max(0) as u64;
        self.entries
            .lock()
            .expect("memory")
            .values()
            .filter(|entry| entry.last_referenced_at < cutoff)
            .cloned()
            .collect()
    }

    pub fn count(&self) -> usize {
        self.entries.lock().expect("memory").len()
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// The web
// ---------------------------------------------------------------------------

/// Crude HTML to text. Good enough to read; not a parser.
fn readable(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() / 2);
    let mut in_tag = false;
    let mut skipping: Option<&str> = None;
    let lower = raw.to_lowercase();
    let bytes: Vec<char> = raw.chars().collect();
    let lower_chars: Vec<char> = lower.chars().collect();
    let mut index = 0;

    while index < bytes.len() {
        // Script and style contents are not prose and confuse everything.
        if let Some(tag) = skipping {
            let close = format!("</{tag}");
            if lower_chars[index..].starts_with(&close.chars().collect::<Vec<_>>()[..]) {
                skipping = None;
                in_tag = true;
            }
            index += 1;
            continue;
        }
        if lower_chars[index..].starts_with(&['<', 's', 'c', 'r', 'i', 'p', 't']) {
            skipping = Some("script");
            index += 1;
            continue;
        }
        if lower_chars[index..].starts_with(&['<', 's', 't', 'y', 'l', 'e']) {
            skipping = Some("style");
            index += 1;
            continue;
        }
        match bytes[index] {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            ch if !in_tag => out.push(ch),
            _ => {}
        }
        index += 1;
    }

    let unescaped = out
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
    unescaped.split_whitespace().collect::<Vec<_>>().join(" ")
}

async fn fetch(arguments: Value) -> ToolResult {
    let url = arguments
        .get("url")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return ToolResult::failed(format!("{url:?} is not an http(s) URL"));
    }
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent("cookie-backend/0.1")
        .build()
    {
        Ok(client) => client,
        Err(e) => return ToolResult::failed(format!("could not build a client: {e}")),
    };
    let response = match client.get(&url).send().await {
        Ok(response) => response,
        Err(e) => return ToolResult::failed(format!("could not fetch {url}: {e}")),
    };
    let status = response.status();
    if !status.is_success() {
        return ToolResult::failed(format!("{url} returned {status}"));
    }
    let body = response.text().await.unwrap_or_default();
    let text = readable(&body);
    let truncated: String = text.chars().take(MAX_PAGE_CHARS).collect();
    let preview: String = truncated.chars().take(200).collect();
    ToolResult::ok(
        format!("fetched {url} ({} characters)", text.chars().count()),
        format!(
            "HTTP {status}, {} characters, begins: {preview}",
            text.chars().count()
        ),
    )
    .with_data(json!(truncated))
}

async fn search(arguments: Value) -> ToolResult {
    let query = arguments
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if query.is_empty() {
        return ToolResult::failed("search needs a query");
    }
    // DuckDuckGo's HTML endpoint: no API key, no account, no quota, which
    // matters for something meant to run unattended on a machine in a
    // cupboard.
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent("Mozilla/5.0 (compatible; cookie-backend/0.1)")
        .build()
    {
        Ok(client) => client,
        Err(e) => return ToolResult::failed(format!("could not build a client: {e}")),
    };
    let response = match client
        .post("https://html.duckduckgo.com/html/")
        .form(&[("q", query.as_str())])
        .send()
        .await
    {
        Ok(response) => response,
        Err(e) => return ToolResult::failed(format!("the search failed: {e}")),
    };
    let body = response.text().await.unwrap_or_default();

    let mut results = Vec::new();
    for chunk in body.split("class=\"result__a\"").skip(1) {
        let Some(title_start) = chunk.find('>') else {
            continue;
        };
        let Some(title_end) = chunk[title_start..].find("</a>") else {
            continue;
        };
        let title = readable(&chunk[title_start..title_start + title_end]);
        let url = chunk
            .rsplit_once("href=\"")
            .and_then(|(before, _)| before.rsplit_once("href=\"").map(|(_, u)| u))
            .unwrap_or_default();
        if !title.is_empty() {
            results.push(json!({"title": title, "url": url}));
        }
        if results.len() >= 6 {
            break;
        }
    }

    if results.is_empty() {
        return ToolResult {
            ok: false,
            summary: format!("no results for {query:?}"),
            evidence: "the search returned nothing usable".into(),
            data: None,
            error: None,
        };
    }
    let titles: Vec<String> = results
        .iter()
        .take(3)
        .filter_map(|r| r["title"].as_str().map(str::to_owned))
        .collect();
    let first = results[0]["title"].as_str().unwrap_or_default().to_string();
    ToolResult::ok(
        format!(
            "{} results for {query:?}: {}",
            results.len(),
            titles.join("; ")
        ),
        format!("top result: {first}"),
    )
    .with_data(json!(results))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

fn handler<F, Fut>(f: F) -> Handler
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ToolResult> + Send + 'static,
{
    Arc::new(move |arguments| f(arguments).boxed())
}

/// The backend-side tools.
pub fn tools(memory: Arc<Memory>) -> Vec<Tool> {
    let remember = {
        let memory = Arc::clone(&memory);
        handler(move |arguments: Value| {
            let memory = Arc::clone(&memory);
            async move {
                let key = arguments
                    .get("key")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                let value = arguments
                    .get("value")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if key.is_empty() || value.is_empty() {
                    return ToolResult::failed("remembering needs a key and a value");
                }
                let tags = arguments
                    .get("tags")
                    .and_then(Value::as_array)
                    .map(|tags| {
                        tags.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect()
                    })
                    .unwrap_or_default();
                memory.remember(&key, &value, tags);
                let preview: String = value.chars().take(120).collect();
                ToolResult::ok(format!("remembered {key}"), format!("{key} = {preview}"))
            }
        })
    };

    let recall = {
        let memory = Arc::clone(&memory);
        handler(move |arguments: Value| {
            let memory = Arc::clone(&memory);
            async move {
                let query = arguments
                    .get("query")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let hits = memory.recall(query, 5);
                if hits.is_empty() {
                    return ToolResult {
                        ok: false,
                        summary: "I don't know anything about that".into(),
                        evidence: "no matching memory".into(),
                        data: None,
                        error: None,
                    };
                }
                let listed = hits
                    .iter()
                    .map(|entry| format!("{}: {}", entry.key, entry.value))
                    .collect::<Vec<_>>()
                    .join("; ");
                let count = hits.len();
                ToolResult::ok(listed, format!("{count} memories matched"))
                    .with_data(serde_json::to_value(hits).unwrap_or(Value::Null))
            }
        })
    };

    let forget = {
        let memory = Arc::clone(&memory);
        handler(move |arguments: Value| {
            let memory = Arc::clone(&memory);
            async move {
                let key = arguments
                    .get("key")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if memory.forget(&key) {
                    ToolResult::ok(format!("forgot {key}"), format!("{key} removed"))
                } else {
                    ToolResult::failed(format!("I had nothing stored under {key:?}"))
                }
            }
        })
    };

    vec![
        Tool {
            name: "web.search",
            version: 1,
            summary: "search the web and return titles and links",
            capabilities: &["web", "research"],
            parameters: &[("query", "string")],
            required: &["query"],
            runs: Runs::Backend,
            risk: Risk::Safe,
            handler: Some(handler(search)),
        },
        Tool {
            name: "web.fetch",
            version: 1,
            summary: "fetch a page and return its readable text",
            capabilities: &["web", "research"],
            parameters: &[("url", "string")],
            required: &["url"],
            runs: Runs::Backend,
            risk: Risk::Safe,
            handler: Some(handler(fetch)),
        },
        Tool {
            name: "memory.remember",
            version: 1,
            summary: "store a fact for later",
            capabilities: &["memory"],
            parameters: &[
                ("key", "string"),
                ("value", "string"),
                ("tags", "list of strings"),
            ],
            required: &["key", "value"],
            runs: Runs::Backend,
            risk: Risk::Normal,
            handler: Some(remember),
        },
        Tool {
            name: "memory.recall",
            version: 1,
            summary: "look up what you were told before",
            capabilities: &["memory"],
            parameters: &[("query", "string")],
            required: &["query"],
            runs: Runs::Backend,
            risk: Risk::Safe,
            handler: Some(recall),
        },
        Tool {
            name: "memory.forget",
            version: 1,
            summary: "delete one stored fact",
            capabilities: &["memory"],
            parameters: &[("key", "string")],
            required: &["key"],
            runs: Runs::Backend,
            risk: Risk::Normal,
            handler: Some(forget),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory() -> (Arc<Memory>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (Arc::new(Memory::open(dir.path().join("memory.json"))), dir)
    }

    #[test]
    fn memory_persists_and_records_when_it_last_mattered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.json");
        Memory::open(&path).remember("github token", "in the keychain", vec!["setup".into()]);

        let reopened = Memory::open(&path);
        let hits = reopened.recall("keychain", 5);
        assert_eq!(hits[0].value, "in the keychain");
        assert!(hits[0].last_referenced_at >= hits[0].created_at);
    }

    #[test]
    fn stale_returns_candidates_rather_than_deleting_them() {
        let (memory, _dir) = memory();
        memory.remember("old thing", "value", vec![]);
        assert!(memory.stale(60.0).is_empty());
        assert_eq!(memory.stale(-1.0).len(), 1);
        // Nothing was deleted by asking.
        assert_eq!(memory.count(), 1);
    }

    #[test]
    fn updating_a_memory_keeps_its_history() {
        let (memory, _dir) = memory();
        let first = memory.remember("editor", "vim", vec!["setup".into()]);
        let second = memory.remember("editor", "vscode", vec![]);
        assert_eq!(second.created_at, first.created_at);
        assert_eq!(second.value, "vscode");
        assert_eq!(
            second.tags,
            vec!["setup".to_string()],
            "tags survive an update"
        );
    }

    #[tokio::test]
    async fn the_memory_tools_round_trip() {
        let (memory, _dir) = memory();
        let tools = tools(Arc::clone(&memory));
        let find = |name: &str| {
            tools
                .iter()
                .find(|t| t.name == name)
                .unwrap()
                .handler
                .clone()
                .unwrap()
        };

        assert!(
            find("memory.remember")(json!({"key": "editor", "value": "vscode"}))
                .await
                .ok
        );
        let recalled = find("memory.recall")(json!({"query": "editor"})).await;
        assert!(recalled.summary.contains("vscode"));
        assert!(find("memory.forget")(json!({"key": "editor"})).await.ok);
        assert!(!find("memory.recall")(json!({"query": "editor"})).await.ok);
    }

    #[tokio::test]
    async fn a_tool_given_nonsense_fails_rather_than_panicking() {
        let result = fetch(json!({"url": "not-a-url"})).await;
        assert!(!result.ok);
        assert!(result.summary.contains("http"));
    }

    #[test]
    fn markup_is_stripped_before_a_model_sees_it() {
        let html = "<html><head><style>p{color:red}</style></head><body>\
                    <script>alert(1)</script><p>Hello &amp; welcome</p></body></html>";
        let text = readable(html);
        assert!(text.contains("Hello & welcome"), "{text}");
        assert!(!text.contains("alert"));
        assert!(!text.contains("color:red"));
    }
}
