//! Getting structured data out of models that were asked for it politely.
//!
//! A model told to answer with one JSON object will, often enough to matter,
//! answer with a JSON object wrapped in a code fence, or prefixed with
//! "Sure!", or followed by an explanation, or with `<think>` tags around its
//! reasoning because that is what Qwen3 does. On a small model this is not an
//! edge case; it is Tuesday.
//!
//! Being strict means a working assistant fails on a stray backtick. Being
//! sloppy means garbage reaches the orchestrator. So: extract the first
//! balanced JSON object, tolerate the usual wrappers, and return `None` when
//! there is genuinely nothing — the caller then degrades to conversation
//! rather than failing.

use serde_json::Value;

/// Remove `<think>` blocks, including one left unterminated by a cut stream.
///
/// Reasoning must never reach the user: hearing a model think aloud is both a
/// privacy problem and deeply odd.
pub fn strip_reasoning(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let lower = text.to_lowercase();
    let mut cursor = 0usize;
    while let Some(start) = lower[cursor..].find("<think>") {
        let start = cursor + start;
        out.push_str(&text[cursor..start]);
        match lower[start..].find("</think>") {
            Some(end) => cursor = start + end + "</think>".len(),
            None => {
                // Unterminated: everything after the tag is reasoning.
                return out.trim().to_string();
            }
        }
    }
    out.push_str(&text[cursor..]);
    out.trim().to_string()
}

/// The first balanced JSON object in `text`.
///
/// Scans rather than pattern-matches, so nested objects and braces inside
/// strings do not truncate the match.
pub fn find_json_object(text: &str) -> Option<Value> {
    let cleaned = strip_reasoning(text);
    if let Some(fenced) = fenced_block(&cleaned) {
        if let Some(value) = first_object(&fenced) {
            return Some(value);
        }
    }
    first_object(&cleaned)
}

fn fenced_block(text: &str) -> Option<String> {
    let start = text.find("```")?;
    let after = &text[start + 3..];
    let body_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
    let body = &after[body_start..];
    let end = body.find("```")?;
    Some(body[..end].to_string())
}

fn first_object(text: &str) -> Option<Value> {
    let bytes: Vec<char> = text.chars().collect();
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut escaped = false;

    for (index, ch) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if *ch == '\\' {
                escaped = true;
            } else if *ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(index);
                }
                depth += 1;
            }
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if let Some(begin) = start {
                        let candidate: String = bytes[begin..=index].iter().collect();
                        match serde_json::from_str::<Value>(&candidate) {
                            Ok(Value::Object(map)) => return Some(Value::Object(map)),
                            _ => start = None,
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Clean a model's prose for speech.
///
/// Strips reasoning, code fences and the markdown that creeps in however
/// firmly the prompt asks for none — a synthesiser reads asterisks aloud.
pub fn spoken_text(text: &str) -> String {
    let mut out = strip_reasoning(text);
    while let Some(block) = fenced_block(&out) {
        out = out.replacen(&format!("```{block}```"), " ", 1);
        if let Some(start) = out.find("```") {
            // Unbalanced fence: drop the remainder rather than read it aloud.
            out.truncate(start);
            break;
        }
    }
    let filtered: String = out
        .chars()
        .filter(|c| !matches!(c, '*' | '_' | '`' | '#'))
        .collect();
    filtered
        .lines()
        .map(|line| line.trim_start_matches(['-', '•', ' ']).trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Convenience: a string field, trimmed, or empty.
pub fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_object_parses() {
        assert_eq!(
            find_json_object(r#"{"kind": "chat"}"#).unwrap()["kind"],
            "chat"
        );
    }

    #[test]
    fn a_fenced_object_with_a_preamble_parses() {
        let text = "Sure! Here you go:\n```json\n{\"kind\": \"task\", \"weight\": \"heavy\"}\n```\nHope that helps.";
        let parsed = find_json_object(text).unwrap();
        assert_eq!(parsed["weight"], "heavy");
    }

    #[test]
    fn reasoning_never_parses_as_the_answer() {
        let text = r#"<think>the user wants {"kind": "chat"} probably</think>{"kind": "task"}"#;
        assert_eq!(find_json_object(text).unwrap()["kind"], "task");
    }

    #[test]
    fn unterminated_reasoning_is_discarded() {
        assert_eq!(strip_reasoning("answer <think>still going"), "answer");
    }

    #[test]
    fn nested_objects_and_braces_in_strings_survive() {
        let text =
            r#"{"steps": [{"what": "say {hello}", "done_when": "it is said"}], "say": "ok"}"#;
        let parsed = find_json_object(text).unwrap();
        assert_eq!(parsed["steps"][0]["what"], "say {hello}");
    }

    #[test]
    fn nothing_usable_is_none_rather_than_a_guess() {
        assert!(find_json_object("I'm not sure what you mean.").is_none());
        assert!(find_json_object("").is_none());
        assert!(find_json_object("{ not json at all").is_none());
    }

    #[test]
    fn spoken_text_removes_what_a_synthesiser_would_read_aloud() {
        let text = "<think>hmm</think>**Done.** Here is the code:\n```py\nx=1\n```\n- it works";
        let spoken = spoken_text(text);
        assert!(!spoken.contains("think"));
        assert!(!spoken.contains('*'));
        assert!(!spoken.contains("x=1"), "{spoken}");
        assert!(spoken.starts_with("Done."), "{spoken}");
    }
}
