"""Getting structured data out of models that were asked for it politely.

A model told to answer with one JSON object will, often enough to matter,
answer with a JSON object wrapped in a code fence, or prefixed with "Sure!",
or followed by an explanation, or with `<think>` tags around its reasoning
because that is what Qwen3 does. On a small model this is not an edge case; it
is Tuesday.

Being strict here means a working assistant fails on a stray backtick. Being
sloppy means garbage reaches the orchestrator. So: extract the first balanced
JSON object, tolerate the usual wrappers, and give the caller a typed default
when there is genuinely nothing usable — the caller then degrades to
conversation rather than crashing.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Qwen3 and friends emit their reasoning in these. It must never reach the
#: user, and it must never be parsed as an answer.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove `<think>` blocks and any unterminated trailing one."""
    text = _THINK.sub("", text)
    # A stream cut off mid-thought leaves an opening tag with no close.
    opening = text.lower().rfind("<think>")
    if opening != -1:
        text = text[:opening]
    return text.strip()


def find_json_object(text: str) -> dict[str, Any] | None:
    """The first balanced JSON object in `text`, or `None`.

    Scans rather than regexes, so nested objects and braces inside strings do
    not truncate the match.
    """
    text = strip_reasoning(text)
    fenced = _FENCE.search(text)
    if fenced:
        candidate = _first_object(fenced.group(1))
        if candidate is not None:
            return candidate
    return _first_object(text)


def _first_object(text: str) -> dict[str, Any] | None:
    depth = 0
    start = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    value = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(value, dict):
                    return value
                start = None
    return None


def spoken_text(text: str) -> str:
    """Clean a model's prose for speech.

    Strips reasoning, code fences and the markdown that creeps in however
    firmly the prompt asks for none, because a synthesiser reads asterisks
    aloud.
    """
    text = strip_reasoning(text)
    text = _FENCE.sub(" ", text)
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()
