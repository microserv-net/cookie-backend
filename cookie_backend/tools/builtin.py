"""Tools that run here, on the backend.

Deliberately few. Anything touching the user's files, applications, screen or
shell belongs on their machine and is declared in `frontend.py` instead — this
process should not be able to read your home directory even if a model asks it
to nicely.

What is left is what genuinely lives here: the web, and memory.
"""

from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from . import BACKEND, NORMAL, SAFE, Tool, ToolRegistry, ToolResult

#: Pages are truncated before they reach a model. A 400 KB article costs more
#: context than it is worth, and the useful part is almost always near the top.
MAX_PAGE_CHARS = 6_000
_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_MARKUP = re.compile(r"<[^>]+>")


def _readable(raw: str) -> str:
    """Crude HTML to text. Good enough to read; not a parser."""
    text = _TAGS.sub(" ", raw)
    text = _MARKUP.sub(" ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


async def _fetch(arguments: dict) -> ToolResult:
    url = str(arguments.get("url", "")).strip()
    if not url.startswith(("http://", "https://")):
        return ToolResult.failed(f"{url!r} is not an http(s) URL")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            response = await client.get(url, headers={"user-agent": "cookie-backend/0.1"})
            response.raise_for_status()
    except httpx.HTTPError as e:
        return ToolResult.failed(f"could not fetch {url}: {e}")

    body = _readable(response.text)
    truncated = body[:MAX_PAGE_CHARS]
    return ToolResult(
        ok=True,
        summary=f"fetched {url} ({len(body)} characters)",
        data=truncated,
        evidence=f"HTTP {response.status_code}, {len(body)} characters, "
        f"begins: {truncated[:200]}",
    )


async def _search(arguments: dict) -> ToolResult:
    """Web search via DuckDuckGo's HTML endpoint.

    No API key, no account, no quota — which matters for something meant to
    run unattended on a machine in a cupboard.
    """
    query = str(arguments.get("query", "")).strip()
    if not query:
        return ToolResult.failed("search needs a query")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            response = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"user-agent": "Mozilla/5.0 (compatible; cookie-backend/0.1)"},
            )
            response.raise_for_status()
    except httpx.HTTPError as e:
        return ToolResult.failed(f"the search failed: {e}")

    results = []
    for match in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', response.text, re.DOTALL
    ):
        link, title = match.group(1), _readable(match.group(2))
        if title:
            results.append({"title": title, "url": html.unescape(link)})
        if len(results) >= 6:
            break

    if not results:
        return ToolResult(ok=False, summary=f"no results for {query!r}",
                          evidence="the search returned nothing usable")
    listed = "; ".join(f"{r['title']}" for r in results[:3])
    return ToolResult(
        ok=True,
        summary=f"{len(results)} results for {query!r}: {listed}",
        data=results,
        evidence=f"top result: {results[0]['title']} ({results[0]['url']})",
    )


class Memory:
    """Persistent memory with enough metadata to be deleted intelligently.

    The user should be able to say "forget what I have not asked about in two
    months", which needs `last_referenced_at` and not just `created_at` — the
    distinction between when something was learned and when it last mattered.

    Nothing expires on its own. Backend memory persisting by default is the
    opposite of the generated-audio policy on the frontend, and deliberately
    so: audio is a by-product, this is what Cookie knows.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._entries = json.loads(self.path.read_text()).get("entries", {})
        except (json.JSONDecodeError, OSError):
            self._entries = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entries": self._entries}, indent=2))
        tmp.replace(self.path)

    def remember(self, key: str, value: str, tags: list[str] | None = None) -> dict:
        now = time.time()
        existing = self._entries.get(key)
        entry = {
            "key": key,
            "value": value,
            "tags": tags or (existing or {}).get("tags", []),
            "created_at": (existing or {}).get("created_at", now),
            "updated_at": now,
            "last_referenced_at": now,
        }
        self._entries[key] = entry
        self._save()
        return entry

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        """Substring match over keys, values and tags.

        Not embeddings: on this machine an embedding model is another thing
        resident in RAM, and for a few hundred facts about one person's setup
        substring matching is not the bottleneck. Revisit when it is.
        """
        needle = query.lower().strip()
        hits = []
        for entry in self._entries.values():
            haystack = f"{entry['key']} {entry['value']} {' '.join(entry.get('tags', []))}".lower()
            if not needle or needle in haystack:
                hits.append(entry)
        hits.sort(key=lambda e: e.get("last_referenced_at", 0), reverse=True)
        now = time.time()
        for entry in hits[:limit]:
            entry["last_referenced_at"] = now
        if hits:
            self._save()
        return hits[:limit]

    def forget(self, key: str) -> bool:
        if key not in self._entries:
            return False
        del self._entries[key]
        self._save()
        return True

    def stale(self, older_than_days: float) -> list[dict]:
        """Entries nobody has asked about in a while.

        The safeguard against "delete some of your memory" wiping everything:
        this returns candidates, and deleting them is a separate, explicit act.
        """
        cutoff = time.time() - older_than_days * 86_400
        return [
            entry
            for entry in self._entries.values()
            if entry.get("last_referenced_at", entry.get("created_at", 0)) < cutoff
        ]

    def count(self) -> int:
        return len(self._entries)


def register_builtin(registry: ToolRegistry, memory: Memory) -> ToolRegistry:
    """Add the backend-side tools to `registry`."""

    async def remember(arguments: dict) -> ToolResult:
        key = str(arguments.get("key", "")).strip()
        value = str(arguments.get("value", "")).strip()
        if not key or not value:
            return ToolResult.failed("remembering needs a key and a value")
        tags = arguments.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        memory.remember(key, value, tags)
        return ToolResult(ok=True, summary=f"remembered {key}",
                          evidence=f"{key} = {value[:120]}")

    async def recall(arguments: dict) -> ToolResult:
        hits = memory.recall(str(arguments.get("query", "")))
        if not hits:
            return ToolResult(ok=False, summary="I don't know anything about that",
                              evidence="no matching memory")
        listed = "; ".join(f"{h['key']}: {h['value']}" for h in hits)
        return ToolResult(ok=True, summary=listed, data=hits,
                          evidence=f"{len(hits)} memories matched")

    async def forget(arguments: dict) -> ToolResult:
        key = str(arguments.get("key", "")).strip()
        if memory.forget(key):
            return ToolResult(ok=True, summary=f"forgot {key}", evidence=f"{key} removed")
        return ToolResult.failed(f"I had nothing stored under {key!r}")

    async def now(arguments: dict) -> ToolResult:
        stamp = time.strftime("%A %d %B %Y, %H:%M")
        return ToolResult(ok=True, summary=stamp, evidence=f"local clock: {stamp}")

    for tool in [
        Tool("web.search", 1, "search the web and return titles and links",
             ("web", "research"), {"query": "string"}, ("query",), BACKEND, SAFE, _search),
        Tool("web.fetch", 1, "fetch a page and return its readable text",
             ("web", "research"), {"url": "string"}, ("url",), BACKEND, SAFE, _fetch),
        Tool("memory.remember", 1, "store a fact for later",
             ("memory",), {"key": "string", "value": "string", "tags": "list of strings"},
             ("key", "value"), BACKEND, NORMAL, remember),
        Tool("memory.recall", 1, "look up what you were told before",
             ("memory",), {"query": "string"}, ("query",), BACKEND, SAFE, recall),
        Tool("memory.forget", 1, "delete one stored fact",
             ("memory",), {"key": "string"}, ("key",), BACKEND, NORMAL, forget),
        Tool("time.now", 1, "the current date and time",
             ("time",), {}, (), BACKEND, SAFE, now),
    ]:
        registry.register(tool)
    return registry
