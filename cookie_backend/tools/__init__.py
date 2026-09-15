"""The tool system.

Three properties matter more than the tools themselves:

**Versioned contracts.** A tool is `name@version` with a stated parameter
schema. Adding or replacing one must never require touching the orchestrator,
and a frontend running last month's build must not break because a tool grew
an argument.

**Discovery, not enumeration.** There will eventually be dozens of tools.
Sending every schema on every request is the obvious approach and it is wrong
on a machine where context costs seconds: the router names a capability, and
only the tools tagged with it reach the model. `describe_for_model` is
deliberately terse for the same reason.

**A tool declares where it runs.** Filesystem and shell work belongs on the
user's laptop, not here; web and memory belong here. The registry records
which, and the orchestrator dispatches accordingly. That boundary is a
security property: the backend asks for `filesystem.search` with arguments, it
does not ship a shell string.

Risk is declared per tool and is what the frontend's confirmation policy keys
off. Nothing in this package decides whether an action is *allowed* — that is
the frontend's job, because it is the machine being acted upon.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

#: Where a tool executes.
BACKEND = "backend"
FRONTEND = "frontend"

#: How much damage a tool can do if the model is wrong about wanting it.
SAFE = "safe"          # read-only: listing, searching, fetching
NORMAL = "normal"      # writes that are recoverable: edit a file, commit
DANGEROUS = "dangerous"  # deletes, installs, credentials
CRITICAL = "critical"  # irreversible: disk operations, force pushes


@dataclass
class ToolResult:
    """What a tool hands back.

    `summary` is for the model and for speech; `data` is for tools that feed
    other tools; `evidence` is what the validator judges against, and is the
    field that turns "the worker says it worked" into something checkable.
    """

    ok: bool
    summary: str
    data: Any = None
    evidence: str = ""
    error: str | None = None

    def to_message(self) -> dict:
        message = {"ok": self.ok, "summary": self.summary}
        if self.evidence:
            message["evidence"] = self.evidence
        if self.error:
            message["error"] = self.error
        return message

    @classmethod
    def failed(cls, error: str) -> "ToolResult":
        return cls(ok=False, summary=error, error=error, evidence=error)


#: A backend tool: arguments in, result out.
Handler = Callable[[dict], Awaitable[ToolResult]]


@dataclass
class Tool:
    """One versioned capability."""

    name: str
    version: int
    summary: str
    #: Capability tags the router can ask for: "files", "web", "memory"…
    capabilities: tuple[str, ...]
    #: JSON-schema-ish, kept minimal because it goes into a prompt.
    parameters: dict[str, str] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    runs: str = BACKEND
    risk: str = SAFE
    handler: Handler | None = None

    @property
    def id(self) -> str:
        return f"{self.name}@v{self.version}"

    def describe_for_model(self) -> str:
        """One line. Every extra word here is paid for on every request."""
        args = ", ".join(
            f"{key}{'' if key in self.required else '?'}: {kind}"
            for key, kind in self.parameters.items()
        )
        return f"{self.name}({args}) — {self.summary}"

    def validate(self, arguments: dict) -> str | None:
        """Return a complaint, or `None` if the arguments are usable.

        Checked here rather than in the tool so that a hallucinated argument
        name is caught before anything runs, and the model gets told what it
        got wrong rather than a traceback.
        """
        missing = [key for key in self.required if not str(arguments.get(key, "")).strip()]
        if missing:
            return f"{self.name} needs {', '.join(missing)}"
        unknown = [key for key in arguments if key not in self.parameters]
        if unknown:
            return f"{self.name} has no argument {', '.join(unknown)}"
        return None


class ToolRegistry:
    """Everything available, and the means to find the few that matter."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        """Register a tool. A newer version supersedes an older one by name."""
        existing = self._tools.get(tool.name)
        if existing and existing.version > tool.version:
            return existing
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        # Accept "name@v2" as well as "name": models copy the id back at us.
        return self._tools.get(name.split("@")[0])

    def all(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def capabilities(self) -> list[str]:
        """Every capability tag, for the router to choose from."""
        tags = {tag for tool in self._tools.values() for tag in tool.capabilities}
        return sorted(tags)

    def discover(self, capabilities: list[str], limit: int = 8) -> list[Tool]:
        """The tools worth showing the model for this request.

        Ordered by how many of the requested capabilities each covers, so a
        tool that matches two beats one that matches one. Capped, because an
        unbounded list defeats the point.
        """
        wanted = {c.strip().lower() for c in capabilities if c.strip()}
        if not wanted:
            return []
        scored = []
        for tool in self._tools.values():
            overlap = len(wanted & {c.lower() for c in tool.capabilities})
            if overlap:
                scored.append((overlap, tool.name, tool))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [tool for _, _, tool in scored[:limit]]

    def describe(self, tools: list[Tool]) -> str:
        """The block that goes into the worker's prompt."""
        return "\n".join(f"- {tool.describe_for_model()}" for tool in tools)


@dataclass
class ToolCall:
    """A model's request to use a tool."""

    id: str
    name: str
    arguments: dict

    @classmethod
    def from_json(cls, value: dict, call_id: str) -> "ToolCall | None":
        name = str(value.get("tool", "")).strip()
        if not name:
            return None
        arguments = value.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        return cls(id=call_id, name=name, arguments=arguments)


def now_ms() -> int:
    return int(time.time() * 1000)
