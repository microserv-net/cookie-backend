"""Tools that run on the user's machine.

The backend never touches the user's filesystem, shell, applications or
screen. It asks. That is a security boundary, not a layering preference: a
model that has been talked into something should be able to reach at most a
request for `filesystem.delete`, which the frontend can refuse, confirm or
classify — because the frontend is the machine being acted upon and the only
one that knows whether the user is standing there.

The mechanism is deliberately small. A tool call goes out on the existing
reply stream:

    {"type":"tool.request","id":"call-7","tool":"filesystem.search",
     "arguments":{"pattern":"*.rs","root":"~/projects"},"risk":"safe"}

and the frontend posts the answer back:

    POST /v1/tool-result
    {"id":"call-7","ok":true,"summary":"4 matches","evidence":"...","data":[...]}

Correlation is by `id`, and every call has a deadline — a frontend that has
gone away must not leave a turn hanging forever. Timing out is reported to
the model as a failed tool call, which it can then plan around, rather than
as an exception nobody sees.

Declared here rather than discovered from the frontend because the model needs
the schemas before the first call, and a capability list that changes under
the orchestrator is harder to reason about than one that is versioned and
checked in. The frontend advertises which of these it actually implements
when it connects; anything it does not implement is simply never offered.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from . import CRITICAL, DANGEROUS, FRONTEND, NORMAL, SAFE, Tool, ToolRegistry, ToolResult

#: How long to wait for the frontend to answer a tool call.
DEFAULT_TIMEOUT_SECONDS = 120.0


#: The catalogue. Risk levels are advisory to the backend and authoritative to
#: the frontend, which owns the confirmation policy.
FRONTEND_TOOLS: tuple[Tool, ...] = (
    Tool("filesystem.search", 1, "find files by name or pattern",
         ("files", "projects"), {"pattern": "string", "root": "string"},
         ("pattern",), FRONTEND, SAFE),
    Tool("filesystem.read", 1, "read a file, or a range of its lines",
         ("files", "code"), {"path": "string", "from_line": "integer", "to_line": "integer"},
         ("path",), FRONTEND, SAFE),
    Tool("filesystem.write", 1, "write or replace a file's contents",
         ("files", "code"), {"path": "string", "content": "string"},
         ("path", "content"), FRONTEND, NORMAL),
    Tool("filesystem.delete", 1, "delete a file or directory",
         ("files",), {"path": "string"}, ("path",), FRONTEND, DANGEROUS),
    Tool("shell.which", 1, "check whether a command exists and what it is",
         ("shell", "system"), {"command": "string"}, ("command",), FRONTEND, SAFE),
    Tool("shell.run", 1, "run a command and return its output and exit code",
         ("shell", "code", "system"),
         {"command": "string", "arguments": "list of strings", "cwd": "string",
          "timeout_seconds": "integer"},
         ("command",), FRONTEND, NORMAL),
    Tool("app.open", 1, "open an application",
         ("applications",), {"name": "string"}, ("name",), FRONTEND, SAFE),
    Tool("app.close", 1, "close an application or window",
         ("applications",), {"name": "string"}, ("name",), FRONTEND, NORMAL),
    Tool("browser.open", 1, "open a URL in the default browser",
         ("web", "applications"), {"url": "string"}, ("url",), FRONTEND, SAFE),
    Tool("vscode.workspace", 1, "the open workspace, file, selection and diagnostics",
         ("code", "projects"), {}, (), FRONTEND, SAFE),
    Tool("vscode.open", 1, "open a file in the editor, optionally at a line",
         ("code",), {"path": "string", "line": "integer"}, ("path",), FRONTEND, SAFE),
    Tool("git.status", 1, "branch, staged and unstaged changes",
         ("git", "code"), {"repository": "string"}, (), FRONTEND, SAFE),
    Tool("git.diff", 1, "the current diff, optionally for one path",
         ("git", "code"), {"repository": "string", "path": "string"}, (), FRONTEND, SAFE),
    Tool("git.commit", 1, "stage and commit with a message",
         ("git", "code"), {"repository": "string", "message": "string"},
         ("message",), FRONTEND, NORMAL),
    Tool("git.push", 1, "push the current branch",
         ("git", "code"), {"repository": "string", "force": "boolean"}, (), FRONTEND, CRITICAL),
    Tool("screen.capture", 1, "a screenshot, described or returned",
         ("screen",), {"display": "integer"}, (), FRONTEND, NORMAL),
)


@dataclass
class PendingCall:
    id: str
    future: asyncio.Future = field(repr=False)


class FrontendBridge:
    """Dispatches tool calls to the frontend and waits for the answers.

    One instance per turn. Calls are correlated by id; results arrive out of
    band on `/v1/tool-result` and are handed to whoever is waiting.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self._pending: dict[str, asyncio.Future] = {}
        #: Which frontend tools this particular frontend says it implements.
        #: Empty means "we have not been told", which is treated as all of
        #: them — an older frontend that does not advertise still works.
        self.supported: set[str] = set()

    def advertise(self, names: list[str] | None) -> None:
        self.supported = {str(n) for n in names} if names else set()

    def offers(self, tool: Tool) -> bool:
        if tool.runs != FRONTEND:
            return True
        return not self.supported or tool.name in self.supported

    def deliver(self, call_id: str, result: ToolResult) -> bool:
        """Called by the HTTP endpoint when the frontend answers."""
        future = self._pending.pop(call_id, None)
        if future is None or future.done():
            return False
        future.set_result(result)
        return True

    async def call(self, call_id: str, emit, tool: Tool, arguments: dict) -> ToolResult:
        """Send the request and wait for the frontend, or time out."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[call_id] = future
        await emit(
            {
                "type": "tool.request",
                "id": call_id,
                "tool": tool.name,
                "version": tool.version,
                "arguments": arguments,
                "risk": tool.risk,
            }
        )
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            self._pending.pop(call_id, None)
            # Reported as a failed call rather than raised: the model can plan
            # around "that did not answer", but not around an exception.
            return ToolResult.failed(
                f"{tool.name} did not answer within {self.timeout:.0f} seconds"
            )
        except asyncio.CancelledError:
            self._pending.pop(call_id, None)
            raise


def register_frontend(registry: ToolRegistry) -> ToolRegistry:
    for tool in FRONTEND_TOOLS:
        registry.register(tool)
    return registry
