"""The tool layer: contracts, discovery, backend tools, frontend dispatch."""

import asyncio
import json
import tempfile
from pathlib import Path

from cookie_backend.config import Config, ModelRole
from cookie_backend.orchestrator import Orchestrator
from cookie_backend.tasks import TaskManager
from cookie_backend.tools import SAFE, Tool, ToolRegistry, ToolResult
from cookie_backend.tools.builtin import Memory, register_builtin
from cookie_backend.tools.frontend import FrontendBridge, register_frontend

from test_orchestrator import ScriptedProvider, collector


def registry() -> ToolRegistry:
    memory = Memory(Path(tempfile.mkdtemp()) / "memory.json")
    return register_frontend(register_builtin(ToolRegistry(), memory))


# --- contracts and discovery ------------------------------------------------


def test_discovery_returns_only_what_was_asked_for():
    r = registry()
    found = {t.name for t in r.discover(["memory"])}
    assert found == {"memory.remember", "memory.recall", "memory.forget"}
    assert "shell.run" not in found


def test_discovery_prefers_tools_covering_more_of_the_request():
    r = registry()
    found = [t.name for t in r.discover(["git", "code"], limit=3)]
    # git.* tools carry both tags; filesystem.read carries only "code".
    assert all(name.startswith("git.") for name in found), found


def test_nothing_is_offered_when_nothing_is_asked_for():
    assert registry().discover([]) == []


def test_a_newer_version_supersedes_an_older_one():
    r = ToolRegistry()
    r.register(Tool("a.b", 1, "old", ("x",)))
    r.register(Tool("a.b", 2, "new", ("x",)))
    assert r.get("a.b").summary == "new"
    # And an older one does not clobber a newer one.
    r.register(Tool("a.b", 1, "older still", ("x",)))
    assert r.get("a.b").version == 2


def test_tools_are_addressable_by_id_as_well_as_name():
    r = registry()
    assert r.get("web.fetch@v1") is r.get("web.fetch")


def test_arguments_are_checked_before_anything_runs():
    tool = registry().get("filesystem.write")
    assert tool.validate({"path": "/tmp/x"}) == "filesystem.write needs content"
    assert "no argument" in tool.validate({"path": "/tmp/x", "content": "y", "mode": "0644"})
    assert tool.validate({"path": "/tmp/x", "content": "y"}) is None


def test_descriptions_are_one_line_and_mark_optional_arguments():
    line = registry().get("filesystem.read").describe_for_model()
    assert "\n" not in line
    assert "path: string" in line and "from_line?: integer" in line


def test_risk_is_declared_so_the_frontend_can_decide():
    r = registry()
    assert r.get("filesystem.read").risk == SAFE
    assert r.get("filesystem.delete").risk == "dangerous"
    assert r.get("git.push").risk == "critical"


# --- memory ------------------------------------------------------------------


def test_memory_persists_and_records_when_it_last_mattered():
    path = Path(tempfile.mkdtemp()) / "memory.json"
    memory = Memory(path)
    memory.remember("github token location", "in the keychain", ["setup"])
    assert Memory(path).recall("keychain")[0]["value"] == "in the keychain"

    entry = Memory(path).recall("github")[0]
    # Referencing a memory updates when it last mattered, which is what makes
    # "forget what I haven't asked about" answerable.
    assert entry["last_referenced_at"] >= entry["created_at"]


def test_stale_returns_candidates_rather_than_deleting_them():
    memory = Memory(Path(tempfile.mkdtemp()) / "memory.json")
    memory.remember("old thing", "value")
    assert memory.stale(older_than_days=60) == []
    assert len(memory.stale(older_than_days=-1)) == 1
    # Nothing was deleted by asking.
    assert memory.count() == 1


async def test_memory_tools_round_trip():
    memory = Memory(Path(tempfile.mkdtemp()) / "memory.json")
    r = register_builtin(ToolRegistry(), memory)
    assert (await r.get("memory.remember").handler(
        {"key": "editor", "value": "vscode"})).ok
    recalled = await r.get("memory.recall").handler({"query": "editor"})
    assert "vscode" in recalled.summary
    assert (await r.get("memory.forget").handler({"key": "editor"})).ok
    assert not (await r.get("memory.recall").handler({"query": "editor"})).ok


async def test_a_tool_given_nonsense_fails_rather_than_raising():
    r = registry()
    result = await r.get("web.fetch").handler({"url": "not-a-url"})
    assert not result.ok and "http" in result.summary


# --- frontend dispatch -------------------------------------------------------


async def test_a_frontend_tool_call_goes_out_and_the_answer_comes_back():
    bridge = FrontendBridge(timeout=5)
    r = registry()
    sent = []

    async def emit(message):
        sent.append(message)
        # The frontend answers.
        bridge.deliver(message["id"], ToolResult(ok=True, summary="4 matches",
                                                 evidence="found 4 files"))

    result = await bridge.call("call-1", emit, r.get("filesystem.search"),
                               {"pattern": "*.rs"})
    assert result.ok and result.summary == "4 matches"
    assert sent[0]["type"] == "tool.request"
    assert sent[0]["tool"] == "filesystem.search"
    assert sent[0]["risk"] == "safe"


async def test_a_frontend_that_never_answers_times_out_as_a_failed_call():
    bridge = FrontendBridge(timeout=0.2)

    async def emit(message):
        pass  # the frontend has gone away

    result = await bridge.call("call-2", emit, registry().get("shell.run"),
                               {"command": "ls"})
    assert not result.ok
    assert "did not answer" in result.summary


def test_a_frontend_only_offers_what_it_implements():
    bridge = FrontendBridge()
    r = registry()
    assert bridge.offers(r.get("vscode.workspace")), "unknown means all, for old frontends"
    bridge.advertise(["filesystem.search", "shell.run"])
    assert bridge.offers(r.get("filesystem.search"))
    assert not bridge.offers(r.get("vscode.workspace"))
    # Backend tools are unaffected by what the frontend can do.
    assert bridge.offers(r.get("web.fetch"))


def test_late_and_duplicate_results_are_ignored():
    bridge = FrontendBridge()
    assert not bridge.deliver("never-asked", ToolResult(ok=True, summary="?"))


# --- the orchestrator using them --------------------------------------------


def build_with_tools(script, bridge=None):
    config = Config(roles={
        "router": ModelRole(model="qwen3:1.7b"),
        "worker": ModelRole(model="qwen3:4b"),
        "architect": ModelRole(model="qwen3:8b"),
    })
    tasks = TaskManager()
    task = tasks.create("working", "heavy")
    provider = ScriptedProvider(script)
    orch = Orchestrator(config, provider, tasks, registry(), bridge or FrontendBridge(timeout=5))
    return orch, task, provider


async def test_only_the_relevant_tool_schemas_reach_the_model():
    """Token efficiency is a design property here, so it is asserted."""
    orch, task, provider = build_with_tools({
        "router": ['{"kind": "task", "weight": "heavy", "capabilities": ["memory"]}'],
        "architect": [json.dumps({"say": "Right.", "steps": [
            {"what": "remember it", "done_when": "it is stored"}]})],
        "worker": ['{"result": "stored", "ok": true, "evidence": "saved"}'],
        "validator": ['{"passed": true, "reason": "stored"}'],
        "summary": ["Noted."],
    })
    _, emit = collector()
    await orch.run("remember that my editor is vscode", task, emit)

    worker_prompts = [prompt for who, prompt in provider.calls if who == "worker"]
    # The worker was shown memory tools and not the twenty others.
    system_seen = orch.registry.describe(orch.tools_for(["memory"]))
    assert "memory.remember" in system_seen
    assert "shell.run" not in system_seen
    assert worker_prompts, "the worker was never called"


async def test_a_tool_result_becomes_the_evidence_the_validator_judges():
    """The point of tools: evidence from the world, not from a model."""
    bridge = FrontendBridge(timeout=5)
    orch, task, _ = build_with_tools({
        "router": ['{"kind": "task", "weight": "heavy", "capabilities": ["code", "shell"]}'],
        "architect": [json.dumps({"say": "Let me check.", "steps": [
            {"what": "run the test suite", "done_when": "the suite passes"}]})],
        "worker": [
            '{"tool": "shell.run", "arguments": {"command": "cargo test"}}',
            '{"result": "ran the suite", "ok": true, "evidence": "exit code 0"}',
        ],
        "validator": ['{"passed": true, "reason": "exit code 0"}'],
        "summary": ["All the tests pass."],
    }, bridge)

    async def emit(message):
        if message.get("type") == "tool.request":
            bridge.deliver(message["id"], ToolResult(
                ok=True, summary="exit code 0, 42 passed",
                evidence="exit code 0; 42 passed; 0 failed"))

    outcome = await orch.run("run the tests", task, emit)
    assert outcome.succeeded
    evidence = outcome.attempts[0].evidence
    assert "42 passed" in evidence, evidence
    assert evidence.startswith("shell.run:")


async def test_a_failing_tool_is_reported_to_the_model_not_hidden():
    bridge = FrontendBridge(timeout=5)
    orch, task, _ = build_with_tools({
        "router": ['{"kind": "task", "weight": "heavy", "capabilities": ["shell"]}'],
        "architect": [json.dumps({"say": "One moment.", "steps": [
            {"what": "run the build", "done_when": "it compiles"}]})],
        "worker": [
            '{"tool": "shell.run", "arguments": {"command": "cargo build"}}',
            '{"result": "the build failed", "ok": false, "evidence": "two type errors"}',
            '{"result": "read the error", "ok": true, "evidence": "E0308 on line 12"}',
        ],
        "validator": [
            '{"passed": false, "reason": "it did not compile"}',
            '{"passed": true, "reason": "the cause is identified"}',
        ],
        "replan": [json.dumps({"say": "", "steps": [
            {"what": "read the first error", "done_when": "the cause is known"}]})],
        "summary": ["The build is failing on a type error."],
    }, bridge)

    async def emit(message):
        if message.get("type") == "tool.request":
            bridge.deliver(message["id"], ToolResult(
                ok=False, summary="exit code 101",
                evidence="error[E0308]: mismatched types"))

    outcome = await orch.run("build it", task, emit)
    assert not outcome.attempts[0].passed
    assert "E0308" in outcome.attempts[0].evidence


async def test_a_hallucinated_tool_is_refused_and_explained():
    orch, task, _ = build_with_tools({
        "router": ['{"kind": "task", "weight": "heavy", "capabilities": ["files"]}'],
        "architect": [json.dumps({"say": "", "steps": [
            {"what": "find it", "done_when": "found"}]})],
        "worker": [
            '{"tool": "filesystem.teleport", "arguments": {"path": "/tmp"}}',
            '{"result": "could not", "ok": false, "evidence": "no such tool"}',
            '{"result": "found it", "ok": true, "evidence": "~/.config/app.toml"}',
        ],
        "validator": [
            '{"passed": false, "reason": "nothing was found"}',
            '{"passed": true, "reason": "the file was found"}',
        ],
        "replan": [json.dumps({"say": "", "steps": [
            {"what": "search instead", "done_when": "found"}]})],
        "summary": ["I could not find that."],
    })
    _, emit = collector()
    outcome = await orch.run("find my config", task, emit)
    assert "no tool called filesystem.teleport" in outcome.attempts[0].evidence
