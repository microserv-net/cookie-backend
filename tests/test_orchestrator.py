"""The orchestration, driven by a scripted model.

Every test here is about a decision rather than a mechanism: that the
architect cannot mark its own homework, that a failed approach is not tried
twice, that reasoning never reaches the user's ears.
"""

import json

import pytest

from cookie_backend.config import Config, ModelRole
from cookie_backend.ollama import ModelError
from cookie_backend.orchestrator import Orchestrator, Plan, Step
from cookie_backend.tasks import TaskManager


class ScriptedProvider:
    """Answers per role, in order, from a script.

    Keyed by role so a test can say "the router says task, the architect says
    this plan, the worker claims success, the validator disagrees" without
    caring about call ordering.
    """

    def __init__(self, script: dict[str, list[str]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, str]] = []

    async def make_room_for(self, role):
        return []

    async def chat(self, role, messages, options=None):
        system = messages[0]["content"]
        who = self._role_of(system)
        self.calls.append((who, messages[-1]["content"]))
        queue = self.script.get(who)
        if not queue:
            raise AssertionError(f"no scripted reply left for {who}")
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        yield reply

    @staticmethod
    def _role_of(system: str) -> str:
        if "classify requests" in system:
            return "router"
        if "previous plan did not work" in system:
            return "replan"
        if "You plan work" in system:
            return "architect"
        if "You check whether" in system:
            return "validator"
        if "Say what happened" in system:
            return "summary"
        if "carry out one step" in system:
            return "worker"
        return "chat"


def build(script):
    config = Config(roles={
        "router": ModelRole(model="qwen3:1.7b"),
        "worker": ModelRole(model="qwen3:4b"),
        "architect": ModelRole(model="qwen3:8b"),
    })
    tasks = TaskManager()
    task = tasks.create("working", "heavy")
    provider = ScriptedProvider(script)
    return Orchestrator(config, provider, tasks), task, provider


def collector():
    spoken = []

    async def emit(message):
        if message["type"] == "delta":
            spoken.append(message["text"])

    return spoken, emit


def plan(*steps):
    return json.dumps({
        "say": "Right, I'll take a look.",
        "steps": [{"what": w, "done_when": d} for w, d in steps],
    })


async def test_a_question_is_answered_without_planning():
    orch, task, provider = build({
        "router": ['{"kind": "chat", "weight": "light"}'],
        "chat": ["It is half past four."],
    })
    spoken, emit = collector()
    outcome = await orch.run("what time is it", task, emit)
    assert outcome.succeeded
    assert "".join(spoken) == "It is half past four."
    # The architect was never consulted: that is the point of the router.
    assert not any(who == "architect" for who, _ in provider.calls)


async def test_a_task_is_planned_executed_and_validated():
    orch, task, _ = build({
        "router": ['{"kind": "task", "weight": "heavy"}'],
        "architect": [plan(("run the tests", "the suite passes"))],
        "worker": ['{"result": "ran the suite", "ok": true, "evidence": "0 failures"}'],
        "validator": ['{"passed": true, "reason": "no failures reported"}'],
        "summary": ["All the tests pass now."],
    })
    spoken, emit = collector()
    outcome = await orch.run("fix the failing tests", task, emit)
    assert outcome.succeeded
    assert len(outcome.attempts) == 1
    assert "All the tests pass now." in "".join(spoken)


async def test_the_worker_cannot_mark_its_own_homework():
    """The decision this whole loop exists for."""
    orch, task, _ = build({
        "router": ['{"kind": "task", "weight": "heavy"}'],
        "architect": [plan(("fix the auth bug", "the tests pass"))],
        "replan": [plan(("read the failing test first", "the expectation is understood"))],
        "worker": [
            '{"result": "fixed it", "ok": true, "evidence": ""}',
            '{"result": "read the test", "ok": true, "evidence": "it expects a 401"}',
        ],
        "validator": [
            '{"passed": false, "reason": "three tests still fail"}',
            '{"passed": true, "reason": "the expectation is now clear"}',
        ],
        "summary": ["I had to look at the test first, but it is sorted."],
    })
    spoken, emit = collector()
    outcome = await orch.run("fix the auth bug", task, emit)

    # The worker claimed success and was overruled, which forced a replan.
    assert outcome.attempts[0].passed is False
    assert outcome.replans == 1
    assert outcome.succeeded


async def test_a_failed_approach_is_not_proposed_twice():
    same = plan(("try the same thing", "it works"))
    orch, task, _ = build({
        "router": ['{"kind": "task", "weight": "heavy"}'],
        "architect": [same],
        "replan": [same],  # the architect repeats itself
        "worker": ['{"result": "tried", "ok": false, "evidence": "no"}'],
        "validator": ['{"passed": false, "reason": "still broken"}'],
        "summary": ["I could not get that working."],
    })
    spoken, emit = collector()
    outcome = await orch.run("do the thing", task, emit)
    assert not outcome.succeeded
    assert outcome.gave_up_because is not None
    # Exactly one execution: the repeat was caught before doing it again.
    assert len(outcome.attempts) == 1


async def test_replanning_is_bounded():
    orch, task, _ = build({
        "router": ['{"kind": "task", "weight": "heavy"}'],
        "architect": [plan(("attempt one", "works"))],
        "replan": [plan(("attempt two", "works")), plan(("attempt three", "works"))],
        "worker": ['{"result": "tried", "ok": false, "evidence": "nope"}'] * 3,
        "validator": ['{"passed": false, "reason": "no"}'] * 3,
        "summary": ["I tried a few things and none of them worked."],
    })
    spoken, emit = collector()
    outcome = await orch.run("do the thing", task, emit)
    assert not outcome.succeeded
    assert len(outcome.attempts) == 3, "should stop after the bounded retries"


async def test_reasoning_never_reaches_the_user():
    orch, task, _ = build({
        "router": ['{"kind": "chat", "weight": "light"}'],
        "chat": ["<think>The user is asking about the weather, I should be brief</think>"
                 "It is raining."],
    })
    spoken, emit = collector()
    await orch.run("what is it doing outside", task, emit)
    said = "".join(spoken)
    assert "think" not in said.lower()
    assert said.strip() == "It is raining."


async def test_an_unparseable_router_falls_back_to_conversation():
    orch, task, _ = build({
        "router": ["I'm not sure what you mean by that."],
        "chat": ["Say again?"],
    })
    spoken, emit = collector()
    outcome = await orch.run("mumble", task, emit)
    assert outcome.succeeded
    assert "Say again?" in "".join(spoken)


async def test_a_dead_router_still_answers():
    orch, task, _ = build({
        "router": [ModelError("ollama is not running")],
        "chat": ["I can still hear you."],
    })
    spoken, emit = collector()
    outcome = await orch.run("hello", task, emit)
    assert outcome.succeeded
    assert "I can still hear you." in "".join(spoken)


async def test_an_architect_that_produces_nothing_says_so():
    orch, task, _ = build({
        "router": ['{"kind": "task", "weight": "heavy"}'],
        "architect": ["I would love to help but I have no idea."],
    })
    spoken, emit = collector()
    outcome = await orch.run("do something impossible", task, emit)
    assert not outcome.succeeded
    assert "could not work out how" in "".join(spoken)


def test_plans_are_fingerprinted_by_approach_not_wording():
    a = Plan(steps=[Step("Run the tests", "they pass")], say="Right.")
    b = Plan(steps=[Step("run the tests", "they pass")], say="Let me see.")
    c = Plan(steps=[Step("read the tests", "they pass")], say="Right.")
    assert a.fingerprint() == b.fingerprint(), "rewording the preamble is not a new idea"
    assert a.fingerprint() != c.fingerprint()
