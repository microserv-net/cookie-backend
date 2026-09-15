"""Protocol-level tests, driven through the real app with a fake model.

No Ollama, no model, no network — the same rule the frontend keeps. If these
ever need a running model, something has leaked out of the provider
abstraction and that is the bug.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import httpx
import pytest

from cookie_backend.auth import DeviceStore
from cookie_backend.config import Config, ModelRole
from cookie_backend.ollama import ModelError
from cookie_backend.server import create_app


class FakeProvider:
    """A model that says what it is told to, slowly enough to interrupt.

    Role-aware, because the orchestrator asks different models different
    questions: the router is answered with a classification, the architect
    with a plan, and anything conversational with `fragments`.
    """

    def __init__(self, fragments=("Hello. ", "How can I help?"), delay=0.0, error=None,
                 route=None):
        self.fragments = fragments
        self.delay = delay
        self.error = error
        #: Override the routing decision; by default long turns become tasks.
        self.route = route
        self.calls = []
        self.evicted = []

    async def available(self):
        return True

    async def installed_models(self):
        return ["qwen3:4b"]

    async def loaded_models(self):
        return []

    async def make_room_for(self, role):
        return self.evicted

    async def chat(self, role, messages, options=None):
        self.calls.append((role.model, messages))
        if self.error:
            raise self.error
        system = messages[0]["content"]
        user = messages[-1]["content"]

        if "classify requests" in system:
            decision = self.route or (
                {"kind": "task", "weight": "heavy"}
                if len(user.split()) > 12
                else {"kind": "chat", "weight": "light"}
            )
            yield json.dumps(decision)
            return
        if "You plan work" in system or "previous plan did not work" in system:
            yield json.dumps({
                "say": "Right.",
                "steps": [{"what": "do the thing", "done_when": "it is done"}],
            })
            return
        if "carry out one step" in system:
            yield json.dumps({"result": "did it", "ok": True, "evidence": "it is done"})
            return
        if "You check whether" in system:
            yield json.dumps({"passed": True, "reason": "the evidence shows it"})
            return

        for fragment in self.fragments:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield fragment


def build(provider=None, paired=False, **overrides):
    tmp = Path(tempfile.mkdtemp())
    config = Config(
        roles={"router": ModelRole(model="qwen3:1.7b"),
               "worker": ModelRole(model="qwen3:4b"),
               "architect": ModelRole(model="qwen3:8b")},
        data_dir=tmp,
        **overrides,
    )
    devices = DeviceStore(tmp / "devices.json")
    token = None
    if paired:
        token = devices.complete_pairing(devices.begin_pairing(), "laptop")
    app = create_app(config, provider=provider or FakeProvider(), devices=devices)
    return app, token


def client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def turn(ac, text, token=None, **scheduling):
    headers = {"authorization": f"Bearer {token}"} if token else {}
    body = {
        "protocol": "cookie-interface/1",
        "session_id": "s", "utterance_id": "u",
        "text": text, "final": True,
        "interface": {"speech": True, "listening": True, "visual": "orb"},
        "scheduling": {"priority": "normal", "preempt": False, **scheduling},
        "active_tasks": [],
    }
    messages = []
    async with ac.stream("POST", "/api/v1/chat", json=body, headers=headers) as response:
        assert response.status_code == 200, await response.aread()
        async for line in response.aiter_lines():
            if line.strip():
                messages.append(json.loads(line))
    return messages


async def test_health_is_open_and_names_the_protocol():
    app, _ = build()
    async with client(app) as ac:
        body = (await ac.get("/api/v1/health")).json()
    assert body["protocol"] == "cookie-interface/1"
    assert body["status"] == "ok"


async def test_a_turn_streams_deltas_and_terminates():
    app, _ = build(FakeProvider(route={"kind": "chat", "weight": "light"}))
    async with client(app) as ac:
        messages = await turn(ac, "hello")
    kinds = [m["type"] for m in messages]
    assert "delta" in kinds
    assert kinds[-1] == "end"
    spoken = "".join(m["text"] for m in messages if m["type"] == "delta")
    assert spoken == "Hello. How can I help?"


async def test_a_turn_reports_its_task_lifecycle():
    app, _ = build()
    async with client(app) as ac:
        messages = await turn(ac, "hello")
    tasks = [m for m in messages if m["type"] == "task"]
    assert [t["state"] for t in tasks][-1] == "completed"
    assert all(t["id"] and t["title"] for t in tasks)


async def test_a_question_skips_the_architect_and_a_task_does_not():
    """The router earning its place: most turns never load the 8b model."""
    provider = FakeProvider()
    app, _ = build(provider)

    async with client(app) as ac:
        short = await turn(ac, "what time is it")
    models_used = [model for model, _ in provider.calls]
    assert "qwen3:1.7b" in models_used, "the router should see every turn"
    assert "qwen3:8b" not in models_used, "a question must not load the architect"
    assert [t for t in short if t["type"] == "task"][0]["weight"] == "light"

    provider.calls.clear()
    async with client(app) as ac:
        long = await turn(ac, " ".join(["please do the thing"] * 8))
    assert "qwen3:8b" in [model for model, _ in provider.calls]
    assert [t for t in long if t["type"] == "task"][0]["weight"] == "heavy"


async def test_model_failures_are_spoken_not_swallowed():
    app, _ = build(FakeProvider(error=ModelError("the model qwen3:4b is not installed")))
    async with client(app) as ac:
        messages = await turn(ac, "hello")
    spoken = " ".join(m["text"] for m in messages if m["type"] == "delta")
    assert "not installed" in spoken
    assert [m for m in messages if m["type"] == "task"][-1]["state"] == "failed"


async def test_cancelling_stops_a_turn_in_flight():
    app, _ = build(FakeProvider(fragments=["a "] * 60, delay=0.05,
                                route={"kind": "chat", "weight": "light"}))
    async with client(app) as ac:
        collected = []

        async def listen():
            async with ac.stream("POST", "/api/v1/chat", json={
                "text": "tell me a long story", "final": True,
                "scheduling": {"preempt": False},
            }) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        collected.append(json.loads(line))

        runner = asyncio.create_task(listen())
        await asyncio.sleep(0.25)
        cancelled = (await ac.post("/api/v1/cancel", json={})).json()["cancelled"]
        await asyncio.wait_for(runner, timeout=10)

    assert cancelled, "nothing was cancelled"
    # It stopped early rather than reading all sixty fragments.
    assert len([m for m in collected if m["type"] == "delta"]) < 60
    assert collected[-1]["type"] == "end"


async def test_pairing_closes_the_door_behind_it():
    app, _ = build()
    async with client(app) as ac:
        # Nothing paired yet: open, so you can always get in to pair.
        assert (await ac.post("/api/v1/tasks")).status_code in (405, 200)
        assert (await ac.get("/api/v1/tasks")).status_code == 200

        store = app.state.devices
        code = store.begin_pairing()
        paired = await ac.post("/api/v1/pair", json={"code": code, "device_name": "laptop"})
        token = paired.json()["token"]

        # Now it is shut.
        assert (await ac.get("/api/v1/tasks")).status_code == 401
        ok = await ac.get("/api/v1/tasks", headers={"authorization": f"Bearer {token}"})
        assert ok.status_code == 200


async def test_a_bad_pairing_code_is_refused():
    app, _ = build()
    async with client(app) as ac:
        response = await ac.post("/api/v1/pair", json={"code": "not-a-code", "device_name": "x"})
    assert response.status_code == 403


async def test_models_endpoint_reports_residency():
    app, token = build(paired=True)
    async with client(app) as ac:
        body = (await ac.get("/api/v1/models",
                             headers={"authorization": f"Bearer {token}"})).json()
    assert body["roles"]["worker"]["model"] == "qwen3:4b"
    assert body["roles"]["worker"]["installed"] is True
    assert body["roles"]["architect"]["installed"] is False
    assert body["budget_gb"] > 0


async def test_preemption_suspends_heavy_work_and_resumes_it():
    """The behaviour the whole scheduling design exists for."""
    provider = FakeProvider(fragments=["a "] * 30, delay=0.05)
    app, _ = build(provider)
    async with client(app) as ac:
        heavy_messages = []

        async def heavy():
            async with ac.stream("POST", "/api/v1/chat", json={
                "text": " ".join(["word"] * 40), "final": True,
                "scheduling": {"preempt": False},
            }) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        heavy_messages.append(json.loads(line))

        runner = asyncio.create_task(heavy())
        await asyncio.sleep(0.2)

        # A short aside, marked the way the frontend marks one.
        await turn(ac, "what time is it", preempt=True, priority="interactive")
        await asyncio.wait_for(runner, timeout=20)

    states = [m["state"] for m in heavy_messages if m["type"] == "task"]
    assert "suspended" in states, states
    # Suspended, then picked back up, then finished: nothing was lost.
    assert states.index("suspended") < states.index("running", states.index("suspended"))
    assert states[-1] == "completed"


async def test_the_tool_catalogue_says_where_each_tool_runs():
    app, token = build(paired=True)
    async with client(app) as ac:
        body = (await ac.get("/api/v1/tools",
                             headers={"authorization": f"Bearer {token}"})).json()
    by_name = {t["name"]: t for t in body["tools"]}
    assert by_name["shell.run"]["runs"] == "frontend"
    assert by_name["web.fetch"]["runs"] == "backend"
    assert by_name["git.push"]["risk"] == "critical"
    assert "files" in body["capabilities"]


async def test_tool_results_are_correlated_by_id():
    app, _ = build()
    async with client(app) as ac:
        # Nobody is waiting for this one, which is not an error: the turn has
        # moved on, and the frontend should not be made to care.
        response = await ac.post("/api/v1/tool-result",
                                 json={"id": "call-nobody-wants", "ok": True,
                                       "summary": "done"})
        assert response.status_code == 200
        assert response.json()["accepted"] is False

        response = await ac.post("/api/v1/tool-result", json={"ok": True})
        assert response.status_code == 400


async def test_a_frontend_only_gets_offered_what_it_says_it_implements():
    app, _ = build()
    async with client(app) as ac:
        await ac.post("/api/v1/chat", json={
            "text": "hello", "final": True,
            "tools": ["filesystem.search", "shell.run"],
            "scheduling": {"preempt": False},
        })
    bridge = app.state.bridge
    registry = app.state.registry
    assert bridge.offers(registry.get("shell.run"))
    assert not bridge.offers(registry.get("vscode.workspace"))
