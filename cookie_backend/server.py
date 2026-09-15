"""The HTTP surface the frontend talks to.

Implements `docs/protocol.md`. Everything streams: a turn is answered with
newline-delimited JSON as the model produces it, so Cookie starts speaking
the first sentence while the rest is still being generated.

Routing, planning and validation live in `orchestrator.py`; this module is
the transport, the authentication and the task lifecycle around them.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import PROTOCOL, __version__
from .auth import DeviceStore, PairingError
from .config import Config
from .ollama import ModelError, OllamaProvider
from .orchestrator import Orchestrator
from .tasks import Cancelled, TaskManager
from .tools import ToolRegistry, ToolResult
from .tools.builtin import Memory, register_builtin
from .tools.frontend import FrontendBridge, register_frontend

#: Kept short on purpose. The reply is going to be *spoken*, and a model that
#: writes an essay produces an assistant nobody lets finish a sentence.
SPOKEN_SYSTEM_PROMPT = (
    "You are Cookie, a voice assistant. Your replies are read aloud, so: "
    "speak in short, natural sentences; never use markdown, bullet points, "
    "code blocks or emoji; do not narrate what you are about to do. "
    "If you do not know something, say so briefly."
)

#: A turn longer than this is treated as real work rather than conversation.
HEAVY_WORD_COUNT = 12


def create_app(
    config: Config,
    *,
    provider: OllamaProvider | None = None,
    devices: DeviceStore | None = None,
) -> FastAPI:
    """Build the application.

    Dependencies are injected so the test-suite can drive the whole protocol
    without Ollama, a model, or a network — the same rule the frontend keeps.
    """
    app = FastAPI(title="cookie-backend", version=__version__)
    app.state.config = config
    app.state.provider = provider or OllamaProvider(config)
    app.state.devices = devices or DeviceStore(config.data_dir / "devices.json")
    app.state.tasks = TaskManager(max_concurrent_heavy=config.max_concurrent_heavy)
    app.state.memory = Memory(config.data_dir / "memory.json")
    app.state.registry = register_frontend(
        register_builtin(ToolRegistry(), app.state.memory)
    )
    #: One bridge per process rather than per turn: tool results arrive on a
    #: separate request, and correlating them by call id is simpler than
    #: routing them to the right turn's bridge.
    app.state.bridge = FrontendBridge()

    base = config.base_path.rstrip("/")

    # --- authentication --------------------------------------------------

    async def require_device(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """Every endpoint but health and pairing needs a paired device.

        When nothing has been paired yet the backend is open, because a
        locked-out machine with no way in is worse than a machine on your own
        network that has not been paired. The moment the first device pairs,
        this closes.
        """
        store: DeviceStore = request.app.state.devices
        if store.is_empty():
            return None
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        device = store.authenticate(token)
        if device is None:
            raise HTTPException(
                status_code=401,
                detail="not paired. Run `cookie-backend pair` on the backend machine.",
            )
        return device

    # --- status ----------------------------------------------------------

    @app.get(f"{base}/v1/health")
    async def health() -> JSONResponse:
        provider: OllamaProvider = app.state.provider
        ollama_up = await provider.available()
        store: DeviceStore = app.state.devices
        return JSONResponse(
            {
                "status": "ok" if ollama_up else "degraded",
                "backend": "cookie-backend",
                "version": __version__,
                "protocol": PROTOCOL,
                "ollama": "up" if ollama_up else "unreachable",
                "paired_devices": len(store.devices()),
                "active_tasks": len(app.state.tasks.active()),
            }
        )

    @app.get(f"{base}/v1/models")
    async def models(_=Depends(require_device)) -> JSONResponse:
        """What is configured, installed and resident.

        Residency is the thing worth watching on a small machine, so it is
        reported rather than left to `ollama ps` on a different terminal.
        """
        provider: OllamaProvider = app.state.provider
        try:
            installed = await provider.installed_models()
            loaded = await provider.loaded_models()
        except ModelError as e:
            return JSONResponse({"error": str(e)}, status_code=503)
        return JSONResponse(
            {
                "roles": {
                    name: {"model": role.model, "keep_alive": role.keep_alive,
                           "installed": role.model in installed}
                    for name, role in config.roles.items()
                },
                "loaded": [
                    {"model": m.name, "size_gb": round(m.size_gb, 2),
                     "expires_at": m.expires_at}
                    for m in loaded
                ],
                "resident_gb": round(sum(m.size_gb for m in loaded), 2),
                "budget_gb": config.model_memory_gb,
            }
        )

    # --- pairing ---------------------------------------------------------

    @app.post(f"{base}/v1/pair")
    async def pair(request: Request) -> JSONResponse:
        body = await _json_body(request)
        store: DeviceStore = app.state.devices
        try:
            token = store.complete_pairing(
                str(body.get("code", "")),
                str(body.get("device_name", "frontend")),
            )
        except PairingError as e:
            # 403 rather than 401: the request was understood and refused.
            raise HTTPException(status_code=403, detail=str(e)) from e
        return JSONResponse({"token": token, "protocol": PROTOCOL})

    # --- work ------------------------------------------------------------

    @app.post(f"{base}/v1/cancel")
    async def cancel(request: Request, _=Depends(require_device)) -> JSONResponse:
        body = await _json_body(request)
        task_id = body.get("task_id")
        cancelled = await app.state.tasks.cancel([task_id] if task_id else None)
        return JSONResponse({"cancelled": cancelled})

    @app.get(f"{base}/v1/tools")
    async def tools(_=Depends(require_device)) -> JSONResponse:
        """What this backend can do, and where each tool runs.

        The frontend reads this to know which local tools it is expected to
        implement, and clients can read it to see what a request might touch.
        """
        registry: ToolRegistry = app.state.registry
        return JSONResponse(
            {
                "capabilities": registry.capabilities(),
                "tools": [
                    {"name": t.name, "version": t.version, "summary": t.summary,
                     "capabilities": list(t.capabilities), "runs": t.runs,
                     "risk": t.risk, "parameters": t.parameters,
                     "required": list(t.required)}
                    for t in registry.all()
                ],
            }
        )

    @app.post(f"{base}/v1/tool-result")
    async def tool_result(request: Request, _=Depends(require_device)) -> JSONResponse:
        """The frontend answering a `tool.request` it received on the stream.

        Correlated by `id`. An unknown id is not an error worth failing on —
        it means the turn moved on, usually because the call timed out or was
        cancelled, and the frontend should not be made to care.
        """
        body = await _json_body(request)
        call_id = str(body.get("id", ""))
        if not call_id:
            raise HTTPException(status_code=400, detail="a tool result needs an id")
        result = ToolResult(
            ok=bool(body.get("ok", False)),
            summary=str(body.get("summary", "")),
            data=body.get("data"),
            evidence=str(body.get("evidence", "")),
            error=body.get("error"),
        )
        accepted = app.state.bridge.deliver(call_id, result)
        return JSONResponse({"accepted": accepted})

    @app.get(f"{base}/v1/tasks")
    async def tasks(_=Depends(require_device)) -> JSONResponse:
        return JSONResponse(
            {
                "active": [
                    {"id": t.id, "title": t.title, "state": t.state,
                     "weight": t.weight, "detail": t.detail,
                     "age_seconds": round(t.age_seconds)}
                    for t in app.state.tasks.active()
                ]
            }
        )

    @app.post(f"{base}/v1/chat")
    async def chat(request: Request, _=Depends(require_device)) -> StreamingResponse:
        body = await _json_body(request)
        return StreamingResponse(
            _run_turn(app, body),
            media_type="application/x-ndjson",
            headers={"cache-control": "no-store"},
        )

    return app


async def _json_body(request: Request) -> dict:
    try:
        return await request.json()
    except Exception as e:  # noqa: BLE001 - any malformed body is one failure
        raise HTTPException(status_code=400, detail=f"expected a JSON body: {e}") from e


async def _run_turn(app: FastAPI, body: dict) -> AsyncIterator[bytes]:
    """One conversational turn, as newline-delimited JSON."""
    manager: TaskManager = app.state.tasks
    provider: OllamaProvider = app.state.provider
    config: Config = app.state.config

    text = str(body.get("text", "")).strip()
    scheduling = body.get("scheduling") or {}
    preempt_requested = bool(scheduling.get("preempt"))

    outbox: asyncio.Queue[dict | None] = asyncio.Queue()

    async def forward(message: dict) -> None:
        await outbox.put(message)

    manager.subscribe(forward)

    heavy = len(text.split()) > HEAVY_WORD_COUNT
    task = manager.create(
        title="thinking about that" if not heavy else "working on that",
        weight="heavy" if heavy else "light",
    )

    # The frontend tells us what it can do when it opens a turn; anything it
    # does not implement is never offered to the model.
    app.state.bridge.advertise(body.get("tools"))
    orchestrator = Orchestrator(
        config, provider, manager, app.state.registry, app.state.bridge
    )

    async def drive() -> None:
        stood_aside = []
        try:
            # Ask heavier work to stand aside *before* touching a model: the
            # gap between turns is itself a checkpoint, and the whole point is
            # not to have two large models resident at once.
            if manager.should_preempt(preempt_requested):
                stood_aside = await manager.stand_aside(
                    manager.heavy_running(), "paused while I deal with this"
                )

            await manager.set_state(task, "running")
            outcome = await orchestrator.run(
                text or "(the user said nothing audible)", task, outbox.put
            )
            if not task.cancelled:
                await manager.set_state(
                    task, "completed" if outcome.succeeded else "failed"
                )

        except Cancelled:
            await manager.set_state(task, "cancelled")
        except ModelError as e:
            # Model failures are spoken, not swallowed: the user asked a
            # question and deserves to know why there is no answer.
            await outbox.put({"type": "delta", "text": str(e)})
            await manager.set_state(task, "failed", str(e))
        except Exception as e:  # noqa: BLE001
            await outbox.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
            await manager.set_state(task, "failed", str(e))
        finally:
            if stood_aside:
                await manager.resume(stood_aside)
            await outbox.put(None)

    runner = asyncio.create_task(drive())
    try:
        while True:
            message = await outbox.get()
            if message is None:
                break
            yield (json.dumps(message) + "\n").encode()
        yield (json.dumps({"type": "end"}) + "\n").encode()
    finally:
        manager.unsubscribe(forward)
        if not runner.done():
            # The frontend hung up. That is an interruption of *speech*, not a
            # cancellation of work — but with nowhere to send the rest of this
            # reply, finishing it would only waste the machine.
            await manager.cancel([task.id])
            await asyncio.sleep(0)
        manager.forget(task.id)
