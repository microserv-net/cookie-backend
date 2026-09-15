"""Router, architect, worker, and the loop that keeps them honest.

```
utterance ─▶ router (1.7b)  what kind of thing is this?
                 │
        ┌────────┴────────┐
        ▼                 ▼
   chat: worker      task: architect (8b) plans
   answers and            │
   that is all            ▼
                     worker (4b) executes a step
                          │
                          ▼
                     validator checks it against the step's own
                     success condition — not against how confident
                     the worker sounded
                          │
                    ┌─────┴─────┐
                 passed       failed
                    │             │
                next step    architect replans with the evidence
```

Three properties this is built around:

**The architect does not get to mark its own homework.** The worker executes
and the validator checks against a condition the architect had to state in
advance. An architect that believes its own output is how you get an assistant
that cheerfully reports success while the tests are still red.

**Replanning is bounded and remembers.** Failed approaches are fingerprinted,
so the architect cannot propose the same thing twice; after a small number of
attempts the assistant says what went wrong instead of trying forever.

**Every model call is a checkpoint.** On a machine that holds one large model,
the gaps between calls are where something more urgent gets to go first — see
`tasks.checkpoint`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from . import prompts
from .config import Config
from .ollama import ModelError, OllamaProvider
from .parsing import find_json_object, spoken_text
from .tasks import Task, TaskManager
from .tools import FRONTEND, ToolCall, ToolRegistry, ToolResult
from .tools.frontend import FrontendBridge

#: How many times the architect may replan before we stop and explain.
MAX_REPLANS = 2
#: Cap on steps per plan, whatever the architect thinks.
MAX_STEPS = 5
#: Tool calls allowed within one step, before we make the worker report.
#: Without this a model that likes a tool will happily use it forever.
MAX_TOOL_CALLS = 6


@dataclass
class Step:
    what: str
    done_when: str


@dataclass
class Plan:
    steps: list[Step]
    say: str = ""

    def fingerprint(self) -> str:
        """Identity of an *approach*, for detecting repetition.

        Deliberately the steps and not `say`: rewording the preamble is not a
        new idea, and letting it count as one is how a loop hides.
        """
        joined = "|".join(f"{s.what}→{s.done_when}" for s in self.steps)
        return hashlib.sha256(joined.lower().encode()).hexdigest()[:16]


@dataclass
class Attempt:
    """What was tried and what came of it. Fed back to the architect."""

    step: Step
    result: str
    evidence: str
    passed: bool


@dataclass
class Outcome:
    succeeded: bool
    attempts: list[Attempt] = field(default_factory=list)
    replans: int = 0
    gave_up_because: str | None = None


#: Anything the orchestrator wants said or shown. `kind` is a protocol message
#: type: "delta" for speech, "task" handled by the manager.
Emit = Callable[[dict], Awaitable[None]]


async def _discard(_message: dict) -> None:
    """Sink for tool calls made outside a streaming turn (tests, retries)."""


class Orchestrator:
    """One turn, start to finish."""

    def __init__(
        self,
        config: Config,
        provider: OllamaProvider,
        tasks: TaskManager,
        registry: ToolRegistry | None = None,
        bridge: FrontendBridge | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tasks = tasks
        self.registry = registry or ToolRegistry()
        self.bridge = bridge or FrontendBridge()

    # --- model access ----------------------------------------------------

    async def _complete(self, role_name: str, system: str, user: str, task: Task) -> str:
        """One non-streaming model call, with a checkpoint in front of it."""
        await self.tasks.checkpoint(task)
        role = self.config.role(role_name)
        evicted = await self.provider.make_room_for(role)
        if evicted:
            await self.tasks.set_state(
                task, task.state, f"unloading {', '.join(evicted)} to make room"
            )
        chunks: list[str] = []
        async for fragment in self.provider.chat(
            role,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
        ):
            chunks.append(fragment)
        return "".join(chunks)

    async def _speak(self, role_name: str, system: str, user: str, task: Task,
                     emit: Emit) -> str:
        """A model call whose output is spoken as it arrives."""
        await self.tasks.checkpoint(task)
        role = self.config.role(role_name)
        buffer: list[str] = []
        async for fragment in self.provider.chat(
            role,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
        ):
            if task.cancelled:
                break
            buffer.append(fragment)
            # Reasoning must never be spoken, so anything containing an open
            # `<think>` is held back until the whole reply is in.
            if "<think>" not in "".join(buffer).lower():
                await emit({"type": "delta", "text": fragment})
        whole = "".join(buffer)
        if "<think>" in whole.lower():
            cleaned = spoken_text(whole)
            if cleaned:
                await emit({"type": "delta", "text": cleaned})
            return cleaned
        return whole

    # --- the pipeline ----------------------------------------------------

    async def route(self, text: str, task: Task) -> tuple[str, str, list[str]]:
        """(kind, weight, capabilities).

        Falls back to conversation, which is always safe: an assistant that
        answers when it should have acted is a disappointment; one that acts
        when it should have answered is a hazard.
        """
        available = ", ".join(self.registry.capabilities()) or "none"
        system = f"{prompts.ROUTER} {available}"
        try:
            raw = await self._complete("router", system, text, task)
        except ModelError:
            return "chat", "light", []
        parsed = find_json_object(raw) or {}
        kind = parsed.get("kind") if parsed.get("kind") in ("chat", "task") else "chat"
        weight = parsed.get("weight") if parsed.get("weight") in ("light", "heavy") else "light"
        wanted = parsed.get("capabilities")
        capabilities = [str(c) for c in wanted] if isinstance(wanted, list) else []
        return kind, weight, capabilities

    def tools_for(self, capabilities: list[str]) -> list:
        """The tools worth showing the model, filtered by what is reachable."""
        return [
            tool
            for tool in self.registry.discover(capabilities)
            if self.bridge.offers(tool)
        ]

    async def _use_tool(self, call: ToolCall, task: Task, emit: Emit) -> ToolResult:
        """Run one tool call, wherever it lives."""
        tool = self.registry.get(call.name)
        if tool is None:
            return ToolResult.failed(f"there is no tool called {call.name}")
        complaint = tool.validate(call.arguments)
        if complaint:
            return ToolResult.failed(complaint)

        await self.tasks.set_state(task, task.state, f"using {tool.name}")
        # A tool call is a checkpoint: we are about to wait on something
        # anyway, so it is free to let a more urgent turn go first.
        await self.tasks.checkpoint(task)

        if tool.runs == FRONTEND:
            return await self.bridge.call(call.id, emit, tool, call.arguments)
        if tool.handler is None:
            return ToolResult.failed(f"{tool.name} is declared but not implemented here")
        try:
            return await tool.handler(call.arguments)
        except Exception as e:  # noqa: BLE001 - a broken tool is a failed step
            return ToolResult.failed(f"{tool.name} failed: {type(e).__name__}: {e}")

    async def plan(self, objective: str, task: Task, history: list[Attempt],
                   avoid: set[str]) -> Plan | None:
        """Ask the architect for a plan, or a *different* plan."""
        if history:
            evidence = "\n".join(
                f"- tried: {a.step.what}\n  result: {a.result}\n  evidence: {a.evidence}"
                f"\n  verdict: {'passed' if a.passed else 'FAILED'}"
                for a in history
            )
            system = prompts.ARCHITECT_REPLAN
            user = f"Objective: {objective}\n\nWhat has been tried:\n{evidence}"
        else:
            system = prompts.ARCHITECT
            user = f"Objective: {objective}"

        raw = await self._complete("architect", system, user, task)
        parsed = find_json_object(raw)
        if not parsed:
            return None
        steps = [
            Step(what=str(s.get("what", "")).strip(), done_when=str(s.get("done_when", "")).strip())
            for s in (parsed.get("steps") or [])
            if str(s.get("what", "")).strip()
        ][:MAX_STEPS]
        if not steps:
            return None
        plan = Plan(steps=steps, say=spoken_text(str(parsed.get("say", ""))))
        if plan.fingerprint() in avoid:
            # The architect has proposed something already known not to work.
            # Treating this as "no plan" sends us to the give-up path, which
            # explains itself, rather than round the loop again.
            return None
        return plan

    async def execute(
        self,
        objective: str,
        step: Step,
        task: Task,
        tools: list | None = None,
        emit: Emit | None = None,
    ) -> Attempt:
        """Worker does the step, using tools; validator decides whether it counts."""
        tools = tools or []
        system = prompts.WORKER
        if tools:
            system += "\n\nTools available:\n" + self.registry.describe(tools)

        transcript = (
            f"Objective: {objective}\nStep: {step.what}\nSucceeds when: {step.done_when}"
        )
        tool_evidence: list[str] = []
        parsed: dict = {}
        raw = ""

        for call_number in range(MAX_TOOL_CALLS + 1):
            raw = await self._complete("worker", system, transcript, task)
            parsed = find_json_object(raw) or {}
            call = ToolCall.from_json(parsed, f"{task.id}-call-{call_number}") if tools else None
            if call is None:
                break
            if call_number == MAX_TOOL_CALLS:
                tool_evidence.append("stopped after too many tool calls")
                break
            result = await self._use_tool(call, task, emit or _discard)
            tool_evidence.append(f"{call.name}: {result.evidence or result.summary}")
            # The model sees exactly what happened, including failures, and
            # decides what to do about it.
            transcript += (
                f"\n\nYou used {call.name} with {call.arguments}.\n"
                f"Result: {'ok' if result.ok else 'FAILED'} — {result.summary}"
            )
            if result.data is not None:
                transcript += f"\nData: {str(result.data)[:1500]}"

        result = spoken_text(str(parsed.get("result", ""))) or spoken_text(raw)[:400]
        evidence = str(parsed.get("evidence", "")).strip()
        if tool_evidence:
            # Tool output is *external* evidence, which is the entire reason
            # the validator is worth more than a second opinion.
            evidence = "; ".join(tool_evidence) + (f"; {evidence}" if evidence else "")
        claimed = bool(parsed.get("ok", False))

        # The worker's own verdict is an input, not the answer.
        verdict = await self._complete(
            "worker",
            prompts.VALIDATOR,
            f"Step: {step.what}\nSucceeds when: {step.done_when}\n"
            f"Worker reported: {result}\nEvidence: {evidence or '(none given)'}",
            task,
        )
        checked = find_json_object(verdict) or {}
        passed = bool(checked.get("passed", False))
        reason = str(checked.get("reason", "")).strip()

        return Attempt(
            step=step,
            result=result,
            evidence=evidence or reason,
            # No evidence and no independent pass means it did not happen,
            # however confident the worker was.
            passed=passed and (claimed or bool(evidence)),
        )

    async def run(self, text: str, task: Task, emit: Emit) -> Outcome:
        """Handle one utterance. Speech and progress go out through `emit`."""
        kind, weight, capabilities = await self.route(text, task)
        tools = self.tools_for(capabilities)
        await self.tasks.set_state(
            task,
            "running",
            "answering" if kind == "chat" else "planning",
        )

        if kind == "chat":
            await self._speak("worker", prompts.SPOKEN, text, task, emit)
            return Outcome(succeeded=True)

        outcome = Outcome(succeeded=False)
        avoid: set[str] = set()

        for attempt_number in range(MAX_REPLANS + 1):
            plan = await self.plan(text, task, outcome.attempts, avoid)
            if plan is None:
                outcome.gave_up_because = (
                    "I could not come up with an approach I have not already tried."
                    if attempt_number
                    else "I could not work out how to do that."
                )
                break
            avoid.add(plan.fingerprint())
            outcome.replans = attempt_number

            if plan.say:
                await emit({"type": "delta", "text": plan.say + " "})

            failed: Attempt | None = None
            for index, step in enumerate(plan.steps, start=1):
                await self.tasks.set_state(
                    task, "running", f"step {index} of {len(plan.steps)}: {step.what}"
                )
                result = await self.execute(text, step, task, tools, emit)
                outcome.attempts.append(result)
                if not result.passed:
                    failed = result
                    break

            if failed is None:
                outcome.succeeded = True
                break

            await self.tasks.set_state(
                task, "running", f"that did not work: {failed.evidence or 'no evidence'}"
            )

        await self._summarise(text, outcome, task, emit)
        return outcome

    async def _summarise(self, objective: str, outcome: Outcome, task: Task,
                         emit: Emit) -> None:
        """Say what happened. The user watched none of it."""
        if outcome.gave_up_because and not outcome.attempts:
            await emit({"type": "delta", "text": outcome.gave_up_because})
            return
        transcript = "\n".join(
            f"- {a.step.what}: {a.result} ({'worked' if a.passed else 'did not work'})"
            for a in outcome.attempts
        )
        status = "It worked." if outcome.succeeded else "It did not work in the end."
        if outcome.gave_up_because:
            status += f" {outcome.gave_up_because}"
        try:
            await self._speak(
                "worker",
                prompts.SUMMARISE,
                f"Request: {objective}\n\nWhat happened:\n{transcript}\n\n{status}",
                task,
                emit,
            )
        except ModelError:
            # Losing the model at the last step must not lose the news.
            await emit({"type": "delta", "text": status})


async def stream_turn(
    text: str,
    task: Task,
    orchestrator: Orchestrator,
    emit: Emit,
) -> Outcome:
    """Convenience wrapper used by the server."""
    return await orchestrator.run(text, task, emit)
