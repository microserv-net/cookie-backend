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

#: How many times the architect may replan before we stop and explain.
MAX_REPLANS = 2
#: Cap on steps per plan, whatever the architect thinks.
MAX_STEPS = 5


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


class Orchestrator:
    """One turn, start to finish."""

    def __init__(
        self,
        config: Config,
        provider: OllamaProvider,
        tasks: TaskManager,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tasks = tasks

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

    async def route(self, text: str, task: Task) -> tuple[str, str]:
        """(kind, weight). Falls back to conversation, which is always safe."""
        try:
            raw = await self._complete("router", prompts.ROUTER, text, task)
        except ModelError:
            return "chat", "light"
        parsed = find_json_object(raw) or {}
        kind = parsed.get("kind") if parsed.get("kind") in ("chat", "task") else "chat"
        weight = parsed.get("weight") if parsed.get("weight") in ("light", "heavy") else "light"
        return kind, weight

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

    async def execute(self, objective: str, step: Step, task: Task) -> Attempt:
        """Worker does the step; validator decides whether it counts."""
        raw = await self._complete(
            "worker",
            prompts.WORKER,
            f"Objective: {objective}\nStep: {step.what}\nSucceeds when: {step.done_when}",
            task,
        )
        parsed = find_json_object(raw) or {}
        result = spoken_text(str(parsed.get("result", ""))) or spoken_text(raw)[:400]
        evidence = str(parsed.get("evidence", "")).strip()
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
        kind, weight = await self.route(text, task)
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
                result = await self.execute(text, step, task)
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
