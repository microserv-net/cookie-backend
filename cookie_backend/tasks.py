"""Tasks, and the part that makes one-model-at-a-time hardware bearable.

A task is one thing the user asked for. It runs until it finishes, fails, is
cancelled, or is asked to **stand aside** — which is the interesting case.

Standing aside is cooperative. Nothing preempts anything by force: a task
checks `await checkpoint()` at the points where it is already between things,
and if something more urgent has arrived it waits there until that is done.
Checkpoints are exactly the boundaries where stopping costs nothing:

  * a model call has just returned,
  * a model is about to be loaded or swapped,
  * a tool is running and we are waiting on it,
  * we are between steps of a plan.

Never mid-generation, and never by discarding work. The distinction between
this and cancellation is the whole point: being asked to wait must not lose
what you have done, and being cancelled must only ever happen because someone
asked for it.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable

Weight = str  # "light" | "normal" | "heavy"
State = str  # "queued" | "running" | "suspended" | "completed" | "failed" | "cancelled"


@dataclass
class Task:
    id: str
    title: str
    weight: Weight = "normal"
    state: State = "queued"
    detail: str | None = None
    created_at: float = field(default_factory=time.time)
    #: Set while this task is asked to stand aside.
    _stand_aside: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _cancelled: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def to_message(self) -> dict:
        """The protocol's task message. See docs/protocol.md."""
        message = {
            "type": "task",
            "id": self.id,
            "state": self.state,
            "title": self.title,
            "weight": self.weight,
        }
        if self.detail:
            message["detail"] = self.detail
        return message


class Cancelled(Exception):
    """Raised at a checkpoint when the task has been cancelled."""


class TaskManager:
    """Everything in flight, and who has to wait for whom."""

    def __init__(self, max_concurrent_heavy: int = 1) -> None:
        self.max_concurrent_heavy = max_concurrent_heavy
        self._tasks: dict[str, Task] = {}
        self._listeners: list[Callable[[dict], Awaitable[None]]] = []

    # --- lifecycle -------------------------------------------------------

    def create(self, title: str, weight: Weight = "normal") -> Task:
        task = Task(id=f"task-{uuid.uuid4().hex[:8]}", title=title, weight=weight)
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def active(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.state in ("queued", "running", "suspended")]

    def heavy_running(self) -> list[Task]:
        return [t for t in self.active() if t.weight == "heavy" and t.state == "running"]

    def forget(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)

    # --- events ----------------------------------------------------------

    def subscribe(self, listener: Callable[[dict], Awaitable[None]]) -> None:
        """Receive every task message, for streaming to the frontend."""
        self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[dict], Awaitable[None]]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def _announce(self, task: Task) -> None:
        message = task.to_message()
        for listener in list(self._listeners):
            try:
                await listener(message)
            except Exception:  # noqa: BLE001 - a broken listener is not a task failure
                self.unsubscribe(listener)

    async def set_state(self, task: Task, state: State, detail: str | None = None) -> None:
        task.state = state
        if detail is not None:
            task.detail = detail
        await self._announce(task)

    # --- the interesting part -------------------------------------------

    async def stand_aside(self, tasks: Iterable[Task], reason: str) -> list[Task]:
        """Ask running heavy work to yield at its next checkpoint."""
        asked = []
        for task in tasks:
            if task.state != "running" or task.weight != "heavy":
                continue
            task._stand_aside.set()
            await self.set_state(task, "suspended", reason)
            asked.append(task)
        return asked

    async def resume(self, tasks: Iterable[Task]) -> None:
        for task in tasks:
            if task.cancelled:
                continue
            task._stand_aside.clear()
            await self.set_state(task, "running", "picking that back up")

    async def checkpoint(self, task: Task) -> None:
        """Call this wherever stopping is free.

        Returns immediately unless the task has been asked to stand aside, in
        which case it waits until it is allowed to continue. Raises
        `Cancelled` if the task has been cancelled — cancellation is the only
        thing that ends a task from outside, and it can only arrive because
        somebody asked for it.
        """
        if task.cancelled:
            raise Cancelled(task.id)
        while task._stand_aside.is_set():
            if task.cancelled:
                raise Cancelled(task.id)
            await asyncio.sleep(0.05)
        if task.cancelled:
            raise Cancelled(task.id)

    async def cancel(self, task_ids: Iterable[str] | None = None) -> list[str]:
        """Abandon work. `None` cancels everything in flight."""
        targets = list(task_ids) if task_ids else [t.id for t in self.active()]
        cancelled = []
        for task_id in targets:
            task = self._tasks.get(task_id)
            if task is None or task.state in ("completed", "failed", "cancelled"):
                continue
            task._cancelled.set()
            # A cancelled task must not sit waiting at a checkpoint forever.
            task._stand_aside.clear()
            await self.set_state(task, "cancelled", "cancelled")
            cancelled.append(task_id)
        return cancelled

    def should_preempt(self, requested: bool) -> bool:
        """Whether an incoming turn justifies asking heavy work to wait.

        The frontend has already decided a human is waiting; this is the
        backend's side of the same question — is anything actually in the way?
        """
        return requested and bool(self.heavy_running())
