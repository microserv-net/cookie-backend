
import asyncio, pytest
from cookie_backend.tasks import TaskManager, Cancelled

async def test_checkpoint_is_free_when_nothing_is_urgent():
    m = TaskManager()
    t = m.create("building", "heavy")
    await m.set_state(t, "running")
    await asyncio.wait_for(m.checkpoint(t), timeout=0.5)

async def test_standing_aside_holds_at_the_checkpoint_then_resumes():
    m = TaskManager()
    heavy = m.create("building", "heavy")
    await m.set_state(heavy, "running")
    progress = []

    async def work():
        for step in range(3):
            await m.checkpoint(heavy)
            progress.append(step)
            await asyncio.sleep(0.02)

    runner = asyncio.create_task(work())
    await asyncio.sleep(0.01)
    await m.stand_aside([heavy], "answering something short")
    await asyncio.sleep(0.15)
    held = len(progress)
    assert heavy.state == "suspended"
    await m.resume([heavy])
    await asyncio.wait_for(runner, timeout=2)
    # It paused, then finished what it was doing: nothing was lost.
    assert held < 3 and progress == [0, 1, 2]
    assert heavy.state == "running"

async def test_cancellation_ends_the_task_and_releases_a_held_checkpoint():
    m = TaskManager()
    t = m.create("building", "heavy")
    await m.set_state(t, "running")
    await m.stand_aside([t], "waiting")

    async def work():
        with pytest.raises(Cancelled):
            await m.checkpoint(t)
        return "stopped"

    runner = asyncio.create_task(work())
    await asyncio.sleep(0.05)
    assert await m.cancel([t.id]) == [t.id]
    assert await asyncio.wait_for(runner, timeout=2) == "stopped"
    assert t.state == "cancelled"

async def test_preemption_is_only_requested_when_something_is_in_the_way():
    m = TaskManager()
    assert not m.should_preempt(True)
    heavy = m.create("building", "heavy")
    await m.set_state(heavy, "running")
    assert m.should_preempt(True)
    assert not m.should_preempt(False)

async def test_task_messages_match_the_protocol():
    m = TaskManager()
    t = m.create("fixing the tests", "heavy")
    seen = []
    m.subscribe(lambda msg: asyncio.sleep(0, result=seen.append(msg)))
    await m.set_state(t, "running", "running the suite")
    assert seen[-1] == {"type": "task", "id": t.id, "state": "running",
                        "title": "fixing the tests", "weight": "heavy",
                        "detail": "running the suite"}
