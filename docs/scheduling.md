# Scheduling on one machine

The first deployment is a mini PC with 16 GB of RAM, of which about 12 is
usable. Qwen3:8b at Q4 is 5-6 GB. That means one heavy model resident, or two
smaller ones — and the difference between an assistant that feels alive and
one that feels stuck is entirely in what happens when you speak while it is
busy.

## The interface does the labelling

The backend cannot tell from the text whether a new utterance is adding to the
job it is already doing or interrupting with something small. The frontend
can: it knows a human is standing there, how long they have been waiting, and
that they just spoke a short sentence rather than typing a specification. So
every turn arrives labelled:

```json
"scheduling": { "priority": "interactive", "preempt": true }
```

| Situation | priority | preempt |
|---|---|---|
| nothing running | normal | false |
| heavy task running, 12 words or fewer | interactive | true |
| heavy task running, longer utterance | normal | false |
| explicitly told to wait | background | false |

A long request is a new job, not an aside, so it queues rather than thrashing
the machine.

## Preempt means yield, not abandon

`preempt: true` asks for a suspension at the next **natural boundary** — a
point where you are already between things and nothing is lost by stopping:

- a model call has just returned,
- you are about to load or swap a model,
- a tool is running and you are waiting on it,
- you are between steps of a plan.

Never mid-generation, and never by discarding work. Report it:

```json
{"type":"task","id":"task-18","state":"suspended","detail":"paused while I deal with this"}
```

and when you pick it back up:

```json
{"type":"task","id":"task-18","state":"running","detail":"picking that back up"}
```

The frontend shows a suspended task as paused rather than stuck, and stops
asking for preemption while it is suspended — a suspended heavy task is not
occupying the machine.

`reference/echo_backend.py` implements exactly this handshake, so the
behaviour can be seen before any model exists.

## Model residency

The natural boundaries above are also the only sensible points to unload a
model. A rough starting policy for the first machine:

- The router (1.7b, ~1.5 GB) stays resident. It is on the path of every
  request, and reloading it would be felt on all of them.
- The worker (4b) stays resident while a task is active.
- The architect (8b) is loaded for planning and unloaded when planning is
  done, because holding it alongside the worker is what pushes this machine
  into swap.

This should be measured rather than assumed, and it is configuration, not
code: moving to a larger machine should mean changing model names and
residency settings, not touching the orchestrator.

## Ignoring all of this is allowed

A backend that ignores the scheduling hints is still correct — it just makes a
constrained machine feel worse. Nothing else in the protocol depends on
honouring them.
