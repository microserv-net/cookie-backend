# Architecture (design, not implementation)

Everything here is a plan. Nothing in this document is built yet; the only
committed decisions are in [protocol.md](protocol.md), because the frontend
already depends on those.

## Three models, smallest first

```
                    ┌──────────────┐
   utterance ──────▶│ router 1.7b  │  what kind of thing is this?
                    └──────┬───────┘
              ┌────────────┴────────────┐
              ▼                         ▼
      routine / small              complex / multi-step
              │                         │
       ┌──────────────┐          ┌──────────────┐
       │ worker 4b    │          │ architect 8b │  plan
       └──────┬───────┘          └──────┬───────┘
              │◀────────────────────────┘
              ▼
       tool execution   (here, or on the frontend)
              │
              ▼
       worker validates
        ┌─────┴─────┐
     success      failure ──▶ architect replans, with evidence
```

The point of the router is not intelligence, it is **cost**. Most utterances
never need an 8b model, and on this hardware loading one has a price measured
in seconds.

## The validation loop is the important part

The architect must not be allowed to assume its own output worked. The worker
executes, then validates against something external — a compilation, a test
run, a linter, a diff, a command's exit code — and produces a *structured*
failure report rather than a sentence. The architect replans against that
evidence.

Bounded retries, loop detection, detection of repeated failed approaches, and
preservation of the work that did succeed rather than starting over.

## Tools are plugins, and discovery is what keeps prompts small

A large tool ecosystem and a small context window are in tension. Dumping
every tool schema into every request is the obvious approach and it is wrong
on this hardware. Instead: a registry with capability metadata, the router
identifies the relevant capability, and only those schemas reach the model.

Tool contracts are versioned. Adding a tool must not require touching the
orchestrator.

## Where work happens

| Frontend | Backend |
|---|---|
| microphone, STT, TTS, orb | routing, planning, validation |
| local files, shell, apps, windows | tool orchestration, memory |
| screen capture, VS Code | web research, GitHub |
| permissions, confirmation UI | task state, model lifecycle |

Some tools span both. The backend asks for a structured operation —
`filesystem.search` with arguments — rather than shipping a shell string, and
the frontend executes it through a controlled layer with risk classifications
and confirmation. That boundary is a security property, not a layering
preference: an unauthenticated endpoint that runs arbitrary shell on the LAN
is the failure mode this design exists to avoid.

## Open questions

- **Implementation language.** See the README. It depends on how much tool
  execution ends up on this side rather than the frontend's.
- **Memory.** Persistent by default, with metadata rich enough to answer
  "delete what I have not asked about in two months" — which needs
  `last_referenced_at`, not just `created_at`, and safeguards so a vague
  deletion request cannot wipe everything.
- **Pairing.** One exchange, a stored credential, no per-request approval,
  revocable. The frontend already expects a bearer token.
