# cookie-backend

Cookie's mind. Runs on a machine you own; talks to
[cookie-frontend](https://github.com/microserv-net/cookie-frontend) over an
authenticated HTTP protocol.

**Nothing here is implemented yet.** This repository currently contains the
contract, the design, and a reference server you can run today — so the
frontend can be developed against something real while the actual backend is
built.

```
cookie-frontend (your laptop)          cookie-backend (your server)
  microphone, voice, orb        ──▶      router → worker → architect
  local tools, screen, VS Code  ◀──      tools, memory, web, GitHub
         authenticated HTTP                     Ollama
```

## What is here

| | |
|---|---|
| `docs/protocol.md` | the contract with the frontend. Authoritative. |
| `docs/architecture.md` | the planned design, marked as design. |
| `docs/scheduling.md` | how one machine holding one large model stays responsive. |
| `reference/echo_backend.py` | a conforming backend in ~200 lines of stdlib Python. Run it, talk to Cookie, watch the whole path work. |
| `tools/conformance.py` | checks any backend against the contract. |

## Try it now

```bash
python3 reference/echo_backend.py --port 8080
```

Then point the frontend at it:

```bash
cookie-interface --backend http://127.0.0.1:8080/api
```

Say something. It will repeat it back, streamed sentence by sentence, with a
fake task so you can watch the progress path work end to end. Ask it to do
something long and then interrupt — that exercises the part that is easy to
get wrong.

```bash
python3 tools/conformance.py http://127.0.0.1:8080/api
```

## What is deliberately not decided yet

The implementation language. The reference is stdlib Python because it must be
readable and runnable by anyone in ten seconds — that is not a vote. The real
backend runs Ollama-hosted models and a versioned tool system, and the choice
between Python and Rust for it depends on how much of the tool execution ends
up on this side rather than the frontend's. Deciding it in a README before any
of that is built would be guessing.

What *is* decided is the protocol, because the frontend already speaks it.
