# cookie-backend

Cookie's mind. Runs on a machine you own; talks to
[cookie-frontend](https://github.com/microserv-net/cookie-frontend) over an
authenticated HTTP protocol.

Currently implemented: the protocol, authentication and pairing, the model
abstraction over Ollama, and the task lifecycle including cooperative
preemption. You can talk to a real model through Cookie today. The
orchestration on top — router, architect, worker, the validation loop, tools —
is next.

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
| `docs/architecture.md` | the design; implemented parts marked. |
| `docs/tools.md` | the tool system, the wire protocol, and how to add one. |
| `docs/scheduling.md` | how one machine holding one large model stays responsive. |
| `cookie_backend/` | the backend itself: server, auth, Ollama provider, tasks. |
| `reference/echo_backend.py` | a conforming backend in ~200 lines of stdlib Python, for developing the frontend without any of this running. |
| `tools/conformance.py` | checks any backend against the contract. Both the real server and the reference pass it. |

## Running it

```bash
cargo install --path .
cookie-backend init          # writes the default config, tells you where
ollama pull qwen3:4b         # and qwen3:1.7b, qwen3:8b when you want them
cookie-backend doctor        # is everything ready?
cookie-backend               # serve
```

Then on your laptop:

```bash
cookie-backend pair                       # on this machine: prints a code
# exchange it for a token via POST /v1/pair, then:
export COOKIE_BACKEND_TOKEN=<token>
cookie-interface --backend http://<this-machine>:8080/api
```

Say something. It goes microphone → recognition → here → Ollama → back →
spoken, and the orb follows the whole way.

### Without Ollama

The backend still starts, still pairs, and still answers — it tells you it
cannot reach a model rather than going quiet. `cargo test` covers the whole
protocol with no Ollama, no model and no network.

## Rust, like the frontend

One language across the system. The alternative — a Rust frontend and a
scripting-language backend — trades a genuine property for a convenience:
the tool contracts, the protocol types and the risk classifications exist on
both sides of the wire, and having them checked by the same compiler is worth
more than faster iteration on this side. It also means one toolchain to
install on the backend machine and one binary to deploy, with no interpreter
or virtual environment to keep alive next to Ollama.

## Model residency

The first machine holds roughly one large model in memory, so `keep_alive` is
not a tuning detail. The router stays resident because it is on the path of
every request; the architect is loaded for planning and evicted afterwards,
because holding it alongside the worker is what pushes 16 GB into swap.
`GET /v1/models` reports what is resident against the budget, and the server
evicts before loading when it has to. See [docs/scheduling.md](docs/scheduling.md).
