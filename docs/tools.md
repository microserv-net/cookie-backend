# Tools

Without tools the assistant can plan and talk but cannot touch anything, and
the validator is a second opinion from another model rather than a fact. Tools
are where "the tests pass" stops being a claim and becomes an exit code.

## Where a tool runs is part of its contract

| | |
|---|---|
| **backend** | the web, memory. Things that genuinely live on the server. |
| **frontend** | files, shell, applications, VS Code, git, the screen. |

The backend never touches your filesystem, shell or screen. It *asks*. A model
that has been talked into something can reach at most a request for
`filesystem.delete`, which your machine can refuse, confirm, or classify —
because your machine is the one being acted upon and the only one that knows
whether you are standing in front of it.

That is why there is no `shell(command_string)` on this side. The backend
sends `shell.run` with structured arguments; the frontend decides what that
means and whether to do it.

## Discovery, not enumeration

There are twenty-odd tools and there will be more. Sending every schema on
every request is the obvious approach and it is wrong on a machine where
context costs seconds.

So the router names *capabilities* — `files`, `git`, `web`, `memory`, `code`,
`shell`, `screen`, `applications`, `projects`, `system`, `research`, `time` —
and only the tools tagged with those reach the model. A request about a
repository never sees the memory tools.

Descriptions are one line each, for the same reason:

```
- shell.run(command: string, arguments?: list of strings, cwd?: string,
  timeout_seconds?: integer) — run a command and return its output and exit code
```

## Versioned contracts

A tool is `name@vN`. Registering a higher version supersedes a lower one by
name; registering a lower one does nothing. Models are allowed to refer to
either `shell.run` or `shell.run@v1`, because they will.

Arguments are checked against the schema *before* anything runs, so a
hallucinated argument name comes back as "shell.run has no argument `shell`"
rather than a traceback. A hallucinated *tool* comes back as "there is no tool
called filesystem.teleport". Both are things the model can plan around.

## Risk

Declared per tool: `safe`, `normal`, `dangerous`, `critical`.

Nothing in the backend decides whether an action is *allowed*. The risk level
travels with the request and the frontend's confirmation policy keys off it —
`git.push` is critical, `filesystem.read` is safe, and the difference is
whether you get asked.

## The wire protocol

A tool call goes out on the turn's existing reply stream:

```json
{"type":"tool.request","id":"task-3f2a-call-0","tool":"filesystem.search",
 "version":1,"arguments":{"pattern":"*.rs","root":"~/projects"},"risk":"safe"}
```

The frontend posts the answer back on a separate request:

```http
POST /api/v1/tool-result
{"id":"task-3f2a-call-0","ok":true,"summary":"4 matches",
 "evidence":"4 files under ~/projects","data":["src/main.rs", "..."]}
```

`summary` is what the model reads and what may be spoken. `evidence` is what
the validator judges against. `data` is for tools feeding other tools and is
truncated before it reaches a model.

Every call has a deadline. A frontend that has gone away must not leave a turn
hanging, so a timeout is reported to the model as a failed tool call — which
it can plan around — rather than as an exception nobody sees.

An unknown `id` is accepted and ignored: it means the turn moved on, and the
frontend should not be made to care.

## Advertising

A frontend lists what it implements when it opens a turn:

```json
{"text":"...", "tools":["filesystem.search","filesystem.read","shell.run"]}
```

Anything not listed is never offered to the model, so a request that would
need it is planned differently rather than failing halfway. A frontend that
advertises nothing is assumed to implement everything, which keeps older
builds working.

`GET /v1/tools` returns the full catalogue with schemas, risk levels and where
each one runs.

## Adding a tool

Backend-side, in `src/tools/builtin.rs`:

```rust
Tool {
    name: "web.fetch",
    version: 1,
    summary: "fetch a page and return its readable text",
    capabilities: &["web", "research"],
    parameters: &[("url", "string")],
    required: &["url"],
    runs: Runs::Backend,
    risk: Risk::Safe,
    handler: Some(handler(fetch)),
}
```

Frontend-side, add it to `tools()` in `src/tools/frontend.rs` and implement it
in the frontend. Nothing in the orchestrator changes either way — that is
the property the whole package is arranged around.

## Memory

Backend memory persists indefinitely by default, which is the opposite of the
frontend's generated-audio policy and deliberately so: audio is a by-product,
this is what Cookie knows.

Entries carry `created_at`, `updated_at` and `last_referenced_at`, because
"forget what I have not asked about in two months" needs the distinction
between when something was learned and when it last mattered. `stale()`
returns candidates; deleting them is a separate, explicit act, so a vague
request cannot wipe everything.

Recall is substring matching over keys, values and tags. Not embeddings: an
embedding model is another thing resident in RAM on a machine that has none
spare, and for a few hundred facts about one person's setup it is not the
bottleneck. Worth revisiting when it is.
