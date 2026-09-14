#!/usr/bin/env python3
"""Check a backend against the protocol in docs/protocol.md.

    python3 tools/conformance.py http://127.0.0.1:8080/api

Every check maps to a sentence in the contract. A backend that passes all of
these will work with the frontend; one that fails any of them will
misbehave in a way that is hard to diagnose from the frontend's side, which
is exactly why this exists.

Stdlib only, so it runs anywhere the backend does.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request

PROTOCOL = "cookie-interface/1"

passed: list[str] = []
failed: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        passed.append(name)
        print(f"  ok      {name}")
    else:
        failed.append((name, detail))
        print(f"  FAILED  {name}" + (f" — {detail}" if detail else ""))
    return condition


def post_stream(url: str, body: dict, timeout: float = 30.0):
    """POST and yield decoded NDJSON/SSE objects as they arrive."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json",
                 "accept": "application/x-ndjson, text/event-stream, application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        if "ndjson" in content_type or "event-stream" in content_type:
            for raw in response:
                line = raw.decode().strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    return
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield {"type": "__unparseable__", "raw": line}
        else:
            yield {"type": "__whole__", "body": json.loads(response.read())}


def turn(base: str, text: str, **scheduling):
    return post_stream(base.rstrip("/") + "/v1/chat", {
        "protocol": PROTOCOL,
        "session_id": "conformance",
        "utterance_id": "u-" + str(int(time.time() * 1000)),
        "text": text,
        "final": True,
        "interface": {"speech": True, "listening": True, "visual": "orb"},
        "scheduling": {"priority": scheduling.get("priority", "normal"),
                       "preempt": scheduling.get("preempt", False)},
        "active_tasks": scheduling.get("active_tasks", []),
    })


def main(base: str) -> int:
    print(f"checking {base}\n")

    # --- health ----------------------------------------------------------
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/v1/health", timeout=5) as r:
            health_ok = r.status == 200
            body = r.read()
    except Exception as e:  # noqa: BLE001 - any failure is the same failure here
        check("health responds", False, str(e))
        print("\nnothing else can be checked without health. Is it running?")
        return 1
    check("health responds 200", health_ok)
    check("health returns JSON", _is_json(body), "the frontend parses it")

    # --- a turn ----------------------------------------------------------
    messages = []
    started = time.monotonic()
    first_delta_at = None
    try:
        for message in turn(base, "hello, can you hear me"):
            messages.append(message)
            if first_delta_at is None and message.get("type") in ("delta", "text", "token"):
                first_delta_at = time.monotonic() - started
    except Exception as e:  # noqa: BLE001
        check("a turn completes", False, str(e))
        return 1

    check("a turn completes", bool(messages))
    check("nothing unparseable was sent",
          not any(m.get("type") == "__unparseable__" for m in messages),
          "every line must be one JSON object")

    whole = [m for m in messages if m.get("type") == "__whole__"]
    if whole:
        reply = whole[0]["body"]
        check("non-streaming reply names its text",
              any(k in reply for k in ("reply", "text", "message", "content")),
              "frontend looks for reply/text/message/content")
        print("\n  note: this backend answers in one piece rather than streaming.")
        print("  That is allowed, but the voice will not start until you finish.")
    else:
        deltas = [m for m in messages if m.get("type") in ("delta", "text", "token")]
        check("the reply streams as deltas", bool(deltas))
        check("every delta carries text",
              all(isinstance(m.get("text") or m.get("delta"), str) for m in deltas))
        check("the stream is terminated",
              any(m.get("type") in ("end", "done") for m in messages),
              "without an end the frontend waits for the connection to close")
        if first_delta_at is not None:
            check("the first delta arrives within 10s", first_delta_at < 10,
                  f"took {first_delta_at:.1f}s")

    # --- tasks -----------------------------------------------------------
    tasks = [m for m in messages if m.get("type") in ("task", "task.update", "progress")]
    if tasks:
        check("task messages carry an id",
              all(t.get("id") or t.get("task_id") for t in tasks))
        states = {t.get("state") for t in tasks if t.get("state")}
        known = {"queued", "running", "suspended", "completed", "failed",
                 "cancelled", "canceled", "paused", "yielded", "done",
                 "finished", "error"}
        check("task states are from the documented set",
              states <= known, f"saw {states - known}")
    else:
        print("\n  note: no task messages. Allowed, but the orb cannot show")
        print("  progress and 'what are you working on?' has no answer.")

    # --- cancellation ----------------------------------------------------
    cancelled = {"done": False}

    def long_turn():
        try:
            for _ in turn(base, "please do something that takes a while, "
                                "at length, with many steps involved in it"):
                pass
        except Exception:  # noqa: BLE001 - a dropped stream is a valid outcome
            pass
        cancelled["done"] = True

    thread = threading.Thread(target=long_turn, daemon=True)
    thread.start()
    time.sleep(0.6)
    try:
        request = urllib.request.Request(
            base.rstrip("/") + "/v1/cancel",
            data=json.dumps({"protocol": PROTOCOL, "session_id": "conformance",
                             "task_id": None}).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as r:
            check("cancel is accepted", r.status in (200, 202, 204))
    except urllib.error.HTTPError as e:
        check("cancel is accepted", False, f"HTTP {e.code}")
    except Exception as e:  # noqa: BLE001
        check("cancel is accepted", False, str(e))
    thread.join(timeout=20)
    check("cancel actually stops the turn", cancelled["done"],
          "the turn was still running 20s after cancelling")

    # --- report ----------------------------------------------------------
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    if failed:
        print("\nwhat to fix:")
        for name, detail in failed:
            print(f"  {name}" + (f" — {detail}" if detail else ""))
        return 1
    print("\nThis backend will work with cookie-interface.")
    return 0


def _is_json(raw: bytes) -> bool:
    try:
        json.loads(raw)
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
