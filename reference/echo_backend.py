#!/usr/bin/env python3
"""A conforming Cookie backend in the standard library.

It does not think. It echoes, slowly, with a fake task attached — which is
exactly enough to exercise every path the frontend has: streaming replies,
task progress, suspension, cancellation, health.

Run it while building the real backend, or while working on the frontend, so
there is always something real on the other end of the wire.

    python3 reference/echo_backend.py --port 8080
    cookie-interface --backend http://127.0.0.1:8080/api

Deliberately stdlib-only and single-file. It is a reference for the protocol,
not a foundation for the implementation.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL = "cookie-interface/1"

# Tasks the frontend believes are running, keyed by id. A real backend would
# have a task manager; this is the smallest thing that behaves correctly.
TASKS: dict[str, dict] = {}
TASKS_LOCK = threading.Lock()


def sentences(text: str) -> list[str]:
    """Split into speakable pieces, so the reply streams the way a model's
    output would rather than arriving in one lump."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # --- plumbing --------------------------------------------------------

    def log_message(self, fmt, *args):  # quieter than the default
        print(f"  {self.address_string()} {fmt % args}")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("content-length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    def _begin_stream(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "application/x-ndjson")
        self.send_header("cache-control", "no-store")
        # Chunked, because the whole point is that the length is not known
        # when the first sentence is already being spoken.
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()

    def _emit(self, obj: dict) -> None:
        """Write one NDJSON line as an HTTP chunk."""
        line = (json.dumps(obj) + "\n").encode()
        self.wfile.write(b"%x\r\n" % len(line) + line + b"\r\n")
        self.wfile.flush()

    def _end_stream(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # --- routes ----------------------------------------------------------

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/v1/health"):
            self._json(200, {"status": "ok", "backend": "echo-reference",
                             "protocol": PROTOCOL})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        if path.endswith("/v1/chat"):
            self.handle_chat()
        elif path.endswith("/v1/cancel"):
            self.handle_cancel()
        else:
            self._json(404, {"error": "not found"})

    def handle_cancel(self) -> None:
        request = self._read_json()
        task_id = request.get("task_id")
        with TASKS_LOCK:
            targets = [task_id] if task_id else list(TASKS)
            for tid in targets:
                if tid in TASKS:
                    TASKS[tid]["cancelled"] = True
        print(f"  cancelled: {targets}")
        self._json(200, {"cancelled": targets})

    def handle_chat(self) -> None:
        request = self._read_json()
        text = (request.get("text") or "").strip()
        scheduling = request.get("scheduling") or {}
        priority = scheduling.get("priority", "normal")
        preempt = bool(scheduling.get("preempt"))

        print(f"  heard: {text!r}  priority={priority} preempt={preempt}")

        # Cooperative preemption. A real backend would suspend its own heavy
        # work at a model or tool boundary; here we mark whatever is running
        # as suspended, answer, and put it back. The frontend shows the
        # suspension, which is the behaviour that matters to the user.
        suspended: list[str] = []
        if preempt:
            with TASKS_LOCK:
                for tid, task in TASKS.items():
                    if task["state"] == "running" and task["weight"] == "heavy":
                        task["state"] = "suspended"
                        suspended.append(tid)

        task_id = f"task-{uuid.uuid4().hex[:8]}"
        # A long request is treated as heavy work, a short one as a quick
        # answer. The real router will do this with a model; the shape of the
        # messages is identical either way.
        heavy = len(text.split()) > 12
        with TASKS_LOCK:
            TASKS[task_id] = {
                "state": "running",
                "weight": "heavy" if heavy else "light",
                "cancelled": False,
            }

        self._begin_stream()
        try:
            self._emit({"type": "task", "id": task_id, "state": "running",
                        "title": "thinking about that",
                        "weight": "heavy" if heavy else "light"})

            for tid in suspended:
                self._emit({"type": "task", "id": tid, "state": "suspended",
                            "detail": "paused while I deal with this"})

            reply = f"You said: {text}." if text else "I didn't catch that."
            if heavy:
                reply += " That was a long one, so I'd normally take a while over it."

            for piece in sentences(reply):
                with TASKS_LOCK:
                    if TASKS[task_id]["cancelled"]:
                        self._emit({"type": "task", "id": task_id,
                                    "state": "cancelled"})
                        self._emit({"type": "end"})
                        return
                self._emit({"type": "delta", "text": piece + " "})
                # Pacing makes streaming visible; a real model provides its
                # own delay for free.
                time.sleep(0.25)

            self._emit({"type": "task", "id": task_id, "state": "completed"})

            for tid in suspended:
                with TASKS_LOCK:
                    if tid in TASKS and not TASKS[tid]["cancelled"]:
                        TASKS[tid]["state"] = "running"
                self._emit({"type": "task", "id": tid, "state": "running",
                            "detail": "picking that back up"})

            self._emit({"type": "end"})
        except (BrokenPipeError, ConnectionResetError):
            # The frontend went away mid-reply. Not an error: it happens
            # every time somebody interrupts.
            print("  client disconnected mid-reply")
        finally:
            self._end_stream_quietly()
            with TASKS_LOCK:
                TASKS.pop(task_id, None)

    def _end_stream_quietly(self) -> None:
        try:
            self._end_stream()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to accept from the LAN or Tailscale")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"echo backend on http://{args.host}:{args.port}/api")
    print(f"  cookie-interface --backend http://{args.host}:{args.port}/api")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    main()
