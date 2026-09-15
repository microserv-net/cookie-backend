"""Shared test setup, and the app factory CI uses for a live conformance run."""

import tempfile
from pathlib import Path

from cookie_backend.config import Config, ModelRole
from cookie_backend.server import create_app


def app_for_ci():
    """A real server with a throwaway data directory.

    Used by the conformance job, which drives the actual HTTP surface rather
    than the test client — the two find different bugs.
    """
    config = Config(
        roles={"router": ModelRole(model="qwen3:1.7b"),
               "worker": ModelRole(model="qwen3:4b"),
               "architect": ModelRole(model="qwen3:8b")},
        data_dir=Path(tempfile.mkdtemp()),
    )
    return create_app(config)
