"""Configuration.

TOML, because `tomllib` is in the standard library and the frontend already
uses it — one format across the system is worth more than the merits of any
particular one.

Model names never appear in source. Moving to a bigger machine should mean
editing this file, not the orchestrator.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = """\
# cookie-backend configuration
# Regenerate this file with: cookie-backend init

[server]
host = "0.0.0.0"          # reachable from your laptop and over Tailscale
port = 8080
base_path = "/api"

[models.router]
provider = "ollama"
model = "qwen3:1.7b"
# Resident: it is on the path of every request and reloading it would be felt
# on all of them.
keep_alive = "30m"

[models.worker]
provider = "ollama"
model = "qwen3:4b"
keep_alive = "10m"

[models.architect]
provider = "ollama"
model = "qwen3:8b"
# Unloaded promptly: holding this alongside the worker is what pushes a 16 GB
# machine into swap.
keep_alive = "2m"

[ollama]
endpoint = "http://127.0.0.1:11434"
request_timeout_seconds = 600

[limits]
# Roughly how much model weight can be resident at once, in gigabytes. The
# scheduler uses this to decide whether loading the architect means evicting
# the worker first.
model_memory_gb = 9.0
max_concurrent_heavy = 1
"""


@dataclass
class ModelRole:
    """One named role and the model that currently fills it."""

    provider: str = "ollama"
    model: str = "qwen3:4b"
    keep_alive: str = "10m"
    options: dict = field(default_factory=dict)


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8080
    base_path: str = "/api"
    ollama_endpoint: str = "http://127.0.0.1:11434"
    request_timeout_seconds: float = 600.0
    model_memory_gb: float = 9.0
    max_concurrent_heavy: int = 1
    roles: dict[str, ModelRole] = field(default_factory=dict)
    data_dir: Path = field(default_factory=lambda: data_home())

    def role(self, name: str) -> ModelRole:
        """The model filling `name`, falling back to the worker.

        Falling back rather than raising matters: a config written before a
        role existed should degrade, not refuse to start.
        """
        if name in self.roles:
            return self.roles[name]
        if "worker" in self.roles:
            return self.roles["worker"]
        return ModelRole()

    @property
    def chat_path(self) -> str:
        return self.base_path.rstrip("/") + "/v1/chat"


def config_home() -> Path:
    """Platform-independent config directory, honouring XDG and an override."""
    override = os.environ.get("COOKIE_BACKEND_HOME")
    if override:
        return Path(override) / "config"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "cookie-backend"


def data_home() -> Path:
    """Where tokens, memory and state live."""
    override = os.environ.get("COOKIE_BACKEND_HOME")
    if override:
        return Path(override) / "data"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "cookie-backend"


def config_file() -> Path:
    return config_home() / "config.toml"


def load(path: Path | None = None) -> Config:
    """Load configuration, writing the default file on first run."""
    path = path or config_file()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG)

    raw = tomllib.loads(path.read_text())
    server = raw.get("server", {})
    ollama = raw.get("ollama", {})
    limits = raw.get("limits", {})

    roles: dict[str, ModelRole] = {}
    for name, section in (raw.get("models") or {}).items():
        roles[name] = ModelRole(
            provider=section.get("provider", "ollama"),
            model=section.get("model", "qwen3:4b"),
            keep_alive=section.get("keep_alive", "10m"),
            options=section.get("options", {}),
        )

    return Config(
        host=server.get("host", "0.0.0.0"),
        port=int(server.get("port", 8080)),
        base_path=server.get("base_path", "/api"),
        ollama_endpoint=ollama.get("endpoint", "http://127.0.0.1:11434"),
        request_timeout_seconds=float(ollama.get("request_timeout_seconds", 600)),
        model_memory_gb=float(limits.get("model_memory_gb", 9.0)),
        max_concurrent_heavy=int(limits.get("max_concurrent_heavy", 1)),
        roles=roles,
    )
