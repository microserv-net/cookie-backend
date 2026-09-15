"""The model provider.

Only this module knows that the models are hosted by Ollama, and only the
configuration knows which models they are. Everything above talks about
*roles* — router, worker, architect — so replacing a model, or eventually the
host, is a configuration change rather than a rewrite.

The one non-obvious responsibility here is residency. The first machine holds
roughly one large model in RAM, so `keep_alive` is not a tuning detail: it is
the difference between the architect being available and the worker being
evicted to make room for it. Ollama's own lifecycle is used rather than
reimplemented, because it already knows what is loaded and we do not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from .config import Config, ModelRole


class ModelError(Exception):
    """Something went wrong that the user should hear about in plain words."""


@dataclass
class LoadedModel:
    name: str
    size_bytes: int
    expires_at: str | None = None

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1_000_000_000


class OllamaProvider:
    """Thin async client over the Ollama HTTP API."""

    def __init__(self, config: Config, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.endpoint = config.ollama_endpoint.rstrip("/")
        self._client = client
        self._owns_client = client is None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.request_timeout_seconds)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # --- introspection ---------------------------------------------------

    async def available(self) -> bool:
        """Is Ollama up? Used by the health endpoint and by `doctor`."""
        try:
            client = await self._http()
            response = await client.get(f"{self.endpoint}/api/version", timeout=3.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def installed_models(self) -> list[str]:
        client = await self._http()
        try:
            response = await client.get(f"{self.endpoint}/api/tags", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise ModelError(f"could not ask Ollama what it has: {e}") from e
        return [m["name"] for m in response.json().get("models", [])]

    async def loaded_models(self) -> list[LoadedModel]:
        """What is resident right now, and how much memory it is using."""
        client = await self._http()
        try:
            response = await client.get(f"{self.endpoint}/api/ps", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise ModelError(f"could not ask Ollama what is loaded: {e}") from e
        return [
            LoadedModel(
                name=m.get("name", "?"),
                size_bytes=int(m.get("size", 0)),
                expires_at=m.get("expires_at"),
            )
            for m in response.json().get("models", [])
        ]

    async def resident_gb(self) -> float:
        try:
            return sum(m.size_gb for m in await self.loaded_models())
        except ModelError:
            return 0.0

    # --- residency -------------------------------------------------------

    async def unload(self, model: str) -> None:
        """Evict a model now.

        `keep_alive: 0` is Ollama's own idiom for this, so we are not fighting
        its lifecycle — we are using it at the point where we happen to know
        the model is no longer needed, which Ollama cannot know.
        """
        client = await self._http()
        try:
            await client.post(
                f"{self.endpoint}/api/generate",
                json={"model": model, "keep_alive": 0, "prompt": ""},
                timeout=30.0,
            )
        except httpx.HTTPError:
            # Eviction is an optimisation. Failing to evict costs memory, not
            # correctness, and should never surface to the user.
            pass

    async def make_room_for(self, role: ModelRole) -> list[str]:
        """Evict what has to go before `role` can be loaded.

        Returns the models evicted, so the caller can say so in the activity
        log rather than leaving a ten-second pause unexplained.
        """
        budget = self.config.model_memory_gb
        try:
            loaded = await self.loaded_models()
        except ModelError:
            return []
        resident = sum(m.size_gb for m in loaded)
        if resident < budget * 0.75:
            return []

        evicted = []
        # Evict the largest thing that is not the model we are about to use;
        # on this machine that is nearly always the architect finishing up.
        for model in sorted(loaded, key=lambda m: m.size_gb, reverse=True):
            if model.name.startswith(role.model.split(":")[0]) and model.name == role.model:
                continue
            await self.unload(model.name)
            evicted.append(model.name)
            resident -= model.size_gb
            if resident < budget * 0.6:
                break
        return evicted

    # --- generation ------------------------------------------------------

    async def chat(
        self,
        role: ModelRole,
        messages: list[dict],
        *,
        options: dict | None = None,
    ) -> AsyncIterator[str]:
        """Stream a reply, token by token.

        Yields text fragments as they arrive. The caller is free to stop
        consuming — that is how cancellation reaches the model, and closing the
        response is what tells Ollama to stop generating.
        """
        client = await self._http()
        payload = {
            "model": role.model,
            "messages": messages,
            "stream": True,
            "keep_alive": role.keep_alive,
            "options": {**role.options, **(options or {})},
        }
        try:
            async with client.stream(
                "POST", f"{self.endpoint}/api/chat", json=payload
            ) as response:
                if response.status_code == 404:
                    raise ModelError(
                        f"the model {role.model} is not installed. "
                        f"Run: ollama pull {role.model}"
                    )
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if message.get("error"):
                        raise ModelError(str(message["error"]))
                    fragment = (message.get("message") or {}).get("content", "")
                    if fragment:
                        yield fragment
                    if message.get("done"):
                        return
        except httpx.ConnectError as e:
            raise ModelError(
                f"I can't reach Ollama at {self.endpoint}. Is it running?"
            ) from e
        except httpx.HTTPError as e:
            raise ModelError(f"the model call failed: {e}") from e
