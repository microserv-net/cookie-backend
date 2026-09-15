"""Pairing and authentication.

The threat here is not a stranger on the internet. It is that this endpoint
can eventually run commands on your laptop, and it lives on a home network
with a television and a doorbell on it. An unauthenticated endpoint that
powerful is the failure mode the whole design exists to avoid.

The shape is: pair once, then never think about it again.

    backend:   cookie-backend pair
               → prints a short code, valid for ten minutes, single use
    frontend:  exchanges the code for a token, stores it
    thereafter: Authorization: Bearer <token>

Tokens are stored hashed, never in plaintext, so a leaked state file does not
hand over access. They can be listed and revoked by name, because "pair
another machine" and "that laptop is gone" are both things that happen.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path

#: Long enough that guessing is hopeless, short enough to read aloud.
PAIRING_CODE_WORDS = 3
PAIRING_TTL_SECONDS = 600

_WORDS = (
    "amber apple anchor basil beacon birch candle cedar cinder clover copper "
    "cotton dahlia ember fennel ginger harbour hazel indigo ivory juniper "
    "kettle lantern linen marble meadow nutmeg olive orchid pebble quartz "
    "quince rowan saffron sorrel tamarind thistle umber velvet walnut willow"
).split()


@dataclass
class Device:
    """A frontend that has been paired."""

    name: str
    token_hash: str
    created_at: float
    last_seen_at: float | None = None


class PairingError(Exception):
    """The code was wrong, expired, or already used."""


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def generate_code() -> str:
    """A human-speakable pairing code, e.g. `cedar-quartz-willow`.

    Words rather than hex because this gets read across a room, and a code
    people mistype is a code people disable.
    """
    return "-".join(secrets.choice(_WORDS) for _ in range(PAIRING_CODE_WORDS))


class DeviceStore:
    """Paired devices, persisted as JSON with restrictive permissions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._devices: dict[str, Device] = {}
        self._pending: dict[str, float] = {}  # code -> expiry
        self._load()

    # --- persistence -----------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            # A corrupt store must not lock you out of your own machine; it
            # means re-pairing, which is one command.
            return
        for entry in raw.get("devices", []):
            device = Device(**entry)
            self._devices[device.name] = device

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"devices": [asdict(d) for d in self._devices.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    # --- pairing ---------------------------------------------------------

    def begin_pairing(self, ttl: int = PAIRING_TTL_SECONDS) -> str:
        """Issue a single-use pairing code."""
        code = generate_code()
        self._pending[code] = time.time() + ttl
        return code

    def pending_codes(self) -> list[str]:
        self._expire_pending()
        return list(self._pending)

    def _expire_pending(self) -> None:
        now = time.time()
        self._pending = {c: e for c, e in self._pending.items() if e > now}

    def complete_pairing(self, code: str, device_name: str) -> str:
        """Exchange a code for a token. The code is consumed either way."""
        self._expire_pending()
        # Constant-time comparison against each candidate: the set is tiny, and
        # a dictionary lookup on a secret is a timing oracle.
        matched = None
        for candidate in self._pending:
            if hmac.compare_digest(candidate, code):
                matched = candidate
        if matched is None:
            raise PairingError("that pairing code is not valid. Run `cookie-backend pair` again.")
        del self._pending[matched]

        token = secrets.token_urlsafe(32)
        name = device_name.strip() or "unnamed device"
        self._devices[name] = Device(
            name=name,
            token_hash=_hash(token),
            created_at=time.time(),
        )
        self._save()
        return token

    # --- use -------------------------------------------------------------

    def authenticate(self, token: str | None) -> Device | None:
        """Return the device this token belongs to, or `None`."""
        if not token:
            return None
        digest = _hash(token)
        for device in self._devices.values():
            if hmac.compare_digest(device.token_hash, digest):
                device.last_seen_at = time.time()
                return device
        return None

    def devices(self) -> list[Device]:
        return sorted(self._devices.values(), key=lambda d: d.created_at)

    def revoke(self, name: str) -> bool:
        if name not in self._devices:
            return False
        del self._devices[name]
        self._save()
        return True

    def is_empty(self) -> bool:
        return not self._devices
