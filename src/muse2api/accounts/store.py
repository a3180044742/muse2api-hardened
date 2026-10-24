"""Account persistence. ``AccountStore`` is the interface; ``JsonAccountStore`` is the default.

A SQLite/Redis store can be added by implementing the same protocol.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Protocol

from ..private_files import atomic_private_write, redact_secrets, secure_file_mode
from .model import Account


class AccountStore(Protocol):
    async def load(self) -> list[Account]: ...
    async def save(self, accounts: list[Account]) -> None: ...


class JsonAccountStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    async def load(self) -> list[Account]:
        if not self.path.is_file():
            return []
        await asyncio.to_thread(secure_file_mode, self.path)
        raw = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        if not raw.strip():
            return []
        return [Account.model_validate(item) for item in json.loads(raw)]

    async def save(self, accounts: list[Account]) -> None:
        payload = json.dumps(
            [self._private_dump(a) for a in accounts], ensure_ascii=False, indent=2
        )
        async with self._lock:
            await asyncio.to_thread(self._atomic_write, payload)

    def _atomic_write(self, payload: str) -> None:
        atomic_private_write(self.path, payload, prefix=".accounts.")

    @staticmethod
    def _private_dump(account: Account) -> dict:
        data = account.model_dump(mode="json")
        if data.get("last_error"):
            data["last_error"] = redact_secrets(data["last_error"], limit=500)
        return data
