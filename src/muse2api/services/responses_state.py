"""Process-local, bounded state for the Responses continuation protocol."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class ResponseSnapshot:
    owner_id: str
    response_id: str
    model: str
    created_at: int
    expires_at: float
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    pending_calls: dict[str, dict[str, Any]] = field(default_factory=dict)


class ResponsesState:
    """TTL state scoped to an API key and capped to limit process memory use.

    State is intentionally process-local: callers must use full input history
    after a restart or when ``previous_response_id`` has expired.
    """

    def __init__(self, ttl_seconds: float = 900, max_entries: int = 2048) -> None:
        self.ttl_seconds = max(30.0, ttl_seconds)
        self.max_entries = max(16, max_entries)
        self._responses: OrderedDict[tuple[str, str], ResponseSnapshot] = OrderedDict()
        self._calls: dict[tuple[str, str], tuple[str, str]] = {}
        self._reserved_calls: set[tuple[str, str, str]] = set()
        self._lock = RLock()

    def _purge(self, now: float) -> None:
        expired = [key for key, value in self._responses.items() if value.expires_at <= now]
        for key in expired:
            self._remove(key)
        while len(self._responses) > self.max_entries:
            self._remove(next(iter(self._responses)))

    def _remove(self, key: tuple[str, str]) -> None:
        snapshot = self._responses.pop(key, None)
        if snapshot is None:
            return
        for call_id in snapshot.pending_calls:
            self._calls.pop((snapshot.owner_id, call_id), None)
            self._reserved_calls.discard((snapshot.owner_id, snapshot.response_id, call_id))

    def put(self, snapshot: ResponseSnapshot, *, now: float | None = None) -> None:
        with self._lock:
            current = time.monotonic() if now is None else now
            self._purge(current)
            key = (snapshot.owner_id, snapshot.response_id)
            self._remove(key)
            snapshot.expires_at = current + self.ttl_seconds
            self._responses[key] = snapshot
            for call_id in snapshot.pending_calls:
                self._calls[(snapshot.owner_id, call_id)] = key
            self._purge(current)

    def get(self, owner_id: str, response_id: str, *, now: float | None = None) -> ResponseSnapshot | None:
        with self._lock:
            current = time.monotonic() if now is None else now
            self._purge(current)
            key = (owner_id, response_id)
            snapshot = self._responses.get(key)
            if snapshot is not None:
                self._responses.move_to_end(key)
            return snapshot

    def get_for_call(self, owner_id: str, call_id: str, *, now: float | None = None) -> ResponseSnapshot | None:
        with self._lock:
            current = time.monotonic() if now is None else now
            self._purge(current)
            key = self._calls.get((owner_id, call_id))
            if key is None:
                return None
            snapshot = self._responses.get(key)
            if snapshot is not None:
                self._responses.move_to_end(key)
            return snapshot

    def reserve_calls(self, owner_id: str, response_id: str, call_ids: set[str],
                      *, now: float | None = None) -> bool:
        """Atomically reserve pending calls so concurrent replays are rejected."""
        with self._lock:
            current = time.monotonic() if now is None else now
            self._purge(current)
            key = (owner_id, response_id)
            snapshot = self._responses.get(key)
            if snapshot is None or not call_ids:
                return False
            reservations = {(owner_id, response_id, call_id) for call_id in call_ids}
            if any(call_id not in snapshot.pending_calls for call_id in call_ids):
                return False
            if reservations & self._reserved_calls:
                return False
            self._reserved_calls.update(reservations)
            return True

    def release_calls(self, owner_id: str, response_id: str, call_ids: set[str]) -> None:
        with self._lock:
            for call_id in call_ids:
                self._reserved_calls.discard((owner_id, response_id, call_id))

    def consume_calls(self, owner_id: str, response_id: str, call_ids: set[str],
                      *, now: float | None = None) -> None:
        """Make returned function-call IDs one-use after their outputs succeed."""
        with self._lock:
            current = time.monotonic() if now is None else now
            self._purge(current)
            snapshot = self._responses.get((owner_id, response_id))
            if snapshot is None:
                return
            for call_id in call_ids:
                snapshot.pending_calls.pop(call_id, None)
                self._calls.pop((owner_id, call_id), None)
                self._reserved_calls.discard((owner_id, response_id, call_id))
