"""Facts about the muse.ai web app shared by all drivers.

Everything here was observed from the public web client and may change without
notice. Keep upstream-specific constants in this module so that a site update
is a one-file fix.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
import uuid
from collections.abc import Awaitable, Callable

import httpx

from ..drivers.base import SessionInfo
from ..errors import UpstreamAuthError, UpstreamError

SITE = "https://muse.ai"
NEW_THREAD_URL = f"{SITE}/thread/new"
SESSION_API = f"{SITE}/api/session"
VM_WAKE_API = f"{SITE}/api/hatch/vm/wake"

# Cookies that together make up a logged-in session.  The current web client
# no longer emits ``hatch_gw`` for every account; the session endpoint remains
# the authority, so that cookie is forwarded when present but is optional.
SESSION_COOKIES = ("hatch_sess", "hatch_vml", "hatch_native_auth_device")

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)

# A wake response is only an acknowledgement.  The session endpoint is the
# source of truth for whether the workspace is actually ready.
WAKE_CONFIRM_SECONDS = 20.0
WAKE_POLL_SECONDS = 2.0
MAX_WAKE_CONFIRM_SECONDS = 120.0


def _bounded_seconds(value: object, default: float) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(seconds):
        return default
    return min(max(seconds, 0.0), MAX_WAKE_CONFIRM_SECONDS)


def cookie_header(cookies: dict[str, str]) -> str:
    return "; ".join(f"{k}={v}" for k, v in cookies.items() if v)


def missing_session_cookies(cookies: dict[str, str]) -> list[str]:
    return [name for name in SESSION_COOKIES if not cookies.get(name)]


def wake_confirm_seconds() -> float:
    """Read the bounded wake confirmation window from the environment."""
    return _bounded_seconds(os.environ.get("MUSE2API_WAKE_CONFIRM_S", WAKE_CONFIRM_SECONDS),
                            WAKE_CONFIRM_SECONDS)


def _merge_cookies(resp: httpx.Response, jar: dict[str, str], expires: dict[str, int]) -> None:
    """Merge Set-Cookie values without retaining any response text."""
    for cookie in resp.cookies.jar:
        if cookie.value:
            jar[cookie.name] = cookie.value
        if cookie.expires:
            expires[cookie.name] = int(cookie.expires)


def _json_body(resp: httpx.Response) -> dict:
    if not resp.headers.get("content-type", "").startswith("application/json"):
        return {}
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


async def renew_session(
    cookies: dict[str, str], *, wake_vm: bool = True, timeout: float = 15.0,
    confirm_s: float | None = None, poll_s: float = WAKE_POLL_SECONDS,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> SessionInfo:
    """Hit the session endpoint, which rotates short-lived cookies via Set-Cookie,
    and optionally wake the account's cloud workspace so the first request is fast.

    The wake response is not treated as proof of readiness.  When a VM id is
    present, the session endpoint is polled until it reports ``RUNNING`` or the
    bounded confirmation window expires.  Poll responses can rotate cookies too.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Origin": SITE,
        "Referer": NEW_THREAD_URL,
        "Accept": "application/json",
    }
    jar = dict(cookies)
    expires: dict[str, int] = {}
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        resp = await client.get(SESSION_API, headers={"Cookie": cookie_header(jar)})
        if resp.status_code in (401, 403):
            raise UpstreamAuthError(f"session rejected by upstream (HTTP {resp.status_code})")
        if resp.status_code >= 400:
            raise UpstreamError(f"session endpoint returned HTTP {resp.status_code}")

        _merge_cookies(resp, jar, expires)
        body = _json_body(resp)
        meta = {
            key: (str(body[source])[:128] if body.get(source) is not None else None)
            for key, source in (
                ("session_status", "status"),
                ("vm_id", "vm_id"),
                ("vm_state", "vm_state"),
            )
        }

        if wake_vm and meta["vm_id"] and meta["vm_state"] != "DISABLED":
            try:
                w = await client.post(
                    VM_WAKE_API,
                    headers={"Cookie": cookie_header(jar)},
                    json={"vm_id": meta["vm_id"], "retry_count": 0,
                          "connect_attempt_id": str(uuid.uuid4())},
                )
                meta["vm_wake_http"] = str(w.status_code)
            except httpx.HTTPError as exc:
                meta["vm_wake_http"] = type(exc).__name__
            meta["vm_state_after"] = await _confirm_running(
                client, jar, expires, timeout=timeout,
                confirm_s=wake_confirm_seconds() if confirm_s is None else confirm_s,
                poll_s=poll_s, sleep=sleep, clock=clock, request_headers=headers,
            )
            meta["vm_wake_ok"] = (
                None if meta["vm_state_after"] is None
                else meta["vm_state_after"] == "RUNNING"
            )

    return SessionInfo(ok=meta["session_status"] == "assigned", cookies=jar,
                       cookie_expires=expires, meta=meta)


async def _confirm_running(
    client: httpx.AsyncClient,
    jar: dict[str, str],
    expires: dict[str, int],
    *,
    timeout: float,
    confirm_s: float,
    poll_s: float,
    sleep: Callable[[float], Awaitable[object]],
    clock: Callable[[], float],
    request_headers: dict[str, str],
) -> str | None:
    """Poll session state while tolerating transient upstream failures."""
    window = _bounded_seconds(confirm_s, WAKE_CONFIRM_SECONDS)
    interval = max(0.01, _bounded_seconds(poll_s, WAKE_POLL_SECONDS))
    deadline = clock() + window
    state: str | None = None
    while True:
        remaining = deadline - clock()
        try:
            response = await client.get(
                SESSION_API,
                headers={**request_headers, "Cookie": cookie_header(jar)},
                timeout=max(1.0, min(timeout, max(remaining, 0.001))),
            )
        except httpx.HTTPError:
            response = None
        if response is not None and response.status_code < 400:
            _merge_cookies(response, jar, expires)
            seen = _json_body(response).get("vm_state")
            if seen is not None:
                state = str(seen)[:128]
                if state == "RUNNING":
                    return state
        if clock() + interval > deadline:
            return state
        await sleep(interval)
