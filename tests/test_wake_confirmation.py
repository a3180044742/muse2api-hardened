"""Focused regression tests for session renewal and VM wake scheduling."""

from __future__ import annotations

import httpx
import pytest

from muse2api.accounts import Account, AccountPool, JsonAccountStore
from muse2api.accounts.pool import note_wake
from muse2api.upstream import muse

COOKIES = {
    "hatch_sess": "s1",
    "hatch_gw": "g1",
    "hatch_vml": "v1",
    "hatch_native_auth_device": "d1",
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _session(state: str, **headers: str) -> httpx.Response:
    return httpx.Response(200, json={"status": "assigned", "vm_id": "vm-1",
                                    "vm_state": state}, headers=headers)


@pytest.fixture
def scripted_upstream(monkeypatch: pytest.MonkeyPatch):
    real = httpx.AsyncClient
    state: dict[str, object] = {
        "sessions": [], "wake": httpx.Response(200), "gets": 0, "posts": 0,
        "get_headers": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            observed = state["get_headers"]
            assert isinstance(observed, list)
            observed.append({name: request.headers.get(name) for name in
                             ("user-agent", "origin", "referer", "accept", "cookie")})
        if request.method == "POST":
            state["posts"] = int(state["posts"]) + 1
            wake = state["wake"]
            if isinstance(wake, Exception):
                raise wake
            assert isinstance(wake, httpx.Response)
            return wake
        sessions = state["sessions"]
        assert isinstance(sessions, list) and sessions
        index = min(int(state["gets"]), len(sessions) - 1)
        state["gets"] = int(state["gets"]) + 1
        response = sessions[index]
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, httpx.Response)
        return response

    monkeypatch.setattr(
        muse.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    return state


async def test_wake_confirmation_polls_state_and_merges_rotated_cookie(scripted_upstream):
    scripted_upstream["sessions"] = [
        _session("SUSPENDED"),
        _session("STARTING"),
        _session(
            "RUNNING",
            **{"set-cookie": "hatch_vml=v2; Path=/; Expires=Mon, 21 Oct 2030 07:28:00 GMT"},
        ),
    ]
    clock = FakeClock()
    info = await muse.renew_session(
        dict(COOKIES), confirm_s=20, poll_s=2, sleep=clock.sleep, clock=clock,
    )
    assert info.ok
    assert info.meta["vm_state_after"] == "RUNNING"
    assert info.meta["vm_wake_ok"] is True
    assert info.cookies["hatch_vml"] == "v2"
    assert info.cookie_expires["hatch_vml"] == 1_918_798_080
    assert scripted_upstream["posts"] == 1
    assert scripted_upstream["gets"] == 3
    observed_headers = scripted_upstream["get_headers"]
    assert isinstance(observed_headers, list)
    assert all(headers["user-agent"] == muse.USER_AGENT for headers in observed_headers)
    assert all(headers["origin"] == muse.SITE for headers in observed_headers)
    assert all(headers["referer"] == muse.NEW_THREAD_URL for headers in observed_headers)
    assert all(headers["accept"] == "application/json" for headers in observed_headers)
    assert clock.sleeps == [2]


async def test_wake_response_code_is_not_used_as_readiness(scripted_upstream):
    scripted_upstream["sessions"] = [_session("RUNNING")]
    scripted_upstream["wake"] = httpx.Response(403, json={"error": "already running"})
    info = await muse.renew_session(dict(COOKIES), confirm_s=20)
    assert info.meta["vm_wake_http"] == "403"
    assert info.meta["vm_wake_ok"] is True
    assert info.meta["vm_state_after"] == "RUNNING"


async def test_unconfirmed_wake_defers_video_only(tmp_path):
    pool = AccountPool(JsonAccountStore(tmp_path / "accounts.json"), acquire_timeout=0.2)
    await pool.upsert(Account(id="a0", cookies=dict(COOKIES)))
    await pool.upsert(Account(id="a1", cookies=dict(COOKIES)))
    note_wake(pool.get("a0"), {"vm_wake_ok": False, "vm_state_after": "SUSPENDED"})

    video = await pool.acquire(kind="video")
    assert video.id == "a1"
    await pool.release(video)
    general = await pool.acquire(exclude={"a1"})
    assert general.id == "a0"
    await pool.release(general)


async def test_wake_block_never_makes_video_unavailable(tmp_path):
    pool = AccountPool(JsonAccountStore(tmp_path / "accounts.json"), acquire_timeout=0.2)
    for account_id in ("a0", "a1"):
        account = Account(id=account_id, cookies=dict(COOKIES))
        note_wake(account, {"vm_wake_ok": False})
        await pool.upsert(account)
    account = await pool.acquire(kind="video")
    assert account.id in {"a0", "a1"}
    await pool.release(account)


async def test_confirmed_wake_clears_only_own_block(tmp_path):
    pool = AccountPool(JsonAccountStore(tmp_path / "accounts.json"), acquire_timeout=0.2)
    account = Account(id="a0", cookies=dict(COOKIES))
    await pool.upsert(account)
    note_wake(account, {"vm_wake_ok": False})
    assert "video" in account.meta["kind_blocked_until"]
    note_wake(account, {"vm_wake_ok": True, "vm_state_after": "RUNNING"})
    assert "video" not in account.meta["kind_blocked_until"]


def test_public_account_exposes_wake_status_but_not_vm_id():
    account = Account(
        id="a0",
        meta={"vm_id": "private-vm-id", "vm_wake_ok": True,
              "vm_state_after": "RUNNING"},
    )
    public = account.public()
    assert public["meta"]["vm_wake_ok"] is True
    assert public["meta"]["vm_state_after"] == "RUNNING"
    assert "vm_id" not in public["meta"]


def test_wake_window_is_finite_and_bounded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MUSE2API_WAKE_CONFIRM_S", "inf")
    assert muse.wake_confirm_seconds() == muse.WAKE_CONFIRM_SECONDS
    monkeypatch.setenv("MUSE2API_WAKE_CONFIRM_S", "9999")
    assert muse.wake_confirm_seconds() == muse.MAX_WAKE_CONFIRM_SECONDS


async def test_malformed_kind_block_metadata_is_ignored(tmp_path):
    pool = AccountPool(JsonAccountStore(tmp_path / "accounts.json"), acquire_timeout=0.2)
    account = Account(id="a0", cookies=dict(COOKIES), meta={"kind_blocked_until": "bad"})
    await pool.upsert(account)
    note_wake(account, {"vm_wake_ok": False})
    lease = await pool.acquire(kind="video")
    assert lease.id == "a0"
    await pool.release(lease)


def test_malformed_wake_outcome_does_not_clear_or_create_block():
    account = Account(id="a0", meta={"kind_blocked_until": {"video": 9999999999}})
    note_wake(account, {"vm_wake_ok": "false"})
    assert account.meta["kind_blocked_until"]["video"] == 9999999999

