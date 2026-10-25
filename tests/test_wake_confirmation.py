"""Focused regression tests for session renewal and VM wake scheduling."""

from __future__ import annotations

import time

import httpx
import pytest

from muse2api.accounts import Account, AccountPool, JsonAccountStore
from muse2api.accounts.pool import WAKE_BLOCK_SECONDS, note_wake
from muse2api.drivers.mock import MockDriver
from muse2api.services.gateway import Gateway
from muse2api.upstream import muse

COOKIES = {
    "hatch_sess": "s1",
    "hatch_gw": "g1",
    "hatch_vml": "v1",
    "hatch_native_auth_device": "d1",
}


def test_current_session_cookie_set_does_not_require_optional_gateway_cookie():
    cookies = {key: value for key, value in COOKIES.items() if key != "hatch_gw"}
    assert muse.missing_session_cookies(cookies) == []


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


async def test_never_running_is_not_confirmed_and_stops_at_deadline(scripted_upstream):
    scripted_upstream["sessions"] = [_session("SUSPENDED")]
    clock = FakeClock()
    info = await muse.renew_session(
        dict(COOKIES), confirm_s=6, poll_s=2, sleep=clock.sleep, clock=clock,
    )
    assert info.meta["vm_wake_ok"] is False
    assert info.meta["vm_state_after"] == "SUSPENDED"
    assert clock.now <= 6
    assert scripted_upstream["gets"] == 5


@pytest.mark.parametrize("failure", [
    httpx.Response(503, text="busy"),
    httpx.ConnectError("connection reset"),
])
async def test_transient_confirmation_errors_leave_wake_unknown(scripted_upstream, failure):
    scripted_upstream["sessions"] = [_session("SUSPENDED"), failure]
    info = await muse.renew_session(dict(COOKIES), confirm_s=4, poll_s=2)
    assert info.ok
    assert info.meta["vm_wake_ok"] is None
    assert info.meta["vm_state_after"] is None


async def test_confirmation_continues_after_error_until_running(scripted_upstream):
    scripted_upstream["sessions"] = [
        _session("SUSPENDED"), httpx.Response(502), _session("RUNNING"),
    ]
    info = await muse.renew_session(dict(COOKIES), confirm_s=10, poll_s=2)
    assert info.meta["vm_wake_ok"] is True
    assert info.meta["vm_state_after"] == "RUNNING"


async def test_wake_http_status_does_not_override_confirmed_running(scripted_upstream):
    scripted_upstream["sessions"] = [_session("RUNNING")]
    scripted_upstream["wake"] = httpx.Response(403, json={"error": "already running"})
    info = await muse.renew_session(dict(COOKIES), confirm_s=20)
    assert info.meta["vm_wake_http"] == "403"
    assert info.meta["vm_wake_ok"] is True


async def test_rejected_wake_on_stopped_vm_is_not_confirmed(scripted_upstream):
    scripted_upstream["sessions"] = [_session("SUSPENDED")]
    scripted_upstream["wake"] = httpx.Response(403, json={"error": "forbidden"})
    info = await muse.renew_session(dict(COOKIES), confirm_s=4, poll_s=2)
    assert info.meta["vm_wake_ok"] is False
    assert info.meta["vm_state_after"] == "SUSPENDED"


async def test_no_wake_is_attempted_without_a_vm(scripted_upstream):
    scripted_upstream["sessions"] = [httpx.Response(200, json={"status": "assigned"})]
    info = await muse.renew_session(dict(COOKIES))
    assert "vm_wake_ok" not in info.meta
    assert scripted_upstream["posts"] == 0


async def test_video_wake_block_does_not_replace_a_longer_block(tmp_path):
    pool = AccountPool(JsonAccountStore(tmp_path / "accounts.json"), acquire_timeout=0.2)
    account = Account(id="a0", cookies=dict(COOKIES))
    longer = time.time() + 12 * 3600
    account.meta["kind_blocked_until"] = {"video": longer}
    await pool.upsert(account)
    note_wake(account, {"vm_wake_ok": False})
    note_wake(account, {"vm_wake_ok": True})
    assert account.meta["kind_blocked_until"]["video"] == longer


async def test_renewal_without_wake_preserves_existing_block(tmp_path):
    pool = AccountPool(JsonAccountStore(tmp_path / "accounts.json"), acquire_timeout=0.2)
    account = Account(id="a0", cookies=dict(COOKIES))
    await pool.upsert(account)
    note_wake(account, {"vm_wake_ok": False})
    before = dict(account.meta["kind_blocked_until"])
    note_wake(account, {"session_status": "assigned"})
    assert account.meta["kind_blocked_until"] == before
    assert time.time() + WAKE_BLOCK_SECONDS - 5 < before["video"]


async def test_video_routing_falls_back_when_every_account_is_blocked(tmp_path):
    pool = AccountPool(JsonAccountStore(tmp_path / "accounts.json"), acquire_timeout=0.2)
    for account_id in ("a0", "a1"):
        account = Account(id=account_id, cookies=dict(COOKIES))
        note_wake(account, {"vm_wake_ok": False})
        await pool.upsert(account)
    lease = await pool.acquire(kind="video")
    assert lease.id in {"a0", "a1"}
    await pool.release(lease)


async def test_gateway_routes_video_around_an_unconfirmed_account(tmp_path):
    pool = AccountPool(JsonAccountStore(tmp_path / "accounts.json"), acquire_timeout=0.2)
    for account_id in ("a0", "a1"):
        await pool.upsert(Account(id=account_id, cookies=dict(COOKIES)))
    note_wake(pool.get("a0"), {"vm_wake_ok": False})
    gateway = Gateway(pool, MockDriver())

    async def which(account: Account) -> str:
        return account.id

    for _ in range(3):
        assert await gateway._run(which, kind="video") == "a1"
    assert {await gateway._run(which) for _ in range(4)} == {"a0", "a1"}


async def test_admin_reports_wake_state_and_reset_clears_block(client, admin):
    response = client.post(
        "/admin/accounts", headers=admin,
        json={"cookies": {"hatch_sess": "fixture-session"}},
    )
    account_id = response.json()["account"]["id"]
    pool = client.app.state.services.pool
    account = pool.get(account_id)
    account.meta.update({"vm_wake_ok": False, "vm_state_after": "SUSPENDED"})
    note_wake(account, account.meta)
    listed = next(
        item["meta"] for item in client.get("/admin/accounts", headers=admin).json()["data"]
        if item["id"] == account_id
    )
    assert listed["vm_wake_ok"] is False
    assert "kind_blocked_until" not in listed
    assert account.meta["kind_blocked_until"]["video"] > time.time()
    client.post(f"/admin/accounts/{account_id}/reset", headers=admin)
    assert "kind_blocked_until" not in pool.get(account_id).meta

