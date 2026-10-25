from __future__ import annotations

import json
import os

import pytest

from muse2api.accounts import Account, JsonAccountStore
from muse2api.auth.keys import KeyStore
from muse2api.errors import UpstreamError
from muse2api.private_files import atomic_private_write, redact_secrets
from muse2api.services.tasks import TaskManager


def test_redact_secrets_covers_headers_json_and_token_shapes():
    google_key = "".join(("AIza", "012345678901234567890123456789012345678"))
    message = (
        'Authorization: Bearer abc.def; refresh_token="refresh-secret"; '
        'Authorization: OAuth oauth-secret; Authorization: Api-Key api-secret; '
        'cookie="sid=one; csrf=two" sk-abcdefghijklmnop '
        f'{google_key} "authorization":"Bearer json-secret"; authorization=plain-secret'
    )

    result = redact_secrets(message)

    for secret in ("abc.def", "oauth-secret", "api-secret", "refresh-secret", "sid=one", "csrf=two",
                   "sk-abcdefghijklmnop", google_key):
        assert secret not in result
    assert "json-secret" not in result
    assert "plain-secret" not in result
    assert '"authorization":"[REDACTED]"' in result
    assert "Bearer [REDACTED]" in result


def test_redact_secrets_covers_query_and_runtime_credential_shapes():
    message = (
        "https://gateway.invalid/v1?x-api-key=query-secret&session_id=session-secret&"
        "jwt=jwt-secret&signature=sig-secret "
        'x-goog-api-key="google-secret" session_id=runtime-secret '
        "eyJheader.payloadvalue.signaturevalue"
    )

    result = redact_secrets(message)

    for secret in (
        "query-secret", "session-secret", "jwt-secret", "sig-secret",
        "google-secret", "runtime-secret", "eyJheader.payloadvalue.signaturevalue",
    ):
        assert secret not in result
    assert "[REDACTED]" in result


def test_error_body_redacts_credentials_before_http_or_sse_serialization():
    error = UpstreamError(
        "upstream rejected Authorization: Bearer body-bearer-fixture "
        "refresh_token=body-refresh-fixture"
    )

    body = error.to_body()
    message = body["error"]["message"]

    assert "body-bearer-fixture" not in message
    assert "body-refresh-fixture" not in message
    assert "Bearer [REDACTED]" in message
    assert body["error"]["code"] == "upstream_error"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode bits")
async def test_account_and_key_stores_use_private_modes_and_harden_existing_files(tmp_path):
    account_path = tmp_path / "accounts.json"
    account_store = JsonAccountStore(account_path)
    await account_store.save([Account(label="test", cookies={"session": "secret"})])
    assert account_path.stat().st_mode & 0o777 == 0o600
    account_path.chmod(0o644)
    await account_store.load()
    assert account_path.stat().st_mode & 0o777 == 0o600

    keys_path = tmp_path / "keys.json"
    key_store = KeyStore(keys_path)
    await key_store.create("test")
    assert keys_path.stat().st_mode & 0o777 == 0o600
    keys_path.chmod(0o644)
    await KeyStore(keys_path).load()
    assert keys_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode bits")
def test_atomic_private_write_replaces_file_and_keeps_temporary_files_private(tmp_path):
    path = tmp_path / "state.json"
    atomic_private_write(path, "{}", prefix=".state.")

    assert path.read_text(encoding="utf-8") == "{}"
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.glob(".state.*.tmp")) == []


async def test_task_errors_are_redacted_before_persistence_and_logging(tmp_path, caplog):
    manager = TaskManager(tmp_path / "tasks.json")

    async def fail(_progress):
        raise RuntimeError(
            "upstream Bearer task-bearer-fixture refresh_token=task-refresh-fixture"
        )

    task = manager.submit("test", {}, fail)
    await manager._running[task.id]

    persisted = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))[0]
    error = persisted["error"]["message"]
    assert "task-bearer-fixture" not in error
    assert "task-refresh-fixture" not in error
    assert "task-bearer-fixture" not in caplog.text
    assert "task-refresh-fixture" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_compose_publishes_to_loopback_by_default():
    from pathlib import Path

    root = Path(__file__).parents[1]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    example = (root / ".env.example").read_text(encoding="utf-8")

    assert "${MUSE2API_PUBLISH_HOST:-127.0.0.1}:18610:18610" in compose
    assert "MUSE2API_PUBLISH_HOST=127.0.0.1" in example
