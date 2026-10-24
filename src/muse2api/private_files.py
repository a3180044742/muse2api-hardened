"""Helpers for files that contain credentials or private request metadata."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:access_token|refresh_token|id_token|api[_-]?key|"
    r"x-api-key|x-goog-api-key|cookie|set-cookie|password|"
    r"secret|client_secret|session(?:[_-]?id)?|jwt|signature|sig|token)[\"']?\s*[:=]\s*)"
    r"([\"']?)([^\"'\s,;&}\]]+)\2"
)
_QUOTED_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:access_token|refresh_token|id_token|api[_-]?key|"
    r"x-api-key|x-goog-api-key|authorization|cookie|set-cookie|password|"
    r"secret|client_secret|session(?:[_-]?id)?|jwt|signature|sig|token)[\"']?\s*[:=]\s*)"
    r"([\"'])(.*?)\2"
)
_AUTH_HEADER = re.compile(
    r"(?i)(\b(?:authorization|proxy-authorization)\s*:\s*)"
    r"(Bearer|Basic|Token|Api-Key|OAuth|Digest)\s+([^\r\n,;&}\]]+)"
)
_AUTH_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:authorization|proxy-authorization)\s*=\s*)"
    r"(?:(Bearer|Basic|Token|Api-Key|OAuth|Digest)\s+)?([^\"'\s,;&}\]]+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:api[_-]?key|x-api-key|x-goog-api-key|access[_-]?token|"
    r"refresh[_-]?token|id[_-]?token|client[_-]?secret|authorization|cookie|"
    r"session(?:[_-]?id)?|jwt|signature|sig|token|key)=)[^&#\s]+"
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b")
_TOKEN_PREFIX = re.compile(
    r"\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{12,}|m2a-[A-Za-z0-9_-]{16,}|"
    r"AIza[A-Za-z0-9_-]{20,})\b"
)


def redact_secrets(value: str, *, limit: int | None = None) -> str:
    """Remove common credentials from diagnostic text before it is persisted."""
    value = _AUTH_HEADER.sub(
        lambda match: f"{match.group(1)}{match.group(2)} [REDACTED]",
        value,
    )
    value = _AUTH_ASSIGNMENT.sub(
        lambda match: match.group(1)
        + (match.group(2) + " " if match.group(2) else "")
        + "[REDACTED]",
        value,
    )
    value = _QUOTED_SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]\2", value)
    value = _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]\2", value)
    value = _QUERY_SECRET.sub(r"\1[REDACTED]", value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    value = _JWT.sub("[JWT_REDACTED]", value)
    value = _TOKEN_PREFIX.sub("[REDACTED]", value)
    return value[:limit] if limit is not None else value


def secure_file_mode(path: Path) -> None:
    """Restrict an existing private file to its owner (mode 0600 on POSIX)."""
    if path.is_symlink():
        raise OSError(f"refusing to change permissions on symlink: {path}")
    if path.exists():
        path.chmod(0o600)


def atomic_private_write(path: Path, content: str, *, prefix: str) -> None:
    """Atomically replace a private text file without creating a public-mode temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        secure_file_mode(path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
