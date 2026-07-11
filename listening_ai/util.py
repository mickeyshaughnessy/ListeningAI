"""Small shared helpers used across store backends and the demo server."""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string ending in ``Z``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
