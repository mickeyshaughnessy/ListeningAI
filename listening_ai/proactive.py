"""
Proactive / unprompted message policy.

Hosts (e.g. GreenDial GET /Doc) use this to decide whether an agent may
speak without a user prompt. Pure gate logic — no LLM, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        s = ts.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None


def _utc_now(now: Optional[datetime] = None) -> datetime:
    if now is None:
        return datetime.utcnow()
    if now.tzinfo is not None:
        return now.astimezone(timezone.utc).replace(tzinfo=None)
    return now


@dataclass
class ProactivePolicy:
    """
    Rate limits for unprompted assistant messages.

    Defaults match GreenDial Doc: max 2/day, 6h min interval, 30m quiet
    after recent chat activity.
    """

    min_interval_hours: float = 6.0
    max_per_day: int = 2
    quiet_after_activity_minutes: float = 30.0
    require_notifications_enabled: bool = True
    require_profile_or_transcript: bool = True

    def evaluate(
        self,
        *,
        last_sent_at: Optional[str] = None,
        sent_dates: Optional[Sequence[str]] = None,
        last_activity_at: Optional[str] = None,
        notifications_enabled: bool = True,
        has_profile: bool = False,
        has_transcript: bool = False,
        force: bool = False,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Return a decision dict:

          allowed: bool
          reason: str | None   — gate that blocked (or None if allowed)
          next_eligible_at: str | None  — ISO UTC when next attempt may pass
        """
        now_dt = _utc_now(now)
        today = now_dt.strftime("%Y-%m-%d")

        if force:
            return {
                "allowed": True,
                "reason": None,
                "next_eligible_at": None,
            }

        if self.require_notifications_enabled and not notifications_enabled:
            return {
                "allowed": False,
                "reason": "notifications_disabled",
                "next_eligible_at": None,
            }

        if self.require_profile_or_transcript and not (has_profile or has_transcript):
            return {
                "allowed": False,
                "reason": "empty_context",
                "next_eligible_at": None,
            }

        # Daily cap
        dates = [d for d in (sent_dates or []) if isinstance(d, str)]
        today_count = sum(1 for d in dates if d[:10] == today)
        if today_count >= self.max_per_day:
            # Next eligible: start of next UTC day
            tomorrow = (now_dt + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return {
                "allowed": False,
                "reason": "daily_cap",
                "next_eligible_at": tomorrow.isoformat(),
            }

        # Min interval since last unprompted send
        last_sent = _parse_iso(last_sent_at)
        if last_sent is not None:
            min_delta = timedelta(hours=self.min_interval_hours)
            earliest = last_sent + min_delta
            if now_dt < earliest:
                return {
                    "allowed": False,
                    "reason": "min_interval",
                    "next_eligible_at": earliest.isoformat(),
                }

        # Quiet window after any chat activity
        last_act = _parse_iso(last_activity_at)
        if last_act is not None and self.quiet_after_activity_minutes > 0:
            quiet_until = last_act + timedelta(minutes=self.quiet_after_activity_minutes)
            if now_dt < quiet_until:
                return {
                    "allowed": False,
                    "reason": "recent_activity",
                    "next_eligible_at": quiet_until.isoformat(),
                }

        # Next eligible after a successful send would be min_interval from now
        next_after = now_dt + timedelta(hours=self.min_interval_hours)
        return {
            "allowed": True,
            "reason": None,
            "next_eligible_at": next_after.isoformat(),
        }

    def record_send(
        self,
        state: Optional[Dict[str, Any]] = None,
        *,
        message_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Update a host-owned state dict after a successful unprompted send.

        Expected shape (host may store under user['doc_unprompted']):
          last_sent_at, sent_dates (list of YYYY-MM-DD), last_message_id
        """
        now_dt = _utc_now(now)
        out = dict(state or {})
        out["last_sent_at"] = now_dt.isoformat()
        today = now_dt.strftime("%Y-%m-%d")
        dates = list(out.get("sent_dates") or [])
        dates.append(today)
        # Keep last 14 day stamps (enough for daily-cap math)
        out["sent_dates"] = dates[-14:]
        if message_id:
            out["last_message_id"] = message_id
        return out


# GreenDial Doc defaults as a named instance
DOC_DEFAULT_POLICY = ProactivePolicy(
    min_interval_hours=6.0,
    max_per_day=2,
    quiet_after_activity_minutes=30.0,
    require_notifications_enabled=True,
    require_profile_or_transcript=True,
)


def is_nothing_message(text: Optional[str]) -> bool:
    """True if the model declined to send anything useful."""
    if not text or not str(text).strip():
        return True
    t = str(text).strip().lower()
    sentinels = (
        "nothing",
        "no message",
        "skip",
        "n/a",
        "(none)",
        "none",
        "no-op",
        "noop",
    )
    if t in sentinels:
        return True
    if t.startswith("nothing to say") or t.startswith("no unprompted"):
        return True
    return False
