"""Unit tests for ProactivePolicy (no network)."""
import unittest
from datetime import datetime, timedelta

from listening_ai.proactive import (
    DOC_DEFAULT_POLICY,
    ProactivePolicy,
    is_nothing_message,
)


class ProactivePolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = ProactivePolicy(
            min_interval_hours=6.0,
            max_per_day=2,
            quiet_after_activity_minutes=30.0,
        )
        self.now = datetime(2026, 7, 13, 12, 0, 0)

    def test_allowed_with_context(self):
        d = self.policy.evaluate(
            has_profile=True,
            has_transcript=False,
            notifications_enabled=True,
            now=self.now,
        )
        self.assertTrue(d["allowed"])
        self.assertIsNone(d["reason"])

    def test_notifications_disabled(self):
        d = self.policy.evaluate(
            has_profile=True,
            notifications_enabled=False,
            now=self.now,
        )
        self.assertFalse(d["allowed"])
        self.assertEqual(d["reason"], "notifications_disabled")

    def test_empty_context(self):
        d = self.policy.evaluate(
            has_profile=False,
            has_transcript=False,
            now=self.now,
        )
        self.assertFalse(d["allowed"])
        self.assertEqual(d["reason"], "empty_context")

    def test_min_interval(self):
        last = (self.now - timedelta(hours=2)).isoformat()
        d = self.policy.evaluate(
            last_sent_at=last,
            has_profile=True,
            now=self.now,
        )
        self.assertFalse(d["allowed"])
        self.assertEqual(d["reason"], "min_interval")
        self.assertIsNotNone(d["next_eligible_at"])

    def test_daily_cap(self):
        d = self.policy.evaluate(
            sent_dates=["2026-07-13", "2026-07-13"],
            has_transcript=True,
            now=self.now,
        )
        self.assertFalse(d["allowed"])
        self.assertEqual(d["reason"], "daily_cap")

    def test_recent_activity(self):
        act = (self.now - timedelta(minutes=10)).isoformat()
        d = self.policy.evaluate(
            last_activity_at=act,
            has_profile=True,
            now=self.now,
        )
        self.assertFalse(d["allowed"])
        self.assertEqual(d["reason"], "recent_activity")

    def test_force_bypasses_gates(self):
        d = self.policy.evaluate(
            notifications_enabled=False,
            has_profile=False,
            force=True,
            now=self.now,
        )
        self.assertTrue(d["allowed"])

    def test_record_send(self):
        state = self.policy.record_send({}, message_id="msg_1", now=self.now)
        self.assertEqual(state["last_message_id"], "msg_1")
        self.assertEqual(state["sent_dates"], ["2026-07-13"])
        self.assertTrue(state["last_sent_at"].startswith("2026-07-13"))

    def test_doc_default_policy_exported(self):
        self.assertEqual(DOC_DEFAULT_POLICY.max_per_day, 2)
        self.assertEqual(DOC_DEFAULT_POLICY.min_interval_hours, 6.0)

    def test_is_nothing_message(self):
        self.assertTrue(is_nothing_message(""))
        self.assertTrue(is_nothing_message("nothing"))
        self.assertTrue(is_nothing_message("SKIP"))
        self.assertFalse(is_nothing_message("How did sleep go last night?"))


if __name__ == "__main__":
    unittest.main()
