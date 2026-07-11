"""Unit tests for JsonFileStore (no network, no boto3 required)."""
import os
import tempfile
import unittest

from listening_ai import Settings, configure, set_store
from listening_ai.store import JsonFileStore, get_store


class JsonFileStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmp.name, "db.json")
        self.store = JsonFileStore(db_path=db_path, token_expiry_seconds=3600)
        set_store(self.store)
        configure(Settings(store_backend="json", data_dir=self.tmp.name, db_path=db_path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_and_login(self):
        user = self.store.create_user("alice", "secret")
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "alice")
        self.assertIn("password_hash", user)

        # duplicate username rejected
        self.assertIsNone(self.store.create_user("Alice", "other"))

        logged = self.store.verify_login("alice", "secret")
        self.assertIsNotNone(logged)
        self.assertEqual(logged["user_id"], user["user_id"])
        self.assertIsNone(self.store.verify_login("alice", "wrong"))

    def test_session(self):
        user = self.store.create_user("bob", "pw")
        token = self.store.create_session(user["user_id"])
        self.assertEqual(self.store.get_user_id_for_token(token), user["user_id"])
        self.assertIsNone(self.store.get_user_id_for_token("nope"))

    def test_profile_and_settings(self):
        user = self.store.create_user("carol", "pw")
        uid = user["user_id"]
        self.store.update_profile(uid, {"name": "Carol", "city": "Denver"})
        self.assertEqual(self.store.get_profile(uid)["name"], "Carol")
        self.store.update_settings(uid, {"tone": "concise"})
        self.assertEqual(self.store.get_settings(uid)["tone"], "concise")

    def test_inbox_and_notifications(self):
        a = self.store.create_user("dave", "pw")
        b = self.store.create_user("erin", "pw")
        msg = self.store.add_message(b["user_id"], "dave", "hello")
        msgs = self.store.list_messages(b["user_id"])
        self.assertEqual(len(msgs), 1)
        self.store.mark_message_read(b["user_id"], msg["id"])
        self.assertTrue(self.store.list_messages(b["user_id"])[0]["read"])

        n = self.store.add_notification(a["user_id"], "ping")
        self.assertEqual(len(self.store.list_notifications(a["user_id"])), 1)
        self.store.mark_notification_read(a["user_id"], n["id"])
        self.assertTrue(self.store.delete_notification(a["user_id"], n["id"]))

    def test_chat_session(self):
        user = self.store.create_user("frank", "pw")
        sid = self.store.create_chat_session(user["user_id"], agent_id="doc")
        self.store.append_chat_messages(sid, [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        chat = self.store.get_chat_session(sid)
        self.assertEqual(len(chat["messages"]), 2)
        sessions = self.store.list_chat_sessions(user["user_id"])
        self.assertEqual(len(sessions), 1)

    def test_module_proxies(self):
        # ensure listening_ai.store.* proxies hit the installed store
        user = get_store().create_user("gina", "pw")
        from listening_ai import store as store_mod
        self.assertEqual(store_mod.get_user(user["user_id"])["username"], "gina")


if __name__ == "__main__":
    unittest.main()
