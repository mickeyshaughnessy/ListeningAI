"""HTTP-level tests for the Flask blueprint (no network / no real LLM)."""
import os
import tempfile
import unittest
from unittest import mock

from flask import Flask

from listening_ai import Settings, configure_app, create_blueprint, default_registry
from listening_ai.store import JsonFileStore


class ApiBlueprintTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmp.name, "db.json")
        store = JsonFileStore(db_path=db_path, token_expiry_seconds=3600)
        configure_app(
            Settings(store_backend="json", data_dir=self.tmp.name, db_path=db_path, openrouter_api_key="k"),
            store=store,
        )
        app = Flask(__name__)
        app.register_blueprint(create_blueprint(tool_registry=default_registry()))
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def _register(self, username="alice", password="secret"):
        r = self.client.post("/register", json={"username": username, "password": password})
        self.assertEqual(r.status_code, 201, r.get_data(as_text=True))
        return r.get_json()

    def test_ping(self):
        r = self.client.get("/ping")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_register_login_account(self):
        data = self._register()
        self.assertIn("token", data)
        token = data["token"]

        r = self.client.post("/login", json={"username": "alice", "password": "secret"})
        self.assertEqual(r.status_code, 200)

        r = self.client.get("/account", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["username"], "alice")

        r = self.client.get("/account")
        self.assertEqual(r.status_code, 401)

    def test_register_validation(self):
        r = self.client.post("/register", json={"username": "", "password": "x"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/register", json={"username": "x", "password": "ab"})
        self.assertEqual(r.status_code, 400)
        self._register("bob", "secret")
        r = self.client.post("/register", json={"username": "Bob", "password": "other"})
        self.assertEqual(r.status_code, 409)

    def test_x_session_token_header(self):
        data = self._register("carol", "secret")
        r = self.client.get("/account", headers={"X-Session-Token": data["token"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["username"], "carol")

    def test_profile_and_settings(self):
        data = self._register("dave", "secret")
        headers = {"Authorization": f"Bearer {data['token']}"}

        r = self.client.put("/profile", headers=headers, json={"updates": {"name": "Dave"}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["profile"]["name"], "Dave")

        r = self.client.get("/profile", headers=headers)
        self.assertEqual(r.get_json()["profile"]["name"], "Dave")

        r = self.client.put("/settings", headers=headers, json={"tone": "brief"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["settings"]["tone"], "brief")

    def test_inbox_and_notifications(self):
        a = self._register("erin", "secret")
        b = self._register("frank", "secret")
        a_headers = {"Authorization": f"Bearer {a['token']}"}
        b_headers = {"Authorization": f"Bearer {b['token']}"}

        r = self.client.post(
            "/inbox",
            headers=a_headers,
            json={"to_username": "frank", "body": "hello frank"},
        )
        self.assertEqual(r.status_code, 201)

        r = self.client.get("/inbox", headers=b_headers)
        msgs = r.get_json()["messages"]
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["body"], "hello frank")

        r = self.client.post(f"/inbox/{msgs[0]['id']}/read", headers=b_headers)
        self.assertTrue(r.get_json()["message"]["read"])

        r = self.client.post("/notifications", headers=a_headers, json={"text": "nudge"})
        self.assertEqual(r.status_code, 201)
        nid = r.get_json()["notification"]["id"]
        r = self.client.post(f"/notifications/{nid}/read", headers=a_headers)
        self.assertTrue(r.get_json()["notification"]["read"])
        r = self.client.delete(f"/notifications/{nid}", headers=a_headers)
        self.assertTrue(r.get_json()["deleted"])

    def test_chat_with_mocked_llm(self):
        data = self._register("gina", "secret")
        headers = {"Authorization": f"Bearer {data['token']}"}

        with mock.patch("listening_ai.controller.llm.call_llm_with_tools") as m:
            m.return_value = {
                "stop_reason": "end_turn",
                "text": "Noted.",
                "tool_calls": [],
                "raw_message": {"role": "assistant", "content": "Noted."},
                "model_used": "mock/model",
                "error": None,
            }
            r = self.client.post("/chat", headers=headers, json={"message": "remember teal"})
            self.assertEqual(r.status_code, 200)
            body = r.get_json()
            self.assertEqual(body["response"], "Noted.")
            self.assertTrue(body["session_id"])
            sid = body["session_id"]

        r = self.client.get("/chat/history", headers=headers, query_string={"session_id": sid})
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(len(r.get_json()["messages"]), 2)

        r = self.client.get("/chat/sessions", headers=headers)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.get_json()["sessions"]), 1)

    def test_chat_requires_message(self):
        data = self._register("hank", "secret")
        headers = {"Authorization": f"Bearer {data['token']}"}
        r = self.client.post("/chat", headers=headers, json={})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
