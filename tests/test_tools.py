"""Unit tests for ToolRegistry and default tools."""
import os
import tempfile
import unittest

from listening_ai import Settings, configure, set_store
from listening_ai.store import JsonFileStore
from listening_ai.tools import ToolRegistry, default_registry


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmp.name, "db.json")
        self.store = JsonFileStore(db_path=db_path, token_expiry_seconds=3600)
        set_store(self.store)
        configure(Settings(store_backend="json", data_dir=self.tmp.name, db_path=db_path))
        self.user = self.store.create_user("tooluser", "secret")
        self.uid = self.user["user_id"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_register_and_execute(self):
        reg = ToolRegistry()
        reg.register(
            "add",
            "Add two numbers",
            {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
            lambda user_id, a, b: a + b,
        )
        self.assertTrue(reg.has_tools())
        self.assertEqual(reg.names(), ["add"])
        self.assertEqual(reg.execute("add", {"a": 2, "b": 3}, self.uid), "5")
        schemas = reg.schemas()
        self.assertEqual(schemas[0]["function"]["name"], "add")

    def test_unknown_tool(self):
        reg = ToolRegistry()
        out = reg.execute("nope", {}, self.uid)
        self.assertIn("unknown tool", out)

    def test_bad_arguments(self):
        reg = ToolRegistry()
        reg.register(
            "needs_x",
            "Needs x",
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            lambda user_id, x: x,
        )
        out = reg.execute("needs_x", {}, self.uid)
        self.assertIn("bad arguments", out)

    def test_handler_exception(self):
        reg = ToolRegistry()

        def boom(user_id):
            raise RuntimeError("kaboom")

        reg.register("boom", "Boom", {"type": "object", "properties": {}}, boom)
        out = reg.execute("boom", {}, self.uid)
        self.assertIn("Error executing boom", out)
        self.assertIn("kaboom", out)

    def test_dict_result_json_encoded(self):
        reg = ToolRegistry()
        reg.register(
            "meta",
            "Meta",
            {"type": "object", "properties": {}},
            lambda user_id: {"ok": True, "user": user_id},
        )
        out = reg.execute("meta", {}, self.uid)
        self.assertIn('"ok": true', out)
        self.assertIn(self.uid, out)

    def test_default_registry_profile_roundtrip(self):
        reg = default_registry()
        self.assertIn("get_profile", reg.names())
        self.assertIn("update_profile", reg.names())

        empty = reg.execute("get_profile", {}, self.uid)
        self.assertIn("empty", empty.lower())

        saved = reg.execute("update_profile", {"field": "city", "value": "Austin"}, self.uid)
        self.assertIn("Austin", saved)
        profile = reg.execute("get_profile", {}, self.uid)
        self.assertIn("Austin", profile)

    def test_default_registry_inbox(self):
        other = self.store.create_user("other", "pw")
        reg = default_registry()
        out = reg.execute(
            "send_message",
            {"to_username": "other", "body": "hi there"},
            self.uid,
        )
        self.assertIn("Message sent", out)
        msgs = reg.execute("list_messages", {}, other["user_id"])
        self.assertIn("hi there", msgs)

    def test_register_from_openai_schema(self):
        reg = ToolRegistry()
        reg.register_from_openai_schema(
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo",
                    "parameters": {
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                        "required": ["msg"],
                    },
                },
            },
            lambda user_id, msg: msg,
        )
        self.assertEqual(reg.execute("echo", {"msg": "yo"}, self.uid), "yo")


if __name__ == "__main__":
    unittest.main()
