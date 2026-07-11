"""Unit tests for ChatController with a mocked LLM."""
import os
import tempfile
import unittest
from unittest import mock

from listening_ai import Settings, configure, set_store
from listening_ai.controller import ChatController
from listening_ai.store import JsonFileStore
from listening_ai.tools import ToolRegistry


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmp.name, "db.json")
        self.store = JsonFileStore(db_path=db_path, token_expiry_seconds=3600)
        set_store(self.store)
        configure(Settings(
            store_backend="json",
            data_dir=self.tmp.name,
            db_path=db_path,
            openrouter_api_key="test-key",
            openrouter_tools_model="test/model",
        ))
        self.user = self.store.create_user("ctrl", "pw")
        self.uid = self.user["user_id"]

        self.reg = ToolRegistry()
        self.reg.register(
            "ping",
            "Return pong",
            {"type": "object", "properties": {}, "required": []},
            lambda user_id: "pong",
        )
        self.controller = ChatController(tool_registry=self.reg, max_steps=4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_handle_message_persists_history(self):
        sid = self.controller.start_session(self.uid)

        with mock.patch("listening_ai.controller.llm.call_llm_with_tools") as m:
            m.return_value = {
                "stop_reason": "end_turn",
                "text": "Hello back",
                "tool_calls": [],
                "raw_message": {"role": "assistant", "content": "Hello back"},
                "model_used": "test/model",
                "error": None,
            }
            result = self.controller.handle_message(self.uid, sid, "hi")

        self.assertEqual(result["response"], "Hello back")
        self.assertEqual(result["session_id"], sid)
        history = self.controller.get_history(sid)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["content"], "Hello back")

    def test_run_loop_executes_tools(self):
        with mock.patch("listening_ai.controller.llm.call_llm_with_tools") as m:
            m.side_effect = [
                {
                    "stop_reason": "tool_calls",
                    "text": None,
                    "tool_calls": [{"id": "1", "name": "ping", "input": {}}],
                    "raw_message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "1",
                            "type": "function",
                            "function": {"name": "ping", "arguments": "{}"},
                        }],
                    },
                    "model_used": "test/model",
                    "error": None,
                },
                {
                    "stop_reason": "end_turn",
                    "text": "Tool said pong",
                    "tool_calls": [],
                    "raw_message": {"role": "assistant", "content": "Tool said pong"},
                    "model_used": "test/model",
                    "error": None,
                },
            ]
            text, model, log = self.controller.run_loop(
                [{"role": "user", "content": "ping please"}],
                self.uid,
            )

        self.assertEqual(text, "Tool said pong")
        self.assertEqual(model, "test/model")
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["name"], "ping")
        self.assertEqual(log[0]["result"], "pong")
        self.assertEqual(m.call_count, 2)

    def test_run_loop_llm_error(self):
        with mock.patch("listening_ai.controller.llm.call_llm_with_tools") as m:
            m.return_value = {
                "stop_reason": "end_turn",
                "text": None,
                "tool_calls": [],
                "raw_message": None,
                "model_used": "test/model",
                "error": "missing_api_key",
            }
            text, _model, log = self.controller.run_loop(
                [{"role": "user", "content": "hi"}],
                self.uid,
            )
        self.assertIn("trouble", text.lower())
        self.assertEqual(log, [])

    def test_mismatched_session_raises(self):
        sid = self.controller.start_session(self.uid)
        other = self.store.create_user("other", "pw")
        with self.assertRaises(ValueError):
            self.controller.handle_message(other["user_id"], sid, "hi")


if __name__ == "__main__":
    unittest.main()
