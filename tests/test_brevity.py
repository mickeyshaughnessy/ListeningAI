"""Tests for reply brevity / listening mode."""
import unittest
from unittest import mock

from listening_ai import Settings, configure, shorten_reply
from listening_ai.brevity import (
    _heuristic_shorten,
    listening_system_suffix,
    normalize_level,
)
from listening_ai.controller import ChatController
from listening_ai.tools import ToolRegistry


class BrevityUnitTests(unittest.TestCase):
    def test_normalize_level(self):
        self.assertEqual(normalize_level("very_short"), "very_short")
        self.assertEqual(normalize_level("very short"), "very_short")
        self.assertEqual(normalize_level("SHORT"), "short")
        self.assertEqual(normalize_level("off"), "none")
        self.assertEqual(normalize_level(None), "none")

    def test_listening_suffix(self):
        self.assertIn("tool", listening_system_suffix("very_short").lower())
        self.assertEqual(listening_system_suffix("none"), "")

    def test_heuristic_very_short(self):
        long = (
            "Great question! I'd be happy to help you with that. "
            "Logged sleep as 6 hours. "
            "You might also consider a consistent bedtime routine going forward."
        )
        out = _heuristic_shorten(
            long,
            "very_short",
            [{"name": "log_health_data", "input": {"field": "sleep"}, "result": "ok"}],
        )
        self.assertIn("Logged", out)
        self.assertLess(len(out), len(long))

    def test_shorten_none_passthrough(self):
        t = "A full paragraph of advice that stays."
        self.assertEqual(shorten_reply(t, level="none", use_llm=False), t)

    def test_shorten_uses_llm_when_enabled(self):
        configure(Settings(openrouter_api_key="k", reply_brevity="very_short"))
        with mock.patch("listening_ai.brevity.llm.completion", return_value="Saved it.") as m:
            out = shorten_reply(
                "I have carefully saved your preference to the profile after reviewing.",
                level="very_short",
                tool_log=[{"name": "update_profile", "input": {"field": "x"}, "result": "ok"}],
                use_llm=True,
            )
        self.assertEqual(out, "Saved it.")
        self.assertTrue(m.called)


class ControllerBrevityTests(unittest.TestCase):
    def setUp(self):
        configure(Settings(
            openrouter_api_key="test-key",
            openrouter_tools_model="test/model",
            reply_brevity="very_short",
        ))
        self.reg = ToolRegistry()
        self.reg.register(
            "save_note",
            "Save a note",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            lambda user_id, text: f"saved:{text}",
        )
        self.controller = ChatController(
            tool_registry=self.reg,
            max_steps=4,
            reply_brevity="very_short",
        )

    def test_run_loop_shortens_and_appends_listening_prompt(self):
        long_reply = (
            "Absolutely, I've gone ahead and taken care of that for you. "
            "Saved your note about water. "
            "Please let me know if there is anything else I can assist with today!"
        )
        with mock.patch("listening_ai.controller.llm.call_llm_with_tools") as m, \
             mock.patch("listening_ai.brevity.llm.completion", return_value="Saved your water note.") as sm:
            m.side_effect = [
                {
                    "stop_reason": "tool_calls",
                    "text": None,
                    "tool_calls": [{"id": "1", "name": "save_note", "input": {"text": "water"}}],
                    "raw_message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "1",
                            "type": "function",
                            "function": {"name": "save_note", "arguments": '{"text":"water"}'},
                        }],
                    },
                    "model_used": "test/model",
                    "error": None,
                },
                {
                    "stop_reason": "end_turn",
                    "text": long_reply,
                    "tool_calls": [],
                    "raw_message": {"role": "assistant", "content": long_reply},
                    "model_used": "test/model",
                    "error": None,
                },
            ]
            text, _model, log = self.controller.run_loop(
                [{"role": "user", "content": "remember I drank water"}],
                "user_1",
            )

        self.assertEqual(text, "Saved your water note.")
        self.assertEqual(log[0]["name"], "save_note")
        # Listening directive injected into system prompt of tool loop
        first_sys = m.call_args_list[0].kwargs.get("system_prompt") or ""
        self.assertIn("Listening mode", first_sys)
        self.assertTrue(sm.called)


if __name__ == "__main__":
    unittest.main()
