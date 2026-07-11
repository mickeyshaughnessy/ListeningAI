"""Unit tests for the OpenRouter LLM helpers (no network)."""
import unittest
from unittest import mock

from listening_ai import Settings, configure, completion, call_llm_with_tools, get_last_model_used
from listening_ai import llm as llm_mod


class LlmHelperTests(unittest.TestCase):
    def setUp(self):
        configure(Settings(
            openrouter_api_key="test-key",
            openrouter_model="test/primary",
            openrouter_tools_model="test/tools",
            openrouter_fallback_models=["test/fb1", "test/fb2"],
            openrouter_tools_fallback_models=["test/tfb1"],
            llm_temperature=0.5,
            llm_max_tokens=100,
        ))
        llm_mod._last_used_model = None

    def test_completion_prompt_form(self):
        fake = {
            "stop_reason": "end_turn",
            "text": "hello world",
            "tool_calls": [],
            "tool_uses": [],
            "raw_message": {"role": "assistant", "content": "hello world"},
            "raw_content": {"role": "assistant", "content": "hello world"},
            "model_used": "test/primary",
            "error": None,
        }
        with mock.patch.object(llm_mod, "call_llm_with_tools", return_value=fake) as m:
            text = completion(prompt="hi", system_prompt="be brief")
        self.assertEqual(text, "hello world")
        kwargs = m.call_args.kwargs
        self.assertIsNone(kwargs.get("tools"))
        self.assertEqual(kwargs["system_prompt"], "be brief")
        # prompt becomes a single user message inside call_llm_with_tools path
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "hi"}])

    def test_missing_api_key(self):
        configure(Settings(openrouter_api_key=""))
        resp = call_llm_with_tools(messages=[{"role": "user", "content": "x"}])
        self.assertEqual(resp["error"], "missing_api_key")
        self.assertIsNone(resp["text"])

    def test_fallback_on_http_error(self):
        class FakeResp:
            def __init__(self, status, body):
                self.status_code = status
                self.text = body if isinstance(body, str) else ""
                self._body = body

            def json(self):
                return self._body

        good = {
            "choices": [{
                "message": {"content": "ok from fb", "tool_calls": []},
                "finish_reason": "stop",
            }]
        }

        def side_effect(*args, **kwargs):
            model = kwargs.get("json", {}).get("model") or (args and args) 
            # inspect payload model from call
            payload = kwargs.get("json") or {}
            if payload.get("model") == "test/primary":
                return FakeResp(500, "boom")
            return FakeResp(200, good)

        with mock.patch.object(llm_mod.requests, "post", side_effect=side_effect):
            resp = call_llm_with_tools(messages=[{"role": "user", "content": "hi"}])
        self.assertIsNone(resp["error"])
        self.assertEqual(resp["text"], "ok from fb")
        self.assertEqual(resp["model_used"], "test/fb1")
        self.assertEqual(get_last_model_used(), "test/fb1")
        # GreenDial aliases present
        self.assertEqual(resp["tool_uses"], [])
        self.assertIsNotNone(resp["raw_content"])

    def test_tool_calls_parsed(self):
        body = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "tc1",
                        "function": {
                            "name": "get_profile",
                            "arguments": '{"field":"x"}',
                        },
                    }],
                },
            }]
        }

        class FakeResp:
            status_code = 200
            text = ""

            def json(self):
                return body

        with mock.patch.object(llm_mod.requests, "post", return_value=FakeResp()):
            resp = call_llm_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "get_profile"}}],
            )
        self.assertEqual(resp["stop_reason"], "tool_calls")
        self.assertEqual(resp["tool_calls"][0]["name"], "get_profile")
        self.assertEqual(resp["tool_calls"][0]["input"], {"field": "x"})
        self.assertEqual(resp["tool_uses"], resp["tool_calls"])


if __name__ == "__main__":
    unittest.main()
