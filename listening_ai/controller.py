"""
ChatController - the reusable agentic chat loop.

Call the LLM with registered tool schemas, execute any tool calls the model
asks for against the ToolRegistry, feed results back, and repeat until the
model stops calling tools (or max_steps is hit).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import llm
from . import store
from .settings import get_settings
from .tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """You are a helpful, concise assistant embedded directly in a website, \
acting as the user's navigator for the site. You listen better than you speak: ask short \
clarifying questions when needed, but prefer to just look things up or take the action yourself \
using your tools rather than making the user repeat themselves or dig through menus.

Guidelines:
- Use tools to read or update the user's profile, settings, inbox, and notifications instead of \
guessing at their state.
- Keep replies short and conversational (a few sentences), not long essays.
- If a tool call fails, tell the user plainly what happened rather than pretending it worked.
"""


class ChatController:
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 6,
    ):
        self.tools = tool_registry if tool_registry is not None else ToolRegistry()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_steps = max_steps

    def start_session(self, user_id: str, agent_id: str = "default") -> str:
        return store.create_chat_session(user_id, agent_id=agent_id)

    def get_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        chat = store.get_chat_session(session_id)
        return chat["messages"] if chat else None

    def handle_message(
        self, user_id: str, session_id: str, user_message: str
    ) -> Dict[str, Any]:
        chat = store.get_chat_session(session_id)
        if not chat or chat["user_id"] != user_id:
            raise ValueError("Unknown or mismatched chat session")

        messages = list(chat["messages"]) + [{"role": "user", "content": user_message}]
        final_text, model_used, tool_log = self.run_loop(messages, user_id)

        store.append_chat_messages(session_id, [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": final_text},
        ])

        return {
            "response": final_text,
            "session_id": session_id,
            "model_used": model_used,
            "tool_calls": tool_log,
        }

    def run_loop(
        self,
        messages: List[Dict[str, Any]],
        user_id: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Run the agentic tool loop over an in-memory message list.

        Host projects (e.g. GreenDial) call this directly when they manage
        their own conversation transcripts and only need the loop itself.

        Returns (final_text, model_used, tool_log).
        """
        tool_log: List[Dict[str, Any]] = []
        final_text = ""
        s = get_settings()
        model_used = s.openrouter_tools_model or s.openrouter_model
        tool_schemas = self.tools.schemas() if self.tools.has_tools() else None
        prompt = system_prompt if system_prompt is not None else self.system_prompt

        # Work on a copy so callers can keep their original list
        working = list(messages)

        for _step in range(self.max_steps):
            resp = llm.call_llm_with_tools(
                messages=working,
                tools=tool_schemas,
                system_prompt=prompt,
            )

            if resp.get("error"):
                if not final_text:
                    final_text = "Sorry, I'm having trouble reaching the language model right now."
                break

            if resp.get("model_used"):
                model_used = resp["model_used"]
            if resp.get("text"):
                final_text = resp["text"]

            tool_calls = resp.get("tool_calls") or []
            if not tool_calls or resp.get("stop_reason") == "end_turn":
                break

            working.append(resp["raw_message"])

            tool_results = []
            for tc in tool_calls:
                result = self.tools.execute(tc["name"], tc["input"], user_id)
                tool_log.append({
                    "name": tc["name"],
                    "input": tc["input"],
                    "result": result,
                })
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
            working.extend(tool_results)

        if not final_text:
            final_text = "Done."
        return final_text, model_used, tool_log

    # Back-compat alias
    def _run_agentic_loop(self, messages, user_id):
        return self.run_loop(messages, user_id)
