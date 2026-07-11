"""
Reply brevity: parse LLM replies into shorter "listen more, speak less" forms.

Levels (Settings.reply_brevity / ChatController.reply_brevity):
  none        — pass through
  short       — a few tight sentences
  very_short  — one breath: one sentence, two only if tools ran

Tool outcomes are preserved: if the model already called tools, shortening
keeps what was *done* and drops monologue, hedges, and essay structure.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from . import llm
from .settings import get_settings

VALID_LEVELS = ("none", "short", "very_short")

# Appended to the agent system prompt so the primary turn also prefers action.
LISTENING_DIRECTIVES = {
    "short": (
        "\n\n# Listening mode (short)\n"
        "- Prefer tools over prose: if the user asked you to do something you can do "
        "with a tool, call the tool first. Do not ask permission when the action is clear.\n"
        "- After tools run, reply in at most 2–3 short sentences confirming what you did "
        "and only the essential next detail.\n"
        "- No essays, bullet walls, or filler openers (\"Great question\", \"I'd be happy to\").\n"
    ),
    "very_short": (
        "\n\n# Listening mode (very short)\n"
        "- PRIORITY: use tools to actually do what the user wants. Calling the right tool "
        "matters more than a polished explanation.\n"
        "- If an action is available via tools (save, log, update, notify, look up), do it "
        "in this turn — do not only describe how you would do it.\n"
        "- Final reply: one short sentence (two only if needed after tools). "
        "State what you did or the single most useful fact. No preamble, no lists, "
        "no \"let me know if\".\n"
    ),
}

_SHORTEN_SYSTEM = {
    "short": (
        "You compress chatbot replies so they listen better than they speak. "
        "Rewrite the draft into 2–3 short sentences max. "
        "Keep concrete facts, numbers, and anything that reports a completed action. "
        "Drop greetings, disclaimers, and essay structure. "
        "Output ONLY the rewritten reply, no quotes or labels."
    ),
    "very_short": (
        "You compress chatbot replies so they listen better than they speak. "
        "Rewrite the draft into ONE short sentence (two only if a tool result must be stated). "
        "If tools ran, lead with what was done (e.g. \"Logged sleep as 6 hours.\"). "
        "Drop all padding. Output ONLY the rewritten reply."
    ),
}


def normalize_level(level: Optional[str]) -> str:
    if not level:
        return "none"
    level = str(level).strip().lower().replace("-", "_")
    if level in ("off", "full", "normal", "default", ""):
        return "none"
    if level in ("very short", "veryshort", "xs", "micro"):
        return "very_short"
    if level not in VALID_LEVELS:
        return "none"
    return level


def listening_system_suffix(level: Optional[str]) -> str:
    level = normalize_level(level)
    return LISTENING_DIRECTIVES.get(level, "")


def _tool_summary(tool_log: Optional[Sequence[Dict[str, Any]]]) -> str:
    if not tool_log:
        return ""
    parts = []
    for entry in tool_log[:8]:
        name = entry.get("name") or "tool"
        inp = entry.get("input") or {}
        result = entry.get("result")
        if isinstance(result, str) and len(result) > 120:
            result = result[:117] + "..."
        # Prefer compact name + key fields
        keys = []
        for k in ("field", "value", "text", "area", "agent_id"):
            if k in inp and inp[k] not in (None, ""):
                keys.append(f"{k}={inp[k]!r}")
        arg = ", ".join(keys) if keys else ""
        parts.append(f"- {name}({arg}) → {result}")
    return "\n".join(parts)


def _heuristic_shorten(text: str, level: str, tool_log: Optional[Sequence[Dict[str, Any]]]) -> str:
    """Offline fallback: no extra LLM call."""
    text = (text or "").strip()
    if not text:
        if tool_log:
            names = [e.get("name") for e in tool_log if e.get("name")]
            if names:
                return f"Done — {', '.join(names)}."
            return "Done."
        return text

    # Prefer first non-empty paragraph / sentence units
    # Split on blank lines first, then sentences
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    body = paragraphs[0] if paragraphs else text
    # Strip markdown headings / bullets for first-pass
    body = re.sub(r"^#+\s*", "", body, flags=re.M)
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if level == "very_short":
        # If tools ran and first sentence is fluff, try one that mentions save/log/update
        if tool_log and sentences:
            actiony = re.compile(
                r"\b(saved|logged|updated|set|recorded|done|noted|queued|sent)\b",
                re.I,
            )
            for s in sentences:
                if actiony.search(s):
                    return s
            return sentences[0]
        return sentences[0] if sentences else text[:160].rstrip() + ("…" if len(text) > 160 else "")

    # short: up to 3 sentences
    return " ".join(sentences[:3]) if sentences else text


def shorten_reply(
    text: str,
    level: Optional[str] = None,
    tool_log: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    use_llm: bool = True,
) -> str:
    """
    Parse ``text`` into a shorter reply according to ``level``.

    When ``use_llm`` is True and an API key is configured, uses a low-token
    rewrite pass; otherwise (or on failure) uses a local heuristic.
    """
    level = normalize_level(level if level is not None else get_settings().reply_brevity)
    text = text or ""
    if level == "none" or not text.strip():
        return text

    if use_llm:
        s = get_settings()
        if s.openrouter_api_key:
            try:
                tools_block = _tool_summary(tool_log)
                user_prompt = f"Draft reply:\n{text.strip()}"
                if tools_block:
                    user_prompt += (
                        f"\n\nTools already executed this turn (preserve outcomes):\n{tools_block}"
                    )
                max_tokens = 80 if level == "very_short" else 160
                shortened = llm.completion(
                    prompt=user_prompt,
                    system_prompt=_SHORTEN_SYSTEM[level],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                if shortened and not shortened.startswith("Sorry, I couldn't reach"):
                    cleaned = shortened.strip().strip('"').strip("'")
                    # Guard against model expanding instead of compressing
                    if len(cleaned) < len(text) * 1.15 or len(cleaned) < 280:
                        print(f"[Brevity] {level}: {len(text)} → {len(cleaned)} chars")
                        return cleaned
            except Exception as e:
                print(f"[Brevity] LLM shorten failed ({e}); heuristic fallback")

    return _heuristic_shorten(text, level, tool_log)
