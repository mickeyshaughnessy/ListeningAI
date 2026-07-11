"""
OpenRouter completion wrapper with model fallback chains.

Supports both plain chat and OpenAI-style tool/function calling.
Reads credentials and model lists from ``listening_ai.settings``.

Host projects (e.g. GreenDial) should call these helpers for all generic LLM
work rather than re-implementing OpenRouter clients.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

from .settings import get_settings

_last_used_model = None


def get_last_model_used():
    """Return the model id that last produced a successful completion."""
    return _last_used_model


def _api_headers():
    s = get_settings()
    headers = {
        "Authorization": f"Bearer {s.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    if s.openrouter_site_url:
        headers["HTTP-Referer"] = s.openrouter_site_url
    if s.openrouter_site_name:
        headers["X-Title"] = s.openrouter_site_name
    return headers


def _build_sequence(primary, fallbacks):
    seq = [primary] + [m for m in (fallbacks or []) if m != primary]
    seen = set()
    ordered = []
    for m in seq:
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def _normalize_messages(
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build an OpenAI-style messages list from prompt and/or messages."""
    all_messages: List[Dict[str, Any]] = []
    if system_prompt:
        all_messages.append({"role": "system", "content": system_prompt})
    if messages:
        all_messages.extend(list(messages))
    elif prompt is not None:
        all_messages.append({"role": "user", "content": prompt})
    return all_messages


def completion(
    prompt: Optional[str] = None,
    *,
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    use_fallback: bool = False,
) -> str:
    """
    Plain chat completion (no tools). Returns reply text, or a short apology
    if every model in the fallback chain failed.

    Accepts either a single ``prompt`` string (becomes a user message) or a
    full ``messages`` list. Host apps often use the prompt form for one-shot
    generations (notifications, suggestions, etc.).
    """
    s = get_settings()
    if use_fallback and model is None and s.openrouter_fallback_models:
        model = s.openrouter_fallback_models[0]
    elif model is None:
        model = s.openrouter_model

    resp = call_llm_with_tools(
        messages=_normalize_messages(messages=messages, prompt=prompt, system_prompt=None),
        tools=None,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.get("text") or "Sorry, I couldn't reach the language model just now."


def call_llm(
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    prompt: Optional[str] = None,
) -> str:
    """
    Plain chat completion returning reply text.

    Preferred over ``completion`` when you already have a multi-turn messages
    list. ``completion`` is the ergonomic one-shot wrapper.
    """
    return completion(
        prompt=prompt,
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


def call_llm_with_tools(
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    LLM completion with optional tool/function calling (OpenAI-compatible).

    Returns dict:
        {
          "stop_reason": "end_turn" | "tool_calls",
          "text":        str | None,
          "tool_calls":  [{"id", "name", "input"}, ...],
          "raw_message": dict,
          "model_used":  str,
          "error":       str | None,
        }

    Also includes GreenDial-compat aliases ``tool_uses`` (== tool_calls) and
    ``raw_content`` (== raw_message).
    """
    global _last_used_model
    s = get_settings()

    primary = model or (s.openrouter_tools_model if tools else s.openrouter_model)
    fallbacks = s.openrouter_tools_fallback_models if tools else s.openrouter_fallback_models
    sequence = _build_sequence(primary, fallbacks)

    temperature = s.llm_temperature if temperature is None else temperature
    max_tokens = max_tokens or s.llm_max_tokens

    all_messages = _normalize_messages(
        messages=messages, prompt=prompt, system_prompt=system_prompt
    )

    def _err(msg: str) -> Dict[str, Any]:
        return {
            "stop_reason": "end_turn",
            "text": None,
            "tool_calls": [],
            "tool_uses": [],
            "raw_message": None,
            "raw_content": None,
            "model_used": primary,
            "error": msg,
        }

    if not s.openrouter_api_key:
        return _err("missing_api_key")

    if not all_messages:
        return _err("empty_messages")

    last_err = "unknown"
    for attempt, m in enumerate(sequence):
        try:
            payload: Dict[str, Any] = {
                "model": m,
                "messages": all_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            label = "LLM Tools" if tools else "LLM"
            print(f"[{label}] Calling {m} (attempt={attempt + 1}/{len(sequence)})...")

            resp = requests.post(
                s.openrouter_api_url,
                headers=_api_headers(),
                json=payload,
                timeout=45,
            )

            if resp.status_code == 429:
                last_err = "rate_limited"
                print(f"[{label}] Rate limited on {m}")
                continue
            if resp.status_code >= 400:
                last_err = f"http_{resp.status_code}: {resp.text[:200]}"
                print(f"[{label}] Error {resp.status_code} on {m}: {resp.text[:200]}")
                continue

            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            msg_obj = choice.get("message", {})
            if not choice or not msg_obj:
                last_err = "no_choices"
                print(f"[{label}] No choices from {m}")
                continue

            text = msg_obj.get("content")
            raw_tool_calls = msg_obj.get("tool_calls") or []
            finish_reason = choice.get("finish_reason", "stop")

            tool_calls = []
            for tc in raw_tool_calls:
                try:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        parsed = json.loads(args) if args.strip() else {}
                    else:
                        parsed = args if isinstance(args, dict) else {}
                    tool_calls.append({
                        "id": tc.get("id", f"tc_{len(tool_calls)}"),
                        "name": func.get("name", ""),
                        "input": parsed if isinstance(parsed, dict) else {},
                    })
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    last_err = f"bad_tool_call_json: {e}"
                    print(f"[{label}] Could not parse tool call: {e}")
                    continue

            raw_message = {"role": "assistant", "content": text}
            if raw_tool_calls:
                raw_message["tool_calls"] = raw_tool_calls

            _last_used_model = m
            if attempt > 0:
                print(f"[{label}] Succeeded on fallback {m} (attempt {attempt + 1})")

            return {
                "stop_reason": "tool_calls" if (finish_reason == "tool_calls" or tool_calls) else "end_turn",
                "text": text,
                "tool_calls": tool_calls,
                "tool_uses": tool_calls,  # GreenDial alias
                "raw_message": raw_message,
                "raw_content": raw_message,  # GreenDial alias
                "model_used": m,
                "error": None,
            }

        except requests.exceptions.Timeout:
            last_err = "timeout"
            print(f"[LLM] Timeout on {m}")
        except requests.exceptions.RequestException as e:
            last_err = f"connection_error: {e}"
            print(f"[LLM] Connection error on {m}: {e}")
        except Exception as e:
            last_err = f"unexpected_error: {e}"
            print(f"[LLM] Unexpected error on {m}: {e}")

        if attempt + 1 < len(sequence):
            print(f"[LLM] {m} failed ({last_err}), trying next")

    print(f"[LLM] All {len(sequence)} models exhausted, last error: {last_err}")
    return _err(last_err)
