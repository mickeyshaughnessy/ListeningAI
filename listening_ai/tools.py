"""
Tool registry for the agentic chat loop.

Each tool is an OpenAI-compatible function schema plus a Python handler.
The default registry covers the generic surface every sibling site needs
(profile, settings, inbox, notifications); host projects extend it with
their own domain tools via ``registry.register(...)``.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from . import store


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> "ToolRegistry":
        """
        parameters: JSON Schema ``parameters`` object (OpenAI function-calling format).
        handler: callable(user_id, **kwargs) -> str | dict | list
        """
        self._tools[name] = {
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
            "handler": handler,
        }
        return self

    def register_from_openai_schema(
        self, schema: Dict[str, Any], handler: Callable[..., Any]
    ) -> "ToolRegistry":
        """Register a tool given a full OpenAI ``{type, function: {...}}`` schema."""
        func = schema.get("function") or {}
        return self.register(
            func.get("name", ""),
            func.get("description", ""),
            func.get("parameters") or {"type": "object", "properties": {}},
            handler,
        )

    def schemas(self) -> List[Dict[str, Any]]:
        return [t["schema"] for t in self._tools.values()]

    def has_tools(self) -> bool:
        return bool(self._tools)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def execute(self, name: str, inputs: Any, user_id: str) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: unknown tool '{name}'"
        try:
            result = tool["handler"](user_id, **(inputs or {}))
            if isinstance(result, (dict, list)):
                return json.dumps(result)
            return str(result)
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"Error executing {name}: {e}"


# ------------------------------------------------------------ handlers --

def _get_profile(user_id):
    profile = store.get_profile(user_id)
    return profile if profile else "Profile is empty - no data yet."


def _update_profile(user_id, field, value):
    updated = store.update_profile(user_id, {field: value})
    return f"Saved profile.{field} = {value!r}" if updated is not None else "Error: user not found."


def _get_settings(user_id):
    return store.get_settings(user_id) or {}


def _update_settings(user_id, field, value):
    updated = store.update_settings(user_id, {field: value})
    return f"Saved settings.{field} = {value!r}" if updated is not None else "Error: user not found."


def _list_messages(user_id, unread_only=False):
    messages = store.list_messages(user_id)
    if unread_only:
        messages = [m for m in messages if not m["read"]]
    if not messages:
        return "Inbox is empty."
    return messages[-20:]


def _send_message(user_id, to_username, body):
    recipient_id = store.get_user_id_by_username(to_username)
    if not recipient_id:
        return f"Error: no user named '{to_username}'."
    sender = store.get_user(user_id)
    sender_name = sender["username"] if sender else user_id
    msg = store.add_message(recipient_id, sender_name, body)
    return f"Message sent to {to_username}: {msg['body']!r}"


def _list_notifications(user_id, unread_only=False):
    notifications = store.list_notifications(user_id)
    if unread_only:
        notifications = [n for n in notifications if not n["read"]]
    if not notifications:
        return "No notifications."
    return notifications[-20:]


def _create_notification(user_id, text):
    notif = store.add_notification(user_id, text)
    return f"Notification queued: {notif['text']!r}"


def _mark_notification_read(user_id, notification_id):
    notif = store.mark_notification_read(user_id, notification_id)
    return f"Marked notification {notification_id} as read." if notif else "Error: notification not found."


def default_registry() -> ToolRegistry:
    """Generic tools available in every ListeningAI deployment."""
    registry = ToolRegistry()

    registry.register(
        "get_profile",
        "Read the current user's profile (freeform key/value facts remembered about them). "
        "Call this at the start of a conversation to recall context.",
        {"type": "object", "properties": {}, "required": []},
        _get_profile,
    )
    registry.register(
        "update_profile",
        "Save or update one field in the user's profile, e.g. their name, goals, or preferences.",
        {
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "Profile field name"},
                "value": {"type": "string", "description": "Value to save"},
            },
            "required": ["field", "value"],
        },
        _update_profile,
    )
    registry.register(
        "get_settings",
        "Read the current user's site settings (e.g. tone, notification preferences).",
        {"type": "object", "properties": {}, "required": []},
        _get_settings,
    )
    registry.register(
        "update_settings",
        "Save or update one setting for the current user.",
        {
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "Setting name"},
                "value": {"type": "string", "description": "Value to save"},
            },
            "required": ["field", "value"],
        },
        _update_settings,
    )
    registry.register(
        "list_messages",
        "Read the user's inbox (messages sent to them by other users).",
        {
            "type": "object",
            "properties": {
                "unread_only": {"type": "boolean", "description": "Only return unread messages", "default": False},
            },
            "required": [],
        },
        _list_messages,
    )
    registry.register(
        "send_message",
        "Send a message from the current user to another user's inbox by username.",
        {
            "type": "object",
            "properties": {
                "to_username": {"type": "string", "description": "Recipient's username"},
                "body": {"type": "string", "description": "Message text"},
            },
            "required": ["to_username", "body"],
        },
        _send_message,
    )
    registry.register(
        "list_notifications",
        "Read the user's notifications (system alerts, reminders).",
        {
            "type": "object",
            "properties": {
                "unread_only": {"type": "boolean", "description": "Only return unread notifications", "default": False},
            },
            "required": [],
        },
        _list_notifications,
    )
    registry.register(
        "create_notification",
        "Queue a notification or reminder for the user, e.g. a follow-up or scheduled nudge.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Notification text shown to the user"},
            },
            "required": ["text"],
        },
        _create_notification,
    )
    registry.register(
        "mark_notification_read",
        "Mark one of the user's notifications as read given its id.",
        {
            "type": "object",
            "properties": {
                "notification_id": {"type": "string", "description": "Notification id"},
            },
            "required": ["notification_id"],
        },
        _mark_notification_read,
    )
    return registry
