"""
Pluggable storage for ListeningAI.

Usage::

    from listening_ai import configure, Settings
    from listening_ai.store import build_store, set_store

    settings = Settings(store_backend="spaces", spaces_key=..., ...)
    configure(settings)
    set_store(build_store(settings))

Or let ``listening_ai.configure_app()`` / the demo server do it for you.

Module-level functions (``create_user``, ``get_profile``, ...) delegate to the
active store so tools, auth, controller, and blueprint stay store-agnostic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..settings import Settings, get_settings
from .base import BaseStore
from .json_file import JsonFileStore

# SpacesStore imported lazily so boto3 is optional

_store: Optional[BaseStore] = None


def build_store(settings: Optional[Settings] = None) -> BaseStore:
    """Construct a store backend from Settings."""
    settings = settings or get_settings()
    backend = (settings.store_backend or "json").lower()

    if backend in ("json", "file", "json_file"):
        return JsonFileStore(
            db_path=settings.resolved_db_path(),
            token_expiry_seconds=settings.token_expiry_seconds,
        )

    if backend in ("spaces", "s3", "do_spaces"):
        from .spaces import SpacesStore
        return SpacesStore(
            key=settings.spaces_key,
            secret=settings.spaces_secret,
            bucket=settings.spaces_bucket,
            region=settings.spaces_region,
            endpoint=settings.resolved_spaces_endpoint(),
            prefix=settings.spaces_prefix,
            token_expiry_seconds=settings.token_expiry_seconds,
        )

    raise ValueError(
        f"Unknown store_backend={settings.store_backend!r}. "
        "Use 'json' or 'spaces'."
    )


def set_store(store: BaseStore) -> BaseStore:
    """Install a store instance as the process-wide default."""
    global _store
    _store = store
    return store


def get_store() -> BaseStore:
    """Return the active store, building a default from settings if needed."""
    global _store
    if _store is None:
        _store = build_store()
    return _store


# ---------------------------------------------------------------------------
# Module-level proxies — keep call sites (tools/auth/controller/blueprint)
# looking like the original flat store.py API.
# ---------------------------------------------------------------------------

def create_user(username, password):
    return get_store().create_user(username, password)


def verify_login(username, password):
    return get_store().verify_login(username, password)


def get_user(user_id):
    return get_store().get_user(user_id)


def get_user_id_by_username(username):
    return get_store().get_user_id_by_username(username)


def create_session(user_id):
    return get_store().create_session(user_id)


def get_user_id_for_token(token):
    return get_store().get_user_id_for_token(token)


def get_profile(user_id):
    return get_store().get_profile(user_id)


def update_profile(user_id, updates):
    return get_store().update_profile(user_id, updates)


def get_settings_for_user(user_id):
    """User settings (not library Settings). Named to avoid clash with settings.py."""
    return get_store().get_settings(user_id)


# Keep the original name for callers that did store.get_settings(user_id)
def get_settings(user_id):  # type: ignore[misc]
    return get_store().get_settings(user_id)


def update_settings(user_id, updates):
    return get_store().update_settings(user_id, updates)


def add_message(user_id, sender, body):
    return get_store().add_message(user_id, sender, body)


def list_messages(user_id):
    return get_store().list_messages(user_id)


def mark_message_read(user_id, message_id):
    return get_store().mark_message_read(user_id, message_id)


def add_notification(user_id, text, meta=None):
    return get_store().add_notification(user_id, text, meta=meta)


def list_notifications(user_id):
    return get_store().list_notifications(user_id)


def mark_notification_read(user_id, notification_id):
    return get_store().mark_notification_read(user_id, notification_id)


def delete_notification(user_id, notification_id):
    return get_store().delete_notification(user_id, notification_id)


def create_chat_session(user_id, agent_id="default"):
    return get_store().create_chat_session(user_id, agent_id=agent_id)


def get_chat_session(session_id):
    return get_store().get_chat_session(session_id)


def append_chat_messages(session_id, messages):
    return get_store().append_chat_messages(session_id, messages)


def list_chat_sessions(user_id):
    return get_store().list_chat_sessions(user_id)


__all__ = [
    "BaseStore",
    "JsonFileStore",
    "build_store",
    "set_store",
    "get_store",
    "create_user",
    "verify_login",
    "get_user",
    "get_user_id_by_username",
    "create_session",
    "get_user_id_for_token",
    "get_profile",
    "update_profile",
    "get_settings",
    "update_settings",
    "add_message",
    "list_messages",
    "mark_message_read",
    "add_notification",
    "list_notifications",
    "mark_notification_read",
    "delete_notification",
    "create_chat_session",
    "get_chat_session",
    "append_chat_messages",
    "list_chat_sessions",
]
