"""JSON-file-backed store — default backend for local demos and tests."""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from werkzeug.security import check_password_hash, generate_password_hash

from ..util import utc_now_iso
from .base import BaseStore


def _now() -> str:
    return utc_now_iso()


_EMPTY_DB = {
    "users": {},
    "usernames": {},
    "sessions": {},
    "messages": {},
    "notifications": {},
    "chats": {},
}


class JsonFileStore(BaseStore):
    def __init__(self, db_path: str, token_expiry_seconds: int = 86400 * 7):
        self.db_path = db_path
        self.token_expiry_seconds = token_expiry_seconds
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)) or ".", exist_ok=True)
        if not os.path.exists(self.db_path):
            return json.loads(json.dumps(_EMPTY_DB))
        with open(self.db_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = json.loads(json.dumps(_EMPTY_DB))
        for key, default in _EMPTY_DB.items():
            data.setdefault(key, json.loads(json.dumps(default)))
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)) or ".", exist_ok=True)
        tmp_path = self.db_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.db_path)

    # ---------------------------------------------------------------- users --

    def create_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        username_key = username.strip().lower()
        with self._lock:
            db = self._load()
            if username_key in db["usernames"]:
                return None
            user_id = f"user_{uuid.uuid4().hex[:12]}"
            user = {
                "user_id": user_id,
                "username": username.strip(),
                "password_hash": generate_password_hash(password),
                "profile": {},
                "settings": {"tone": "friendly", "notifications_enabled": True},
                "created_at": _now(),
            }
            db["users"][user_id] = user
            db["usernames"][username_key] = user_id
            self._save(db)
            return dict(user)

    def verify_login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        username_key = username.strip().lower()
        with self._lock:
            db = self._load()
            user_id = db["usernames"].get(username_key)
            if not user_id:
                return None
            user = db["users"].get(user_id)
            if not user or not check_password_hash(user["password_hash"], password):
                return None
            return dict(user)

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            db = self._load()
            user = db["users"].get(user_id)
            return dict(user) if user else None

    def get_user_id_by_username(self, username: str) -> Optional[str]:
        with self._lock:
            db = self._load()
            return db["usernames"].get(username.strip().lower())

    # ------------------------------------------------------------- sessions --

    def create_session(self, user_id: str) -> str:
        token = secrets.token_hex(24)
        with self._lock:
            db = self._load()
            db["sessions"][token] = {
                "user_id": user_id,
                "created_at": _now(),
                "expires_at": time.time() + self.token_expiry_seconds,
            }
            self._save(db)
        return token

    def get_user_id_for_token(self, token: str) -> Optional[str]:
        with self._lock:
            db = self._load()
            session = db["sessions"].get(token)
            if not session:
                return None
            if session["expires_at"] < time.time():
                del db["sessions"][token]
                self._save(db)
                return None
            return session["user_id"]

    # ------------------------------------------------------------- profile --

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        user = self.get_user(user_id)
        return (user or {}).get("profile", {})

    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            db = self._load()
            user = db["users"].get(user_id)
            if not user:
                return None
            user.setdefault("profile", {}).update(updates or {})
            self._save(db)
            return dict(user["profile"])

    def get_settings(self, user_id: str) -> Dict[str, Any]:
        user = self.get_user(user_id)
        return (user or {}).get("settings", {})

    def update_settings(self, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            db = self._load()
            user = db["users"].get(user_id)
            if not user:
                return None
            user.setdefault("settings", {}).update(updates or {})
            self._save(db)
            return dict(user["settings"])

    # -------------------------------------------------------------- inbox --

    def add_message(self, user_id: str, sender: str, body: str) -> Dict[str, Any]:
        message = {
            "id": uuid.uuid4().hex[:12],
            "from": sender,
            "body": body,
            "ts": _now(),
            "read": False,
        }
        with self._lock:
            db = self._load()
            db["messages"].setdefault(user_id, []).append(message)
            self._save(db)
        return message

    def list_messages(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            db = self._load()
            return list(db["messages"].get(user_id, []))

    def mark_message_read(self, user_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            db = self._load()
            for m in db["messages"].get(user_id, []):
                if m["id"] == message_id:
                    m["read"] = True
                    self._save(db)
                    return m
            return None

    # -------------------------------------------------------- notifications --

    def add_notification(
        self, user_id: str, text: str, meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        notification = {
            "id": uuid.uuid4().hex[:12],
            "text": text,
            "meta": meta or {},
            "ts": _now(),
            "read": False,
        }
        with self._lock:
            db = self._load()
            db["notifications"].setdefault(user_id, []).append(notification)
            self._save(db)
        return notification

    def list_notifications(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            db = self._load()
            return list(db["notifications"].get(user_id, []))

    def mark_notification_read(
        self, user_id: str, notification_id: str
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            db = self._load()
            for n in db["notifications"].get(user_id, []):
                if n["id"] == notification_id:
                    n["read"] = True
                    self._save(db)
                    return n
            return None

    def delete_notification(self, user_id: str, notification_id: str) -> bool:
        with self._lock:
            db = self._load()
            items = db["notifications"].get(user_id, [])
            before = len(items)
            db["notifications"][user_id] = [n for n in items if n["id"] != notification_id]
            self._save(db)
            return len(db["notifications"][user_id]) < before

    # ---------------------------------------------------------------- chat --

    def create_chat_session(self, user_id: str, agent_id: str = "default") -> str:
        session_id = uuid.uuid4().hex
        with self._lock:
            db = self._load()
            db["chats"][session_id] = {
                "id": session_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "messages": [],
                "created_at": _now(),
                "updated_at": _now(),
            }
            self._save(db)
        return session_id

    def get_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            db = self._load()
            chat = db["chats"].get(session_id)
            return dict(chat) if chat else None

    def append_chat_messages(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            db = self._load()
            chat = db["chats"].get(session_id)
            if not chat:
                return None
            chat["messages"].extend(messages)
            chat["updated_at"] = _now()
            self._save(db)
            return list(chat["messages"])

    def list_chat_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            db = self._load()
            return [c for c in db["chats"].values() if c["user_id"] == user_id]
