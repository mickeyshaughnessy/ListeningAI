"""DigitalOcean Spaces / S3-compatible store.

Object layout under ``prefix`` (default ``listening_ai/``)::

    {prefix}users/{user_id}.json
    {prefix}usernames/{username_key}.json   # {"user_id": "..."}
    {prefix}sessions/{token}.json
    {prefix}inbox/{user_id}.json
    {prefix}notifications/{user_id}.json
    {prefix}chats/{session_id}.json
    {prefix}chat_index/{user_id}/{session_id}.json  # lightweight index

Requires the optional ``boto3`` dependency::

    pip install listening-ai[spaces]
"""
from __future__ import annotations

import json
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


class SpacesStore(BaseStore):
    def __init__(
        self,
        *,
        key: str,
        secret: str,
        bucket: str,
        region: str = "sfo3",
        endpoint: str = "",
        prefix: str = "listening_ai/",
        token_expiry_seconds: int = 86400 * 7,
        client: Any = None,
    ):
        self.bucket = bucket
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"
        self.token_expiry_seconds = token_expiry_seconds
        self._lock = threading.Lock()

        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as e:
                raise ImportError(
                    "SpacesStore requires boto3. Install with: pip install listening-ai[spaces]"
                ) from e
            endpoint_url = endpoint or f"https://{region}.digitaloceanspaces.com"
            self._client = boto3.client(
                "s3",
                region_name=region,
                endpoint_url=endpoint_url,
                aws_access_key_id=key,
                aws_secret_access_key=secret,
            )

    # ------------------------------------------------------------- low-level --

    def _key(self, path: str) -> str:
        return f"{self.prefix}{path}"

    def _get_json(self, path: str) -> Optional[Any]:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=self._key(path))
            return json.loads(resp["Body"].read().decode("utf-8"))
        except Exception as e:
            code = ""
            resp = getattr(e, "response", None)
            if isinstance(resp, dict):
                code = resp.get("Error", {}).get("Code", "") or ""
            # botocore ClientError NoSuchKey / 404, or missing object on S3-compat stacks
            msg = str(e).lower()
            if code in ("NoSuchKey", "404", "NotFound") or "nosuchkey" in msg or "not found" in msg:
                return None
            # Some mocks/stubs raise generic errors for missing keys
            if type(e).__name__ in ("NoSuchKey", "ClientError") and not code:
                return None
            raise

    def _put_json(self, path: str, data: Any) -> None:
        self._client.put_object(
            Bucket=self.bucket,
            Key=self._key(path),
            Body=json.dumps(data, indent=2),
            ContentType="application/json",
        )

    def _delete(self, path: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=self._key(path))
        except Exception:
            pass

    def _list_keys(self, path_prefix: str) -> List[str]:
        full = self._key(path_prefix)
        keys: List[str] = []
        token = None
        while True:
            kwargs: Dict[str, Any] = {"Bucket": self.bucket, "Prefix": full}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self._client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return keys

    # ---------------------------------------------------------------- users --

    def create_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        username_key = username.strip().lower()
        with self._lock:
            if self._get_json(f"usernames/{username_key}.json") is not None:
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
            self._put_json(f"users/{user_id}.json", user)
            self._put_json(f"usernames/{username_key}.json", {"user_id": user_id})
            return dict(user)

    def verify_login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        username_key = username.strip().lower()
        mapping = self._get_json(f"usernames/{username_key}.json")
        if not mapping:
            return None
        user = self._get_json(f"users/{mapping['user_id']}.json")
        if not user or not check_password_hash(user.get("password_hash", ""), password):
            return None
        return dict(user)

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        user = self._get_json(f"users/{user_id}.json")
        return dict(user) if user else None

    def get_user_id_by_username(self, username: str) -> Optional[str]:
        mapping = self._get_json(f"usernames/{username.strip().lower()}.json")
        return mapping.get("user_id") if mapping else None

    # ------------------------------------------------------------- sessions --

    def create_session(self, user_id: str) -> str:
        token = secrets.token_hex(24)
        self._put_json(
            f"sessions/{token}.json",
            {
                "user_id": user_id,
                "created_at": _now(),
                "expires_at": time.time() + self.token_expiry_seconds,
            },
        )
        return token

    def get_user_id_for_token(self, token: str) -> Optional[str]:
        session = self._get_json(f"sessions/{token}.json")
        if not session:
            return None
        if session.get("expires_at", 0) < time.time():
            self._delete(f"sessions/{token}.json")
            return None
        return session.get("user_id")

    # ------------------------------------------------------------- profile --

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        user = self.get_user(user_id)
        return (user or {}).get("profile", {})

    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            user = self._get_json(f"users/{user_id}.json")
            if not user:
                return None
            profile = user.setdefault("profile", {})
            for key, value in (updates or {}).items():
                if value is None:
                    profile.pop(key, None)  # null clears the field
                else:
                    profile[key] = value
            self._put_json(f"users/{user_id}.json", user)
            return dict(user["profile"])

    def get_settings(self, user_id: str) -> Dict[str, Any]:
        user = self.get_user(user_id)
        return (user or {}).get("settings", {})

    def update_settings(self, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            user = self._get_json(f"users/{user_id}.json")
            if not user:
                return None
            user.setdefault("settings", {}).update(updates or {})
            self._put_json(f"users/{user_id}.json", user)
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
            items = self._get_json(f"inbox/{user_id}.json") or []
            items.append(message)
            self._put_json(f"inbox/{user_id}.json", items)
        return message

    def list_messages(self, user_id: str) -> List[Dict[str, Any]]:
        return list(self._get_json(f"inbox/{user_id}.json") or [])

    def mark_message_read(self, user_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            items = self._get_json(f"inbox/{user_id}.json") or []
            for m in items:
                if m.get("id") == message_id:
                    m["read"] = True
                    self._put_json(f"inbox/{user_id}.json", items)
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
            items = self._get_json(f"notifications/{user_id}.json") or []
            items.append(notification)
            self._put_json(f"notifications/{user_id}.json", items)
        return notification

    def list_notifications(self, user_id: str) -> List[Dict[str, Any]]:
        return list(self._get_json(f"notifications/{user_id}.json") or [])

    def mark_notification_read(
        self, user_id: str, notification_id: str
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            items = self._get_json(f"notifications/{user_id}.json") or []
            for n in items:
                if n.get("id") == notification_id:
                    n["read"] = True
                    self._put_json(f"notifications/{user_id}.json", items)
                    return n
            return None

    def delete_notification(self, user_id: str, notification_id: str) -> bool:
        with self._lock:
            items = self._get_json(f"notifications/{user_id}.json") or []
            new_items = [n for n in items if n.get("id") != notification_id]
            if len(new_items) == len(items):
                return False
            self._put_json(f"notifications/{user_id}.json", new_items)
            return True

    # ---------------------------------------------------------------- chat --

    def create_chat_session(self, user_id: str, agent_id: str = "default") -> str:
        session_id = uuid.uuid4().hex
        chat = {
            "id": session_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "messages": [],
            "created_at": _now(),
            "updated_at": _now(),
        }
        self._put_json(f"chats/{session_id}.json", chat)
        self._put_json(
            f"chat_index/{user_id}/{session_id}.json",
            {
                "id": session_id,
                "agent_id": agent_id,
                "created_at": chat["created_at"],
                "updated_at": chat["updated_at"],
            },
        )
        return session_id

    def get_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        chat = self._get_json(f"chats/{session_id}.json")
        return dict(chat) if chat else None

    def append_chat_messages(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            chat = self._get_json(f"chats/{session_id}.json")
            if not chat:
                return None
            chat["messages"].extend(messages)
            chat["updated_at"] = _now()
            self._put_json(f"chats/{session_id}.json", chat)
            # refresh index
            self._put_json(
                f"chat_index/{chat['user_id']}/{session_id}.json",
                {
                    "id": session_id,
                    "agent_id": chat.get("agent_id", "default"),
                    "created_at": chat.get("created_at"),
                    "updated_at": chat["updated_at"],
                    "message_count": len(chat["messages"]),
                },
            )
            return list(chat["messages"])

    def list_chat_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        keys = self._list_keys(f"chat_index/{user_id}/")
        sessions: List[Dict[str, Any]] = []
        for key in keys:
            if not key.endswith(".json"):
                continue
            # strip bucket prefix to get store-relative path
            rel = key[len(self.prefix):] if key.startswith(self.prefix) else key
            meta = self._get_json(rel)
            if not meta:
                continue
            # attach full message list length if available
            chat = self.get_chat_session(meta["id"])
            if chat:
                sessions.append(chat)
            else:
                sessions.append({
                    "id": meta["id"],
                    "user_id": user_id,
                    "agent_id": meta.get("agent_id", "default"),
                    "messages": [],
                    "created_at": meta.get("created_at"),
                    "updated_at": meta.get("updated_at"),
                })
        return sessions
