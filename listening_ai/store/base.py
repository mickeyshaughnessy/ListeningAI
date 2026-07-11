"""Abstract store interface for ListeningAI persistence."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseStore(ABC):
    """Persistence contract used by auth, tools, controller, and blueprint.

    Host projects either use a built-in backend (JsonFileStore, SpacesStore)
    or implement this interface against their existing storage.
    """

    # ---------------------------------------------------------------- users --

    @abstractmethod
    def create_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Create a user. Returns the user dict, or None if username taken."""

    @abstractmethod
    def verify_login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Validate credentials. Returns user dict on success, else None."""

    @abstractmethod
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load a user by id."""

    @abstractmethod
    def get_user_id_by_username(self, username: str) -> Optional[str]:
        """Resolve a username to user_id (case-insensitive)."""

    # ------------------------------------------------------------- sessions --

    @abstractmethod
    def create_session(self, user_id: str) -> str:
        """Issue a bearer token for user_id and return it."""

    @abstractmethod
    def get_user_id_for_token(self, token: str) -> Optional[str]:
        """Resolve a bearer token to user_id, or None if invalid/expired."""

    # ------------------------------------------------------------- profile --

    @abstractmethod
    def get_profile(self, user_id: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Merge updates into profile. Returns the new profile, or None if missing."""

    @abstractmethod
    def get_settings(self, user_id: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def update_settings(self, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ...

    # -------------------------------------------------------------- inbox --

    @abstractmethod
    def add_message(self, user_id: str, sender: str, body: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def list_messages(self, user_id: str) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def mark_message_read(self, user_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        ...

    # -------------------------------------------------------- notifications --

    @abstractmethod
    def add_notification(
        self, user_id: str, text: str, meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def list_notifications(self, user_id: str) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def mark_notification_read(
        self, user_id: str, notification_id: str
    ) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def delete_notification(self, user_id: str, notification_id: str) -> bool:
        ...

    # ---------------------------------------------------------------- chat --

    @abstractmethod
    def create_chat_session(self, user_id: str, agent_id: str = "default") -> str:
        """Create a chat session and return its id."""

    @abstractmethod
    def get_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def append_chat_messages(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Append messages to a chat session. Returns the full message list."""

    @abstractmethod
    def list_chat_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        ...
