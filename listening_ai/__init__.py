"""
listening_ai - a reusable "listens better than it speaks" chatbot controller.

Pulls together an LLM wrapper (OpenRouter, with model fallback), a pluggable
store (JSON file or DigitalOcean Spaces), token auth, a tool registry, and an
agentic chat controller into one Flask blueprint that any sibling project can
mount — then extend with its own domain tools.
"""
from __future__ import annotations

from typing import Any, Optional

from .settings import Settings, configure, get_settings
from .controller import ChatController
from .tools import ToolRegistry, default_registry
from .blueprint import create_blueprint
from .store import BaseStore, JsonFileStore, build_store, set_store, get_store
from .util import utc_now_iso
from .llm import (
    call_llm,
    call_llm_with_tools,
    completion,
    get_last_model_used,
)

__version__ = "0.1.1"

__all__ = [
    "Settings",
    "configure",
    "get_settings",
    "configure_app",
    "ChatController",
    "ToolRegistry",
    "default_registry",
    "create_blueprint",
    "BaseStore",
    "JsonFileStore",
    "SpacesStore",
    "build_store",
    "set_store",
    "get_store",
    "utc_now_iso",
    "call_llm",
    "call_llm_with_tools",
    "completion",
    "get_last_model_used",
    "__version__",
]


def __getattr__(name: str):
    # Lazy export so importing listening_ai without boto3 still works
    if name == "SpacesStore":
        from .store.spaces import SpacesStore
        return SpacesStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def configure_app(
    settings: Optional[Settings] = None,
    store: Optional[BaseStore] = None,
    **kwargs: Any,
) -> Settings:
    """
    One-shot setup for host projects.

    Configures settings, installs a store (built from settings if not passed),
    and returns the active Settings.

    Example (GreenDial)::

        import config
        from listening_ai import Settings, configure_app

        configure_app(
            Settings.from_config_module(config),
            store_backend="spaces",
            spaces_prefix="greendial/listening_ai/",
        )
    """
    if settings is None and kwargs:
        # Allow configure_app(openrouter_api_key=..., store_backend="spaces", ...)
        settings = configure(**kwargs)
    elif settings is not None:
        settings = configure(settings, **kwargs) if kwargs else configure(settings)
    else:
        settings = configure()

    if store is not None:
        set_store(store)
    else:
        set_store(build_store(settings))

    return settings
