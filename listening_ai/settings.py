"""
Runtime settings for the listening_ai library.

Host projects call `configure(...)` once at startup (typically from their
own config.py). The demo server does the same from ListeningAI's config.py.
Library modules read settings via `get_settings()` — never via a hard-coded
`import config` of the host app.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Settings:
    # --- LLM (OpenRouter) ---
    openrouter_api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-oss-20b:free"
    openrouter_tools_model: str = "openai/gpt-oss-20b:free"
    openrouter_fallback_models: List[str] = field(default_factory=lambda: [
        "openai/gpt-oss-120b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
    ])
    openrouter_tools_fallback_models: List[str] = field(default_factory=lambda: [
        "openai/gpt-oss-120b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ])
    openrouter_site_url: str = "https://github.com/mickeyshaughnessy/ListeningAI"
    openrouter_site_name: str = "ListeningAI"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 800

    # Reply brevity: "none" | "short" | "very_short"
    # Post-parse LLM replies so the bot listens more than it speaks.
    # "very_short" also steers the agent loop to prioritize tool use.
    reply_brevity: str = "none"

    # --- Auth ---
    token_expiry_seconds: int = 86400 * 7

    # --- Storage backend: "json" | "spaces" ---
    store_backend: str = "json"

    # JSON file store
    data_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data"))
    db_path: Optional[str] = None  # defaults to data_dir/db.json

    # DigitalOcean Spaces / S3-compatible store
    spaces_key: str = ""
    spaces_secret: str = ""
    spaces_region: str = "sfo3"
    spaces_endpoint: str = ""  # e.g. https://sfo3.digitaloceanspaces.com
    spaces_bucket: str = ""
    spaces_prefix: str = "listening_ai/"

    def resolved_db_path(self) -> str:
        if self.db_path:
            return self.db_path
        return os.path.join(self.data_dir, "db.json")

    def resolved_spaces_endpoint(self) -> str:
        if self.spaces_endpoint:
            return self.spaces_endpoint
        if self.spaces_region:
            return f"https://{self.spaces_region}.digitaloceanspaces.com"
        return ""

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "Settings":
        """Build Settings from a dict or config module's __dict__-like mapping.

        Accepts both snake_case Settings field names and the ALL_CAPS names
        used by sibling project config.py modules.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: Dict[str, Any] = {}

        # Direct snake_case
        for k, v in data.items():
            if k in known and not k.startswith("_") and v is not None:
                kwargs[k] = v

        # ALL_CAPS aliases used by sibling apps
        aliases = {
            "OPENROUTER_API_URL": "openrouter_api_url",
            "OPENROUTER_API_KEY": "openrouter_api_key",
            "OPENROUTER_MODEL": "openrouter_model",
            "OPENROUTER_TOOLS_MODEL": "openrouter_tools_model",
            "OPENROUTER_FALLBACK_MODELS": "openrouter_fallback_models",
            "OPENROUTER_TOOLS_FALLBACK_MODELS": "openrouter_tools_fallback_models",
            "OPENROUTER_SITE_URL": "openrouter_site_url",
            "OPENROUTER_SITE_NAME": "openrouter_site_name",
            "LLM_TEMPERATURE": "llm_temperature",
            "LLM_MAX_TOKENS": "llm_max_tokens",
            "REPLY_BREVITY": "reply_brevity",
            "LISTENING_AI_REPLY_BREVITY": "reply_brevity",
            "TOKEN_EXPIRY_SECONDS": "token_expiry_seconds",
            "DATA_DIR": "data_dir",
            "DB_PATH": "db_path",
            "STORE_BACKEND": "store_backend",
            "DO_SPACES_KEY": "spaces_key",
            "DO_SPACES_SECRET": "spaces_secret",
            "DO_SPACES_REGION": "spaces_region",
            "DO_SPACES_ENDPOINT": "spaces_endpoint",
            "DO_SPACES_BUCKET": "spaces_bucket",
            "S3_PREFIX": "spaces_prefix",
            "SPACES_PREFIX": "spaces_prefix",
            "LISTENING_AI_PREFIX": "spaces_prefix",
            "LISTENING_AI_STORE": "store_backend",
        }
        for src, dest in aliases.items():
            if dest not in kwargs and src in data and data[src] is not None:
                kwargs[dest] = data[src]

        return cls(**kwargs)

    @classmethod
    def from_config_module(cls, module: Any) -> "Settings":
        data = {k: getattr(module, k) for k in dir(module) if k.isupper()}
        return cls.from_mapping(data)

    @classmethod
    def from_env(cls) -> "Settings":
        env = {
            "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
            "OPENROUTER_API_URL": os.environ.get("OPENROUTER_API_URL"),
            "OPENROUTER_MODEL": os.environ.get("OPENROUTER_MODEL"),
            "OPENROUTER_TOOLS_MODEL": os.environ.get("OPENROUTER_TOOLS_MODEL"),
            "STORE_BACKEND": os.environ.get("LISTENING_AI_STORE", "json"),
            "DATA_DIR": os.environ.get("LISTENING_AI_DATA_DIR"),
            "DO_SPACES_KEY": os.environ.get("DO_SPACES_KEY") or os.environ.get("AWS_ACCESS_KEY_ID"),
            "DO_SPACES_SECRET": os.environ.get("DO_SPACES_SECRET") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "DO_SPACES_REGION": os.environ.get("DO_SPACES_REGION") or os.environ.get("AWS_REGION"),
            "DO_SPACES_ENDPOINT": os.environ.get("DO_SPACES_ENDPOINT"),
            "DO_SPACES_BUCKET": os.environ.get("DO_SPACES_BUCKET") or os.environ.get("S3_BUCKET"),
            "S3_PREFIX": os.environ.get("S3_PREFIX") or os.environ.get("LISTENING_AI_PREFIX"),
            "REPLY_BREVITY": os.environ.get("LISTENING_AI_REPLY_BREVITY")
            or os.environ.get("REPLY_BREVITY"),
            "TOKEN_EXPIRY_SECONDS": (
                int(os.environ["LISTENING_AI_TOKEN_EXPIRY"])
                if os.environ.get("LISTENING_AI_TOKEN_EXPIRY") else None
            ),
        }
        # Drop None values so defaults apply
        return cls.from_mapping({k: v for k, v in env.items() if v is not None and v != ""})


_settings: Optional[Settings] = None


def configure(settings: Optional[Settings] = None, **kwargs: Any) -> Settings:
    """Set global library settings. Call once at process startup.

    Accepts either a Settings instance or keyword overrides. If neither is
    given, builds from environment variables.
    """
    global _settings
    if settings is None:
        base = Settings.from_env()
        if kwargs:
            data = asdict(base)
            data.update(kwargs)
            _settings = Settings.from_mapping(data)
        else:
            _settings = base
    else:
        if kwargs:
            data = asdict(settings)
            data.update(kwargs)
            _settings = Settings.from_mapping(data)
        else:
            _settings = settings
    return _settings


def get_settings() -> Settings:
    """Return the active Settings, auto-configuring from env if needed."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
