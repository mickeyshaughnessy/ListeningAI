"""
Local dev/test server for the listening_ai library.

Run with:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    pip install -e ".[spaces]"   # or: pip install -e .
    python api_server.py

Then open index.html (served at / by this same app) to chat against it.

Storage backend is selected via config / env:
    LISTENING_AI_STORE=json     (default, local data/db.json)
    LISTENING_AI_STORE=spaces   (DigitalOcean Spaces / S3)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from flask import Flask, send_from_directory
from flask_cors import CORS

from listening_ai import Settings, configure_app, create_blueprint, default_registry

# Optional: load sibling-style config.py if present
_config: Optional[Any] = None
try:
    import config as _config  # type: ignore[no-redef]
    _settings = Settings.from_config_module(_config)
except ImportError:
    _settings = Settings.from_env()

# Allow env to force store backend without editing config
if os.environ.get("LISTENING_AI_STORE"):
    _settings.store_backend = os.environ["LISTENING_AI_STORE"]

configure_app(_settings)

app = Flask(__name__, static_folder=None)
CORS(app)

registry = default_registry()


def _echo_action(user_id, action, **params):
    """
    Example custom tool. Real integrations replace this with domain actions
    (submit_bid, load_preset, etc.).
    """
    return f"(demo) would perform action {action!r} with params {params} for {user_id}"


registry.register(
    "demo_site_action",
    "Example placeholder for a project-specific action (e.g. submitting a bid, "
    "adjusting a simulation control). Replace with real tools for your site.",
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Name of the action to perform"},
        },
        "required": ["action"],
    },
    _echo_action,
)

app.register_blueprint(create_blueprint(tool_registry=registry))


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/health")
def health():
    from listening_ai import get_settings, get_store

    s = get_settings()
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "store": type(get_store()).__name__,
        "store_backend": s.store_backend,
        "version": __import__("listening_ai").__version__,
    }


def _resolve_run_options() -> tuple[str, int, bool]:
    host = "0.0.0.0"
    port = 5099
    debug = True
    if _config is not None:
        host = getattr(_config, "FLASK_HOST", host)
        port = int(getattr(_config, "FLASK_PORT", port))
        debug = bool(getattr(_config, "DEBUG", debug))
    return host, port, debug


def _warn_if_missing_api_key() -> None:
    key = ""
    if _config is not None:
        key = getattr(_config, "OPENROUTER_API_KEY", "") or ""
    if not key and not os.environ.get("OPENROUTER_API_KEY"):
        print("WARNING: OPENROUTER_API_KEY is not set — chat calls will fail.")


if __name__ == "__main__":
    host, port, debug = _resolve_run_options()
    _warn_if_missing_api_key()
    print(f"ListeningAI demo on http://{host}:{port}  store={_settings.store_backend}")
    app.run(host=host, port=port, debug=debug)
