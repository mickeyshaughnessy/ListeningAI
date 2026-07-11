# ListeningAI

An AI system that listens better than it speaks.

Reusable framework for AI-first tools built around a conversational control
interface, a tool API, memory, and an agentic chat loop. Extracted from the
pattern that keeps showing up across sibling projects (GreenDial, Robot Services
Exchange, love-matcher, TheUSDX / Acme Redactors, Agreed).

**GreenDial is the reference production deployment.**

## Install

```bash
# Library only
pip install -e .

# With DigitalOcean Spaces / S3 support
pip install -e ".[spaces]"

# Dev (pytest + spaces extra)
pip install -e ".[dev]"

# From another project in this workspace (e.g. GreenDial)
pip install -e ../ListeningAI[spaces]
```

Package name on installs: `listening-ai`. Import name: `listening_ai`.

## Quick start (demo console)

```bash
pip install -e ".[spaces]"
cp config.py.example config.py   # optional; or just export OPENROUTER_API_KEY
export OPENROUTER_API_KEY="sk-or-v1-..."
python api_server.py
```

Open `http://localhost:5099/`. Run `python simulate.py` against a live server for
scripted end-to-end checks.

Unit tests (no network, LLM mocked):

```bash
python -m unittest discover -s tests -v
# or, after pip install -e ".[dev]":
pytest
```

## Architecture

| Module | Role |
|--------|------|
| `listening_ai.settings` | `Settings` dataclass + `configure()` / `get_settings()` |
| `listening_ai.store` | Pluggable persistence: `JsonFileStore`, `SpacesStore` |
| `listening_ai.llm` | OpenRouter `completion` / `call_llm` / `call_llm_with_tools` + model fallbacks |
| `listening_ai.auth` | `Authorization: Bearer` (also accepts `X-Session-Token`) |
| `listening_ai.tools` | `ToolRegistry` + generic default tools |
| `listening_ai.controller` | `ChatController` agentic loop (`run_loop` / `handle_message`) |
| `listening_ai.brevity` | Reply shorten levels (`none` / `short` / `very_short`) + tool-first listening prompts |
| `listening_ai.blueprint` | Flask blueprint: auth, profile, settings, inbox, chat |
| `listening_ai.util` | Small shared helpers (e.g. UTC timestamps) |

## HTTP API (blueprint)

Default routes (no `url_prefix`):

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/ping` | no | Liveness |
| POST | `/register` | no | Create user + session |
| POST | `/login` | no | Session token |
| GET | `/account` | yes | Current user |
| GET/PUT | `/profile` | yes | Freeform profile KV |
| GET/PUT | `/settings` | yes | User preferences |
| GET/POST | `/inbox` | yes | List / send messages |
| POST | `/inbox/<id>/read` | yes | Mark message read |
| GET/POST | `/notifications` | yes | List / create |
| POST | `/notifications/<id>/read` | yes | Mark read |
| DELETE | `/notifications/<id>` | yes | Delete |
| GET | `/chat/sessions` | yes | List chat sessions |
| GET | `/chat/history?session_id=` | yes | Transcript |
| POST | `/chat` | yes | Agentic chat turn |

Auth: `Authorization: Bearer <token>` or `X-Session-Token: <token>`.

The demo server also serves `GET /` (console UI) and `GET /health`.

## Storage backends

### JSON file (default)

```python
from listening_ai import Settings, configure_app

configure_app(Settings(store_backend="json", data_dir="./data"))
```

### DigitalOcean Spaces (S3-compatible)

```python
from listening_ai import Settings, configure_app

configure_app(Settings(
    store_backend="spaces",
    spaces_key="...",
    spaces_secret="...",
    spaces_region="sfo3",
    spaces_endpoint="https://sfo3.digitaloceanspaces.com",
    spaces_bucket="mithril-media",
    spaces_prefix="listening_ai/",   # or greendial/listening_ai/
    openrouter_api_key="...",
))
```

Object layout under the prefix:

```
{prefix}users/{user_id}.json
{prefix}usernames/{username}.json
{prefix}sessions/{token}.json
{prefix}inbox/{user_id}.json
{prefix}notifications/{user_id}.json
{prefix}chats/{session_id}.json
{prefix}chat_index/{user_id}/{session_id}.json
```

You can also implement `listening_ai.store.BaseStore` against an existing host
database and pass it to `configure_app(..., store=my_store)`.

## Wire into a host app

```python
from flask import Flask
from listening_ai import configure_app, create_blueprint, default_registry, Settings
import config  # your existing config module

configure_app(
    Settings.from_config_module(config),
    store_backend="spaces",
    spaces_prefix="myapp/listening_ai/",
)

app = Flask(__name__)
registry = default_registry()
registry.register(
    "submit_bid",
    "Submit a bid on a job for the current user.",
    {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "amount": {"type": "number"},
        },
        "required": ["job_id", "amount"],
    },
    my_submit_bid_handler,  # def handler(user_id, job_id, amount): ...
)
app.register_blueprint(create_blueprint(tool_registry=registry, url_prefix="/listening"))
```

### Reply brevity (listen more than you speak)

```python
from listening_ai import Settings, configure_app, ChatController

configure_app(Settings(
    openrouter_api_key="...",
    reply_brevity="very_short",  # none | short | very_short
))
# Or per controller:
controller = ChatController(tool_registry=registry, reply_brevity="very_short")
```

With `short` / `very_short`:
1. The agent system prompt is extended to **prefer tools over monologue**.
2. The final reply is **parsed into a shorter form** (LLM rewrite with local fallback), keeping tool outcomes.

GreenDial sets `reply_brevity="very_short"` via `listening_bridge.py`.

### Use only the agentic loop (no blueprint)

GreenDial does this for Doc/specialist tool use while keeping its own `/auth`
and `/chat` routes:

```python
from listening_ai import ChatController, ToolRegistry

registry = ToolRegistry()
# register domain tools...
controller = ChatController(tool_registry=registry, system_prompt=DOC_PROMPT)
final_text, model_used, tool_log = controller.run_loop(messages, user_id)
```

### Use only the LLM client (no store / blueprint)

For one-shot completions (suggestions, notifications, plain chat) after
`configure(...)` with your OpenRouter key:

```python
from listening_ai import Settings, configure, completion, call_llm_with_tools

configure(Settings.from_config_module(config))  # or Settings.from_env()

text = completion(prompt="Summarize sleep tips in one sentence.")
# multi-turn / tools:
resp = call_llm_with_tools(messages=[...], tools=schemas, system_prompt="...")
```

## Default tools

Every `default_registry()` installation includes:

- `get_profile` / `update_profile`
- `get_settings` / `update_settings`
- `list_messages` / `send_message`
- `list_notifications` / `create_notification` / `mark_notification_read`

Host apps add domain tools with `registry.register(name, description, parameters, handler)`.

## GreenDial reference deployment

GreenDial installs this package, configures Spaces under
`greendial/listening_ai/`, mounts the blueprint at `/listening`, and routes its
internal agentic health-tool loop through `ChatController.run_loop`. See
`GreenDial/listening_bridge.py`.

## Env vars

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | LLM key |
| `OPENROUTER_API_URL` | Override OpenRouter completions URL |
| `OPENROUTER_MODEL` / `OPENROUTER_TOOLS_MODEL` | Primary models |
| `LISTENING_AI_STORE` | `json` or `spaces` |
| `LISTENING_AI_DATA_DIR` | JSON store directory |
| `LISTENING_AI_PREFIX` | Spaces key prefix |
| `LISTENING_AI_TOKEN_EXPIRY` | Session lifetime (seconds) |
| `DO_SPACES_KEY` / `DO_SPACES_SECRET` | Spaces credentials |
| `DO_SPACES_BUCKET` / `DO_SPACES_REGION` / `DO_SPACES_ENDPOINT` | Spaces location |

## License

MIT — see [LICENSE](LICENSE).
