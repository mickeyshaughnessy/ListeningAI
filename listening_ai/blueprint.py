"""
Flask blueprint exposing the reusable ListeningAI surface: auth, profile,
settings, inbox, notifications, and the core chat/agentic endpoints.

Usage from a host project::

    from flask import Flask
    from listening_ai import configure_app, create_blueprint, default_registry

    configure_app(...)  # settings + store
    app = Flask(__name__)
    registry = default_registry()
    registry.register("submit_bid", "...", {...}, my_handler)
    app.register_blueprint(create_blueprint(tool_registry=registry))
"""
from flask import Blueprint, request, jsonify

from . import store
from .auth import require_auth
from .controller import ChatController
from .tools import default_registry


def create_blueprint(tool_registry=None, system_prompt=None, url_prefix="", name="listening_ai"):
    bp = Blueprint(name, __name__, url_prefix=url_prefix)
    registry = tool_registry if tool_registry is not None else default_registry()
    controller = ChatController(tool_registry=registry, system_prompt=system_prompt)

    # ---------------------------------------------------------- system --

    @bp.route("/ping", methods=["GET"])
    def ping():
        return jsonify({"status": "ok", "service": "listening_ai"})

    # ------------------------------------------------------------ auth --

    @bp.route("/register", methods=["POST"])
    def register():
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400
        if len(username) > 64:
            return jsonify({"error": "username must be 64 characters or fewer"}), 400
        if len(password) < 4:
            return jsonify({"error": "password must be at least 4 characters"}), 400

        user = store.create_user(username, password)
        if not user:
            return jsonify({"error": "username already taken"}), 409

        token = store.create_session(user["user_id"])
        return jsonify({
            "user_id": user["user_id"],
            "username": user["username"],
            "token": token,
            "profile": user["profile"],
            "settings": user["settings"],
        }), 201

    @bp.route("/login", methods=["POST"])
    def login():
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        user = store.verify_login(username, password)
        if not user:
            return jsonify({"error": "invalid username or password"}), 401

        token = store.create_session(user["user_id"])
        return jsonify({
            "user_id": user["user_id"],
            "username": user["username"],
            "token": token,
            "profile": user["profile"],
            "settings": user["settings"],
        })

    @bp.route("/account", methods=["GET"])
    @require_auth
    def account():
        user = store.get_user(request.user_id)
        if not user:
            return jsonify({"error": "not found"}), 404
        return jsonify({
            "user_id": user["user_id"],
            "username": user["username"],
            "profile": user.get("profile", {}),
            "settings": user.get("settings", {}),
            "created_at": user.get("created_at") or user.get("created"),
        })

    # --------------------------------------------------------- profile --

    @bp.route("/profile", methods=["GET"])
    @require_auth
    def get_profile():
        return jsonify({"profile": store.get_profile(request.user_id)})

    @bp.route("/profile", methods=["PUT"])
    @require_auth
    def put_profile():
        data = request.get_json(silent=True) or {}
        updates = data.get("updates", data)
        profile = store.update_profile(request.user_id, updates)
        return jsonify({"profile": profile})

    # -------------------------------------------------------- settings --

    @bp.route("/settings", methods=["GET"])
    @require_auth
    def get_settings_route():
        return jsonify({"settings": store.get_settings(request.user_id)})

    @bp.route("/settings", methods=["PUT"])
    @require_auth
    def put_settings():
        data = request.get_json(silent=True) or {}
        updates = data.get("updates", data)
        settings = store.update_settings(request.user_id, updates)
        return jsonify({"settings": settings})

    # ----------------------------------------------------------- inbox --

    @bp.route("/inbox", methods=["GET"])
    @require_auth
    def get_inbox():
        return jsonify({"messages": store.list_messages(request.user_id)})

    @bp.route("/inbox", methods=["POST"])
    @require_auth
    def post_inbox():
        data = request.get_json(silent=True) or {}
        to_username = (data.get("to_username") or "").strip()
        body = data.get("body") or ""
        if not to_username or not body:
            return jsonify({"error": "to_username and body are required"}), 400
        recipient_id = store.get_user_id_by_username(to_username)
        if not recipient_id:
            return jsonify({"error": f"no user named '{to_username}'"}), 404
        sender = store.get_user(request.user_id)
        message = store.add_message(recipient_id, sender["username"], body)
        return jsonify({"message": message}), 201

    @bp.route("/inbox/<message_id>/read", methods=["POST"])
    @require_auth
    def read_inbox_message(message_id):
        message = store.mark_message_read(request.user_id, message_id)
        if not message:
            return jsonify({"error": "not found"}), 404
        return jsonify({"message": message})

    # ---------------------------------------------------- notifications --

    @bp.route("/notifications", methods=["GET"])
    @require_auth
    def get_notifications():
        return jsonify({"notifications": store.list_notifications(request.user_id)})

    @bp.route("/notifications", methods=["POST"])
    @require_auth
    def post_notification():
        data = request.get_json(silent=True) or {}
        text = data.get("text") or ""
        if not text:
            return jsonify({"error": "text is required"}), 400
        notification = store.add_notification(request.user_id, text, meta=data.get("meta"))
        return jsonify({"notification": notification}), 201

    @bp.route("/notifications/<notification_id>/read", methods=["POST"])
    @require_auth
    def read_notification(notification_id):
        notification = store.mark_notification_read(request.user_id, notification_id)
        if not notification:
            return jsonify({"error": "not found"}), 404
        return jsonify({"notification": notification})

    @bp.route("/notifications/<notification_id>", methods=["DELETE"])
    @require_auth
    def delete_notification(notification_id):
        deleted = store.delete_notification(request.user_id, notification_id)
        return jsonify({"deleted": deleted})

    # -------------------------------------------------------------- chat --

    @bp.route("/chat/sessions", methods=["GET"])
    @require_auth
    def list_chat_sessions():
        sessions = store.list_chat_sessions(request.user_id)
        return jsonify({"sessions": [
            {
                "id": s["id"],
                "agent_id": s.get("agent_id", "default"),
                "created_at": s.get("created_at"),
                "updated_at": s.get("updated_at"),
                "message_count": len(s.get("messages") or []),
            }
            for s in sessions
        ]})

    @bp.route("/chat/history", methods=["GET"])
    @require_auth
    def chat_history():
        session_id = request.args.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id query param is required"}), 400
        chat = store.get_chat_session(session_id)
        if not chat or chat["user_id"] != request.user_id:
            return jsonify({"error": "not found"}), 404
        return jsonify({"session_id": session_id, "messages": chat["messages"]})

    @bp.route("/chat", methods=["POST"])
    @require_auth
    def chat():
        data = request.get_json(silent=True) or {}
        message = data.get("message")
        if not message:
            return jsonify({"error": "message is required"}), 400

        session_id = data.get("session_id")
        if session_id:
            chat_session = store.get_chat_session(session_id)
            if not chat_session or chat_session["user_id"] != request.user_id:
                return jsonify({"error": "invalid session_id"}), 404
        else:
            session_id = controller.start_session(
                request.user_id, agent_id=data.get("agent_id", "default")
            )

        result = controller.handle_message(request.user_id, session_id, message)
        return jsonify(result)

    return bp
