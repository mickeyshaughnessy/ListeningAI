"""
Token auth using ``Authorization: Bearer <token>``.

Matches the majority pattern across sibling projects (love-matcher, Agreed,
TheUSDX, theservicesexchange). Resolves tokens via the active store.
"""
from functools import wraps

from flask import request, jsonify

from . import store


def get_token_from_request():
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[len("Bearer "):].strip()
    # Also accept X-Session-Token for GreenDial-style clients
    alt = request.headers.get("X-Session-Token", "")
    if alt:
        return alt.strip()
    return None


def get_current_user_id():
    token = get_token_from_request()
    if not token:
        return None
    return store.get_user_id_for_token(token)


def require_auth(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"error": "Unauthorized"}), 401
        request.user_id = user_id
        return view_func(*args, **kwargs)
    return wrapped
