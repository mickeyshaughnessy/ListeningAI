"""
Integration-test-style simulation, in the spirit of the sibling projects'
demand_monitor.py / supply_monitor.py / int_tests.py / integration_test.py:
a scripted "user" that drives the live API end-to-end over HTTP so the
library can be exercised and improved without a browser.

Usage:
    python api_server.py &
    python simulate.py

Optional env:
    LISTENING_AI_BASE_URL  (default http://localhost:5099)
"""
from __future__ import annotations

import os
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("LISTENING_AI_BASE_URL", "http://localhost:5099").rstrip("/")


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "OK  " if condition else "FAIL"
    suffix = f" — {detail}" if detail and not condition else ""
    print(f"[{status}] {label}{suffix}")
    if not condition:
        sys.exit(1)


def main() -> None:
    session = requests.Session()
    username = f"sim_user_{uuid.uuid4().hex[:8]}"
    password = "hunter2-test"

    try:
        r = session.get(f"{BASE_URL}/ping", timeout=5)
    except requests.exceptions.RequestException as e:
        print(f"Cannot reach {BASE_URL}: {e}")
        print("Start the demo server first: python api_server.py")
        sys.exit(1)

    check("ping", r.status_code == 200)

    try:
        health = session.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code == 200:
            body = health.json()
            print(f"      health: store={body.get('store')} version={body.get('version')}")
    except requests.exceptions.RequestException:
        pass

    r = session.post(
        f"{BASE_URL}/register",
        json={"username": username, "password": password},
        timeout=10,
    )
    check("register", r.status_code == 201, r.text[:200])
    token = r.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    r = session.post(
        f"{BASE_URL}/login",
        json={"username": username, "password": password},
        timeout=10,
    )
    check("login", r.status_code == 200)

    r = session.get(f"{BASE_URL}/account", headers=headers, timeout=10)
    check("account (authed)", r.status_code == 200 and r.json()["username"] == username)

    r = session.get(f"{BASE_URL}/account", timeout=10)
    check("account rejects missing token", r.status_code == 401)

    r = session.put(
        f"{BASE_URL}/profile",
        headers=headers,
        json={"updates": {"name": "Sim User"}},
        timeout=10,
    )
    check("update profile", r.status_code == 200 and r.json()["profile"]["name"] == "Sim User")

    r = session.get(f"{BASE_URL}/profile", headers=headers, timeout=10)
    check("get profile", r.status_code == 200 and r.json()["profile"]["name"] == "Sim User")

    r = session.put(
        f"{BASE_URL}/settings",
        headers=headers,
        json={"updates": {"tone": "concise"}},
        timeout=10,
    )
    check("update settings", r.status_code == 200 and r.json()["settings"]["tone"] == "concise")

    r = session.post(
        f"{BASE_URL}/notifications",
        headers=headers,
        json={"text": "Test reminder"},
        timeout=10,
    )
    check("create notification", r.status_code == 201)
    notif_id = r.json()["notification"]["id"]

    r = session.get(f"{BASE_URL}/notifications", headers=headers, timeout=10)
    check("list notifications", r.status_code == 200 and len(r.json()["notifications"]) >= 1)

    r = session.post(f"{BASE_URL}/notifications/{notif_id}/read", headers=headers, timeout=10)
    check("mark notification read", r.status_code == 200 and r.json()["notification"]["read"] is True)

    # second user to exercise inbox messaging between users
    other_username = f"sim_user_{uuid.uuid4().hex[:8]}"
    r = session.post(
        f"{BASE_URL}/register",
        json={"username": other_username, "password": password},
        timeout=10,
    )
    check("register second user", r.status_code == 201)

    r = session.post(
        f"{BASE_URL}/inbox",
        headers=headers,
        json={"to_username": other_username, "body": "hello!"},
        timeout=10,
    )
    check("send inbox message", r.status_code == 201)

    other_token = requests.post(
        f"{BASE_URL}/login",
        json={"username": other_username, "password": password},
        timeout=10,
    ).json()["token"]
    r = session.get(
        f"{BASE_URL}/inbox",
        headers={"Authorization": f"Bearer {other_token}"},
        timeout=10,
    )
    check("recipient sees message", r.status_code == 200 and len(r.json()["messages"]) == 1)

    # --- chat / agentic tool use ---
    print("\n--- chat (requires OPENROUTER_API_KEY to be set) ---")
    r = session.post(
        f"{BASE_URL}/chat",
        headers=headers,
        json={"message": "Please remember that my favorite color is teal."},
        timeout=90,
    )
    if r.status_code != 200:
        print(f"[SKIP] chat endpoint returned {r.status_code}: {r.text[:200]}")
    else:
        data = r.json()
        check("chat returns a session_id", bool(data.get("session_id")))
        print(f"assistant: {data['response']}")
        print(f"tool_calls: {data.get('tool_calls')}")
        session_id = data["session_id"]

        time.sleep(1)
        r = session.post(
            f"{BASE_URL}/chat",
            headers=headers,
            json={"message": "What's my favorite color?", "session_id": session_id},
            timeout=90,
        )
        data = r.json()
        print(f"assistant: {data['response']}")
        check("chat keeps session continuity", data["session_id"] == session_id)

        r = session.get(
            f"{BASE_URL}/chat/history",
            headers=headers,
            params={"session_id": session_id},
            timeout=10,
        )
        check("chat history readable", r.status_code == 200 and len(r.json()["messages"]) >= 4)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
