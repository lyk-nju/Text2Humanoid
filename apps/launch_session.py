from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Text2Humanoid session and send one command.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    req = urllib.request.Request(f"{base}/sessions", method="POST")
    with urllib.request.urlopen(req) as resp:
        session = json.loads(resp.read().decode("utf-8"))
    payload = json.dumps({"text": args.text}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/sessions/{session['session_id']}/commands",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req):
        pass
    print(session["session_id"])


if __name__ == "__main__":
    main()
