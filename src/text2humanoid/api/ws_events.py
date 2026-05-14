from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class WebSocketHub:
    def __init__(self) -> None:
        self._clients: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._clients[session_id].append(ws)

    def disconnect(self, session_id: str, ws: WebSocket) -> None:
        if session_id not in self._clients:
            return
        self._clients[session_id] = [client for client in self._clients[session_id] if client is not ws]
        if not self._clients[session_id]:
            self._clients.pop(session_id, None)

    async def broadcast(self, session_id: str, event: dict[str, Any]) -> None:
        for ws in list(self._clients.get(session_id, [])):
            await ws.send_json(event)
