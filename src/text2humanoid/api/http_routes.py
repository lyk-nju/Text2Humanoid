from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from text2humanoid.api.schemas import CreateSessionResponse, PromptCommandSchema, StatusResponse
from text2humanoid.api.ws_events import WebSocketHub
from text2humanoid.contracts.commands import PromptCommand, TrajectoryCondition, TrajectoryPoint
from text2humanoid.infra.artifact_store import ArtifactStore
from text2humanoid.orchestrator.session_manager import SessionManager


def _to_contract(payload: PromptCommandSchema) -> PromptCommand:
    trajectory = None
    if payload.trajectory is not None:
        trajectory = TrajectoryCondition(
            waypoints=[TrajectoryPoint(**point.model_dump()) for point in payload.trajectory.waypoints],
            token_aligned_traj=payload.trajectory.token_aligned_traj,
            token_mask=payload.trajectory.token_mask,
            metadata=dict(payload.trajectory.metadata),
        )
    return PromptCommand(
        text=payload.text,
        trajectory=trajectory,
        submit_time=payload.submit_time,
        transition_mode=payload.transition_mode,
        command_id=payload.command_id,
        metadata=dict(payload.metadata),
    )


def create_app(session_manager: SessionManager, artifact_store: ArtifactStore) -> FastAPI:
    app = FastAPI(title="Text2Humanoid API")
    hub = WebSocketHub()
    app.state.session_manager = session_manager
    app.state.artifact_store = artifact_store
    app.state.ws_hub = hub

    @app.post("/sessions", response_model=CreateSessionResponse)
    async def create_session() -> CreateSessionResponse:
        session_id = session_manager.create_session()
        return CreateSessionResponse(session_id=session_id)

    @app.get("/sessions/{session_id}/status", response_model=StatusResponse)
    async def get_status(session_id: str) -> StatusResponse:
        status = session_manager.get_status(session_id)
        return StatusResponse(**status.to_dict())

    @app.post("/sessions/{session_id}/commands")
    async def push_command(session_id: str, payload: PromptCommandSchema) -> dict[str, str]:
        command = _to_contract(payload)
        session_manager.push_command(session_id, command)
        await hub.broadcast(session_id, {"type": "command.accepted", "command": command.to_dict()})
        return {"status": "ok"}

    @app.post("/sessions/{session_id}/reset")
    async def reset_session(session_id: str) -> dict[str, str]:
        session_manager.reset_session(session_id)
        await hub.broadcast(session_id, {"type": "session.reset"})
        return {"status": "ok"}

    @app.post("/sessions/{session_id}/stop")
    async def stop_session(session_id: str) -> dict[str, str]:
        session_manager.stop_session(session_id)
        await hub.broadcast(session_id, {"type": "session.stopped"})
        return {"status": "ok"}

    @app.post("/sessions/{session_id}/export")
    async def export_session(session_id: str) -> dict[str, str]:
        export_path = artifact_store.export_status_bundle(session_id, session_manager.get_status(session_id))
        return {"status": "ok", "path": str(export_path)}

    @app.post("/sessions/{session_id}/refill/start")
    async def start_refill(session_id: str) -> dict[str, str]:
        session_manager.start_refill_loop(session_id)
        return {"status": "ok", "session_id": session_id}

    @app.post("/sessions/{session_id}/refill/stop")
    async def stop_refill(session_id: str) -> dict[str, str]:
        session_manager.stop_refill_loop(session_id)
        return {"status": "ok", "session_id": session_id}

    @app.websocket("/ws/{session_id}")
    async def status_stream(ws: WebSocket, session_id: str) -> None:
        await hub.connect(session_id, ws)
        try:
            await ws.send_json({"type": "status", "payload": session_manager.get_status(session_id).to_dict()})
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(session_id, ws)

    return app
