from __future__ import annotations

from text2humanoid.contracts.status import SessionPhase


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    SessionPhase.IDLE.value: {SessionPhase.WARMING.value, SessionPhase.STOPPED.value},
    SessionPhase.WARMING.value: {SessionPhase.RUNNING.value, SessionPhase.ERROR.value, SessionPhase.STOPPED.value},
    SessionPhase.RUNNING.value: {
        SessionPhase.DEGRADED.value,
        SessionPhase.RESETTING.value,
        SessionPhase.STOPPED.value,
        SessionPhase.ERROR.value,
    },
    SessionPhase.DEGRADED.value: {
        SessionPhase.RUNNING.value,
        SessionPhase.RESETTING.value,
        SessionPhase.STOPPED.value,
        SessionPhase.ERROR.value,
    },
    SessionPhase.RESETTING.value: {SessionPhase.WARMING.value, SessionPhase.STOPPED.value, SessionPhase.ERROR.value},
    SessionPhase.ERROR.value: {SessionPhase.RESETTING.value, SessionPhase.STOPPED.value},
    SessionPhase.STOPPED.value: set(),
}


def can_transition(src: str, dst: str) -> bool:
    return dst in ALLOWED_TRANSITIONS.get(src, set())
