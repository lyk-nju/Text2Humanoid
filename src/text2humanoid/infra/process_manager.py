from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class ManagedProcess:
    name: str
    proc: subprocess.Popen


class ProcessManager:
    def __init__(self) -> None:
        self._processes: dict[str, ManagedProcess] = {}

    def spawn(self, name: str, cmd: list[str], cwd: str | None = None) -> None:
        self._processes[name] = ManagedProcess(name=name, proc=subprocess.Popen(cmd, cwd=cwd))

    def stop(self, name: str) -> None:
        proc = self._processes.pop(name, None)
        if proc is None:
            return
        proc.proc.terminate()

    def poll(self, name: str) -> int | None:
        proc = self._processes.get(name)
        if proc is None:
            return None
        return proc.proc.poll()
