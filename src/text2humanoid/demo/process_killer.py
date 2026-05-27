from __future__ import annotations

from dataclasses import dataclass
import os
import signal
import subprocess
import time
from typing import Callable


_DEMO_COMMAND_MARKERS = (
    "apps/demo_console_server.py",
    "apps/launch_full_demo.py",
    "apps/replay_bfmzero_chunk.py",
    "apps/run_text_to_bfmzero.py",
    "rl_policy/bfm_zero.py",
    "sim_env.base_sim",
    "generate_ldf.py",
)
_STOPPER_MARKERS = (
    "apps/app_stop.py",
    "grep ",
    "rg ",
)


@dataclass(frozen=True, slots=True)
class DemoProcess:
    pid: int
    command: str


@dataclass(frozen=True, slots=True)
class StopResult:
    matched: int
    signaled: int
    processes: tuple[DemoProcess, ...]


def is_demo_process(command: str) -> bool:
    if any(marker in command for marker in _STOPPER_MARKERS):
        return False
    return any(marker in command for marker in _DEMO_COMMAND_MARKERS)


def parse_ps_output(ps_output: str, *, current_pid: int | None = None) -> list[DemoProcess]:
    current_pid = os.getpid() if current_pid is None else current_pid
    processes: list[DemoProcess] = []
    for raw_line in ps_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        command = command.strip()
        if is_demo_process(command):
            processes.append(DemoProcess(pid=pid, command=command))
    return processes


def read_process_table() -> str:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout


def wait_for_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def stop_demo_processes(
    *,
    ps_output: str | None = None,
    kill_func: Callable[[int, int], None] = os.kill,
    wait_func: Callable[[int, float], bool] = wait_for_exit,
    dry_run: bool = False,
    force: bool = False,
    timeout_sec: float = 3.0,
    current_pid: int | None = None,
) -> StopResult:
    table = read_process_table() if ps_output is None else ps_output
    processes = tuple(parse_ps_output(table, current_pid=current_pid))
    signaled = 0
    if dry_run:
        return StopResult(matched=len(processes), signaled=0, processes=processes)

    for process in processes:
        try:
            kill_func(process.pid, signal.SIGTERM)
            signaled += 1
        except ProcessLookupError:
            continue
        if force and not wait_func(process.pid, timeout_sec):
            try:
                kill_func(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return StopResult(matched=len(processes), signaled=signaled, processes=processes)
