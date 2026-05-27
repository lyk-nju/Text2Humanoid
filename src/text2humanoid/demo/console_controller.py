from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
from typing import Callable, Any

from .stream_preview import Streaming263DPreviewBuffer


_TEXT2HUMANOID_DIR = Path(__file__).resolve().parents[3]
_TEXT2MOTION_ROOT = _TEXT2HUMANOID_DIR.parent
_APPS_DIR = _TEXT2HUMANOID_DIR / "apps"
_SRC_DIR = _TEXT2HUMANOID_DIR / "src"
_DEFAULT_PYTHON = "/home/lai/anaconda3/envs/flooddiffusion/bin/python"


def _default_stream_metrics() -> dict[str, dict[str, float | int]]:
    return {
        "generation": {"fps": 0.0, "buffer_frames": 0},
        "retarget": {"fps": 0.0, "buffer_frames": 0},
        "motion_control": {"fps": 0.0, "buffer_frames": 0},
    }


@dataclass(slots=True)
class GenerateJobRequest:
    text: str
    out_prefix: str = "demo_console"
    generation_steps: int = 150
    frames: int | None = 300
    inspect: bool = False
    auto_push: bool = True
    mark_end: bool = True
    no_realtime: bool = False


@dataclass(slots=True)
class StreamJobRequest:
    text: str
    history_length: int = 30
    denoise_steps: int = 10
    chunk_frames: int = 16
    context_frames: int = 30
    startup_buffer_frames: int = 25
    output_fps: float = 50.0


class _ManagedProcess:
    def __init__(self, name: str, proc, log_file) -> None:
        self.name = name
        self.proc = proc
        self.log_file = log_file

    def close_log(self) -> None:
        try:
            self.log_file.close()
        except Exception:
            pass


class _StreamSession:
    def __init__(self, request: StreamJobRequest) -> None:
        self.request = request
        self.stream_id = uuid.uuid4().hex[:12]
        self.stop_event = threading.Event()
        self._lock = threading.RLock()
        self._text = request.text
        self.thread: threading.Thread | None = None
        self.error = ""

    def should_stop(self) -> bool:
        return self.stop_event.is_set()

    def current_text(self) -> str:
        with self._lock:
            return self._text

    def update_text(self, text: str) -> None:
        with self._lock:
            self._text = text


class DemoController:
    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        video_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        bfmzero_root: str | Path | None = None,
        bfmzero_python: str = _DEFAULT_PYTHON,
        text2humanoid_python: str = _DEFAULT_PYTHON,
        flooddiffusion_python: str = _DEFAULT_PYTHON,
        run_text_to_bfmzero: Callable[[list[str]], int] | None = None,
        replay_bfmzero_chunk: Callable[[list[str]], int] | None = None,
        stream_to_bfmzero: Callable[[StreamJobRequest, Callable[[], bool], Callable[[], str]], int] | None = None,
    ) -> None:
        self.output_dir = self._resolve_text2humanoid_path(output_dir or "assets/saved")
        self.video_dir = self._resolve_text2humanoid_path(video_dir or "assets/video")
        self.log_dir = self._resolve_text2humanoid_path(log_dir or "artifacts/logs/demo_console")
        self.bfmzero_root = Path(bfmzero_root or (_TEXT2MOTION_ROOT / "BFM-Zero")).expanduser().resolve()
        self.bfmzero_python = bfmzero_python
        self.text2humanoid_python = text2humanoid_python
        self.flooddiffusion_python = flooddiffusion_python
        self.run_text_to_bfmzero = run_text_to_bfmzero or self._run_text_to_bfmzero_subprocess
        self.replay_bfmzero_chunk = replay_bfmzero_chunk or self._replay_bfmzero_chunk_subprocess
        self.stream_to_bfmzero = stream_to_bfmzero or self._stream_to_bfmzero_default
        self._stream_runner = None
        self._stream_preview = Streaming263DPreviewBuffer()
        self._stream_metrics = _default_stream_metrics()

        self.keyboard_command_file = self.log_dir / "bfmzero_policy.keys"
        self.sim_keyboard_command_file = self.log_dir / "bfmzero_sim.keys"
        self._processes: dict[str, _ManagedProcess] = {}
        self._events: list[dict] = []
        self._lock = threading.RLock()
        self._stage = "idle"
        self._latest_artifacts: dict[str, str] = {}
        self._latest_job_id = ""
        self._task_thread: threading.Thread | None = None
        self._stream_session: _StreamSession | None = None
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _run_text2humanoid_app(self, app_name: str, argv: list[str]) -> int:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_SRC_DIR)
        result = subprocess.run(
            [self.text2humanoid_python, str(_APPS_DIR / app_name), *argv],
            cwd=_TEXT2HUMANOID_DIR,
            env=env,
        )
        return int(result.returncode)

    def _run_text_to_bfmzero_subprocess(self, argv: list[str]) -> int:
        return self._run_text2humanoid_app("run_text_to_bfmzero.py", argv)

    def _replay_bfmzero_chunk_subprocess(self, argv: list[str]) -> int:
        return self._run_text2humanoid_app("replay_bfmzero_chunk.py", argv)

    def _stream_to_bfmzero_default(
        self,
        request: StreamJobRequest,
        should_stop: Callable[[], bool],
        current_text: Callable[[], str],
    ) -> int:
        if self._stream_runner is None:
            from text2humanoid.streaming.text_to_bfmzero import build_streaming_text_to_bfmzero_runner

            self._stream_runner = build_streaming_text_to_bfmzero_runner(
                progress_callback=self._set_stage,
                generated_motion_callback=self._record_stream_preview_motion,
                metrics_callback=self._record_stream_metrics,
            )
        return int(self._stream_runner(request, should_stop, current_text))

    def _resolve_text2humanoid_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (_TEXT2HUMANOID_DIR / path).resolve()

    def _set_stage(self, stage: str, message: str) -> None:
        with self._lock:
            self._stage = stage
            self._events.append(
                {
                    "id": len(self._events),
                    "ts": time.time(),
                    "stage": stage,
                    "message": message,
                }
            )

    def _record_stream_preview_motion(self, motion: Any) -> None:
        try:
            appended = self._stream_preview.append_motion(motion)
            if appended:
                status = self._stream_preview.status()
                self._set_stage(
                    "stream_preview",
                    f"Queued 263D preview frames={appended} queued_frames={status['queued_frames']}",
                )
        except Exception as exc:
            self._set_stage("stream_preview_failed", str(exc))

    def _record_stream_metrics(self, metrics: dict[str, dict[str, float | int]]) -> None:
        with self._lock:
            merged = _default_stream_metrics()
            for section, values in metrics.items():
                if section in merged:
                    merged[section].update(values)
            self._stream_metrics = merged

    def _spawn(self, name: str, cmd: list[str], cwd: Path) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{name}.log"
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=log_file, stderr=log_file, text=True)
        self._processes[name] = _ManagedProcess(name=name, proc=proc, log_file=log_file)
        self._set_stage("process", f"Started {name}: {log_path}")

    def start_sim(self) -> dict:
        if "sim" not in self._processes or self._processes["sim"].proc.poll() is not None:
            if self.sim_keyboard_command_file.exists():
                self.sim_keyboard_command_file.unlink()
            self._spawn(
                "sim",
                [
                    self.bfmzero_python,
                    "-m",
                    "sim_env.base_sim",
                    "--robot_config",
                    "./config/robot/g1.yaml",
                    "--scene_config",
                    "./config/scene/g1_29dof.yaml",
                    "--keyboard_command_file",
                    str(self.sim_keyboard_command_file),
                ],
                self.bfmzero_root,
            )
        return {"status": "ok"}

    def start_policy(self) -> dict:
        if self.keyboard_command_file.exists():
            self.keyboard_command_file.unlink()
        if "policy" not in self._processes or self._processes["policy"].proc.poll() is not None:
            self._spawn(
                "policy",
                [
                    self.bfmzero_python,
                    "rl_policy/bfm_zero.py",
                    "--robot_config",
                    "config/robot/g1.yaml",
                    "--policy_config",
                    "config/policy/motivo_newG1.yaml",
                    "--model_path",
                    "./model/exported/FBcprAuxModel_policy_test.onnx",
                    "--task",
                    "config/exp/tracking_online/walking.yaml",
                    "--keyboard_command_file",
                    str(self.keyboard_command_file),
                ],
                self.bfmzero_root,
            )
        return {"status": "ok"}

    def send_policy_key(self, key: str) -> dict:
        if key not in {"i", "]", "[", "p", "o"}:
            raise ValueError(f"Unsupported policy key: {key}")
        self.keyboard_command_file.parent.mkdir(parents=True, exist_ok=True)
        with self.keyboard_command_file.open("a", encoding="utf-8") as f:
            f.write(f"{key}\n")
        self._set_stage("policy_key", f"Sent policy key {key}")
        return {"status": "ok", "key": key}

    def send_sim_key(self, key: str) -> dict:
        if key not in {"9"}:
            raise ValueError(f"Unsupported sim key: {key}")
        self.sim_keyboard_command_file.parent.mkdir(parents=True, exist_ok=True)
        with self.sim_keyboard_command_file.open("a", encoding="utf-8") as f:
            f.write(f"{key}\n")
        self._set_stage("sim_key", f"Sent sim key {key}")
        return {"status": "ok", "key": key}

    def _output_prefix(self, request: GenerateJobRequest) -> Path:
        path = Path(request.out_prefix).expanduser()
        if path.is_absolute():
            return path
        return self.output_dir / path

    def _build_generate_args(self, request: GenerateJobRequest, prefix: Path) -> list[str]:
        args = [
            "--text",
            request.text,
            "--out-prefix",
            str(prefix),
            "--generation-steps",
            str(request.generation_steps),
            "--python",
            self.flooddiffusion_python,
            "--dry-run",
            "--video-dir",
            str(self.video_dir),
        ]
        if request.frames is not None:
            args.extend(["--frames", str(request.frames)])
        if request.inspect:
            args.append("--inspect")
        return args

    def _build_replay_args(self, request: GenerateJobRequest, bfmzero_path: Path) -> list[str]:
        args = ["--input", str(bfmzero_path)]
        if request.mark_end:
            args.append("--mark-end")
        if request.no_realtime:
            args.append("--no-realtime")
        return args

    def generate_and_push(self, request: GenerateJobRequest) -> dict:
        job_id = uuid.uuid4().hex[:12]
        self._latest_job_id = job_id
        prefix = self._output_prefix(request)
        bfmzero_path = prefix.with_name(f"{prefix.name}_bfmzero.npz")

        try:
            self._set_stage("preparing_generation", "Preparing generation")
            self._set_stage("loading_t5", "Loading T5 / model assets")
            self._set_stage("encoding_text", "Encoding text")
            self._set_stage("generating_263d", "Generating 263D motion")
            rc = self.run_text_to_bfmzero(self._build_generate_args(request, prefix))
            if rc != 0:
                raise RuntimeError(f"run_text_to_bfmzero failed with code {rc}")

            self._latest_artifacts = {
                "motion_263": str(prefix.with_name(f"{prefix.name}_motion_263.npz")),
                "motion_140": str(prefix.with_name(f"{prefix.name}_motion_140.npy")),
                "retarget": str(prefix.with_name(f"{prefix.name}_mte_filter_true.npz")),
                "bfmzero": str(bfmzero_path),
                "video_263": str(self.video_dir / f"{prefix.name}_motion_263.mp4"),
                "video_140": str(self.video_dir / f"{prefix.name}_motion_140.mp4"),
                "video_retarget": str(self.video_dir / f"{prefix.name}_retarget.mp4"),
                "video_diagnostics": str(self.video_dir / f"{prefix.name}_retarget_diagnostics.mp4"),
            }
            self._set_stage("retargeting", "Retargeting complete")
            if request.inspect:
                self._set_stage("rendering", "Rendering inspect videos complete")

            if request.auto_push:
                self._set_stage("pushing", "Pushing latest motion to BFM-Zero")
                rc = self.replay_bfmzero_chunk(self._build_replay_args(request, bfmzero_path))
                if rc != 0:
                    raise RuntimeError(f"replay_bfmzero_chunk failed with code {rc}")

            self._set_stage("finished", "Finish")
            return {"status": "finished", "job_id": job_id, "artifacts": dict(self._latest_artifacts)}
        except Exception as exc:
            self._set_stage("failed", str(exc))
            return {"status": "failed", "job_id": job_id, "error": str(exc)}

    def start_generate_task(self, request: GenerateJobRequest) -> dict:
        if self._task_thread is not None and self._task_thread.is_alive():
            return {"status": "busy", "job_id": self._latest_job_id}
        job_id = uuid.uuid4().hex[:12]
        self._latest_job_id = job_id
        self._task_thread = threading.Thread(target=self.generate_and_push, args=(request,), daemon=True)
        self._task_thread.start()
        return {"status": "accepted", "job_id": job_id}

    def _run_stream_task(self, session: _StreamSession) -> None:
        try:
            self._set_stage("stream_running", f"Streaming text: {session.current_text()}")
            rc = self.stream_to_bfmzero(session.request, session.should_stop, session.current_text)
            if rc != 0:
                raise RuntimeError(f"stream_to_bfmzero failed with code {rc}")
            self._set_stage("stream_stopped", "Stream stopped")
        except Exception as exc:
            session.error = str(exc)
            self._set_stage("stream_failed", str(exc))

    def start_stream_task(self, request: StreamJobRequest) -> dict:
        if self._stream_session is not None:
            thread = self._stream_session.thread
            if thread is not None and thread.is_alive():
                return {"status": "busy", "stream_id": self._stream_session.stream_id}

        session = _StreamSession(request)
        self._stream_preview.reset()
        self._record_stream_metrics(_default_stream_metrics())
        session.thread = threading.Thread(target=self._run_stream_task, args=(session,), daemon=True)
        self._stream_session = session
        self._set_stage("stream_starting", f"Starting stream: {request.text}")
        session.thread.start()
        return {"status": "accepted", "stream_id": session.stream_id}

    def update_stream_text(self, text: str) -> dict:
        if self._stream_session is None:
            return {"status": "idle", "error": "no active stream"}
        thread = self._stream_session.thread
        if thread is None or not thread.is_alive():
            return {"status": "idle", "error": "no active stream"}
        self._stream_session.update_text(text)
        self._set_stage("stream_text", f"Updated stream text: {text}")
        return {"status": "ok", "text": text}

    def stop_stream_task(self) -> dict:
        if self._stream_session is None:
            return {"status": "idle"}
        session = self._stream_session
        thread = session.thread
        if thread is None or not thread.is_alive():
            return {"status": "stopped"}

        self._set_stage("stream_stopping", "Stopping stream")
        session.stop_event.set()
        thread.join(timeout=2.0)
        if thread.is_alive():
            return {"status": "stopping", "stream_id": session.stream_id}
        return {"status": "stopped", "stream_id": session.stream_id}

    def status(self) -> dict:
        with self._lock:
            processes = {}
            for name, managed in self._processes.items():
                processes[name] = "running" if managed.proc.poll() is None else "stopped"
            stream_running = False
            stream_text = ""
            stream_id = ""
            stream_error = ""
            if self._stream_session is not None:
                thread = self._stream_session.thread
                stream_running = bool(thread is not None and thread.is_alive())
                stream_text = self._stream_session.current_text()
                stream_id = self._stream_session.stream_id
                stream_error = self._stream_session.error
            return {
                "stage": self._stage,
                "job_id": self._latest_job_id,
                "processes": processes,
                "stream": {
                    "running": stream_running,
                    "stream_id": stream_id,
                    "text": stream_text,
                    "error": stream_error,
                    "preview": self._stream_preview.status(),
                    "metrics": {name: dict(values) for name, values in self._stream_metrics.items()},
                },
                "artifacts": dict(self._latest_artifacts),
            }

    def stream_preview_frame(self) -> dict:
        return self._stream_preview.pop_frame()

    def events(self, since: int = 0) -> list[dict]:
        with self._lock:
            return [event for event in self._events if int(event["id"]) >= int(since)]

    def stop_all(self) -> dict:
        self.stop_stream_task()
        for name, managed in list(self._processes.items())[::-1]:
            proc = managed.proc
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            managed.close_log()
            self._processes.pop(name, None)
        self._set_stage("idle", "Stopped all demo processes")
        return {"status": "ok"}

    def _launch_external_app_stop(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_SRC_DIR)
        log_path = self.log_dir / "app_stop.log"
        with log_path.open("a", encoding="utf-8") as log_file:
            subprocess.Popen(
                [
                    self.text2humanoid_python,
                    str(_APPS_DIR / "app_stop.py"),
                    "--force",
                    "--timeout-sec",
                    "2",
                ],
                cwd=_TEXT2HUMANOID_DIR,
                env=env,
                stdout=log_file,
                stderr=log_file,
                text=True,
            )

    def stop_app(self) -> dict:
        self.stop_all()

        def delayed_stop() -> None:
            time.sleep(0.2)
            self._launch_external_app_stop()

        threading.Thread(target=delayed_stop, daemon=True).start()
        self._set_stage("stopping", "Stopping Text2Humanoid demo app")
        return {"status": "stopping"}
