from __future__ import annotations

import argparse
from builtins import input as input
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys
import time


_TEXT2HUMANOID_DIR = Path(__file__).resolve().parents[1]
_TEXT2MOTION_ROOT = _TEXT2HUMANOID_DIR.parent
_BFM_ZERO_DIR = _TEXT2MOTION_ROOT / "BFM-Zero"
_DEFAULT_PYTHON = "/home/lai/anaconda3/envs/flooddiffusion/bin/python"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_bfmzero_chunk import main as replay_bfmzero_chunk_main
from run_text_to_bfmzero import main as run_text_to_bfmzero_main


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the FloodDiffusion -> retarget -> BFM-Zero -> MuJoCo demo."
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--out-prefix", default="assets/saved/full_demo")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--generation-steps", type=int, default=150)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--video-dir", default="assets/video")
    parser.add_argument("--render-stride", type=int, default=1)

    parser.add_argument("--skip-sim", action="store_true", help="Reuse an already running MuJoCo simulation")
    parser.add_argument("--skip-policy", action="store_true", help="Reuse an already running BFM-Zero policy")
    parser.add_argument("--auto-publish", action="store_true", help="Replay without waiting for manual BFM-Zero key setup")
    parser.add_argument("--manual-policy-keys", action="store_true", help="Ask the user to press i, ], [ manually")
    parser.add_argument("--policy-warmup-sec", type=float, default=5.0)
    parser.add_argument("--init-wait-sec", type=float, default=2.0)
    parser.add_argument("--policy-action-wait-sec", type=float, default=0.2)
    parser.add_argument("--log-dir", default="artifacts/logs/full_demo")

    parser.add_argument("--bfmzero-root", default=str(_BFM_ZERO_DIR))
    parser.add_argument("--bfmzero-python", default=_DEFAULT_PYTHON)
    parser.add_argument("--flooddiffusion-python", default=_DEFAULT_PYTHON)
    parser.add_argument("--flooddiffusion-config", default="configs/stream.yaml")
    parser.add_argument("--system-config", default="configs/system/local_dev.yaml")
    parser.add_argument("--retarget-fps", type=float, default=None)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=5592)
    parser.add_argument("--mark-end", action="store_true", default=True)
    parser.add_argument("--no-mark-end", action="store_false", dest="mark_end")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--startup-delay-sec", type=float, default=0.5)
    return parser.parse_args(argv)


def _resolve_text2humanoid_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (_TEXT2HUMANOID_DIR / path).resolve()


def _spawn_process(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    log_dir: Path,
) -> _ManagedProcess:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=log_file,
        stderr=log_file,
        text=True,
    )
    print(f"Started {name}: pid={getattr(proc, 'pid', 'unknown')} log={log_path}")
    return _ManagedProcess(name=name, proc=proc, log_file=log_file)


def _append_policy_key(command_file: Path, key: str) -> None:
    command_file.parent.mkdir(parents=True, exist_ok=True)
    with command_file.open("a", encoding="utf-8") as f:
        f.write(f"{key}\n")
        f.flush()


def _send_policy_key_sequence(args: argparse.Namespace, command_file: Path) -> None:
    print("Sending BFM-Zero policy keys: i -> ] -> [")
    _append_policy_key(command_file, "i")
    if args.init_wait_sec > 0:
        time.sleep(args.init_wait_sec)
    _append_policy_key(command_file, "]")
    if args.policy_action_wait_sec > 0:
        time.sleep(args.policy_action_wait_sec)
    _append_policy_key(command_file, "[")


def _stop_processes(processes: list[_ManagedProcess]) -> None:
    for managed in reversed(processes):
        proc = managed.proc
        if proc.poll() is not None:
            managed.close_log()
            continue
        print(f"Stopping {managed.name}...")
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
        finally:
            managed.close_log()


def _build_text_to_bfmzero_args(args: argparse.Namespace, prefix: Path) -> list[str]:
    cmd = [
        "--text",
        args.text,
        "--out-prefix",
        str(prefix),
        "--generation-steps",
        str(args.generation_steps),
        "--python",
        args.flooddiffusion_python,
        "--flooddiffusion-config",
        args.flooddiffusion_config,
        "--system-config",
        args.system_config,
        "--output-fps",
        str(args.output_fps),
        "--device",
        args.device,
        "--dry-run",
    ]
    if args.frames is not None:
        cmd.extend(["--frames", str(args.frames)])
    if args.retarget_fps is not None:
        cmd.extend(["--retarget-fps", str(args.retarget_fps)])
    if args.inspect:
        cmd.append("--inspect")
        cmd.extend(["--video-dir", args.video_dir])
        cmd.extend(["--render-stride", str(args.render_stride)])
    return cmd


def _build_replay_args(args: argparse.Namespace, bfmzero_path: Path) -> list[str]:
    cmd = [
        "--input",
        str(bfmzero_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--startup-delay-sec",
        str(args.startup_delay_sec),
    ]
    if args.mark_end:
        cmd.append("--mark-end")
    if args.no_realtime:
        cmd.append("--no-realtime")
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prefix = _resolve_text2humanoid_path(args.out_prefix)
    bfmzero_path = prefix.with_name(f"{prefix.name}_bfmzero.npz")
    log_dir = _resolve_text2humanoid_path(args.log_dir)
    bfmzero_root = Path(args.bfmzero_root).expanduser().resolve()
    keyboard_command_file = log_dir / "bfmzero_policy.keys"
    if keyboard_command_file.exists():
        keyboard_command_file.unlink()
    processes: list[_ManagedProcess] = []

    try:
        if not args.skip_sim:
            processes.append(
                _spawn_process(
                    name="mujoco_sim",
                    cmd=[
                        args.bfmzero_python,
                        "-m",
                        "sim_env.base_sim",
                        "--robot_config",
                        "./config/robot/g1.yaml",
                        "--scene_config",
                        "./config/scene/g1_29dof.yaml",
                    ],
                    cwd=bfmzero_root,
                    log_dir=log_dir,
                )
            )
        if not args.skip_policy:
            processes.append(
                _spawn_process(
                    name="bfmzero_policy",
                    cmd=[
                        args.bfmzero_python,
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
                        str(keyboard_command_file),
                    ],
                    cwd=bfmzero_root,
                    log_dir=log_dir,
                )
            )
            if args.policy_warmup_sec > 0:
                time.sleep(args.policy_warmup_sec)

        rc = run_text_to_bfmzero_main(_build_text_to_bfmzero_args(args, prefix))
        if rc != 0:
            return rc

        if not args.skip_policy and not args.manual_policy_keys:
            _send_policy_key_sequence(args, keyboard_command_file)
        elif not args.auto_publish:
            input("BFM-Zero policy ready? Press i, then ], then [, then Enter here to replay... ")

        rc = replay_bfmzero_chunk_main(_build_replay_args(args, bfmzero_path))
        if rc != 0:
            return rc
        print(f"Full demo replayed: {bfmzero_path}")
        print(f"Logs: {log_dir}")
        return 0
    finally:
        _stop_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
