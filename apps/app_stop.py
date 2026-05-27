from __future__ import annotations

import argparse
from collections.abc import Sequence

from text2humanoid.demo.process_killer import stop_demo_processes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop Text2Humanoid demo processes.")
    parser.add_argument("--dry-run", action="store_true", help="Print matching processes without signaling them")
    parser.add_argument("--force", action="store_true", help="Send SIGKILL after SIGTERM if a process stays alive")
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    parser.add_argument(
        "--ps-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = stop_demo_processes(
        ps_output=args.ps_output,
        dry_run=args.dry_run,
        force=args.force,
        timeout_sec=args.timeout_sec,
    )
    action = "matched" if args.dry_run else "stopped"
    print(f"{action}={result.matched} signaled={result.signaled}")
    for process in result.processes:
        print(f"{process.pid} {process.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
