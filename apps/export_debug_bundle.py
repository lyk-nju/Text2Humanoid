from __future__ import annotations

import argparse
from pathlib import Path

from text2humanoid.infra.artifact_store import ArtifactStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a status artifact bundle.")
    parser.add_argument("--root", default="./artifacts")
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()

    store = ArtifactStore(Path(args.root))
    print(store.session_dir(args.session_id))


if __name__ == "__main__":
    main()
