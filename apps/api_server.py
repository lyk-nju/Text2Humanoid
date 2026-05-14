from __future__ import annotations

import argparse

import yaml

from text2humanoid.api import create_app
from text2humanoid.infra.config_loader import build_components, resolve_config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Text2Humanoid API server")
    parser.add_argument("--config", type=str, default="configs/system/local_dev.yaml")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    session_manager, artifact_store = build_components(cfg)
    app = create_app(session_manager, artifact_store)

    import uvicorn
    uvicorn.run(app, host=cfg["host"], port=int(cfg["port"]))


if __name__ == "__main__":
    main()
