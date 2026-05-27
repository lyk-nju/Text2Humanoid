from __future__ import annotations

import argparse

from text2humanoid.demo.console_controller import DemoController
from text2humanoid.demo.console_server import create_demo_console_server


DEFAULT_PYTHON = "/home/lai/anaconda3/envs/flooddiffusion/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser(description="Text2Humanoid local demo console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--output-dir", default="assets/saved")
    parser.add_argument("--video-dir", default="assets/video")
    parser.add_argument("--log-dir", default="artifacts/logs/demo_console")
    parser.add_argument("--text2humanoid-python", default=DEFAULT_PYTHON)
    parser.add_argument("--flooddiffusion-python", default=DEFAULT_PYTHON)
    args = parser.parse_args()

    controller = DemoController(
        output_dir=args.output_dir,
        video_dir=args.video_dir,
        log_dir=args.log_dir,
        text2humanoid_python=args.text2humanoid_python,
        flooddiffusion_python=args.flooddiffusion_python,
    )
    server = create_demo_console_server(controller, host=args.host, port=args.port)
    print(f"Text2Humanoid Demo Console: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        controller.stop_all()
        server.server_close()


if __name__ == "__main__":
    main()
