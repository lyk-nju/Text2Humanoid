from __future__ import annotations

import argparse
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved reference clip.")
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    data = np.load(args.path, allow_pickle=True)
    for key in sorted(data.files):
        print(f"{key}: {data[key].shape}")


if __name__ == "__main__":
    main()
