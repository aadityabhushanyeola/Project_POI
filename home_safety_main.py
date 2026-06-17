from __future__ import annotations

import argparse

from src.home_safety.runner import run_home_safety


def main() -> None:
    parser = argparse.ArgumentParser(description="CCTV home safety hazard analysis")
    parser.add_argument("--config", default="config/home_safety.yaml")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Analyze video and stop at end")
    args = parser.parse_args()
    run_home_safety(args.config, no_display=args.no_display, offline=args.offline)


if __name__ == "__main__":
    main()
