"""User entrypoint for nano-rl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nano_rl.config import RunIntent, launch_config_to_dict, load_launch_config
from nano_rl.exceptions import NanoRLError
from nano_rl.runtime.ray.driver import RayDriver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nano-rl runtime entrypoint")
    parser.add_argument("--config", required=True, type=Path, help="YAML runtime config")
    parser.add_argument(
        "--emit-resolved-config",
        action="store_true",
        help="print resolved LaunchConfig JSON and exit",
    )
    parser.add_argument(
        "--skip-artifact-validation",
        action="store_true",
        help="skip configured dataset artifact probes for local dry-run development",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_launch_config(args.config)
        if args.emit_resolved_config or config.run.emit_resolved_config:
            print(json.dumps(launch_config_to_dict(config), indent=2, sort_keys=True))
            return 0

        driver = RayDriver(config)
        if config.run.intent == RunIntent.VALIDATE:
            result = driver.validate(validate_artifacts=not args.skip_artifact_validation)
        elif config.run.intent == RunIntent.DRY_RUN:
            result = driver.dry_run(validate_artifacts=not args.skip_artifact_validation)
        else:
            result = driver.train(validate_artifacts=not args.skip_artifact_validation)

        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except NanoRLError as exc:
        print(f"nano-rl: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
