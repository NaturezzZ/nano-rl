"""User entrypoint for nano-rl."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from nano_rl.config import LaunchConfig, RunIntent, launch_config_to_dict, load_launch_config
from nano_rl.exceptions import NanoRLError
from nano_rl.runtime.ray.driver import RayDriver


RESOLVED_CONFIG_DIR = Path(__file__).resolve().parent / ".nano_rl"
RESOLVED_CONFIG_PATH = RESOLVED_CONFIG_DIR / "resolved-config.json"
RUN_RESULT_PATH = RESOLVED_CONFIG_DIR / "run-result.json"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nano-rl runtime entrypoint")
    parser.add_argument("--config", required=True, type=Path, help="YAML runtime config")
    parser.add_argument(
        "--emit-resolved-config",
        action="store_true",
        help=f"write resolved LaunchConfig JSON to {RESOLVED_CONFIG_PATH} and exit",
    )
    parser.add_argument(
        "--skip-artifact-validation",
        action="store_true",
        help="skip configured dataset artifact probes for local dry-run development",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_launch_config(args.config)
        write_resolved_config(config)
        logging.getLogger(__name__).info(
            "loaded LaunchConfig: config=%s intent=%s start_ray_actors=%s canonical_mode=%s",
            args.config,
            config.run.intent.value,
            config.run.start_ray_actors,
            config.canonical_mode,
        )
        if args.emit_resolved_config or config.run.emit_resolved_config:
            return 0

        driver = RayDriver(config)
        if config.run.intent == RunIntent.VALIDATE:
            result = driver.validate(validate_artifacts=not args.skip_artifact_validation)
        elif config.run.intent == RunIntent.DRY_RUN:
            result = driver.dry_run(validate_artifacts=not args.skip_artifact_validation)
        else:
            result = driver.train(validate_artifacts=not args.skip_artifact_validation)

        write_run_result(result)
        return 0
    except NanoRLError as exc:
        print(f"nano-rl: {exc}", file=sys.stderr)
        return 2


def write_resolved_config(config: LaunchConfig) -> Path:
    RESOLVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(launch_config_to_dict(config), indent=2, sort_keys=True)
    RESOLVED_CONFIG_PATH.write_text(payload + "\n", encoding="utf-8")
    logging.getLogger(__name__).info("wrote resolved LaunchConfig: path=%s", RESOLVED_CONFIG_PATH)
    return RESOLVED_CONFIG_PATH


def write_run_result(result: dict[str, object]) -> Path:
    RESOLVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True, default=str)
    RUN_RESULT_PATH.write_text(payload + "\n", encoding="utf-8")
    logging.getLogger(__name__).info("wrote run result: path=%s", RUN_RESULT_PATH)
    return RUN_RESULT_PATH


def configure_logging() -> None:
    level_name = os.environ.get("NANO_RL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT)


if __name__ == "__main__":
    raise SystemExit(main())
