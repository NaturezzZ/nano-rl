"""Run one deterministic local runtime iteration.

This script composes the pure-Python modules without starting Ray, vLLM, or
FSDP2.  It is a development smoke path for checking that config, GPU leases,
coordinators, queue, registry, and role adapters fit together.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nano_rl.config import load_launch_config
from nano_rl.runtime.controller import ControllerCore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local deterministic nano-rl smoke iteration")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prompt", action="append", default=[], help="prompt to include; can be repeated")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prompts = args.prompt or ["hello nano-rl", "test rollout"]
    config = load_launch_config(args.config)
    result = ControllerCore(config).run_smoke_iteration(prompts)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
