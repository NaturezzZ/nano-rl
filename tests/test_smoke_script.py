from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_script_runs_one_local_iteration() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_local_runtime.py",
            "--config",
            "docs/examples/disaggregated.yaml",
            "--prompt",
            "hello",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["new_weight"]["version_id"] == 1
    assert payload["queue"]["acked"] == 1
