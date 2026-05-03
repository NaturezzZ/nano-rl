from __future__ import annotations

from pathlib import Path

import pytest

from nano_rl.config import load_launch_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def disaggregated_gpu_plan():
    return load_launch_config(ROOT / "docs/examples/disaggregated.yaml").gpu_plan
