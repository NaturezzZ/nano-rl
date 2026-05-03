from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from nano_rl.config import CanonicalMode, ConfigError, load_launch_config


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text())


def test_collocated_example_resolves_to_shared_gpu_plan() -> None:
    config = load_launch_config(ROOT / "docs/examples/collocated.yaml")

    assert config.canonical_mode == CanonicalMode.FULLY_SYNC
    assert config.model.model_path == "/mnt/hdfs/nano-ai/models/qwen"
    assert config.model.tokenizer_path == "/mnt/hdfs/nano-ai/models/qwen"
    assert config.gpu_plan.physical_gpu_ids == tuple(range(8))
    assert config.gpu_plan.rollout_only_gpu_ids == ()
    assert config.gpu_plan.shared_gpu_ids == tuple(range(8))
    assert config.gpu_plan.rollout_replica_count == 4
    assert config.gpu_plan.rollout_worker_count == 8
    assert config.gpu_plan.trainer_rank_count == 8


def test_disaggregated_example_resolves_rollout_only_and_shared_gpu_plan() -> None:
    config = load_launch_config(ROOT / "docs/examples/disaggregated.yaml")

    assert config.canonical_mode == CanonicalMode.STANDALONE_HYBRID
    assert config.rollout.partial_rollout.enabled is True
    assert config.rollout.partial_rollout.mixed_policy_samples == "train"
    assert config.gpu_plan.rollout_only_gpu_ids == (0, 1, 2, 3)
    assert config.gpu_plan.shared_gpu_ids == (4, 5, 6, 7)
    assert [replica.gpu_ids for replica in config.gpu_plan.rollout_replicas] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]
    assert [rank.gpu_id for rank in config.gpu_plan.trainer_ranks] == [4, 5, 6, 7]


def test_model_path_does_not_require_storage_source_type() -> None:
    raw = _load_yaml("docs/examples/collocated.yaml")
    raw = copy.deepcopy(raw)
    raw["model"] = {"model_path": "/opt/nano-models/qwen"}
    temp = ROOT / ".tmp-local-model-path.yaml"
    temp.write_text(yaml.safe_dump(raw))
    try:
        config = load_launch_config(temp)
    finally:
        temp.unlink(missing_ok=True)

    assert config.model.model_path == "/opt/nano-models/qwen"
    assert config.model.tokenizer_path == "/opt/nano-models/qwen"


def test_shared_gpus_require_trainer_rank_class() -> None:
    raw = _load_yaml("docs/examples/disaggregated.yaml")
    raw["runtime"]["ray"]["gpu_manager"]["role_classes"].pop("trainer_rank")
    temp = ROOT / ".tmp-invalid-config.yaml"
    temp.write_text(yaml.safe_dump(raw))
    try:
        with pytest.raises(ConfigError, match="shared GPUs require"):
            load_launch_config(temp)
    finally:
        temp.unlink(missing_ok=True)


def test_internal_modes_are_not_user_yaml_modes() -> None:
    raw = _load_yaml("docs/examples/collocated.yaml")
    raw["mode"] = "fully_sync"
    temp = ROOT / ".tmp-invalid-mode.yaml"
    temp.write_text(yaml.safe_dump(raw))
    try:
        with pytest.raises(ConfigError):
            load_launch_config(temp)
    finally:
        temp.unlink(missing_ok=True)


def test_topology_counts_must_match_placement() -> None:
    raw = _load_yaml("docs/examples/disaggregated.yaml")
    raw = copy.deepcopy(raw)
    raw["runtime"]["ray"]["placement"]["trainer"]["num_ranks"] = 8
    temp = ROOT / ".tmp-invalid-placement.yaml"
    temp.write_text(yaml.safe_dump(raw))
    try:
        with pytest.raises(ConfigError, match="trainer.num_ranks"):
            load_launch_config(temp)
    finally:
        temp.unlink(missing_ok=True)


def test_role_class_paths_must_import() -> None:
    raw = _load_yaml("docs/examples/collocated.yaml")
    raw["runtime"]["ray"]["gpu_manager"]["role_classes"]["rollout_worker"] = "nano_rl.runtime.roles.MissingRole"
    temp = ROOT / ".tmp-invalid-role-class.yaml"
    temp.write_text(yaml.safe_dump(raw))
    try:
        with pytest.raises(ConfigError, match="class does not exist"):
            load_launch_config(temp)
    finally:
        temp.unlink(missing_ok=True)


def test_gpu_manager_must_be_enabled() -> None:
    raw = _load_yaml("docs/examples/collocated.yaml")
    raw = copy.deepcopy(raw)
    raw["runtime"]["ray"]["gpu_manager"]["enabled"] = False
    temp = ROOT / ".tmp-disabled-gpu-manager.yaml"
    temp.write_text(yaml.safe_dump(raw))
    try:
        with pytest.raises(ConfigError, match="Input should be True"):
            load_launch_config(temp)
    finally:
        temp.unlink(missing_ok=True)


def test_rollout_lifecycle_regions_must_align_with_tensor_parallel_groups() -> None:
    raw = _load_yaml("docs/examples/disaggregated.yaml")
    raw = copy.deepcopy(raw)
    raw["runtime"]["ray"]["gpu_manager"]["topology"]["rollout_only_gpus"] = 3
    raw["runtime"]["ray"]["gpu_manager"]["topology"]["shared_gpus"] = 5
    raw["runtime"]["ray"]["placement"]["trainer"]["num_ranks"] = 5
    raw["parallel"]["trainer"]["fsdp_world_size"] = 5
    temp = ROOT / ".tmp-invalid-tp-region.yaml"
    temp.write_text(yaml.safe_dump(raw))
    try:
        with pytest.raises(ConfigError, match="rollout_only_gpus must be divisible"):
            load_launch_config(temp)
    finally:
        temp.unlink(missing_ok=True)
