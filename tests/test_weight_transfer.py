from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from nano_rl.config import ConfigError, load_launch_config
from nano_rl.runtime.protocols import WeightFormat, WeightMeta, WeightShardSourceKind, WeightTransferMethod
from nano_rl.runtime.weight_transfer import WeightTransferError, WeightTransferPlanner


ROOT = Path(__file__).resolve().parents[1]


def _weight(version_id: int = 3) -> WeightMeta:
    return WeightMeta(
        version_id=version_id,
        created_at=datetime.utcnow(),
        model_path="/weights/policy-v3",
        artifact_uri="/weights/policy-v3",
        manifest_uri="/weights/policy-v3/manifest.json",
        format=WeightFormat.VLLM_COMPATIBLE,
        checksum="checksum-v3",
    )


def test_locality_aware_transfer_reshards_shared_replicas_and_pulls_rollout_only_artifacts() -> None:
    config = load_launch_config(ROOT / "docs/examples/disaggregated.yaml")
    planner = WeightTransferPlanner(
        method=config.weight_transfer.method,
        allow_rollout_only_artifact_pull=config.weight_transfer.allow_rollout_only_artifact_pull,
    )

    plan = planner.build_plan(
        _weight(),
        rollout_replicas=config.gpu_plan.rollout_replicas,
        trainer_ranks=config.gpu_plan.trainer_ranks,
    )

    assert plan.method == WeightTransferMethod.LOCALITY_AWARE_CHECKPOINT
    assert plan.sources["rollout-dp-0"].kind == WeightShardSourceKind.ARTIFACT_PULL
    assert plan.sources["rollout-dp-1"].kind == WeightShardSourceKind.ARTIFACT_PULL
    assert plan.sources["rollout-dp-2"].kind == WeightShardSourceKind.SHARED_GPU_RESHARD
    assert plan.sources["rollout-dp-2"].source_rank_ids == (0, 1)
    assert plan.sources["rollout-dp-2"].source_gpu_ids == (4, 5)
    assert plan.sources["rollout-dp-3"].kind == WeightShardSourceKind.SHARED_GPU_RESHARD
    assert plan.sources["rollout-dp-3"].source_rank_ids == (2, 3)


def test_objectref_transfer_config_uses_ray_object_ref_sources_for_every_replica() -> None:
    config = load_launch_config(ROOT / "docs/examples/disaggregated.yaml")
    planner = WeightTransferPlanner(method=WeightTransferMethod.OBJECT_REF)

    plan = planner.build_plan(
        _weight(version_id=9),
        rollout_replicas=config.gpu_plan.rollout_replicas,
        trainer_ranks=config.gpu_plan.trainer_ranks,
    )

    assert plan.method == WeightTransferMethod.OBJECT_REF
    assert {source.kind for source in plan.sources.values()} == {WeightShardSourceKind.RAY_OBJECT_REF}
    assert plan.sources["rollout-dp-0"].object_ref_key == "weight:9:rollout-dp-0"


def test_locality_aware_transfer_requires_artifact_pull_for_rollout_only_gpus() -> None:
    config = load_launch_config(ROOT / "docs/examples/disaggregated.yaml")
    planner = WeightTransferPlanner(
        method=WeightTransferMethod.LOCALITY_AWARE_CHECKPOINT,
        allow_rollout_only_artifact_pull=False,
    )

    with pytest.raises(WeightTransferError, match="rollout-only GPUs"):
        planner.build_plan(
            _weight(),
            rollout_replicas=config.gpu_plan.rollout_replicas,
            trainer_ranks=config.gpu_plan.trainer_ranks,
        )


def test_config_rejects_locality_aware_transfer_without_rollout_only_artifact_pull(tmp_path: Path) -> None:
    raw = (ROOT / "docs/examples/disaggregated.yaml").read_text()
    raw = raw.replace("allow_rollout_only_artifact_pull: true", "allow_rollout_only_artifact_pull: false")
    config_path = tmp_path / "invalid-weight-transfer.yaml"
    config_path.write_text(raw)

    with pytest.raises(ConfigError, match="allow_rollout_only_artifact_pull=true"):
        load_launch_config(config_path)
