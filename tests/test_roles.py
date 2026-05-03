from __future__ import annotations

import pytest

from nano_rl.exceptions import SlotStateError
from nano_rl.runtime.coordinators import GenerationRequest
from nano_rl.runtime.roles import (
    RewardActorRole,
    RolloutReplicaControllerRole,
    RolloutWorkerRole,
    TrainerRankRole,
    bootstrap_weight_meta,
)
from nano_rl.runtime.sample_queue import SampleQueueActorCore
from nano_rl.runtime.slot import GpuLease, RoleName


def test_rollout_worker_generates_versioned_sample() -> None:
    meta = bootstrap_weight_meta(
        model_path="/models/qwen",
        tokenizer_path="/models/qwen",
    )
    worker = RolloutWorkerRole("worker-0", gpu_id=0)
    worker.activate_weight(
        meta,
        lease=GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=7),
    )
    sample = worker.generate(
        GenerationRequest(
            request_id="r",
            replica_id="rollout-dp-0",
            prompt="prompt",
            target_policy_version=0,
            metadata={"controller_step": 1},
        ),
        RewardActorRole(),
        lease=GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=7),
    )

    assert sample.policy_version == 0
    assert sample.weight_checksum == meta.checksum
    assert sample.meta["worker_id"] == "worker-0"
    assert sample.meta["lease_epoch"] == 7
    assert sample.reward > 0


def test_rollout_replica_controller_validates_all_tp_leases() -> None:
    meta = bootstrap_weight_meta(
        model_path="/models/qwen",
        tokenizer_path="/models/qwen",
    )
    replica = RolloutReplicaControllerRole(
        replica_id="rollout-dp-0",
        gpu_ids=(0, 1),
        worker_ids=("rollout-dp-0-tp-0", "rollout-dp-0-tp-1"),
    )
    replica.activate_weight(
        meta,
        leases=(
            GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="rollout-dp-0-tp-0", lease_epoch=1),
            GpuLease(gpu_id=1, role=RoleName.ROLLOUT, holder_id="rollout-dp-0-tp-1", lease_epoch=2),
        ),
    )

    sample = replica.generate(
        GenerationRequest(
            request_id="r",
            replica_id="rollout-dp-0",
            prompt="prompt",
            target_policy_version=0,
            metadata={},
        ),
        RewardActorRole(),
        leases=(
            GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="rollout-dp-0-tp-0", lease_epoch=1),
            GpuLease(gpu_id=1, role=RoleName.ROLLOUT, holder_id="rollout-dp-0-tp-1", lease_epoch=2),
        ),
    )

    assert replica.active_lease_epochs == {0: 1, 1: 2}
    assert sample.meta["lease_epochs"] == {"0": 1, "1": 2}


def test_rollout_worker_rejects_trainer_lease() -> None:
    worker = RolloutWorkerRole("worker-0", gpu_id=0)

    with pytest.raises(SlotStateError, match="not rollout"):
        worker.generate(
            GenerationRequest(
                request_id="r",
                replica_id="rollout-dp-0",
                prompt="prompt",
                target_policy_version=0,
                metadata={},
            ),
            RewardActorRole(),
            lease=GpuLease(gpu_id=0, role=RoleName.TRAINER, holder_id="worker-0", lease_epoch=0),
        )


def test_rollout_weight_activation_requires_rollout_lease() -> None:
    meta = bootstrap_weight_meta(
        model_path="/models/qwen",
        tokenizer_path="/models/qwen",
    )
    worker = RolloutWorkerRole("worker-0", gpu_id=0)
    replica = RolloutReplicaControllerRole(
        replica_id="rollout-dp-0",
        gpu_ids=(0,),
        worker_ids=("worker-0",),
    )

    with pytest.raises(SlotStateError, match="not rollout"):
        worker.activate_weight(
            meta,
            lease=GpuLease(gpu_id=0, role=RoleName.TRAINER, holder_id="worker-0", lease_epoch=0),
        )

    with pytest.raises(SlotStateError, match="not rollout"):
        replica.activate_weight(
            meta,
            leases=(GpuLease(gpu_id=0, role=RoleName.TRAINER, holder_id="worker-0", lease_epoch=0),),
        )


def test_trainer_rank_exports_next_weight_after_optimization() -> None:
    meta = bootstrap_weight_meta(
        model_path="/models/qwen",
        tokenizer_path="/models/qwen",
    )
    queue = SampleQueueActorCore(max_policy_lag=1, sample_ttl_sec=60, queue_high_watermark=10)
    # A minimal batch is easier to construct through the queue path.
    worker = RolloutWorkerRole("worker-0", gpu_id=0)
    worker.activate_weight(
        meta,
        lease=GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=0),
    )
    sample = worker.generate(
        GenerationRequest(
            request_id="r",
            replica_id="rollout-dp-0",
            prompt="prompt",
            target_policy_version=0,
            metadata={},
        ),
        RewardActorRole(),
        lease=GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=0),
    )
    queue.submit_sample(sample, current_policy_version=0)
    batch = queue.reserve_train_batch(reserved_by="trainer", current_policy_version=0, max_sequences=1)
    assert batch is not None

    rank = TrainerRankRole(rank=0, gpu_id=4)
    stats = rank.optimize(
        batch,
        lease=GpuLease(gpu_id=4, role=RoleName.TRAINER, holder_id="trainer-rank-0", lease_epoch=1),
    )
    new_weight = rank.export_weight(meta)

    assert stats.num_sequences == 1
    assert new_weight.version_id == 1
    assert new_weight.parent_version == 0
