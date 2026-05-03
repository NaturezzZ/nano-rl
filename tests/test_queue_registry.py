from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nano_rl.runtime.protocols import WeightFormat, WeightMeta
from nano_rl.runtime.protocols import PolicySegment, SampleRecord, WeightStatus
from nano_rl.runtime.sample_queue import QueueDecision, SampleQueueActorCore
from nano_rl.runtime.weight_registry import WeightRegistryActorCore


def _sample(sample_id: str, policy_version: int, now: datetime) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        policy_version=policy_version,
        prompt="p",
        response="r",
        tokens=[1, 2],
        logprobs=[-0.1, -0.2],
        reward=1.0,
        created_at=now,
        expires_at=now + timedelta(seconds=60),
    )


def test_sample_queue_accepts_reserves_acks_and_releases() -> None:
    now = datetime.utcnow()
    queue = SampleQueueActorCore(max_policy_lag=2, sample_ttl_sec=60, queue_high_watermark=10)

    assert queue.submit_sample(_sample("s1", 3, now), current_policy_version=4).accepted
    assert queue.submit_sample(_sample("s2", 4, now), current_policy_version=4).accepted

    batch = queue.reserve_train_batch(
        reserved_by="trainer",
        current_policy_version=4,
        max_sequences=2,
        now=now,
    )
    assert batch is not None
    assert batch.num_sequences == 2
    assert batch.policy_version_histogram == {3: 1, 4: 1}

    queue.release_batch(batch.train_batch_id)
    assert queue.stats()["pending"] == 2

    batch = queue.reserve_train_batch(reserved_by="trainer", current_policy_version=4, max_sequences=1, now=now)
    assert batch is not None
    queue.ack_batch(batch.train_batch_id)
    assert queue.stats()["acked"] == 1


def test_sample_queue_drops_stale_samples() -> None:
    now = datetime.utcnow()
    queue = SampleQueueActorCore(max_policy_lag=1, sample_ttl_sec=60, queue_high_watermark=10)
    result = queue.submit_sample(_sample("old", 1, now), current_policy_version=3, now=now)

    assert result.decision == QueueDecision.DROPPED_POLICY_LAG
    assert queue.stats()["dropped"] == 1


def test_sample_queue_uses_oldest_partial_rollout_policy_segment_for_lag() -> None:
    now = datetime.utcnow()
    queue = SampleQueueActorCore(max_policy_lag=1, sample_ttl_sec=60, queue_high_watermark=10)
    record = _sample("partial", 3, now)
    record.policy_segments = [
        PolicySegment(start_token=0, end_token=1, policy_version=1),
        PolicySegment(start_token=1, end_token=2, policy_version=3),
    ]

    result = queue.submit_sample(record, current_policy_version=3, now=now)

    assert result.decision == QueueDecision.DROPPED_POLICY_LAG


def test_weight_registry_activation_requires_all_workers() -> None:
    registry = WeightRegistryActorCore()
    meta = WeightMeta(
        version_id=1,
        created_at=datetime.utcnow(),
        model_path="/models/qwen/weights/v1",
        format=WeightFormat.SAFETENSORS,
        checksum="abc",
    )

    registered = registry.register(meta)
    assert registered.status == WeightStatus.REGISTERED

    registry.begin_activation(1, {"w0", "w1"})
    assert registry.ack_activation(1, "w0").status == WeightStatus.ACTIVATING
    assert registry.ack_activation(1, "w1").status == WeightStatus.ACTIVE_GLOBAL
    assert registry.latest_active_global().version_id == 1

    with pytest.raises(Exception):
        registry.ack_activation(1, "w2")
