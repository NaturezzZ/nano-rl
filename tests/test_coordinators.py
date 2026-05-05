from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from nano_rl.config import load_launch_config
from nano_rl.runtime.coordinators import CoordinatorError, RolloutManagerCore, TrainerCoordinatorCore
from nano_rl.runtime.protocols import SampleRecord
from nano_rl.runtime.sample_queue import SampleQueueActorCore


ROOT = Path(__file__).resolve().parents[1]


def test_rollout_manager_dispatches_round_robin_and_can_pause() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    manager = RolloutManagerCore(config.gpu_plan.rollout_replicas)

    requests = manager.dispatch_prompts(["a", "b", "c"], target_policy_version=2, controller_step=10)
    assert [request.replica_id for request in requests] == [
        "rollout-dp-0",
        "rollout-dp-1",
        "rollout-dp-2",
    ]

    manager.pause_for_weight("activate-v3")
    with pytest.raises(CoordinatorError, match="paused"):
        manager.dispatch_prompts(["d"], target_policy_version=3, controller_step=11)


def test_rollout_manager_pumps_backlog_until_capacity_or_backpressure() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    manager = RolloutManagerCore(config.gpu_plan.rollout_replicas, max_in_flight_per_replica=1)
    manager.enqueue_prompts(["a", "b", "c", "d", "e"])

    requests = manager.dispatch_from_backlog(
        target_policy_version=2,
        controller_step=10,
        output_queue_depth=0,
        output_queue_high_watermark=10,
    )
    assert [request.replica_id for request in requests] == [
        "rollout-dp-0",
        "rollout-dp-1",
        "rollout-dp-2",
        "rollout-dp-3",
    ]
    assert manager.stats()["input_backlog"] == 1
    assert manager.stats()["in_flight"] == 4

    blocked = manager.dispatch_from_backlog(
        target_policy_version=2,
        controller_step=11,
        output_queue_depth=9,
        output_queue_high_watermark=9,
    )
    assert blocked == []

    manager.complete_request(requests[0].request_id)
    resumed = manager.dispatch_from_backlog(
        target_policy_version=2,
        controller_step=12,
        output_queue_depth=0,
        output_queue_high_watermark=10,
    )
    assert len(resumed) == 1
    assert resumed[0].prompt == "e"


def test_rollout_manager_preserves_prompt_metadata() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    manager = RolloutManagerCore(config.gpu_plan.rollout_replicas, max_in_flight_per_replica=1)
    manager.enqueue_prompts(
        [
            {
                "prompt_id": "p0",
                "prompt": "hello",
                "metadata": {"prompt_tokens": 9, "source": "mock"},
            }
        ]
    )

    requests = manager.dispatch_from_backlog(
        target_policy_version=2,
        controller_step=10,
        output_queue_depth=0,
        output_queue_high_watermark=10,
    )

    assert len(requests) == 1
    assert requests[0].prompt == "hello"
    assert requests[0].metadata["prompt_id"] == "p0"
    assert requests[0].metadata["prompt_tokens"] == 9
    assert requests[0].metadata["source"] == "mock"
    assert requests[0].metadata["controller_step"] == 10


def test_trainer_coordinator_assigns_shared_gpus_and_reserves_batch() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    coordinator = TrainerCoordinatorCore(config.gpu_plan.trainer_ranks)
    assert coordinator.world_size == 4
    assert [assignment.gpu_id for assignment in coordinator.rank_assignments] == [4, 5, 6, 7]

    queue = SampleQueueActorCore(max_policy_lag=2, sample_ttl_sec=60, queue_high_watermark=10)
    now = datetime.utcnow()
    queue.submit_sample(
        SampleRecord(
            sample_id="s",
            policy_version=1,
            prompt="p",
            response="r",
            tokens=[1],
            logprobs=[0.0],
            reward=1.0,
            created_at=now,
        ),
        current_policy_version=1,
        now=now,
    )

    batch = coordinator.reserve_batch(queue, current_policy_version=1, max_sequences=1)
    assert batch is not None
    assert batch.reserved_by == "trainer-group-0"
