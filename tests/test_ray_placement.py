from __future__ import annotations

from pathlib import Path

import pytest

from nano_rl.config import load_launch_config
from nano_rl.runtime.ray.actors import (
    build_gpu_lease_manager_actor_class,
    build_metrics_actor_class,
    build_rollout_replica_controller_actor_class,
    build_rollout_manager_actor_class,
    build_rollout_worker_actor_class,
    build_sample_queue_actor_class,
    build_trainer_rank_actor_class,
    build_trainer_coordinator_actor_class,
    build_weight_registry_actor_class,
)
from nano_rl.runtime.ray.driver import RayDriver
from nano_rl.runtime.ray.placement import build_ray_launch_plan


ROOT = Path(__file__).resolve().parents[1]


def test_ray_launch_plan_uses_role_scoped_gpu_resources() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    plan = build_ray_launch_plan(config)

    assert plan.actors_requesting_ray_gpus == ()
    rollout_replica_resources = {
        actor.name: actor.resources
        for actor in plan.actors
        if actor.actor_type == "RolloutReplicaControllerActor"
    }
    trainer_rank_resources = {
        next(iter(actor.resources))
        for actor in plan.actors
        if actor.actor_type == "TrainerRankActor"
    }

    assert rollout_replica_resources == {
        "rollout-dp-0": {"rollout_gpu_0": 1, "rollout_gpu_1": 1},
        "rollout-dp-1": {"rollout_gpu_2": 1, "rollout_gpu_3": 1},
        "rollout-dp-2": {"rollout_gpu_4": 1, "rollout_gpu_5": 1},
        "rollout-dp-3": {"rollout_gpu_6": 1, "rollout_gpu_7": 1},
    }
    assert trainer_rank_resources == {f"train_gpu_{gpu_id}" for gpu_id in range(4, 8)}
    assert plan.node_custom_resources["rollout_gpu_4"] == 1
    assert plan.node_custom_resources["train_gpu_4"] == 1
    assert all(
        actor.resources == {}
        for actor in plan.actors
        if actor.actor_type == "RolloutWorkerActor"
    )
    rollout_manager = next(actor for actor in plan.actors if actor.actor_type == "RolloutManagerActor")
    assert len(rollout_manager.init_args[0]) == 4
    rollout_replica = next(actor for actor in plan.actors if actor.name == "rollout-dp-2")
    assert rollout_replica.num_gpus == 0
    assert rollout_replica.resources == {"rollout_gpu_4": 1, "rollout_gpu_5": 1}
    assert rollout_replica.init_args[:3] == (
        "rollout-dp-2",
        [4, 5],
        ["rollout-dp-2-tp-0", "rollout-dp-2-tp-1"],
    )
    rollout_backend_config = rollout_replica.init_args[3]
    assert rollout_backend_config["tensor_parallel_size"] == 2
    assert rollout_backend_config["gpu_ids"] == [4, 5]
    assert rollout_backend_config["holder_ids"] == ["rollout-dp-2-tp-0", "rollout-dp-2-tp-1"]
    trainer_rank = next(actor for actor in plan.actors if actor.name == "trainer-rank-0")
    assert trainer_rank.num_gpus == 0
    assert trainer_rank.resources == {"train_gpu_4": 1}
    assert trainer_rank.init_args[:3] == (0, 4, 0)
    trainer_backend_config = trainer_rank.init_args[3]
    assert trainer_backend_config["backend"] == "fsdp2"
    assert trainer_backend_config["rank"] == 0
    assert trainer_backend_config["world_size"] == 4
    assert trainer_backend_config["gpu_id"] == 4
    assert trainer_backend_config["holder_id"] == "trainer-rank-0"


def test_ray_driver_dry_run_includes_launch_plan() -> None:
    config = load_launch_config(ROOT / "recipes/collocated.yaml")
    dry_run = RayDriver(config).dry_run()
    actors = dry_run["ray_launch_plan"]["actors"]

    assert any(actor["actor_type"] == "GpuLeaseManagerActor" for actor in actors)
    assert any(actor["actor_type"] == "RolloutReplicaControllerActor" for actor in actors)
    assert all(actor["num_gpus"] == 0 for actor in actors)


def test_mock_launch_plan_uses_configured_actor_memory_hint() -> None:
    config = load_launch_config(ROOT / "recipes/mock_disaggregated.yaml")
    plan = build_ray_launch_plan(config)

    assert config.mock.ray_actor_memory_mb == 64
    assert {actor.memory_bytes for actor in plan.actors} == {64 * 1024 * 1024}


def test_ray_actor_builders_are_importable_when_ray_is_available() -> None:
    pytest.importorskip("ray")

    builders = [
        build_gpu_lease_manager_actor_class,
        build_rollout_manager_actor_class,
        build_rollout_replica_controller_actor_class,
        build_rollout_worker_actor_class,
        build_trainer_coordinator_actor_class,
        build_trainer_rank_actor_class,
        build_sample_queue_actor_class,
        build_weight_registry_actor_class,
        build_metrics_actor_class,
    ]

    assert all(builder() is not None for builder in builders)
