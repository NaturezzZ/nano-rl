from __future__ import annotations

from pathlib import Path
from typing import Any

from nano_rl.config import load_launch_config
from nano_rl.runtime.ray.driver import RayDriver
from nano_rl.runtime.ray.launcher import RayActorGraphLauncher, RemoteActorClassBuilder
from nano_rl.runtime.ray.placement import build_ray_launch_plan


ROOT = Path(__file__).resolve().parents[1]


class FakeRemoteActorClass:
    def __init__(self, actor_type: str, calls: list[dict[str, Any]]) -> None:
        self.actor_type = actor_type
        self.calls = calls
        self.options_payload: dict[str, Any] | None = None

    def options(self, **kwargs: Any) -> "FakeRemoteActorClass":
        self.options_payload = kwargs
        return self

    def remote(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        assert self.options_payload is not None
        handle = {
            "actor_type": self.actor_type,
            "options": self.options_payload,
            "args": args,
            "kwargs": kwargs,
        }
        self.calls.append(handle)
        return handle


def _fake_builders(
    actor_types: set[str],
    calls: list[dict[str, Any]],
) -> dict[str, RemoteActorClassBuilder]:
    return {
        actor_type: (lambda actor_type=actor_type: FakeRemoteActorClass(actor_type, calls))
        for actor_type in actor_types
    }


def test_actor_graph_launcher_uses_zero_ray_gpus_and_role_resources() -> None:
    config = load_launch_config(ROOT / "docs/examples/disaggregated.yaml")
    plan = build_ray_launch_plan(config)
    calls: list[dict[str, Any]] = []

    graph = RayActorGraphLauncher(
        plan,
        namespace=config.runtime.ray.namespace,
        actor_class_builders=_fake_builders({spec.actor_type for spec in plan.actors}, calls),
    ).start(dry_run=False)

    assert set(graph.handles) == {spec.name for spec in plan.actors}
    assert len(calls) == len(plan.actors)
    assert all(call["options"]["num_gpus"] == 0 for call in calls)
    assert all(call["options"]["namespace"] == config.runtime.ray.namespace for call in calls)

    rollout_dp_2 = next(
        call for call in calls if call["options"]["resources"] == {"rollout_gpu_4": 1, "rollout_gpu_5": 1}
    )
    train_gpu_4 = next(
        call for call in calls if call["options"]["resources"] == {"train_gpu_4": 1}
    )
    assert rollout_dp_2["actor_type"] == "RolloutReplicaControllerActor"
    assert train_gpu_4["actor_type"] == "TrainerRankActor"
    rollout_worker = next(call for call in calls if call["options"]["name"] == "rollout-dp-2-tp-0")
    assert rollout_worker["options"]["resources"] == {}
    assert graph.resource_summary["shared_gpu_ids"] == [4, 5, 6, 7]


def test_actor_graph_launcher_passes_init_args_from_launch_plan() -> None:
    config = load_launch_config(ROOT / "docs/examples/disaggregated.yaml")
    plan = build_ray_launch_plan(config)
    calls: list[dict[str, Any]] = []

    RayActorGraphLauncher(
        plan,
        namespace=config.runtime.ray.namespace,
        actor_class_builders=_fake_builders({spec.actor_type for spec in plan.actors}, calls),
    ).start(dry_run=False)

    gpu_manager = next(call for call in calls if call["options"]["name"] == "gpu-lease-manager")
    rollout_manager = next(call for call in calls if call["options"]["name"] == "rollout-manager")
    trainer_rank_0 = next(call for call in calls if call["options"]["name"] == "trainer-rank-0")
    rollout_worker = next(call for call in calls if call["options"]["name"] == "rollout-dp-2-tp-0")

    assert gpu_manager["args"] == next(
        spec.init_args for spec in plan.actors if spec.name == "gpu-lease-manager"
    )
    assert len(rollout_manager["args"][0]) == config.gpu_plan.rollout_replica_count
    assert trainer_rank_0["args"][:3] == (0, 4, 0)
    assert trainer_rank_0["args"][3]["backend"] == "fsdp2"
    assert trainer_rank_0["args"][3]["world_size"] == config.gpu_plan.trainer_rank_count
    assert trainer_rank_0["args"][3]["holder_id"] == "trainer-rank-0"
    assert rollout_worker["args"] == ("rollout-dp-2-tp-0", 4, 2, 0)
    rollout_replica = next(call for call in calls if call["options"]["name"] == "rollout-dp-2")
    assert rollout_replica["args"][:3] == (
        "rollout-dp-2",
        [4, 5],
        ["rollout-dp-2-tp-0", "rollout-dp-2-tp-1"],
    )
    assert rollout_replica["args"][3]["tensor_parallel_size"] == 2
    assert rollout_replica["args"][3]["gpu_ids"] == [4, 5]
    assert rollout_replica["args"][3]["holder_ids"] == [
        "rollout-dp-2-tp-0",
        "rollout-dp-2-tp-1",
    ]


def test_actor_graph_dry_run_does_not_create_remote_actors() -> None:
    config = load_launch_config(ROOT / "docs/examples/collocated.yaml")
    plan = build_ray_launch_plan(config)
    calls: list[dict[str, Any]] = []

    graph = RayActorGraphLauncher(
        plan,
        namespace=config.runtime.ray.namespace,
        actor_class_builders=_fake_builders({spec.actor_type for spec in plan.actors}, calls),
    ).start(dry_run=True)

    assert graph.dry_run is True
    assert graph.handles == {}
    assert calls == []
    assert graph.resource_summary["actors_requesting_ray_gpus"] == []


def test_ray_driver_train_still_returns_plan_without_starting_actor_graph(monkeypatch) -> None:
    config = load_launch_config(ROOT / "docs/examples/collocated.yaml")

    def fail_if_actor_graph_starts(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("train() must not start the Ray actor graph by default")

    monkeypatch.setattr(RayActorGraphLauncher, "start", fail_if_actor_graph_starts)

    result = RayDriver(config).train(validate_artifacts=False)

    assert "ray_launch_plan" in result
    assert result["execution_status"] == "planned_backend_integrated"
    assert result["start_ray_actors"] is False
    assert all(
        actor["num_gpus"] == 0
        for actor in result["ray_launch_plan"]["actors"]
    )


def test_ray_driver_train_can_start_actor_graph_when_config_requests_it() -> None:
    base_config = load_launch_config(ROOT / "docs/examples/collocated.yaml")
    config = base_config.model_copy(
        update={"run": base_config.run.model_copy(update={"start_ray_actors": True})}
    )
    plan = build_ray_launch_plan(config)
    calls: list[dict[str, Any]] = []

    result = RayDriver(config).train(
        validate_artifacts=False,
        actor_class_builders=_fake_builders({spec.actor_type for spec in plan.actors}, calls),
        start_ray=False,
    )

    assert result["execution_status"] == "ray_actor_graph_started"
    assert result["start_ray_actors"] is True
    assert result["ray_actor_graph"]["dry_run"] is False
    assert "controller" in result["ray_actor_graph"]["handles"]
    assert "trainer-rank-0" in result["ray_actor_graph"]["handles"]
    assert len(calls) == len(plan.actors)
    assert all(call["options"]["num_gpus"] == 0 for call in calls)
