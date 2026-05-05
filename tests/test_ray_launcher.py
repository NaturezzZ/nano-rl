from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from nano_rl.config import load_launch_config
from nano_rl.exceptions import RayClusterError
from nano_rl.runtime.ray.cluster import RayClusterController
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


class FakeRayModule:
    def __init__(
        self,
        *,
        fail_auto_connect: bool = False,
        fail_every_connect: bool = False,
        initialized: bool = False,
    ) -> None:
        self.fail_auto_connect = fail_auto_connect
        self.fail_every_connect = fail_every_connect
        self.initialized = initialized
        self.init_calls: list[dict[str, Any]] = []

    def is_initialized(self) -> bool:
        return self.initialized

    def init(self, **kwargs: Any) -> object:
        self.init_calls.append(kwargs)
        address = kwargs.get("address")
        if self.fail_every_connect and address:
            raise RuntimeError("connection failed")
        if self.fail_auto_connect and address == "auto":
            raise RuntimeError("no cluster found")
        self.initialized = True
        return object()


def _fake_builders(
    actor_types: set[str],
    calls: list[dict[str, Any]],
) -> dict[str, RemoteActorClassBuilder]:
    return {
        actor_type: (lambda actor_type=actor_type: FakeRemoteActorClass(actor_type, calls))
        for actor_type in actor_types
    }


def test_actor_graph_launcher_uses_zero_ray_gpus_and_role_resources(caplog: pytest.LogCaptureFixture) -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    plan = build_ray_launch_plan(config)
    calls: list[dict[str, Any]] = []
    caplog.set_level(logging.INFO, logger="nano_rl.runtime.ray.launcher")

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
    messages = [record.getMessage() for record in caplog.records]
    assert any("starting Ray actor graph creation" in message for message in messages)
    assert any("creating Ray actor: name=controller" in message for message in messages)
    assert any("Ray actor graph creation completed" in message for message in messages)


def test_actor_graph_launcher_passes_init_args_from_launch_plan() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
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


def test_actor_graph_launcher_passes_mock_memory_hint_to_ray_options() -> None:
    config = load_launch_config(ROOT / "recipes/mock_disaggregated.yaml")
    plan = build_ray_launch_plan(config)
    calls: list[dict[str, Any]] = []

    graph = RayActorGraphLauncher(
        plan,
        namespace=config.runtime.ray.namespace,
        actor_class_builders=_fake_builders({spec.actor_type for spec in plan.actors}, calls),
    ).start(dry_run=False)

    expected = 64 * 1024 * 1024
    assert all(call["options"]["memory"] == expected for call in calls)
    assert set(graph.resource_summary["actor_memory_bytes"].values()) == {expected}


def test_actor_graph_dry_run_does_not_create_remote_actors() -> None:
    config = load_launch_config(ROOT / "recipes/collocated.yaml")
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


def test_ray_cluster_controller_auto_connects_existing_cluster() -> None:
    ray = FakeRayModule()
    result = RayClusterController(
        namespace="nano-rl",
        ray_address="auto",
        node_custom_resources={"rollout_gpu_0": 1},
        ray_module=ray,
    ).ensure_initialized()

    assert result.startup_mode == "connected_existing"
    assert result.created_local_cluster is False
    assert ray.init_calls == [{"namespace": "nano-rl", "address": "auto", "logging_level": "warning"}]


def test_ray_cluster_controller_auto_falls_back_to_local_cluster(caplog: pytest.LogCaptureFixture) -> None:
    ray = FakeRayModule(fail_auto_connect=True)
    caplog.set_level(logging.INFO, logger="nano_rl.runtime.ray.cluster")
    result = RayClusterController(
        namespace="nano-rl",
        ray_address="auto",
        node_custom_resources={"rollout_gpu_0": 1, "train_gpu_0": 1},
        ray_module=ray,
    ).ensure_initialized()

    assert result.startup_mode == "created_local_after_auto_failed"
    assert result.created_local_cluster is True
    assert result.fallback_reason == "RuntimeError: no cluster found"
    assert ray.init_calls == [
        {"namespace": "nano-rl", "address": "auto", "logging_level": "warning"},
        {
            "namespace": "nano-rl",
            "resources": {"rollout_gpu_0": 1, "train_gpu_0": 1},
            "logging_level": "warning",
        },
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any("Ray runtime: checking for existing Ray cluster" in message for message in messages)
    assert any("Ray runtime: no existing Ray cluster found; creating local Ray cluster" in message for message in messages)
    assert any("Ray runtime: local Ray cluster created" in message for message in messages)
    assert not any("no cluster found" in message for message in messages)
    assert not any("custom_resources" in message for message in messages)


def test_ray_cluster_controller_explicit_address_fails_without_local_fallback() -> None:
    ray = FakeRayModule(fail_every_connect=True)

    with pytest.raises(RayClusterError, match="failed to connect to Ray cluster"):
        RayClusterController(
            namespace="nano-rl",
            ray_address="ray://head:10001",
            node_custom_resources={"rollout_gpu_0": 1},
            ray_module=ray,
        ).ensure_initialized()

    assert ray.init_calls == [
        {"namespace": "nano-rl", "address": "ray://head:10001", "logging_level": "warning"}
    ]


def test_actor_graph_launcher_records_auto_fallback_cluster_startup() -> None:
    config = load_launch_config(ROOT / "recipes/collocated.yaml")
    plan = build_ray_launch_plan(config)
    calls: list[dict[str, Any]] = []
    ray = FakeRayModule(fail_auto_connect=True)

    graph = RayActorGraphLauncher(
        plan,
        namespace=config.runtime.ray.namespace,
        actor_class_builders=_fake_builders({spec.actor_type for spec in plan.actors}, calls),
        start_ray=True,
        ray_address="auto",
        ray_module=ray,
    ).start(dry_run=False)

    assert graph.ray_cluster is not None
    assert graph.ray_cluster.startup_mode == "created_local_after_auto_failed"
    assert graph.to_dict()["ray_cluster"]["created_local_cluster"] is True
    assert "controller" in graph.handles


def test_ray_driver_train_still_returns_plan_without_starting_actor_graph(monkeypatch) -> None:
    config = load_launch_config(ROOT / "recipes/collocated.yaml")

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
    base_config = load_launch_config(ROOT / "recipes/collocated.yaml")
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


def test_ray_driver_train_runs_training_loop_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    base_config = load_launch_config(ROOT / "recipes/mock_collocated.yaml")
    config = base_config.model_copy(
        update={"run": base_config.run.model_copy(update={"start_ray_actors": True})}
    )

    class FakeGraph:
        handles = {"controller": object()}

        def to_dict(self) -> dict[str, object]:
            return {"handles": {"controller": "fake"}, "dry_run": False}

    def fake_start_actor_graph(self: RayDriver, **kwargs: Any) -> FakeGraph:
        return FakeGraph()

    def fake_run_training_loop(self: Any) -> dict[str, object]:
        return {
            "status": "completed",
            "steps_completed": 2,
            "final_weight": {"version_id": 2},
        }

    monkeypatch.setattr(RayDriver, "start_actor_graph", fake_start_actor_graph)
    monkeypatch.setattr("nano_rl.runtime.ray.driver.RayTrainingLoop.run", fake_run_training_loop)

    result = RayDriver(config).train(validate_artifacts=False, start_ray=False, run_training_loop=True)

    assert result["execution_status"] == "ray_training_completed"
    assert result["training_result"]["steps_completed"] == 2
    assert result["training_result"]["final_weight"]["version_id"] == 2
