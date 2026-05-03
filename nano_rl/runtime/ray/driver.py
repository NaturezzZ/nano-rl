"""Ray driver entrypoint."""

from __future__ import annotations

from nano_rl.config import LaunchConfig
from nano_rl.runtime.artifacts import validate_input_artifacts
from nano_rl.runtime.controller import ControllerCore
from nano_rl.runtime.ray.launcher import (
    RayActorGraph,
    RayActorGraphLauncher,
    RemoteActorClassBuilder,
)
from nano_rl.runtime.ray.placement import build_ray_launch_plan


class RayDriver:
    """Thin driver that owns Ray startup boundaries.

    Planning is always available without importing Ray backend dependencies.
    When ``run.start_ray_actors`` is enabled, ``train`` also starts the Ray
    actor graph whose rollout and trainer actors own the concrete backend
    adapters.
    """

    def __init__(self, launch_config: LaunchConfig):
        self.launch_config = launch_config

    def validate(self, *, validate_artifacts: bool = True) -> dict[str, object]:
        if validate_artifacts:
            validate_input_artifacts(self.launch_config)
        return self._planned_runtime()

    def dry_run(self, *, validate_artifacts: bool = False) -> dict[str, object]:
        if validate_artifacts:
            validate_input_artifacts(self.launch_config)
        return self._planned_runtime()

    def train(
        self,
        *,
        validate_artifacts: bool = True,
        actor_class_builders: dict[str, RemoteActorClassBuilder] | None = None,
        start_ray: bool = True,
    ) -> dict[str, object]:
        if validate_artifacts:
            validate_input_artifacts(self.launch_config)
        plan = self._planned_runtime()
        if not self.launch_config.run.start_ray_actors:
            plan["execution_status"] = "planned_backend_integrated"
            plan["start_ray_actors"] = False
            return plan

        graph = self.start_actor_graph(
            actor_class_builders=actor_class_builders,
            start_ray=start_ray,
        )
        plan["execution_status"] = "ray_actor_graph_started"
        plan["start_ray_actors"] = True
        plan["ray_actor_graph"] = graph.to_dict()
        return plan

    def build_actor_graph(
        self,
        *,
        actor_class_builders: dict[str, RemoteActorClassBuilder] | None = None,
        dry_run: bool = True,
    ) -> RayActorGraph:
        ray_launch_plan = build_ray_launch_plan(self.launch_config)
        launcher = RayActorGraphLauncher(
            ray_launch_plan,
            namespace=self.launch_config.runtime.ray.namespace,
            actor_class_builders=actor_class_builders,
            ray_address=self.launch_config.runtime.ray.address,
        )
        return launcher.start(dry_run=dry_run)

    def start_actor_graph(
        self,
        *,
        actor_class_builders: dict[str, RemoteActorClassBuilder] | None = None,
        start_ray: bool = False,
    ) -> RayActorGraph:
        ray_launch_plan = build_ray_launch_plan(self.launch_config)
        launcher = RayActorGraphLauncher(
            ray_launch_plan,
            namespace=self.launch_config.runtime.ray.namespace,
            actor_class_builders=actor_class_builders,
            ray_address=self.launch_config.runtime.ray.address,
            start_ray=start_ray,
        )
        return launcher.start(dry_run=False)

    def _planned_runtime(self) -> dict[str, object]:
        plan = ControllerCore(self.launch_config).dry_run()
        ray_launch_plan = build_ray_launch_plan(self.launch_config)
        plan["ray_launch_plan"] = ray_launch_plan.model_dump(mode="json")
        return plan
