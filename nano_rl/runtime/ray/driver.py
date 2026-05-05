"""Ray driver entrypoint."""

from __future__ import annotations

import logging

from nano_rl.config import LaunchConfig
from nano_rl.runtime.artifacts import validate_input_artifacts
from nano_rl.runtime.controller import ControllerCore
from nano_rl.runtime.ray.launcher import (
    RayActorGraph,
    RayActorGraphLauncher,
    RemoteActorClassBuilder,
)
from nano_rl.runtime.ray.placement import build_ray_launch_plan
from nano_rl.runtime.ray.training import RayTrainingLoop


logger = logging.getLogger(__name__)


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
        logger.info("validate started: validate_artifacts=%s", validate_artifacts)
        if validate_artifacts:
            validate_input_artifacts(self.launch_config)
        plan = self._planned_runtime()
        logger.info("validate completed: actor_count=%s", len(plan["ray_launch_plan"]["actors"]))
        return plan

    def dry_run(self, *, validate_artifacts: bool = False) -> dict[str, object]:
        logger.info("dry_run started: validate_artifacts=%s", validate_artifacts)
        if validate_artifacts:
            validate_input_artifacts(self.launch_config)
        plan = self._planned_runtime()
        logger.info("dry_run completed: actor_count=%s", len(plan["ray_launch_plan"]["actors"]))
        return plan

    def train(
        self,
        *,
        validate_artifacts: bool = True,
        actor_class_builders: dict[str, RemoteActorClassBuilder] | None = None,
        start_ray: bool = True,
        run_training_loop: bool | None = None,
    ) -> dict[str, object]:
        logger.info(
            "train started: validate_artifacts=%s start_ray_actors=%s start_ray=%s",
            validate_artifacts,
            self.launch_config.run.start_ray_actors,
            start_ray,
        )
        if validate_artifacts:
            validate_input_artifacts(self.launch_config)
        plan = self._planned_runtime()
        if not self.launch_config.run.start_ray_actors:
            logger.info("train planned only: run.start_ray_actors=false actor_count=%s", len(plan["ray_launch_plan"]["actors"]))
            plan["execution_status"] = "planned_backend_integrated"
            plan["start_ray_actors"] = False
            return plan

        logger.info("starting Ray actor graph: actor_count=%s", len(plan["ray_launch_plan"]["actors"]))
        try:
            graph = self.start_actor_graph(
                actor_class_builders=actor_class_builders,
                start_ray=start_ray,
            )
            graph_dict = graph.to_dict()
            should_run_training = run_training_loop
            if should_run_training is None:
                should_run_training = start_ray and actor_class_builders is None
            if should_run_training:
                logger.info("starting Ray training loop")
                training_result = RayTrainingLoop(self.launch_config, graph).run()
                logger.info(
                    "Ray training loop completed: steps_completed=%s final_weight_version=%s",
                    training_result["steps_completed"],
                    training_result["final_weight"]["version_id"],
                )
            else:
                training_result = None
        finally:
            if start_ray:
                logger.info("shutting down Ray runtime after actor graph startup attempt")
                self._shutdown_ray_runtime()
        plan["execution_status"] = "ray_training_completed" if training_result is not None else "ray_actor_graph_started"
        plan["start_ray_actors"] = True
        plan["ray_actor_graph"] = graph_dict
        if training_result is not None:
            plan["training_result"] = training_result
        logger.info("Ray actor graph started: handle_count=%s", len(graph.handles))
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
            dedup_logs=self.launch_config.runtime.ray.dedup_logs,
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
            dedup_logs=self.launch_config.runtime.ray.dedup_logs,
        )
        return launcher.start(dry_run=False)

    def _planned_runtime(self) -> dict[str, object]:
        logger.info("building backend-integrated runtime plan")
        plan = ControllerCore(self.launch_config).dry_run()
        ray_launch_plan = build_ray_launch_plan(self.launch_config)
        plan["ray_launch_plan"] = ray_launch_plan.model_dump(mode="json")
        logger.info(
            "runtime plan built: rollout_replicas=%s trainer_ranks=%s actor_count=%s",
            plan["rollout_replica_count"],
            plan["trainer_rank_count"],
            len(ray_launch_plan.actors),
        )
        return plan

    def _shutdown_ray_runtime(self) -> None:
        import ray

        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray runtime shutdown completed")
