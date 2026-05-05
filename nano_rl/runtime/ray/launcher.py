"""Ray actor graph startup boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Mapping

from nano_rl.runtime.ray.actors import (
    build_gpu_lease_manager_actor_class,
    build_metrics_actor_class,
    build_rollout_manager_actor_class,
    build_rollout_replica_controller_actor_class,
    build_rollout_worker_actor_class,
    build_sample_queue_actor_class,
    build_trainer_coordinator_actor_class,
    build_trainer_rank_actor_class,
    build_weight_registry_actor_class,
)
from nano_rl.runtime.ray.cluster import RayClusterController, RayClusterStartupResult
from nano_rl.runtime.ray.placement import RayActorSpec, RayLaunchPlan


RemoteActorClassBuilder = Callable[[], Any]
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RayActorGraph:
    """Structured result of actor graph construction."""

    handles: Mapping[str, Any]
    launch_plan: RayLaunchPlan
    resource_summary: dict[str, Any]
    dry_run: bool = False
    ray_cluster: RayClusterStartupResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "handles": {name: repr(handle) for name, handle in self.handles.items()},
            "launch_plan": self.launch_plan.model_dump(mode="json"),
            "resource_summary": self.resource_summary,
            "dry_run": self.dry_run,
            "ray_cluster": self.ray_cluster.to_dict() if self.ray_cluster else None,
        }


@dataclass
class RayActorGraphLauncher:
    """Create Ray actors from a declarative ``RayLaunchPlan``.

    The launcher is intentionally backend-agnostic: it does not import vLLM or
    FSDP2, and it never assigns long-lived actors a Ray GPU token. Tests can
    inject fake actor builders that implement the Ray ``options().remote()``
    surface without importing or starting Ray.
    """

    launch_plan: RayLaunchPlan
    namespace: str | None = None
    actor_class_builders: Mapping[str, RemoteActorClassBuilder] | None = None
    start_ray: bool = False
    ray_address: str | None = None
    dedup_logs: bool = True
    ray_module: Any | None = None
    _handles: dict[str, Any] = field(default_factory=dict, init=False)
    _ray_cluster: RayClusterStartupResult | None = field(default=None, init=False)

    def start(self, *, dry_run: bool = False) -> RayActorGraph:
        self._validate_no_ray_gpu_tokens()
        if dry_run:
            logger.info("Ray actor graph dry run: actor_count=%s", len(self.launch_plan.actors))
            return RayActorGraph(
                handles={},
                launch_plan=self.launch_plan,
                resource_summary=self.resource_summary(),
                dry_run=True,
            )

        self._ray_cluster = None
        if self.start_ray:
            self._ray_cluster = self._ensure_ray_initialized()

        handles: dict[str, Any] = {}
        builders = self._actor_class_builders()
        logger.info("starting Ray actor graph creation: actor_count=%s", len(self.launch_plan.actors))
        for spec in self.launch_plan.actors:
            builder = builders.get(spec.actor_type)
            if builder is None:
                raise ValueError(f"no Ray actor builder registered for actor_type={spec.actor_type!r}")
            logger.info(
                "creating Ray actor: name=%s type=%s num_cpus=%s memory_bytes=%s resources=%s",
                spec.name,
                spec.actor_type,
                spec.num_cpus,
                spec.memory_bytes,
                dict(spec.resources),
            )
            started_at = time.monotonic()
            remote_class = builder()
            try:
                handles[spec.name] = remote_class.options(**self._actor_options(spec)).remote(
                    *spec.init_args,
                    **spec.init_kwargs,
                )
            except Exception:
                logger.exception("Ray actor creation failed: name=%s type=%s", spec.name, spec.actor_type)
                raise
            logger.info(
                "created Ray actor handle: name=%s type=%s elapsed_ms=%.1f",
                spec.name,
                spec.actor_type,
                (time.monotonic() - started_at) * 1000,
            )

        self._handles = handles
        logger.info("Ray actor graph creation completed: handle_count=%s", len(handles))
        return RayActorGraph(
            handles=handles,
            launch_plan=self.launch_plan,
            resource_summary=self.resource_summary(),
            dry_run=False,
            ray_cluster=self._ray_cluster,
        )

    def resource_summary(self) -> dict[str, Any]:
        rollout_resources = sorted(
            resource
            for resource in self.launch_plan.node_custom_resources
            if resource.startswith("rollout_gpu_")
        )
        train_resources = sorted(
            resource
            for resource in self.launch_plan.node_custom_resources
            if resource.startswith("train_gpu_")
        )
        shared_gpu_ids = sorted(
            {
                int(resource.removeprefix("rollout_gpu_"))
                for resource in rollout_resources
            }
            & {
                int(resource.removeprefix("train_gpu_"))
                for resource in train_resources
            }
        )
        actor_counts: dict[str, int] = {}
        for spec in self.launch_plan.actors:
            actor_counts[spec.actor_type] = actor_counts.get(spec.actor_type, 0) + 1

        return {
            "actor_count": len(self.launch_plan.actors),
            "actor_counts_by_type": actor_counts,
            "node_custom_resources": dict(self.launch_plan.node_custom_resources),
            "rollout_resources": rollout_resources,
            "train_resources": train_resources,
            "shared_gpu_ids": shared_gpu_ids,
            "actors_requesting_ray_gpus": [
                spec.name for spec in self.launch_plan.actors_requesting_ray_gpus
            ],
            "actor_memory_bytes": {
                spec.name: spec.memory_bytes
                for spec in self.launch_plan.actors
                if spec.memory_bytes is not None
            },
        }

    def _actor_options(self, spec: RayActorSpec) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_cpus": spec.num_cpus,
            "num_gpus": 0,
            "resources": dict(spec.resources),
            "name": spec.name,
        }
        if spec.memory_bytes is not None:
            options["memory"] = spec.memory_bytes
        if self.namespace:
            options["namespace"] = self.namespace
        return options

    def _actor_class_builders(self) -> Mapping[str, RemoteActorClassBuilder]:
        if self.actor_class_builders is not None:
            return self.actor_class_builders
        return default_actor_class_builders()

    def _validate_no_ray_gpu_tokens(self) -> None:
        offenders = self.launch_plan.actors_requesting_ray_gpus
        if offenders:
            names = ", ".join(spec.name for spec in offenders)
            raise ValueError(f"long-lived Ray actors must not request num_gpus>0: {names}")

    def _ensure_ray_initialized(self) -> RayClusterStartupResult:
        return RayClusterController(
            namespace=self.namespace,
            ray_address=self.ray_address,
            node_custom_resources=self.launch_plan.node_custom_resources,
            node_num_cpus=self.launch_plan.node_num_cpus,
            dedup_logs=self.dedup_logs,
            ray_module=self.ray_module,
        ).ensure_initialized()


def default_actor_class_builders() -> Mapping[str, RemoteActorClassBuilder]:
    return {
        "ControllerActor": build_controller_actor_class,
        "GpuLeaseManagerActor": build_gpu_lease_manager_actor_class,
        "RolloutManagerActor": build_rollout_manager_actor_class,
        "TrainerCoordinatorActor": build_trainer_coordinator_actor_class,
        "SampleQueueActor": build_sample_queue_actor_class,
        "WeightRegistryActor": build_weight_registry_actor_class,
        "MetricsActor": build_metrics_actor_class,
        "RewardActor": build_reward_actor_class,
        "RolloutReplicaControllerActor": build_rollout_replica_controller_actor_class,
        "RolloutWorkerActor": build_rollout_worker_actor_class,
        "TrainerRankActor": build_trainer_rank_actor_class,
    }


def build_controller_actor_class():
    import ray

    @ray.remote(num_cpus=1, num_gpus=0)
    class ControllerActor:
        def __init__(self) -> None:
            _configure_actor_logging()
            logger.info("ControllerActor init started")
            self.started = True
            logger.info("ControllerActor init completed")

        def ping(self) -> bool:
            return self.started

    return ControllerActor


def build_reward_actor_class():
    import ray

    @ray.remote(num_cpus=0, num_gpus=0)
    class RewardActor:
        def __init__(self) -> None:
            _configure_actor_logging()
            logger.info("RewardActor init started")
            self.started = True
            logger.info("RewardActor init completed")

        def ping(self) -> bool:
            return self.started

    return RewardActor


def _configure_actor_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
