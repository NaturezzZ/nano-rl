"""Ray actor placement plan for quantity-only GPU topology."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nano_rl.config import LaunchConfig, TrainerBackendName


class RayActorSpec(BaseModel):
    """Declarative actor launch spec used before real Ray actor startup."""

    model_config = ConfigDict(frozen=True)

    name: str
    actor_type: str
    import_path: str | None = None
    num_cpus: float = Field(default=0, ge=0)
    num_gpus: float = Field(default=0, ge=0)
    resources: dict[str, float] = Field(default_factory=dict)
    init_args: tuple[Any, ...] = Field(default_factory=tuple)
    init_kwargs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RayLaunchPlan(BaseModel):
    """Resolved Ray actor/resource plan.

    ``node_custom_resources`` is the local-node resource declaration that lets
    rollout and trainer actors overlap on the same physical GPU id without
    sharing a Ray ``num_gpus`` token.  CUDA execution is still controlled by
    ``GpuLeaseManagerActor``.
    """

    model_config = ConfigDict(frozen=True)

    node_custom_resources: dict[str, float]
    actors: tuple[RayActorSpec, ...]
    backend_summary: dict[str, str] = Field(default_factory=dict)

    @property
    def actors_requesting_ray_gpus(self) -> tuple[RayActorSpec, ...]:
        return tuple(actor for actor in self.actors if actor.num_gpus > 0)


def build_ray_launch_plan(config: LaunchConfig) -> RayLaunchPlan:
    """Build the Ray actor/resource plan without importing Ray."""

    placement = config.runtime.ray.placement
    role_classes = config.runtime.ray.gpu_manager.role_classes

    node_custom_resources: dict[str, float] = {}
    actors: list[RayActorSpec] = [
        RayActorSpec(name="controller", actor_type="ControllerActor", num_cpus=1),
        RayActorSpec(
            name="gpu-lease-manager",
            actor_type="GpuLeaseManagerActor",
            num_cpus=1,
            init_args=(config.gpu_plan.model_dump(mode="json"),),
        ),
        RayActorSpec(
            name="rollout-manager",
            actor_type="RolloutManagerActor",
            import_path=role_classes.rollout_manager,
            num_cpus=1,
            init_args=(
                [replica.model_dump(mode="json") for replica in config.gpu_plan.rollout_replicas],
                max(1, config.control.max_pending_rollout_refs // config.gpu_plan.rollout_replica_count),
            ),
        ),
        RayActorSpec(
            name="trainer-coordinator",
            actor_type="TrainerCoordinatorActor",
            num_cpus=1,
            init_args=([rank.model_dump(mode="json") for rank in config.gpu_plan.trainer_ranks],),
        ),
        RayActorSpec(
            name="sample-queue",
            actor_type="SampleQueueActor",
            num_cpus=1,
            init_args=(
                config.control.max_policy_lag,
                config.control.sample_ttl_sec,
                config.control.queue_high_watermark,
            ),
        ),
        RayActorSpec(name="weight-registry", actor_type="WeightRegistryActor", num_cpus=1),
        RayActorSpec(name="metrics", actor_type="MetricsActor", num_cpus=1),
    ]

    reward = placement.reward
    if reward and reward.num_actors:
        for index in range(reward.num_actors):
            actors.append(
                RayActorSpec(
                    name=f"reward-{index}",
                    actor_type="RewardActor",
                    num_cpus=reward.cpus_per_actor or 0,
                )
            )

    for replica in config.gpu_plan.rollout_replicas:
        rollout_resources = {f"rollout_gpu_{gpu_id}": 1 for gpu_id in replica.gpu_ids}
        node_custom_resources.update(rollout_resources)
        actors.append(
            RayActorSpec(
                name=replica.replica_id,
                actor_type="RolloutReplicaControllerActor",
                import_path=replica.controller_class,
                num_cpus=1,
                num_gpus=0,
                resources=rollout_resources,
                init_args=(
                    replica.replica_id,
                    list(replica.gpu_ids),
                    list(replica.worker_ids),
                    {
                        "backend": config.rollout.backend,
                        "model_path": config.model.model_path,
                        "tokenizer_path": config.model.tokenizer_path,
                        "tensor_parallel_size": len(replica.gpu_ids),
                        "dtype": config.rollout.vllm.dtype,
                        "max_model_len": config.rollout.vllm.max_model_len,
                        "trust_remote_code": config.rollout.vllm.trust_remote_code,
                        "engine_kwargs": config.rollout.vllm.engine_kwargs,
                        "sampling_params": config.rollout.vllm.sampling_params,
                        "gpu_ids": list(replica.gpu_ids),
                        "holder_ids": list(replica.worker_ids),
                        "mock": config.rollout.mock.model_dump(mode="json"),
                    },
                    config.reward.model_dump(mode="json"),
                ),
                metadata={
                    "replica_id": replica.replica_id,
                    "dp_rank": replica.dp_rank,
                    "topology": replica.topology,
                    "gpu_ids": list(replica.gpu_ids),
                    "worker_ids": list(replica.worker_ids),
                    "backend": config.rollout.backend,
                    "weight_transfer_method": config.weight_transfer.method,
                },
            )
        )

    for worker in config.gpu_plan.rollout_workers:
        actors.append(
            RayActorSpec(
                name=worker.worker_id,
                actor_type="RolloutWorkerActor",
                import_path=worker.worker_class,
                num_gpus=0,
                resources={},
                init_args=(worker.worker_id, worker.gpu_id, worker.dp_rank, worker.tp_rank),
                metadata={
                    "replica_id": worker.replica_id,
                    "dp_rank": worker.dp_rank,
                    "tp_rank": worker.tp_rank,
                    "gpu_id": worker.gpu_id,
                    "topology": worker.topology,
                },
            )
        )

    for rank in config.gpu_plan.trainer_ranks:
        node_custom_resources[rank.ray_resource] = 1
        actors.append(
            RayActorSpec(
                name=f"trainer-rank-{rank.rank}",
                actor_type="TrainerRankActor",
                import_path=rank.trainer_class,
                num_cpus=placement.trainer.cpus_per_rank,
                num_gpus=0,
                resources={rank.ray_resource: 1},
                init_args=(
                    rank.rank,
                    rank.gpu_id,
                    0,
                    {
                        "backend": config.trainer.backend,
                        "rank": rank.rank,
                        "world_size": config.gpu_plan.trainer_rank_count,
                        "gpu_id": rank.gpu_id,
                        "holder_id": f"trainer-rank-{rank.rank}",
                        "group_epoch": 0,
                        "model_path": config.model.model_path,
                        "checkpoint_dir": config.trainer.checkpoint_dir,
                        "extra": _trainer_backend_extra(config),
                    },
                ),
                metadata={"rank": rank.rank, "gpu_id": rank.gpu_id, "backend": config.trainer.backend},
            )
        )

    _validate_role_resource_uniqueness(actors)
    return RayLaunchPlan(
        node_custom_resources=node_custom_resources,
        actors=tuple(actors),
        backend_summary={
            "rollout": config.rollout.backend,
            "trainer": config.trainer.backend,
            "weight_store": config.weight_transfer.store.backend,
            "data": config.data.source_type,
            "reward": config.reward.backend,
        },
    )


def _trainer_backend_extra(config: LaunchConfig) -> dict[str, Any]:
    extra: dict[str, Any] = {"learning_rate": config.algorithm.learning_rate}
    if config.trainer.backend == TrainerBackendName.FSDP2:
        if config.trainer.fsdp2 is None:
            raise ValueError("trainer.fsdp2 is required when trainer.backend=fsdp2")
        extra.update(
            {
                "mixed_precision": config.trainer.fsdp2.mixed_precision,
                "sharding": config.trainer.fsdp2.sharding,
            }
        )
    elif config.trainer.backend == TrainerBackendName.MOCK:
        extra["mock"] = config.trainer.mock.model_dump(mode="json")
        if str(config.weight_transfer.store.backend) != "checkpoint":
            extra["weight_store"] = config.weight_transfer.store.model_dump(mode="json")
    return extra


def _validate_role_resource_uniqueness(actors: list[RayActorSpec]) -> None:
    rollout_resources: set[str] = set()
    train_resources: set[str] = set()
    for actor in actors:
        for resource in actor.resources:
            if resource.startswith("rollout_gpu_"):
                if resource in rollout_resources:
                    raise ValueError(f"duplicate rollout resource assignment: {resource}")
                rollout_resources.add(resource)
            if resource.startswith("train_gpu_"):
                if resource in train_resources:
                    raise ValueError(f"duplicate train resource assignment: {resource}")
                train_resources.add(resource)
