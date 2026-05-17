"""Ray actor placement plan for quantity-only GPU topology."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nano_rl.config import LaunchConfig, TrainerBackendName
from nano_rl.runtime.slot import GpuTopology, RolloutReplicaSpec


class RayActorSpec(BaseModel):
    """Declarative actor launch spec used before real Ray actor startup."""

    model_config = ConfigDict(frozen=True)

    name: str
    actor_type: str
    import_path: str | None = None
    num_cpus: float = Field(default=0, ge=0)
    num_gpus: float = Field(default=0, ge=0)
    memory_bytes: int | None = Field(default=None, ge=1)
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
    node_num_cpus: int | None = Field(default=None, ge=1)
    backend_summary: dict[str, str] = Field(default_factory=dict)

    @property
    def actors_requesting_ray_gpus(self) -> tuple[RayActorSpec, ...]:
        return tuple(actor for actor in self.actors if actor.num_gpus > 0)


def build_ray_launch_plan(config: LaunchConfig) -> RayLaunchPlan:
    """Build the Ray actor/resource plan without importing Ray."""

    placement = config.runtime.ray.placement
    role_classes = config.runtime.ray.gpu_manager.role_classes
    toggle_offload = config.runtime.ray.gpu_manager.hybrid_toggle.offload
    mock_actor_memory_bytes = _mock_actor_memory_bytes(config)

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
        vllm_weight_sync_backend = _vllm_weight_sync_backend_for_replica(config, replica)
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
                        "engine_kwargs": _vllm_engine_kwargs(config),
                        "sampling_params": config.rollout.vllm.sampling_params,
                        "weight_sync_backend": vllm_weight_sync_backend,
                        "require_weight_sync": config.rollout.vllm.require_weight_sync,
                        "gpu_ids": list(replica.gpu_ids),
                        "holder_ids": list(replica.worker_ids),
                        "offload_strategy": toggle_offload.rollout_engine,
                        "vllm_sleep_level": toggle_offload.vllm_sleep_level or 2,
                        "residual_gpu_memory_budget_mb": toggle_offload.residual_gpu_memory_budget_mb,
                        "huggingface": config.rollout.huggingface.model_dump(mode="json"),
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
                    "weight_sync_backend": vllm_weight_sync_backend,
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
                        "rendezvous": config.trainer.fsdp2.rendezvous if config.trainer.fsdp2 else None,
                        "store_endpoint": config.trainer.fsdp2.store_endpoint if config.trainer.fsdp2 else None,
                        "model_path": config.model.model_path,
                        "checkpoint_dir": config.trainer.checkpoint_dir,
                        "extra": _trainer_backend_extra(config),
                    },
                ),
                metadata={"rank": rank.rank, "gpu_id": rank.gpu_id, "backend": config.trainer.backend},
            )
        )

    if mock_actor_memory_bytes is not None:
        actors = [
            actor.model_copy(update={"memory_bytes": mock_actor_memory_bytes})
            for actor in actors
        ]

    _validate_role_resource_uniqueness(actors)
    return RayLaunchPlan(
        node_custom_resources=node_custom_resources,
        actors=tuple(actors),
        node_num_cpus=config.runtime.local.cpus,
        backend_summary={
            "rollout": config.rollout.backend,
            "trainer": config.trainer.backend,
            "weight_store": config.weight_transfer.store.backend,
            "data": config.data.source_type,
            "reward": config.reward.backend,
        },
    )


def _mock_actor_memory_bytes(config: LaunchConfig) -> int | None:
    if not config.mock.enabled or config.mock.ray_actor_memory_mb is None:
        return None
    return config.mock.ray_actor_memory_mb * 1024 * 1024


def _trainer_backend_extra(config: LaunchConfig) -> dict[str, Any]:
    extra: dict[str, Any] = {"learning_rate": config.algorithm.learning_rate}
    if config.trainer.backend == TrainerBackendName.FSDP2:
        if config.trainer.fsdp2 is None:
            raise ValueError("trainer.fsdp2 is required when trainer.backend=fsdp2")
        extra.update(
            {
                "mixed_precision": config.trainer.fsdp2.mixed_precision,
                "sharding": config.trainer.fsdp2.sharding,
                "dist_backend": config.trainer.fsdp2.dist_backend,
                "trust_remote_code": config.trainer.fsdp2.trust_remote_code,
            }
        )
        if config.trainer.fsdp2.max_length is not None:
            extra["max_length"] = config.trainer.fsdp2.max_length
    elif config.trainer.backend == TrainerBackendName.MOCK:
        extra["mock"] = config.trainer.mock.model_dump(mode="json")
        if str(config.weight_transfer.store.backend) != "checkpoint":
            extra["weight_store"] = config.weight_transfer.store.model_dump(mode="json")
    return extra


def _vllm_engine_kwargs(config: LaunchConfig) -> dict[str, Any]:
    engine_kwargs = dict(config.rollout.vllm.engine_kwargs)
    if config.runtime.ray.gpu_manager.hybrid_toggle.offload.rollout_engine == "vllm_sleep":
        engine_kwargs["enable_sleep_mode"] = True
    return engine_kwargs


def _vllm_weight_sync_backend_for_replica(config: LaunchConfig, replica: RolloutReplicaSpec) -> str:
    configured = config.rollout.vllm.weight_sync_backend
    if configured != "auto":
        return configured
    if replica.topology == GpuTopology.SHARED:
        return "ipc"
    return "nccl"


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
