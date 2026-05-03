"""YAML-first runtime configuration models.

The public YAML surface keeps user-facing topology names such as
``collocated`` and ``disaggregated``.  ``LaunchConfig`` is the resolved,
internal representation that the runtime consumes.
"""

from __future__ import annotations

from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nano_rl.exceptions import ConfigError
from nano_rl.runtime.protocols import WeightTransferMethod
from nano_rl.runtime.slot import (
    GpuTopology,
    ResolvedGpuPlan,
    RolloutReplicaSpec,
    RolloutWorkerSpec,
    TrainerRankSpec,
)


class RunIntent(StrEnum):
    TRAIN = "train"
    DRY_RUN = "dry_run"
    VALIDATE = "validate"


class UserMode(StrEnum):
    COLLOCATED = "collocated"
    DISAGGREGATED = "disaggregated"


class CanonicalMode(StrEnum):
    FULLY_SYNC = "fully_sync"
    STANDALONE_HYBRID = "standalone_hybrid"


class SourceType(StrEnum):
    HDFS_URI = "hdfs_uri"
    HDFS_FUSE_PATH = "hdfs_fuse_path"


class RunConfig(BaseModel):
    intent: RunIntent
    emit_resolved_config: bool = False
    start_ray_actors: bool = False


class LocalRuntimeConfig(BaseModel):
    num_gpus: int = Field(ge=1)
    cpus: int = Field(ge=1)


class HdfsConfig(BaseModel):
    cli: str
    read_probe_timeout_sec: int = Field(default=30, ge=1)


class HdfsFuseConfig(BaseModel):
    mount_root: str
    require_mount: bool = True
    read_probe_timeout_sec: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _validate_mount_root(self) -> "HdfsFuseConfig":
        if not self.mount_root.startswith("/"):
            raise ValueError("runtime.storage.hdfs_fuse.mount_root must be absolute")
        return self


class StorageConfig(BaseModel):
    allowed_input_sources: list[SourceType]
    hdfs: HdfsConfig | None = None
    hdfs_fuse: HdfsFuseConfig | None = None


class GpuTopologyCounts(BaseModel):
    rollout_only_gpus: int = Field(ge=0)
    shared_gpus: int = Field(ge=0)
    idle_gpus: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return self.rollout_only_gpus + self.shared_gpus + self.idle_gpus

    @property
    def rollout_capable(self) -> int:
        return self.rollout_only_gpus + self.shared_gpus


class RoleClassesConfig(BaseModel):
    rollout_manager: str
    rollout_replica_controller: str
    rollout_worker: str
    trainer_rank: str | None = None


class ToggleOffloadConfig(BaseModel):
    trainer_model: Literal["cpu_pinned", "cpu"]
    trainer_optimizer: Literal["cpu_pinned", "cpu"]
    rollout_engine: Literal["vllm_sleep", "teardown_and_reload"]
    vllm_sleep_level: Literal[1, 2] | None = None
    cpu_memory_budget_gb: float = Field(ge=1)
    residual_gpu_memory_budget_mb: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_vllm_sleep(self) -> "ToggleOffloadConfig":
        if self.rollout_engine == "vllm_sleep" and self.vllm_sleep_level is None:
            raise ValueError("vllm_sleep_level is required when rollout_engine=vllm_sleep")
        return self


class HybridToggleConfig(BaseModel):
    strategy: Literal["swap_on_toggle", "dual_resident"]
    rollout_drain_timeout_sec: int = Field(ge=1)
    train_drain_timeout_sec: int = Field(ge=1)
    cuda_quiesce_timeout_sec: int = Field(ge=1)
    offload: ToggleOffloadConfig


class GpuManagerConfig(BaseModel):
    enabled: Literal[True]
    lease_manager: Literal[True]
    role_classes: RoleClassesConfig
    topology: GpuTopologyCounts
    hybrid_toggle: HybridToggleConfig


class TrainerPlacementConfig(BaseModel):
    num_ranks: int = Field(ge=1)
    gpus_per_rank: float = Field(ge=0)
    cpus_per_rank: float = Field(default=0, ge=0)


class RolloutPlacementConfig(BaseModel):
    num_replicas: int = Field(ge=1)
    gpus_per_replica: int = Field(ge=1)
    tensor_parallel_size: int = Field(ge=1)


class RewardPlacementConfig(BaseModel):
    num_actors: int | None = Field(default=None, ge=1)
    cpus_per_actor: float | None = Field(default=None, ge=0)


class PlacementConfig(BaseModel):
    trainer: TrainerPlacementConfig
    rollout: RolloutPlacementConfig
    reward: RewardPlacementConfig | None = None


class RayConfig(BaseModel):
    address: str
    namespace: str
    gpu_manager: GpuManagerConfig
    placement: PlacementConfig


class RuntimeSection(BaseModel):
    backend: Literal["ray"]
    local: LocalRuntimeConfig
    storage: StorageConfig
    ray: RayConfig


class TrainerParallelConfig(BaseModel):
    data_parallel_size: int = Field(ge=1)
    fsdp_world_size: int = Field(ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)


class RolloutParallelConfig(BaseModel):
    data_parallel_size: int = Field(ge=1)
    tensor_parallel_size: int = Field(ge=1)


class ParallelConfig(BaseModel):
    trainer: TrainerParallelConfig
    rollout: RolloutParallelConfig


class ModelConfig(BaseModel):
    model_path: str
    tokenizer_path: str | None = None

    @model_validator(mode="after")
    def _normalize_tokenizer_path(self) -> "ModelConfig":
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        return self


class DataConfig(BaseModel):
    source_type: SourceType
    data_path: str
    prompt_column: str = "prompt"


class AlgorithmConfig(BaseModel):
    name: Literal["ppo"]
    max_steps: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    seed: int = Field(default=0, ge=0)


class Fsdp2Config(BaseModel):
    mixed_precision: Literal["bf16", "fp16"]
    sharding: Literal["full_shard", "hybrid_shard"]


class TrainerConfig(BaseModel):
    backend: Literal["fsdp2"]
    global_batch_size: int = Field(ge=1)
    checkpoint_dir: str | None = None
    fsdp2: Fsdp2Config


class PolicyPinConfig(BaseModel):
    latest_ratio: float = Field(default=1.0, ge=0, le=1)
    lagged_ratio: float = Field(default=0.0, ge=0, le=1)


class PrecisionMixConfig(BaseModel):
    bf16: float = Field(default=1.0, ge=0, le=1)
    fp8: float = Field(default=0.0, ge=0, le=1)


class RolloutHybridConfig(BaseModel):
    enabled: bool = True
    precision_mix: PrecisionMixConfig = Field(default_factory=PrecisionMixConfig)
    policy_pin: PolicyPinConfig = Field(default_factory=PolicyPinConfig)


class PartialRolloutConfig(BaseModel):
    enabled: bool = False
    pause_mode: Literal["keep"] = "keep"
    clear_cache: Literal[True] = True
    mixed_policy_samples: Literal["train", "drop"] = "drop"


class RolloutConfig(BaseModel):
    backend: Literal["vllm"]
    partial_rollout: PartialRolloutConfig = Field(default_factory=PartialRolloutConfig)
    hybrid: RolloutHybridConfig = Field(default_factory=RolloutHybridConfig)


class ControlConfig(BaseModel):
    max_policy_lag: int = Field(ge=0)
    sample_ttl_sec: int = Field(ge=1)
    queue_high_watermark: int = Field(ge=1)
    max_pending_rollout_refs: int = Field(ge=1)
    max_pending_train_refs: int = Field(ge=1)


class WeightTransferConfig(BaseModel):
    method: WeightTransferMethod = WeightTransferMethod.LOCALITY_AWARE_CHECKPOINT
    allow_rollout_only_artifact_pull: bool = True
    max_versions_in_flight: int = Field(default=2, ge=1)


class RuntimeConfig(BaseModel):
    """User-facing YAML configuration."""

    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    runtime: RuntimeSection
    mode: UserMode
    parallel: ParallelConfig
    model: ModelConfig
    data: DataConfig
    algorithm: AlgorithmConfig
    trainer: TrainerConfig
    rollout: RolloutConfig
    weight_transfer: WeightTransferConfig = Field(default_factory=WeightTransferConfig)
    control: ControlConfig

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "RuntimeConfig":
        topology = self.runtime.ray.gpu_manager.topology
        placement = self.runtime.ray.placement
        rollout_dp = self.parallel.rollout.data_parallel_size
        rollout_tp = self.parallel.rollout.tensor_parallel_size

        if self.mode == UserMode.COLLOCATED:
            if topology.rollout_only_gpus != 0:
                raise ValueError("collocated mode requires rollout_only_gpus=0")
            if topology.shared_gpus < 1:
                raise ValueError("collocated mode requires at least one shared GPU")
            if self.control.max_policy_lag != 0:
                raise ValueError("collocated mode requires max_policy_lag=0")
            if self.rollout.hybrid.policy_pin.lagged_ratio != 0:
                raise ValueError("collocated mode requires policy_pin.lagged_ratio=0")
        else:
            if topology.rollout_only_gpus < 1 or topology.shared_gpus < 1:
                raise ValueError("disaggregated mode requires rollout_only_gpus>0 and shared_gpus>0")
            if self.control.max_policy_lag < 1:
                raise ValueError("disaggregated mode requires max_policy_lag>=1")

        if topology.total > self.runtime.local.num_gpus:
            raise ValueError("GPU topology counts exceed runtime.local.num_gpus")

        if topology.shared_gpus > 0 and not self.runtime.ray.gpu_manager.role_classes.trainer_rank:
            raise ValueError("shared GPUs require runtime.ray.gpu_manager.role_classes.trainer_rank")

        _validate_import_path(
            "runtime.ray.gpu_manager.role_classes.rollout_manager",
            self.runtime.ray.gpu_manager.role_classes.rollout_manager,
        )
        _validate_import_path(
            "runtime.ray.gpu_manager.role_classes.rollout_replica_controller",
            self.runtime.ray.gpu_manager.role_classes.rollout_replica_controller,
        )
        _validate_import_path(
            "runtime.ray.gpu_manager.role_classes.rollout_worker",
            self.runtime.ray.gpu_manager.role_classes.rollout_worker,
        )
        if self.runtime.ray.gpu_manager.role_classes.trainer_rank:
            _validate_import_path(
                "runtime.ray.gpu_manager.role_classes.trainer_rank",
                self.runtime.ray.gpu_manager.role_classes.trainer_rank,
            )

        if placement.trainer.num_ranks != topology.shared_gpus:
            raise ValueError("placement.trainer.num_ranks must equal shared_gpus")

        if placement.trainer.gpus_per_rank != 1:
            raise ValueError("v0.1 requires placement.trainer.gpus_per_rank=1")

        rollout_gpu_claim = placement.rollout.num_replicas * placement.rollout.gpus_per_replica
        if rollout_gpu_claim != topology.rollout_capable:
            raise ValueError("placement.rollout.num_replicas * gpus_per_replica must equal rollout_only_gpus + shared_gpus")

        if placement.rollout.num_replicas != rollout_dp:
            raise ValueError("placement.rollout.num_replicas must equal parallel.rollout.data_parallel_size")

        if placement.rollout.gpus_per_replica != rollout_tp:
            raise ValueError("placement.rollout.gpus_per_replica must equal parallel.rollout.tensor_parallel_size")

        if topology.rollout_capable != rollout_dp * rollout_tp:
            raise ValueError("rollout GPU count must equal rollout data_parallel_size * tensor_parallel_size")

        if topology.rollout_only_gpus % rollout_tp != 0:
            raise ValueError("rollout_only_gpus must be divisible by rollout tensor_parallel_size")

        if topology.shared_gpus % rollout_tp != 0:
            raise ValueError("shared_gpus must be divisible by rollout tensor_parallel_size")

        if self.parallel.trainer.fsdp_world_size != placement.trainer.num_ranks:
            raise ValueError("parallel.trainer.fsdp_world_size must equal trainer.num_ranks")

        if self.parallel.rollout.tensor_parallel_size != placement.rollout.tensor_parallel_size:
            raise ValueError("parallel.rollout.tensor_parallel_size must match rollout placement")

        if self.parallel.trainer.tensor_parallel_size != 1:
            raise ValueError("v0.1 requires trainer.tensor_parallel_size=1")

        if self.parallel.trainer.pipeline_parallel_size != 1:
            raise ValueError("v0.1 requires trainer.pipeline_parallel_size=1")

        if (
            self.weight_transfer.method == WeightTransferMethod.LOCALITY_AWARE_CHECKPOINT
            and topology.rollout_only_gpus > 0
            and not self.weight_transfer.allow_rollout_only_artifact_pull
        ):
            raise ValueError(
                "locality_aware_checkpoint requires allow_rollout_only_artifact_pull=true "
                "when rollout_only_gpus>0"
            )

        if self.data.source_type not in self.runtime.storage.allowed_input_sources:
            raise ValueError("data.source_type must be listed in allowed_input_sources")

        _validate_source_uri("data.data_path", self.data.source_type, self.data.data_path)
        return self

    @property
    def canonical_mode(self) -> CanonicalMode:
        if self.mode == UserMode.COLLOCATED:
            return CanonicalMode.FULLY_SYNC
        return CanonicalMode.STANDALONE_HYBRID

    def to_launch_config(self) -> "LaunchConfig":
        return LaunchConfig.from_runtime_config(self)


class LaunchConfig(BaseModel):
    """Resolved immutable runtime configuration."""

    model_config = ConfigDict(frozen=True)

    run: RunConfig
    canonical_mode: CanonicalMode
    user_mode: UserMode
    runtime: RuntimeSection
    parallel: ParallelConfig
    model: ModelConfig
    data: DataConfig
    algorithm: AlgorithmConfig
    trainer: TrainerConfig
    rollout: RolloutConfig
    weight_transfer: WeightTransferConfig
    control: ControlConfig
    gpu_plan: ResolvedGpuPlan

    @classmethod
    def from_runtime_config(cls, config: RuntimeConfig) -> "LaunchConfig":
        return cls(
            run=config.run,
            canonical_mode=config.canonical_mode,
            user_mode=config.mode,
            runtime=config.runtime,
            parallel=config.parallel,
            model=config.model,
            data=config.data,
            algorithm=config.algorithm,
            trainer=config.trainer,
            rollout=config.rollout,
            weight_transfer=config.weight_transfer,
            control=config.control,
            gpu_plan=_build_gpu_plan(config),
        )

    @property
    def train_gpu_count(self) -> int:
        return self.gpu_plan.trainer_rank_count

    @property
    def rollout_gpu_count(self) -> int:
        return self.gpu_plan.rollout_gpu_count


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse YAML config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config file must contain a YAML mapping: {path}")
    try:
        return RuntimeConfig.model_validate(raw)
    except Exception as exc:  # pydantic raises ValidationError; keep caller API stable.
        raise ConfigError(str(exc)) from exc


def load_launch_config(path: str | Path) -> LaunchConfig:
    return load_runtime_config(path).to_launch_config()


def _build_gpu_plan(config: RuntimeConfig) -> ResolvedGpuPlan:
    role_classes = config.runtime.ray.gpu_manager.role_classes
    counts = config.runtime.ray.gpu_manager.topology
    rollout_tp = config.parallel.rollout.tensor_parallel_size

    physical_gpu_ids = tuple(range(config.runtime.local.num_gpus))
    rollout_only_gpu_ids = tuple(physical_gpu_ids[: counts.rollout_only_gpus])
    shared_start = counts.rollout_only_gpus
    shared_stop = shared_start + counts.shared_gpus
    shared_gpu_ids = tuple(physical_gpu_ids[shared_start:shared_stop])
    idle_gpu_ids = tuple(physical_gpu_ids[shared_stop:])

    rollout_gpu_ids = rollout_only_gpu_ids + shared_gpu_ids
    rollout_replicas: list[RolloutReplicaSpec] = []
    rollout_workers: list[RolloutWorkerSpec] = []

    for dp_rank, offset in enumerate(range(0, len(rollout_gpu_ids), rollout_tp)):
        gpu_ids = tuple(rollout_gpu_ids[offset : offset + rollout_tp])
        topology = GpuTopology.SHARED if all(gpu_id in shared_gpu_ids for gpu_id in gpu_ids) else GpuTopology.ROLLOUT_ONLY
        replica_id = f"rollout-dp-{dp_rank}"
        worker_ids: list[str] = []
        for tp_rank, gpu_id in enumerate(gpu_ids):
            worker_id = f"{replica_id}-tp-{tp_rank}"
            worker_ids.append(worker_id)
            rollout_workers.append(
                RolloutWorkerSpec(
                    worker_id=worker_id,
                    replica_id=replica_id,
                    dp_rank=dp_rank,
                    tp_rank=tp_rank,
                    gpu_id=gpu_id,
                    topology=topology,
                    worker_class=role_classes.rollout_worker,
                    ray_resource=f"rollout_gpu_{gpu_id}",
                )
            )
        rollout_replicas.append(
            RolloutReplicaSpec(
                replica_id=replica_id,
                dp_rank=dp_rank,
                topology=topology,
                gpu_ids=gpu_ids,
                worker_ids=tuple(worker_ids),
                controller_class=role_classes.rollout_replica_controller,
                worker_class=role_classes.rollout_worker,
            )
        )

    trainer_ranks = tuple(
        TrainerRankSpec(
            rank=rank,
            gpu_id=gpu_id,
            trainer_class=role_classes.trainer_rank or "",
            ray_resource=f"train_gpu_{gpu_id}",
        )
        for rank, gpu_id in enumerate(shared_gpu_ids)
    )

    return ResolvedGpuPlan(
        physical_gpu_ids=physical_gpu_ids,
        rollout_only_gpu_ids=rollout_only_gpu_ids,
        shared_gpu_ids=shared_gpu_ids,
        idle_gpu_ids=idle_gpu_ids,
        rollout_replicas=tuple(rollout_replicas),
        rollout_workers=tuple(rollout_workers),
        trainer_ranks=trainer_ranks,
    )


def _validate_source_uri(field: str, source_type: SourceType, uri: str) -> None:
    if source_type == SourceType.HDFS_URI:
        if not uri.startswith("hdfs://"):
            raise ValueError(f"{field} must start with hdfs:// when source_type=hdfs_uri")
    elif source_type == SourceType.HDFS_FUSE_PATH:
        if not uri.startswith("/"):
            raise ValueError(f"{field} must be absolute when source_type=hdfs_fuse_path")
    else:
        raise ValueError(f"unsupported source_type for {field}: {source_type}")


def _validate_import_path(field: str, path: str) -> None:
    module_name, sep, attr = path.rpartition(".")
    if not sep or not module_name or not attr:
        raise ValueError(f"{field} must be a dotted import path")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"{field} module cannot be imported: {module_name}") from exc
    if not hasattr(module, attr):
        raise ValueError(f"{field} class does not exist: {path}")


def launch_config_to_dict(config: LaunchConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")
