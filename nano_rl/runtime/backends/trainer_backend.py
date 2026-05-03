"""Trainer backend boundary shared by fake and real FSDP2 implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from hashlib import sha256
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
import os

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nano_rl.exceptions import NanoRLError, SlotStateError
from nano_rl.runtime.protocols import TrainBatch, WeightFormat, WeightMeta
from nano_rl.runtime.slot import GpuLease, RoleName


class BackendUnavailableError(NanoRLError):
    """Raised when a requested backend dependency or capability is unavailable."""


class BackendStateError(NanoRLError):
    """Raised when a backend method is called in an invalid lifecycle state."""


class TrainerProcessGroupMetadata(BaseModel):
    """Future process group metadata; construction does not initialize dist."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    group_epoch: int = Field(ge=0)
    rendezvous: str | None = None
    store_endpoint: str | None = None
    comm_epoch: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _rank_must_be_in_world(self) -> "TrainerProcessGroupMetadata":
        if self.rank >= self.world_size:
            raise ValueError(f"rank {self.rank} must be smaller than world_size {self.world_size}")
        return self


class TrainerBackendConfig(BaseModel):
    """Configuration needed by one trainer rank backend instance."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["fsdp2", "fake"] = "fsdp2"
    rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    gpu_id: int = Field(ge=0)
    holder_id: str | None = None
    group_epoch: int = Field(default=0, ge=0)
    rendezvous: str | None = None
    store_endpoint: str | None = None
    comm_epoch: int = Field(default=0, ge=0)
    model_path: str | None = None
    optimizer_name: str = "adamw"
    checkpoint_dir: str | None = None
    set_cuda_visible_devices: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _rank_must_be_in_world(self) -> "TrainerBackendConfig":
        if self.rank >= self.world_size:
            raise ValueError(f"rank {self.rank} must be smaller than world_size {self.world_size}")
        return self

    @property
    def expected_holder_id(self) -> str:
        return self.holder_id or f"trainer-rank-{self.rank}"

    @property
    def process_group(self) -> TrainerProcessGroupMetadata:
        return TrainerProcessGroupMetadata(
            rank=self.rank,
            world_size=self.world_size,
            group_epoch=self.group_epoch,
            rendezvous=self.rendezvous,
            store_endpoint=self.store_endpoint,
            comm_epoch=self.comm_epoch,
        )


class TrainStateBundle(BaseModel):
    """Serializable description of one rank's CPU-standby train state."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    group_epoch: int = Field(ge=0)
    comm_epoch: int = Field(ge=0)
    train_step: int = Field(ge=0)
    weight_version: int | None = Field(default=None, ge=0)
    model_state_uri: str | None = None
    optimizer_state_uri: str | None = None
    scheduler_state_uri: str | None = None
    rng_state_uri: str | None = None
    scaler_state_uri: str | None = None
    residency: Literal["cpu_standby", "unloaded"] = "cpu_standby"
    metadata: dict[str, Any] = Field(default_factory=dict)


class OptimizerStepResult(BaseModel):
    """Result returned after a backend completes one optimizer update."""

    rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    group_epoch: int = Field(ge=0)
    comm_epoch: int = Field(ge=0)
    train_step: int = Field(ge=0)
    train_batch_id: str
    num_sequences: int = Field(ge=0)
    num_tokens: int = Field(ge=0)
    loss: float
    lease_gpu_id: int = Field(ge=0)
    lease_epoch: int = Field(ge=0)
    weight_version: int | None = Field(default=None, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)


class TrainerBackend(ABC):
    """Interface implemented by trainer rank execution backends."""

    def __init__(self, config: TrainerBackendConfig):
        self.config = config

    @abstractmethod
    def initialize_rank(self) -> TrainStateBundle:
        """Initialize rank-local backend metadata without consuming a batch."""

    @abstractmethod
    def hydrate(self, state: TrainStateBundle | None = None, *, lease: GpuLease) -> TrainStateBundle:
        """Move or rebuild train state into GPU-active residency after lease validation."""

    @abstractmethod
    def optimize(self, batch: TrainBatch, *, lease: GpuLease) -> OptimizerStepResult:
        """Run one optimizer update under a valid trainer GPU lease."""

    @abstractmethod
    def export_weight(self, parent: WeightMeta) -> WeightMeta:
        """Export a backend-readable weight artifact; only rank 0 may publish."""

    @abstractmethod
    def offload(self, *, lease: GpuLease) -> TrainStateBundle:
        """Move model/optimizer state out of GPU residency after lease validation."""

    def _assert_trainer_lease(self, lease: GpuLease) -> None:
        if lease.role != RoleName.TRAINER:
            raise SlotStateError(f"trainer rank {self.config.rank} lease role is {lease.role}, not trainer")
        if lease.gpu_id != self.config.gpu_id:
            raise SlotStateError(
                f"trainer rank {self.config.rank} lease gpu is {lease.gpu_id}, expected {self.config.gpu_id}"
            )
        if lease.holder_id != self.config.expected_holder_id:
            raise SlotStateError(
                f"trainer rank {self.config.rank} lease holder is {lease.holder_id}, "
                f"expected {self.config.expected_holder_id}"
            )

    def _set_cuda_visible_devices(self) -> None:
        if self.config.set_cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)


class FakeTrainerBackend(TrainerBackend):
    """Deterministic trainer backend for CPU-only tests and local smoke paths."""

    def __init__(self, config: TrainerBackendConfig):
        super().__init__(config)
        self._initialized = False
        self._gpu_resident = False
        self._train_step = 0
        self._weight_version: int | None = None

    def initialize_rank(self) -> TrainStateBundle:
        self._initialized = True
        return self._state_bundle(residency="unloaded")

    def hydrate(self, state: TrainStateBundle | None = None, *, lease: GpuLease) -> TrainStateBundle:
        self._assert_trainer_lease(lease)
        if not self._initialized:
            self.initialize_rank()
        if state is not None:
            self._assert_state_matches_rank(state)
            self._train_step = state.train_step
            self._weight_version = state.weight_version
        self._gpu_resident = True
        return self._state_bundle(residency="cpu_standby")

    def optimize(self, batch: TrainBatch, *, lease: GpuLease) -> OptimizerStepResult:
        self._assert_trainer_lease(lease)
        if not self._initialized:
            self.initialize_rank()
        self._gpu_resident = True
        self._train_step += 1
        loss = 1.0 / max(1, batch.num_sequences + self.config.rank + self._train_step)
        return OptimizerStepResult(
            rank=self.config.rank,
            world_size=self.config.world_size,
            group_epoch=self.config.group_epoch,
            comm_epoch=self.config.comm_epoch,
            train_step=self._train_step,
            train_batch_id=batch.train_batch_id,
            num_sequences=batch.num_sequences,
            num_tokens=batch.num_tokens,
            loss=loss,
            lease_gpu_id=lease.gpu_id,
            lease_epoch=lease.lease_epoch,
            weight_version=self._weight_version,
            metrics={"fake_loss": loss},
        )

    def export_weight(self, parent: WeightMeta) -> WeightMeta:
        if self.config.rank != 0:
            raise BackendStateError(f"trainer rank {self.config.rank} cannot export weights; rank 0 owns publish")
        version_id = parent.version_id + 1
        checksum = sha256(f"{parent.checksum}:{version_id}:{self._train_step}".encode()).hexdigest()
        self._weight_version = version_id
        return WeightMeta(
            version_id=version_id,
            parent_version=parent.version_id,
            trainer_step=self._train_step,
            created_at=datetime.utcnow(),
            model_path=parent.model_path,
            tokenizer_path=parent.tokenizer_path,
            artifact_uri=parent.artifact_uri,
            manifest_uri=parent.manifest_uri,
            format=WeightFormat.VLLM_COMPATIBLE,
            checksum=checksum,
            created_by=f"trainer-rank-{self.config.rank}",
        )

    def offload(self, *, lease: GpuLease) -> TrainStateBundle:
        self._assert_trainer_lease(lease)
        if not self._initialized:
            raise BackendStateError(f"trainer rank {self.config.rank} has not been initialized")
        self._gpu_resident = False
        return self._state_bundle(residency="cpu_standby")

    def _state_bundle(self, *, residency: Literal["cpu_standby", "unloaded"]) -> TrainStateBundle:
        return TrainStateBundle(
            rank=self.config.rank,
            world_size=self.config.world_size,
            group_epoch=self.config.group_epoch,
            comm_epoch=self.config.comm_epoch,
            train_step=self._train_step,
            weight_version=self._weight_version,
            residency=residency,
            metadata={
                "backend": "fake",
                "gpu_resident": self._gpu_resident,
                "process_group": self.config.process_group.model_dump(mode="json"),
            },
        )

    def _assert_state_matches_rank(self, state: TrainStateBundle) -> None:
        if state.rank != self.config.rank:
            raise BackendStateError(f"state rank {state.rank} does not match backend rank {self.config.rank}")
        if state.world_size != self.config.world_size:
            raise BackendStateError(
                f"state world_size {state.world_size} does not match backend world_size {self.config.world_size}"
            )
        if state.group_epoch != self.config.group_epoch:
            raise BackendStateError(
                f"state group_epoch {state.group_epoch} does not match backend group_epoch {self.config.group_epoch}"
            )


def build_trainer_backend(config: TrainerBackendConfig | Mapping[str, Any]) -> TrainerBackend:
    """Build a trainer backend from runtime config."""

    resolved = config if isinstance(config, TrainerBackendConfig) else TrainerBackendConfig.model_validate(config)
    if resolved.backend == "fake":
        return FakeTrainerBackend(resolved)
    if resolved.backend == "fsdp2":
        from nano_rl.runtime.backends.fsdp2_backend import Fsdp2TrainerBackend

        return Fsdp2TrainerBackend(resolved)
    raise BackendStateError(f"unsupported trainer backend: {resolved.backend}")


def checkpoint_version_dir(config: TrainerBackendConfig, version_id: int) -> Path:
    checkpoint_dir = config.checkpoint_dir or config.extra.get("checkpoint_dir")
    if not checkpoint_dir:
        raise BackendStateError("trainer backend requires checkpoint_dir to export weights")
    return Path(str(checkpoint_dir)).expanduser().resolve() / f"version-{version_id}"
