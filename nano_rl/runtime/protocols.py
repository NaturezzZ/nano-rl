"""Structured runtime protocol models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class WeightStatus(StrEnum):
    REGISTERED = "registered"
    ACTIVATING = "activating"
    ACTIVE_GLOBAL = "active_global"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class WeightFormat(StrEnum):
    HF = "hf"
    SAFETENSORS = "safetensors"
    VLLM_COMPATIBLE = "vllm_compatible"


class WeightTransferMethod(StrEnum):
    OBJECT_REF = "objectref"
    LOCALITY_AWARE_CHECKPOINT = "locality_aware_checkpoint"


class WeightShardSourceKind(StrEnum):
    RAY_OBJECT_REF = "ray_object_ref"
    SHARED_GPU_RESHARD = "shared_gpu_reshard"
    ARTIFACT_PULL = "artifact_pull"


class WeightMeta(BaseModel):
    version_id: int = Field(ge=0)
    created_at: datetime
    model_path: str
    format: WeightFormat
    checksum: str
    parent_version: int | None = Field(default=None, ge=0)
    trainer_step: int | None = Field(default=None, ge=0)
    tokenizer_path: str | None = None
    artifact_uri: str | None = None
    manifest_uri: str | None = None
    tokenizer_hash: str | None = None
    chat_template_hash: str | None = None
    created_by: str | None = None
    status: WeightStatus = WeightStatus.REGISTERED


class WeightShardSource(BaseModel):
    replica_id: str
    kind: WeightShardSourceKind
    target_gpu_ids: tuple[int, ...] = ()
    target_worker_ids: tuple[str, ...] = ()
    source_rank_ids: tuple[int, ...] = ()
    source_gpu_ids: tuple[int, ...] = ()
    artifact_uri: str | None = None
    manifest_uri: str | None = None
    object_ref_key: str | None = None
    reason: str


class WeightTransferPlan(BaseModel):
    version_id: int = Field(ge=0)
    method: WeightTransferMethod
    artifact_uri: str | None = None
    manifest_uri: str | None = None
    sources: dict[str, WeightShardSource]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def source_for_replica(self, replica_id: str) -> WeightShardSource:
        return self.sources[replica_id]


class SampleRecord(BaseModel):
    sample_id: str
    policy_version: int = Field(ge=0)
    prompt: str
    response: str
    tokens: list[int]
    logprobs: list[float]
    reward: float
    request_id: str | None = None
    weight_checksum: str | None = None
    finish_reason: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    response_tokens: int | None = Field(default=None, ge=0)
    old_logprobs: list[float] | None = None
    reward_source: str | None = None
    advantages: list[float] | None = None
    drop_reason: str | None = None
    policy_segments: list["PolicySegment"] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def oldest_behavior_policy_version(self) -> int:
        if not self.policy_segments:
            return self.policy_version
        return min(segment.policy_version for segment in self.policy_segments)


class PolicySegment(BaseModel):
    start_token: int = Field(ge=0)
    end_token: int = Field(ge=0)
    policy_version: int = Field(ge=0)
    weight_checksum: str | None = None


class SampleRef(BaseModel):
    sample_id: str
    policy_version: int = Field(ge=0)
    created_at: datetime
    expires_at: datetime | None = None
    object_ref: Any = None
    num_tokens: int = Field(default=0, ge=0)

    model_config = {"arbitrary_types_allowed": True}


class TrainBatch(BaseModel):
    train_batch_id: str
    sample_refs: list[SampleRef]
    sample_ids: list[str]
    policy_version_min: int = Field(ge=0)
    policy_version_max: int = Field(ge=0)
    policy_version_histogram: dict[int, int]
    num_sequences: int = Field(ge=0)
    num_tokens: int = Field(ge=0)
    reserved_by: str
    reserved_at: datetime
    expires_at: datetime | None = None


class TrainStats(BaseModel):
    rank: int = Field(ge=0)
    group_epoch: int = Field(ge=0)
    train_step: int = Field(ge=0)
    train_batch_id: str
    num_sequences: int = Field(ge=0)
    num_tokens: int = Field(ge=0)
    loss: float


class EventSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class HealthEvent(BaseModel):
    event_id: str
    event_type: str
    severity: EventSeverity
    source_actor: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    gpu_id: int | None = None
    policy_version: int | None = None
    group_epoch: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)
