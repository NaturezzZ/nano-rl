"""CPU-only rollout backend for deterministic runtime tests.

The mock backend implements the same activation and generation surface as the
vLLM rollout backend without importing vLLM or touching CUDA.  It still enforces
the rollout lease contract so controller/actor tests exercise the same ownership
rules before any real backend is wired in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nano_rl.exceptions import SlotStateError
from nano_rl.runtime.backends.vllm_backend import GenerationOutput, VllmBackendConfig
from nano_rl.runtime.protocols import WeightMeta, WeightShardSource
from nano_rl.runtime.slot import GpuLease, RoleName


class MockRolloutBackendError(RuntimeError):
    """Raised for invalid mock rollout backend state transitions."""


class MockRolloutBackendConfig(BaseModel):
    """Configuration for a deterministic CPU-only rollout backend."""

    model_config = ConfigDict(frozen=True)

    tensor_parallel_size: int = Field(default=1, ge=1)
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    gpu_ids: tuple[int, ...] = ()
    holder_id: str | None = None
    holder_ids: tuple[str, ...] = ()
    seed: int = Field(default=0, ge=0)
    response_template: str | None = None
    tokenization: str = "sha256_bytes"
    max_response_tokens: int = Field(default=16, ge=1)
    logprob_mode: str = "linear"
    finish_reason: str = "stop"
    include_policy_segments: bool = False
    response_prefix: str = "mock"
    default_max_tokens: int = Field(default=8, ge=1)

    @classmethod
    def from_rollout_backend_config(cls, config: Mapping[str, Any] | None) -> "MockRolloutBackendConfig":
        raw = dict(config or {})
        nested = raw.pop("mock", None)
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise TypeError("rollout mock config must be a mapping")
            raw.update(nested)
        return cls.model_validate(raw)

    @model_validator(mode="after")
    def _validate_gpu_binding(self) -> "MockRolloutBackendConfig":
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("gpu_ids must be unique")
        if any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise ValueError("gpu_ids must be non-negative")
        if self.gpu_ids and len(self.gpu_ids) != self.tensor_parallel_size:
            raise ValueError(
                "tensor_parallel_size must match gpu_ids length when explicit gpu_ids are configured"
            )
        if self.gpu_ids and self.holder_ids and len(self.holder_ids) != len(self.gpu_ids):
            raise ValueError("holder_ids length must match gpu_ids length")
        return self

    @property
    def holders_by_gpu(self) -> dict[int, str]:
        if not self.gpu_ids or not self.holder_ids:
            return {}
        return dict(zip(self.gpu_ids, self.holder_ids, strict=True))


class MockRolloutBackend:
    """Deterministic CPU-only implementation of the rollout backend protocol."""

    def __init__(self, config: MockRolloutBackendConfig | VllmBackendConfig | Mapping[str, Any] | None = None) -> None:
        self.config = _resolve_config(config)
        self._active_weight: WeightMeta | None = None
        self._active_weight_source: WeightShardSource | None = None
        self._activation_count = 0

    @property
    def active_weight(self) -> WeightMeta | None:
        return self._active_weight

    @property
    def active_weight_source(self) -> WeightShardSource | None:
        return self._active_weight_source

    @property
    def activation_count(self) -> int:
        return self._activation_count

    def activate_weight(
        self,
        meta: WeightMeta,
        *,
        lease: GpuLease | Sequence[GpuLease],
        transfer_source: WeightShardSource | None = None,
    ) -> None:
        self._assert_rollout_lease(lease)
        self._active_weight = meta
        self._active_weight_source = transfer_source
        self._activation_count += 1

    def generate(
        self,
        *,
        prompt: str,
        target_policy_version: int,
        request_metadata: Mapping[str, Any] | None = None,
        lease: GpuLease | Sequence[GpuLease],
        request_id: str | None = None,
    ) -> GenerationOutput:
        leases = self._assert_rollout_lease(lease)
        if self._active_weight is None:
            raise MockRolloutBackendError("cannot generate before activate_weight")
        if self._active_weight.version_id != target_policy_version:
            raise MockRolloutBackendError(
                f"active weight version {self._active_weight.version_id} does not match target {target_policy_version}"
            )

        metadata = dict(request_metadata or {})
        seed_payload = _stable_payload(
            {
                "seed": self.config.seed,
                "prompt": prompt,
                "request_id": request_id,
                "request_metadata": metadata,
                "weight_version": self._active_weight.version_id,
                "weight_checksum": self._active_weight.checksum,
                "sampling_params": self.config.sampling_params,
            }
        )
        digest = hashlib.sha256(seed_payload).hexdigest()
        if self.config.response_template is None:
            response = f"{self.config.response_prefix}[v{self._active_weight.version_id}:{digest[:12]}] {prompt}"
        else:
            response = _render_response(
                self.config.response_template,
                prompt=prompt,
                policy_version=target_policy_version,
                active_weight=self._active_weight,
                request_id=request_id,
                digest=digest,
            )
        tokens = _deterministic_tokens(seed_payload, self._max_tokens())
        logprobs = _deterministic_logprobs(tokens)
        metadata["backend"] = "mock"
        metadata["mock_rollout"] = {
            "backend": "mock_rollout",
            "active_weight_version": self._active_weight.version_id,
            "activation_count": self._activation_count,
            "gpu_ids": [item.gpu_id for item in leases],
            "holder_ids": [item.holder_id for item in leases],
            "lease_epochs": {str(item.gpu_id): item.lease_epoch for item in leases},
            "digest": digest,
            "tokenization": self.config.tokenization,
            "logprob_mode": self.config.logprob_mode,
        }
        if self.config.include_policy_segments:
            metadata["policy_segments"] = [
                {
                    "start_token": 0,
                    "end_token": len(tokens),
                    "policy_version": target_policy_version,
                    "weight_checksum": self._active_weight.checksum,
                }
            ]

        return GenerationOutput(
            request_id=request_id or f"mock-{digest[:16]}",
            prompt=prompt,
            response=response,
            tokens=tokens,
            logprobs=logprobs,
            old_logprobs=list(logprobs),
            finish_reason=self.config.finish_reason,
            policy_version=target_policy_version,
            weight_checksum=self._active_weight.checksum,
            metadata=metadata,
        )

    def _max_tokens(self) -> int:
        raw_value = self.config.sampling_params.get("max_tokens", self.config.max_response_tokens)
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise MockRolloutBackendError(f"mock rollout max_tokens must be an integer, got {raw_value!r}") from exc
        if value < 1:
            raise MockRolloutBackendError("mock rollout max_tokens must be >= 1")
        return value

    def _assert_rollout_lease(self, lease: GpuLease | Sequence[GpuLease]) -> tuple[GpuLease, ...]:
        leases = _lease_sequence(lease)
        actual_gpus = {item.gpu_id for item in leases}
        if len(actual_gpus) != len(leases):
            raise SlotStateError("mock rollout backend received duplicate GPU leases")

        expected_gpus = set(self.config.gpu_ids)
        if expected_gpus and actual_gpus != expected_gpus:
            raise SlotStateError(
                f"mock rollout backend expected gpu leases {sorted(expected_gpus)}, "
                f"got {sorted(actual_gpus)}"
            )

        for item in leases:
            if item.role != RoleName.ROLLOUT:
                raise SlotStateError(f"mock rollout backend lease role is {item.role}, not rollout")

        expected_holders_by_gpu = self.config.holders_by_gpu
        if expected_holders_by_gpu:
            for item in leases:
                expected_holder = expected_holders_by_gpu[item.gpu_id]
                if item.holder_id != expected_holder:
                    raise SlotStateError(
                        f"mock rollout backend lease holder for gpu {item.gpu_id} is {item.holder_id}, "
                        f"expected {expected_holder}"
                    )
            return leases

        expected_holders = set(self.config.holder_ids)
        if expected_holders and {item.holder_id for item in leases} != expected_holders:
            raise SlotStateError(
                f"mock rollout backend expected lease holders {sorted(expected_holders)}, "
                f"got {sorted(item.holder_id for item in leases)}"
            )
        if self.config.holder_id is not None:
            for item in leases:
                if item.holder_id != self.config.holder_id:
                    raise SlotStateError(
                        f"mock rollout backend lease holder is {item.holder_id}, expected {self.config.holder_id}"
                    )
        return leases


def build_mock_rollout_backend(
    config: MockRolloutBackendConfig | VllmBackendConfig | Mapping[str, Any] | None = None,
) -> MockRolloutBackend:
    """Factory hook for tests and future mock backend selection."""

    return MockRolloutBackend(config)


def _resolve_config(
    config: MockRolloutBackendConfig | VllmBackendConfig | Mapping[str, Any] | None,
) -> MockRolloutBackendConfig:
    if isinstance(config, MockRolloutBackendConfig):
        return config
    if isinstance(config, VllmBackendConfig):
        return MockRolloutBackendConfig.model_validate(config.model_dump())
    return MockRolloutBackendConfig.from_rollout_backend_config(config or {})


def _lease_sequence(lease: GpuLease | Sequence[GpuLease]) -> tuple[GpuLease, ...]:
    if isinstance(lease, GpuLease):
        return (lease,)
    leases = tuple(lease)
    if not leases:
        raise SlotStateError("mock rollout backend requires at least one GPU lease")
    return leases


def _stable_payload(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _render_response(
    template: str | None,
    *,
    prompt: str,
    policy_version: int,
    active_weight: WeightMeta,
    request_id: str | None,
    digest: str,
) -> str:
    if template is None:
        return f"mock[v{active_weight.version_id}:{digest[:12]}] {prompt}"
    try:
        return template.format(
            prompt=prompt,
            policy_version=policy_version,
            target_policy_version=policy_version,
            version_id=active_weight.version_id,
            weight_checksum=active_weight.checksum,
            trainer_step=active_weight.trainer_step,
            request_id=request_id or "",
            digest=digest,
        )
    except (IndexError, KeyError) as exc:
        raise MockRolloutBackendError(f"invalid mock rollout response_template: {template}") from exc


def _deterministic_tokens(seed_payload: bytes, count: int) -> list[int]:
    tokens: list[int] = []
    block = seed_payload
    while len(tokens) < count:
        digest = hashlib.sha256(block).digest()
        for index in range(0, len(digest), 2):
            if len(tokens) >= count:
                break
            tokens.append(int.from_bytes(digest[index : index + 2], "big") % 32000)
        block = digest
    return tokens


def _deterministic_logprobs(tokens: Sequence[int]) -> list[float]:
    return [-round(((token % 1000) + 1) / 1000, 3) for token in tokens]
