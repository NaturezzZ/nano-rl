"""vLLM rollout backend boundary.

The Ray role wrappers own placement and lease acquisition.  This module keeps
the CUDA-facing rollout backend surface small enough for worker and replica
actors to call later while remaining importable on machines without vLLM.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable
import importlib
import os
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nano_rl.exceptions import SlotStateError
from nano_rl.runtime.protocols import SampleRecord, WeightMeta
from nano_rl.runtime.slot import GpuLease, RoleName


EngineFactory = Callable[["VllmBackendConfig", WeightMeta], Any]


class VllmBackendError(RuntimeError):
    """Base class for rollout backend failures."""


class VllmBackendUnavailable(VllmBackendError):
    """Raised when real vLLM is requested but not importable."""

    def __init__(self, *, package_name: str, action: str, original_error: BaseException):
        self.package_name = package_name
        self.action = action
        self.original_error = original_error
        super().__init__(
            f"{package_name} is required to {action}; install vLLM or inject a fake engine_factory for tests"
        )


class GenerationOutput(BaseModel):
    """Normalized rollout generation payload passed back to sample assembly."""

    request_id: str | None = None
    prompt: str
    response: str
    tokens: list[int]
    logprobs: list[float]
    old_logprobs: list[float]
    finish_reason: str
    policy_version: int = Field(ge=0)
    weight_checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VllmBackendConfig(BaseModel):
    """Configuration needed to construct one vLLM rollout engine lazily."""

    model_config = ConfigDict(frozen=True)

    model_path: str | None = None
    tokenizer_path: str | None = None
    tensor_parallel_size: int = Field(default=1, ge=1)
    dtype: str | None = None
    max_model_len: int | None = Field(default=None, ge=1)
    trust_remote_code: bool = False
    engine_kwargs: dict[str, Any] = Field(default_factory=dict)
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    gpu_ids: tuple[int, ...] = ()
    holder_id: str | None = None
    holder_ids: tuple[str, ...] = ()
    set_cuda_visible_devices: bool = True

    @model_validator(mode="after")
    def _validate_gpu_binding(self) -> "VllmBackendConfig":
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
    def cuda_visible_devices(self) -> str | None:
        if not self.gpu_ids:
            return None
        return ",".join(str(gpu_id) for gpu_id in self.gpu_ids)

    @property
    def holders_by_gpu(self) -> dict[int, str]:
        if not self.gpu_ids or not self.holder_ids:
            return {}
        return dict(zip(self.gpu_ids, self.holder_ids, strict=True))

    def apply_cuda_visible_devices(self) -> None:
        value = self.cuda_visible_devices
        if self.set_cuda_visible_devices and value is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = value


@runtime_checkable
class RolloutBackend(Protocol):
    """Backend surface consumed by future rollout worker/controller actors."""

    @property
    def active_weight(self) -> WeightMeta | None:
        """Currently loaded policy weights, if any."""

    def activate_weight(self, meta: WeightMeta, *, lease: GpuLease | Sequence[GpuLease]) -> None:
        """Load or switch the engine to a weight version under a rollout lease."""

    def generate(
        self,
        *,
        prompt: str,
        target_policy_version: int,
        request_metadata: Mapping[str, Any] | None = None,
        lease: GpuLease | Sequence[GpuLease],
        request_id: str | None = None,
    ) -> GenerationOutput:
        """Generate one completion under a rollout lease."""


class VllmRolloutBackend:
    """Rollout backend adapter around a lazily constructed vLLM engine."""

    def __init__(self, config: VllmBackendConfig, *, engine_factory: EngineFactory | None = None) -> None:
        self.config = config
        self._engine_factory = engine_factory or _default_engine_factory
        self._engine: Any | None = None
        self._active_weight: WeightMeta | None = None

    @property
    def active_weight(self) -> WeightMeta | None:
        return self._active_weight

    @property
    def engine(self) -> Any | None:
        return self._engine

    def activate_weight(self, meta: WeightMeta, *, lease: GpuLease | Sequence[GpuLease]) -> None:
        self._assert_rollout_lease(lease)
        if self._engine is None:
            config = self._config_for_weight(meta)
            config.apply_cuda_visible_devices()
            self._engine = self._engine_factory(config, meta)
        _notify_engine_weight(self._engine, meta)
        self._active_weight = meta

    def generate(
        self,
        *,
        prompt: str,
        target_policy_version: int,
        request_metadata: Mapping[str, Any] | None = None,
        lease: GpuLease | Sequence[GpuLease],
        request_id: str | None = None,
    ) -> GenerationOutput:
        self._assert_rollout_lease(lease)
        if self._active_weight is None:
            raise VllmBackendError("cannot generate before activate_weight")
        if self._active_weight.version_id != target_policy_version:
            raise VllmBackendError(
                f"active weight version {self._active_weight.version_id} does not match target {target_policy_version}"
            )

        raw_output = _call_generate(
            self._require_engine(),
            prompt=prompt,
            sampling_params=self.config.sampling_params,
            request_metadata=request_metadata or {},
        )
        return _normalize_generation_output(
            raw_output,
            prompt=prompt,
            target_policy_version=target_policy_version,
            weight_checksum=self._active_weight.checksum,
            request_id=request_id,
            request_metadata=request_metadata or {},
        )

    def _config_for_weight(self, meta: WeightMeta) -> VllmBackendConfig:
        return self.config.model_copy(
            update={
                "model_path": self.config.model_path or meta.model_path,
                "tokenizer_path": self.config.tokenizer_path or meta.tokenizer_path,
            }
        )

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise VllmBackendError("vLLM engine has not been initialized")
        return self._engine

    def _assert_rollout_lease(self, lease: GpuLease | Sequence[GpuLease]) -> None:
        leases = _lease_sequence(lease)
        actual_gpus = {item.gpu_id for item in leases}
        if len(actual_gpus) != len(leases):
            raise SlotStateError("rollout backend received duplicate GPU leases")
        expected_gpus = set(self.config.gpu_ids)
        if expected_gpus and actual_gpus != expected_gpus:
            raise SlotStateError(
                f"rollout backend expected gpu leases {sorted(expected_gpus)}, "
                f"got {sorted(actual_gpus)}"
            )

        for item in leases:
            if item.role != RoleName.ROLLOUT:
                raise SlotStateError(f"rollout backend lease role is {item.role}, not rollout")

        expected_holders_by_gpu = self.config.holders_by_gpu
        if expected_holders_by_gpu:
            for item in leases:
                expected_holder = expected_holders_by_gpu[item.gpu_id]
                if item.holder_id != expected_holder:
                    raise SlotStateError(
                        f"rollout backend lease holder for gpu {item.gpu_id} is {item.holder_id}, "
                        f"expected {expected_holder}"
                    )
            return

        expected_holders = set(self.config.holder_ids)
        if expected_holders and {item.holder_id for item in leases} != expected_holders:
            raise SlotStateError(
                f"rollout backend expected lease holders {sorted(expected_holders)}, "
                f"got {sorted(item.holder_id for item in leases)}"
            )
        if self.config.holder_id is not None:
            for item in leases:
                if item.holder_id != self.config.holder_id:
                    raise SlotStateError(
                        f"rollout backend lease holder is {item.holder_id}, expected {self.config.holder_id}"
                    )


def build_rollout_backend(
    config: VllmBackendConfig | Mapping[str, Any] | None = None,
    *,
    engine_factory: EngineFactory | None = None,
) -> VllmRolloutBackend:
    """Factory hook for future Ray role actors."""

    resolved = config if isinstance(config, VllmBackendConfig) else VllmBackendConfig.model_validate(config or {})
    return VllmRolloutBackend(resolved, engine_factory=engine_factory)


def _lease_sequence(lease: GpuLease | Sequence[GpuLease]) -> tuple[GpuLease, ...]:
    if isinstance(lease, GpuLease):
        return (lease,)
    leases = tuple(lease)
    if not leases:
        raise SlotStateError("rollout backend requires at least one GPU lease")
    return leases


def _default_engine_factory(config: VllmBackendConfig, _meta: WeightMeta) -> Any:
    try:
        vllm = importlib.import_module("vllm")
    except ImportError as exc:
        raise VllmBackendUnavailable(package_name="vllm", action="construct rollout backend", original_error=exc) from exc

    kwargs: dict[str, Any] = {
        "model": config.model_path,
        "tokenizer": config.tokenizer_path or config.model_path,
        "tensor_parallel_size": config.tensor_parallel_size,
        "trust_remote_code": config.trust_remote_code,
        **config.engine_kwargs,
    }
    if config.dtype is not None:
        kwargs["dtype"] = config.dtype
    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len
    return vllm.LLM(**kwargs)


def _notify_engine_weight(engine: Any, meta: WeightMeta) -> None:
    for method_name in ("activate_weight", "load_weight", "load_weights"):
        method = getattr(engine, method_name, None)
        if method is not None:
            method(meta)
            return


def _call_generate(
    engine: Any,
    *,
    prompt: str,
    sampling_params: Mapping[str, Any],
    request_metadata: Mapping[str, Any],
) -> Any:
    params = _build_sampling_params(sampling_params)
    try:
        return engine.generate([prompt], sampling_params=params, use_tqdm=False, metadata=dict(request_metadata))
    except TypeError:
        try:
            return engine.generate([prompt], sampling_params=params, use_tqdm=False)
        except TypeError:
            return engine.generate(prompt, sampling_params=params)


def _build_sampling_params(raw_params: Mapping[str, Any]) -> Any:
    if not raw_params:
        return None
    try:
        sampling_params_cls = getattr(importlib.import_module("vllm"), "SamplingParams")
    except ImportError:
        return dict(raw_params)
    return sampling_params_cls(**dict(raw_params))


def _normalize_generation_output(
    raw_output: Any,
    *,
    prompt: str,
    target_policy_version: int,
    weight_checksum: str,
    request_id: str | None,
    request_metadata: Mapping[str, Any],
) -> GenerationOutput:
    request_output = _first_request_output(raw_output)
    completion = _first_completion(request_output)
    resolved_request_id = request_id or _optional_str(_read_attr(request_output, "request_id", default=None))
    response = str(_read_attr(completion, "text", default=""))
    tokens = _normalize_token_ids(_read_attr(completion, "token_ids", default=[]))
    logprobs = _normalize_logprobs(_read_attr(completion, "logprobs", default=[]), tokens)
    finish_reason = str(_read_attr(completion, "finish_reason", default="unknown") or "unknown")
    return GenerationOutput(
        request_id=resolved_request_id,
        prompt=prompt,
        response=response,
        tokens=tokens,
        logprobs=logprobs,
        old_logprobs=list(logprobs),
        finish_reason=finish_reason,
        policy_version=target_policy_version,
        weight_checksum=weight_checksum,
        metadata=dict(request_metadata),
    )


def _first_request_output(raw_output: Any) -> Any:
    if isinstance(raw_output, Sequence) and not isinstance(raw_output, (str, bytes)):
        if not raw_output:
            raise VllmBackendError("vLLM generate returned no request outputs")
        return raw_output[0]
    if raw_output is None:
        raise VllmBackendError("vLLM generate returned no request outputs")
    return raw_output


def _first_completion(request_output: Any) -> Any:
    outputs = _read_attr(request_output, "outputs", default=None)
    if outputs is not None:
        if not outputs:
            raise VllmBackendError("vLLM generate returned a request output without completions")
        return outputs[0]
    return request_output


def _read_attr(value: Any, name: str, *, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_token_ids(raw_tokens: Any) -> list[int]:
    if raw_tokens is None:
        return []
    return [int(token) for token in raw_tokens]


def _normalize_logprobs(raw_logprobs: Any, token_ids: Sequence[int]) -> list[float]:
    if raw_logprobs is None:
        return []
    normalized: list[float] = []
    for index, item in enumerate(raw_logprobs):
        if isinstance(item, Mapping):
            token_id = token_ids[index] if index < len(token_ids) else None
            selected = item.get(token_id) if token_id is not None else None
            if selected is None and item:
                selected = next(iter(item.values()))
            normalized.append(_logprob_value(selected))
        else:
            normalized.append(_logprob_value(item))
    return normalized


def _logprob_value(item: Any) -> float:
    if item is None:
        return 0.0
    if isinstance(item, Mapping):
        return float(item.get("logprob", 0.0))
    return float(getattr(item, "logprob", item))


def generation_output_to_sample_record(
    output: GenerationOutput,
    *,
    reward: float,
    reward_source: str,
    request_metadata: Mapping[str, Any] | None = None,
) -> SampleRecord:
    """Convert normalized vLLM output into the runtime sample protocol."""

    metadata = dict(request_metadata or {})
    metadata.update(output.metadata)
    request_id = output.request_id or _optional_str(metadata.get("request_id"))
    sample_id = _optional_str(metadata.get("sample_id")) or request_id or f"sample-{uuid4().hex}"
    return SampleRecord(
        sample_id=sample_id,
        request_id=request_id,
        policy_version=output.policy_version,
        weight_checksum=output.weight_checksum,
        prompt=output.prompt,
        response=output.response,
        finish_reason=output.finish_reason,
        tokens=list(output.tokens),
        logprobs=list(output.logprobs),
        old_logprobs=list(output.old_logprobs),
        reward=float(reward),
        reward_source=reward_source,
        prompt_tokens=int(metadata.get("prompt_tokens", 0)) if metadata.get("prompt_tokens") is not None else None,
        response_tokens=len(output.tokens),
        meta=metadata,
    )
