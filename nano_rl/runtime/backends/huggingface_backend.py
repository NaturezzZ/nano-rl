"""Hugging Face Transformers rollout backend boundary.

This backend is the lightweight, direct ``transformers`` path for local and
small-model rollout execution.  It intentionally implements the same rollout
protocol as the vLLM and mock adapters: Ray still owns placement, the lease
manager still gates CUDA work, and generated samples still carry policy and
weight-version metadata.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
import importlib
import logging
import os

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nano_rl.exceptions import SlotStateError
from nano_rl.runtime.backends.vllm_backend import GenerationOutput
from nano_rl.runtime.protocols import WeightMeta, WeightShardSource
from nano_rl.runtime.slot import GpuLease, RoleName


HfModelFactory = Callable[["HuggingFaceBackendConfig", WeightMeta, Mapping[str, Any]], Any]
HfTokenizerFactory = Callable[["HuggingFaceBackendConfig", WeightMeta, Mapping[str, Any]], Any]
logger = logging.getLogger(__name__)


class HuggingFaceBackendError(RuntimeError):
    """Base class for Hugging Face rollout backend failures."""


class HuggingFaceBackendUnavailable(HuggingFaceBackendError):
    """Raised when the direct Transformers backend cannot import dependencies."""

    def __init__(self, *, action: str, original_error: BaseException):
        self.package_name = "transformers"
        self.action = action
        self.original_error = original_error
        super().__init__(
            "transformers and torch are required to "
            f"{action}; install the Hugging Face stack or inject fake factories for tests"
        )


class HuggingFaceBackendConfig(BaseModel):
    """Configuration for one direct Transformers rollout replica."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    model_path: str | None = None
    tokenizer_path: str | None = None
    tensor_parallel_size: int = Field(default=1, ge=1)
    dtype: str | None = None
    device: str | None = None
    device_map: str | dict[str, Any] | None = None
    trust_remote_code: bool = False
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)
    skip_special_tokens: bool = True
    set_cuda_visible_devices: bool = True
    gpu_ids: tuple[int, ...] = ()
    holder_id: str | None = None
    holder_ids: tuple[str, ...] = ()
    offload_strategy: Literal["cpu", "teardown_and_reload", "vllm_sleep"] = "cpu"

    @classmethod
    def from_rollout_backend_config(cls, config: Mapping[str, Any] | None) -> "HuggingFaceBackendConfig":
        raw = dict(config or {})
        nested = raw.pop("huggingface", None)
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise TypeError("rollout huggingface config must be a mapping")
            raw.update(nested)
        return cls.model_validate(raw)

    @model_validator(mode="after")
    def _validate_gpu_binding(self) -> "HuggingFaceBackendConfig":
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


class HuggingFaceRolloutBackend:
    """Direct ``transformers.AutoModelForCausalLM.generate`` rollout adapter."""

    def __init__(
        self,
        config: HuggingFaceBackendConfig,
        *,
        model_factory: HfModelFactory | None = None,
        tokenizer_factory: HfTokenizerFactory | None = None,
    ) -> None:
        self.config = config
        self._model_factory = model_factory or _default_model_factory
        self._tokenizer_factory = tokenizer_factory or _default_tokenizer_factory
        self._requires_modules = model_factory is None or tokenizer_factory is None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._modules: dict[str, Any] | None = None
        self._active_weight: WeightMeta | None = None
        self._active_weight_source: WeightShardSource | None = None
        self._loaded_model_path: str | None = None
        self._loaded_tokenizer_path: str | None = None
        self._loaded_weight_checksum: str | None = None
        self._offloaded = False

    @property
    def active_weight(self) -> WeightMeta | None:
        return self._active_weight

    @property
    def active_weight_source(self) -> WeightShardSource | None:
        return self._active_weight_source

    @property
    def model(self) -> Any | None:
        return self._model

    @property
    def tokenizer(self) -> Any | None:
        return self._tokenizer

    def activate_weight(
        self,
        meta: WeightMeta,
        *,
        lease: GpuLease | Sequence[GpuLease],
        transfer_source: WeightShardSource | None = None,
    ) -> None:
        logger.info(
            "HuggingFace activate_weight started: version=%s gpu_ids=%s",
            meta.version_id,
            list(self.config.gpu_ids),
        )
        self._assert_rollout_lease(lease)
        model_path = self._model_path_for_weight(meta, transfer_source=transfer_source)
        tokenizer_path = self._tokenizer_path_for_weight(meta, model_path=model_path)
        if (
            self._model is None
            or self._loaded_model_path != model_path
            or self._loaded_weight_checksum != meta.checksum
        ):
            self._load_model_and_tokenizer(meta, model_path=model_path, tokenizer_path=tokenizer_path)
        elif self._tokenizer is None or self._loaded_tokenizer_path != tokenizer_path:
            self._tokenizer = self._tokenizer_factory(
                self._config_for_paths(model_path=model_path, tokenizer_path=tokenizer_path),
                meta,
                self._require_modules() if self._requires_modules else {},
            )
            self._loaded_tokenizer_path = tokenizer_path
        if self._offloaded:
            self.wake(lease=lease)
        self._active_weight = meta
        self._active_weight_source = transfer_source
        logger.info(
            "HuggingFace activate_weight completed: version=%s model_path=%s checksum=%s",
            meta.version_id,
            model_path,
            meta.checksum,
        )

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
            raise HuggingFaceBackendError("cannot generate before activate_weight")
        if self._offloaded:
            raise HuggingFaceBackendError("HuggingFace rollout backend is offloaded; wake it before generate")
        if self._active_weight.version_id != target_policy_version:
            raise HuggingFaceBackendError(
                f"active weight version {self._active_weight.version_id} does not match target {target_policy_version}"
            )

        tokenizer = self._require_tokenizer()
        model = self._require_model()
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = _move_encoded_to_device(encoded, self._device_for_generation())
        input_ids = _encoded_get(encoded, "input_ids")
        prompt_token_count = _sequence_length(input_ids)
        generation_kwargs = dict(self.config.generation_kwargs)
        generation_kwargs.setdefault("max_new_tokens", 16)
        generation_kwargs.setdefault("return_dict_in_generate", True)
        generation_kwargs.setdefault("output_scores", True)
        raw_output = model.generate(**_encoded_to_kwargs(encoded), **generation_kwargs)
        sequence = _first_sequence(_read_attr(raw_output, "sequences", default=raw_output))
        tokens = _tokens_after_prompt(sequence, prompt_token_count)
        response = str(tokenizer.decode(tokens, skip_special_tokens=self.config.skip_special_tokens))
        logprobs = _transition_logprobs(model, raw_output, tokens)
        finish_reason = _finish_reason(tokens, tokenizer, generation_kwargs)

        metadata = dict(request_metadata or {})
        metadata["backend"] = "huggingface"
        metadata["prompt_tokens"] = prompt_token_count
        metadata["huggingface_rollout"] = {
            "backend": "huggingface",
            "active_weight_version": self._active_weight.version_id,
            "model_path": self._loaded_model_path,
            "tokenizer_path": self._loaded_tokenizer_path,
            "gpu_ids": [item.gpu_id for item in leases],
            "holder_ids": [item.holder_id for item in leases],
            "lease_epochs": {str(item.gpu_id): item.lease_epoch for item in leases},
            "response_tokens": len(tokens),
            "device": self._device_for_generation(),
        }
        logger.info(
            "HuggingFace generate completed: request_id=%s policy_version=%s token_count=%s",
            request_id,
            target_policy_version,
            len(tokens),
        )
        return GenerationOutput(
            request_id=request_id,
            prompt=prompt,
            response=response,
            tokens=tokens,
            logprobs=logprobs,
            old_logprobs=list(logprobs),
            finish_reason=finish_reason,
            policy_version=target_policy_version,
            weight_checksum=self._active_weight.checksum,
            metadata=metadata,
        )

    def offload(self, *, lease: GpuLease | Sequence[GpuLease]) -> dict[str, object]:
        self._assert_rollout_lease(lease)
        action = "cpu_offload"
        if self.config.offload_strategy == "teardown_and_reload":
            self._model = None
            action = "teardown"
        elif self._model is not None:
            _move_model(self._model, "cpu")
            _empty_cuda_cache(self._modules)
        self._offloaded = True
        return self._residency_state("offloaded", action=action)

    def wake(self, *, lease: GpuLease | Sequence[GpuLease]) -> dict[str, object]:
        self._assert_rollout_lease(lease)
        action = "noop"
        if self._active_weight is None:
            self._offloaded = False
            return self._residency_state("active", action="noop_no_weight")
        model_path = self._model_path_for_weight(self._active_weight, transfer_source=self._active_weight_source)
        tokenizer_path = self._tokenizer_path_for_weight(self._active_weight, model_path=model_path)
        if self._model is None:
            self._load_model_and_tokenizer(self._active_weight, model_path=model_path, tokenizer_path=tokenizer_path)
            action = "reload"
        else:
            _move_model(self._model, self._device_for_generation())
            action = "move_to_device"
        self._offloaded = False
        return self._residency_state("active", action=action)

    def _load_model_and_tokenizer(self, meta: WeightMeta, *, model_path: str, tokenizer_path: str) -> None:
        self.config.apply_cuda_visible_devices()
        config = self._config_for_paths(model_path=model_path, tokenizer_path=tokenizer_path)
        modules = self._require_modules() if self._requires_modules else {}
        self._tokenizer = self._tokenizer_factory(config, meta, modules)
        self._model = self._model_factory(config, meta, modules)
        if config.device_map is None:
            _move_model(self._model, self._device_for_generation(modules=modules))
        self._loaded_model_path = model_path
        self._loaded_tokenizer_path = tokenizer_path
        self._loaded_weight_checksum = meta.checksum
        self._offloaded = False

    def _config_for_paths(self, *, model_path: str, tokenizer_path: str) -> HuggingFaceBackendConfig:
        return self.config.model_copy(update={"model_path": model_path, "tokenizer_path": tokenizer_path})

    def _model_path_for_weight(
        self,
        meta: WeightMeta,
        *,
        transfer_source: WeightShardSource | None,
    ) -> str:
        return (
            transfer_source.artifact_uri
            if transfer_source is not None and transfer_source.artifact_uri
            else meta.artifact_uri
            or meta.model_path
            or self.config.model_path
            or ""
        )

    def _tokenizer_path_for_weight(self, meta: WeightMeta, *, model_path: str) -> str:
        return meta.tokenizer_path or self.config.tokenizer_path or model_path

    def _require_modules(self) -> dict[str, Any]:
        if self._modules is None:
            try:
                torch = importlib.import_module("torch")
                transformers = importlib.import_module("transformers")
            except (ImportError, ModuleNotFoundError) as exc:
                raise HuggingFaceBackendUnavailable(
                    action="construct rollout backend",
                    original_error=exc,
                ) from exc
            self._modules = {"torch": torch, "transformers": transformers}
        return self._modules

    def _require_model(self) -> Any:
        if self._model is None:
            raise HuggingFaceBackendError("HuggingFace model has not been initialized")
        return self._model

    def _require_tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise HuggingFaceBackendError("HuggingFace tokenizer has not been initialized")
        return self._tokenizer

    def _device_for_generation(self, *, modules: Mapping[str, Any] | None = None) -> str:
        if self.config.device and self.config.device != "auto":
            return self.config.device
        modules = modules or self._modules
        torch = None if modules is None else modules.get("torch")
        if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _residency_state(self, residency: str, *, action: str) -> dict[str, object]:
        return {
            "backend": "huggingface",
            "residency": residency,
            "action": action,
            "active_weight_version": None if self._active_weight is None else self._active_weight.version_id,
            "has_model": self._model is not None,
            "offloaded": self._offloaded,
            "gpu_ids": list(self.config.gpu_ids),
        }

    def _assert_rollout_lease(self, lease: GpuLease | Sequence[GpuLease]) -> tuple[GpuLease, ...]:
        leases = _lease_sequence(lease)
        actual_gpus = {item.gpu_id for item in leases}
        if len(actual_gpus) != len(leases):
            raise SlotStateError("HuggingFace rollout backend received duplicate GPU leases")
        expected_gpus = set(self.config.gpu_ids)
        if expected_gpus and actual_gpus != expected_gpus:
            raise SlotStateError(
                f"HuggingFace rollout backend expected gpu leases {sorted(expected_gpus)}, "
                f"got {sorted(actual_gpus)}"
            )
        for item in leases:
            if item.role != RoleName.ROLLOUT:
                raise SlotStateError(f"HuggingFace rollout backend lease role is {item.role}, not rollout")

        expected_holders_by_gpu = self.config.holders_by_gpu
        if expected_holders_by_gpu:
            for item in leases:
                expected_holder = expected_holders_by_gpu[item.gpu_id]
                if item.holder_id != expected_holder:
                    raise SlotStateError(
                        f"HuggingFace rollout backend lease holder for gpu {item.gpu_id} is {item.holder_id}, "
                        f"expected {expected_holder}"
                    )
            return leases

        expected_holders = set(self.config.holder_ids)
        if expected_holders and {item.holder_id for item in leases} != expected_holders:
            raise SlotStateError(
                f"HuggingFace rollout backend expected lease holders {sorted(expected_holders)}, "
                f"got {sorted(item.holder_id for item in leases)}"
            )
        if self.config.holder_id is not None:
            for item in leases:
                if item.holder_id != self.config.holder_id:
                    raise SlotStateError(
                        f"HuggingFace rollout backend lease holder is {item.holder_id}, expected {self.config.holder_id}"
                    )
        return leases


def build_huggingface_rollout_backend(
    config: HuggingFaceBackendConfig | Mapping[str, Any] | None = None,
    *,
    model_factory: HfModelFactory | None = None,
    tokenizer_factory: HfTokenizerFactory | None = None,
) -> HuggingFaceRolloutBackend:
    resolved = (
        config
        if isinstance(config, HuggingFaceBackendConfig)
        else HuggingFaceBackendConfig.from_rollout_backend_config(config or {})
    )
    return HuggingFaceRolloutBackend(
        resolved,
        model_factory=model_factory,
        tokenizer_factory=tokenizer_factory,
    )


def _default_model_factory(
    config: HuggingFaceBackendConfig,
    _meta: WeightMeta,
    modules: Mapping[str, Any],
) -> Any:
    transformers = modules["transformers"]
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        **config.model_kwargs,
    }
    torch_dtype = _torch_dtype(config.dtype, modules["torch"])
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if config.device_map is not None:
        kwargs["device_map"] = config.device_map
    return transformers.AutoModelForCausalLM.from_pretrained(config.model_path, **kwargs)


def _default_tokenizer_factory(
    config: HuggingFaceBackendConfig,
    _meta: WeightMeta,
    modules: Mapping[str, Any],
) -> Any:
    transformers = modules["transformers"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.tokenizer_path or config.model_path,
        trust_remote_code=config.trust_remote_code,
        **config.tokenizer_kwargs,
    )
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _torch_dtype(name: str | None, torch: Any) -> Any:
    if name is None:
        return None
    normalized = str(name).lower()
    if normalized == "auto":
        return "auto"
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    attr = aliases.get(normalized, normalized)
    if not hasattr(torch, attr):
        raise HuggingFaceBackendError(f"unsupported HuggingFace rollout dtype: {name}")
    return getattr(torch, attr)


def _lease_sequence(lease: GpuLease | Sequence[GpuLease]) -> tuple[GpuLease, ...]:
    if isinstance(lease, GpuLease):
        return (lease,)
    leases = tuple(lease)
    if not leases:
        raise SlotStateError("HuggingFace rollout backend requires at least one GPU lease")
    return leases


def _move_encoded_to_device(encoded: Any, device: str) -> Any:
    if hasattr(encoded, "to"):
        return encoded.to(device)
    if isinstance(encoded, Mapping):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
    return encoded


def _encoded_to_kwargs(encoded: Any) -> dict[str, Any]:
    if isinstance(encoded, Mapping):
        return dict(encoded)
    if hasattr(encoded, "items"):
        return dict(encoded.items())
    raise HuggingFaceBackendError("tokenizer output must be mapping-like")


def _encoded_get(encoded: Any, key: str) -> Any:
    if isinstance(encoded, Mapping):
        return encoded.get(key)
    if hasattr(encoded, "get"):
        return encoded.get(key)
    return getattr(encoded, key, None)


def _sequence_length(input_ids: Any) -> int:
    if input_ids is None:
        return 0
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    if isinstance(input_ids, Sequence) and not isinstance(input_ids, (str, bytes)):
        first = input_ids[0] if input_ids and isinstance(input_ids[0], Sequence) else input_ids
        return len(first)
    return 0


def _first_sequence(sequences: Any) -> Any:
    if hasattr(sequences, "tolist"):
        sequences = sequences.tolist()
    if isinstance(sequences, Sequence) and not isinstance(sequences, (str, bytes)):
        if not sequences:
            raise HuggingFaceBackendError("HuggingFace generate returned no sequences")
        first = sequences[0]
        if hasattr(first, "tolist"):
            return first.tolist()
        return first
    return sequences


def _tokens_after_prompt(sequence: Any, prompt_token_count: int) -> list[int]:
    if hasattr(sequence, "tolist"):
        sequence = sequence.tolist()
    tokens = [int(token) for token in list(sequence)]
    return tokens[prompt_token_count:]


def _transition_logprobs(model: Any, raw_output: Any, tokens: Sequence[int]) -> list[float]:
    scores = _read_attr(raw_output, "scores", default=None)
    sequences = _read_attr(raw_output, "sequences", default=None)
    method = getattr(model, "compute_transition_scores", None)
    if scores is None or sequences is None or method is None:
        return [0.0 for _ in tokens]
    try:
        transition_scores = method(sequences, scores, normalize_logits=True)
    except TypeError:
        transition_scores = method(sequences, scores)
    values = _first_sequence(transition_scores)
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in list(values)[-len(tokens):]] if tokens else []


def _finish_reason(tokens: Sequence[int], tokenizer: Any, generation_kwargs: Mapping[str, Any]) -> str:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and tokens and tokens[-1] == eos_id:
        return "stop"
    max_new_tokens = generation_kwargs.get("max_new_tokens")
    try:
        if max_new_tokens is not None and len(tokens) >= int(max_new_tokens):
            return "length"
    except (TypeError, ValueError):
        pass
    return "stop"


def _move_model(model: Any, device: str) -> None:
    if hasattr(model, "to"):
        model.to(device)


def _empty_cuda_cache(modules: Mapping[str, Any] | None) -> None:
    if modules is None:
        return
    torch = modules.get("torch")
    cuda = None if torch is None else getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available() and hasattr(cuda, "empty_cache"):
        cuda.empty_cache()


def _read_attr(value: Any, name: str, *, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)
