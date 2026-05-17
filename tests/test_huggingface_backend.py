from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib
import os

import pytest

from nano_rl.runtime.backends import (
    HuggingFaceBackendConfig,
    HuggingFaceBackendUnavailable,
    HuggingFaceRolloutBackend,
    build_huggingface_rollout_backend,
    build_rollout_backend,
)
from nano_rl.runtime.protocols import WeightFormat, WeightMeta
from nano_rl.runtime.slot import GpuLease, RoleName


class FakeBatch(dict):
    def to(self, device: str) -> "FakeBatch":
        self["device"] = device
        return self


class FakeTokenizer:
    eos_token_id = 99
    eos_token = "<eos>"
    pad_token = None

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, prompt: str, *, return_tensors: str) -> FakeBatch:
        self.calls.append({"prompt": prompt, "return_tensors": return_tensors})
        return FakeBatch(input_ids=[[11, 12]], attention_mask=[[1, 1]])

    def decode(self, tokens: list[int], *, skip_special_tokens: bool = True) -> str:
        return "decoded:" + ",".join(str(token) for token in tokens)


@dataclass
class FakeGenerateOutput:
    sequences: list[list[int]]
    scores: list[object] | None = None


class FakeModel:
    def __init__(self, config: HuggingFaceBackendConfig, meta: WeightMeta) -> None:
        self.config = config
        self.meta = meta
        self.devices: list[str] = []
        self.generate_calls: list[dict[str, object]] = []

    def to(self, device: str) -> "FakeModel":
        self.devices.append(device)
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return FakeGenerateOutput(sequences=[kwargs["input_ids"][0] + [31, 32, 99]], scores=[object()])

    def compute_transition_scores(self, sequences, scores, normalize_logits: bool = True):
        return [[-0.1, -0.2, -0.3]]


def test_build_rollout_backend_dispatches_to_huggingface_without_importing_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    models: list[FakeModel] = []
    tokenizers: list[FakeTokenizer] = []

    def model_factory(config: HuggingFaceBackendConfig, meta: WeightMeta, modules) -> FakeModel:
        model = FakeModel(config, meta)
        models.append(model)
        return model

    def tokenizer_factory(config: HuggingFaceBackendConfig, meta: WeightMeta, modules) -> FakeTokenizer:
        tokenizer = FakeTokenizer()
        tokenizers.append(tokenizer)
        return tokenizer

    backend = build_rollout_backend(
        {
            "backend": "huggingface",
            "model_path": "/base-model",
            "tokenizer_path": "/base-tokenizer",
            "tensor_parallel_size": 2,
            "gpu_ids": (4, 6),
            "holder_ids": ("rollout-dp-1-tp-0", "rollout-dp-1-tp-1"),
            "huggingface": {
                "dtype": "bfloat16",
                "device": "cpu",
                "generation_kwargs": {"max_new_tokens": 3, "do_sample": False},
            },
        },
        hf_model_factory=model_factory,
        hf_tokenizer_factory=tokenizer_factory,
    )
    assert isinstance(backend, HuggingFaceRolloutBackend)
    leases = (
        GpuLease(gpu_id=4, role=RoleName.ROLLOUT, holder_id="rollout-dp-1-tp-0", lease_epoch=2),
        GpuLease(gpu_id=6, role=RoleName.ROLLOUT, holder_id="rollout-dp-1-tp-1", lease_epoch=3),
    )
    weight = _weight(version_id=8, checksum="abc123", model_path="/weights/v8", tokenizer_path="/tok/v8")

    backend.activate_weight(weight, lease=leases)
    output = backend.generate(
        prompt="hello",
        target_policy_version=8,
        request_metadata={"controller_step": 11},
        request_id="req-8",
        lease=leases,
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,6"
    assert models[0].config.model_path == "/weights/v8"
    assert models[0].config.tokenizer_path == "/tok/v8"
    assert models[0].devices == ["cpu"]
    assert models[0].generate_calls[0]["max_new_tokens"] == 3
    assert output.request_id == "req-8"
    assert output.response == "decoded:31,32,99"
    assert output.tokens == [31, 32, 99]
    assert output.logprobs == [-0.1, -0.2, -0.3]
    assert output.finish_reason == "stop"
    assert output.policy_version == 8
    assert output.weight_checksum == "abc123"
    assert output.metadata["backend"] == "huggingface"
    assert output.metadata["prompt_tokens"] == 2
    assert output.metadata["huggingface_rollout"]["gpu_ids"] == [4, 6]
    assert output.metadata["huggingface_rollout"]["lease_epochs"] == {"4": 2, "6": 3}


def test_huggingface_backend_requires_activate_and_matching_policy_version() -> None:
    backend = build_huggingface_rollout_backend(
        {"gpu_ids": (0,), "holder_id": "worker", "device": "cpu"},
        model_factory=lambda config, meta, modules: FakeModel(config, meta),
        tokenizer_factory=lambda config, meta, modules: FakeTokenizer(),
    )
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker", lease_epoch=1)

    with pytest.raises(Exception, match="cannot generate before activate_weight"):
        backend.generate(prompt="x", target_policy_version=1, request_metadata={}, lease=lease)

    backend.activate_weight(_weight(version_id=3), lease=lease)
    with pytest.raises(Exception, match="does not match target"):
        backend.generate(prompt="x", target_policy_version=4, request_metadata={}, lease=lease)


def test_huggingface_backend_offload_and_wake_reload_model() -> None:
    models: list[FakeModel] = []

    def model_factory(config: HuggingFaceBackendConfig, meta: WeightMeta, modules) -> FakeModel:
        model = FakeModel(config, meta)
        models.append(model)
        return model

    backend = build_huggingface_rollout_backend(
        {"gpu_ids": (0,), "holder_id": "worker", "device": "cpu", "offload_strategy": "teardown_and_reload"},
        model_factory=model_factory,
        tokenizer_factory=lambda config, meta, modules: FakeTokenizer(),
    )
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker", lease_epoch=1)
    backend.activate_weight(_weight(version_id=5), lease=lease)

    offloaded = backend.offload(lease=lease)
    with pytest.raises(Exception, match="offloaded"):
        backend.generate(prompt="x", target_policy_version=5, request_metadata={}, lease=lease)
    woken = backend.wake(lease=lease)
    output = backend.generate(prompt="ready", target_policy_version=5, request_metadata={}, lease=lease)

    assert offloaded["action"] == "teardown"
    assert woken["action"] == "reload"
    assert len(models) == 2
    assert output.response == "decoded:31,32,99"


def test_huggingface_backend_reloads_when_checksum_changes_at_same_path() -> None:
    models: list[FakeModel] = []

    def model_factory(config: HuggingFaceBackendConfig, meta: WeightMeta, modules) -> FakeModel:
        model = FakeModel(config, meta)
        models.append(model)
        return model

    backend = build_huggingface_rollout_backend(
        {"gpu_ids": (0,), "holder_id": "worker", "device": "cpu"},
        model_factory=model_factory,
        tokenizer_factory=lambda config, meta, modules: FakeTokenizer(),
    )
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker", lease_epoch=1)

    backend.activate_weight(_weight(version_id=1, checksum="a", model_path="/same"), lease=lease)
    backend.activate_weight(_weight(version_id=2, checksum="b", model_path="/same"), lease=lease)

    assert len(models) == 2
    assert backend.active_weight.version_id == 2


def test_huggingface_backend_validates_rollout_lease() -> None:
    backend = build_huggingface_rollout_backend(
        {"gpu_ids": (0,), "holder_id": "worker"},
        model_factory=lambda config, meta, modules: FakeModel(config, meta),
        tokenizer_factory=lambda config, meta, modules: FakeTokenizer(),
    )

    with pytest.raises(Exception, match="not rollout"):
        backend.activate_weight(
            _weight(version_id=1),
            lease=GpuLease(gpu_id=0, role=RoleName.TRAINER, holder_id="worker", lease_epoch=1),
        )
    with pytest.raises(Exception, match="expected gpu leases"):
        backend.activate_weight(
            _weight(version_id=1),
            lease=GpuLease(gpu_id=1, role=RoleName.ROLLOUT, holder_id="worker", lease_epoch=1),
        )
    with pytest.raises(Exception, match="expected worker"):
        backend.activate_weight(
            _weight(version_id=1),
            lease=GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="other", lease_epoch=1),
        )


def test_default_huggingface_backend_reports_structured_error_when_import_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name in {"torch", "transformers"}:
            raise ImportError(f"no {name} in test")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    backend = build_huggingface_rollout_backend({"gpu_ids": (0,), "holder_id": "worker"})

    with pytest.raises(HuggingFaceBackendUnavailable) as exc_info:
        backend.activate_weight(
            _weight(version_id=2),
            lease=GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker", lease_epoch=1),
        )

    assert exc_info.value.package_name == "transformers"
    assert exc_info.value.action == "construct rollout backend"
    assert isinstance(exc_info.value.original_error, ImportError)


def _weight(
    *,
    version_id: int,
    checksum: str = "checksum",
    model_path: str = "/model",
    tokenizer_path: str | None = None,
) -> WeightMeta:
    return WeightMeta(
        version_id=version_id,
        created_at=datetime.utcnow(),
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        format=WeightFormat.HF,
        checksum=checksum,
    )
