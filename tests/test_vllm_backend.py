from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib
import os

import pytest

import nano_rl.runtime.backends.vllm_backend as vllm_backend_module
from nano_rl.runtime.backends import (
    GenerationOutput,
    VllmBackendConfig,
    VllmBackendError,
    VllmBackendUnavailable,
    build_rollout_backend,
    generation_output_to_sample_record,
)
from nano_rl.runtime.protocols import WeightFormat, WeightMeta, WeightShardSource, WeightShardSourceKind
from nano_rl.runtime.slot import GpuLease, RoleName


@dataclass
class FakeLogProb:
    logprob: float


@dataclass
class FakeCompletion:
    text: str
    token_ids: list[int]
    logprobs: list[dict[int, FakeLogProb]]
    finish_reason: str = "stop"


@dataclass
class FakeRequestOutput:
    outputs: list[FakeCompletion]
    request_id: str | None = None


class FakeVllmEngine:
    def __init__(self, config: VllmBackendConfig, meta: WeightMeta) -> None:
        self.config = config
        self.initial_meta = meta
        self.activated: list[tuple[int, str, str | None]] = []
        self.generate_calls: list[dict[str, object]] = []
        self.sleep_calls: list[int] = []
        self.wake_calls = 0

    def activate_weight(self, meta: WeightMeta, *, transfer_source: WeightShardSource | None = None) -> None:
        self.activated.append((meta.version_id, meta.checksum, None if transfer_source is None else transfer_source.kind))

    def sleep(self, level: int = 1) -> None:
        self.sleep_calls.append(level)

    def wake_up(self) -> None:
        self.wake_calls += 1

    def generate(self, prompts, *, sampling_params=None, use_tqdm=False, metadata=None):
        prompt = prompts[0]
        token_ids = [101, len(prompt), 202]
        logprobs = [
            {101: FakeLogProb(-0.1)},
            {len(prompt): FakeLogProb(-0.2)},
            {202: FakeLogProb(-0.3)},
        ]
        self.generate_calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
                "metadata": metadata,
            }
        )
        return [
            FakeRequestOutput(
                outputs=[FakeCompletion(f"{prompt} -> fake", token_ids, logprobs)],
                request_id=f"engine-{prompt}",
            )
        ]


class FakeVllmEngineWithoutSleep(FakeVllmEngine):
    sleep = None
    offload = None


def test_importing_backend_does_not_import_vllm() -> None:
    module = importlib.import_module("nano_rl.runtime.backends.vllm_backend")

    assert hasattr(module, "VllmRolloutBackend")


def test_fake_engine_activate_and_generate_normalizes_vllm_output() -> None:
    engines: list[FakeVllmEngine] = []

    def factory(config: VllmBackendConfig, meta: WeightMeta) -> FakeVllmEngine:
        engine = FakeVllmEngine(config, meta)
        engines.append(engine)
        return engine

    backend = build_rollout_backend(
        {
            "gpu_ids": (0,),
            "holder_id": "rollout-dp-0-tp-0",
            "sampling_params": {"temperature": 0.0, "max_tokens": 8},
        },
        engine_factory=factory,
    )
    weight = _weight(version_id=7, checksum="abc123", model_path="/weights/v7", tokenizer_path="/tok")
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="rollout-dp-0-tp-0", lease_epoch=3)

    backend.activate_weight(weight, lease=lease)
    output = backend.generate(
        prompt="hello",
        target_policy_version=7,
        request_metadata={"controller_step": 11},
        request_id="req-1",
        lease=lease,
    )

    assert backend.active_weight == weight
    assert engines[0].config.model_path == "/weights/v7"
    assert engines[0].config.tokenizer_path == "/tok"
    assert engines[0].activated == [(7, "abc123", None)]
    assert output.request_id == "req-1"
    assert output.prompt == "hello"
    assert output.response == "hello -> fake"
    assert output.tokens == [101, 5, 202]
    assert output.logprobs == [-0.1, -0.2, -0.3]
    assert output.old_logprobs == output.logprobs
    assert output.finish_reason == "stop"
    assert output.policy_version == 7
    assert output.weight_checksum == "abc123"
    assert output.metadata == {"controller_step": 11}


def test_backend_forwards_weight_transfer_source_to_engine() -> None:
    engines: list[FakeVllmEngine] = []

    def factory(config: VllmBackendConfig, meta: WeightMeta) -> FakeVllmEngine:
        engine = FakeVllmEngine(config, meta)
        engines.append(engine)
        return engine

    backend = build_rollout_backend(
        {"gpu_ids": (4,), "holder_id": "worker-4"},
        engine_factory=factory,
    )
    lease = GpuLease(gpu_id=4, role=RoleName.ROLLOUT, holder_id="worker-4", lease_epoch=1)
    source = WeightShardSource(
        replica_id="rollout-dp-2",
        kind=WeightShardSourceKind.SHARED_GPU_RESHARD,
        target_gpu_ids=(4,),
        source_rank_ids=(0,),
        source_gpu_ids=(4,),
        reason="unit test",
    )

    backend.activate_weight(_weight(version_id=6), lease=lease, transfer_source=source)

    assert backend.active_weight_source == source
    assert engines[0].activated == [(6, "checksum", "shared_gpu_reshard")]


def test_backend_offloads_and_wakes_vllm_engine_before_generate() -> None:
    engines: list[FakeVllmEngine] = []

    def factory(config: VllmBackendConfig, meta: WeightMeta) -> FakeVllmEngine:
        engine = FakeVllmEngine(config, meta)
        engines.append(engine)
        return engine

    backend = build_rollout_backend(
        {"gpu_ids": (0,), "holder_id": "worker-0", "vllm_sleep_level": 2},
        engine_factory=factory,
    )
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=1)
    backend.activate_weight(_weight(version_id=4), lease=lease)

    offloaded = backend.offload(lease=lease)
    with pytest.raises(Exception, match="offloaded"):
        backend.generate(prompt="blocked", target_policy_version=4, request_metadata={}, lease=lease)
    woken = backend.wake(lease=lease)
    output = backend.generate(prompt="ready", target_policy_version=4, request_metadata={}, lease=lease)

    assert offloaded["residency"] == "offloaded"
    assert woken["residency"] == "active"
    assert engines[0].sleep_calls == [2]
    assert engines[0].wake_calls == 1
    assert output.response == "ready -> fake"


def test_vllm_sleep_forces_sleep_mode_engine_kwarg() -> None:
    engines: list[FakeVllmEngine] = []

    def factory(config: VllmBackendConfig, meta: WeightMeta) -> FakeVllmEngine:
        engine = FakeVllmEngine(config, meta)
        engines.append(engine)
        return engine

    backend = build_rollout_backend(
        {
            "gpu_ids": (0,),
            "holder_id": "worker-0",
            "engine_kwargs": {"enable_sleep_mode": False},
        },
        engine_factory=factory,
    )
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=1)

    backend.activate_weight(_weight(version_id=9), lease=lease)

    assert engines[0].config.engine_kwargs["enable_sleep_mode"] is True


def test_vllm_sleep_does_not_teardown_when_sleep_api_is_missing() -> None:
    engines: list[FakeVllmEngineWithoutSleep] = []

    def factory(config: VllmBackendConfig, meta: WeightMeta) -> FakeVllmEngineWithoutSleep:
        engine = FakeVllmEngineWithoutSleep(config, meta)
        engines.append(engine)
        return engine

    backend = build_rollout_backend(
        {"gpu_ids": (0,), "holder_id": "worker-0"},
        engine_factory=factory,
    )
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=1)
    backend.activate_weight(_weight(version_id=10), lease=lease)

    with pytest.raises(VllmBackendError, match="cannot keep engine resident"):
        backend.offload(lease=lease)

    assert backend.engine is engines[0]


def test_vllm_sleep_enforces_residual_gpu_memory_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    engines: list[FakeVllmEngine] = []

    def factory(config: VllmBackendConfig, meta: WeightMeta) -> FakeVllmEngine:
        engine = FakeVllmEngine(config, meta)
        engines.append(engine)
        return engine

    monkeypatch.setattr(
        vllm_backend_module,
        "_visible_gpu_memory_snapshot_mb",
        lambda _config: (
            {
                "local_device": 0,
                "gpu_id": 0,
                "used_mb": 4096,
                "free_mb": 1024,
                "total_mb": 5120,
            },
        ),
    )
    backend = build_rollout_backend(
        {
            "gpu_ids": (0,),
            "holder_id": "worker-0",
            "residual_gpu_memory_budget_mb": 2048,
        },
        engine_factory=factory,
    )
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=1)
    backend.activate_weight(_weight(version_id=11), lease=lease)

    with pytest.raises(VllmBackendError, match="residual GPU memory above budget"):
        backend.offload(lease=lease)

    assert engines[0].sleep_calls == [2]
    with pytest.raises(Exception, match="offloaded"):
        backend.generate(prompt="blocked", target_policy_version=11, request_metadata={}, lease=lease)


def test_fake_engine_output_converts_to_sample_record() -> None:
    backend = build_rollout_backend(
        {"gpu_ids": (0,), "holder_id": "worker-0"},
        engine_factory=FakeVllmEngine,
    )
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=1)
    backend.activate_weight(_weight(version_id=5, checksum="weight-5"), lease=lease)
    output = backend.generate(
        prompt="sample prompt",
        target_policy_version=5,
        request_metadata={"controller_step": 9, "prompt_tokens": 2},
        request_id="req-5",
        lease=lease,
    )

    sample = generation_output_to_sample_record(
        output,
        reward=0.75,
        reward_source="unit_reward",
        request_metadata={"batch_id": "batch-1"},
    )

    assert sample.sample_id == "req-5"
    assert sample.request_id == "req-5"
    assert sample.policy_version == 5
    assert sample.weight_checksum == "weight-5"
    assert sample.prompt == "sample prompt"
    assert sample.response == "sample prompt -> fake"
    assert sample.tokens == [101, 13, 202]
    assert sample.logprobs == [-0.1, -0.2, -0.3]
    assert sample.old_logprobs == [-0.1, -0.2, -0.3]
    assert sample.reward == 0.75
    assert sample.reward_source == "unit_reward"
    assert sample.prompt_tokens == 2
    assert sample.response_tokens == 3
    assert sample.meta == {"batch_id": "batch-1", "controller_step": 9, "prompt_tokens": 2}


def test_sample_record_uses_metadata_request_id_and_sample_id_when_output_has_none() -> None:
    output = GenerationOutput(
        prompt="p",
        response="r",
        tokens=[1],
        logprobs=[-0.1],
        old_logprobs=[-0.2],
        finish_reason="length",
        policy_version=4,
        weight_checksum="checksum-4",
        metadata={"request_id": "metadata-request"},
    )

    sample = generation_output_to_sample_record(
        output,
        reward=1,
        reward_source="reward",
        request_metadata={"sample_id": "sample-from-metadata", "request_id": "request-from-metadata"},
    )

    assert sample.sample_id == "sample-from-metadata"
    assert sample.request_id == "metadata-request"
    assert sample.policy_version == 4
    assert sample.old_logprobs == [-0.2]
    assert sample.reward == 1.0
    assert sample.meta == {
        "sample_id": "sample-from-metadata",
        "request_id": "metadata-request",
    }


def test_backend_validates_rollout_gpu_and_holder_lease() -> None:
    backend = build_rollout_backend(
        {"gpu_ids": (0,), "holder_id": "rollout-dp-0-tp-0"},
        engine_factory=FakeVllmEngine,
    )
    weight = _weight(version_id=1)

    with pytest.raises(Exception, match="not rollout"):
        backend.activate_weight(
            weight,
            lease=GpuLease(gpu_id=0, role=RoleName.TRAINER, holder_id="rollout-dp-0-tp-0", lease_epoch=1),
        )
    with pytest.raises(Exception, match="expected gpu leases"):
        backend.activate_weight(
            weight,
            lease=GpuLease(gpu_id=1, role=RoleName.ROLLOUT, holder_id="rollout-dp-0-tp-0", lease_epoch=1),
        )
    with pytest.raises(Exception, match="expected rollout-dp-0-tp-0"):
        backend.activate_weight(
            weight,
            lease=GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="other", lease_epoch=1),
        )


def test_backend_validates_multi_gpu_holder_ids_and_sets_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    engines: list[FakeVllmEngine] = []

    def factory(config: VllmBackendConfig, meta: WeightMeta) -> FakeVllmEngine:
        engine = FakeVllmEngine(config, meta)
        engines.append(engine)
        return engine

    backend = build_rollout_backend(
        {
            "tensor_parallel_size": 2,
            "gpu_ids": (4, 6),
            "holder_ids": ("rollout-dp-1-tp-0", "rollout-dp-1-tp-1"),
        },
        engine_factory=factory,
    )
    weight = _weight(version_id=8)
    leases = (
        GpuLease(gpu_id=4, role=RoleName.ROLLOUT, holder_id="rollout-dp-1-tp-0", lease_epoch=2),
        GpuLease(gpu_id=6, role=RoleName.ROLLOUT, holder_id="rollout-dp-1-tp-1", lease_epoch=3),
    )

    backend.activate_weight(weight, lease=leases)
    output = backend.generate(prompt="tp", target_policy_version=8, request_metadata={}, lease=leases)

    assert output.request_id == "engine-tp"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,6"
    assert engines[0].config.cuda_visible_devices == "4,6"

    swapped_holders = (
        GpuLease(gpu_id=4, role=RoleName.ROLLOUT, holder_id="rollout-dp-1-tp-1", lease_epoch=2),
        GpuLease(gpu_id=6, role=RoleName.ROLLOUT, holder_id="rollout-dp-1-tp-0", lease_epoch=3),
    )
    with pytest.raises(Exception, match="lease holder for gpu 4"):
        backend.generate(prompt="bad", target_policy_version=8, request_metadata={}, lease=swapped_holders)


def test_backend_rejects_mismatched_holder_ids_length() -> None:
    with pytest.raises(ValueError, match="holder_ids length must match gpu_ids length"):
        VllmBackendConfig.model_validate(
            {
                "tensor_parallel_size": 2,
                "gpu_ids": (0, 1),
                "holder_ids": ("only-one-holder",),
            }
        )


def test_default_backend_reports_structured_error_when_vllm_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "vllm":
            raise ImportError("no vllm in test")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    backend = build_rollout_backend({"gpu_ids": (0,), "holder_id": "worker"})

    with pytest.raises(VllmBackendUnavailable) as exc_info:
        backend.activate_weight(
            _weight(version_id=2),
            lease=GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker", lease_epoch=1),
        )

    assert exc_info.value.package_name == "vllm"
    assert exc_info.value.action == "construct rollout backend"
    assert isinstance(exc_info.value.original_error, ImportError)


def test_generate_requires_active_target_policy_version() -> None:
    backend = build_rollout_backend({"gpu_ids": (0,), "holder_id": "worker"}, engine_factory=FakeVllmEngine)
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker", lease_epoch=1)
    backend.activate_weight(_weight(version_id=3), lease=lease)

    with pytest.raises(Exception, match="does not match target"):
        backend.generate(prompt="x", target_policy_version=4, request_metadata={}, lease=lease)


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
        format=WeightFormat.VLLM_COMPATIBLE,
        checksum=checksum,
    )
