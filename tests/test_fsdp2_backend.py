from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nano_rl.exceptions import SlotStateError
from nano_rl.runtime.backends import (
    BackendStateError,
    BackendUnavailableError,
    FakeTrainerBackend,
    Fsdp2TrainerBackend,
    OptimizerStepResult,
    TrainerBackendConfig,
    TrainerProcessGroupMetadata,
    TrainStateBundle,
    build_trainer_backend,
)
from nano_rl.runtime.protocols import SampleRecord, SampleRef, TrainBatch, WeightFormat, WeightMeta
from nano_rl.runtime.slot import GpuLease, RoleName


def _config(rank: int = 0, gpu_id: int = 4, world_size: int = 2) -> TrainerBackendConfig:
    return TrainerBackendConfig(
        backend="fake",
        rank=rank,
        world_size=world_size,
        gpu_id=gpu_id,
        group_epoch=3,
        rendezvous="env://",
        store_endpoint="127.0.0.1:29500",
        comm_epoch=9,
    )


def _lease(rank: int = 0, gpu_id: int = 4, *, role: RoleName = RoleName.TRAINER) -> GpuLease:
    return GpuLease(gpu_id=gpu_id, role=role, holder_id=f"trainer-rank-{rank}", lease_epoch=11)


def _batch() -> TrainBatch:
    now = datetime.utcnow()
    return TrainBatch(
        train_batch_id="train-batch-1",
        sample_refs=[
            SampleRef(sample_id="s0", policy_version=7, created_at=now, num_tokens=3),
            SampleRef(sample_id="s1", policy_version=7, created_at=now, num_tokens=5),
        ],
        sample_ids=["s0", "s1"],
        policy_version_min=7,
        policy_version_max=7,
        policy_version_histogram={7: 2},
        num_sequences=2,
        num_tokens=8,
        reserved_by="trainer-coordinator",
        reserved_at=now,
    )


def _weight(version_id: int = 7) -> WeightMeta:
    return WeightMeta(
        version_id=version_id,
        created_at=datetime.utcnow(),
        model_path="/models/qwen",
        tokenizer_path="/models/qwen",
        format=WeightFormat.HF,
        checksum="abc123",
        trainer_step=0,
        created_by="test",
    )


def _record(sample_id: str, *, prompt: str = "prompt", response: str = "response") -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        policy_version=7,
        prompt=prompt,
        response=response,
        tokens=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        reward=1.0,
    )


def _batch_with_records(*, object_ref: Any | None = None) -> TrainBatch:
    now = datetime.utcnow()
    record = _record("s0")
    return TrainBatch(
        train_batch_id="train-batch-records",
        sample_refs=[
            SampleRef(
                sample_id="s0",
                policy_version=7,
                created_at=now,
                object_ref=record if object_ref is None else object_ref,
                num_tokens=3,
            )
        ],
        sample_ids=["s0"],
        policy_version_min=7,
        policy_version_max=7,
        policy_version_histogram={7: 1},
        num_sequences=1,
        num_tokens=3,
        reserved_by="trainer-coordinator",
        reserved_at=now,
    )


class FakeTensor:
    def __init__(self, value: Any = None):
        self.value = value
        self.devices: list[Any] = []
        self.detach_calls = 0

    def to(self, device: Any) -> "FakeTensor":
        self.devices.append(device)
        return self

    def clone(self) -> "FakeTensor":
        return FakeTensor(self.value)

    def detach(self) -> "FakeTensor":
        self.detach_calls += 1
        return self

    def cpu(self) -> "FakeTensor":
        self.devices.append("cpu")
        return self


class FakeLoss:
    def __init__(self, value: float):
        self.value = value
        self.backward_calls = 0

    def backward(self) -> None:
        self.backward_calls += 1

    def detach(self) -> "FakeLoss":
        return self

    def cpu(self) -> "FakeLoss":
        return self

    def item(self) -> float:
        return self.value


class FakeModel:
    def __init__(self) -> None:
        self.to_devices: list[Any] = []
        self.forward_calls: list[dict[str, Any]] = []
        self.saved_dirs: list[Path] = []
        self.saved_state_dicts: list[dict[str, Any] | None] = []
        self.losses: list[FakeLoss] = []

    def to(self, device: Any) -> "FakeModel":
        self.to_devices.append(device)
        return self

    def parameters(self) -> list[str]:
        return ["param"]

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        loss = FakeLoss(2.5)
        self.losses.append(loss)
        self.forward_calls.append(kwargs)
        return SimpleNamespace(loss=loss)

    def save_pretrained(self, output_dir: Path, *, state_dict: dict[str, Any] | None = None) -> None:
        self.saved_dirs.append(Path(output_dir))
        self.saved_state_dicts.append(state_dict)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "model.txt").write_text("fake model\n", encoding="utf-8")


class FakeTokenizer:
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.pad_token: str | None = None
        self.calls: list[dict[str, Any]] = []
        self.saved_dirs: list[Path] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, FakeTensor]:
        self.calls.append({"texts": texts, **kwargs})
        return {"input_ids": FakeTensor([1, 2, 3]), "attention_mask": FakeTensor([1, 1, 1])}

    def save_pretrained(self, output_dir: Path) -> None:
        self.saved_dirs.append(Path(output_dir))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "tokenizer.txt").write_text("fake tokenizer\n", encoding="utf-8")


class FakeAdamW:
    def __init__(self, params: list[str], *, lr: float, weight_decay: float) -> None:
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_calls = 0
        self.zero_grad_calls: list[dict[str, Any]] = []
        self.state: dict[str, dict[str, FakeTensor]] = {}

    def step(self) -> None:
        self.step_calls += 1
        self.state.setdefault("param", {})["exp_avg"] = FakeTensor("avg")
        self.state.setdefault("param", {})["exp_avg_sq"] = FakeTensor("avg_sq")

    def zero_grad(self, **kwargs: Any) -> None:
        self.zero_grad_calls.append(kwargs)


class FakeStateDictOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeRayObjectRef:
    def __init__(self, payload: Any) -> None:
        self.payload = payload


class FakeFsdp2Modules:
    def __init__(self, *, cuda_available: bool = True, ray_module: Any | None = None) -> None:
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.optimizers: list[FakeAdamW] = []
        self.clip_calls: list[tuple[list[str], float]] = []
        self.empty_cache_calls = 0
        self.init_process_group_calls: list[dict[str, Any]] = []
        self.fully_shard_calls: list[FakeModel] = []
        self.state_dict_calls: list[dict[str, Any]] = []
        self.ray_module = ray_module

        def is_available() -> bool:
            return cuda_available

        def empty_cache() -> None:
            self.empty_cache_calls += 1

        def adamw(params: list[str], *, lr: float, weight_decay: float) -> FakeAdamW:
            optimizer = FakeAdamW(params, lr=lr, weight_decay=weight_decay)
            self.optimizers.append(optimizer)
            return optimizer

        def clip_grad_norm_(params: list[str], max_norm: float) -> None:
            self.clip_calls.append((params, max_norm))

        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=is_available, empty_cache=empty_cache),
            device=lambda name: name,
            optim=SimpleNamespace(AdamW=adamw),
            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=clip_grad_norm_)),
            bfloat16="bf16",
            float16="fp16",
        )
        self.distributed = SimpleNamespace(
            is_initialized=lambda: bool(self.init_process_group_calls),
            init_process_group=self._init_process_group,
        )
        self.fsdp = SimpleNamespace(fully_shard=self._fully_shard)
        self.transformers = SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=self._model_from_pretrained),
            AutoTokenizer=SimpleNamespace(from_pretrained=self._tokenizer_from_pretrained),
        )
        self.checkpoint_state_dict = SimpleNamespace(
            get_model_state_dict=self._get_model_state_dict,
            StateDictOptions=FakeStateDictOptions,
        )

    def _init_process_group(self, **kwargs: Any) -> None:
        self.init_process_group_calls.append(kwargs)

    def _fully_shard(self, model: FakeModel) -> None:
        self.fully_shard_calls.append(model)
        return None

    def _get_model_state_dict(self, model: FakeModel, *, options: FakeStateDictOptions) -> dict[str, FakeTensor]:
        self.state_dict_calls.append({"model": model, "options": options})
        return {"model.layers.0.weight": FakeTensor("full-weight")}

    def _model_from_pretrained(self, model_path: str, **kwargs: Any) -> FakeModel:
        self.model.from_pretrained = {"model_path": model_path, **kwargs}
        return self.model

    def _tokenizer_from_pretrained(self, model_path: str, **kwargs: Any) -> FakeTokenizer:
        self.tokenizer.from_pretrained = {"model_path": model_path, **kwargs}
        return self.tokenizer


def _install_fake_fsdp2_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool = True,
    ray_module: Any | None = None,
) -> FakeFsdp2Modules:
    from nano_rl.runtime.backends import fsdp2_backend

    modules = FakeFsdp2Modules(cuda_available=cuda_available, ray_module=ray_module)
    by_name = {
        "torch": modules.torch,
        "torch.distributed": modules.distributed,
        "torch.distributed.checkpoint.state_dict": modules.checkpoint_state_dict,
        "torch.distributed._composable.fsdp": modules.fsdp,
        "transformers": modules.transformers,
    }
    if ray_module is not None:
        by_name["ray"] = ray_module

    def fake_import_module(name: str):
        if name in by_name:
            return by_name[name]
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(fsdp2_backend, "import_module", fake_import_module)
    return modules


def test_fake_backend_optimizes_predictably_and_exports_rank0_weight() -> None:
    backend = FakeTrainerBackend(_config(rank=0, gpu_id=4))
    initial_state = backend.initialize_rank()

    assert isinstance(initial_state, TrainStateBundle)
    assert initial_state.rank == 0
    assert initial_state.world_size == 2
    assert initial_state.group_epoch == 3
    assert initial_state.comm_epoch == 9
    assert initial_state.train_step == 0

    result = backend.optimize(_batch(), lease=_lease(rank=0, gpu_id=4))
    exported = backend.export_weight(_weight(version_id=7))

    assert isinstance(result, OptimizerStepResult)
    assert result.train_step == 1
    assert result.loss == pytest.approx(1 / 3)
    assert result.lease_gpu_id == 4
    assert result.lease_epoch == 11
    assert exported.version_id == 8
    assert exported.parent_version == 7
    assert exported.trainer_step == 1
    assert exported.format == WeightFormat.VLLM_COMPATIBLE
    assert exported.created_by == "trainer-rank-0"


@pytest.mark.parametrize(
    ("lease", "match"),
    [
        (_lease(rank=0, gpu_id=4, role=RoleName.ROLLOUT), "not trainer"),
        (_lease(rank=0, gpu_id=5), "expected 4"),
        (GpuLease(gpu_id=4, role=RoleName.TRAINER, holder_id="trainer-rank-1", lease_epoch=11), "expected trainer-rank-0"),
    ],
)
def test_fake_backend_fails_fast_on_wrong_lease(lease: GpuLease, match: str) -> None:
    backend = FakeTrainerBackend(_config(rank=0, gpu_id=4))

    with pytest.raises(SlotStateError, match=match):
        backend.optimize(_batch(), lease=lease)


def test_fake_backend_rejects_non_rank0_export() -> None:
    backend = FakeTrainerBackend(_config(rank=1, gpu_id=5))
    backend.optimize(_batch(), lease=_lease(rank=1, gpu_id=5))

    with pytest.raises(BackendStateError, match="rank 1 cannot export"):
        backend.export_weight(_weight())


def test_fake_backend_hydrate_and_offload_preserve_process_group_metadata() -> None:
    backend = FakeTrainerBackend(_config(rank=0, gpu_id=4))
    hydrated = backend.hydrate(None, lease=_lease(rank=0, gpu_id=4))
    result = backend.optimize(_batch(), lease=_lease(rank=0, gpu_id=4))
    offloaded = backend.offload(lease=_lease(rank=0, gpu_id=4))

    assert hydrated.metadata["process_group"]["rank"] == 0
    assert hydrated.metadata["process_group"]["world_size"] == 2
    assert hydrated.metadata["process_group"]["group_epoch"] == 3
    assert hydrated.metadata["process_group"]["rendezvous"] == "env://"
    assert hydrated.metadata["process_group"]["store_endpoint"] == "127.0.0.1:29500"
    assert hydrated.metadata["process_group"]["comm_epoch"] == 9
    assert offloaded.train_step == result.train_step
    assert offloaded.residency == "cpu_standby"


def test_process_group_metadata_validates_rank_bounds() -> None:
    with pytest.raises(ValueError, match="smaller than world_size"):
        TrainerProcessGroupMetadata(rank=2, world_size=2, group_epoch=0)


def test_build_trainer_backend_factory_returns_requested_backend() -> None:
    fake = build_trainer_backend(_config(rank=0, gpu_id=4))
    fsdp2 = build_trainer_backend(
        TrainerBackendConfig(backend="fsdp2", rank=0, world_size=1, gpu_id=0, group_epoch=0)
    )

    assert isinstance(fake, FakeTrainerBackend)
    assert isinstance(fsdp2, Fsdp2TrainerBackend)


def test_fsdp2_backend_runs_lm_step_exports_and_offloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    modules = _install_fake_fsdp2_modules(monkeypatch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    backend = Fsdp2TrainerBackend(
        TrainerBackendConfig(
            backend="fsdp2",
            rank=0,
            world_size=1,
            gpu_id=3,
            group_epoch=5,
            model_path="/fake/hf-model",
            checkpoint_dir=str(tmp_path),
            extra={"learning_rate": 3.0e-5, "weight_decay": 0.1, "max_grad_norm": 0.7, "mixed_precision": "bf16"},
        )
    )
    lease = _lease(rank=0, gpu_id=3)

    initialized = backend.initialize_rank()
    hydrated = backend.hydrate(initialized, lease=lease)
    result = backend.optimize(_batch_with_records(), lease=lease)
    exported = backend.export_weight(_weight(version_id=7))
    offloaded = backend.offload(lease=lease)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert initialized.metadata["backend"] == "fsdp2"
    assert initialized.residency == "cpu_standby"
    assert hydrated.metadata["hydrated"] is True
    assert result.train_step == 1
    assert result.loss == pytest.approx(2.5)
    assert result.metrics == {"lm_loss": 2.5}
    assert modules.model.from_pretrained["model_path"] == "/fake/hf-model"
    assert modules.model.from_pretrained["torch_dtype"] == "bf16"
    assert modules.tokenizer.pad_token == modules.tokenizer.eos_token
    assert modules.tokenizer.calls[0]["texts"] == ["prompt\nresponse"]
    assert modules.optimizers[0].lr == pytest.approx(3.0e-5)
    assert modules.optimizers[0].weight_decay == pytest.approx(0.1)
    assert modules.optimizers[0].step_calls == 1
    assert modules.optimizers[0].zero_grad_calls == [{"set_to_none": True}]
    assert modules.clip_calls == [(["param"], 0.7)]
    assert modules.model.losses[0].backward_calls == 1
    assert exported.version_id == 8
    assert exported.parent_version == 7
    assert exported.trainer_step == 1
    assert exported.model_path == str(tmp_path / "version-8")
    assert exported.tokenizer_path == str(tmp_path / "version-8")
    assert exported.format == WeightFormat.VLLM_COMPATIBLE
    assert exported.created_by == "trainer-rank-0"
    assert exported.checksum
    assert (tmp_path / "version-8" / "model.txt").exists()
    assert (tmp_path / "version-8" / "tokenizer.txt").exists()
    assert modules.model.to_devices[-1] == "cpu"
    assert modules.optimizers[0].state["param"]["exp_avg"].devices[-1] == "cpu"
    assert modules.optimizers[0].state["param"]["exp_avg_sq"].devices[-1] == "cpu"
    rehydrated = backend.hydrate(offloaded, lease=lease)
    assert rehydrated.metadata["hydrated"] is True
    assert modules.optimizers[0].state["param"]["exp_avg"].devices[-1] == "cuda:0"
    assert modules.optimizers[0].state["param"]["exp_avg_sq"].devices[-1] == "cuda:0"
    assert modules.empty_cache_calls == 2
    assert offloaded.train_step == 1


def test_fsdp2_backend_materializes_sample_record_from_ray_object_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_ref = FakeRayObjectRef(_record("s0", prompt="ray prompt", response="ray response"))
    ray_module = SimpleNamespace(ObjectRef=FakeRayObjectRef, get=lambda ref: ref.payload)
    modules = _install_fake_fsdp2_modules(monkeypatch, ray_module=ray_module)
    backend = Fsdp2TrainerBackend(
        TrainerBackendConfig(
            backend="fsdp2",
            rank=0,
            world_size=1,
            gpu_id=0,
            group_epoch=0,
            model_path="/fake/hf-model",
            checkpoint_dir=str(tmp_path),
        )
    )

    result = backend.optimize(_batch_with_records(object_ref=fake_ref), lease=_lease(rank=0, gpu_id=0))

    assert result.loss == pytest.approx(2.5)
    assert modules.tokenizer.calls[0]["texts"] == ["ray prompt\nray response"]


def test_fsdp2_backend_multi_rank_without_rendezvous_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_fsdp2_modules(monkeypatch)
    backend = Fsdp2TrainerBackend(
        TrainerBackendConfig(
            backend="fsdp2",
            rank=0,
            world_size=2,
            gpu_id=0,
            group_epoch=0,
            model_path="/fake/hf-model",
        )
    )

    with pytest.raises(BackendStateError, match="requires rendezvous"):
        backend.initialize_rank()


def test_fsdp2_backend_multi_rank_initializes_process_group_and_fully_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    modules = _install_fake_fsdp2_modules(monkeypatch)
    backend = Fsdp2TrainerBackend(
        TrainerBackendConfig(
            backend="fsdp2",
            rank=1,
            world_size=2,
            gpu_id=1,
            group_epoch=0,
            rendezvous="env://",
            model_path="/fake/hf-model",
            checkpoint_dir=str(tmp_path),
        )
    )

    backend.initialize_rank()

    assert modules.init_process_group_calls == [
        {"backend": "nccl", "init_method": "env://", "rank": 1, "world_size": 2}
    ]
    assert modules.fully_shard_calls == [modules.model]


def test_fsdp2_backend_multi_rank_export_uses_full_state_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    modules = _install_fake_fsdp2_modules(monkeypatch)
    backend = Fsdp2TrainerBackend(
        TrainerBackendConfig(
            backend="fsdp2",
            rank=0,
            world_size=2,
            gpu_id=0,
            group_epoch=0,
            rendezvous="env://",
            model_path="/fake/hf-model",
            checkpoint_dir=str(tmp_path),
        )
    )

    backend.initialize_rank()
    exported = backend.export_weight(_weight(version_id=7))

    assert exported.version_id == 8
    assert modules.state_dict_calls
    assert modules.state_dict_calls[0]["model"] is modules.model
    assert modules.state_dict_calls[0]["options"].kwargs == {"full_state_dict": True, "cpu_offload": True}
    assert modules.model.saved_state_dicts[0]["model.layers.0.weight"].value == "full-weight"
    assert modules.model.saved_state_dicts[0]["model.layers.0.weight"].devices[-1] == "cpu"
    assert (tmp_path / "version-8" / "model.txt").exists()


def test_fsdp2_backend_non_publishing_rank_participates_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    modules = _install_fake_fsdp2_modules(monkeypatch)
    backend = Fsdp2TrainerBackend(
        TrainerBackendConfig(
            backend="fsdp2",
            rank=1,
            world_size=2,
            gpu_id=1,
            group_epoch=0,
            rendezvous="env://",
            model_path="/fake/hf-model",
            checkpoint_dir=str(tmp_path),
        )
    )

    backend.initialize_rank()
    participant = backend.export_weight(_weight(version_id=7), publish=False)

    assert participant.version_id == 8
    assert participant.created_by == "trainer-rank-1"
    assert participant.checksum == "abc123"
    assert modules.state_dict_calls
    assert modules.model.saved_dirs == []


def test_fsdp2_backend_requires_sample_record_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_fsdp2_modules(monkeypatch)
    backend = Fsdp2TrainerBackend(
        TrainerBackendConfig(
            backend="fsdp2",
            rank=0,
            world_size=1,
            gpu_id=0,
            group_epoch=0,
            model_path="/fake/hf-model",
            checkpoint_dir=str(tmp_path),
        )
    )

    with pytest.raises(BackendStateError, match=r"sample_refs\[\*\]\.object_ref"):
        backend.optimize(_batch(), lease=_lease(rank=0, gpu_id=0))


def test_fsdp2_backend_missing_import_raises_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from nano_rl.runtime.backends import fsdp2_backend

    def missing_import(name: str):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(fsdp2_backend, "import_module", missing_import)
    backend = Fsdp2TrainerBackend(
        TrainerBackendConfig(backend="fsdp2", rank=0, world_size=1, gpu_id=0, group_epoch=0)
    )

    with pytest.raises(BackendUnavailableError, match="PyTorch FSDP2 backend is unavailable"):
        backend.initialize_rank()
