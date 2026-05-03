"""Guarded PyTorch FSDP2 trainer backend boundary."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

from nano_rl.runtime.backends.trainer_backend import (
    BackendStateError,
    BackendUnavailableError,
    OptimizerStepResult,
    TrainerBackend,
    TrainerBackendConfig,
    TrainStateBundle,
    checkpoint_version_dir,
)
from nano_rl.runtime.protocols import SampleRecord, TrainBatch, WeightFormat, WeightMeta
from nano_rl.runtime.slot import GpuLease


class Fsdp2TrainerBackend(TrainerBackend):
    """Trainer backend boundary for one PyTorch FSDP2 rank.

    The class is import-safe on machines without PyTorch.  Methods that would
    touch PyTorch call ``_require_fsdp2`` first and translate dependency gaps
    into ``BackendUnavailableError``.
    """

    def __init__(self, config: TrainerBackendConfig):
        super().__init__(config)
        self._initialized = False
        self._hydrated = False
        self._train_step = 0
        self._weight_version: int | None = None
        self._torch_modules: dict[str, ModuleType] | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._optimizer: Any | None = None
        self._device: Any | None = None

    def initialize_rank(self) -> TrainStateBundle:
        modules = self._require_fsdp2()
        self._maybe_init_process_group(modules)
        self._ensure_model_and_optimizer(modules)
        self._initialized = True
        return self._state_bundle(residency="unloaded")

    def hydrate(self, state: TrainStateBundle | None = None, *, lease: GpuLease) -> TrainStateBundle:
        self._assert_trainer_lease(lease)
        modules = self._require_fsdp2()
        if not self._initialized:
            self.initialize_rank()
        if state is not None:
            self._assert_state_matches_rank(state)
            self._train_step = state.train_step
            self._weight_version = state.weight_version
        self._move_model_to_device(modules)
        self._hydrated = True
        return self._state_bundle(residency="cpu_standby")

    def optimize(self, batch: TrainBatch, *, lease: GpuLease) -> OptimizerStepResult:
        self._assert_trainer_lease(lease)
        modules = self._require_fsdp2()
        if not self._initialized:
            self.initialize_rank()
        if not self._hydrated:
            self.hydrate(None, lease=lease)

        model = self._require_model()
        tokenizer = self._require_tokenizer()
        optimizer = self._require_optimizer()
        device = self._require_device()
        texts = self._batch_texts(batch)
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.extra.get("max_length"),
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        else:
            encoded = {
                name: value.to(device) if hasattr(value, "to") else value
                for name, value in encoded.items()
                if value is not None
            }
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise BackendStateError("FSDP2 tokenizer output is missing input_ids")
        labels = input_ids.clone() if hasattr(input_ids, "clone") else input_ids
        outputs = model(**encoded, labels=labels)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()
        max_grad_norm = self.config.extra.get("max_grad_norm")
        if max_grad_norm is not None:
            modules["torch"].nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self._train_step += 1
        loss_value = float(loss.detach().cpu().item())
        return OptimizerStepResult(
            rank=self.config.rank,
            world_size=self.config.world_size,
            group_epoch=self.config.group_epoch,
            comm_epoch=self.config.comm_epoch,
            train_step=self._train_step,
            train_batch_id=batch.train_batch_id,
            num_sequences=batch.num_sequences,
            num_tokens=batch.num_tokens,
            loss=loss_value,
            lease_gpu_id=lease.gpu_id,
            lease_epoch=lease.lease_epoch,
            weight_version=self._weight_version,
            metrics={"lm_loss": loss_value},
        )

    def export_weight(self, parent: WeightMeta) -> WeightMeta:
        if self.config.rank != 0:
            raise BackendStateError(f"trainer rank {self.config.rank} cannot export weights; rank 0 owns publish")
        self._require_fsdp2()
        model = self._require_model()
        tokenizer = self._require_tokenizer()
        version_id = parent.version_id + 1
        output_dir = checkpoint_version_dir(self.config, version_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        checksum = _hash_checkpoint_dir(output_dir)
        self._weight_version = version_id
        return WeightMeta(
            version_id=version_id,
            parent_version=parent.version_id,
            trainer_step=self._train_step,
            created_at=datetime.utcnow(),
            model_path=str(output_dir),
            tokenizer_path=str(output_dir),
            artifact_uri=str(output_dir),
            format=WeightFormat.VLLM_COMPATIBLE,
            checksum=checksum,
            created_by=f"trainer-rank-{self.config.rank}",
        )

    def offload(self, *, lease: GpuLease) -> TrainStateBundle:
        self._assert_trainer_lease(lease)
        modules = self._require_fsdp2()
        if not self._initialized:
            raise BackendStateError(f"trainer rank {self.config.rank} has not been initialized")
        if self._model is not None:
            self._model.to("cpu")
        if modules["torch"].cuda.is_available():
            modules["torch"].cuda.empty_cache()
        self._hydrated = False
        return self._state_bundle(residency="cpu_standby")

    def _require_fsdp2(self) -> dict[str, ModuleType]:
        if self._torch_modules is not None:
            return self._torch_modules

        self._set_cuda_visible_devices()
        try:
            torch = import_module("torch")
            distributed = import_module("torch.distributed")
            fsdp_module = import_module("torch.distributed._composable.fsdp")
            transformers = import_module("transformers")
        except (ImportError, ModuleNotFoundError) as exc:
            raise BackendUnavailableError(
                "PyTorch FSDP2 backend is unavailable: could not import torch, transformers, or torch.distributed._composable.fsdp"
            ) from exc

        if not hasattr(distributed, "init_process_group"):
            raise BackendUnavailableError("PyTorch distributed is unavailable: missing init_process_group")
        if not hasattr(distributed, "is_initialized"):
            raise BackendUnavailableError("PyTorch distributed is unavailable: missing is_initialized")
        if not any(hasattr(fsdp_module, name) for name in ("fully_shard", "FSDPModule")):
            raise BackendUnavailableError(
                "PyTorch FSDP2 backend is unavailable: missing fully_shard/FSDPModule entrypoint"
            )

        self._torch_modules = {
            "torch": torch,
            "distributed": distributed,
            "fsdp": fsdp_module,
            "transformers": transformers,
        }
        return self._torch_modules

    def _maybe_init_process_group(self, modules: dict[str, ModuleType]) -> None:
        if self.config.world_size <= 1:
            return
        distributed = modules["distributed"]
        if distributed.is_initialized():
            return
        if not self.config.rendezvous and not self.config.store_endpoint:
            raise BackendStateError(
                "multi-rank FSDP2 initialization requires rendezvous or store_endpoint"
            )
        init_method = self.config.rendezvous
        if init_method is None and self.config.store_endpoint:
            init_method = f"tcp://{self.config.store_endpoint}"
        distributed.init_process_group(
            backend=str(self.config.extra.get("dist_backend", "nccl")),
            init_method=init_method,
            rank=self.config.rank,
            world_size=self.config.world_size,
        )

    def _ensure_model_and_optimizer(self, modules: dict[str, ModuleType]) -> None:
        if self._model is not None and self._optimizer is not None:
            return
        model_path = self.config.model_path
        if not model_path:
            raise BackendStateError("FSDP2 trainer backend requires model_path")

        transformers = modules["transformers"]
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=bool(self.config.extra.get("trust_remote_code", False)),
            torch_dtype=self._torch_dtype(modules),
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=bool(self.config.extra.get("trust_remote_code", False)),
        )
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if self.config.world_size > 1:
            fully_shard = getattr(modules["fsdp"], "fully_shard", None)
            if fully_shard is None:
                raise BackendUnavailableError("PyTorch FSDP2 backend is unavailable: missing fully_shard")
            maybe_model = fully_shard(model)
            if maybe_model is not None:
                model = maybe_model

        self._model = model
        self._tokenizer = tokenizer
        self._move_model_to_device(modules)
        self._optimizer = self._build_optimizer(modules)

    def _build_optimizer(self, modules: dict[str, ModuleType]) -> Any:
        optimizer_name = self.config.optimizer_name.lower()
        if optimizer_name not in {"adamw", "adamw_torch"}:
            raise BackendStateError(f"unsupported trainer optimizer: {self.config.optimizer_name}")
        return modules["torch"].optim.AdamW(
            self._require_model().parameters(),
            lr=float(self.config.extra.get("learning_rate", 1.0e-6)),
            weight_decay=float(self.config.extra.get("weight_decay", 0.0)),
        )

    def _move_model_to_device(self, modules: dict[str, ModuleType]) -> None:
        if self._model is None:
            self._ensure_model_and_optimizer(modules)
            return
        torch = modules["torch"]
        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)

    def _torch_dtype(self, modules: dict[str, ModuleType]) -> Any:
        dtype = self.config.extra.get("mixed_precision")
        torch = modules["torch"]
        if dtype == "bf16":
            return torch.bfloat16
        if dtype == "fp16":
            return torch.float16
        return None

    def _batch_texts(self, batch: TrainBatch) -> list[str]:
        texts: list[str] = []
        for ref in batch.sample_refs:
            record = _sample_record_from_ref(ref.object_ref)
            if record is None:
                raise BackendStateError(
                    "FSDP2 optimize requires TrainBatch.sample_refs[*].object_ref to contain SampleRecord payloads"
                )
            texts.append(f"{record.prompt}\n{record.response}")
        if not texts:
            raise BackendStateError("FSDP2 optimize requires a non-empty TrainBatch")
        return texts

    def _require_model(self) -> Any:
        if self._model is None:
            raise BackendStateError(f"trainer rank {self.config.rank} has no initialized model")
        return self._model

    def _require_tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise BackendStateError(f"trainer rank {self.config.rank} has no initialized tokenizer")
        return self._tokenizer

    def _require_optimizer(self) -> Any:
        if self._optimizer is None:
            raise BackendStateError(f"trainer rank {self.config.rank} has no initialized optimizer")
        return self._optimizer

    def _require_device(self) -> Any:
        if self._device is None:
            raise BackendStateError(f"trainer rank {self.config.rank} has no initialized device")
        return self._device

    def _state_bundle(self, *, residency: str) -> TrainStateBundle:
        return TrainStateBundle(
            rank=self.config.rank,
            world_size=self.config.world_size,
            group_epoch=self.config.group_epoch,
            comm_epoch=self.config.comm_epoch,
            train_step=self._train_step,
            weight_version=self._weight_version,
            residency=residency,  # type: ignore[arg-type]
            metadata={
                "backend": "fsdp2",
                "initialized": self._initialized,
                "hydrated": self._hydrated,
                "process_group": self.config.process_group.model_dump(mode="json"),
                "model_path": self.config.model_path,
                "optimizer_name": self.config.optimizer_name,
                "checkpoint_dir": self.config.checkpoint_dir,
            },
        )

    def _assert_state_matches_rank(self, state: TrainStateBundle) -> None:
        expected: dict[str, Any] = {
            "rank": self.config.rank,
            "world_size": self.config.world_size,
            "group_epoch": self.config.group_epoch,
            "comm_epoch": self.config.comm_epoch,
        }
        for field_name, expected_value in expected.items():
            actual = getattr(state, field_name)
            if actual != expected_value:
                raise BackendStateError(
                    f"state {field_name} {actual} does not match backend {field_name} {expected_value}"
                )


def _sample_record_from_ref(value: Any) -> SampleRecord | None:
    payload = _resolve_sample_ref_payload(value)
    if isinstance(payload, SampleRecord):
        return payload
    if isinstance(payload, Mapping):
        try:
            return SampleRecord.model_validate(payload)
        except Exception as exc:
            raise BackendStateError("FSDP2 optimize received an invalid SampleRecord payload") from exc
    return None


def _resolve_sample_ref_payload(value: Any) -> Any:
    if value is None or isinstance(value, SampleRecord) or isinstance(value, Mapping):
        return value

    try:
        ray = import_module("ray")
    except (ImportError, ModuleNotFoundError):
        return value

    object_ref_type = getattr(ray, "ObjectRef", None)
    is_object_ref = False
    if object_ref_type is not None:
        try:
            is_object_ref = isinstance(value, object_ref_type)
        except TypeError:
            is_object_ref = False
    if not is_object_ref:
        return value

    try:
        return ray.get(value)
    except Exception as exc:
        raise BackendStateError("FSDP2 optimize failed to materialize SampleRecord object_ref") from exc


def _hash_checkpoint_dir(path: Path) -> str:
    digest = sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        digest.update(str(item.relative_to(path)).encode())
        digest.update(str(item.stat().st_size).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
