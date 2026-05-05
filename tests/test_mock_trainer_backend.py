"""Tests for the deterministic mock trainer backend."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Literal

import pytest

from nano_rl.runtime.backends.trainer_backend import (
    BackendStateError,
    FakeTrainerBackend,
    MockTrainerBackend,
    TrainerBackendConfig,
    build_trainer_backend,
)
from nano_rl.runtime.protocols import TrainBatch, WeightFormat, WeightMeta
from nano_rl.runtime.slot import GpuLease, RoleName


def _config(
    *,
    backend: Literal["mock", "fake"] = "mock",
    rank: int = 0,
    gpu_id: int = 4,
    world_size: int = 2,
) -> TrainerBackendConfig:
    return TrainerBackendConfig(
        backend=backend,
        rank=rank,
        world_size=world_size,
        gpu_id=gpu_id,
        group_epoch=3,
        rendezvous="env://",
        store_endpoint="127.0.0.1:29500",
        comm_epoch=9,
    )


def _lease(*, rank: int = 0, gpu_id: int = 4) -> GpuLease:
    return GpuLease(
        gpu_id=gpu_id,
        role=RoleName.TRAINER,
        holder_id=f"trainer-rank-{rank}",
        lease_epoch=11,
    )


def _batch() -> TrainBatch:
    return TrainBatch(
        train_batch_id="batch-1",
        sample_refs=[],
        sample_ids=["sample-0", "sample-1"],
        policy_version_min=7,
        policy_version_max=7,
        policy_version_histogram={7: 2},
        num_sequences=2,
        num_tokens=17,
        reserved_by="trainer-rank-0",
        reserved_at=datetime(2026, 1, 1),
    )


def _weight(version_id: int = 5) -> WeightMeta:
    return WeightMeta(
        version_id=version_id,
        created_at=datetime(2026, 1, 1),
        model_path="/mock/model",
        format=WeightFormat.HF,
        checksum="parent-checksum",
    )


def test_mock_backend_lifecycle_metadata_loss_and_rank0_export() -> None:
    backend = MockTrainerBackend(_config())

    initialized = backend.initialize_rank()
    hydrated = backend.hydrate(initialized, lease=_lease())
    result = backend.optimize(_batch(), lease=_lease())
    offloaded = backend.offload(lease=_lease())
    exported = backend.export_weight(_weight())

    assert initialized.metadata["backend"] == "mock"
    assert hydrated.metadata["backend"] == "mock"
    assert hydrated.metadata["requested_backend"] == "mock"
    assert hydrated.metadata["compat_aliases"] == ["fake"]
    assert hydrated.metadata["process_group"]["rank"] == 0
    assert hydrated.metadata["process_group"]["world_size"] == 2
    assert hydrated.metadata["process_group"]["group_epoch"] == 3
    assert hydrated.metadata["process_group"]["comm_epoch"] == 9
    assert result.loss == pytest.approx(1 / 3)
    assert result.metrics["mock_loss"] == pytest.approx(result.loss)
    assert result.metrics["fake_loss"] == pytest.approx(result.loss)
    assert offloaded.train_step == 1
    assert offloaded.residency == "cpu_standby"
    assert exported.version_id == 6
    assert exported.parent_version == 5
    assert exported.trainer_step == 1
    assert exported.created_by == "trainer-rank-0"
    assert exported.format == WeightFormat.VLLM_COMPATIBLE
    assert exported.checksum == sha256("parent-checksum:6:1".encode()).hexdigest()


def test_mock_backend_is_deterministic_across_instances() -> None:
    first = MockTrainerBackend(_config())
    second = MockTrainerBackend(_config())

    first_result = first.optimize(_batch(), lease=_lease())
    second_result = second.optimize(_batch(), lease=_lease())
    first_export = first.export_weight(_weight())
    second_export = second.export_weight(_weight())

    assert first_result.loss == second_result.loss
    assert first_result.metrics == second_result.metrics
    assert first_export.checksum == second_export.checksum


def test_fake_backend_alias_builds_mock_backend_with_compat_metadata() -> None:
    assert FakeTrainerBackend is MockTrainerBackend

    backend = build_trainer_backend(_config(backend="fake"))
    state = backend.initialize_rank()

    assert isinstance(backend, MockTrainerBackend)
    assert state.metadata["backend"] == "mock"
    assert state.metadata["requested_backend"] == "fake"
    assert state.metadata["compat_aliases"] == ["fake"]


def test_mock_backend_rejects_non_rank0_export() -> None:
    backend = MockTrainerBackend(_config(rank=1, gpu_id=5))
    backend.optimize(_batch(), lease=_lease(rank=1, gpu_id=5))

    with pytest.raises(BackendStateError, match="rank 1 cannot export"):
        backend.export_weight(_weight())
