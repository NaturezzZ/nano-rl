from __future__ import annotations

from datetime import datetime

import pytest

from nano_rl.runtime.backends.mock_rollout_backend import (
    MockRolloutBackend,
    MockRolloutBackendConfig,
    MockRolloutBackendError,
    build_mock_rollout_backend,
)
from nano_rl.runtime.backends.vllm_backend import VllmBackendConfig
from nano_rl.runtime.protocols import WeightFormat, WeightMeta, WeightShardSource, WeightShardSourceKind
from nano_rl.runtime.slot import GpuLease, RoleName


def test_mock_rollout_activate_and_generate_is_deterministic() -> None:
    backend = build_mock_rollout_backend(
        {
            "gpu_ids": (0,),
            "holder_id": "rollout-dp-0-tp-0",
            "sampling_params": {"temperature": 0.0, "max_tokens": 4},
        }
    )
    weight = _weight(version_id=7, checksum="abc123")
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="rollout-dp-0-tp-0", lease_epoch=3)

    backend.activate_weight(weight, lease=lease)
    first = backend.generate(
        prompt="hello",
        target_policy_version=7,
        request_metadata={"controller_step": 11},
        request_id="req-1",
        lease=lease,
    )
    second = backend.generate(
        prompt="hello",
        target_policy_version=7,
        request_metadata={"controller_step": 11},
        request_id="req-1",
        lease=lease,
    )

    assert backend.active_weight == weight
    assert backend.activation_count == 1
    assert first == second
    assert first.request_id == "req-1"
    assert first.prompt == "hello"
    assert first.response.startswith("mock[v7:")
    assert first.response.endswith("] hello")
    assert len(first.tokens) == 4
    assert len(first.logprobs) == 4
    assert first.old_logprobs == first.logprobs
    assert first.finish_reason == "stop"
    assert first.policy_version == 7
    assert first.weight_checksum == "abc123"
    assert first.metadata["controller_step"] == 11
    assert first.metadata["mock_rollout"]["backend"] == "mock_rollout"
    assert first.metadata["mock_rollout"]["active_weight_version"] == 7
    assert first.metadata["mock_rollout"]["gpu_ids"] == [0]
    assert first.metadata["mock_rollout"]["holder_ids"] == ["rollout-dp-0-tp-0"]
    assert first.metadata["mock_rollout"]["lease_epochs"] == {"0": 3}


def test_mock_rollout_requires_activate_before_generate() -> None:
    backend = build_mock_rollout_backend({"gpu_ids": (0,), "holder_id": "worker-0"})
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=1)

    with pytest.raises(MockRolloutBackendError, match="cannot generate before activate_weight"):
        backend.generate(prompt="x", target_policy_version=1, request_metadata={}, lease=lease)


def test_mock_rollout_requires_active_target_policy_version() -> None:
    backend = build_mock_rollout_backend({"gpu_ids": (0,), "holder_id": "worker-0"})
    lease = GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=1)
    backend.activate_weight(_weight(version_id=3), lease=lease)

    with pytest.raises(MockRolloutBackendError, match="does not match target"):
        backend.generate(prompt="x", target_policy_version=4, request_metadata={}, lease=lease)


def test_mock_rollout_validates_rollout_gpu_and_holder_lease() -> None:
    backend = build_mock_rollout_backend({"gpu_ids": (0,), "holder_id": "worker-0"})
    weight = _weight(version_id=1)

    with pytest.raises(Exception, match="not rollout"):
        backend.activate_weight(
            weight,
            lease=GpuLease(gpu_id=0, role=RoleName.TRAINER, holder_id="worker-0", lease_epoch=1),
        )
    with pytest.raises(Exception, match="expected gpu leases"):
        backend.activate_weight(
            weight,
            lease=GpuLease(gpu_id=1, role=RoleName.ROLLOUT, holder_id="worker-0", lease_epoch=1),
        )
    with pytest.raises(Exception, match="expected worker-0"):
        backend.activate_weight(
            weight,
            lease=GpuLease(gpu_id=0, role=RoleName.ROLLOUT, holder_id="other", lease_epoch=1),
        )


def test_mock_rollout_validates_multi_gpu_holder_ids() -> None:
    backend = build_mock_rollout_backend(
        {
            "tensor_parallel_size": 2,
            "gpu_ids": (4, 6),
            "holder_ids": ("rollout-dp-1-tp-0", "rollout-dp-1-tp-1"),
        }
    )
    leases = (
        GpuLease(gpu_id=4, role=RoleName.ROLLOUT, holder_id="rollout-dp-1-tp-0", lease_epoch=2),
        GpuLease(gpu_id=6, role=RoleName.ROLLOUT, holder_id="rollout-dp-1-tp-1", lease_epoch=3),
    )
    swapped_holders = (
        GpuLease(gpu_id=4, role=RoleName.ROLLOUT, holder_id="rollout-dp-1-tp-1", lease_epoch=2),
        GpuLease(gpu_id=6, role=RoleName.ROLLOUT, holder_id="rollout-dp-1-tp-0", lease_epoch=3),
    )

    backend.activate_weight(_weight(version_id=8), lease=leases)
    output = backend.generate(prompt="tp", target_policy_version=8, request_metadata={}, lease=leases)

    assert output.policy_version == 8
    assert output.metadata["mock_rollout"]["gpu_ids"] == [4, 6]
    with pytest.raises(Exception, match="lease holder for gpu 4"):
        backend.generate(prompt="bad", target_policy_version=8, request_metadata={}, lease=swapped_holders)


def test_mock_rollout_tracks_weight_transfer_source() -> None:
    backend = build_mock_rollout_backend({"gpu_ids": (4,), "holder_id": "worker-4"})
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


def test_mock_rollout_config_accepts_vllm_backend_config_without_importing_vllm() -> None:
    backend = MockRolloutBackend(
        VllmBackendConfig(
            tensor_parallel_size=2,
            gpu_ids=(2, 3),
            holder_ids=("worker-2", "worker-3"),
            sampling_params={"max_tokens": 2},
        )
    )
    leases = (
        GpuLease(gpu_id=2, role=RoleName.ROLLOUT, holder_id="worker-2", lease_epoch=1),
        GpuLease(gpu_id=3, role=RoleName.ROLLOUT, holder_id="worker-3", lease_epoch=1),
    )

    backend.activate_weight(_weight(version_id=2), lease=leases)
    output = backend.generate(prompt="compat", target_policy_version=2, request_metadata={}, lease=leases)

    assert output.response.startswith("mock[v2:")
    assert output.tokens and len(output.tokens) == 2


def test_mock_rollout_rejects_mismatched_holder_ids_length() -> None:
    with pytest.raises(ValueError, match="holder_ids length must match gpu_ids length"):
        MockRolloutBackendConfig.model_validate(
            {
                "tensor_parallel_size": 2,
                "gpu_ids": (0, 1),
                "holder_ids": ("only-one-holder",),
            }
        )


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
