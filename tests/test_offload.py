from __future__ import annotations

import pytest

from nano_rl.config import load_launch_config
from nano_rl.exceptions import GpuRoleUnsupported, SlotStateError
from nano_rl.runtime.offload import (
    GpuResidencyManagerCore,
    OffloadEventType,
    ResidencyHookContext,
    ResidencyHookReport,
    ResidencyOperation,
    ResidencyState,
    cpu_pinned_trainer_strategy,
    vllm_sleep_strategy,
)
from nano_rl.runtime.slot import GpuLease, GpuLeaseManagerCore, RoleName


def _lease_manager() -> GpuLeaseManagerCore:
    config = load_launch_config("docs/examples/disaggregated.yaml")
    return GpuLeaseManagerCore(config.gpu_plan)


def _residency_manager(lease_manager: GpuLeaseManagerCore, **kwargs: object) -> GpuResidencyManagerCore:
    return GpuResidencyManagerCore(
        lease_manager,
        offload_timeout_sec=5,
        hydrate_timeout_sec=5,
        residual_gpu_memory_budget_mb=2048,
        **kwargs,
    )


def test_shared_gpu_enters_and_exits_train_window() -> None:
    lease_manager = _lease_manager()
    seen: list[tuple[str, ResidencyOperation, RoleName, int]] = []

    def record(context: ResidencyHookContext) -> ResidencyHookReport:
        seen.append((context.strategy, context.operation, context.role, context.lease_epoch))
        return ResidencyHookReport(duration_sec=0.1, residual_gpu_memory_mb=128)

    manager = _residency_manager(
        lease_manager,
        rollout_strategy=vllm_sleep_strategy(offload_hook=record, hydrate_hook=record),
        trainer_strategy=cpu_pinned_trainer_strategy(offload_hook=record, hydrate_hook=record),
    )

    rollout_lease = lease_manager.current_lease(RoleName.ROLLOUT, 4, holder_id="rollout-dp-2-tp-0")
    enter = manager.enter_train(rollout_lease, trainer_holder_id="trainer-rank-0")

    assert enter.success is True
    assert enter.lease == GpuLease(gpu_id=4, role=RoleName.TRAINER, holder_id="trainer-rank-0", lease_epoch=1)
    assert enter.state == ResidencyState.TRAINER_ACTIVE
    lease_manager.assert_active(RoleName.TRAINER, 4, holder_id="trainer-rank-0", expected_epoch=1)
    assert manager.get_state(4) == ResidencyState.TRAINER_ACTIVE

    exit_result = manager.exit_train(enter.lease, rollout_holder_id="rollout-dp-2-tp-0")

    assert exit_result.success is True
    assert exit_result.lease == GpuLease(gpu_id=4, role=RoleName.ROLLOUT, holder_id="rollout-dp-2-tp-0", lease_epoch=2)
    assert exit_result.state == ResidencyState.ROLLOUT_ACTIVE
    lease_manager.assert_active(RoleName.ROLLOUT, 4, holder_id="rollout-dp-2-tp-0", expected_epoch=2)
    assert seen == [
        ("vllm_sleep", ResidencyOperation.OFFLOAD, RoleName.ROLLOUT, 0),
        ("cpu_pinned", ResidencyOperation.HYDRATE, RoleName.TRAINER, 1),
        ("cpu_pinned", ResidencyOperation.OFFLOAD, RoleName.TRAINER, 1),
        ("vllm_sleep", ResidencyOperation.HYDRATE, RoleName.ROLLOUT, 2),
    ]


def test_rollout_only_gpu_rejects_trainer_transition() -> None:
    lease_manager = _lease_manager()
    manager = _residency_manager(lease_manager)
    rollout_lease = lease_manager.current_lease(RoleName.ROLLOUT, 0, holder_id="rollout-dp-0-tp-0")

    with pytest.raises(GpuRoleUnsupported):
        manager.enter_train(rollout_lease, trainer_holder_id="trainer-rank-x")


def test_wrong_epoch_rejects_before_strategy_hook_runs() -> None:
    lease_manager = _lease_manager()
    called = False

    def fail_if_called(context: ResidencyHookContext) -> ResidencyHookReport:
        nonlocal called
        called = True
        return ResidencyHookReport()

    manager = _residency_manager(
        lease_manager,
        rollout_strategy=vllm_sleep_strategy(offload_hook=fail_if_called),
    )
    stale_lease = GpuLease(gpu_id=4, role=RoleName.ROLLOUT, holder_id="rollout-dp-2-tp-0", lease_epoch=99)

    with pytest.raises(SlotStateError, match="expected 99"):
        manager.enter_train(stale_lease, trainer_holder_id="trainer-rank-0")

    assert called is False
    assert manager.get_state(4) == ResidencyState.ROLLOUT_ACTIVE


def test_strategy_hook_failure_returns_structured_reason_and_degrades() -> None:
    lease_manager = _lease_manager()

    def offload_fails(context: ResidencyHookContext) -> ResidencyHookReport:
        return ResidencyHookReport(success=False, reason=f"{context.strategy} refused sleep")

    manager = _residency_manager(
        lease_manager,
        rollout_strategy=vllm_sleep_strategy(offload_hook=offload_fails),
    )
    rollout_lease = lease_manager.current_lease(RoleName.ROLLOUT, 4, holder_id="rollout-dp-2-tp-0")

    result = manager.enter_train(rollout_lease, trainer_holder_id="trainer-rank-0")

    assert result.success is False
    assert result.event_type == OffloadEventType.SHARED_GPU_OFFLOAD_FAILED
    assert result.reason == "vllm_sleep refused sleep"
    assert result.state == ResidencyState.DEGRADED
    assert manager.get_state(4) == ResidencyState.DEGRADED
    lease_manager.assert_active(RoleName.ROLLOUT, 4, holder_id="rollout-dp-2-tp-0", expected_epoch=0)


def test_timeout_is_reported_as_toggle_timeout() -> None:
    lease_manager = _lease_manager()

    def slow_hydrate(context: ResidencyHookContext) -> ResidencyHookReport:
        return ResidencyHookReport(duration_sec=6)

    manager = _residency_manager(
        lease_manager,
        trainer_strategy=cpu_pinned_trainer_strategy(hydrate_hook=slow_hydrate),
    )
    rollout_lease = lease_manager.current_lease(RoleName.ROLLOUT, 4, holder_id="rollout-dp-2-tp-0")

    result = manager.enter_train(rollout_lease, trainer_holder_id="trainer-rank-0")

    assert result.success is False
    assert result.event_type == OffloadEventType.SHARED_GPU_TOGGLE_TIMEOUT
    assert "budget 5.000s" in result.reason
    assert result.state == ResidencyState.DEGRADED
