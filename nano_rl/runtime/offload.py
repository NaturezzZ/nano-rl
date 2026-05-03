"""CPU-only shared GPU offload and hydrate state machine.

This module is the executable boundary between the lease manager and concrete
CUDA backends.  Strategies expose hooks for vLLM/FSDP2 integrations, while the
core remains deterministic and testable without touching CUDA.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from nano_rl.exceptions import GpuRoleUnsupported, SlotStateError
from nano_rl.runtime.slot import GpuLease, GpuLeaseManagerCore, GpuTopology, RoleName


class ResidencyState(StrEnum):
    ROLLOUT_ACTIVE = "ROLLOUT_ACTIVE"
    ROLLOUT_DRAINING = "ROLLOUT_DRAINING"
    ROLLOUT_OFFLOADED = "ROLLOUT_OFFLOADED"
    TRAINER_HYDRATING = "TRAINER_HYDRATING"
    TRAINER_ACTIVE = "TRAINER_ACTIVE"
    TRAINER_OFFLOADED = "TRAINER_OFFLOADED"
    ROLLOUT_WAKING = "ROLLOUT_WAKING"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class ResidencyTransition(StrEnum):
    ENTER_TRAIN = "ENTER_TRAIN"
    ROLLOUT_OFFLOADED = "ROLLOUT_OFFLOADED"
    TRAINER_HYDRATING = "TRAINER_HYDRATING"
    TRAINER_HYDRATED = "TRAINER_HYDRATED"
    EXIT_TRAIN = "EXIT_TRAIN"
    TRAINER_OFFLOADED = "TRAINER_OFFLOADED"
    ROLLOUT_HYDRATED = "ROLLOUT_HYDRATED"


class OffloadEventType(StrEnum):
    SHARED_GPU_OFFLOAD_FAILED = "SHARED_GPU_OFFLOAD_FAILED"
    SHARED_GPU_HYDRATE_FAILED = "SHARED_GPU_HYDRATE_FAILED"
    SHARED_GPU_TOGGLE_TIMEOUT = "SHARED_GPU_TOGGLE_TIMEOUT"


class ResidencyHookReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool = True
    reason: str | None = None
    duration_sec: float = Field(default=0, ge=0)
    residual_gpu_memory_mb: int = Field(default=0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class ResidencyTransitionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    transition: ResidencyTransition
    from_state: ResidencyState
    to_state: ResidencyState
    gpu_id: int = Field(ge=0)
    role: RoleName
    holder_id: str
    lease_epoch: int = Field(ge=0)


class _ResidencyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    gpu_id: int = Field(ge=0)
    role: RoleName
    holder_id: str
    lease_epoch: int = Field(ge=0)
    state: ResidencyState
    strategy: str
    reason: str | None = None
    event_type: OffloadEventType | None = None
    duration_sec: float = Field(default=0, ge=0)
    residual_gpu_memory_mb: int = Field(default=0, ge=0)
    lease: GpuLease | None = None
    transitions: tuple[ResidencyTransitionRecord, ...] = ()
    metadata: dict[str, object] = Field(default_factory=dict)


class OffloadResult(_ResidencyResult):
    pass


class HydrateResult(_ResidencyResult):
    pass


class ResidencyOperation(StrEnum):
    OFFLOAD = "offload"
    HYDRATE = "hydrate"


class ResidencyHookContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpu_id: int = Field(ge=0)
    role: RoleName
    holder_id: str
    lease_epoch: int = Field(ge=0)
    operation: ResidencyOperation
    strategy: str


ResidencyHook = Callable[[ResidencyHookContext], ResidencyHookReport]


class GpuResidencyStrategy(Protocol):
    @property
    def name(self) -> str: ...

    def offload(self, context: ResidencyHookContext) -> ResidencyHookReport: ...

    def hydrate(self, context: ResidencyHookContext) -> ResidencyHookReport: ...


class HookedGpuResidencyStrategy:
    """Named residency strategy with optional backend hooks."""

    def __init__(
        self,
        name: str,
        *,
        offload_hook: ResidencyHook | None = None,
        hydrate_hook: ResidencyHook | None = None,
    ) -> None:
        self._name = name
        self._offload_hook = offload_hook
        self._hydrate_hook = hydrate_hook

    @property
    def name(self) -> str:
        return self._name

    def offload(self, context: ResidencyHookContext) -> ResidencyHookReport:
        if self._offload_hook is None:
            return ResidencyHookReport()
        return self._offload_hook(context)

    def hydrate(self, context: ResidencyHookContext) -> ResidencyHookReport:
        if self._hydrate_hook is None:
            return ResidencyHookReport()
        return self._hydrate_hook(context)


def vllm_sleep_strategy(
    *,
    offload_hook: ResidencyHook | None = None,
    hydrate_hook: ResidencyHook | None = None,
) -> GpuResidencyStrategy:
    return HookedGpuResidencyStrategy("vllm_sleep", offload_hook=offload_hook, hydrate_hook=hydrate_hook)


def teardown_and_reload_strategy(
    *,
    offload_hook: ResidencyHook | None = None,
    hydrate_hook: ResidencyHook | None = None,
) -> GpuResidencyStrategy:
    return HookedGpuResidencyStrategy("teardown_and_reload", offload_hook=offload_hook, hydrate_hook=hydrate_hook)


def cpu_pinned_trainer_strategy(
    *,
    offload_hook: ResidencyHook | None = None,
    hydrate_hook: ResidencyHook | None = None,
) -> GpuResidencyStrategy:
    return HookedGpuResidencyStrategy("cpu_pinned", offload_hook=offload_hook, hydrate_hook=hydrate_hook)


class GpuResidencyManagerCore:
    """Lease-aware shared GPU offload/hydrate state machine."""

    def __init__(
        self,
        lease_manager: GpuLeaseManagerCore,
        *,
        rollout_strategy: GpuResidencyStrategy | None = None,
        trainer_strategy: GpuResidencyStrategy | None = None,
        offload_timeout_sec: float,
        hydrate_timeout_sec: float,
        residual_gpu_memory_budget_mb: int,
    ) -> None:
        self.lease_manager = lease_manager
        self.rollout_strategy = rollout_strategy or vllm_sleep_strategy()
        self.trainer_strategy = trainer_strategy or cpu_pinned_trainer_strategy()
        self.offload_timeout_sec = offload_timeout_sec
        self.hydrate_timeout_sec = hydrate_timeout_sec
        self.residual_gpu_memory_budget_mb = residual_gpu_memory_budget_mb
        self._states = {
            gpu_id: ResidencyState.ROLLOUT_ACTIVE
            for gpu_id in lease_manager.plan.shared_gpu_ids
        }

    def get_state(self, gpu_id: int) -> ResidencyState:
        self._assert_shared_gpu(gpu_id)
        return self._states[gpu_id]

    def enter_train(self, rollout_lease: GpuLease, *, trainer_holder_id: str) -> HydrateResult:
        self._assert_current_lease(rollout_lease, expected_role=RoleName.ROLLOUT)
        self._assert_state(rollout_lease.gpu_id, ResidencyState.ROLLOUT_ACTIVE)

        transitions: list[ResidencyTransitionRecord] = []
        self._move(
            transitions,
            ResidencyTransition.ENTER_TRAIN,
            rollout_lease,
            ResidencyState.ROLLOUT_ACTIVE,
            ResidencyState.ROLLOUT_DRAINING,
        )

        offload = self._run_offload(self.rollout_strategy, rollout_lease)
        if not offload.success:
            return self._hydrate_failure_from_offload(offload, transitions)

        self._move(
            transitions,
            ResidencyTransition.ROLLOUT_OFFLOADED,
            rollout_lease,
            ResidencyState.ROLLOUT_DRAINING,
            ResidencyState.ROLLOUT_OFFLOADED,
        )

        trainer_lease = self.lease_manager.grant(
            RoleName.TRAINER,
            rollout_lease.gpu_id,
            trainer_holder_id,
            reason="rollout_offloaded",
        )
        self._move(
            transitions,
            ResidencyTransition.TRAINER_HYDRATING,
            trainer_lease,
            ResidencyState.ROLLOUT_OFFLOADED,
            ResidencyState.TRAINER_HYDRATING,
        )

        hydrate = self._run_hydrate(self.trainer_strategy, trainer_lease)
        if not hydrate.success:
            self._states[trainer_lease.gpu_id] = ResidencyState.DEGRADED
            return HydrateResult(
                success=False,
                gpu_id=trainer_lease.gpu_id,
                role=trainer_lease.role,
                holder_id=trainer_lease.holder_id,
                lease_epoch=trainer_lease.lease_epoch,
                state=ResidencyState.DEGRADED,
                strategy=self.trainer_strategy.name,
                reason=hydrate.reason,
                event_type=hydrate.event_type,
                duration_sec=hydrate.duration_sec,
                residual_gpu_memory_mb=hydrate.residual_gpu_memory_mb,
                lease=trainer_lease,
                transitions=tuple(transitions),
                metadata=hydrate.metadata,
            )

        self._move(
            transitions,
            ResidencyTransition.TRAINER_HYDRATED,
            trainer_lease,
            ResidencyState.TRAINER_HYDRATING,
            ResidencyState.TRAINER_ACTIVE,
        )
        return HydrateResult(
            success=True,
            gpu_id=trainer_lease.gpu_id,
            role=trainer_lease.role,
            holder_id=trainer_lease.holder_id,
            lease_epoch=trainer_lease.lease_epoch,
            state=ResidencyState.TRAINER_ACTIVE,
            strategy=self.trainer_strategy.name,
            duration_sec=hydrate.duration_sec,
            residual_gpu_memory_mb=hydrate.residual_gpu_memory_mb,
            lease=trainer_lease,
            transitions=tuple(transitions),
            metadata=hydrate.metadata,
        )

    def exit_train(self, trainer_lease: GpuLease, *, rollout_holder_id: str) -> HydrateResult:
        self._assert_current_lease(trainer_lease, expected_role=RoleName.TRAINER)
        self._assert_state(trainer_lease.gpu_id, ResidencyState.TRAINER_ACTIVE)

        transitions: list[ResidencyTransitionRecord] = []
        self._move(
            transitions,
            ResidencyTransition.EXIT_TRAIN,
            trainer_lease,
            ResidencyState.TRAINER_ACTIVE,
            ResidencyState.TRAINER_OFFLOADED,
        )

        offload = self._run_offload(self.trainer_strategy, trainer_lease)
        if not offload.success:
            return self._hydrate_failure_from_offload(offload, transitions)

        rollout_lease = self.lease_manager.grant(
            RoleName.ROLLOUT,
            trainer_lease.gpu_id,
            rollout_holder_id,
            reason="trainer_offloaded",
        )
        self._move(
            transitions,
            ResidencyTransition.TRAINER_OFFLOADED,
            rollout_lease,
            ResidencyState.TRAINER_OFFLOADED,
            ResidencyState.ROLLOUT_WAKING,
        )

        hydrate = self._run_hydrate(self.rollout_strategy, rollout_lease)
        if not hydrate.success:
            self._states[rollout_lease.gpu_id] = ResidencyState.DEGRADED
            return HydrateResult(
                success=False,
                gpu_id=rollout_lease.gpu_id,
                role=rollout_lease.role,
                holder_id=rollout_lease.holder_id,
                lease_epoch=rollout_lease.lease_epoch,
                state=ResidencyState.DEGRADED,
                strategy=self.rollout_strategy.name,
                reason=hydrate.reason,
                event_type=hydrate.event_type,
                duration_sec=hydrate.duration_sec,
                residual_gpu_memory_mb=hydrate.residual_gpu_memory_mb,
                lease=rollout_lease,
                transitions=tuple(transitions),
                metadata=hydrate.metadata,
            )

        self._move(
            transitions,
            ResidencyTransition.ROLLOUT_HYDRATED,
            rollout_lease,
            ResidencyState.ROLLOUT_WAKING,
            ResidencyState.ROLLOUT_ACTIVE,
        )
        return HydrateResult(
            success=True,
            gpu_id=rollout_lease.gpu_id,
            role=rollout_lease.role,
            holder_id=rollout_lease.holder_id,
            lease_epoch=rollout_lease.lease_epoch,
            state=ResidencyState.ROLLOUT_ACTIVE,
            strategy=self.rollout_strategy.name,
            duration_sec=hydrate.duration_sec,
            residual_gpu_memory_mb=hydrate.residual_gpu_memory_mb,
            lease=rollout_lease,
            transitions=tuple(transitions),
            metadata=hydrate.metadata,
        )

    def _run_offload(self, strategy: GpuResidencyStrategy, lease: GpuLease) -> OffloadResult:
        try:
            report = strategy.offload(self._context(lease, ResidencyOperation.OFFLOAD, strategy.name))
        except Exception as exc:  # backend exceptions are converted into structured runtime events.
            report = ResidencyHookReport(success=False, reason=str(exc))
        event_type, reason = self._check_report(report, operation=ResidencyOperation.OFFLOAD)
        if event_type is not None:
            self._states[lease.gpu_id] = ResidencyState.DEGRADED
        return OffloadResult(
            success=event_type is None,
            gpu_id=lease.gpu_id,
            role=lease.role,
            holder_id=lease.holder_id,
            lease_epoch=lease.lease_epoch,
            state=self._states[lease.gpu_id],
            strategy=strategy.name,
            reason=reason,
            event_type=event_type,
            duration_sec=report.duration_sec,
            residual_gpu_memory_mb=report.residual_gpu_memory_mb,
            lease=lease,
            metadata=report.metadata,
        )

    def _run_hydrate(self, strategy: GpuResidencyStrategy, lease: GpuLease) -> HydrateResult:
        try:
            report = strategy.hydrate(self._context(lease, ResidencyOperation.HYDRATE, strategy.name))
        except Exception as exc:  # backend exceptions are converted into structured runtime events.
            report = ResidencyHookReport(success=False, reason=str(exc))
        event_type, reason = self._check_report(report, operation=ResidencyOperation.HYDRATE)
        if event_type is not None:
            self._states[lease.gpu_id] = ResidencyState.DEGRADED
        return HydrateResult(
            success=event_type is None,
            gpu_id=lease.gpu_id,
            role=lease.role,
            holder_id=lease.holder_id,
            lease_epoch=lease.lease_epoch,
            state=self._states[lease.gpu_id],
            strategy=strategy.name,
            reason=reason,
            event_type=event_type,
            duration_sec=report.duration_sec,
            residual_gpu_memory_mb=report.residual_gpu_memory_mb,
            lease=lease,
            metadata=report.metadata,
        )

    def _hydrate_failure_from_offload(
        self,
        offload: OffloadResult,
        transitions: list[ResidencyTransitionRecord],
    ) -> HydrateResult:
        return HydrateResult(
            success=False,
            gpu_id=offload.gpu_id,
            role=offload.role,
            holder_id=offload.holder_id,
            lease_epoch=offload.lease_epoch,
            state=offload.state,
            strategy=offload.strategy,
            reason=offload.reason,
            event_type=offload.event_type,
            duration_sec=offload.duration_sec,
            residual_gpu_memory_mb=offload.residual_gpu_memory_mb,
            lease=offload.lease,
            transitions=tuple(transitions),
            metadata=offload.metadata,
        )

    def _check_report(
        self,
        report: ResidencyHookReport,
        *,
        operation: ResidencyOperation,
    ) -> tuple[OffloadEventType | None, str | None]:
        if report.duration_sec > self._timeout_for(operation):
            return OffloadEventType.SHARED_GPU_TOGGLE_TIMEOUT, (
                f"{operation} took {report.duration_sec:.3f}s, "
                f"budget {self._timeout_for(operation):.3f}s"
            )
        if report.residual_gpu_memory_mb > self.residual_gpu_memory_budget_mb:
            return self._failure_event_for(operation), (
                f"residual gpu memory {report.residual_gpu_memory_mb}MB exceeds "
                f"budget {self.residual_gpu_memory_budget_mb}MB"
            )
        if not report.success:
            return self._failure_event_for(operation), report.reason or f"{operation} hook failed"
        return None, None

    def _failure_event_for(self, operation: ResidencyOperation) -> OffloadEventType:
        if operation == ResidencyOperation.OFFLOAD:
            return OffloadEventType.SHARED_GPU_OFFLOAD_FAILED
        return OffloadEventType.SHARED_GPU_HYDRATE_FAILED

    def _timeout_for(self, operation: ResidencyOperation) -> float:
        if operation == ResidencyOperation.OFFLOAD:
            return self.offload_timeout_sec
        return self.hydrate_timeout_sec

    def _context(self, lease: GpuLease, operation: ResidencyOperation, strategy: str) -> ResidencyHookContext:
        return ResidencyHookContext(
            gpu_id=lease.gpu_id,
            role=lease.role,
            holder_id=lease.holder_id,
            lease_epoch=lease.lease_epoch,
            operation=operation,
            strategy=strategy,
        )

    def _move(
        self,
        transitions: list[ResidencyTransitionRecord],
        transition: ResidencyTransition,
        lease: GpuLease,
        from_state: ResidencyState,
        to_state: ResidencyState,
    ) -> None:
        self._assert_state(lease.gpu_id, from_state)
        self._states[lease.gpu_id] = to_state
        transitions.append(
            ResidencyTransitionRecord(
                transition=transition,
                from_state=from_state,
                to_state=to_state,
                gpu_id=lease.gpu_id,
                role=lease.role,
                holder_id=lease.holder_id,
                lease_epoch=lease.lease_epoch,
            )
        )

    def _assert_current_lease(self, lease: GpuLease, *, expected_role: RoleName) -> None:
        if lease.role != expected_role:
            raise SlotStateError(f"expected {expected_role} lease for gpu {lease.gpu_id}, got {lease.role}")
        self._assert_shared_gpu(lease.gpu_id)
        self.lease_manager.assert_active(
            lease.role,
            lease.gpu_id,
            holder_id=lease.holder_id,
            expected_epoch=lease.lease_epoch,
        )

    def _assert_shared_gpu(self, gpu_id: int) -> None:
        state = self.lease_manager.get_state(gpu_id)
        if state.topology != GpuTopology.SHARED:
            raise GpuRoleUnsupported(gpu_id, RoleName.TRAINER)
        if gpu_id not in self._states:
            raise SlotStateError(f"gpu {gpu_id} has no residency state")

    def _assert_state(self, gpu_id: int, expected: ResidencyState) -> None:
        actual = self.get_state(gpu_id)
        if actual != expected:
            raise SlotStateError(f"gpu {gpu_id} residency state is {actual}, expected {expected}")
