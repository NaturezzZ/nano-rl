"""Resolved GPU plan and lease manager core.

The runtime keeps rollout and trainer actors as separate failure domains.  This
module owns the deterministic GPU assignment plan and a CPU-only lease manager
that gates which role may execute CUDA on a physical GPU at a given lease epoch.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from nano_rl.exceptions import GpuRoleUnsupported, SlotStateError


class GpuTopology(StrEnum):
    ROLLOUT_ONLY = "rollout_only"
    SHARED = "shared"
    IDLE = "idle"


class RoleName(StrEnum):
    ROLLOUT = "rollout"
    TRAINER = "trainer"


class GpuLeasePhase(StrEnum):
    IDLE = "IDLE"
    ROLLOUT_ACTIVE = "ROLLOUT_ACTIVE"
    TRAINER_ACTIVE = "TRAINER_ACTIVE"
    FAILED = "FAILED"


class RolloutWorkerSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_id: str
    replica_id: str
    dp_rank: int = Field(ge=0)
    tp_rank: int = Field(ge=0)
    gpu_id: int = Field(ge=0)
    topology: GpuTopology
    worker_class: str
    ray_resource: str


class RolloutReplicaSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    replica_id: str
    dp_rank: int = Field(ge=0)
    topology: GpuTopology
    gpu_ids: tuple[int, ...]
    worker_ids: tuple[str, ...]
    controller_class: str
    worker_class: str


class TrainerRankSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=0)
    gpu_id: int = Field(ge=0)
    trainer_class: str
    ray_resource: str


class ResolvedGpuPlan(BaseModel):
    """Deterministic GPU assignment expanded from quantity-only YAML."""

    model_config = ConfigDict(frozen=True)

    physical_gpu_ids: tuple[int, ...]
    rollout_only_gpu_ids: tuple[int, ...]
    shared_gpu_ids: tuple[int, ...]
    idle_gpu_ids: tuple[int, ...]
    rollout_replicas: tuple[RolloutReplicaSpec, ...]
    rollout_workers: tuple[RolloutWorkerSpec, ...]
    trainer_ranks: tuple[TrainerRankSpec, ...]

    @property
    def total_gpus(self) -> int:
        return len(self.physical_gpu_ids)

    @property
    def rollout_gpu_count(self) -> int:
        return len(self.rollout_only_gpu_ids) + len(self.shared_gpu_ids)

    @property
    def shared_gpu_count(self) -> int:
        return len(self.shared_gpu_ids)

    @property
    def rollout_replica_count(self) -> int:
        return len(self.rollout_replicas)

    @property
    def rollout_worker_count(self) -> int:
        return len(self.rollout_workers)

    @property
    def trainer_rank_count(self) -> int:
        return len(self.trainer_ranks)

    def rollout_worker_for_gpu(self, gpu_id: int) -> RolloutWorkerSpec:
        for worker in self.rollout_workers:
            if worker.gpu_id == gpu_id:
                return worker
        raise SlotStateError(f"no rollout worker assigned to gpu {gpu_id}")


class GpuLease(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpu_id: int = Field(ge=0)
    role: RoleName
    holder_id: str
    lease_epoch: int = Field(ge=0)


class GpuLeaseState(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpu_id: int = Field(ge=0)
    topology: GpuTopology
    phase: GpuLeasePhase
    active_role: RoleName | None = None
    holder_id: str | None = None
    lease_epoch: int = Field(default=0, ge=0)
    failed_reason: str | None = None


class GpuLeaseManagerCore:
    """CPU-only active-role gate for physical GPUs.

    Rollout and trainer actors are independent actors/processes.  They should
    use role-specific Ray custom resources for placement and call this manager
    to acquire a lease before executing CUDA work on an assigned physical GPU.
    """

    def __init__(self, plan: ResolvedGpuPlan):
        self.plan = plan
        self._states: dict[int, GpuLeaseState] = {}
        self._supported_roles: dict[int, set[RoleName]] = {}

        rollout_gpus = set(plan.rollout_only_gpu_ids) | set(plan.shared_gpu_ids)
        shared_gpus = set(plan.shared_gpu_ids)

        for gpu_id in plan.physical_gpu_ids:
            roles: set[RoleName] = set()
            topology = GpuTopology.IDLE
            phase = GpuLeasePhase.IDLE
            active_role: RoleName | None = None
            holder_id: str | None = None

            if gpu_id in rollout_gpus:
                roles.add(RoleName.ROLLOUT)
                worker = plan.rollout_worker_for_gpu(gpu_id)
                topology = worker.topology
                phase = GpuLeasePhase.ROLLOUT_ACTIVE
                active_role = RoleName.ROLLOUT
                holder_id = worker.worker_id
            if gpu_id in shared_gpus:
                roles.add(RoleName.TRAINER)
                topology = GpuTopology.SHARED

            self._supported_roles[gpu_id] = roles
            self._states[gpu_id] = GpuLeaseState(
                gpu_id=gpu_id,
                topology=topology,
                phase=phase,
                active_role=active_role,
                holder_id=holder_id,
            )

    def states(self) -> list[dict[str, object]]:
        return [state.model_dump(mode="json") for _, state in sorted(self._states.items())]

    def get_state(self, gpu_id: int) -> GpuLeaseState:
        try:
            return self._states[gpu_id]
        except KeyError as exc:
            raise SlotStateError(f"unknown gpu {gpu_id}") from exc

    def assert_active(
        self,
        role: RoleName,
        gpu_id: int,
        *,
        holder_id: str | None = None,
        expected_epoch: int | None = None,
    ) -> None:
        self._assert_role_supported(gpu_id, role)
        state = self.get_state(gpu_id)
        if state.phase == GpuLeasePhase.FAILED:
            raise SlotStateError(f"gpu {gpu_id} is failed: {state.failed_reason}")
        if state.active_role != role:
            raise SlotStateError(f"gpu {gpu_id} active role is {state.active_role}, not {role}")
        if holder_id is not None and state.holder_id != holder_id:
            raise SlotStateError(f"gpu {gpu_id} holder is {state.holder_id}, not {holder_id}")
        if expected_epoch is not None and state.lease_epoch != expected_epoch:
            raise SlotStateError(f"gpu {gpu_id} lease_epoch is {state.lease_epoch}, expected {expected_epoch}")

    def current_lease(self, role: RoleName, gpu_id: int, *, holder_id: str | None = None) -> GpuLease:
        """Return the current active lease token after validating it."""

        self.assert_active(role, gpu_id, holder_id=holder_id)
        state = self.get_state(gpu_id)
        if state.holder_id is None:
            raise SlotStateError(f"gpu {gpu_id} has no active holder")
        return GpuLease(
            gpu_id=gpu_id,
            role=role,
            holder_id=state.holder_id,
            lease_epoch=state.lease_epoch,
        )

    def grant(self, role: RoleName, gpu_id: int, holder_id: str, *, reason: str = "manual") -> GpuLease:
        del reason
        self._assert_role_supported(gpu_id, role)
        state = self.get_state(gpu_id)
        if state.phase == GpuLeasePhase.FAILED:
            raise SlotStateError(f"gpu {gpu_id} is failed: {state.failed_reason}")

        if state.active_role == role and state.holder_id == holder_id:
            return GpuLease(gpu_id=gpu_id, role=role, holder_id=holder_id, lease_epoch=state.lease_epoch)

        next_epoch = state.lease_epoch + 1
        phase = GpuLeasePhase.ROLLOUT_ACTIVE if role == RoleName.ROLLOUT else GpuLeasePhase.TRAINER_ACTIVE
        self._states[gpu_id] = GpuLeaseState(
            gpu_id=gpu_id,
            topology=state.topology,
            phase=phase,
            active_role=role,
            holder_id=holder_id,
            lease_epoch=next_epoch,
        )
        return GpuLease(gpu_id=gpu_id, role=role, holder_id=holder_id, lease_epoch=next_epoch)

    def mark_failed(self, gpu_id: int, reason: str) -> None:
        state = self.get_state(gpu_id)
        self._states[gpu_id] = GpuLeaseState(
            gpu_id=gpu_id,
            topology=state.topology,
            phase=GpuLeasePhase.FAILED,
            lease_epoch=state.lease_epoch + 1,
            failed_reason=reason,
        )

    def _assert_role_supported(self, gpu_id: int, role: RoleName) -> None:
        if gpu_id not in self._supported_roles:
            raise SlotStateError(f"unknown gpu {gpu_id}")
        if role not in self._supported_roles[gpu_id]:
            raise GpuRoleUnsupported(gpu_id, role)
