"""Controller core for planning and dry-run execution."""

from __future__ import annotations

from dataclasses import dataclass

from nano_rl.config import CanonicalMode, LaunchConfig
from nano_rl.runtime.coordinators import RolloutManagerCore, TrainerCoordinatorCore
from nano_rl.runtime.metrics import MetricsActorCore
from nano_rl.runtime.offload import GpuResidencyManagerCore
from nano_rl.runtime.protocols import EventSeverity, SampleRecord, TrainStats, WeightMeta
from nano_rl.runtime.roles import (
    RewardActorRole,
    RolloutReplicaControllerRole,
    TrainerRankRole,
    bootstrap_weight_meta,
)
from nano_rl.runtime.sample_queue import QueueError, SampleQueueActorCore
from nano_rl.runtime.slot import GpuLease, GpuLeaseManagerCore, RoleName
from nano_rl.runtime.weight_registry import WeightRegistryActorCore
from nano_rl.runtime.weight_transfer import WeightTransferPlanner


@dataclass(frozen=True)
class RuntimePlan:
    canonical_mode: CanonicalMode
    total_gpus: int
    rollout_gpu_count: int
    shared_gpu_count: int
    rollout_replica_count: int
    rollout_worker_count: int
    trainer_rank_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_mode": self.canonical_mode,
            "total_gpus": self.total_gpus,
            "rollout_gpu_count": self.rollout_gpu_count,
            "shared_gpu_count": self.shared_gpu_count,
            "rollout_replica_count": self.rollout_replica_count,
            "rollout_worker_count": self.rollout_worker_count,
            "trainer_rank_count": self.trainer_rank_count,
        }


class ControllerCore:
    """Local core for ControllerActor behavior."""

    def __init__(self, launch_config: LaunchConfig) -> None:
        self.launch_config = launch_config
        self.metrics = MetricsActorCore()
        self.sample_queue = SampleQueueActorCore(
            max_policy_lag=launch_config.control.max_policy_lag,
            sample_ttl_sec=launch_config.control.sample_ttl_sec,
            queue_high_watermark=launch_config.control.queue_high_watermark,
        )
        self.weight_registry = WeightRegistryActorCore()
        self.weight_transfer = WeightTransferPlanner(
            method=launch_config.weight_transfer.method,
            allow_rollout_only_artifact_pull=launch_config.weight_transfer.allow_rollout_only_artifact_pull,
        )
        self.last_weight_transfer_plan = None
        self.gpu_leases = GpuLeaseManagerCore(launch_config.gpu_plan)
        toggle = launch_config.runtime.ray.gpu_manager.hybrid_toggle
        self.gpu_residency = GpuResidencyManagerCore(
            self.gpu_leases,
            offload_timeout_sec=toggle.cuda_quiesce_timeout_sec,
            hydrate_timeout_sec=toggle.cuda_quiesce_timeout_sec,
            residual_gpu_memory_budget_mb=toggle.offload.residual_gpu_memory_budget_mb,
        )
        max_in_flight_per_replica = max(
            1,
            launch_config.control.max_pending_rollout_refs // launch_config.gpu_plan.rollout_replica_count,
        )
        self.rollout_manager = RolloutManagerCore(
            launch_config.gpu_plan.rollout_replicas,
            max_in_flight_per_replica=max_in_flight_per_replica,
        )
        self.trainer_coordinator = TrainerCoordinatorCore(launch_config.gpu_plan.trainer_ranks)
        self.rollout_roles = {
            replica.replica_id: RolloutReplicaControllerRole(
                replica_id=replica.replica_id,
                gpu_ids=replica.gpu_ids,
                worker_ids=replica.worker_ids,
            )
            for replica in launch_config.gpu_plan.rollout_replicas
        }
        self.reward_role = RewardActorRole()
        self.trainer_ranks = {
            assignment.rank: TrainerRankRole(
                rank=assignment.rank,
                gpu_id=assignment.gpu_id,
                group_epoch=self.trainer_coordinator.group_epoch,
            )
            for assignment in self.trainer_coordinator.rank_assignments
        }

    def build_plan(self) -> RuntimePlan:
        return RuntimePlan(
            canonical_mode=self.launch_config.canonical_mode,
            total_gpus=self.launch_config.gpu_plan.total_gpus,
            rollout_gpu_count=self.launch_config.gpu_plan.rollout_gpu_count,
            shared_gpu_count=self.launch_config.gpu_plan.shared_gpu_count,
            rollout_replica_count=self.launch_config.gpu_plan.rollout_replica_count,
            rollout_worker_count=self.launch_config.gpu_plan.rollout_worker_count,
            trainer_rank_count=self.launch_config.gpu_plan.trainer_rank_count,
        )

    def dry_run(self) -> dict[str, object]:
        plan = self.build_plan().to_dict()
        plan["gpu_plan"] = self.launch_config.gpu_plan.model_dump(mode="json")
        plan["gpu_lease_states"] = self.gpu_leases.states()
        plan["gpu_residency_states"] = {
            gpu_id: self.gpu_residency.get_state(gpu_id).value
            for gpu_id in self.launch_config.gpu_plan.shared_gpu_ids
        }
        plan["queue"] = self.sample_queue.stats()
        plan["rollout_manager"] = self.rollout_manager.stats()
        plan["trainer_coordinator"] = self.trainer_coordinator.to_dict()
        plan["weight_transfer"] = self.launch_config.weight_transfer.model_dump(mode="json")
        return plan

    def enter_train_window(self) -> list[dict[str, object]]:
        states: list[dict[str, object]] = []
        for assignment in self.trainer_coordinator.rank_assignments:
            rollout_worker = self.launch_config.gpu_plan.rollout_worker_for_gpu(assignment.gpu_id)
            rollout_lease = self.gpu_leases.current_lease(
                RoleName.ROLLOUT,
                assignment.gpu_id,
                holder_id=rollout_worker.worker_id,
            )
            result = self.gpu_residency.enter_train(
                rollout_lease,
                trainer_holder_id=f"trainer-rank-{assignment.rank}",
            )
            if not result.success:
                self.metrics.emit(
                    str(result.event_type or "SHARED_GPU_TOGGLE_FAILED"),
                    source_actor="ControllerActor",
                    severity=EventSeverity.ERROR,
                    gpu_id=assignment.gpu_id,
                    details=result.model_dump(mode="json"),
                )
                raise RuntimeError(f"failed to enter train window on gpu {assignment.gpu_id}: {result.reason}")
            self.gpu_leases.assert_active(
                RoleName.TRAINER,
                assignment.gpu_id,
                holder_id=f"trainer-rank-{assignment.rank}",
                expected_epoch=result.lease_epoch,
            )
            states.append(self.gpu_leases.get_state(assignment.gpu_id).model_dump(mode="json"))
        return states

    def return_to_rollout(self) -> list[dict[str, object]]:
        states: list[dict[str, object]] = []
        for gpu_id in self.launch_config.gpu_plan.shared_gpu_ids:
            worker = self.launch_config.gpu_plan.rollout_worker_for_gpu(gpu_id)
            rank = self._trainer_rank_for_gpu(gpu_id)
            trainer_lease = self.gpu_leases.current_lease(
                RoleName.TRAINER,
                gpu_id,
                holder_id=f"trainer-rank-{rank}",
            )
            result = self.gpu_residency.exit_train(trainer_lease, rollout_holder_id=worker.worker_id)
            if not result.success:
                self.metrics.emit(
                    str(result.event_type or "SHARED_GPU_TOGGLE_FAILED"),
                    source_actor="ControllerActor",
                    severity=EventSeverity.ERROR,
                    gpu_id=gpu_id,
                    details=result.model_dump(mode="json"),
                )
                raise RuntimeError(f"failed to return gpu {gpu_id} to rollout: {result.reason}")
            self.gpu_leases.assert_active(
                RoleName.ROLLOUT,
                gpu_id,
                holder_id=worker.worker_id,
                expected_epoch=result.lease_epoch,
            )
            states.append(self.gpu_leases.get_state(gpu_id).model_dump(mode="json"))
        return states

    def bootstrap_initial_weight(self) -> WeightMeta:
        active = self.weight_registry.latest_active_global()
        if active is not None:
            return active

        meta = bootstrap_weight_meta(
            model_path=self.launch_config.model.model_path,
            tokenizer_path=self.launch_config.model.tokenizer_path,
        )
        registered = self.weight_registry.register(meta)
        return self._activate_weight_for_rollout(registered)

    def submit_prompts(self, prompts: list[str]) -> int:
        return self.rollout_manager.enqueue_prompts(prompts)

    def pump_rollout(
        self,
        *,
        current_weight: WeightMeta | None = None,
        controller_step: int = 0,
        max_new_requests: int | None = None,
    ) -> list[SampleRecord]:
        weight = current_weight or self.bootstrap_initial_weight()
        requests = self.rollout_manager.dispatch_from_backlog(
            target_policy_version=weight.version_id,
            controller_step=controller_step,
            output_queue_depth=self._output_queue_depth(),
            output_queue_high_watermark=self.launch_config.control.queue_high_watermark,
            max_new_requests=max_new_requests,
        )

        samples: list[SampleRecord] = []
        for request in requests:
            role = self.rollout_roles[request.replica_id]
            sample = role.generate(request, self.reward_role, leases=self._rollout_leases_for_replica(role))
            self.rollout_manager.complete_request(request.request_id)
            self.sample_queue.submit_sample(
                sample,
                current_policy_version=weight.version_id,
                object_ref=sample,
            )
            samples.append(sample)
        return samples

    def pump_rollout_until_blocked(
        self,
        *,
        current_weight: WeightMeta | None = None,
        controller_step: int = 0,
        max_total_requests: int | None = None,
    ) -> list[SampleRecord]:
        weight = current_weight or self.bootstrap_initial_weight()
        samples: list[SampleRecord] = []
        while max_total_requests is None or len(samples) < max_total_requests:
            remaining = None if max_total_requests is None else max_total_requests - len(samples)
            tick_samples = self.pump_rollout(
                current_weight=weight,
                controller_step=controller_step,
                max_new_requests=remaining,
            )
            if not tick_samples:
                break
            samples.extend(tick_samples)
        return samples

    def run_smoke_iteration(self, prompts: list[str]) -> dict[str, object]:
        """Run one deterministic local rollout->train->publish iteration."""

        if not prompts:
            raise ValueError("prompts must not be empty")

        current_weight = self.bootstrap_initial_weight()
        self.submit_prompts(prompts)
        samples = self.pump_rollout_until_blocked(
            current_weight=current_weight,
            controller_step=0,
            max_total_requests=len(prompts),
        )

        train_window_states = self.enter_train_window()
        batch = None
        try:
            batch = self.trainer_coordinator.reserve_batch(
                self.sample_queue,
                current_policy_version=current_weight.version_id,
                max_sequences=len(samples),
            )
            if batch is None:
                self.metrics.emit(
                    "TRAIN_BATCH_EMPTY",
                    source_actor="ControllerActor",
                    severity=EventSeverity.WARNING,
                    details={"prompt_count": len(prompts)},
                )
                self.return_to_rollout()
                raise RuntimeError("smoke iteration produced no train batch")

            train_stats: list[TrainStats] = []
            for rank in sorted(self.trainer_ranks):
                role = self.trainer_ranks[rank]
                train_stats.append(role.optimize(batch, lease=self._trainer_lease_for_rank(role)))

            new_weight = self.trainer_ranks[0].export_weight(current_weight)
            registered = self.weight_registry.register(new_weight)
            self.sample_queue.ack_batch(batch.train_batch_id)
        except Exception:
            if batch is not None:
                self._release_batch_safely(batch.train_batch_id)
            self.return_to_rollout()
            raise

        rollout_states = self.return_to_rollout()
        active_weight = self._activate_weight_for_rollout(registered)

        return {
            "initial_weight": current_weight.model_dump(mode="json"),
            "new_weight": active_weight.model_dump(mode="json"),
            "samples": [sample.model_dump(mode="json") for sample in samples],
            "train_batch": batch.model_dump(mode="json"),
            "train_stats": [stats.model_dump(mode="json") for stats in train_stats],
            "train_window_states": train_window_states,
            "rollout_states": rollout_states,
            "weight_transfer_plan": (
                None
                if self.last_weight_transfer_plan is None
                else self.last_weight_transfer_plan.model_dump(mode="json")
            ),
            "queue": self.sample_queue.stats(),
            "registry": [meta.model_dump(mode="json") for meta in self.weight_registry.all_versions()],
        }

    def _activate_weight_for_rollout(self, meta: WeightMeta) -> WeightMeta:
        replica_ids = set(self.rollout_roles)
        transfer_plan = self.weight_transfer.build_plan(
            meta,
            rollout_replicas=self.launch_config.gpu_plan.rollout_replicas,
            trainer_ranks=self.launch_config.gpu_plan.trainer_ranks,
        )
        self.rollout_manager.pause_for_weight(f"activate-v{meta.version_id}")
        self.weight_registry.begin_activation(meta.version_id, replica_ids)
        active = meta
        try:
            for replica_id, role in self.rollout_roles.items():
                leases = self._rollout_leases_for_replica(role)
                role.activate_weight(
                    meta,
                    leases=leases,
                    transfer_source=transfer_plan.source_for_replica(replica_id),
                )
                self._assert_rollout_leases_current(leases)
                active = self.weight_registry.ack_activation(meta.version_id, replica_id)
            self.last_weight_transfer_plan = transfer_plan
            self.rollout_manager.resume()
            return active
        except Exception:
            self.weight_registry.mark_failed(meta.version_id, "rollout activation failed")
            self.rollout_manager.resume()
            raise

    def _rollout_leases_for_replica(self, role: RolloutReplicaControllerRole) -> tuple[GpuLease, ...]:
        return tuple(
            self.gpu_leases.current_lease(RoleName.ROLLOUT, gpu_id, holder_id=worker_id)
            for gpu_id, worker_id in zip(role.gpu_ids, role.worker_ids, strict=True)
        )

    def _assert_rollout_leases_current(self, leases: tuple[GpuLease, ...]) -> None:
        for lease in leases:
            self.gpu_leases.assert_active(
                RoleName.ROLLOUT,
                lease.gpu_id,
                holder_id=lease.holder_id,
                expected_epoch=lease.lease_epoch,
            )

    def _trainer_lease_for_rank(self, role: TrainerRankRole) -> GpuLease:
        if role.gpu_id is None:
            raise RuntimeError(f"trainer rank {role.rank} has no assigned gpu")
        return self.gpu_leases.current_lease(
            RoleName.TRAINER,
            role.gpu_id,
            holder_id=f"trainer-rank-{role.rank}",
        )

    def _trainer_rank_for_gpu(self, gpu_id: int) -> int:
        for assignment in self.trainer_coordinator.rank_assignments:
            if assignment.gpu_id == gpu_id:
                return assignment.rank
        raise RuntimeError(f"no trainer rank assigned to gpu {gpu_id}")

    def _output_queue_depth(self) -> int:
        stats = self.sample_queue.stats()
        return stats["pending"] + stats["reserved_samples"]

    def _release_batch_safely(self, train_batch_id: str) -> None:
        try:
            self.sample_queue.release_batch(train_batch_id)
        except QueueError:
            return
