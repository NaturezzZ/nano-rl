"""Optional Ray actor wrappers around local runtime cores."""

from __future__ import annotations

from dataclasses import asdict

from nano_rl.runtime.backends import (
    TrainerBackendConfig,
    VllmBackendConfig,
    VllmRolloutBackend,
    build_trainer_backend,
    generation_output_to_sample_record,
)
from nano_rl.runtime.coordinators import GenerationRequest, RolloutManagerCore, TrainerCoordinatorCore
from nano_rl.runtime.metrics import MetricsActorCore
from nano_rl.runtime.protocols import (
    EventSeverity,
    SampleRecord,
    TrainBatch,
    TrainStats,
    WeightMeta,
    WeightShardSource,
)
from nano_rl.runtime.roles import (
    RewardActorRole,
    RolloutReplicaControllerRole,
    RolloutWorkerRole,
    TrainerRankRole,
)
from nano_rl.runtime.sample_queue import SampleQueueActorCore
from nano_rl.runtime.slot import (
    GpuLease,
    GpuLeaseManagerCore,
    ResolvedGpuPlan,
    RoleName,
    RolloutReplicaSpec,
    TrainerRankSpec,
)
from nano_rl.runtime.weight_registry import WeightRegistryActorCore


def build_gpu_lease_manager_actor_class():
    """Return a CPU Ray actor class for the GPU lease manager.

    Importing Ray is deferred so config parsing and unit tests do not require
    starting or importing the Ray runtime.  This actor intentionally does not
    request ``num_gpus``; rollout and trainer execution actors use role-specific
    custom resources plus lease tokens for CUDA access.
    """

    import ray

    @ray.remote(num_cpus=1)
    class GpuLeaseManagerActor:
        def __init__(self, plan: dict):
            self._core = GpuLeaseManagerCore(ResolvedGpuPlan.model_validate(plan))

        def states(self) -> list[dict[str, object]]:
            return self._core.states()

        def assert_active(
            self,
            role: str,
            gpu_id: int,
            holder_id: str | None = None,
            expected_epoch: int | None = None,
        ) -> bool:
            self._core.assert_active(RoleName(role), gpu_id, holder_id=holder_id, expected_epoch=expected_epoch)
            return True

        def current_lease(self, role: str, gpu_id: int, holder_id: str | None = None) -> dict[str, object]:
            return self._core.current_lease(RoleName(role), gpu_id, holder_id=holder_id).model_dump(mode="json")

        def grant(self, role: str, gpu_id: int, holder_id: str, reason: str = "manual") -> dict[str, object]:
            return self._core.grant(RoleName(role), gpu_id, holder_id, reason=reason).model_dump(mode="json")

        def mark_failed(self, gpu_id: int, reason: str) -> None:
            self._core.mark_failed(gpu_id, reason)

    return GpuLeaseManagerActor


def build_rollout_manager_actor_class():
    """Return the CPU Ray actor class for global rollout lifecycle control."""

    import ray

    @ray.remote(num_cpus=1)
    class RolloutManagerActor:
        def __init__(self, replicas: list[dict], max_in_flight_per_replica: int = 1):
            self._core = RolloutManagerCore(
                tuple(RolloutReplicaSpec.model_validate(replica) for replica in replicas),
                max_in_flight_per_replica=max_in_flight_per_replica,
            )

        def enqueue_prompts(self, prompts: list[str]) -> int:
            return self._core.enqueue_prompts(prompts)

        def pause_for_weight(self, reason: str) -> None:
            self._core.pause_for_weight(reason)

        def resume(self) -> None:
            self._core.resume()

        def dispatch_from_backlog(
            self,
            target_policy_version: int,
            controller_step: int,
            output_queue_depth: int,
            output_queue_high_watermark: int,
            max_new_requests: int | None = None,
        ) -> list[dict[str, object]]:
            return [
                asdict(request)
                for request in self._core.dispatch_from_backlog(
                    target_policy_version=target_policy_version,
                    controller_step=controller_step,
                    output_queue_depth=output_queue_depth,
                    output_queue_high_watermark=output_queue_high_watermark,
                    max_new_requests=max_new_requests,
                )
            ]

        def complete_request(self, request_id: str) -> dict[str, object]:
            return asdict(self._core.complete_request(request_id))

        def stats(self) -> dict[str, object]:
            return self._core.stats()

    return RolloutManagerActor


def build_rollout_replica_controller_actor_class():
    """Return a CPU Ray actor class for one rollout DP replica controller."""

    import ray

    @ray.remote(num_cpus=1)
    class RolloutReplicaControllerActor:
        def __init__(
            self,
            replica_id: str,
            gpu_ids: list[int],
            worker_ids: list[str],
            backend_config: dict[str, object] | None = None,
        ):
            self._role = RolloutReplicaControllerRole(
                replica_id=replica_id,
                gpu_ids=tuple(gpu_ids),
                worker_ids=tuple(worker_ids),
            )
            self._reward = RewardActorRole()
            self._backend = None
            if backend_config is not None:
                resolved = VllmBackendConfig.model_validate(backend_config)
                self._backend = VllmRolloutBackend(resolved)

        def activate_weight(
            self,
            meta: dict[str, object],
            leases: list[dict[str, object]],
            transfer_source: dict[str, object] | None = None,
        ) -> None:
            weight = WeightMeta.model_validate(meta)
            parsed_leases = tuple(GpuLease.model_validate(lease) for lease in leases)
            parsed_source = None if transfer_source is None else WeightShardSource.model_validate(transfer_source)
            self._role.activate_weight(weight, leases=parsed_leases, transfer_source=parsed_source)
            if self._backend is not None:
                self._backend.activate_weight(weight, lease=parsed_leases, transfer_source=parsed_source)

        def generate(self, request: dict[str, object], leases: list[dict[str, object]]) -> dict[str, object]:
            parsed_request = GenerationRequest(**request)
            parsed_leases = tuple(GpuLease.model_validate(lease) for lease in leases)
            if self._backend is not None:
                output = self._backend.generate(
                    prompt=parsed_request.prompt,
                    target_policy_version=parsed_request.target_policy_version,
                    request_metadata={
                        "replica_id": parsed_request.replica_id,
                        "gpu_ids": list(self._role.gpu_ids),
                        "worker_ids": list(self._role.worker_ids),
                        **parsed_request.metadata,
                    },
                    lease=parsed_leases,
                    request_id=parsed_request.request_id,
                )
                reward = self._reward.score(parsed_request.prompt, output.response)
                sample = generation_output_to_sample_record(
                    output,
                    reward=reward,
                    reward_source=self._reward.name,
                    request_metadata=parsed_request.metadata,
                )
                return sample.model_dump(mode="json")
            sample = self._role.generate(
                parsed_request,
                self._reward,
                leases=parsed_leases,
            )
            return sample.model_dump(mode="json")

        def state(self) -> dict[str, object]:
            return {
                "replica_id": self._role.replica_id,
                "gpu_ids": list(self._role.gpu_ids),
                "worker_ids": list(self._role.worker_ids),
                "active_policy_version": self._role.active_policy_version,
                "active_weight_checksum": self._role.active_weight_checksum,
                "active_lease_epochs": self._role.active_lease_epochs,
                "active_weight_source": (
                    None if self._role.active_weight_source is None else self._role.active_weight_source.model_dump(mode="json")
                ),
                "backend": None if self._backend is None else type(self._backend).__name__,
            }

    return RolloutReplicaControllerActor


def build_rollout_worker_actor_class():
    """Return a Ray actor class for one rollout TP worker.

    The class default does not request Ray GPUs.  Launch code should apply
    ``resources={"rollout_gpu_i": 1}`` from ``RayLaunchPlan``.
    """

    import ray

    @ray.remote(num_cpus=0, num_gpus=0)
    class RolloutWorkerActor:
        def __init__(self, worker_id: str, gpu_id: int, dp_rank: int, tp_rank: int):
            self._role = RolloutWorkerRole(
                worker_id=worker_id,
                gpu_id=gpu_id,
                dp_rank=dp_rank,
                tp_rank=tp_rank,
            )
            self._reward = RewardActorRole()

        def activate_weight(self, meta: dict[str, object], lease: dict[str, object]) -> None:
            self._role.activate_weight(
                WeightMeta.model_validate(meta),
                lease=GpuLease.model_validate(lease),
            )

        def generate(self, request: dict[str, object], lease: dict[str, object]) -> dict[str, object]:
            sample = self._role.generate(
                GenerationRequest(**request),
                self._reward,
                lease=GpuLease.model_validate(lease),
            )
            return sample.model_dump(mode="json")

        def state(self) -> dict[str, object]:
            return {
                "worker_id": self._role.worker_id,
                "gpu_id": self._role.gpu_id,
                "dp_rank": self._role.dp_rank,
                "tp_rank": self._role.tp_rank,
                "active_policy_version": self._role.active_policy_version,
                "active_weight_checksum": self._role.active_weight_checksum,
                "active_lease_epoch": self._role.active_lease_epoch,
            }

    return RolloutWorkerActor


def build_trainer_rank_actor_class():
    """Return a Ray actor class for one trainer rank.

    The class default does not request Ray GPUs.  Launch code should apply
    ``resources={"train_gpu_i": 1}`` from ``RayLaunchPlan``.
    """

    import ray

    @ray.remote(num_cpus=0, num_gpus=0)
    class TrainerRankActor:
        def __init__(
            self,
            rank: int,
            gpu_id: int,
            group_epoch: int = 0,
            backend_config: dict[str, object] | None = None,
        ):
            self._role = TrainerRankRole(rank=rank, gpu_id=gpu_id, group_epoch=group_epoch)
            self._backend = None
            if backend_config is not None:
                resolved = TrainerBackendConfig.model_validate(backend_config)
                self._backend = build_trainer_backend(resolved)

        def initialize_rank(self) -> dict[str, object] | None:
            if self._backend is None:
                return None
            return self._backend.initialize_rank().model_dump(mode="json")

        def hydrate(self, state: dict[str, object] | None, lease: dict[str, object]) -> dict[str, object] | None:
            if self._backend is None:
                return None
            return self._backend.hydrate(
                None if state is None else self._backend_state(state),
                lease=GpuLease.model_validate(lease),
            ).model_dump(mode="json")

        def optimize(self, batch: dict[str, object], lease: dict[str, object]) -> dict[str, object]:
            parsed_batch = TrainBatch.model_validate(batch)
            parsed_lease = GpuLease.model_validate(lease)
            if self._backend is not None:
                result = self._backend.optimize(parsed_batch, lease=parsed_lease)
                return TrainStats(
                    rank=result.rank,
                    group_epoch=result.group_epoch,
                    train_step=result.train_step,
                    train_batch_id=result.train_batch_id,
                    num_sequences=result.num_sequences,
                    num_tokens=result.num_tokens,
                    loss=result.loss,
                ).model_dump(mode="json")
            stats = self._role.optimize(
                parsed_batch,
                lease=parsed_lease,
            )
            return stats.model_dump(mode="json")

        def export_weight(self, parent: dict[str, object]) -> dict[str, object]:
            weight = WeightMeta.model_validate(parent)
            if self._backend is not None:
                return self._backend.export_weight(weight).model_dump(mode="json")
            return self._role.export_weight(weight).model_dump(mode="json")

        def offload(self, lease: dict[str, object]) -> dict[str, object] | None:
            if self._backend is None:
                return None
            return self._backend.offload(lease=GpuLease.model_validate(lease)).model_dump(mode="json")

        def state(self) -> dict[str, object]:
            return {
                "rank": self._role.rank,
                "gpu_id": self._role.gpu_id,
                "group_epoch": self._role.group_epoch,
                "train_step": self._role.train_step,
                "backend": None if self._backend is None else type(self._backend).__name__,
            }

        def _backend_state(self, state: dict[str, object]):
            from nano_rl.runtime.backends import TrainStateBundle

            return TrainStateBundle.model_validate(state)

    return TrainerRankActor


def build_trainer_coordinator_actor_class():
    """Return the CPU Ray actor class for trainer group coordination."""

    import ray

    @ray.remote(num_cpus=1)
    class TrainerCoordinatorActor:
        def __init__(self, trainer_ranks: list[dict]):
            self._core = TrainerCoordinatorCore(
                tuple(TrainerRankSpec.model_validate(rank) for rank in trainer_ranks),
            )

        def rebuild_group(self) -> int:
            return self._core.rebuild_group()

        def to_dict(self) -> dict[str, object]:
            return self._core.to_dict()

    return TrainerCoordinatorActor


def build_sample_queue_actor_class():
    """Return the CPU Ray actor class for sample queue state."""

    import ray

    @ray.remote(num_cpus=1)
    class SampleQueueActor:
        def __init__(self, max_policy_lag: int, sample_ttl_sec: int, queue_high_watermark: int):
            self._core = SampleQueueActorCore(
                max_policy_lag=max_policy_lag,
                sample_ttl_sec=sample_ttl_sec,
                queue_high_watermark=queue_high_watermark,
            )

        def submit_sample(
            self,
            record: dict[str, object],
            current_policy_version: int,
            object_ref: object | None = None,
        ) -> dict[str, object]:
            result = self._core.submit_sample(
                SampleRecord.model_validate(record),
                current_policy_version=current_policy_version,
                object_ref=object_ref,
            )
            return {
                "decision": result.decision.value,
                "sample_id": result.sample_id,
                "reason": result.reason,
                "accepted": result.accepted,
            }

        def reserve_train_batch(
            self,
            reserved_by: str,
            current_policy_version: int,
            max_sequences: int,
            max_tokens: int | None = None,
            lease_ttl_sec: int = 300,
        ) -> dict[str, object] | None:
            batch = self._core.reserve_train_batch(
                reserved_by=reserved_by,
                current_policy_version=current_policy_version,
                max_sequences=max_sequences,
                max_tokens=max_tokens,
                lease_ttl_sec=lease_ttl_sec,
            )
            return None if batch is None else batch.model_dump(mode="json")

        def ack_batch(self, train_batch_id: str) -> None:
            self._core.ack_batch(train_batch_id)

        def release_batch(self, train_batch_id: str) -> None:
            self._core.release_batch(train_batch_id)

        def stats(self) -> dict[str, int]:
            return self._core.stats()

    return SampleQueueActor


def build_weight_registry_actor_class():
    """Return the CPU Ray actor class for weight version state."""

    import ray

    @ray.remote(num_cpus=1)
    class WeightRegistryActor:
        def __init__(self):
            self._core = WeightRegistryActorCore()

        def register(self, meta: dict[str, object]) -> dict[str, object]:
            return self._core.register(WeightMeta.model_validate(meta)).model_dump(mode="json")

        def begin_activation(self, version_id: int, required_workers: list[str]) -> dict[str, object]:
            return self._core.begin_activation(version_id, set(required_workers)).model_dump(mode="json")

        def ack_activation(self, version_id: int, worker_id: str) -> dict[str, object]:
            return self._core.ack_activation(version_id, worker_id).model_dump(mode="json")

        def mark_failed(self, version_id: int, reason: str) -> dict[str, object]:
            return self._core.mark_failed(version_id, reason).model_dump(mode="json")

        def latest_registered(self) -> dict[str, object] | None:
            meta = self._core.latest_registered()
            return None if meta is None else meta.model_dump(mode="json")

        def latest_active_global(self) -> dict[str, object] | None:
            meta = self._core.latest_active_global()
            return None if meta is None else meta.model_dump(mode="json")

        def all_versions(self) -> list[dict[str, object]]:
            return [meta.model_dump(mode="json") for meta in self._core.all_versions()]

    return WeightRegistryActor


def build_metrics_actor_class():
    """Return the CPU Ray actor class for health events."""

    import ray

    @ray.remote(num_cpus=1)
    class MetricsActor:
        def __init__(self):
            self._core = MetricsActorCore()

        def emit(
            self,
            event_type: str,
            source_actor: str,
            severity: str = EventSeverity.INFO.value,
            gpu_id: int | None = None,
            policy_version: int | None = None,
            group_epoch: int | None = None,
            details: dict[str, object] | None = None,
        ) -> dict[str, object]:
            return self._core.emit(
                event_type,
                source_actor=source_actor,
                severity=EventSeverity(severity),
                gpu_id=gpu_id,
                policy_version=policy_version,
                group_epoch=group_epoch,
                details=details,
            ).model_dump(mode="json")

        def events(self) -> list[dict[str, object]]:
            return [event.model_dump(mode="json") for event in self._core.events()]

        def counts_by_type(self) -> dict[str, int]:
            return self._core.counts_by_type()

    return MetricsActor
