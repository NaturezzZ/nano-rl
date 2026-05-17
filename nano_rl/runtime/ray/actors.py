"""Optional Ray actor wrappers around local runtime cores."""

from __future__ import annotations

from dataclasses import asdict
import logging

from nano_rl.runtime.backends import (
    build_rollout_backend,
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
from nano_rl.runtime.reward_backend import build_reward_backend
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


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
logger = logging.getLogger(__name__)


def _configure_actor_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _config_backend_name(config: dict[str, object] | None, default: str) -> str:
    if config is None:
        return "none"
    return str(config.get("backend", default))


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
            _configure_actor_logging()
            logger.info("GpuLeaseManagerActor init started")
            self._core = GpuLeaseManagerCore(ResolvedGpuPlan.model_validate(plan))
            logger.info("GpuLeaseManagerActor init completed: gpu_count=%s", len(self._core.states()))

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
            _configure_actor_logging()
            logger.info(
                "RolloutManagerActor init started: replica_count=%s max_in_flight_per_replica=%s",
                len(replicas),
                max_in_flight_per_replica,
            )
            self._core = RolloutManagerCore(
                tuple(RolloutReplicaSpec.model_validate(replica) for replica in replicas),
                max_in_flight_per_replica=max_in_flight_per_replica,
            )
            logger.info("RolloutManagerActor init completed")

        def enqueue_prompts(self, prompts: list[object]) -> int:
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
            reward_config: dict[str, object] | None = None,
        ):
            _configure_actor_logging()
            logger.info(
                "RolloutReplicaControllerActor init started: replica_id=%s gpu_ids=%s worker_ids=%s rollout_backend=%s",
                replica_id,
                gpu_ids,
                worker_ids,
                _config_backend_name(backend_config, "vllm"),
            )
            self._role = RolloutReplicaControllerRole(
                replica_id=replica_id,
                gpu_ids=tuple(gpu_ids),
                worker_ids=tuple(worker_ids),
            )
            logger.info("RolloutReplicaControllerActor reward backend build started: replica_id=%s", replica_id)
            self._reward = RewardActorRole(build_reward_backend(reward_config))
            logger.info("RolloutReplicaControllerActor reward backend build completed: replica_id=%s", replica_id)
            self._backend = None
            if backend_config is not None:
                logger.info(
                    "RolloutReplicaControllerActor rollout backend build started: replica_id=%s backend=%s",
                    replica_id,
                    _config_backend_name(backend_config, "vllm"),
                )
                self._backend = build_rollout_backend(backend_config)
                logger.info(
                    "RolloutReplicaControllerActor rollout backend build completed: replica_id=%s backend=%s",
                    replica_id,
                    type(self._backend).__name__,
                )
            logger.info("RolloutReplicaControllerActor init completed: replica_id=%s", replica_id)

        def activate_weight(
            self,
            meta: dict[str, object],
            leases: list[dict[str, object]],
            transfer_source: dict[str, object] | None = None,
        ) -> None:
            weight = WeightMeta.model_validate(meta)
            logger.info(
                "RolloutReplicaControllerActor activate_weight started: replica_id=%s version=%s lease_count=%s",
                self._role.replica_id,
                weight.version_id,
                len(leases),
            )
            parsed_leases = tuple(GpuLease.model_validate(lease) for lease in leases)
            parsed_source = None if transfer_source is None else WeightShardSource.model_validate(transfer_source)
            self._role.activate_weight(weight, leases=parsed_leases, transfer_source=parsed_source)
            if self._backend is not None:
                self._backend.activate_weight(weight, lease=parsed_leases, transfer_source=parsed_source)
            logger.info(
                "RolloutReplicaControllerActor activate_weight completed: replica_id=%s version=%s",
                self._role.replica_id,
                weight.version_id,
            )

        def generate(self, request: dict[str, object], leases: list[dict[str, object]]) -> dict[str, object]:
            parsed_request = GenerationRequest(**request)
            logger.info(
                "RolloutReplicaControllerActor generate started: replica_id=%s request_id=%s target_policy_version=%s",
                self._role.replica_id,
                parsed_request.request_id,
                parsed_request.target_policy_version,
            )
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
                reward = self._reward.score_result(parsed_request.prompt, output.response)
                sample = generation_output_to_sample_record(
                    output,
                    reward=reward.reward,
                    reward_source=reward.reward_source,
                    request_metadata=parsed_request.metadata,
                )
                logger.info(
                    "RolloutReplicaControllerActor generate completed: replica_id=%s request_id=%s sample_id=%s",
                    self._role.replica_id,
                    parsed_request.request_id,
                    sample.sample_id,
                )
                return sample.model_dump(mode="json")
            sample = self._role.generate(
                parsed_request,
                self._reward,
                leases=parsed_leases,
            )
            logger.info(
                "RolloutReplicaControllerActor generate completed: replica_id=%s request_id=%s sample_id=%s",
                self._role.replica_id,
                parsed_request.request_id,
                sample.sample_id,
            )
            return sample.model_dump(mode="json")

        def offload(self, leases: list[dict[str, object]]) -> dict[str, object] | None:
            logger.info(
                "RolloutReplicaControllerActor offload started: replica_id=%s lease_count=%s",
                self._role.replica_id,
                len(leases),
            )
            parsed_leases = tuple(GpuLease.model_validate(lease) for lease in leases)
            state = None
            if self._backend is not None:
                state = self._backend.offload(lease=parsed_leases)
            logger.info(
                "RolloutReplicaControllerActor offload completed: replica_id=%s state=%s",
                self._role.replica_id,
                None if state is None else state.get("residency"),
            )
            return state

        def wake(self, leases: list[dict[str, object]]) -> dict[str, object] | None:
            logger.info(
                "RolloutReplicaControllerActor wake started: replica_id=%s lease_count=%s",
                self._role.replica_id,
                len(leases),
            )
            parsed_leases = tuple(GpuLease.model_validate(lease) for lease in leases)
            state = None
            if self._backend is not None:
                state = self._backend.wake(lease=parsed_leases)
            logger.info(
                "RolloutReplicaControllerActor wake completed: replica_id=%s state=%s",
                self._role.replica_id,
                None if state is None else state.get("residency"),
            )
            return state

        def state(self) -> dict[str, object]:
            backend_active_weight = None if self._backend is None else getattr(self._backend, "active_weight", None)
            backend_active_source = None if self._backend is None else getattr(self._backend, "active_weight_source", None)
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
                "backend_active_policy_version": (
                    None if backend_active_weight is None else backend_active_weight.version_id
                ),
                "backend_active_weight_checksum": None if backend_active_weight is None else backend_active_weight.checksum,
                "backend_active_weight_source": (
                    None if backend_active_source is None else backend_active_source.model_dump(mode="json")
                ),
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
        def __init__(
            self,
            worker_id: str,
            gpu_id: int,
            dp_rank: int,
            tp_rank: int,
            reward_config: dict[str, object] | None = None,
        ):
            _configure_actor_logging()
            logger.info(
                "RolloutWorkerActor init started: worker_id=%s gpu_id=%s dp_rank=%s tp_rank=%s",
                worker_id,
                gpu_id,
                dp_rank,
                tp_rank,
            )
            self._role = RolloutWorkerRole(
                worker_id=worker_id,
                gpu_id=gpu_id,
                dp_rank=dp_rank,
                tp_rank=tp_rank,
            )
            logger.info("RolloutWorkerActor reward backend build started: worker_id=%s", worker_id)
            self._reward = RewardActorRole(build_reward_backend(reward_config))
            logger.info("RolloutWorkerActor init completed: worker_id=%s", worker_id)

        def activate_weight(self, meta: dict[str, object], lease: dict[str, object]) -> None:
            weight = WeightMeta.model_validate(meta)
            logger.info(
                "RolloutWorkerActor activate_weight started: worker_id=%s version=%s",
                self._role.worker_id,
                weight.version_id,
            )
            self._role.activate_weight(
                weight,
                lease=GpuLease.model_validate(lease),
            )
            logger.info(
                "RolloutWorkerActor activate_weight completed: worker_id=%s version=%s",
                self._role.worker_id,
                weight.version_id,
            )

        def generate(self, request: dict[str, object], lease: dict[str, object]) -> dict[str, object]:
            parsed_request = GenerationRequest(**request)
            logger.info(
                "RolloutWorkerActor generate started: worker_id=%s request_id=%s target_policy_version=%s",
                self._role.worker_id,
                parsed_request.request_id,
                parsed_request.target_policy_version,
            )
            sample = self._role.generate(
                parsed_request,
                self._reward,
                lease=GpuLease.model_validate(lease),
            )
            logger.info(
                "RolloutWorkerActor generate completed: worker_id=%s request_id=%s sample_id=%s",
                self._role.worker_id,
                parsed_request.request_id,
                sample.sample_id,
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
            _configure_actor_logging()
            logger.info(
                "TrainerRankActor init started: rank=%s gpu_id=%s group_epoch=%s trainer_backend=%s",
                rank,
                gpu_id,
                group_epoch,
                _config_backend_name(backend_config, "fsdp2"),
            )
            self._role = TrainerRankRole(rank=rank, gpu_id=gpu_id, group_epoch=group_epoch)
            self._backend = None
            if backend_config is not None:
                logger.info(
                    "TrainerRankActor trainer backend build started: rank=%s backend=%s",
                    rank,
                    _config_backend_name(backend_config, "fsdp2"),
                )
                self._backend = build_trainer_backend(backend_config)
                logger.info(
                    "TrainerRankActor trainer backend build completed: rank=%s backend=%s",
                    rank,
                    type(self._backend).__name__,
                )
            logger.info("TrainerRankActor init completed: rank=%s", rank)

        def initialize_rank(self) -> dict[str, object] | None:
            if self._backend is None:
                logger.info("TrainerRankActor initialize_rank skipped: rank=%s backend=none", self._role.rank)
                return None
            logger.info("TrainerRankActor initialize_rank started: rank=%s", self._role.rank)
            state = self._backend.initialize_rank()
            logger.info(
                "TrainerRankActor initialize_rank completed: rank=%s residency=%s",
                self._role.rank,
                state.residency,
            )
            return state.model_dump(mode="json")

        def hydrate(self, state: dict[str, object] | None, lease: dict[str, object]) -> dict[str, object] | None:
            if self._backend is None:
                logger.info("TrainerRankActor hydrate skipped: rank=%s backend=none", self._role.rank)
                return None
            logger.info("TrainerRankActor hydrate started: rank=%s gpu_id=%s", self._role.rank, lease.get("gpu_id"))
            hydrated = self._backend.hydrate(
                None if state is None else self._backend_state(state),
                lease=GpuLease.model_validate(lease),
            )
            logger.info(
                "TrainerRankActor hydrate completed: rank=%s residency=%s",
                self._role.rank,
                hydrated.residency,
            )
            return hydrated.model_dump(mode="json")

        def optimize(self, batch: dict[str, object], lease: dict[str, object]) -> dict[str, object]:
            parsed_batch = TrainBatch.model_validate(batch)
            parsed_lease = GpuLease.model_validate(lease)
            logger.info(
                "TrainerRankActor optimize started: rank=%s train_batch_id=%s num_sequences=%s",
                self._role.rank,
                parsed_batch.train_batch_id,
                parsed_batch.num_sequences,
            )
            if self._backend is not None:
                result = self._backend.optimize(parsed_batch, lease=parsed_lease)
                stats = TrainStats(
                    rank=result.rank,
                    group_epoch=result.group_epoch,
                    train_step=result.train_step,
                    train_batch_id=result.train_batch_id,
                    num_sequences=result.num_sequences,
                    num_tokens=result.num_tokens,
                    loss=result.loss,
                )
                logger.info(
                    "TrainerRankActor optimize completed: rank=%s train_step=%s loss=%s",
                    self._role.rank,
                    stats.train_step,
                    stats.loss,
                )
                return stats.model_dump(mode="json")
            stats = self._role.optimize(
                parsed_batch,
                lease=parsed_lease,
            )
            logger.info(
                "TrainerRankActor optimize completed: rank=%s train_step=%s loss=%s",
                self._role.rank,
                stats.train_step,
                stats.loss,
            )
            return stats.model_dump(mode="json")

        def export_weight(self, parent: dict[str, object], publish: bool = True) -> dict[str, object]:
            weight = WeightMeta.model_validate(parent)
            logger.info(
                "TrainerRankActor export_weight started: rank=%s parent_version=%s publish=%s",
                self._role.rank,
                weight.version_id,
                publish,
            )
            if self._backend is not None:
                exported = self._backend.export_weight(weight, publish=publish)
            else:
                exported = self._role.export_weight(weight, publish=publish)
            logger.info(
                "TrainerRankActor export_weight completed: rank=%s version=%s",
                self._role.rank,
                exported.version_id,
            )
            return exported.model_dump(mode="json")

        def offload(self, lease: dict[str, object]) -> dict[str, object] | None:
            if self._backend is None:
                logger.info("TrainerRankActor offload skipped: rank=%s backend=none", self._role.rank)
                return None
            logger.info("TrainerRankActor offload started: rank=%s gpu_id=%s", self._role.rank, lease.get("gpu_id"))
            state = self._backend.offload(lease=GpuLease.model_validate(lease))
            logger.info(
                "TrainerRankActor offload completed: rank=%s residency=%s",
                self._role.rank,
                state.residency,
            )
            return state.model_dump(mode="json")

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
            _configure_actor_logging()
            logger.info("TrainerCoordinatorActor init started: trainer_rank_count=%s", len(trainer_ranks))
            self._core = TrainerCoordinatorCore(
                tuple(TrainerRankSpec.model_validate(rank) for rank in trainer_ranks),
            )
            logger.info("TrainerCoordinatorActor init completed")

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
            _configure_actor_logging()
            logger.info(
                "SampleQueueActor init started: max_policy_lag=%s sample_ttl_sec=%s queue_high_watermark=%s",
                max_policy_lag,
                sample_ttl_sec,
                queue_high_watermark,
            )
            self._core = SampleQueueActorCore(
                max_policy_lag=max_policy_lag,
                sample_ttl_sec=sample_ttl_sec,
                queue_high_watermark=queue_high_watermark,
            )
            logger.info("SampleQueueActor init completed")

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
            _configure_actor_logging()
            logger.info("WeightRegistryActor init started")
            self._core = WeightRegistryActorCore()
            logger.info("WeightRegistryActor init completed")

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
            _configure_actor_logging()
            logger.info("MetricsActor init started")
            self._core = MetricsActorCore()
            logger.info("MetricsActor init completed")

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
