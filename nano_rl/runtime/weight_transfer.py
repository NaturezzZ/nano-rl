"""Weight transfer planning between FSDP2 trainer ranks and vLLM rollout replicas."""

from __future__ import annotations

from collections.abc import Iterable

from nano_rl.exceptions import NanoRLError
from nano_rl.runtime.protocols import (
    WeightMeta,
    WeightShardSource,
    WeightShardSourceKind,
    WeightTransferMethod,
    WeightTransferPlan,
)
from nano_rl.runtime.slot import RolloutReplicaSpec, TrainerRankSpec


class WeightTransferError(NanoRLError):
    """Raised when a weight version cannot be routed to rollout replicas."""


class WeightTransferPlanner:
    """Build per-replica transfer sources for a registered weight version.

    The planner is deliberately independent from the trainer and rollout
    backends. It decides where each rollout TP group should fetch or derive
    weights; backend adapters decide how to materialize that source.
    """

    def __init__(
        self,
        *,
        method: WeightTransferMethod,
        allow_rollout_only_artifact_pull: bool = True,
    ) -> None:
        self.method = method
        self.allow_rollout_only_artifact_pull = allow_rollout_only_artifact_pull

    def build_plan(
        self,
        meta: WeightMeta,
        *,
        rollout_replicas: Iterable[RolloutReplicaSpec],
        trainer_ranks: Iterable[TrainerRankSpec],
    ) -> WeightTransferPlan:
        trainer_rank_by_gpu = {rank.gpu_id: rank.rank for rank in trainer_ranks}
        replicas = tuple(rollout_replicas)
        artifact_uri = meta.artifact_uri or meta.model_path

        if self.method == WeightTransferMethod.OBJECT_REF:
            sources = {
                replica.replica_id: self._object_ref_source(replica, meta)
                for replica in replicas
            }
            return WeightTransferPlan(
                version_id=meta.version_id,
                method=self.method,
                artifact_uri=artifact_uri,
                manifest_uri=meta.manifest_uri,
                sources=sources,
                metadata={"planner": "objectref"},
            )

        if self.method != WeightTransferMethod.LOCALITY_AWARE_CHECKPOINT:
            raise WeightTransferError(f"unsupported weight transfer method: {self.method}")

        sources = {
            replica.replica_id: self._locality_aware_source(
                replica,
                meta,
                trainer_rank_by_gpu=trainer_rank_by_gpu,
                artifact_uri=artifact_uri,
            )
            for replica in replicas
        }
        return WeightTransferPlan(
            version_id=meta.version_id,
            method=self.method,
            artifact_uri=artifact_uri,
            manifest_uri=meta.manifest_uri,
            sources=sources,
            metadata={
                "planner": "locality_aware_checkpoint",
                "shared_source_gpus": sorted(trainer_rank_by_gpu),
            },
        )

    def _object_ref_source(self, replica: RolloutReplicaSpec, meta: WeightMeta) -> WeightShardSource:
        return WeightShardSource(
            replica_id=replica.replica_id,
            kind=WeightShardSourceKind.RAY_OBJECT_REF,
            target_gpu_ids=replica.gpu_ids,
            target_worker_ids=replica.worker_ids,
            object_ref_key=f"weight:{meta.version_id}:{replica.replica_id}",
            reason="configured objectref transfer; rollout materializes this version from Ray object store refs",
        )

    def _locality_aware_source(
        self,
        replica: RolloutReplicaSpec,
        meta: WeightMeta,
        *,
        trainer_rank_by_gpu: dict[int, int],
        artifact_uri: str,
    ) -> WeightShardSource:
        local_ranks = tuple(
            trainer_rank_by_gpu[gpu_id]
            for gpu_id in replica.gpu_ids
            if gpu_id in trainer_rank_by_gpu
        )
        if len(local_ranks) == len(replica.gpu_ids):
            return WeightShardSource(
                replica_id=replica.replica_id,
                kind=WeightShardSourceKind.SHARED_GPU_RESHARD,
                target_gpu_ids=replica.gpu_ids,
                target_worker_ids=replica.worker_ids,
                source_rank_ids=local_ranks,
                source_gpu_ids=replica.gpu_ids,
                artifact_uri=artifact_uri,
                manifest_uri=meta.manifest_uri,
                reason=(
                    "all rollout TP GPUs also own FSDP2 trainer shards; activation can reshard "
                    "local trainer state into the vLLM TP layout"
                ),
            )

        if not self.allow_rollout_only_artifact_pull:
            missing = sorted(set(replica.gpu_ids) - set(trainer_rank_by_gpu))
            raise WeightTransferError(
                f"replica {replica.replica_id} has rollout-only GPUs {missing} but artifact pull is disabled"
            )

        return WeightShardSource(
            replica_id=replica.replica_id,
            kind=WeightShardSourceKind.ARTIFACT_PULL,
            target_gpu_ids=replica.gpu_ids,
            target_worker_ids=replica.worker_ids,
            artifact_uri=artifact_uri,
            manifest_uri=meta.manifest_uri,
            reason=(
                "one or more rollout TP GPUs do not own trainer shards; activation must hydrate "
                "the needed vLLM shards from the version artifact or manifest"
            ),
        )
