"""Deterministic role adapters used by the local smoke runtime.

These classes define the stable import targets used in config examples.  They
are intentionally small deterministic adapters, not real FSDP2 or vLLM
implementations.  Real backends can later implement the same role surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

from nano_rl.exceptions import SlotStateError
from nano_rl.runtime.coordinators import GenerationRequest
from nano_rl.runtime.protocols import SampleRecord, TrainBatch, TrainStats, WeightFormat, WeightMeta
from nano_rl.runtime.slot import GpuLease, RoleName


@dataclass
class RolloutManagerRole:
    """Placeholder import target for the global rollout CPU actor."""

    manager_id: str = "rollout-manager"


@dataclass
class RolloutReplicaControllerRole:
    """Deterministic local adapter for one rollout DP replica.

    Real execution will fan out to one or more ``RolloutWorkerActor`` TP ranks.
    The smoke runtime keeps the interface at replica level so the global rollout
    manager can be tested without vLLM.
    """

    replica_id: str
    gpu_ids: tuple[int, ...] = ()
    worker_ids: tuple[str, ...] = ()
    active_policy_version: int | None = None
    active_weight_checksum: str | None = None
    active_lease_epochs: dict[int, int] | None = None

    def activate_weight(self, meta: WeightMeta, *, leases: tuple[GpuLease, ...]) -> None:
        self._assert_rollout_leases(leases)
        self.active_policy_version = meta.version_id
        self.active_weight_checksum = meta.checksum
        self.active_lease_epochs = {lease.gpu_id: lease.lease_epoch for lease in leases}

    def generate(
        self,
        request: GenerationRequest,
        reward_role: "RewardActorRole",
        *,
        leases: tuple[GpuLease, ...],
    ) -> SampleRecord:
        self._assert_rollout_leases(leases)
        if self.active_policy_version != request.target_policy_version:
            raise RuntimeError(
                f"replica {self.replica_id} active version {self.active_policy_version} "
                f"does not match request version {request.target_policy_version}"
            )
        response = f"{request.prompt} :: response@v{request.target_policy_version}"
        tokens = _stable_tokens(response)
        logprobs = [-0.01 * (index + 1) for index in range(len(tokens))]
        reward = reward_role.score(request.prompt, response)
        return SampleRecord(
            sample_id=request.request_id,
            request_id=request.request_id,
            policy_version=request.target_policy_version,
            weight_checksum=self.active_weight_checksum,
            prompt=request.prompt,
            response=response,
            finish_reason="stop",
            tokens=tokens,
            logprobs=logprobs,
            old_logprobs=logprobs,
            reward=reward,
            reward_source=reward_role.name,
            prompt_tokens=len(_stable_tokens(request.prompt)),
            response_tokens=len(tokens),
            meta={
                "replica_id": self.replica_id,
                "gpu_ids": list(self.gpu_ids),
                "lease_epochs": {str(lease.gpu_id): lease.lease_epoch for lease in leases},
                "worker_ids": list(self.worker_ids),
                **request.metadata,
            },
        )

    def _assert_rollout_leases(self, leases: tuple[GpuLease, ...]) -> None:
        expected_holders = dict(zip(self.gpu_ids, self.worker_ids, strict=True))
        if len(leases) != len(expected_holders):
            raise SlotStateError(
                f"replica {self.replica_id} expected {len(expected_holders)} rollout leases, got {len(leases)}"
            )
        seen: set[int] = set()
        for lease in leases:
            if lease.gpu_id in seen:
                raise SlotStateError(f"replica {self.replica_id} received duplicate lease for gpu {lease.gpu_id}")
            seen.add(lease.gpu_id)
            expected_holder = expected_holders.get(lease.gpu_id)
            if expected_holder is None:
                raise SlotStateError(f"replica {self.replica_id} received lease for unexpected gpu {lease.gpu_id}")
            if lease.role != RoleName.ROLLOUT:
                raise SlotStateError(f"replica {self.replica_id} lease role is {lease.role}, not rollout")
            if lease.holder_id != expected_holder:
                raise SlotStateError(
                    f"replica {self.replica_id} lease holder is {lease.holder_id}, expected {expected_holder}"
                )


@dataclass
class RolloutWorkerRole:
    worker_id: str
    gpu_id: int | None = None
    dp_rank: int | None = None
    tp_rank: int | None = None
    active_policy_version: int | None = None
    active_weight_checksum: str | None = None
    active_lease_epoch: int | None = None

    def activate_weight(self, meta: WeightMeta, *, lease: GpuLease) -> None:
        self._assert_rollout_lease(lease)
        self.active_policy_version = meta.version_id
        self.active_weight_checksum = meta.checksum
        self.active_lease_epoch = lease.lease_epoch

    def generate(self, request: GenerationRequest, reward_role: "RewardActorRole", *, lease: GpuLease) -> SampleRecord:
        self._assert_rollout_lease(lease)
        if self.active_policy_version != request.target_policy_version:
            raise RuntimeError(
                f"worker {self.worker_id} active version {self.active_policy_version} "
                f"does not match request version {request.target_policy_version}"
            )
        response = f"{request.prompt} :: response@v{request.target_policy_version}"
        tokens = _stable_tokens(response)
        logprobs = [-0.01 * (index + 1) for index in range(len(tokens))]
        reward = reward_role.score(request.prompt, response)
        return SampleRecord(
            sample_id=request.request_id,
            request_id=request.request_id,
            policy_version=request.target_policy_version,
            weight_checksum=self.active_weight_checksum,
            prompt=request.prompt,
            response=response,
            finish_reason="stop",
            tokens=tokens,
            logprobs=logprobs,
            old_logprobs=logprobs,
            reward=reward,
            reward_source=reward_role.name,
            prompt_tokens=len(_stable_tokens(request.prompt)),
            response_tokens=len(tokens),
            meta={
                "worker_id": self.worker_id,
                "replica_id": request.replica_id,
                "gpu_id": self.gpu_id,
                "lease_epoch": lease.lease_epoch,
                **request.metadata,
            },
        )

    def _assert_rollout_lease(self, lease: GpuLease) -> None:
        if self.gpu_id is None:
            raise SlotStateError(f"worker {self.worker_id} has no assigned gpu")
        if lease.role != RoleName.ROLLOUT:
            raise SlotStateError(f"worker {self.worker_id} lease role is {lease.role}, not rollout")
        if lease.gpu_id != self.gpu_id:
            raise SlotStateError(f"worker {self.worker_id} lease gpu is {lease.gpu_id}, expected {self.gpu_id}")
        if lease.holder_id != self.worker_id:
            raise SlotStateError(f"worker {self.worker_id} lease holder is {lease.holder_id}")


@dataclass
class TrainerRankRole:
    rank: int
    gpu_id: int | None = None
    group_epoch: int = 0
    train_step: int = 0

    def optimize(self, batch: TrainBatch, *, lease: GpuLease) -> TrainStats:
        self._assert_trainer_lease(lease)
        self.train_step += 1
        loss = 1.0 / max(1, batch.num_sequences + self.rank + self.train_step)
        return TrainStats(
            rank=self.rank,
            group_epoch=self.group_epoch,
            train_step=self.train_step,
            train_batch_id=batch.train_batch_id,
            num_sequences=batch.num_sequences,
            num_tokens=batch.num_tokens,
            loss=loss,
        )

    def _assert_trainer_lease(self, lease: GpuLease) -> None:
        if self.gpu_id is None:
            raise SlotStateError(f"trainer rank {self.rank} has no assigned gpu")
        if lease.role != RoleName.TRAINER:
            raise SlotStateError(f"trainer rank {self.rank} lease role is {lease.role}, not trainer")
        if lease.gpu_id != self.gpu_id:
            raise SlotStateError(f"trainer rank {self.rank} lease gpu is {lease.gpu_id}, expected {self.gpu_id}")
        expected_holder = f"trainer-rank-{self.rank}"
        if lease.holder_id != expected_holder:
            raise SlotStateError(f"trainer rank {self.rank} lease holder is {lease.holder_id}")

    def export_weight(self, parent: WeightMeta) -> WeightMeta:
        if self.rank != 0:
            raise RuntimeError("only rank 0 can export weights in the local smoke runtime")
        version_id = parent.version_id + 1
        checksum = sha256(f"{parent.checksum}:{version_id}:{self.train_step}".encode()).hexdigest()
        return WeightMeta(
            version_id=version_id,
            parent_version=parent.version_id,
            trainer_step=self.train_step,
            created_at=datetime.utcnow(),
            model_path=parent.model_path,
            tokenizer_path=parent.tokenizer_path,
            artifact_uri=parent.artifact_uri,
            manifest_uri=parent.manifest_uri,
            format=WeightFormat.VLLM_COMPATIBLE,
            checksum=checksum,
            created_by=f"trainer-rank-{self.rank}",
        )


@dataclass
class RewardActorRole:
    name: str = "deterministic_length_reward"

    def score(self, prompt: str, response: str) -> float:
        del prompt
        return min(1.0, len(response) / 100.0)


def bootstrap_weight_meta(
    *,
    model_path: str,
    tokenizer_path: str | None,
) -> WeightMeta:
    checksum = sha256(f"bootstrap:{model_path}:{tokenizer_path}".encode()).hexdigest()
    return WeightMeta(
        version_id=0,
        created_at=datetime.utcnow(),
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        format=WeightFormat.HF,
        checksum=checksum,
        created_by="bootstrap",
    )


def _stable_tokens(text: str) -> list[int]:
    digest = sha256(text.encode()).digest()
    return [byte for byte in digest[: min(16, max(1, len(text.split())))]]
