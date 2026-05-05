"""Coordinator cores for rollout and trainer groups."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import cycle
from typing import Any
from uuid import uuid4

from nano_rl.exceptions import NanoRLError
from nano_rl.runtime.protocols import TrainBatch
from nano_rl.runtime.sample_queue import SampleQueueActorCore
from nano_rl.runtime.slot import RolloutReplicaSpec, TrainerRankSpec


class CoordinatorError(NanoRLError):
    """Coordinator state error."""


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    replica_id: str
    prompt: str
    target_policy_version: int
    metadata: dict[str, object]


@dataclass(frozen=True)
class QueuedPrompt:
    prompt: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class RankAssignment:
    rank: int
    gpu_id: int
    trainer_class: str
    ray_resource: str


class RolloutManagerCore:
    def __init__(
        self,
        replicas: tuple[RolloutReplicaSpec, ...] | list[RolloutReplicaSpec],
        *,
        max_in_flight_per_replica: int = 1,
    ) -> None:
        self.replica_ids = [replica.replica_id for replica in replicas]
        if not self.replica_ids:
            raise CoordinatorError("rollout manager requires at least one rollout replica")
        if max_in_flight_per_replica < 1:
            raise CoordinatorError("max_in_flight_per_replica must be >= 1")
        self._replica_cycle = cycle(self.replica_ids)
        self._max_in_flight_per_replica = max_in_flight_per_replica
        self._input_backlog: deque[QueuedPrompt] = deque()
        self._paused_reason: str | None = None
        self._in_flight: dict[str, GenerationRequest] = {}

    @property
    def paused(self) -> bool:
        return self._paused_reason is not None

    def pause_for_weight(self, reason: str) -> None:
        self._paused_reason = reason

    def resume(self) -> None:
        self._paused_reason = None

    def enqueue_prompts(self, prompts: list[str | Mapping[str, Any]]) -> int:
        for prompt in prompts:
            self._input_backlog.append(_coerce_queued_prompt(prompt))
        return len(self._input_backlog)

    def dispatch_prompts(
        self,
        prompts: list[str | Mapping[str, Any]],
        *,
        target_policy_version: int,
        controller_step: int,
    ) -> list[GenerationRequest]:
        if self.paused:
            raise CoordinatorError(f"rollout manager is paused: {self._paused_reason}")
        requests: list[GenerationRequest] = []
        for prompt in prompts:
            queued = _coerce_queued_prompt(prompt)
            request = self._build_request(queued, target_policy_version, controller_step)
            requests.append(request)
        return requests

    def dispatch_from_backlog(
        self,
        *,
        target_policy_version: int,
        controller_step: int,
        output_queue_depth: int,
        output_queue_high_watermark: int,
        max_new_requests: int | None = None,
    ) -> list[GenerationRequest]:
        if self.paused:
            raise CoordinatorError(f"rollout manager is paused: {self._paused_reason}")
        if output_queue_depth >= output_queue_high_watermark:
            return []

        queue_capacity = output_queue_high_watermark - output_queue_depth
        rollout_capacity = self.max_in_flight - len(self._in_flight)
        request_budget = min(queue_capacity, rollout_capacity, len(self._input_backlog))
        if max_new_requests is not None:
            request_budget = min(request_budget, max_new_requests)

        requests: list[GenerationRequest] = []
        for _ in range(request_budget):
            replica_id = self._next_schedulable_replica()
            if replica_id is None:
                break
            prompt = self._input_backlog.popleft()
            request = self._build_request(prompt, target_policy_version, controller_step, replica_id=replica_id)
            requests.append(request)
        return requests

    def complete_request(self, request_id: str) -> GenerationRequest:
        request = self._in_flight.pop(request_id, None)
        if request is None:
            raise CoordinatorError(f"unknown rollout request: {request_id}")
        return request

    @property
    def max_in_flight(self) -> int:
        return len(self.replica_ids) * self._max_in_flight_per_replica

    def in_flight_by_replica(self) -> dict[str, int]:
        counts = {replica_id: 0 for replica_id in self.replica_ids}
        for request in self._in_flight.values():
            counts[request.replica_id] += 1
        return counts

    def stats(self) -> dict[str, object]:
        return {
            "replicas": len(self.replica_ids),
            "in_flight": len(self._in_flight),
            "in_flight_by_replica": self.in_flight_by_replica(),
            "input_backlog": len(self._input_backlog),
            "max_in_flight": self.max_in_flight,
            "paused": self.paused,
        }

    def _build_request(
        self,
        prompt: QueuedPrompt,
        target_policy_version: int,
        controller_step: int,
        *,
        replica_id: str | None = None,
    ) -> GenerationRequest:
        replica_id = replica_id or next(self._replica_cycle)
        request = GenerationRequest(
            request_id=f"rollout-{uuid4().hex}",
            replica_id=replica_id,
            prompt=prompt.prompt,
            target_policy_version=target_policy_version,
            metadata={**prompt.metadata, "controller_step": controller_step},
        )
        self._in_flight[request.request_id] = request
        return request

    def _next_schedulable_replica(self) -> str | None:
        counts = self.in_flight_by_replica()
        for _ in self.replica_ids:
            replica_id = next(self._replica_cycle)
            if counts[replica_id] < self._max_in_flight_per_replica:
                return replica_id
        return None


class TrainerCoordinatorCore:
    def __init__(self, trainer_ranks: tuple[TrainerRankSpec, ...] | list[TrainerRankSpec]):
        if not trainer_ranks:
            raise CoordinatorError("trainer coordinator requires at least one trainer rank")
        self.group_epoch = 0
        self.rank_assignments = tuple(
            RankAssignment(
                rank=rank_spec.rank,
                gpu_id=rank_spec.gpu_id,
                trainer_class=rank_spec.trainer_class,
                ray_resource=rank_spec.ray_resource,
            )
            for rank_spec in trainer_ranks
        )

    @property
    def world_size(self) -> int:
        return len(self.rank_assignments)

    def rebuild_group(self) -> int:
        self.group_epoch += 1
        return self.group_epoch

    def reserve_batch(
        self,
        queue: SampleQueueActorCore,
        *,
        current_policy_version: int,
        max_sequences: int,
        max_tokens: int | None = None,
    ) -> TrainBatch | None:
        return queue.reserve_train_batch(
            reserved_by=f"trainer-group-{self.group_epoch}",
            current_policy_version=current_policy_version,
            max_sequences=max_sequences,
            max_tokens=max_tokens,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "group_epoch": self.group_epoch,
            "world_size": self.world_size,
            "rank_assignments": [assignment.__dict__ for assignment in self.rank_assignments],
        }


def _coerce_queued_prompt(value: str | Mapping[str, Any]) -> QueuedPrompt:
    if isinstance(value, str):
        return QueuedPrompt(prompt=value, metadata={})
    if not isinstance(value, Mapping):
        raise CoordinatorError("rollout prompts must be strings or mappings")
    raw_prompt = value.get("prompt")
    if not isinstance(raw_prompt, str):
        raise CoordinatorError("rollout prompt mapping requires a string 'prompt'")
    raw_metadata = value.get("metadata", {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, Mapping):
        raise CoordinatorError("rollout prompt mapping metadata must be an object")
    metadata = dict(raw_metadata)
    prompt_id = value.get("prompt_id", value.get("id"))
    if prompt_id is not None:
        metadata.setdefault("prompt_id", str(prompt_id))
    for key, item in value.items():
        if key not in {"prompt", "metadata", "prompt_id", "id"}:
            metadata.setdefault(str(key), item)
    return QueuedPrompt(prompt=raw_prompt, metadata=metadata)
