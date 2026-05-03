"""In-memory SampleQueueActor core."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from uuid import uuid4

from nano_rl.exceptions import QueueError
from nano_rl.runtime.protocols import SampleRecord, SampleRef, TrainBatch


class QueueDecision(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    DROPPED_TTL = "dropped_ttl"
    DROPPED_POLICY_LAG = "dropped_policy_lag"
    DROPPED_HIGH_WATERMARK = "dropped_high_watermark"


@dataclass(frozen=True)
class SubmitResult:
    decision: QueueDecision
    sample_id: str
    reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.decision == QueueDecision.ACCEPTED


class SampleQueueActorCore:
    """Queue state machine for rollout samples.

    This class mirrors the intended actor semantics but stays local and
    deterministic for unit tests.
    """

    def __init__(
        self,
        *,
        max_policy_lag: int,
        sample_ttl_sec: int,
        queue_high_watermark: int,
    ) -> None:
        self.max_policy_lag = max_policy_lag
        self.sample_ttl_sec = sample_ttl_sec
        self.queue_high_watermark = queue_high_watermark
        self._pending: dict[str, SampleRef] = {}
        self._reserved: dict[str, TrainBatch] = {}
        self._acked: set[str] = set()
        self._dropped: dict[str, str] = {}
        self._seen: set[str] = set()

    def submit_sample(
        self,
        record: SampleRecord,
        *,
        current_policy_version: int,
        object_ref: object | None = None,
        now: datetime | None = None,
    ) -> SubmitResult:
        now = now or datetime.utcnow()
        if record.sample_id in self._seen:
            return SubmitResult(QueueDecision.DUPLICATE, record.sample_id, "duplicate sample_id")

        expires_at = record.expires_at or (record.created_at + timedelta(seconds=self.sample_ttl_sec))
        if expires_at <= now:
            self._drop(record.sample_id, "ttl")
            return SubmitResult(QueueDecision.DROPPED_TTL, record.sample_id, "sample expired")

        behavior_policy_version = record.oldest_behavior_policy_version()
        lag = current_policy_version - behavior_policy_version
        if lag > self.max_policy_lag:
            self._drop(record.sample_id, "policy_lag")
            return SubmitResult(
                QueueDecision.DROPPED_POLICY_LAG,
                record.sample_id,
                f"policy lag {lag} exceeds max_policy_lag {self.max_policy_lag}",
            )

        if len(self._pending) + len(self._reserved) >= self.queue_high_watermark:
            self._drop(record.sample_id, "queue_high_watermark")
            return SubmitResult(
                QueueDecision.DROPPED_HIGH_WATERMARK,
                record.sample_id,
                "queue high watermark reached",
            )

        num_tokens = record.response_tokens or len(record.tokens)
        self._pending[record.sample_id] = SampleRef(
            sample_id=record.sample_id,
            policy_version=behavior_policy_version,
            created_at=record.created_at,
            expires_at=expires_at,
            object_ref=object_ref,
            num_tokens=num_tokens,
        )
        self._seen.add(record.sample_id)
        return SubmitResult(QueueDecision.ACCEPTED, record.sample_id)

    def reserve_train_batch(
        self,
        *,
        reserved_by: str,
        current_policy_version: int,
        max_sequences: int,
        max_tokens: int | None = None,
        lease_ttl_sec: int = 300,
        now: datetime | None = None,
    ) -> TrainBatch | None:
        now = now or datetime.utcnow()
        selected: list[SampleRef] = []
        selected_tokens = 0

        for sample_id, ref in list(self._pending.items()):
            if ref.expires_at is not None and ref.expires_at <= now:
                self._pending.pop(sample_id)
                self._drop(sample_id, "ttl")
                continue
            if current_policy_version - ref.policy_version > self.max_policy_lag:
                self._pending.pop(sample_id)
                self._drop(sample_id, "policy_lag")
                continue
            if len(selected) >= max_sequences:
                break
            if max_tokens is not None and selected and selected_tokens + ref.num_tokens > max_tokens:
                break
            selected.append(ref)
            selected_tokens += ref.num_tokens

        if not selected:
            return None

        for ref in selected:
            self._pending.pop(ref.sample_id, None)

        versions = [ref.policy_version for ref in selected]
        histogram: dict[int, int] = {}
        for version in versions:
            histogram[version] = histogram.get(version, 0) + 1

        batch = TrainBatch(
            train_batch_id=f"train-{uuid4().hex}",
            sample_refs=selected,
            sample_ids=[ref.sample_id for ref in selected],
            policy_version_min=min(versions),
            policy_version_max=max(versions),
            policy_version_histogram=histogram,
            num_sequences=len(selected),
            num_tokens=sum(ref.num_tokens for ref in selected),
            reserved_by=reserved_by,
            reserved_at=now,
            expires_at=now + timedelta(seconds=lease_ttl_sec),
        )
        self._reserved[batch.train_batch_id] = batch
        return batch

    def ack_batch(self, train_batch_id: str) -> None:
        batch = self._reserved.pop(train_batch_id, None)
        if batch is None:
            raise QueueError(f"unknown train batch: {train_batch_id}")
        self._acked.update(batch.sample_ids)

    def release_batch(self, train_batch_id: str) -> None:
        batch = self._reserved.pop(train_batch_id, None)
        if batch is None:
            raise QueueError(f"unknown train batch: {train_batch_id}")
        for ref in batch.sample_refs:
            if ref.sample_id not in self._acked and ref.sample_id not in self._dropped:
                self._pending[ref.sample_id] = ref

    def stats(self) -> dict[str, int]:
        return {
            "pending": len(self._pending),
            "reserved_batches": len(self._reserved),
            "reserved_samples": sum(batch.num_sequences for batch in self._reserved.values()),
            "acked": len(self._acked),
            "dropped": len(self._dropped),
            "seen": len(self._seen),
        }

    def _drop(self, sample_id: str, reason: str) -> None:
        self._seen.add(sample_id)
        self._dropped[sample_id] = reason
