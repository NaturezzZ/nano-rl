"""Metrics and health event sink."""

from __future__ import annotations

from uuid import uuid4

from nano_rl.runtime.protocols import EventSeverity, HealthEvent


class MetricsActorCore:
    def __init__(self) -> None:
        self._events: list[HealthEvent] = []

    def emit(
        self,
        event_type: str,
        *,
        source_actor: str,
        severity: EventSeverity = EventSeverity.INFO,
        gpu_id: int | None = None,
        policy_version: int | None = None,
        group_epoch: int | None = None,
        details: dict[str, object] | None = None,
    ) -> HealthEvent:
        event = HealthEvent(
            event_id=uuid4().hex,
            event_type=event_type,
            severity=severity,
            source_actor=source_actor,
            gpu_id=gpu_id,
            policy_version=policy_version,
            group_epoch=group_epoch,
            details=details or {},
        )
        self._events.append(event)
        return event

    def events(self) -> list[HealthEvent]:
        return list(self._events)

    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self._events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts
