"""In-memory WeightRegistryActor core."""

from __future__ import annotations

from dataclasses import dataclass, field

from nano_rl.exceptions import WeightRegistryError
from nano_rl.runtime.protocols import WeightMeta, WeightStatus


@dataclass
class ActivationState:
    version_id: int
    required_workers: set[str]
    acked_workers: set[str] = field(default_factory=set)


class WeightRegistryActorCore:
    """Tracks weight versions and rollout activation acks."""

    def __init__(self) -> None:
        self._versions: dict[int, WeightMeta] = {}
        self._activation: ActivationState | None = None
        self._latest_registered: int | None = None
        self._latest_active_global: int | None = None

    def register(self, meta: WeightMeta) -> WeightMeta:
        existing = self._versions.get(meta.version_id)
        if existing is not None:
            if existing.checksum != meta.checksum:
                raise WeightRegistryError(
                    f"version {meta.version_id} already registered with a different checksum"
                )
            return existing

        registered = meta.model_copy(update={"status": WeightStatus.REGISTERED})
        self._versions[registered.version_id] = registered
        if self._latest_registered is None or registered.version_id > self._latest_registered:
            self._latest_registered = registered.version_id
        return registered

    def begin_activation(self, version_id: int, required_workers: set[str]) -> WeightMeta:
        if not required_workers:
            raise WeightRegistryError("required_workers must not be empty")
        meta = self._require_version(version_id)
        if meta.status == WeightStatus.FAILED:
            raise WeightRegistryError(f"cannot activate failed weight version {version_id}")
        activating = meta.model_copy(update={"status": WeightStatus.ACTIVATING})
        self._versions[version_id] = activating
        self._activation = ActivationState(version_id=version_id, required_workers=set(required_workers))
        return activating

    def ack_activation(self, version_id: int, worker_id: str) -> WeightMeta:
        if self._activation is None or self._activation.version_id != version_id:
            raise WeightRegistryError(f"no active activation for version {version_id}")
        if worker_id not in self._activation.required_workers:
            raise WeightRegistryError(f"worker {worker_id} is not required for version {version_id}")

        self._activation.acked_workers.add(worker_id)
        if self._activation.acked_workers == self._activation.required_workers:
            meta = self._require_version(version_id)
            active = meta.model_copy(update={"status": WeightStatus.ACTIVE_GLOBAL})
            self._versions[version_id] = active
            self._latest_active_global = version_id
            self._activation = None
            return active
        return self._require_version(version_id)

    def mark_failed(self, version_id: int, reason: str) -> WeightMeta:
        del reason
        meta = self._require_version(version_id)
        failed = meta.model_copy(update={"status": WeightStatus.FAILED})
        self._versions[version_id] = failed
        if self._activation and self._activation.version_id == version_id:
            self._activation = None
        return failed

    def latest_registered(self) -> WeightMeta | None:
        if self._latest_registered is None:
            return None
        return self._versions[self._latest_registered]

    def latest_active_global(self) -> WeightMeta | None:
        if self._latest_active_global is None:
            return None
        return self._versions[self._latest_active_global]

    def get(self, version_id: int) -> WeightMeta:
        return self._require_version(version_id)

    def all_versions(self) -> list[WeightMeta]:
        return [self._versions[k] for k in sorted(self._versions)]

    def _require_version(self, version_id: int) -> WeightMeta:
        meta = self._versions.get(version_id)
        if meta is None:
            raise WeightRegistryError(f"unknown weight version: {version_id}")
        return meta

