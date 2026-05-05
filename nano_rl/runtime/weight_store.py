"""Mock weight stores used in CPU-only and checkpoint-free flows."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from nano_rl.runtime.protocols import WeightFormat, WeightMeta, WeightStatus


_EPOCH = datetime(1970, 1, 1)


class WeightStore(Protocol):
    """Small checkpoint replacement interface for mockable weight versions."""

    def store(self, payload: "WeightPayload") -> WeightMeta:
        """Persist a mock weight payload and return its registered metadata."""

    def load(self, version_id: int) -> "WeightPayload":
        """Load a previously stored mock weight payload."""

    def meta(self, version_id: int) -> WeightMeta:
        """Return metadata for a previously stored mock weight payload."""

    def list_versions(self) -> tuple[WeightMeta, ...]:
        """Return registered weight versions ordered by version id."""

    def materialize(self, version_id: int, target_dir: str | Path | None = None) -> Path:
        """Materialize the version as a small local artifact directory."""


@dataclass(frozen=True)
class WeightPayload:
    """Serializable mock payload standing in for real checkpoint tensors."""

    version_id: int
    model_path: str = "mock://model"
    format: WeightFormat = WeightFormat.VLLM_COMPATIBLE
    weights: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parent_version: int | None = None
    trainer_step: int | None = None
    tokenizer_path: str | None = None
    artifact_uri: str | None = None
    tokenizer_hash: str | None = None
    chat_template_hash: str | None = None
    created_by: str = "mock_weight_store"
    created_at: datetime = _EPOCH

    def __post_init__(self) -> None:
        if self.version_id < 0:
            raise ValueError("version_id must be non-negative")
        if self.parent_version is not None and self.parent_version < 0:
            raise ValueError("parent_version must be non-negative")
        if self.trainer_step is not None and self.trainer_step < 0:
            raise ValueError("trainer_step must be non-negative")

    @property
    def checksum(self) -> str:
        return _checksum(self.to_checksum_dict())

    def to_checksum_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_path": self.model_path,
            "format": self.format.value,
            "weights": _jsonable(self.weights),
            "metadata": _jsonable(self.metadata),
            "parent_version": self.parent_version,
            "trainer_step": self.trainer_step,
            "tokenizer_path": self.tokenizer_path,
            "artifact_uri": self.artifact_uri,
            "tokenizer_hash": self.tokenizer_hash,
            "chat_template_hash": self.chat_template_hash,
            "created_by": self.created_by,
        }

    def to_manifest_payload(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_path": self.model_path,
            "format": self.format.value,
            "weights": _jsonable(self.weights),
            "metadata": _jsonable(self.metadata),
            "parent_version": self.parent_version,
            "trainer_step": self.trainer_step,
            "tokenizer_path": self.tokenizer_path,
            "artifact_uri": self.artifact_uri,
            "tokenizer_hash": self.tokenizer_hash,
            "chat_template_hash": self.chat_template_hash,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_manifest_payload(cls, data: Mapping[str, Any]) -> "WeightPayload":
        created_at_raw = data.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_raw)
            if isinstance(created_at_raw, str)
            else _EPOCH
        )
        return cls(
            version_id=int(data["version_id"]),
            model_path=str(data.get("model_path", "mock://model")),
            format=WeightFormat(data.get("format", WeightFormat.VLLM_COMPATIBLE.value)),
            weights=dict(data.get("weights") or {}),
            metadata=dict(data.get("metadata") or {}),
            parent_version=data.get("parent_version"),
            trainer_step=data.get("trainer_step"),
            tokenizer_path=data.get("tokenizer_path"),
            artifact_uri=data.get("artifact_uri"),
            tokenizer_hash=data.get("tokenizer_hash"),
            chat_template_hash=data.get("chat_template_hash"),
            created_by=str(data.get("created_by", "mock_weight_store")),
            created_at=created_at,
        )


class MockMemoryWeightStore:
    """In-memory mock checkpoint store."""

    def __init__(self) -> None:
        self._payloads: dict[int, WeightPayload] = {}
        self._metas: dict[int, WeightMeta] = {}

    def store(self, payload: WeightPayload) -> WeightMeta:
        meta = _build_meta(payload)
        self._payloads[payload.version_id] = payload
        self._metas[payload.version_id] = meta
        return meta

    def load(self, version_id: int) -> WeightPayload:
        try:
            return self._payloads[version_id]
        except KeyError as exc:
            raise KeyError(f"unknown weight version: {version_id}") from exc

    def meta(self, version_id: int) -> WeightMeta:
        try:
            return self._metas[version_id]
        except KeyError as exc:
            raise KeyError(f"unknown weight version: {version_id}") from exc

    def list_versions(self) -> tuple[WeightMeta, ...]:
        return tuple(self._metas[version_id] for version_id in sorted(self._metas))

    def materialize(self, version_id: int, target_dir: str | Path | None = None) -> Path:
        payload = self.load(version_id)
        meta = self.meta(version_id)
        if target_dir is None:
            raise ValueError("target_dir is required for MockMemoryWeightStore.materialize")
        version_dir = Path(target_dir) / _version_dir_name(version_id)
        version_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(version_dir / "manifest.json", payload, meta)
        return version_dir

    put = store
    get = load


class MockFilesystemWeightStore:
    """Filesystem mock checkpoint store that persists small JSON manifests."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._payloads: dict[int, WeightPayload] = {}
        self._metas: dict[int, WeightMeta] = {}
        self._load_existing_manifests()

    def store(self, payload: WeightPayload) -> WeightMeta:
        version_dir = self.root_dir / _version_dir_name(payload.version_id)
        manifest_path = version_dir / "manifest.json"
        meta = _build_meta(
            payload,
            artifact_uri=str(version_dir),
            manifest_uri=str(manifest_path),
        )
        version_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(manifest_path, payload, meta)
        self._payloads[payload.version_id] = payload
        self._metas[payload.version_id] = meta
        return meta

    def load(self, version_id: int) -> WeightPayload:
        self._ensure_loaded(version_id)
        return self._payloads[version_id]

    def meta(self, version_id: int) -> WeightMeta:
        self._ensure_loaded(version_id)
        return self._metas[version_id]

    def list_versions(self) -> tuple[WeightMeta, ...]:
        return tuple(self._metas[version_id] for version_id in sorted(self._metas))

    def materialize(self, version_id: int, target_dir: str | Path | None = None) -> Path:
        self._ensure_loaded(version_id)
        stored_dir = self.root_dir / _version_dir_name(version_id)
        if target_dir is None:
            return stored_dir

        target_version_dir = Path(target_dir) / _version_dir_name(version_id)
        if target_version_dir.exists():
            shutil.rmtree(target_version_dir)
        shutil.copytree(stored_dir, target_version_dir)
        return target_version_dir

    def _ensure_loaded(self, version_id: int) -> None:
        if version_id in self._payloads:
            return

        manifest_path = self.root_dir / _version_dir_name(version_id) / "manifest.json"
        if not manifest_path.exists():
            raise KeyError(f"unknown weight version: {version_id}")

        payload, meta = _read_manifest(manifest_path)
        self._payloads[version_id] = payload
        self._metas[version_id] = meta

    def _load_existing_manifests(self) -> None:
        for manifest_path in sorted(self.root_dir.glob("version-*/manifest.json")):
            payload, meta = _read_manifest(manifest_path)
            self._payloads[payload.version_id] = payload
            self._metas[payload.version_id] = meta

    put = store
    get = load


def build_weight_store(config: str | Mapping[str, Any] | None = None, **kwargs: Any) -> WeightStore:
    """Build a mock weight store without depending on the shared config model."""

    values: dict[str, Any] = dict(kwargs)
    if isinstance(config, str):
        backend = config
    elif config is None:
        backend = values.pop("backend", "mock_memory")
    else:
        values = {**dict(config), **values}
        backend = str(values.pop("backend", values.pop("type", "mock_memory")))

    if backend in {"mock_memory", "memory", "in_memory"}:
        return MockMemoryWeightStore()

    if backend in {"mock_filesystem", "filesystem", "fs"}:
        root_dir = (
            values.get("root_dir")
            or values.get("manifest_dir")
            or values.get("path")
            or values.get("checkpoint_dir")
        )
        if root_dir is None:
            raise ValueError("mock_filesystem weight store requires root_dir or manifest_dir")
        return MockFilesystemWeightStore(root_dir)

    raise ValueError(f"unsupported weight store backend: {backend}")


def build_mock_weight_payload(
    *,
    version_id: int = 0,
    parent_version: int | None = None,
    trainer_step: int | None = None,
    model_path: str = "mock://model",
    format: WeightFormat = WeightFormat.VLLM_COMPATIBLE,
    weights: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    tokenizer_path: str | None = None,
    artifact_uri: str | None = None,
    tokenizer_hash: str | None = None,
    chat_template_hash: str | None = None,
    created_by: str = "mock_weight_store",
) -> WeightPayload:
    """Build a deterministic mock payload for tests and local smoke runs."""

    if weights is None:
        weights = {
            "lm_head.weight": [version_id, trainer_step or 0, 1],
            "transformer.block.0.attn.qkv.weight": [parent_version or 0, version_id, 3],
        }
    if metadata is None:
        metadata = {"source": "mock", "version_id": version_id}

    return WeightPayload(
        version_id=version_id,
        model_path=model_path,
        format=format,
        weights=dict(weights),
        metadata=dict(metadata),
        parent_version=parent_version,
        trainer_step=trainer_step,
        tokenizer_path=tokenizer_path,
        artifact_uri=artifact_uri,
        tokenizer_hash=tokenizer_hash,
        chat_template_hash=chat_template_hash,
        created_by=created_by,
    )


def _build_meta(
    payload: WeightPayload,
    *,
    artifact_uri: str | None = None,
    manifest_uri: str | None = None,
) -> WeightMeta:
    return WeightMeta(
        version_id=payload.version_id,
        created_at=payload.created_at,
        model_path=payload.model_path,
        format=payload.format,
        checksum=payload.checksum,
        parent_version=payload.parent_version,
        trainer_step=payload.trainer_step,
        tokenizer_path=payload.tokenizer_path,
        artifact_uri=artifact_uri or payload.artifact_uri,
        manifest_uri=manifest_uri,
        tokenizer_hash=payload.tokenizer_hash,
        chat_template_hash=payload.chat_template_hash,
        created_by=payload.created_by,
        status=WeightStatus.REGISTERED,
    )


def _write_manifest(path: Path, payload: WeightPayload, meta: WeightMeta) -> None:
    manifest = {
        "schema_version": 1,
        "payload": payload.to_manifest_payload(),
        "meta": _dump_weight_meta(meta),
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_manifest(path: Path) -> tuple[WeightPayload, WeightMeta]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    payload = WeightPayload.from_manifest_payload(manifest["payload"])
    meta_data = manifest["meta"]
    meta = WeightMeta(
        version_id=meta_data["version_id"],
        created_at=datetime.fromisoformat(meta_data["created_at"]),
        model_path=meta_data["model_path"],
        format=WeightFormat(meta_data["format"]),
        checksum=meta_data["checksum"],
        parent_version=meta_data.get("parent_version"),
        trainer_step=meta_data.get("trainer_step"),
        tokenizer_path=meta_data.get("tokenizer_path"),
        artifact_uri=meta_data.get("artifact_uri"),
        manifest_uri=meta_data.get("manifest_uri"),
        tokenizer_hash=meta_data.get("tokenizer_hash"),
        chat_template_hash=meta_data.get("chat_template_hash"),
        created_by=meta_data.get("created_by"),
        status=WeightStatus(meta_data.get("status", WeightStatus.REGISTERED.value)),
    )
    return payload, meta


def _dump_weight_meta(meta: WeightMeta) -> dict[str, Any]:
    if hasattr(meta, "model_dump"):
        return meta.model_dump(mode="json")
    return _jsonable(meta.dict())


def _checksum(data: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(data),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, WeightFormat):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(inner) for inner in value]
    if isinstance(value, list):
        return [_jsonable(inner) for inner in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _version_dir_name(version_id: int) -> str:
    return f"version-{version_id}"
