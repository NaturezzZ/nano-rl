import json
from pathlib import Path

import pytest

from nano_rl.runtime.protocols import WeightFormat, WeightStatus
from nano_rl.runtime.weight_store import (
    MockFilesystemWeightStore,
    MockMemoryWeightStore,
    WeightPayload,
    build_mock_weight_payload,
    build_weight_store,
)


def test_mock_memory_weight_store_returns_deterministic_meta() -> None:
    store = MockMemoryWeightStore()
    payload = build_mock_weight_payload(
        version_id=7,
        parent_version=6,
        trainer_step=13,
        metadata={"suite": "unit"},
    )

    meta = store.store(payload)
    duplicate_meta = store.store(payload)

    assert meta == duplicate_meta
    assert meta.version_id == 7
    assert meta.parent_version == 6
    assert meta.trainer_step == 13
    assert meta.format == WeightFormat.VLLM_COMPATIBLE
    assert meta.status == WeightStatus.REGISTERED
    assert meta.checksum == payload.checksum
    assert store.load(7) == payload
    assert store.list_versions() == (meta,)


def test_mock_memory_weight_store_materializes_manifest(tmp_path: Path) -> None:
    store = MockMemoryWeightStore()
    payload = build_mock_weight_payload(version_id=1)
    meta = store.store(payload)

    materialized_dir = store.materialize(1, tmp_path)
    manifest = json.loads((materialized_dir / "manifest.json").read_text(encoding="utf-8"))

    assert materialized_dir == tmp_path / "version-1"
    assert manifest["payload"]["version_id"] == 1
    assert manifest["meta"]["checksum"] == meta.checksum


def test_mock_filesystem_weight_store_writes_manifest_and_materializes(tmp_path: Path) -> None:
    root_dir = tmp_path / "store"
    store = MockFilesystemWeightStore(root_dir)
    payload = build_mock_weight_payload(version_id=2, trainer_step=5)

    meta = store.store(payload)
    manifest_path = Path(meta.manifest_uri)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path == root_dir / "version-2" / "manifest.json"
    assert meta.artifact_uri == str(root_dir / "version-2")
    assert manifest["schema_version"] == 1
    assert manifest["payload"]["trainer_step"] == 5
    assert manifest["meta"]["checksum"] == payload.checksum

    reloaded_store = MockFilesystemWeightStore(root_dir)
    assert reloaded_store.load(2) == payload
    assert reloaded_store.meta(2) == meta

    materialized_dir = reloaded_store.materialize(2, tmp_path / "materialized")
    materialized_manifest = materialized_dir / "manifest.json"
    assert materialized_dir == tmp_path / "materialized" / "version-2"
    assert materialized_manifest.exists()
    assert json.loads(materialized_manifest.read_text(encoding="utf-8")) == manifest


def test_build_weight_store_factory(tmp_path: Path) -> None:
    memory_store = build_weight_store({"backend": "mock_memory"})
    filesystem_store = build_weight_store(
        {
            "backend": "mock_filesystem",
            "root_dir": tmp_path / "weights",
        }
    )

    assert isinstance(memory_store, MockMemoryWeightStore)
    assert isinstance(filesystem_store, MockFilesystemWeightStore)


def test_build_weight_store_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported weight store backend"):
        build_weight_store({"backend": "real_checkpoint"})


def test_weight_payload_rejects_negative_ids() -> None:
    with pytest.raises(ValueError, match="version_id must be non-negative"):
        WeightPayload(version_id=-1)
