"""Input artifact validation helpers."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from nano_rl.config import LaunchConfig, SourceType
from nano_rl.exceptions import InvalidInputArtifactError


def validate_input_artifacts(config: LaunchConfig) -> None:
    """Fail-fast validation before Ray startup.

    Model and checkpoint paths are normal artifact paths and are not constrained
    to HDFS/HDFS-FUSE source types. Dataset probing keeps the explicit source
    policy because data still declares ``data.source_type``.
    """

    _validate_source("data.data_path", config.data.source_type, config.data.data_path, config)


def _validate_source(field: str, source_type: SourceType, uri: str | None, config: LaunchConfig) -> None:
    if source_type in {SourceType.MOCK_INLINE, SourceType.MOCK_GENERATED}:
        return

    if source_type in {SourceType.LOCAL_CSV, SourceType.MOCK_JSONL}:
        if uri is None:
            raise InvalidInputArtifactError(f"{field} is required for {source_type.value}")
        path = Path(uri)
        if not path.exists():
            raise InvalidInputArtifactError(f"{field} {source_type.value} file does not exist: {uri}")
        if not path.is_file():
            raise InvalidInputArtifactError(f"{field} {source_type.value} path is not a file: {uri}")
        if source_type == SourceType.LOCAL_CSV:
            encoding = config.data.local_csv.encoding if config.data.local_csv else "utf-8"
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise InvalidInputArtifactError(f"{field} local_csv file must include a header row: {uri}")
                if config.data.prompt_column not in reader.fieldnames:
                    raise InvalidInputArtifactError(
                        f"{field} local_csv header is missing prompt_column "
                        f"{config.data.prompt_column!r}: {uri}"
                    )
        return

    if uri is None:
        raise InvalidInputArtifactError(f"{field} is required for source_type: {source_type}")

    if source_type == SourceType.HDFS_URI:
        if not uri.startswith("hdfs://"):
            raise InvalidInputArtifactError(f"{field} must be hdfs:// for hdfs_uri: {uri}")
        hdfs = config.runtime.storage.hdfs
        if hdfs is None:
            raise InvalidInputArtifactError(f"{field} requires runtime.storage.hdfs config")
        try:
            result = subprocess.run(
                [hdfs.cli, "dfs", "-test", "-e", uri],
                check=False,
                timeout=hdfs.read_probe_timeout_sec,
            )
        except FileNotFoundError as exc:
            raise InvalidInputArtifactError(f"HDFS CLI not found for {field}: {hdfs.cli}") from exc
        except subprocess.TimeoutExpired as exc:
            raise InvalidInputArtifactError(f"HDFS probe timed out for {field}: {uri}") from exc
        if result.returncode != 0:
            raise InvalidInputArtifactError(f"HDFS path does not exist for {field}: {uri}")
        return

    if source_type != SourceType.HDFS_FUSE_PATH:
        raise InvalidInputArtifactError(f"{field} has unsupported source_type: {source_type}")

    path = Path(uri)
    mount_root = Path(config.runtime.storage.hdfs_fuse.mount_root if config.runtime.storage.hdfs_fuse else "/")
    try:
        path.resolve().relative_to(mount_root.resolve())
    except ValueError as exc:
        raise InvalidInputArtifactError(f"{field} escapes hdfs_fuse.mount_root: {uri}") from exc
    if not path.exists():
        raise InvalidInputArtifactError(f"{field} path does not exist: {uri}")
    if not path.is_dir() and not path.is_file():
        raise InvalidInputArtifactError(f"{field} is neither file nor directory: {uri}")
