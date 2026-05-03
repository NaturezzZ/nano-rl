#!/usr/bin/env python3
"""Download Hugging Face checkpoints and text datasets for nano-rl tests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_DATASET_ID = "roneneldan/TinyStories"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_DATASET_TEXT_COLUMN = "text"
DEFAULT_DATASET_MAX_BYTES: int | None = None


SIZE_SUFFIXES = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "m": 1000 * 1000,
    "mb": 1000 * 1000,
    "g": 1000 * 1000 * 1000,
    "gb": 1000 * 1000 * 1000,
    "ki": 1024,
    "kib": 1024,
    "mi": 1024 * 1024,
    "mib": 1024 * 1024,
    "gi": 1024 * 1024 * 1024,
    "gib": 1024 * 1024 * 1024,
}


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return value


def byte_size(raw: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*", raw)
    if not match:
        raise argparse.ArgumentTypeError(
            "value must be a byte size like 100M, 100MB, 95MiB, or 0"
        )

    value = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix not in SIZE_SUFFIXES:
        raise argparse.ArgumentTypeError(f"unsupported size suffix: {match.group(2)}")
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return int(value * SIZE_SUFFIXES[suffix])


def parse_patterns(values: list[str] | None) -> list[str] | None:
    if not values:
        return None

    patterns: list[str] = []
    for value in values:
        patterns.extend(part.strip() for part in value.split(",") if part.strip())
    return patterns or None


def safe_repo_name(repo_id: str, revision: str | None = None) -> str:
    parts = [repo_id.replace("/", "__")]
    if revision:
        parts.append(revision.replace("/", "__"))
    return "--".join(parts)


def clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def normalize_dataset_limits(args: argparse.Namespace) -> None:
    if args.skip_dataset:
        return

    if args.dataset_max_bytes == 0:
        args.dataset_max_bytes = None

    if args.dataset_max_bytes is not None and args.dataset_format == "snapshot":
        raise SystemExit(
            "`--dataset-format snapshot` cannot enforce `--dataset-max-bytes`. "
            "Use the default `save-to-disk` format, or pass `--dataset-max-bytes 0` "
            "to explicitly disable the size cap."
        )

    if args.dataset_max_bytes is not None and args.keep_dataset_columns:
        raise SystemExit(
            "`--keep-dataset-columns` cannot be combined with `--dataset-max-bytes`, "
            "because non-text columns may exceed the configured cap. Use the default "
            "text-only output, or pass `--dataset-max-bytes 0` to disable the cap."
        )


def import_huggingface_hub() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install with `pip install -r requirements.txt`."
        ) from exc
    return snapshot_download


def import_datasets() -> tuple[Any, Any]:
    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install -r requirements.txt`."
        ) from exc
    return Dataset, load_dataset


def download_model(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.skip_model:
        return None

    snapshot_download = import_huggingface_hub()
    model_name = args.model_output_name or safe_repo_name(args.model_id, args.model_revision)
    model_dir = args.output_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model_id,
        repo_type="model",
        revision=args.model_revision,
        local_dir=str(model_dir),
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        token=args.hf_token,
        allow_patterns=parse_patterns(args.model_allow_pattern),
        ignore_patterns=parse_patterns(args.model_ignore_pattern),
    )

    return {
        "kind": "model",
        "format": "huggingface_model_snapshot",
        "repo_id": args.model_id,
        "revision": args.model_revision,
        "path": str(model_dir.resolve()),
    }


def dataset_load_kwargs(args: argparse.Namespace, *, streaming: bool) -> dict[str, Any]:
    return clean_kwargs(
        {
            "path": args.dataset_id,
            "name": args.dataset_name,
            "split": args.dataset_split,
            "revision": args.dataset_revision,
            "streaming": streaming,
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "token": args.hf_token,
        }
    )


def save_streaming_dataset(args: argparse.Namespace, dataset_dir: Path) -> dict[str, Any]:
    Dataset, load_dataset = import_datasets()
    iterable = load_dataset(**dataset_load_kwargs(args, streaming=True))

    rows: list[dict[str, Any]] = []
    text_bytes = 0
    for row in iterable:
        if args.dataset_text_column not in row:
            available = ", ".join(sorted(row.keys()))
            raise SystemExit(
                f"Dataset row does not contain text column `{args.dataset_text_column}`. "
                f"Available columns: {available}"
            )

        text_value = row[args.dataset_text_column]
        if text_value is None:
            text_value = ""
        elif not isinstance(text_value, str):
            text_value = str(text_value)

        row_text_bytes = len(text_value.encode("utf-8"))
        if (
            args.dataset_max_bytes is not None
            and text_bytes + row_text_bytes > args.dataset_max_bytes
        ):
            if not rows:
                raise SystemExit(
                    f"First dataset row exceeds --dataset-max-bytes={args.dataset_max_bytes}."
                )
            break

        if args.keep_dataset_columns:
            normalized_row = dict(row)
            normalized_row[args.dataset_text_column] = text_value
        else:
            normalized_row = {args.dataset_text_column: text_value}

        rows.append(normalized_row)
        text_bytes += row_text_bytes
        if args.dataset_max_rows is not None and len(rows) >= args.dataset_max_rows:
            break

    if not rows:
        raise SystemExit("No dataset rows were downloaded.")

    dataset = Dataset.from_list(rows)
    dataset.save_to_disk(str(dataset_dir))
    return {
        "rows": len(dataset),
        "columns": list(dataset.column_names),
        "text_bytes": text_bytes,
        "max_text_bytes": args.dataset_max_bytes,
    }


def save_materialized_dataset(args: argparse.Namespace, dataset_dir: Path) -> dict[str, Any]:
    _, load_dataset = import_datasets()
    dataset = load_dataset(**dataset_load_kwargs(args, streaming=False))

    if args.dataset_text_column not in dataset.column_names:
        available = ", ".join(dataset.column_names)
        raise SystemExit(
            f"Dataset split does not contain text column `{args.dataset_text_column}`. "
            f"Available columns: {available}"
        )

    if args.dataset_max_rows:
        limit = min(args.dataset_max_rows, len(dataset))
        dataset = dataset.select(range(limit))

    if not args.keep_dataset_columns:
        remove_columns = [
            column for column in dataset.column_names if column != args.dataset_text_column
        ]
        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)

    dataset.save_to_disk(str(dataset_dir))
    return {"rows": len(dataset), "columns": list(dataset.column_names)}


def download_dataset_save_to_disk(args: argparse.Namespace) -> dict[str, Any]:
    dataset_name = args.dataset_output_name or safe_repo_name(
        args.dataset_id,
        args.dataset_revision,
    )
    dataset_dir = args.output_dir / "datasets" / dataset_name / args.dataset_split
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_max_rows or args.dataset_max_bytes is not None:
        saved = save_streaming_dataset(args, dataset_dir)
    else:
        saved = save_materialized_dataset(args, dataset_dir)

    manifest = {
        "kind": "dataset",
        "format": "huggingface_datasets_save_to_disk",
        "repo_id": args.dataset_id,
        "name": args.dataset_name,
        "revision": args.dataset_revision,
        "split": args.dataset_split,
        "text_column": args.dataset_text_column,
        "path": str(dataset_dir.resolve()),
        **saved,
    }
    (dataset_dir / "nano_rl_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def download_dataset_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_download = import_huggingface_hub()
    dataset_name = args.dataset_output_name or safe_repo_name(
        args.dataset_id,
        args.dataset_revision,
    )
    dataset_dir = args.output_dir / "datasets" / dataset_name / "snapshot"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        revision=args.dataset_revision,
        local_dir=str(dataset_dir),
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        token=args.hf_token,
    )

    return {
        "kind": "dataset",
        "format": "huggingface_dataset_repo_snapshot",
        "repo_id": args.dataset_id,
        "name": args.dataset_name,
        "revision": args.dataset_revision,
        "split": args.dataset_split,
        "text_column": args.dataset_text_column,
        "path": str(dataset_dir.resolve()),
    }


def download_dataset(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.skip_dataset:
        return None

    if args.dataset_format == "snapshot":
        return download_dataset_snapshot(args)
    return download_dataset_save_to_disk(args)


def write_manifest(args: argparse.Namespace, assets: list[dict[str, Any]]) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "assets_manifest.json"
    manifest = {
        "format_version": 1,
        "output_dir": str(args.output_dir.resolve()),
        "assets": assets,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Hugging Face model checkpoint and a pure-text Hugging Face "
            "dataset for nano-rl local tests. By default this downloads "
            f"{DEFAULT_MODEL_ID} and the complete {DEFAULT_DATASET_ID} "
            f"{DEFAULT_DATASET_SPLIT} split."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hf"),
        help="Directory where models/, datasets/, and assets_manifest.json are written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Optional Hugging Face access token. Public assets do not need this; "
            "HF_TOKEN/HUGGING_FACE_HUB_TOKEN environment variables also work."
        ),
    )

    model = parser.add_argument_group("model")
    model.add_argument("--skip-model", action="store_true", help="Do not download a model.")
    model.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model repo id. Default: {DEFAULT_MODEL_ID}",
    )
    model.add_argument("--model-revision", default=None, help="Optional model git revision.")
    model.add_argument(
        "--model-output-name",
        default=None,
        help="Optional local directory name under output-dir/models.",
    )
    model.add_argument(
        "--model-allow-pattern",
        action="append",
        help="Optional comma-separated or repeated huggingface_hub allow_patterns.",
    )
    model.add_argument(
        "--model-ignore-pattern",
        action="append",
        help="Optional comma-separated or repeated huggingface_hub ignore_patterns.",
    )

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Do not download a dataset.",
    )
    dataset.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_DATASET_ID}",
    )
    dataset.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset config/name passed to datasets.load_dataset.",
    )
    dataset.add_argument(
        "--dataset-split",
        default=DEFAULT_DATASET_SPLIT,
        help=f"Dataset split to download. Default: {DEFAULT_DATASET_SPLIT}",
    )
    dataset.add_argument(
        "--dataset-revision",
        default=None,
        help="Optional dataset git revision.",
    )
    dataset.add_argument(
        "--dataset-text-column",
        default=DEFAULT_DATASET_TEXT_COLUMN,
        help=f"Pure-text column to keep or validate. Default: {DEFAULT_DATASET_TEXT_COLUMN}",
    )
    dataset.add_argument(
        "--dataset-output-name",
        default=None,
        help="Optional local directory name under output-dir/datasets.",
    )
    dataset.add_argument(
        "--dataset-max-rows",
        type=positive_int,
        default=None,
        help=(
            "Optional row limit for quick tests. When set, streaming download is used "
            "and the sampled rows are saved in Hugging Face datasets format."
        ),
    )
    dataset.add_argument(
        "--dataset-max-bytes",
        type=byte_size,
        default=DEFAULT_DATASET_MAX_BYTES,
        help=(
            "Optional maximum UTF-8 bytes from dataset-text-column to save. "
            "By default the complete split is saved. Pass 100M or another size "
            "for smaller smoke-test assets; pass 0 to explicitly disable the cap."
        ),
    )
    dataset.add_argument(
        "--dataset-format",
        choices=["save-to-disk", "snapshot"],
        default="save-to-disk",
        help=(
            "`save-to-disk` writes a load_from_disk-compatible text dataset; "
            "`snapshot` preserves the upstream dataset repository files."
        ),
    )
    dataset.add_argument(
        "--keep-dataset-columns",
        action="store_true",
        help="Keep all dataset columns instead of saving only dataset-text-column.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser()
    args.cache_dir = args.cache_dir.expanduser() if args.cache_dir else None
    normalize_dataset_limits(args)

    assets = [
        asset
        for asset in (download_model(args), download_dataset(args))
        if asset is not None
    ]
    manifest_path = write_manifest(args, assets)

    print(f"Wrote manifest: {manifest_path.resolve()}")
    for asset in assets:
        print(f"{asset['kind']}: {asset['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
