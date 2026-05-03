from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "download_hf_assets.py"
SPEC = importlib.util.spec_from_file_location("download_hf_assets", SCRIPT_PATH)
assert SPEC is not None
download_hf_assets = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(download_hf_assets)


def test_download_hf_assets_defaults_are_full_local_assets() -> None:
    parser = download_hf_assets.build_parser()
    args = parser.parse_args([])
    download_hf_assets.normalize_dataset_limits(args)

    assert args.model_id == "Qwen/Qwen3-0.6B"
    assert args.dataset_id == "roneneldan/TinyStories"
    assert args.dataset_split == "train"
    assert args.dataset_text_column == "text"
    assert args.dataset_max_rows is None
    assert args.dataset_max_bytes is None


def test_dataset_max_bytes_zero_explicitly_disables_cap() -> None:
    parser = download_hf_assets.build_parser()
    args = parser.parse_args(["--dataset-max-bytes", "0"])
    download_hf_assets.normalize_dataset_limits(args)

    assert args.dataset_max_bytes is None
