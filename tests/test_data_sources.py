from __future__ import annotations

import csv

import pytest

from nano_rl.config import DataConfig
from nano_rl.runtime.data_sources import PromptRecord, build_prompt_source


def test_mock_inline_builds_prompt_records_from_flat_dict() -> None:
    source = build_prompt_source(
        {
            "source_type": "mock_inline",
            "prompts": [
                "first prompt",
                {
                    "prompt": "second prompt",
                    "prompt_id": "custom-id",
                    "metadata": {"split": "eval"},
                    "difficulty": "easy",
                },
            ],
        }
    )

    assert list(source.iter_prompts()) == [
        PromptRecord(prompt_id="mock_inline:0", prompt="first prompt", metadata={}),
        PromptRecord(
            prompt_id="custom-id",
            prompt="second prompt",
            metadata={"split": "eval", "difficulty": "easy"},
        ),
    ]


def test_mock_inline_supports_nested_options_and_limit() -> None:
    source = build_prompt_source(
        {
            "source_type": "mock_inline",
            "mock_inline": {
                "prompt_column": "question",
                "prompts": [
                    {"question": "q0", "id": "q0"},
                    {"question": "q1", "id": "q1"},
                ],
            },
        }
    )

    assert list(source.iter_prompts(limit=1)) == [
        PromptRecord(prompt_id="q0", prompt="q0", metadata={}),
    ]


def test_mock_generated_uses_template_start_index_and_seeded_shuffle() -> None:
    config = {
        "source_type": "mock_generated",
        "count": 5,
        "template": "question-{index}",
        "start_index": 10,
        "shuffle": True,
        "seed": 7,
    }

    records = list(build_prompt_source(config).iter_prompts())
    repeated = list(build_prompt_source(config).iter_prompts())

    assert records == repeated
    assert [record.prompt for record in records] != [f"question-{index}" for index in range(10, 15)]
    assert sorted(record.metadata["index"] for record in records) == [10, 11, 12, 13, 14]


def test_mock_generated_without_shuffle_keeps_index_order() -> None:
    source = build_prompt_source(
        {
            "source_type": "mock_generated",
            "mock_generated": {
                "count": 3,
                "template": "prompt {index}",
                "start_index": 4,
            },
        }
    )

    assert [record.prompt for record in source.iter_prompts()] == [
        "prompt 4",
        "prompt 5",
        "prompt 6",
    ]


def test_mock_profile_shapes_prompt_tokens_and_loader_metadata() -> None:
    source = build_prompt_source(
        {
            "source_type": "mock_generated",
            "mock_profile": {
                "enabled": True,
                "sleep_enabled": False,
                "prompt_length_distribution": "uniform",
                "min_prompt_tokens": 12,
                "mean_prompt_tokens": 16,
                "max_prompt_tokens": 24,
                "pad_prompts": True,
                "load_base_ms": 3,
                "load_ms_per_1k_tokens": 10,
                "max_load_ms": 10,
            },
            "mock_generated": {
                "count": 2,
                "template": "prompt {index}",
            },
        }
    )

    records = list(source.iter_prompts())

    assert len(records) == 2
    assert all(12 <= record.metadata["prompt_tokens"] <= 24 for record in records)
    assert all(record.metadata["mock_data"]["source_type"] == "mock_generated" for record in records)
    assert all(record.metadata["mock_data"]["target_prompt_tokens"] >= 12 for record in records)
    assert all(record.prompt.startswith("prompt ") for record in records)


def test_mock_jsonl_reads_prompt_column_and_preserves_metadata(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"question": "hello", "prompt_id": "p0", "metadata": {"split": "train"}, "topic": "math"}\n'
        '{"question": "world", "metadata": {"split": "eval"}}\n',
        encoding="utf-8",
    )

    source = build_prompt_source(
        {
            "source_type": "mock_jsonl",
            "data_path": str(path),
            "prompt_column": "question",
        }
    )

    assert list(source.iter_prompts()) == [
        PromptRecord(prompt_id="p0", prompt="hello", metadata={"split": "train", "topic": "math"}),
        PromptRecord(prompt_id="mock_jsonl:2", prompt="world", metadata={"split": "eval"}),
    ]


def test_local_csv_reads_prompt_column_and_preserves_metadata(tmp_path) -> None:
    path = tmp_path / "prompts.csv"
    path.write_text(
        "act,prompt,for_devs,type\n"
        'Ethereum Developer,"write a tiny contract",TRUE,TEXT\n'
        'Linux Terminal,"reply with pwd output",FALSE,TEXT\n',
        encoding="utf-8",
    )

    source = build_prompt_source(
        {
            "source_type": "local_csv",
            "data_path": str(path),
            "prompt_column": "prompt",
            "local_csv": {"encoding": "utf-8"},
        }
    )

    assert list(source.iter_prompts(limit=2)) == [
        PromptRecord(
            prompt_id="local_csv:1",
            prompt="write a tiny contract",
            metadata={"act": "Ethereum Developer", "for_devs": "TRUE", "type": "TEXT"},
        ),
        PromptRecord(
            prompt_id="local_csv:2",
            prompt="reply with pwd output",
            metadata={"act": "Linux Terminal", "for_devs": "FALSE", "type": "TEXT"},
        ),
    ]


def test_local_csv_validates_prompt_column(tmp_path) -> None:
    path = tmp_path / "prompts.csv"
    path.write_text("text\nhello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing prompt_column 'prompt'"):
        build_prompt_source(
            {
                "source_type": "local_csv",
                "data_path": str(path),
                "prompt_column": "prompt",
            }
        )


def test_local_csv_raises_csv_field_limit_for_long_prompts(tmp_path) -> None:
    path = tmp_path / "prompts.csv"
    long_prompt = "token " * 200
    path.write_text(f'prompt\n"{long_prompt}"\n', encoding="utf-8")
    csv.field_size_limit(32)

    records = list(
        build_prompt_source(
            {
                "source_type": "local_csv",
                "data_path": str(path),
                "prompt_column": "prompt",
            }
        ).iter_prompts()
    )

    assert records == [
        PromptRecord(prompt_id="local_csv:1", prompt=long_prompt, metadata={}),
    ]


def test_mock_jsonl_validates_prompt_column(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt": "hello"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing prompt_column 'question'"):
        build_prompt_source(
            {
                "source_type": "mock_jsonl",
                "data_path": str(path),
                "prompt_column": "question",
            }
        )


def test_build_prompt_source_accepts_data_config_and_reports_unsupported_sources() -> None:
    data = DataConfig(source_type="hdfs_uri", data_path="/mnt/hdfs/prompts.jsonl")

    with pytest.raises(ValueError, match="Unsupported prompt source 'hdfs_uri'"):
        build_prompt_source(data)
