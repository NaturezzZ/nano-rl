from __future__ import annotations

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
