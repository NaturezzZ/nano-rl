"""Prompt data source adapters for local and mock runtime paths."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


MOCK_INLINE = "mock_inline"
MOCK_GENERATED = "mock_generated"
MOCK_JSONL = "mock_jsonl"
SUPPORTED_MOCK_SOURCES = (MOCK_INLINE, MOCK_GENERATED, MOCK_JSONL)


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One prompt payload consumed by rollout scheduling."""

    prompt_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptSource(Protocol):
    """Common interface for prompt providers."""

    def iter_prompts(self, *, limit: int | None = None) -> Iterator[PromptRecord]:
        """Yield prompt records, optionally capped by ``limit``."""


class MockInlinePromptSource:
    """Prompt source backed by prompts embedded directly in config."""

    def __init__(
        self,
        prompts: Iterable[str | Mapping[str, Any]],
        *,
        prompt_column: str = "prompt",
        repeat: int = 1,
    ) -> None:
        if repeat < 1:
            raise ValueError("mock_inline.repeat must be >= 1")
        base_records = tuple(
            _record_from_inline_item(item, index=index, prompt_column=prompt_column)
            for index, item in enumerate(prompts)
        )
        records: list[PromptRecord] = []
        for repeat_index in range(repeat):
            for index, record in enumerate(base_records):
                metadata = dict(record.metadata)
                if repeat > 1:
                    metadata = {"repeat_index": repeat_index, **metadata, "prompt_index": index}
                records.append(
                    PromptRecord(
                        prompt_id=f"{record.prompt_id}:repeat-{repeat_index}" if repeat > 1 else record.prompt_id,
                        prompt=record.prompt,
                        metadata=metadata,
                    )
                )
        self._records = tuple(records)

    def iter_prompts(self, *, limit: int | None = None) -> Iterator[PromptRecord]:
        yield from _limit_records(self._records, limit)


class MockGeneratedPromptSource:
    """Deterministic generated prompt source for CPU-only smoke paths."""

    def __init__(
        self,
        *,
        count: int,
        template: str = "mock prompt {index}",
        start_index: int = 0,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if count < 0:
            raise ValueError("mock_generated.count must be >= 0")
        indices = list(range(start_index, start_index + count))
        if shuffle:
            random.Random(seed).shuffle(indices)
        self._records = tuple(
            PromptRecord(
                prompt_id=f"{MOCK_GENERATED}:{index}",
                prompt=template.format(index=index),
                metadata={"index": index, "ordinal": ordinal},
            )
            for ordinal, index in enumerate(indices)
        )

    def iter_prompts(self, *, limit: int | None = None) -> Iterator[PromptRecord]:
        yield from _limit_records(self._records, limit)


class MockJsonlPromptSource:
    """Prompt source backed by a local JSONL file."""

    def __init__(self, data_path: str | Path, *, prompt_column: str = "prompt") -> None:
        if not prompt_column:
            raise ValueError("mock_jsonl.prompt_column must be non-empty")
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"mock_jsonl.data_path does not exist or is not a file: {path}")
        self._records = tuple(_read_jsonl_records(path, prompt_column=prompt_column))

    def iter_prompts(self, *, limit: int | None = None) -> Iterator[PromptRecord]:
        yield from _limit_records(self._records, limit)


def build_prompt_source(config: Any, *, seed: int | None = None) -> PromptSource:
    """Build a prompt source from LaunchConfig, DataConfig, or a data dict."""

    resolved_seed = _coerce_seed(config, seed)
    data = _coerce_data_mapping(config)
    source_type = _coerce_source_type(data.get("source_type"))
    options = _source_options(data, source_type)

    if source_type == MOCK_INLINE:
        prompts = options.get("prompts")
        if prompts is None:
            raise ValueError("mock_inline.prompts is required")
        if isinstance(prompts, str) or not isinstance(prompts, Iterable):
            raise TypeError("mock_inline.prompts must be an iterable of strings or objects")
        return MockInlinePromptSource(
            prompts,
            prompt_column=str(options.get("prompt_column", "prompt")),
            repeat=_coerce_int(options.get("repeat", 1), "mock_inline.repeat"),
        )

    if source_type == MOCK_GENERATED:
        if "count" not in options:
            raise ValueError("mock_generated.count is required")
        return MockGeneratedPromptSource(
            count=_coerce_int(options["count"], "mock_generated.count"),
            template=str(options.get("template", "mock prompt {index}")),
            start_index=_coerce_int(options.get("start_index", 0), "mock_generated.start_index"),
            shuffle=_coerce_shuffle(options.get("shuffle", False)),
            seed=_coerce_int(options.get("seed", resolved_seed), "mock_generated.seed"),
        )

    if source_type == MOCK_JSONL:
        data_path = options.get("data_path") or options.get("path")
        if not data_path:
            raise ValueError("mock_jsonl.data_path is required")
        return MockJsonlPromptSource(data_path, prompt_column=str(options.get("prompt_column", "prompt")))

    supported = ", ".join(SUPPORTED_MOCK_SOURCES)
    raise ValueError(f"Unsupported prompt source '{source_type}'. Supported mock sources: {supported}")


def _coerce_data_mapping(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        if "source_type" not in config and isinstance(config.get("data"), Mapping):
            return dict(config["data"])
        return dict(config)

    data = getattr(config, "data", None)
    if data is not None:
        return _coerce_data_mapping(data)

    if hasattr(config, "model_dump"):
        return dict(config.model_dump())

    if hasattr(config, "dict"):
        return dict(config.dict())

    values: dict[str, Any] = {}
    for key in ("source_type", "data_path", "prompt_column"):
        if hasattr(config, key):
            values[key] = getattr(config, key)
    if values:
        return values

    raise TypeError("build_prompt_source expects LaunchConfig, DataConfig, or a data dict")


def _coerce_seed(config: Any, explicit_seed: int | None) -> int:
    if explicit_seed is not None:
        return explicit_seed
    mock_config = getattr(config, "mock", None)
    seed = getattr(mock_config, "seed", None)
    if seed is not None:
        return int(seed)
    if isinstance(config, Mapping):
        raw_mock = config.get("mock")
        if isinstance(raw_mock, Mapping) and raw_mock.get("seed") is not None:
            return int(raw_mock["seed"])
    return 0


def _coerce_source_type(value: Any) -> str:
    if value is None:
        raise ValueError("data.source_type is required")
    enum_value = getattr(value, "value", value)
    return str(enum_value)


def _source_options(data: Mapping[str, Any], source_type: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    nested = data.get(source_type)
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise TypeError(f"{source_type} options must be a mapping")
        options.update(nested)

    nested_keys = set(SUPPORTED_MOCK_SOURCES)
    for key, value in data.items():
        if key == "source_type" or key in nested_keys:
            continue
        options[key] = value
    return options


def _record_from_inline_item(item: str | Mapping[str, Any], *, index: int, prompt_column: str) -> PromptRecord:
    if isinstance(item, str):
        return PromptRecord(prompt_id=f"{MOCK_INLINE}:{index}", prompt=item, metadata={})

    if not isinstance(item, Mapping):
        raise TypeError("mock_inline.prompts entries must be strings or objects")

    if prompt_column not in item:
        raise ValueError(f"mock_inline prompt entry {index} is missing prompt_column '{prompt_column}'")
    prompt = item[prompt_column]
    if not isinstance(prompt, str):
        raise TypeError(f"mock_inline prompt entry {index} column '{prompt_column}' must be a string")

    raw_metadata = item.get("metadata", {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, Mapping):
        raise TypeError(f"mock_inline prompt entry {index} metadata must be an object")

    metadata = dict(raw_metadata)
    for key, value in item.items():
        if key not in {prompt_column, "metadata", "prompt_id", "id"}:
            metadata[key] = value

    prompt_id = item.get("prompt_id", item.get("id", f"{MOCK_INLINE}:{index}"))
    return PromptRecord(prompt_id=str(prompt_id), prompt=prompt, metadata=metadata)


def _read_jsonl_records(path: Path, *, prompt_column: str) -> Iterator[PromptRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"mock_jsonl invalid JSON on line {line_number}: {exc.msg}") from exc

            if not isinstance(row, Mapping):
                raise ValueError(f"mock_jsonl line {line_number} must be a JSON object")
            if prompt_column not in row:
                raise ValueError(f"mock_jsonl line {line_number} is missing prompt_column '{prompt_column}'")

            prompt = row[prompt_column]
            if not isinstance(prompt, str):
                raise TypeError(f"mock_jsonl line {line_number} column '{prompt_column}' must be a string")

            raw_metadata = row.get("metadata", {})
            if raw_metadata is None:
                raw_metadata = {}
            if not isinstance(raw_metadata, Mapping):
                raise TypeError(f"mock_jsonl line {line_number} metadata must be an object")

            metadata = dict(raw_metadata)
            for key, value in row.items():
                if key not in {prompt_column, "metadata", "prompt_id", "id"}:
                    metadata[key] = value

            prompt_id = row.get("prompt_id", row.get("id", f"{MOCK_JSONL}:{line_number}"))
            yield PromptRecord(prompt_id=str(prompt_id), prompt=prompt, metadata=metadata)


def _limit_records(records: Iterable[PromptRecord], limit: int | None) -> Iterator[PromptRecord]:
    if limit is not None and limit < 0:
        raise ValueError("PromptSource.iter_prompts limit must be >= 0")

    yielded = 0
    for record in records:
        if limit is not None and yielded >= limit:
            return
        yielded += 1
        yield PromptRecord(prompt_id=record.prompt_id, prompt=record.prompt, metadata=dict(record.metadata))


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _coerce_shuffle(value: Any) -> bool:
    if value in {False, None, "false", "False", "none", "None"}:
        return False
    if value in {True, "true", "True", "deterministic"}:
        return True
    raise ValueError("mock_generated.shuffle must be false or deterministic")
