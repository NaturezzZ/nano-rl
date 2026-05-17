"""Prompt data source adapters for local and mock runtime paths."""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol


MOCK_INLINE = "mock_inline"
MOCK_GENERATED = "mock_generated"
MOCK_JSONL = "mock_jsonl"
LOCAL_CSV = "local_csv"
MOCK_PROFILE = "mock_profile"
SUPPORTED_PROMPT_SOURCES = (MOCK_INLINE, MOCK_GENERATED, MOCK_JSONL, LOCAL_CSV)


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One prompt payload consumed by rollout scheduling."""

    prompt_id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MockDataProfile:
    """Optional mock-only data shape and loader latency profile."""

    enabled: bool = False
    sleep_enabled: bool = False
    prompt_length_distribution: str = "fixed"
    min_prompt_tokens: int = 1
    mean_prompt_tokens: int = 64
    max_prompt_tokens: int = 512
    length_jitter: float = 0.65
    pad_prompts: bool = False
    load_base_ms: float = 0.0
    load_ms_per_1k_tokens: float = 0.0
    load_jitter_ms: float = 0.0
    max_load_ms: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MockDataProfile":
        raw = dict(value or {})
        if not raw:
            return cls()
        profile = cls(
            enabled=_coerce_bool(raw.get("enabled", False), "mock_profile.enabled"),
            sleep_enabled=_coerce_bool(raw.get("sleep_enabled", False), "mock_profile.sleep_enabled"),
            prompt_length_distribution=str(raw.get("prompt_length_distribution", "fixed")),
            min_prompt_tokens=_coerce_int(raw.get("min_prompt_tokens", 1), "mock_profile.min_prompt_tokens"),
            mean_prompt_tokens=_coerce_int(raw.get("mean_prompt_tokens", 64), "mock_profile.mean_prompt_tokens"),
            max_prompt_tokens=_coerce_int(raw.get("max_prompt_tokens", 512), "mock_profile.max_prompt_tokens"),
            length_jitter=_coerce_float(raw.get("length_jitter", 0.65), "mock_profile.length_jitter"),
            pad_prompts=_coerce_bool(raw.get("pad_prompts", False), "mock_profile.pad_prompts"),
            load_base_ms=_coerce_float(raw.get("load_base_ms", 0), "mock_profile.load_base_ms"),
            load_ms_per_1k_tokens=_coerce_float(raw.get("load_ms_per_1k_tokens", 0), "mock_profile.load_ms_per_1k_tokens"),
            load_jitter_ms=_coerce_float(raw.get("load_jitter_ms", 0), "mock_profile.load_jitter_ms"),
            max_load_ms=_coerce_float(raw.get("max_load_ms", 0), "mock_profile.max_load_ms"),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.prompt_length_distribution not in {"fixed", "uniform", "lognormal", "chat_mixture"}:
            raise ValueError("mock_profile.prompt_length_distribution must be fixed, uniform, lognormal, or chat_mixture")
        if self.min_prompt_tokens < 1:
            raise ValueError("mock_profile.min_prompt_tokens must be >= 1")
        if self.max_prompt_tokens < self.min_prompt_tokens:
            raise ValueError("mock_profile.max_prompt_tokens must be >= min_prompt_tokens")
        if self.mean_prompt_tokens < 1 or self.mean_prompt_tokens > self.max_prompt_tokens:
            raise ValueError("mock_profile.mean_prompt_tokens must be between 1 and max_prompt_tokens")
        if min(
            self.length_jitter,
            self.load_base_ms,
            self.load_ms_per_1k_tokens,
            self.load_jitter_ms,
            self.max_load_ms,
        ) < 0:
            raise ValueError("mock_profile numeric timing fields must be >= 0")


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
        profile: MockDataProfile | None = None,
    ) -> None:
        if repeat < 1:
            raise ValueError("mock_inline.repeat must be >= 1")
        self._profile = profile or MockDataProfile()
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
                    _apply_mock_data_profile(
                        PromptRecord(
                            prompt_id=f"{record.prompt_id}:repeat-{repeat_index}" if repeat > 1 else record.prompt_id,
                            prompt=record.prompt,
                            metadata=metadata,
                        ),
                        source_type=MOCK_INLINE,
                        profile=self._profile,
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
        profile: MockDataProfile | None = None,
    ) -> None:
        if count < 0:
            raise ValueError("mock_generated.count must be >= 0")
        self._profile = profile or MockDataProfile()
        indices = list(range(start_index, start_index + count))
        if shuffle:
            random.Random(seed).shuffle(indices)
        self._records = tuple(
            _apply_mock_data_profile(
                PromptRecord(
                    prompt_id=f"{MOCK_GENERATED}:{index}",
                    prompt=template.format(index=index),
                    metadata={"index": index, "ordinal": ordinal},
                ),
                source_type=MOCK_GENERATED,
                profile=self._profile,
            )
            for ordinal, index in enumerate(indices)
        )

    def iter_prompts(self, *, limit: int | None = None) -> Iterator[PromptRecord]:
        yield from _limit_records(self._records, limit)


class MockJsonlPromptSource:
    """Prompt source backed by a local JSONL file."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        prompt_column: str = "prompt",
        profile: MockDataProfile | None = None,
    ) -> None:
        if not prompt_column:
            raise ValueError("mock_jsonl.prompt_column must be non-empty")
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"mock_jsonl.data_path does not exist or is not a file: {path}")
        self._profile = profile or MockDataProfile()
        self._records = tuple(
            _apply_mock_data_profile(record, source_type=MOCK_JSONL, profile=self._profile)
            for record in _read_jsonl_records(path, prompt_column=prompt_column)
        )

    def iter_prompts(self, *, limit: int | None = None) -> Iterator[PromptRecord]:
        yield from _limit_records(self._records, limit)


class LocalCsvPromptSource:
    """Prompt source backed by a local CSV file."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        prompt_column: str = "prompt",
        encoding: str = "utf-8",
        profile: MockDataProfile | None = None,
    ) -> None:
        if not prompt_column:
            raise ValueError("local_csv.prompt_column must be non-empty")
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"local_csv.data_path does not exist or is not a file: {path}")
        self._profile = profile or MockDataProfile()
        self._records = tuple(
            _apply_mock_data_profile(record, source_type=LOCAL_CSV, profile=self._profile)
            for record in _read_csv_records(path, prompt_column=prompt_column, encoding=encoding)
        )

    def iter_prompts(self, *, limit: int | None = None) -> Iterator[PromptRecord]:
        yield from _limit_records(self._records, limit)


def build_prompt_source(config: Any, *, seed: int | None = None) -> PromptSource:
    """Build a prompt source from LaunchConfig, DataConfig, or a data dict."""

    resolved_seed = _coerce_seed(config, seed)
    data = _coerce_data_mapping(config)
    source_type = _coerce_source_type(data.get("source_type"))
    options = _source_options(data, source_type)
    profile = MockDataProfile.from_mapping(data.get(MOCK_PROFILE) if isinstance(data.get(MOCK_PROFILE), Mapping) else None)

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
            profile=profile,
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
            profile=profile,
        )

    if source_type == MOCK_JSONL:
        data_path = options.get("data_path") or options.get("path")
        if not data_path:
            raise ValueError("mock_jsonl.data_path is required")
        return MockJsonlPromptSource(
            data_path,
            prompt_column=str(options.get("prompt_column", "prompt")),
            profile=profile,
        )

    if source_type == LOCAL_CSV:
        data_path = options.get("data_path") or options.get("path")
        if not data_path:
            raise ValueError("local_csv.data_path is required")
        return LocalCsvPromptSource(
            data_path,
            prompt_column=str(options.get("prompt_column", "prompt")),
            encoding=str(options.get("encoding", "utf-8")),
            profile=profile,
        )

    supported = ", ".join(SUPPORTED_PROMPT_SOURCES)
    raise ValueError(f"Unsupported prompt source '{source_type}'. Supported prompt sources: {supported}")


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

    nested_keys = set(SUPPORTED_PROMPT_SOURCES) | {MOCK_PROFILE}
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


def _read_csv_records(path: Path, *, prompt_column: str, encoding: str) -> Iterator[PromptRecord]:
    _raise_csv_field_size_limit()
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("local_csv file must include a header row")
        if prompt_column not in reader.fieldnames:
            raise ValueError(f"local_csv header is missing prompt_column '{prompt_column}'")

        for row_number, row in enumerate(reader, start=1):
            prompt = row.get(prompt_column)
            if prompt is None:
                raise ValueError(f"local_csv row {row_number} is missing prompt_column '{prompt_column}'")
            if not isinstance(prompt, str):
                raise TypeError(f"local_csv row {row_number} column '{prompt_column}' must be a string")

            metadata = {
                key: value
                for key, value in row.items()
                if key not in {prompt_column, "prompt_id", "id"} and key is not None
            }
            prompt_id = row.get("prompt_id") or row.get("id") or f"{LOCAL_CSV}:{row_number}"
            yield PromptRecord(prompt_id=str(prompt_id), prompt=prompt, metadata=metadata)


def _raise_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


def _limit_records(records: Iterable[PromptRecord], limit: int | None) -> Iterator[PromptRecord]:
    if limit is not None and limit < 0:
        raise ValueError("PromptSource.iter_prompts limit must be >= 0")

    yielded = 0
    for record in records:
        if limit is not None and yielded >= limit:
            return
        yielded += 1
        _sleep_for_record(record)
        yield PromptRecord(prompt_id=record.prompt_id, prompt=record.prompt, metadata=dict(record.metadata))


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _coerce_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a number") from exc


def _coerce_bool(value: Any, field_name: str) -> bool:
    if value in {True, "true", "True", "1", 1}:
        return True
    if value in {False, None, "false", "False", "0", 0}:
        return False
    raise ValueError(f"{field_name} must be boolean")


def _coerce_shuffle(value: Any) -> bool:
    if value in {False, None, "false", "False", "none", "None"}:
        return False
    if value in {True, "true", "True", "deterministic"}:
        return True
    raise ValueError("mock_generated.shuffle must be false or deterministic")


def _apply_mock_data_profile(
    record: PromptRecord,
    *,
    source_type: str,
    profile: MockDataProfile,
) -> PromptRecord:
    if not profile.enabled:
        return record

    current_tokens = _estimate_prompt_tokens(record.prompt)
    target_tokens = _sample_prompt_token_count(
        profile,
        seed_payload=f"{source_type}:{record.prompt_id}:{record.prompt}".encode("utf-8"),
        fallback=current_tokens,
    )
    prompt = record.prompt
    if profile.pad_prompts and current_tokens < target_tokens:
        prompt = _pad_prompt(prompt, target_tokens, seed_payload=f"{record.prompt_id}:{record.prompt}".encode("utf-8"))
        current_tokens = _estimate_prompt_tokens(prompt)

    load_sleep_ms = _sample_load_sleep_ms(
        profile,
        prompt_tokens=current_tokens,
        seed_payload=f"load:{source_type}:{record.prompt_id}:{current_tokens}".encode("utf-8"),
    )
    metadata = dict(record.metadata)
    metadata["prompt_tokens"] = current_tokens
    metadata["mock_data"] = {
        "source_type": source_type,
        "prompt_length_distribution": profile.prompt_length_distribution,
        "target_prompt_tokens": target_tokens,
        "prompt_tokens": current_tokens,
        "load_sleep_ms": round(load_sleep_ms, 3),
        "sleep_enabled": profile.sleep_enabled,
    }
    return PromptRecord(prompt_id=record.prompt_id, prompt=prompt, metadata=metadata)


def _sample_prompt_token_count(profile: MockDataProfile, *, seed_payload: bytes, fallback: int) -> int:
    if profile.prompt_length_distribution == "fixed":
        raw = profile.mean_prompt_tokens if profile.pad_prompts else fallback
    else:
        rng = _stable_rng(seed_payload)
        if profile.prompt_length_distribution == "uniform":
            raw = rng.randint(profile.min_prompt_tokens, profile.max_prompt_tokens)
        elif profile.prompt_length_distribution == "lognormal":
            median = max(profile.min_prompt_tokens, profile.mean_prompt_tokens)
            raw = int(round(rng.lognormvariate(_safe_log(median), max(0.01, profile.length_jitter))))
        elif profile.prompt_length_distribution == "chat_mixture":
            raw = _sample_chat_mixture_tokens(profile, rng)
        else:
            raw = fallback
    return _clamp_int(raw, profile.min_prompt_tokens, profile.max_prompt_tokens)


def _sample_chat_mixture_tokens(profile: MockDataProfile, rng: random.Random) -> int:
    draw = rng.random()
    if draw < 0.55:
        low = profile.min_prompt_tokens
        high = max(low, min(profile.max_prompt_tokens, profile.mean_prompt_tokens))
    elif draw < 0.85:
        low = max(profile.min_prompt_tokens, profile.mean_prompt_tokens // 2)
        high = max(low, min(profile.max_prompt_tokens, profile.mean_prompt_tokens * 2))
    elif draw < 0.97:
        low = max(profile.min_prompt_tokens, profile.mean_prompt_tokens)
        high = max(low, min(profile.max_prompt_tokens, profile.mean_prompt_tokens * 4))
    else:
        low = max(profile.min_prompt_tokens, profile.max_prompt_tokens // 2)
        high = profile.max_prompt_tokens
    return rng.randint(low, high)


def _sample_load_sleep_ms(profile: MockDataProfile, *, prompt_tokens: int, seed_payload: bytes) -> float:
    if not profile.sleep_enabled:
        return 0.0
    base = profile.load_base_ms + (prompt_tokens / 1000.0) * profile.load_ms_per_1k_tokens
    jitter = 0.0
    if profile.load_jitter_ms > 0:
        jitter = _stable_rng(seed_payload).uniform(0, profile.load_jitter_ms)
    sleep_ms = base + jitter
    if profile.max_load_ms > 0:
        sleep_ms = min(sleep_ms, profile.max_load_ms)
    return max(0.0, sleep_ms)


def _sleep_for_record(record: PromptRecord) -> None:
    mock_data = record.metadata.get("mock_data")
    if not isinstance(mock_data, Mapping) or not mock_data.get("sleep_enabled"):
        return
    sleep_ms = float(mock_data.get("load_sleep_ms", 0) or 0)
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)


def _estimate_prompt_tokens(prompt: str) -> int:
    stripped = prompt.strip()
    if not stripped:
        return 1
    return max(1, len(stripped.split()))


def _pad_prompt(prompt: str, target_tokens: int, *, seed_payload: bytes) -> str:
    current = _estimate_prompt_tokens(prompt)
    if current >= target_tokens:
        return prompt
    needed = target_tokens - current
    words: list[str] = []
    block = seed_payload
    while len(words) < needed:
        digest = sha256(block).hexdigest()
        for index in range(0, len(digest), 8):
            if len(words) >= needed:
                break
            words.append(f"ctx_{digest[index:index + 8]}")
        block = digest.encode("utf-8")
    suffix = " ".join(words)
    return f"{prompt} {suffix}" if prompt else suffix


def _stable_rng(seed_payload: bytes) -> random.Random:
    seed = int.from_bytes(sha256(seed_payload).digest()[:8], "big")
    return random.Random(seed)


def _safe_log(value: int | float) -> float:
    import math

    return math.log(max(float(value), 1.0))


def _clamp_int(value: int | float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))
