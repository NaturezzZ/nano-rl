"""Reward backend interfaces and deterministic mock implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class RewardResult:
    reward: float
    reward_source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RewardBackend(Protocol):
    name: str

    def score(
        self,
        prompt: str,
        response: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> RewardResult:
        """Score one prompt/response pair."""


class MockRewardBackend:
    """Deterministic length reward for CPU-only mock runtime paths."""

    def __init__(
        self,
        *,
        scale: float = 100.0,
        cap: float = 1.0,
        name: str = "deterministic_length_reward",
    ) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        if cap < 0:
            raise ValueError("cap must be non-negative")
        if not name:
            raise ValueError("name must be non-empty")
        self.scale = float(scale)
        self.cap = float(cap)
        self.name = name

    def score(
        self,
        prompt: str,
        response: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> RewardResult:
        del prompt
        reward = min(self.cap, len(response) / self.scale)
        return RewardResult(
            reward=float(reward),
            reward_source=self.name,
            metadata=dict(metadata or {}),
        )


class DeterministicLengthRewardBackend(MockRewardBackend):
    """Named backend for the default deterministic length reward behavior."""


def build_reward_backend(config: Any | None = None) -> RewardBackend:
    """Build the configured reward backend.

    v0.1 only ships the deterministic mock reward backend. The factory keeps
    the high-level reward boundary explicit so a real reward model can replace
    it without changing rollout actors.
    """

    data = _to_mapping(config)
    backend = str(data.get("backend", "mock"))
    if backend != "mock":
        raise ValueError(f"unsupported reward backend: {backend}")
    options = data.get("mock", data)
    if not isinstance(options, Mapping):
        raise TypeError("reward.mock must be a mapping")
    return MockRewardBackend(
        scale=float(options.get("scale", 100.0)),
        cap=float(options.get("cap", 1.0)),
        name=str(options.get("name", "deterministic_length_reward")),
    )


def _to_mapping(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    model_dump = getattr(config, "model_dump", None)
    if model_dump is not None:
        return dict(model_dump(mode="python"))
    raise TypeError("reward backend config must be a mapping or pydantic model")
