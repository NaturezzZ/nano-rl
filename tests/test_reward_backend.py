from __future__ import annotations

from typing import Protocol

import pytest

from nano_rl.runtime.reward_backend import (
    DeterministicLengthRewardBackend,
    MockRewardBackend,
    RewardBackend,
    RewardResult,
)


def test_default_mock_reward_matches_existing_length_reward() -> None:
    backend = MockRewardBackend()

    result = backend.score("prompt", "x" * 25)

    assert isinstance(result, RewardResult)
    assert result.reward == pytest.approx(0.25)
    assert result.reward_source == "deterministic_length_reward"
    assert result.metadata == {}


def test_default_mock_reward_is_capped_at_one() -> None:
    backend = MockRewardBackend()

    result = backend.score("prompt", "x" * 250)

    assert result.reward == pytest.approx(1.0)
    assert result.reward_source == "deterministic_length_reward"


def test_mock_reward_backend_accepts_scale_cap_name_and_metadata() -> None:
    backend = MockRewardBackend(scale=10.0, cap=0.7, name="unit_reward")

    result = backend.score(
        "prompt",
        "x" * 20,
        metadata={"policy_version": 3, "request_id": "req-1"},
    )

    assert backend.scale == pytest.approx(10.0)
    assert backend.cap == pytest.approx(0.7)
    assert backend.name == "unit_reward"
    assert result.reward == pytest.approx(0.7)
    assert result.reward_source == "unit_reward"
    assert result.metadata == {"policy_version": 3, "request_id": "req-1"}


def test_deterministic_length_reward_backend_uses_same_defaults() -> None:
    backend = DeterministicLengthRewardBackend()

    result = backend.score("ignored prompt", "x" * 40)

    assert result.reward == pytest.approx(0.4)
    assert result.reward_source == "deterministic_length_reward"


def test_mock_reward_backend_validates_configuration() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        MockRewardBackend(scale=0)
    with pytest.raises(ValueError, match="cap must be non-negative"):
        MockRewardBackend(cap=-0.1)
    with pytest.raises(ValueError, match="name must be non-empty"):
        MockRewardBackend(name="")


def test_mock_reward_backend_satisfies_protocol_shape() -> None:
    backend: RewardBackend = MockRewardBackend()
    result = backend.score("p", "response", metadata={"k": "v"})

    assert result.reward == pytest.approx(0.08)
    assert result.reward_source == backend.name
    assert result.metadata == {"k": "v"}


def test_reward_backend_protocol_remains_structural() -> None:
    class CustomRewardBackend:
        name = "custom"

        def score(self, prompt: str, response: str, metadata=None) -> RewardResult:
            return RewardResult(reward=0.5, reward_source=self.name, metadata={"response": response})

    def accepts_reward_backend(backend: RewardBackend) -> RewardResult:
        return backend.score("prompt", "answer")

    result = accepts_reward_backend(CustomRewardBackend())

    assert result.reward == pytest.approx(0.5)
    assert result.reward_source == "custom"
    assert result.metadata == {"response": "answer"}

