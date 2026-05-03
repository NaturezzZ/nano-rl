"""Shared exception types."""

from __future__ import annotations


class NanoRLError(Exception):
    """Base class for nano-rl errors."""


class ConfigError(NanoRLError):
    """Invalid or unreadable runtime configuration."""


class InvalidInputArtifactError(ConfigError):
    """A configured model, tokenizer, or dataset artifact failed validation."""


class RayClusterError(NanoRLError):
    """Ray cluster startup or connection error."""


class SlotError(NanoRLError):
    """Base class for slot state and capability errors."""


class GpuLeaseError(SlotError):
    """Base class for GPU lease state and capability errors."""


class GpuRoleUnsupported(GpuLeaseError):
    """Raised when a physical GPU is asked to activate an unsupported role."""

    def __init__(self, gpu_id: int, role: str):
        self.gpu_id = gpu_id
        self.role = role
        super().__init__(f"gpu {gpu_id} does not support role {role}")


class SlotStateError(SlotError):
    """Raised when a slot transition violates the state machine."""


class QueueError(NanoRLError):
    """Sample queue error."""


class WeightRegistryError(NanoRLError):
    """Weight registry error."""
