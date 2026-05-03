"""Runtime backend boundaries."""

from nano_rl.runtime.backends.fsdp2_backend import Fsdp2TrainerBackend
from nano_rl.runtime.backends.trainer_backend import (
    BackendStateError,
    BackendUnavailableError,
    FakeTrainerBackend,
    OptimizerStepResult,
    TrainerBackend,
    TrainerBackendConfig,
    TrainerProcessGroupMetadata,
    TrainStateBundle,
    build_trainer_backend,
)
from nano_rl.runtime.backends.vllm_backend import (
    GenerationOutput,
    RolloutBackend,
    VllmBackendConfig,
    VllmBackendError,
    VllmBackendUnavailable,
    VllmRolloutBackend,
    build_rollout_backend,
    generation_output_to_sample_record,
)

__all__ = [
    "BackendStateError",
    "BackendUnavailableError",
    "FakeTrainerBackend",
    "Fsdp2TrainerBackend",
    "GenerationOutput",
    "OptimizerStepResult",
    "RolloutBackend",
    "TrainerBackend",
    "TrainerBackendConfig",
    "TrainerProcessGroupMetadata",
    "TrainStateBundle",
    "VllmBackendConfig",
    "VllmBackendError",
    "VllmBackendUnavailable",
    "VllmRolloutBackend",
    "build_rollout_backend",
    "build_trainer_backend",
    "generation_output_to_sample_record",
]
