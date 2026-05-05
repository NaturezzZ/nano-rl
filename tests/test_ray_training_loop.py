from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from nano_rl.config import load_launch_config
from nano_rl.runtime.protocols import WeightFormat, WeightMeta
from nano_rl.runtime.ray.training import RayTrainingLoop
from nano_rl.runtime.slot import GpuLeaseManagerCore, RoleName


ROOT = Path(__file__).resolve().parents[1]


class FakeRay:
    def get(self, value: Any) -> Any:
        return value


class RemoteMethod:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


class FakeActor:
    def __init__(self, **methods: Callable[..., Any]) -> None:
        self._methods = methods

    def __getattr__(self, name: str) -> RemoteMethod:
        return RemoteMethod(self._methods[name])


def test_ray_training_loop_offloads_and_wakes_shared_gpu_residency_before_weight_activation() -> None:
    config = load_launch_config(ROOT / "recipes/mock_collocated.yaml")
    leases = GpuLeaseManagerCore(config.gpu_plan)
    operations: list[str] = []
    exported = _weight(version_id=1).model_dump(mode="json")

    def call_index(prefix: str) -> int:
        return next(index for index, item in enumerate(operations) if item.startswith(prefix))

    handles: dict[str, Any] = {
        "gpu-lease-manager": FakeActor(
            grant=lambda role, gpu_id, holder_id, reason: leases.grant(
                RoleName(role),
                gpu_id,
                holder_id,
                reason=reason,
            ).model_dump(mode="json"),
            current_lease=lambda role, gpu_id, holder_id: leases.current_lease(
                RoleName(role),
                gpu_id,
                holder_id=holder_id,
            ).model_dump(mode="json"),
        ),
        "sample-queue": FakeActor(
            ack_batch=lambda train_batch_id: operations.append(f"queue.ack:{train_batch_id}"),
            release_batch=lambda train_batch_id: operations.append(f"queue.release:{train_batch_id}"),
        ),
        "weight-registry": FakeActor(
            register=lambda meta: operations.append(f"registry.register:v{meta['version_id']}") or meta,
            begin_activation=lambda version, replica_ids: operations.append(f"registry.begin:v{version}"),
            ack_activation=lambda version, replica_id: operations.append(f"registry.ack:{replica_id}:v{version}") or exported,
            mark_failed=lambda version, reason: operations.append(f"registry.failed:v{version}"),
        ),
        "rollout-manager": FakeActor(
            pause_for_weight=lambda reason: operations.append(f"rollout-manager.pause:{reason}"),
            resume=lambda: operations.append("rollout-manager.resume"),
        ),
    }

    for replica in config.gpu_plan.rollout_replicas:
        handles[replica.replica_id] = FakeActor(
            offload=lambda leases, replica_id=replica.replica_id: operations.append(f"rollout.offload:{replica_id}"),
            wake=lambda leases, replica_id=replica.replica_id: operations.append(f"rollout.wake:{replica_id}"),
            activate_weight=lambda meta, leases, source, replica_id=replica.replica_id: operations.append(
                f"rollout.activate:{replica_id}:v{meta['version_id']}"
            ),
        )

    for rank in config.gpu_plan.trainer_ranks:
        handles[f"trainer-rank-{rank.rank}"] = FakeActor(
            hydrate=lambda state, lease, rank=rank.rank: operations.append(f"trainer.hydrate:{rank}") or state,
            optimize=lambda batch, lease, rank=rank.rank: operations.append(f"trainer.optimize:{rank}") or {},
            offload=lambda lease, rank=rank.rank: operations.append(f"trainer.offload:{rank}") or {},
            export_weight=lambda parent, rank=rank.rank: (
                operations.append(f"trainer.export:{rank}:v{parent['version_id']}")
                or exported
            ),
        )

    loop = RayTrainingLoop(
        config,
        SimpleNamespace(handles=handles),
        ray_module=FakeRay(),
    )
    loop._trainer_states = {rank.rank: {"rank": rank.rank} for rank in config.gpu_plan.trainer_ranks}

    result = loop._run_train_step(
        {"train_batch_id": "batch-1"},
        current_weight=_weight(version_id=0).model_dump(mode="json"),
        controller_step=0,
    )

    assert result["active_weight"]["version_id"] == 1
    assert call_index("rollout.offload:") < call_index("trainer.hydrate:")
    assert call_index("trainer.offload:") < call_index("rollout.wake:")
    assert call_index("rollout.wake:") < call_index("rollout.activate:")
    assert call_index("registry.register:v1") < call_index("rollout.activate:")
    assert call_index("trainer.offload:") < call_index("queue.ack:batch-1")
    assert not any(item.startswith("queue.release") for item in operations)


def _weight(version_id: int) -> WeightMeta:
    return WeightMeta(
        version_id=version_id,
        parent_version=None if version_id == 0 else version_id - 1,
        trainer_step=version_id,
        created_at=datetime.utcnow(),
        model_path=f"mock://weights/v{version_id}",
        tokenizer_path=f"mock://weights/v{version_id}",
        artifact_uri=f"mock://weights/v{version_id}",
        format=WeightFormat.VLLM_COMPATIBLE,
        checksum=f"checksum-{version_id}",
        created_by="test",
    )
