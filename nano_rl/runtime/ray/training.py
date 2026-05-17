"""Driver-side Ray training loop orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from itertools import cycle, islice
import logging
from typing import Any

from nano_rl.config import LaunchConfig
from nano_rl.runtime.data_sources import PromptRecord, build_prompt_source
from nano_rl.runtime.protocols import WeightMeta
from nano_rl.runtime.ray.launcher import RayActorGraph
from nano_rl.runtime.roles import bootstrap_weight_meta
from nano_rl.runtime.slot import RoleName, RolloutReplicaSpec, TrainerRankSpec
from nano_rl.runtime.weight_transfer import WeightTransferPlanner


logger = logging.getLogger(__name__)


@dataclass
class PromptBatcher:
    """Small deterministic prompt batcher for bounded v0.1 training runs."""

    records: tuple[PromptRecord, ...]
    _index: int = 0

    @classmethod
    def from_config(cls, config: LaunchConfig, *, total_required: int) -> "PromptBatcher":
        source = build_prompt_source(config, seed=config.mock.seed)
        records = tuple(source.iter_prompts(limit=max(1, total_required)))
        if not records:
            raise RuntimeError("training data source produced no prompts")
        return cls(records=records)

    def next_records(self, count: int) -> list[PromptRecord]:
        if count < 1:
            raise ValueError("prompt batch count must be >= 1")
        iterator = cycle(self.records)
        records = list(islice(iterator, self._index, self._index + count))
        self._index = (self._index + count) % len(self.records)
        return records


class RayTrainingLoop:
    """Run the controller loop against already-started Ray actors.

    This keeps ``main.py`` thin while making the train intent execute the same
    high-level protocol used by the local controller: activate weights, pump
    rollout, reserve samples, toggle shared GPUs to trainer, optimize, publish,
    and reactivate rollout.
    """

    def __init__(
        self,
        launch_config: LaunchConfig,
        graph: RayActorGraph,
        *,
        ray_module: Any | None = None,
    ) -> None:
        self.launch_config = launch_config
        self.graph = graph
        self.handles = graph.handles
        self.ray = ray_module or import_module("ray")
        self._trainer_states: dict[int, dict[str, object] | None] = {}
        self._last_weight_transfer_plan: dict[str, object] | None = None
        self._rollout_active_weight_version: int | None = None

    def run(self) -> dict[str, object]:
        batch_size = self.launch_config.trainer.global_batch_size
        max_steps = self.launch_config.algorithm.max_steps
        prompt_batcher = PromptBatcher.from_config(
            self.launch_config,
            total_required=batch_size * max_steps,
        )

        initial_weight = self._bootstrap_initial_weight()
        current_weight = initial_weight
        steps: list[dict[str, object]] = []

        for step_index in range(max_steps):
            prompt_records = prompt_batcher.next_records(batch_size)
            rollout_result = self._run_rollout_step(
                prompt_records,
                current_weight=current_weight,
                controller_step=step_index,
            )
            if rollout_result["accepted_samples"] < batch_size:
                raise RuntimeError(
                    "rollout did not produce enough accepted samples for a train batch: "
                    f"accepted={rollout_result['accepted_samples']} required={batch_size}"
                )

            batch = self._reserve_train_batch(
                current_weight=current_weight,
                max_sequences=batch_size,
            )
            if batch is None:
                raise RuntimeError(f"train step {step_index} could not reserve a batch")

            train_result = self._run_train_step(
                batch,
                current_weight=current_weight,
                controller_step=step_index,
            )
            current_weight = train_result["active_weight"]
            steps.append(
                {
                    "step": step_index + 1,
                    "policy_version_before": rollout_result["policy_version"],
                    "prompts_submitted": len(prompt_records),
                    "samples_generated": rollout_result["generated_samples"],
                    "samples_accepted": rollout_result["accepted_samples"],
                    "train_batch_id": batch["train_batch_id"],
                    "train_batch_sequences": batch["num_sequences"],
                    "train_batch_tokens": batch["num_tokens"],
                    "train_stats": train_result["train_stats"],
                    "published_weight": current_weight,
                    "weight_transfer_plan": self._last_weight_transfer_plan,
                }
            )

        return {
            "status": "completed",
            "steps_completed": len(steps),
            "max_steps": max_steps,
            "global_batch_size": batch_size,
            "initial_weight": initial_weight,
            "initial_policy_version": initial_weight["version_id"],
            "final_weight": current_weight,
            "steps": steps,
            "queue": self._call("sample-queue", "stats"),
            "registry": self._call("weight-registry", "all_versions"),
            "gpu_lease_states": self._call("gpu-lease-manager", "states"),
            "rollout_manager": self._call("rollout-manager", "stats"),
            "trainer_states": self._trainer_actor_states(),
        }

    def _initialize_trainers(self) -> None:
        refs = [
            self.handles[f"trainer-rank-{rank.rank}"].initialize_rank.remote()
            for rank in self.launch_config.gpu_plan.trainer_ranks
        ]
        states = self._get(refs)
        self._trainer_states = {
            rank.rank: state
            for rank, state in zip(self.launch_config.gpu_plan.trainer_ranks, states, strict=True)
        }

    def _ensure_trainers_initialized(self) -> None:
        if all(
            self._trainer_states.get(rank.rank) is not None
            for rank in self.launch_config.gpu_plan.trainer_ranks
        ):
            return
        self._initialize_trainers()

    def _bootstrap_initial_weight(self) -> dict[str, object]:
        active = self._call("weight-registry", "latest_active_global")
        if active is not None:
            return active

        meta = bootstrap_weight_meta(
            model_path=self.launch_config.model.model_path,
            tokenizer_path=self.launch_config.model.tokenizer_path,
        ).model_dump(mode="json")
        registered = self._call("weight-registry", "register", meta)
        return self._activate_weight_for_rollout(registered)

    def _run_rollout_step(
        self,
        prompt_records: list[PromptRecord],
        *,
        current_weight: dict[str, object],
        controller_step: int,
    ) -> dict[str, object]:
        policy_version = int(current_weight["version_id"])
        self._call("rollout-manager", "enqueue_prompts", [_prompt_payload(record) for record in prompt_records])

        accepted_samples = 0
        generated_samples = 0
        while accepted_samples < len(prompt_records):
            queue_stats = self._call("sample-queue", "stats")
            output_depth = int(queue_stats["pending"]) + int(queue_stats["reserved_samples"])
            requests = self._call(
                "rollout-manager",
                "dispatch_from_backlog",
                policy_version,
                controller_step,
                output_depth,
                self.launch_config.control.queue_high_watermark,
                len(prompt_records) - accepted_samples,
            )
            if not requests:
                break

            sample_refs = []
            for request in requests:
                replica = self._rollout_replica(str(request["replica_id"]))
                leases = self._rollout_leases_for_replica(replica)
                sample_refs.append(self.handles[replica.replica_id].generate.remote(request, leases))
            samples = self._get(sample_refs)

            for request, sample in zip(requests, samples, strict=True):
                self._call("rollout-manager", "complete_request", request["request_id"])
                result = self._call(
                    "sample-queue",
                    "submit_sample",
                    sample,
                    policy_version,
                    sample,
                )
                generated_samples += 1
                if result["accepted"]:
                    accepted_samples += 1

        return {
            "policy_version": policy_version,
            "generated_samples": generated_samples,
            "accepted_samples": accepted_samples,
        }

    def _reserve_train_batch(
        self,
        *,
        current_weight: dict[str, object],
        max_sequences: int,
    ) -> dict[str, object] | None:
        return self._call(
            "sample-queue",
            "reserve_train_batch",
            "trainer-group-0",
            int(current_weight["version_id"]),
            max_sequences,
        )

    def _run_train_step(
        self,
        batch: dict[str, object],
        *,
        current_weight: dict[str, object],
        controller_step: int,
    ) -> dict[str, object]:
        trainer_leases: dict[int, dict[str, object]] = {}
        trainers_offloaded = False
        try:
            self._offload_shared_rollouts()
            self._grant_train_window(controller_step)
            trainer_leases = {
                rank.rank: self._trainer_lease(rank)
                for rank in self.launch_config.gpu_plan.trainer_ranks
            }
            self._ensure_trainers_initialized()
            self._hydrate_trainers(trainer_leases)
            try:
                train_stats = self._optimize_trainers(batch, trainer_leases)
                exported = self._export_weight_from_trainers(current_weight)
                registered = self._call("weight-registry", "register", exported)
                self._sync_native_vllm_weights(registered, trainer_leases)
            except Exception:
                self._offload_trainers_best_effort(trainer_leases)
                trainers_offloaded = True
                raise
            else:
                self._offload_trainers(trainer_leases)
                trainers_offloaded = True
            self._call("sample-queue", "ack_batch", batch["train_batch_id"])
        except Exception:
            if trainer_leases and not trainers_offloaded:
                self._offload_trainers_best_effort(trainer_leases)
            self._call("sample-queue", "release_batch", batch["train_batch_id"])
            raise
        finally:
            self._grant_rollout_window(controller_step)
            self._wake_shared_rollouts()

        active = self._activate_weight_for_rollout(registered)
        return {"train_stats": train_stats, "active_weight": active}

    def _hydrate_trainers(self, leases: dict[int, dict[str, object]]) -> None:
        refs = []
        ranks: list[int] = []
        for rank in self.launch_config.gpu_plan.trainer_ranks:
            ranks.append(rank.rank)
            refs.append(
                self.handles[f"trainer-rank-{rank.rank}"].hydrate.remote(
                    self._trainer_states.get(rank.rank),
                    leases[rank.rank],
                )
            )
        states = self._get(refs)
        for rank, state in zip(ranks, states, strict=True):
            self._trainer_states[rank] = state

    def _optimize_trainers(
        self,
        batch: dict[str, object],
        leases: dict[int, dict[str, object]],
    ) -> list[dict[str, object]]:
        refs = [
            self.handles[f"trainer-rank-{rank.rank}"].optimize.remote(batch, leases[rank.rank])
            for rank in self.launch_config.gpu_plan.trainer_ranks
        ]
        return self._get(refs)

    def _export_weight_from_trainers(self, current_weight: dict[str, object]) -> dict[str, object]:
        refs: list[object] = []
        ranks: list[int] = []
        for rank in self.launch_config.gpu_plan.trainer_ranks:
            ranks.append(rank.rank)
            refs.append(
                self.handles[f"trainer-rank-{rank.rank}"].export_weight.remote(
                    current_weight,
                    rank.rank == 0,
                )
            )
        exports = self._get(refs)
        for rank, exported in zip(ranks, exports, strict=True):
            if rank == 0:
                return exported
        raise RuntimeError("trainer rank 0 is required to publish exported weights")

    def _sync_native_vllm_weights(
        self,
        meta: dict[str, object],
        trainer_leases: dict[int, dict[str, object]],
    ) -> None:
        rollout_config = self.launch_config.rollout
        if str(rollout_config.backend) != "vllm":
            return
        if rollout_config.vllm.weight_sync_backend == "none":
            return

        weight = WeightMeta.model_validate(meta)
        planner = WeightTransferPlanner(
            method=self.launch_config.weight_transfer.method,
            allow_rollout_only_artifact_pull=self.launch_config.weight_transfer.allow_rollout_only_artifact_pull,
        )
        transfer_plan = planner.build_plan(
            weight,
            rollout_replicas=self.launch_config.gpu_plan.rollout_replicas,
            trainer_ranks=self.launch_config.gpu_plan.trainer_ranks,
        )
        self._last_weight_transfer_plan = transfer_plan.model_dump(mode="json")

        refs = []
        for replica in self.launch_config.gpu_plan.rollout_replicas:
            source = transfer_plan.source_for_replica(replica.replica_id).model_dump(mode="json")
            transport = self._native_vllm_transport(source)
            participant_ranks = self._native_vllm_participant_ranks(transport)
            for rank in participant_ranks:
                trainer_handle = self.handles[f"trainer-rank-{rank.rank}"]
                refs.append(
                    trainer_handle.sync_weights_to_vllm.remote(
                        self.handles[replica.replica_id],
                        meta,
                        source,
                        trainer_leases[rank.rank],
                        transport,
                        True,
                    )
                )
        if refs:
            self._get(refs)

    def _native_vllm_transport(self, source: dict[str, object]) -> str:
        configured = self.launch_config.rollout.vllm.weight_sync_backend
        if configured in {"ipc", "nccl"}:
            return configured
        kind = str(source.get("kind"))
        if kind == "shared_gpu_reshard":
            return "ipc"
        return "nccl"

    def _native_vllm_participant_ranks(self, transport: str) -> list[TrainerRankSpec]:
        if transport == "ipc":
            return list(self.launch_config.gpu_plan.trainer_ranks)
        if transport == "nccl":
            return [rank for rank in self.launch_config.gpu_plan.trainer_ranks if rank.rank == 0]
        raise RuntimeError(f"unsupported native vLLM weight sync transport: {transport}")

    def _offload_trainers(self, leases: dict[int, dict[str, object]]) -> None:
        refs = [
            self.handles[f"trainer-rank-{rank.rank}"].offload.remote(leases[rank.rank])
            for rank in self.launch_config.gpu_plan.trainer_ranks
        ]
        states = self._get(refs)
        for rank, state in zip(self.launch_config.gpu_plan.trainer_ranks, states, strict=True):
            self._trainer_states[rank.rank] = state

    def _offload_trainers_best_effort(self, leases: dict[int, dict[str, object]]) -> None:
        try:
            self._offload_trainers(leases)
        except Exception:
            logger.exception("best-effort trainer offload failed after train window error")

    def _activate_weight_for_rollout(self, meta: dict[str, object]) -> dict[str, object]:
        weight = WeightMeta.model_validate(meta)
        if self._rollout_active_weight_version == weight.version_id:
            return meta
        planner = WeightTransferPlanner(
            method=self.launch_config.weight_transfer.method,
            allow_rollout_only_artifact_pull=self.launch_config.weight_transfer.allow_rollout_only_artifact_pull,
        )
        transfer_plan = planner.build_plan(
            weight,
            rollout_replicas=self.launch_config.gpu_plan.rollout_replicas,
            trainer_ranks=self.launch_config.gpu_plan.trainer_ranks,
        )
        self._last_weight_transfer_plan = transfer_plan.model_dump(mode="json")

        replica_ids = [replica.replica_id for replica in self.launch_config.gpu_plan.rollout_replicas]
        self._call("rollout-manager", "pause_for_weight", f"activate-v{weight.version_id}")
        self._call("weight-registry", "begin_activation", weight.version_id, replica_ids)
        active = meta
        try:
            for replica in self.launch_config.gpu_plan.rollout_replicas:
                leases = self._rollout_leases_for_replica(replica)
                source = transfer_plan.source_for_replica(replica.replica_id).model_dump(mode="json")
                self._call(replica.replica_id, "activate_weight", meta, leases, source)
                active = self._call("weight-registry", "ack_activation", weight.version_id, replica.replica_id)
            self._rollout_active_weight_version = weight.version_id
            return active
        except Exception:
            self._call("weight-registry", "mark_failed", weight.version_id, "rollout activation failed")
            raise
        finally:
            self._call("rollout-manager", "resume")

    def _offload_shared_rollouts(self) -> None:
        refs = []
        for replica in self._shared_rollout_replicas():
            leases = self._rollout_leases_for_replica(replica)
            refs.append(self.handles[replica.replica_id].offload.remote(leases))
        if refs:
            self._get(refs)

    def _wake_shared_rollouts(self) -> None:
        refs = []
        for replica in self._shared_rollout_replicas():
            leases = self._rollout_leases_for_replica(replica)
            refs.append(self.handles[replica.replica_id].wake.remote(leases))
        if refs:
            self._get(refs)

    def _shared_rollout_replicas(self) -> list[RolloutReplicaSpec]:
        shared_gpu_ids = set(self.launch_config.gpu_plan.shared_gpu_ids)
        return [
            replica
            for replica in self.launch_config.gpu_plan.rollout_replicas
            if shared_gpu_ids.intersection(replica.gpu_ids)
        ]

    def _grant_train_window(self, controller_step: int) -> None:
        for rank in self.launch_config.gpu_plan.trainer_ranks:
            self._call(
                "gpu-lease-manager",
                "grant",
                RoleName.TRAINER.value,
                rank.gpu_id,
                f"trainer-rank-{rank.rank}",
                f"train-step-{controller_step}",
            )

    def _grant_rollout_window(self, controller_step: int) -> None:
        for gpu_id in self.launch_config.gpu_plan.shared_gpu_ids:
            worker = self.launch_config.gpu_plan.rollout_worker_for_gpu(gpu_id)
            self._call(
                "gpu-lease-manager",
                "grant",
                RoleName.ROLLOUT.value,
                gpu_id,
                worker.worker_id,
                f"rollout-step-{controller_step}",
            )

    def _rollout_leases_for_replica(self, replica: RolloutReplicaSpec) -> list[dict[str, object]]:
        return [
            self._call(
                "gpu-lease-manager",
                "current_lease",
                RoleName.ROLLOUT.value,
                gpu_id,
                holder_id,
            )
            for gpu_id, holder_id in zip(replica.gpu_ids, replica.worker_ids, strict=True)
        ]

    def _trainer_lease(self, rank: TrainerRankSpec) -> dict[str, object]:
        return self._call(
            "gpu-lease-manager",
            "current_lease",
            RoleName.TRAINER.value,
            rank.gpu_id,
            f"trainer-rank-{rank.rank}",
        )

    def _rollout_replica(self, replica_id: str) -> RolloutReplicaSpec:
        for replica in self.launch_config.gpu_plan.rollout_replicas:
            if replica.replica_id == replica_id:
                return replica
        raise RuntimeError(f"unknown rollout replica: {replica_id}")

    def _trainer_actor_states(self) -> list[dict[str, object]]:
        refs = [
            self.handles[f"trainer-rank-{rank.rank}"].state.remote()
            for rank in self.launch_config.gpu_plan.trainer_ranks
        ]
        return self._get(refs)

    def _call(self, actor_name: str, method_name: str, *args: object) -> Any:
        method = getattr(self.handles[actor_name], method_name)
        return self._get(method.remote(*args))

    def _get(self, value: Any) -> Any:
        return self.ray.get(value)


def _prompt_payload(record: PromptRecord) -> dict[str, object]:
    return {
        "prompt_id": record.prompt_id,
        "prompt": record.prompt,
        "metadata": dict(record.metadata),
    }
