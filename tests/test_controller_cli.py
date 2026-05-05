from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import main as cli_main
from nano_rl.config import load_launch_config
from nano_rl.exceptions import SlotStateError
from nano_rl.runtime.protocols import WeightStatus
from nano_rl.runtime.controller import ControllerCore


ROOT = Path(__file__).resolve().parents[1]


def test_controller_dry_run_plan_counts_gpus() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    plan = ControllerCore(config).dry_run()

    assert plan["canonical_mode"] == "standalone_hybrid"
    assert plan["total_gpus"] == 8
    assert plan["rollout_gpu_count"] == 8
    assert plan["shared_gpu_count"] == 4
    assert plan["rollout_replica_count"] == 4
    assert plan["rollout_worker_count"] == 8
    assert plan["trainer_rank_count"] == 4
    assert len(plan["gpu_lease_states"]) == 8
    assert plan["rollout_manager"]["replicas"] == 4


def test_cli_emit_resolved_config_exits_without_artifact_validation(capsys) -> None:
    code = cli_main.main(["--config", str(ROOT / "recipes/collocated.yaml"), "--emit-resolved-config"])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out == ""
    resolved = json.loads(cli_main.RESOLVED_CONFIG_PATH.read_text(encoding="utf-8"))
    assert resolved["canonical_mode"] == "fully_sync"
    assert len(resolved["gpu_plan"]["rollout_workers"]) == 8


def test_cli_train_skip_artifact_validation_outputs_single_runtime_plan(capsys) -> None:
    raw = yaml.safe_load((ROOT / "recipes/disaggregated.yaml").read_text())
    raw["run"]["emit_resolved_config"] = False
    temp = ROOT / ".tmp-cli-train-skip.yaml"
    temp.write_text(yaml.safe_dump(raw))
    try:
        code = cli_main.main(["--config", str(temp), "--skip-artifact-validation"])
    finally:
        temp.unlink(missing_ok=True)
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out == ""
    payload = json.loads(cli_main.RUN_RESULT_PATH.read_text(encoding="utf-8"))
    resolved = json.loads(cli_main.RESOLVED_CONFIG_PATH.read_text(encoding="utf-8"))
    assert resolved["canonical_mode"] == "standalone_hybrid"
    assert payload["canonical_mode"] == "standalone_hybrid"
    assert payload["execution_status"] == "planned_backend_integrated"
    assert payload["start_ray_actors"] is False
    assert payload["ray_launch_plan"]["actors"][1]["actor_type"] == "GpuLeaseManagerActor"


def test_controller_smoke_iteration_composes_runtime_modules() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    result = ControllerCore(config).run_smoke_iteration(["hello", "world"])

    assert result["initial_weight"]["version_id"] == 0
    assert result["new_weight"]["version_id"] == 1
    assert result["new_weight"]["status"] == "active_global"
    assert len(result["samples"]) == 2
    assert result["train_batch"]["num_sequences"] == 2
    assert len(result["train_stats"]) == 4
    assert result["queue"]["acked"] == 2
    assert all(state["phase"] == "ROLLOUT_ACTIVE" for state in result["rollout_states"])
    assert result["weight_transfer_plan"]["method"] == "locality_aware_checkpoint"
    assert result["weight_transfer_plan"]["sources"]["rollout-dp-0"]["kind"] == "artifact_pull"
    assert result["weight_transfer_plan"]["sources"]["rollout-dp-2"]["kind"] == "shared_gpu_reshard"


def test_controller_accepts_prompt_backlog_and_pumps_rollout() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    controller = ControllerCore(config)
    weight = controller.bootstrap_initial_weight()

    backlog = controller.submit_prompts(["a", "b", "c"])
    samples = controller.pump_rollout(current_weight=weight, max_new_requests=2)

    assert backlog == 3
    assert len(samples) == 2
    assert controller.rollout_manager.stats()["input_backlog"] == 1
    assert controller.sample_queue.stats()["pending"] == 2


def test_controller_train_window_updates_residency_state_machine() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    controller = ControllerCore(config)

    train_states = controller.enter_train_window()

    assert all(state["phase"] == "TRAINER_ACTIVE" for state in train_states)
    assert all(
        controller.gpu_residency.get_state(gpu_id) == "TRAINER_ACTIVE"
        for gpu_id in config.gpu_plan.shared_gpu_ids
    )

    rollout_states = controller.return_to_rollout()

    assert all(state["phase"] == "ROLLOUT_ACTIVE" for state in rollout_states)
    assert all(
        controller.gpu_residency.get_state(gpu_id) == "ROLLOUT_ACTIVE"
        for gpu_id in config.gpu_plan.shared_gpu_ids
    )


def test_controller_pumps_rollout_until_budget_is_exhausted() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    controller = ControllerCore(config)
    weight = controller.bootstrap_initial_weight()
    controller.submit_prompts([f"prompt-{index}" for index in range(150)])

    samples = controller.pump_rollout_until_blocked(current_weight=weight, max_total_requests=150)

    assert len(samples) == 150
    assert controller.rollout_manager.stats()["input_backlog"] == 0
    assert controller.sample_queue.stats()["pending"] == 150


def test_controller_rejects_rollout_activation_during_train_window() -> None:
    config = load_launch_config(ROOT / "recipes/collocated.yaml")
    controller = ControllerCore(config)
    active = controller.bootstrap_initial_weight()
    controller.enter_train_window()

    registered = controller.weight_registry.register(
        controller.trainer_ranks[0].export_weight(active),
    )

    with pytest.raises(SlotStateError, match="active role"):
        controller._activate_weight_for_rollout(registered)
    assert controller.weight_registry.get(registered.version_id).status == WeightStatus.FAILED
    assert controller.rollout_manager.paused is False


def test_controller_releases_batch_and_returns_to_rollout_on_train_failure() -> None:
    config = load_launch_config(ROOT / "recipes/disaggregated.yaml")
    controller = ControllerCore(config)

    def fail_optimize(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic trainer failure")

    controller.trainer_ranks[0].optimize = fail_optimize  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="synthetic trainer failure"):
        controller.run_smoke_iteration(["hello"])

    queue_stats = controller.sample_queue.stats()
    assert queue_stats["pending"] == 1
    assert queue_stats["reserved_batches"] == 0
    assert queue_stats["acked"] == 0
    shared_states = [state for state in controller.gpu_leases.states() if state["topology"] == "shared"]
    assert all(state["phase"] == "ROLLOUT_ACTIVE" for state in shared_states)
