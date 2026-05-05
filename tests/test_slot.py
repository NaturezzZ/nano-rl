from __future__ import annotations

import pytest

from nano_rl.config import load_launch_config
from nano_rl.exceptions import GpuRoleUnsupported, SlotStateError
from nano_rl.runtime.slot import GpuLeaseManagerCore, RoleName


def test_rollout_only_gpu_cannot_grant_trainer_lease() -> None:
    config = load_launch_config("recipes/disaggregated.yaml")
    manager = GpuLeaseManagerCore(config.gpu_plan)

    with pytest.raises(GpuRoleUnsupported):
        manager.grant(RoleName.TRAINER, 0, "trainer-rank-x")

    state = manager.get_state(0)
    assert state.active_role == RoleName.ROLLOUT
    assert state.holder_id == "rollout-dp-0-tp-0"


def test_shared_gpu_toggles_between_rollout_and_trainer_leases() -> None:
    config = load_launch_config("recipes/disaggregated.yaml")
    manager = GpuLeaseManagerCore(config.gpu_plan)

    train_lease = manager.grant(RoleName.TRAINER, 4, "trainer-rank-0", reason="train_window")
    assert train_lease.lease_epoch == 1
    manager.assert_active(RoleName.TRAINER, 4, holder_id="trainer-rank-0", expected_epoch=1)

    rollout_lease = manager.grant(RoleName.ROLLOUT, 4, "rollout-dp-2-tp-0", reason="train_window_done")
    assert rollout_lease.lease_epoch == 2
    manager.assert_active(RoleName.ROLLOUT, 4, holder_id="rollout-dp-2-tp-0", expected_epoch=2)


def test_assert_active_rejects_wrong_lease_epoch() -> None:
    config = load_launch_config("recipes/disaggregated.yaml")
    manager = GpuLeaseManagerCore(config.gpu_plan)

    manager.grant(RoleName.TRAINER, 4, "trainer-rank-0")
    with pytest.raises(SlotStateError, match="expected 99"):
        manager.assert_active(RoleName.TRAINER, 4, holder_id="trainer-rank-0", expected_epoch=99)
