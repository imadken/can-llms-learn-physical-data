from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

from effect_pretraining.schedule import ResourceBudget
from effect_pretraining.training_data import paired_update


def test_native_data_noise_time_masks_are_paired_independently_of_arm_execution():
    native = VLAFlowMatching.__new__(VLAFlowMatching)
    arrays = {
        "states": np.arange(80, dtype=np.float32).reshape(10, 8),
        "actions": np.arange(1120, dtype=np.float32).reshape(10, 16, 7),
        "nominal_z": np.arange(60, dtype=np.float32).reshape(10, 6),
        "measured_z": np.arange(60, dtype=np.float32).reshape(10, 6) * 0.2,
    }
    first = paired_update(arrays, native, 101, 20)
    torch.rand(1500)  # Other arms and initialization may consume global RNG.
    second = paired_update(arrays, native, 101, 20)
    for key in ("states", "actions", "nominal_z", "measured_z", "noise", "time", "masked"):
        assert torch.equal(getattr(first, key), getattr(second, key))
    assert np.array_equal(first.indices, second.indices)
    assert ((first.time >= 0.001) & (first.time <= 1)).all()
    other_seed = paired_update(arrays, native, 202, 20)
    assert not torch.equal(first.noise, other_seed.noise)


def test_resource_admission_reserves_contingency_and_honors_earliest_limit():
    now = datetime(2026, 11, 21, 10, tzinfo=UTC)
    budget = ResourceBudget(profiling_started_utc=now, accelerator_seconds_used=28000)
    assert budget.permits(now, 400, 200)
    assert not budget.permits(now, 600, 200)
    budget = ResourceBudget(profiling_started_utc=now - timedelta(hours=12))
    assert not budget.permits(now, 1, 1)
    assert not ResourceBudget().permits(datetime(2026, 11, 28, 17, tzinfo=UTC), 1, 1)


def test_production_cannot_start_from_a_faulted_profile(tmp_path):
    from effect_pretraining.training import profiles_and_projection

    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "profiling.json").write_text('{"status":"fault","paths":{}}')
    with pytest.raises(ValueError, match="Both prescribed profiles"):
        profiles_and_projection(tmp_path, tmp_path / "artifacts")


def test_unavailable_machine_wait_preserves_caps_and_requires_explicit_recovery():
    now = datetime(2026, 11, 22, 10, tzinfo=UTC)
    budget = ResourceBudget(
        profiling_started_utc=now - timedelta(hours=13), unavailable_wait_seconds=2 * 3600
    )
    assert budget.permits(now, 10, 10)
    budget.unavailable_wait_started_utc = now
    assert not budget.permits(now, 10, 10)
    assert budget.accelerator_limit_seconds == 8 * 3600
    assert budget.execution_limit_seconds == 12 * 3600
