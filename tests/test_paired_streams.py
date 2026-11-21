from datetime import UTC, datetime, timedelta

import numpy as np
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
