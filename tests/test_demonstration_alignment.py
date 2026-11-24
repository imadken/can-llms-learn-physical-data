import json

import h5py
import numpy as np
import pytest

from effect_pretraining.coordinates import NumericalCoordinates
from effect_pretraining.training_data import DemonstrationSampler


def test_post_action_observation_uses_next_action_chunk_and_native_camera_rotation(tmp_path):
    path = tmp_path / "raw.hdf5"
    count = 20
    with h5py.File(path, "w") as payload:
        trajectory = payload.create_group("data/demo_0")
        trajectory["actions"] = np.repeat(np.arange(count)[:, None] / 100, 7, axis=1)
        obs = trajectory.create_group("obs")
        obs["ee_pos"] = np.repeat(np.arange(count)[:, None] / 100, 3, axis=1)
        obs["ee_ori"] = np.zeros((count, 3))
        obs["gripper_states"] = np.zeros((count, 2))
        image = np.repeat(np.arange(16, dtype=np.uint8).reshape(4, 4, 1), 3, axis=2)
        for name in ("agentview_rgb", "eye_in_hand_rgb"):
            obs[name] = np.repeat(image[None], count, axis=0)
    manifest = [
        {
            "demonstration_path": str(path),
            "language": f"task {i}",
            "selected_trajectories": [
                {"trajectory_id": "demo_0", "split": "train", "complete_window_count": count - 16}
            ],
        }
        for i in range(10)
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    coordinates = NumericalCoordinates(
        gripper_absolute_limits=(0.04, 0.04), action_min=(-1,) * 7, action_max=(1,) * 7
    )
    sampler = DemonstrationSampler(manifest_path, coordinates)
    try:
        batch = sampler.sample(303, 1)
        assert batch["action"].shape == (32, 16, 7)
        for index, (_, _, offset) in enumerate(batch["window_ids"]):
            assert batch["observation.state"][index, 0].item() == pytest.approx(offset / 100)
            assert batch["action"][index, 0, 0].item() == pytest.approx((offset + 1) / 100)
            assert batch["action"][index, -1, 0].item() == pytest.approx((offset + 16) / 100)
            assert batch["observation.images.camera1"][index, 0, 0, 0].item() == pytest.approx(
                15 / 255
            )
    finally:
        sampler.close()
