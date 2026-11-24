"""The ordinary camera/instruction/current-state deployment interface."""

import numpy as np
import torch
from robosuite.utils.transform_utils import quat2axisangle

from effect_pretraining.coordinates import NumericalCoordinates
from effect_pretraining.native_flow import pad_native


def official_initial_states(suite, task_id: int):
    from pathlib import Path

    from libero.libero import get_libero_path

    task = suite.tasks[task_id]
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    # The pinned official files contain NumPy arrays. Limit deserialization to numeric types.
    numeric_globals = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.float32)),
    ]
    with torch.serialization.safe_globals(numeric_globals):
        states = torch.load(path, weights_only=True)
    values = np.asarray(states)
    if values.ndim != 2 or len(values) < 21 or not np.isfinite(values).all():
        raise ValueError("Invalid official initial-state bank")
    return values


def observation_batch(raw: dict, coordinates: NumericalCoordinates, language: str) -> dict:
    state = np.r_[
        raw["robot0_eef_pos"],
        quat2axisangle(raw["robot0_eef_quat"].copy()),
        raw["robot0_gripper_qpos"],
    ]
    batch = {
        "observation.state": torch.tensor(coordinates.state(state)[None], dtype=torch.float32),
        "task": [language],
    }
    for name, source in [("camera1", "agentview_image"), ("camera2", "robot0_eye_in_hand_image")]:
        image = np.ascontiguousarray(raw[source][::-1, ::-1])
        batch[f"observation.images.{name}"] = (
            torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255
        )
    return batch


@torch.no_grad()
def generate_ordinary(policy, batch: dict, noise: torch.Tensor | None = None) -> torch.Tensor:
    device = next(policy.parameters()).device
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    images, image_masks = policy.prepare_images(batch)
    language, language_masks = policy.prepare_language(batch)
    return policy.model.sample_actions(
        images,
        image_masks,
        language,
        language_masks,
        pad_native(batch["observation.state"], 32),
        noise=noise,
    )[..., :7]
