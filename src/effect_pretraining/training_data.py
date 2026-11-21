"""Paired update streams and fixed multitask demonstration sampling."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from effect_pretraining.coordinates import NumericalCoordinates


class NumericalBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    indices: np.ndarray
    states: torch.Tensor
    actions: torch.Tensor
    nominal_z: torch.Tensor
    measured_z: torch.Tensor
    noise: torch.Tensor
    time: torch.Tensor
    masked: torch.Tensor


def paired_update(
    arrays: dict, native, seed: int, update: int, batch_size: int = 128
) -> NumericalBatch:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 11000, update]))
    indices = rng.integers(len(arrays["states"]), size=batch_size)
    # Native sampling uses CPU Torch RNG; isolate it from module construction and arm execution.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(np.random.SeedSequence([seed, 12000, update]).generate_state(1)[0]))
        noise = native.sample_noise((batch_size, 16, 32), "cpu")
        time = native.sample_time(batch_size, "cpu")
        masked = torch.rand(batch_size) < 0.2
    return NumericalBatch(
        indices=indices,
        states=torch.from_numpy(arrays["states"][indices]),
        actions=torch.from_numpy(arrays["actions"][indices]),
        nominal_z=torch.from_numpy(arrays["nominal_z"][indices]),
        measured_z=torch.from_numpy(arrays["measured_z"][indices]),
        noise=noise,
        time=time,
        masked=masked,
    )


class DemonstrationSampler:
    def __init__(self, manifest: Path, coordinates: NumericalCoordinates, split: str = "train"):
        self.manifest = json.loads(manifest.read_text())
        if len(self.manifest) != 10:
            raise ValueError("The baseline requires all ten tasks")
        self.coordinates = coordinates
        self.handles = [h5py.File(record["demonstration_path"], "r") for record in self.manifest]
        self.windows = []
        for record in self.manifest:
            windows = [
                (item["trajectory_id"], offset)
                for item in record["selected_trajectories"]
                if item["split"] == split
                for offset in range(item["complete_window_count"])
            ]
            if not windows:
                raise ValueError("Every task needs complete permitted windows")
            self.windows.append(windows)

    def sample(self, seed: int, update: int, count: int = 32) -> dict:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 21000, update]))
        states, actions, camera1, camera2, languages, ids = [], [], [], [], [], []
        for task_id in rng.integers(10, size=count):
            name, offset = self.windows[task_id][rng.integers(len(self.windows[task_id]))]
            group = self.handles[task_id]["data"][name]
            state = np.r_[
                group["obs/ee_pos"][offset],
                group["obs/ee_ori"][offset],
                group["obs/gripper_states"][offset],
            ]
            if state.shape != (8,):
                raise ValueError(f"Native eight-coordinate state unavailable: {state.shape}")
            states.append(self.coordinates.state(state))
            # Official raw conversion records obs[j] after action[j]. Predict the next chunk.
            actions.append(self.coordinates.action(group["actions"][offset + 1 : offset + 17]))
            # LIBERO's native camera convention rotates both image axes by 180 degrees.
            camera1.append(group["obs/agentview_rgb"][offset][::-1, ::-1].copy())
            camera2.append(group["obs/eye_in_hand_rgb"][offset][::-1, ::-1].copy())
            languages.append(self.manifest[task_id]["language"])
            ids.append((int(task_id), name, offset))
        return {
            "observation.state": torch.tensor(np.array(states), dtype=torch.float32),
            "action": torch.tensor(np.array(actions), dtype=torch.float32),
            "observation.images.camera1": torch.from_numpy(np.array(camera1))
            .permute(0, 3, 1, 2)
            .float()
            / 255,
            "observation.images.camera2": torch.from_numpy(np.array(camera2))
            .permute(0, 3, 1, 2)
            .float()
            / 255,
            "task": languages,
            "window_ids": ids,
        }

    def close(self):
        for handle in self.handles:
            handle.close()
