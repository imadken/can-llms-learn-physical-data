"""Training-only arrays and deterministic initial token RMS calibration."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from scipy.spatial.transform import Rotation

from effect_pretraining.coordinates import NumericalCoordinates
from effect_pretraining.effect_scaling import EffectTransform, initialize_projector
from effect_pretraining.integrity import sha256_file, verify_release
from effect_pretraining.native_flow import load_native_policy, pad_native


def prepare(root: Path, artifacts: Path) -> None:
    verify_release(root)
    resolved_path = root / "resolved_environment.json"
    resolved = json.loads(resolved_path.read_text())
    paths = resolved["runtime_paths"]
    transform = EffectTransform.model_validate(
        {
            key: value
            for key, value in resolved["effect_transform"].items()
            if key in EffectTransform.model_fields
        }
    )
    interface = json.loads((artifacts / "numerical_interface.json").read_text())
    coordinates = NumericalCoordinates.model_validate(interface["coordinates"])
    with np.load(artifacts / "training_preparation.npz", allow_pickle=False) as saved:
        window_ids = saved["window_ids"].copy()
        nominal = saved["nominal_effects"].copy()
        states = saved["normalized_states"].copy()
        subset = saved["reference_indices"].copy()
    action_windows, measured = [], []
    current_episode, payload = None, None
    for episode, offset in window_ids:
        if episode != current_episode:
            if payload is not None:
                payload.close()
            payload = np.load(
                artifacts / "episodes" / f"episode_{episode:03d}.npz", allow_pickle=False
            )
            current_episode = episode
        action_windows.append(coordinates.action(payload["actions"][offset : offset + 16]))
        dp = payload["positions"][offset + 16] - payload["positions"][offset]
        relative = payload["rotations"][offset].T @ payload["rotations"][offset + 16]
        measured.append(np.r_[dp, Rotation.from_matrix(relative).as_rotvec()])
    payload.close()
    nominal_z = transform.apply(nominal)
    measured_z = transform.apply(np.array(measured))
    np.savez_compressed(
        artifacts / "training_arrays.npz",
        states=states.astype(np.float32),
        actions=np.array(action_windows, dtype=np.float32),
        nominal_z=nominal_z.astype(np.float32),
        measured_z=measured_z.astype(np.float32),
        window_ids=window_ids,
    )
    policy = load_native_policy(Path(paths["checkpoint"]), Path(paths["vlm_assets"]))
    with torch.no_grad():
        state_tokens = policy.model.state_proj(
            pad_native(
                torch.tensor(states[subset], dtype=torch.float32), policy.config.max_state_dim
            )
        )
    records = {}
    for seed in (101, 202):
        projector, calculation = initialize_projector(
            state_tokens, torch.tensor(nominal_z[subset], dtype=torch.float32), 800000 + seed
        )
        path = artifacts / f"initial_projector_seed{seed}.safetensors"
        save_file(projector.state_dict(), str(path))
        summaries = {}
        with torch.no_grad():
            for arm, values in [("K", nominal_z), ("D", measured_z)]:
                embeddings = projector(torch.tensor(values, dtype=torch.float32))
                summaries[arm] = {
                    "input_norm_quantiles": np.quantile(
                        np.linalg.norm(values, axis=1), [0, 0.25, 0.5, 0.75, 1]
                    ).tolist(),
                    "embedding_norm_quantiles": torch.quantile(
                        embeddings.norm(dim=1), torch.tensor([0, 0.25, 0.5, 0.75, 1])
                    ).tolist(),
                }
        records[str(seed)] = calculation | {
            "path": str(path.resolve()),
            "file_sha256": sha256_file(path),
            "reference_window_ids": window_ids[subset].tolist(),
            "reference_count": len(subset),
            "paired_arms": ["P", "K", "D"],
            "input_embedding_summaries": summaries,
        }
    resolved["projector_calibration"] = {
        "status": "training_only_calibrated",
        "calibrated": True,
        "seeds": records,
        "training_arrays_sha256": sha256_file(artifacts / "training_arrays.npz"),
    }
    resolved_path.write_text(json.dumps(resolved, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete_training_windows": len(states),
                "calibration": {
                    seed: {
                        key: value
                        for key, value in record.items()
                        if key not in {"reference_window_ids", "input_embedding_summaries"}
                    }
                    for seed, record in records.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    prepare(Path.cwd(), args.artifacts.resolve())
