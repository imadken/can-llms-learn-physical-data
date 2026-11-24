"""Prescribed fixed snapshot replays and the single ridge residual diagnostic."""

import csv
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from effect_pretraining.coordinates import (
    ControllerMap,
    NumericalCoordinates,
    effect_discrepancy,
    endpoint_effect,
    nominal_effect,
)
from effect_pretraining.effect_scaling import fit_effect_transform, reference_indices
from effect_pretraining.integrity import sha256_file, verify_release
from effect_pretraining.simulation import (
    ContactRecorder,
    capture_snapshot,
    excitation_actions,
    load_snapshot,
    restore_snapshot,
    robot_pose,
    robot_state,
)


def training_windows(artifacts: Path, mapping: ControllerMap, coordinates: NumericalCoordinates):
    features, residuals, nominal, ids, states = [], [], [], [], []
    for episode in range(80):
        path = artifacts / "episodes" / f"episode_{episode:03d}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as payload:
            for offset in range(len(payload["actions"]) - 16 + 1):
                actions = payload["actions"][offset : offset + 16]
                initial_pos, initial_rot = (
                    payload["positions"][offset],
                    payload["rotations"][offset],
                )
                k = nominal_effect(initial_pos, initial_rot, actions, mapping)
                d = endpoint_effect(
                    initial_pos,
                    initial_rot,
                    payload["positions"][offset + 16],
                    payload["rotations"][offset + 16],
                )
                state = coordinates.state(payload["states"][offset])
                features.append(np.r_[state, coordinates.action(actions).ravel()])
                residuals.append(effect_discrepancy(k[None], d[None])[0])
                nominal.append(k)
                states.append(state)
                ids.append((episode, offset))
    return np.array(features), np.array(residuals), np.array(nominal), ids, np.array(states)


def fit_ridge(features: np.ndarray, targets: np.ndarray) -> dict:
    """Minimize mean((XW+b-Y)^2) + 0.001*sum(W^2), intercept unpenalized."""
    feature_mean, feature_std = features.mean(axis=0), features.std(axis=0, ddof=0)
    feature_scale = np.where(feature_std > 1e-12, feature_std, 1.0)
    standardized = (features - feature_mean) / feature_scale
    target_mean = targets.mean(axis=0)
    penalty = 0.001 * targets.size
    coefficients = np.linalg.solve(
        standardized.T @ standardized + penalty * np.eye(features.shape[1]),
        standardized.T @ (targets - target_mean),
    )
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "coefficients": coefficients,
    }


def ridge_prediction(fit: dict, features: np.ndarray) -> np.ndarray:
    return (features - fit["feature_mean"]) / fit["feature_scale"] @ fit["coefficients"] + fit[
        "target_mean"
    ]


def summarize_ridge(targets: np.ndarray, predictions: np.ndarray, mean: np.ndarray) -> dict:
    result = {}
    for channel, dimensions, scale in [
        ("translation", slice(0, 3), 1000),
        ("rotation", slice(3, 6), 180 / np.pi),
    ]:
        y, pred, baseline = targets[:, dimensions], predictions[:, dimensions], mean[dimensions]
        ridge_sse = float(np.square(y - pred).sum())
        mean_sse = float(np.square(y - baseline).sum())
        result[channel] = {
            "count": len(y),
            "ridge_rmse": float(np.sqrt(np.square(y - pred).sum(axis=1).mean()) * scale)
            if len(y)
            else None,
            "zero_rmse": float(np.sqrt(np.square(y).sum(axis=1).mean()) * scale)
            if len(y)
            else None,
            "training_mean_rmse": float(np.sqrt(np.square(y - baseline).sum(axis=1).mean()) * scale)
            if len(y)
            else None,
            "squared_error_reduction": 1 - ridge_sse / mean_sse if mean_sse > 1e-20 else None,
            "units": "mm" if channel == "translation" else "degrees",
            "undefined_ratio_reason": None
            if mean_sse > 1e-20
            else "empty_or_negligible_training_mean_SSE",
        }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def residual_program(root: Path, artifacts: Path) -> None:
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    verify_release(root)
    audit = root / "audit"
    if (artifacts / "residual_run.json").exists():
        raise FileExistsError("The fixed reference replay program must not be silently rerun")
    marker = {
        "started_utc": datetime.now(UTC).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (artifacts / "residual_run.json").write_text(json.dumps(marker, indent=2) + "\n")
    manifest = json.loads((root / "configs/collection_manifest.json").read_text())
    interface = json.loads((artifacts / "numerical_interface.json").read_text())
    mapping = ControllerMap.model_validate(interface["controller_map"])
    coordinates = NumericalCoordinates.model_validate(interface["coordinates"])
    replays = artifacts / "replays"
    replays.mkdir(exist_ok=True)
    scene = (
        Path(get_libero_path("bddl_files"))
        / "libero_goal/push_the_plate_to_the_front_of_the_stove.bddl"
    )
    env = ControlEnv(bddl_file_name=str(scene), has_offscreen_renderer=False, use_camera_obs=False)
    rows, features, residuals, duplicate_rows, snapshot_rows, failures = [], [], [], [], [], []
    started = time.monotonic()
    previous_episode = None
    try:
        for episode, offset in manifest["fixed_snapshot_order"]:
            path = artifacts / "snapshots" / f"ep{episode:03d}_t{offset:03d}.npz"
            if not path.exists():
                snapshot_rows.append({"episode_id": episode, "offset": offset, "status": "missing"})
                continue
            if previous_episode != episode:
                env.reset_from_xml_string(
                    (artifacts / "models" / f"episode_{episode:03d}.xml").read_text()
                )
                previous_episode = episode
                recorder = ContactRecorder(env)
            snapshot = load_snapshot(path)
            snapshot_rows.append(
                {
                    "episode_id": episode,
                    "offset": offset,
                    "status": "valid",
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "intended_reset": "contact_attempt"
                    if episode % 2
                    else "native_free_motion_attempt",
                }
            )
            for reference in range(2):
                restore_snapshot(env, snapshot)
                initial_pos, initial_rot = robot_pose(env)
                state = robot_state(env)
                rng = np.random.default_rng(
                    np.random.SeedSequence([52026, episode, offset, reference])
                )
                actions, boundary = excitation_actions(initial_pos, initial_rot, 16, rng, mapping)
                name = f"ep{episode:03d}_t{offset:03d}_chunk{reference}"
                action_path = replays / f"{name}_actions.npy"
                np.save(
                    action_path, actions, allow_pickle=False
                )  # Saved before any physical replay.
                expected = nominal_effect(initial_pos, initial_rot, actions, mapping)
                outcomes, traces = [], []
                for repeat in range(2):
                    restore_snapshot(env, snapshot)
                    trace, contact_steps, contact_pairs = [], 0, []
                    for action in actions:
                        _, pairs = recorder.step(action)
                        contact_steps += bool(pairs)
                        contact_pairs.append(pairs)
                        integration = capture_snapshot(env).integration_state
                        if not np.isfinite(integration).all():
                            raise FloatingPointError(f"Non-finite replay {name}/{repeat}")
                        trace.append(integration)
                    measured = endpoint_effect(initial_pos, initial_rot, *robot_pose(env))
                    discrepancy = effect_discrepancy(expected[None], measured[None])[0]
                    trace = np.array(trace)
                    np.savez_compressed(
                        replays / f"{name}_repeat{repeat}.npz",
                        integration_states=trace,
                        nominal_effect=expected,
                        measured_effect=measured,
                        finger_displacement=robot_state(env)[6:] - state[6:],
                    )
                    outcomes.append(measured)
                    traces.append(trace)
                    row = {
                        "episode_id": episode,
                        "snapshot_offset": offset,
                        "reference_chunk": reference,
                        "repeat": repeat,
                        "intended_reset": snapshot_rows[-1]["intended_reset"],
                        "actual_robot_object_contact": bool(contact_steps),
                        "contact_control_steps": contact_steps,
                        "translation_residual_mm": float(np.linalg.norm(discrepancy[:3]) * 1000),
                        "rotation_residual_degrees": float(
                            np.linalg.norm(discrepancy[3:]) * 180 / np.pi
                        ),
                        "finger_1_displacement_m": float(robot_state(env)[6] - state[6]),
                        "finger_2_displacement_m": float(robot_state(env)[7] - state[7]),
                        "action_l2": float(np.linalg.norm(actions)),
                        "clipped_action_coordinates": int((np.abs(actions) > 1).sum()),
                        "boundary_reflections": int(boundary.sum()),
                        "status": "valid",
                    }
                    for index in range(6):
                        row[f"nominal_{index}"] = float(expected[index])
                        row[f"measured_{index}"] = float(measured[index])
                        row[f"discrepancy_{index}"] = float(discrepancy[index])
                    rows.append(row)
                    features.append(
                        np.r_[coordinates.state(state), coordinates.action(actions).ravel()]
                    )
                    residuals.append(discrepancy)
                    (replays / f"{name}_repeat{repeat}_contacts.json").write_text(
                        json.dumps(contact_pairs)
                    )
                duplicate = effect_discrepancy(outcomes[0][None], outcomes[1][None])[0]
                identical = np.array_equal(traces[0], traces[1])
                duplicate_rows.append(
                    {
                        "episode_id": episode,
                        "snapshot_offset": offset,
                        "reference_chunk": reference,
                        "translation_duplicate_mm": float(np.linalg.norm(duplicate[:3]) * 1000),
                        "rotation_duplicate_degrees": float(
                            np.linalg.norm(duplicate[3:]) * 180 / np.pi
                        ),
                        "integration_max_abs_difference": float(
                            np.abs(traces[0] - traces[1]).max()
                        ),
                        "integration_bit_identical": identical,
                    }
                )
                if not identical:
                    failures.append(
                        {"reference": name, "reason": "full_snapshot_duplicate_not_bit_identical"}
                    )
            print(
                json.dumps({"episode": episode, "offset": offset, "completed_replays": len(rows)}),
                flush=True,
            )
    finally:
        env.close()
        marker.update(
            wall_seconds=time.monotonic() - started, finished_utc=datetime.now(UTC).isoformat()
        )
        (artifacts / "residual_run.json").write_text(json.dumps(marker, indent=2) + "\n")
        write_csv(audit / "residual_replays.csv", rows)
        write_csv(audit / "duplicate_variability.csv", duplicate_rows)
        (audit / "snapshot_manifest.json").write_text(json.dumps(snapshot_rows, indent=2) + "\n")
        (audit / "replay_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise ValueError(
            "Required full-snapshot replay invariant failed; dependent analysis blocked"
        )
    train_x, train_y, train_k, window_ids, train_states = training_windows(
        artifacts, mapping, coordinates
    )
    transform = fit_effect_transform(train_k, window_ids)
    resolved = json.loads((root / "resolved_environment.json").read_text())
    resolved["effect_transform"] = transform.model_dump() | {
        "fitted": True,
        "source": "nominal_complete_TRAIN_windows_only",
    }
    subset = reference_indices(len(window_ids))
    np.savez_compressed(
        artifacts / "training_preparation.npz",
        window_ids=np.array(window_ids),
        nominal_effects=train_k,
        normalized_states=train_states,
        reference_indices=subset,
    )
    (audit / "training_window_manifest.json").write_text(
        json.dumps(
            {
                "window_count": len(window_ids),
                "ids_sha256": transform.training_window_ids_sha256,
                "canonical_order": "episode_id_then_start_offset",
                "reference_window_ids": [window_ids[i] for i in subset],
                "full_ids_path": str(artifacts / "training_preparation.npz"),
            },
            indent=2,
        )
        + "\n"
    )
    (root / "resolved_environment.json").write_text(json.dumps(resolved, indent=2) + "\n")
    fit = fit_ridge(train_x, train_y)
    predictions = ridge_prediction(fit, np.asarray(features))
    residuals = np.asarray(residuals)
    contacts = np.array([row["actual_robot_object_contact"] for row in rows])
    ridge = {
        "objective": (
            "mean squared error over all training windows and six targets "
            "+ 0.001 * sum(coefficient**2)"
        ),
        "training_window_count": len(train_x),
        "feature_count": train_x.shape[1],
        "intercept_penalized": False,
        "hyperparameter_search": False,
        "heldout_source": (
            "fixed reference replays from evaluation episodes only; "
            "repeats and snapshots are correlated"
        ),
        "overall": summarize_ridge(residuals, predictions, fit["target_mean"]),
        "contact": summarize_ridge(residuals[contacts], predictions[contacts], fit["target_mean"]),
        "no_contact": summarize_ridge(
            residuals[~contacts], predictions[~contacts], fit["target_mean"]
        ),
    }
    np.savez_compressed(artifacts / "ridge_fit.npz", **fit)
    (audit / "ridge_diagnostic.json").write_text(json.dumps(ridge, indent=2) + "\n")
    for row, prediction in zip(rows, predictions, strict=True):
        for index, value in enumerate(prediction):
            row[f"ridge_prediction_{index}"] = float(value)
    write_csv(audit / "residual_replays.csv", rows)
    print(
        json.dumps(
            {"replays": len(rows), "training_windows": len(train_x), "ridge": ridge}, indent=2
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    residual_program(Path.cwd(), args.artifacts.resolve())
