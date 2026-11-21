"""Fixed paired acquisition generation, complete chunk replay, and native evidence criterion."""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from effect_pretraining.coordinates import (
    ControllerMap,
    NumericalCoordinates,
    effect_discrepancy,
    endpoint_effect,
    nominal_effect,
)
from effect_pretraining.effect_scaling import EffectTransform
from effect_pretraining.integrity import sha256_file, verify_release
from effect_pretraining.residuals import write_csv
from effect_pretraining.resources import accelerator_job
from effect_pretraining.simulation import (
    capture_snapshot,
    load_snapshot,
    restore_snapshot,
    robot_pose,
)
from effect_pretraining.training import numerical_model


def score_effect(realized: np.ndarray, goal: np.ndarray) -> dict:
    discrepancy = effect_discrepancy(goal[None], realized[None])[0]
    translation = float(np.linalg.norm(discrepancy[:3]) * 1000)
    rotation = float(np.linalg.norm(discrepancy[3:]) * 180 / np.pi)
    return {
        "translation_error_mm": translation,
        "rotation_error_degrees": rotation,
        "joint_attained": translation <= 10 and rotation <= 5,
    }


def summarize_scores(rows: list[dict]) -> dict:
    return {
        "targets": len(rows),
        "mean_translation_error_mm": float(np.mean([r["translation_error_mm"] for r in rows])),
        "mean_rotation_error_degrees": float(np.mean([r["rotation_error_degrees"] for r in rows])),
        "joint_attainment_percent": float(100 * np.mean([r["joint_attained"] for r in rows])),
    }


def acquisition_evidence(rows: list[dict]) -> dict:
    result = {}
    for arm, domain in [("K", "nominal"), ("D", "physical")]:
        seeds = {}
        for seed in (101, 202):
            groups = {}
            for mode in ("matched", "masked", "alternate", "P"):
                selected = [
                    r
                    for r in rows
                    if r["domain"] == domain
                    and r["seed"] == seed
                    and (
                        (mode == "P" and r["arm"] == "P") or (r["arm"] == arm and r["mode"] == mode)
                    )
                ]
                if not selected:
                    raise ValueError(
                        "All existing paired controls are required for the native criterion"
                    )
                groups[mode] = summarize_scores(selected)
            matched, alternate = groups["matched"], groups["alternate"]
            margin_masked = (
                matched["joint_attainment_percent"] - groups["masked"]["joint_attainment_percent"]
            )
            margin_p = matched["joint_attainment_percent"] - groups["P"]["joint_attainment_percent"]
            t_better = matched["mean_translation_error_mm"] < alternate["mean_translation_error_mm"]
            r_better = (
                matched["mean_rotation_error_degrees"] < alternate["mean_rotation_error_degrees"]
            )
            follows = (
                t_better
                and matched["mean_rotation_error_degrees"]
                <= alternate["mean_rotation_error_degrees"]
            ) or (
                r_better
                and matched["mean_translation_error_mm"] <= alternate["mean_translation_error_mm"]
            )
            seeds[str(seed)] = {
                "groups": groups,
                "joint_margin_over_masked_pp": margin_masked,
                "joint_margin_over_P_pp": margin_p,
                "target_following": follows,
                "criterion_met": margin_masked >= 10 and margin_p >= 10 and follows,
            }
        met = sum(item["criterion_met"] for item in seeds.values())
        result[arm] = {
            "domain": domain,
            "seeds": seeds,
            "label": "positive exploratory acquisition evidence"
            if met == 2
            else "mixed exploratory evidence"
            if met == 1
            else "acquisition not demonstrated within this pilot budget",
        }
    return result


def bank_arrays(root: Path, artifacts: Path, coordinates: NumericalCoordinates) -> dict:
    manifest = json.loads((root / "audit/acquisition_bank_manifest.json").read_text())
    states, positions, rotations, nominal, measured = [], [], [], [], []
    for item in manifest["snapshots"]:
        episode, offset = item["episode_id"], item["offset"]
        with np.load(
            artifacts / "episodes" / f"episode_{episode:03d}.npz", allow_pickle=False
        ) as saved:
            states.append(coordinates.state(saved["states"][offset]))
            positions.append(saved["positions"][offset])
            rotations.append(saved["rotations"][offset])
        k, d = [], []
        for reference in range(2):
            path = (
                artifacts
                / "replays"
                / f"ep{episode:03d}_t{offset:03d}_chunk{reference}_repeat0.npz"
            )
            with np.load(path, allow_pickle=False) as saved:
                k.append(saved["nominal_effect"])
                d.append(saved["measured_effect"])
        nominal.append(k)
        measured.append(d)
    return {
        "states": np.array(states, dtype=np.float32),
        "positions": np.array(positions),
        "rotations": np.array(rotations),
        "nominal": np.array(nominal),
        "measured": np.array(measured),
    }


def generate(root: Path, artifacts: Path) -> None:
    verify_release(root)
    status = json.loads((root / "audit/acquisition_training.json").read_text())
    common = status["common_completed_updates"]
    if common not in (250, 500, 750, 1000):
        raise ValueError("Acquisition evaluation requires a completed common round")
    output = artifacts / "acquisition_generated.npz"
    if output.exists():
        raise FileExistsError("Do not repeat fixed acquisition generations")
    resolved = json.loads((root / "resolved_environment.json").read_text())
    interface = json.loads((artifacts / "numerical_interface.json").read_text())
    coordinates = NumericalCoordinates.model_validate(interface["coordinates"])
    transform = EffectTransform.model_validate(
        {
            key: value
            for key, value in resolved["effect_transform"].items()
            if key in EffectTransform.model_fields
        }
    )
    manifest = json.loads((root / "audit/acquisition_bank_manifest.json").read_text())
    bank = bank_arrays(root, artifacts, coordinates)
    count = len(bank["states"])
    generated = dict(bank)
    summary = {
        "status": "started",
        "common_checkpoint_updates": common,
        "snapshots": count,
        "source_episode_ids": manifest["source_episode_ids"],
        "generation_count": 0,
        "checkpoint_records": {},
    }
    with accelerator_job(artifacts, "fixed_acquisition_generation"):
        for seed in (101, 202):
            noises = []
            for item in manifest["snapshots"]:
                stream = int(
                    np.random.SeedSequence(
                        [62026, seed, item["episode_id"], item["offset"]]
                    ).generate_state(1)[0]
                )
                generator = torch.Generator().manual_seed(stream)
                noises.append(torch.randn(16, 32, generator=generator))
            noise = torch.stack(noises).cuda()
            generated[f"noise_{seed}"] = noise.cpu().numpy()
            state = torch.from_numpy(bank["states"]).cuda()
            for arm in ("P", "K", "D"):
                key = f"{arm}_{seed}"
                record = next(
                    c
                    for c in status["fits"][key]["checkpoints"]
                    if c["completed_updates"] == common
                )
                if sha256_file(Path(record["path"])) != record["sha256"]:
                    raise ValueError("Acquisition checkpoint hash mismatch")
                model = numerical_model(resolved["runtime_paths"], artifacts, seed).cuda().eval()
                payload = torch.load(record["path"], map_location="cpu", weights_only=True)
                model.load_state_dict(payload["trainable"], strict=False)
                del payload
                summary["checkpoint_records"][key] = record
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if arm == "P":
                        values = model.generate(
                            state,
                            torch.zeros(count, 6, device="cuda"),
                            torch.zeros(count, dtype=torch.bool, device="cuda"),
                            noise,
                        )
                        generated[key] = values.float().cpu().numpy()
                        summary["generation_count"] += count
                    else:
                        # K and D receive the same measured goals in the common physical test.
                        effects = transform.apply(bank["measured"]).reshape(count * 2, 6)
                        values = model.generate(
                            state.repeat_interleave(2, 0),
                            torch.tensor(effects, dtype=torch.float32, device="cuda"),
                            torch.zeros(count * 2, dtype=torch.bool, device="cuda"),
                            noise.repeat_interleave(2, 0),
                        )
                        generated[f"{key}_physical"] = (
                            values.float().cpu().numpy().reshape(count, 2, 16, 7)
                        )
                        masked = model.generate(
                            state,
                            torch.zeros(count, 6, device="cuda"),
                            torch.ones(count, dtype=torch.bool, device="cuda"),
                            noise,
                        )
                        generated[f"{key}_masked"] = masked.float().cpu().numpy()
                        summary["generation_count"] += count * 3
                        if arm == "K":
                            native = transform.apply(bank["nominal"]).reshape(count * 2, 6)
                            values = model.generate(
                                state.repeat_interleave(2, 0),
                                torch.tensor(native, dtype=torch.float32, device="cuda"),
                                torch.zeros(count * 2, dtype=torch.bool, device="cuda"),
                                noise.repeat_interleave(2, 0),
                            )
                            generated[f"{key}_nominal"] = (
                                values.float().cpu().numpy().reshape(count, 2, 16, 7)
                            )
                            summary["generation_count"] += count * 2
                if not all(np.isfinite(value).all() for value in generated.values()):
                    raise FloatingPointError("Non-finite acquisition generations")
                print(
                    json.dumps(
                        {"generated_fit": key, "total_unique_chunks": summary["generation_count"]}
                    ),
                    flush=True,
                )
                del model, values
                gc.collect()
                torch.cuda.empty_cache()
            del state, noise
    if summary["generation_count"] > 1024:
        raise ValueError("Generated chunk ceiling exceeded")
    np.savez_compressed(output, **generated)
    with np.load(artifacts / "training_arrays.npz", allow_pickle=False) as training:
        nominal_training = training["nominal_z"]
        physical_input = transform.apply(bank["measured"]).reshape(-1, 6)
        outside = (physical_input < nominal_training.min(axis=0)) | (
            physical_input > nominal_training.max(axis=0)
        )
    separation = [score_effect(pair[0], pair[1]) for pair in bank["measured"]]
    summary.update(
        status="complete",
        generated_array_sha256=sha256_file(output),
        physical_goals_outside_training_nominal_range_percent=float(
            100 * outside.any(axis=1).mean()
        ),
        physical_goal_outside_range_percent_by_coordinate=(100 * outside.mean(axis=0)).tolist(),
        target_separation=separation,
        nearly_coincident_goal_pairs=sum(row["joint_attained"] for row in separation),
    )
    (root / "audit/acquisition_generation.json").write_text(json.dumps(summary, indent=2) + "\n")


def replay_and_score(root: Path, artifacts: Path) -> None:
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    verify_release(root)
    if (root / "audit/acquisition_evidence.json").exists():
        raise FileExistsError("Do not repeat fixed acquisition replays")
    manifest = json.loads((root / "audit/acquisition_bank_manifest.json").read_text())
    interface = json.loads((artifacts / "numerical_interface.json").read_text())
    coordinates = NumericalCoordinates.model_validate(interface["coordinates"])
    mapping = ControllerMap.model_validate(interface["controller_map"])
    generated_path = artifacts / "acquisition_generated.npz"
    metadata = json.loads((root / "audit/acquisition_generation.json").read_text())
    if sha256_file(generated_path) != metadata["generated_array_sha256"]:
        raise ValueError("Generated arrays changed before replay")
    with np.load(generated_path, allow_pickle=False) as saved:
        generated = {key: saved[key].copy() for key in saved.files}
    replays = artifacts / "acquisition_replays"
    replays.mkdir(exist_ok=True)
    bddl = (
        Path(get_libero_path("bddl_files"))
        / "libero_goal/push_the_plate_to_the_front_of_the_stove.bddl"
    )
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False)
    rows, replay_rows, sensitivity = [], [], []
    previous_episode = None
    started = time.monotonic()
    try:
        for index, item in enumerate(manifest["snapshots"]):
            episode, offset = item["episode_id"], item["offset"]
            if previous_episode != episode:
                env.reset_from_xml_string(
                    (artifacts / "models" / f"episode_{episode:03d}.xml").read_text()
                )
                previous_episode = episode
            snapshot = load_snapshot(Path(item["path"]))
            position, rotation = generated["positions"][index], generated["rotations"][index]
            for seed in (101, 202):
                chunks = {"P": generated[f"P_{seed}"][index]}
                for arm in ("K", "D"):
                    chunks[f"{arm}_masked"] = generated[f"{arm}_{seed}_masked"][index]
                    for target in (0, 1):
                        chunks[f"{arm}_{target}"] = generated[f"{arm}_{seed}_physical"][
                            index, target
                        ]
                physical, nominal = {}, {}
                for case, normalized_actions in chunks.items():
                    actions = coordinates.action(normalized_actions, inverse=True)
                    executed = np.clip(actions, coordinates.action_min, coordinates.action_max)
                    restore_snapshot(env, snapshot)
                    traces = []
                    for action in executed:
                        env.step(action)
                        traces.append(capture_snapshot(env).integration_state)
                    if not np.isfinite(traces).all():
                        raise FloatingPointError("Non-finite generated chunk replay")
                    physical[case] = endpoint_effect(position, rotation, *robot_pose(env))
                    nominal[case] = nominal_effect(position, rotation, actions, mapping)
                    name = f"seed{seed}_ep{episode:03d}_t{offset:03d}_{case}"
                    np.savez_compressed(
                        replays / f"{name}.npz",
                        actions=actions,
                        executed_actions=executed,
                        integration_states=np.array(traces),
                        measured_effect=physical[case],
                        nominal_effect=nominal[case],
                    )
                    replay_rows.append(
                        {
                            "seed": seed,
                            "episode_id": episode,
                            "snapshot_offset": offset,
                            "case": case,
                            "control_actions": 16,
                            "status": "valid",
                            "clipped_action_coordinates": int((actions != executed).sum()),
                        }
                    )
                # Native K goals are scored analytically, without extra physical rollouts.
                native_k = [
                    nominal_effect(
                        position,
                        rotation,
                        coordinates.action(
                            generated[f"K_{seed}_nominal"][index, target], inverse=True
                        ),
                        mapping,
                    )
                    for target in (0, 1)
                ]
                for target in (0, 1):
                    for domain, goal_bank in [
                        ("physical", generated["measured"]),
                        ("nominal", generated["nominal"]),
                    ]:
                        goal = goal_bank[index, target]
                        cases = [("P", "P", None)]
                        if domain == "physical":
                            for arm in ("K", "D"):
                                cases.extend(
                                    [
                                        (arm, "matched", f"{arm}_{target}"),
                                        (arm, "masked", f"{arm}_masked"),
                                        (arm, "alternate", f"{arm}_{1 - target}"),
                                    ]
                                )
                        else:
                            cases.extend(
                                [
                                    ("K", "matched", "native_matched"),
                                    ("K", "masked", "K_masked"),
                                    ("K", "alternate", "native_alternate"),
                                ]
                            )
                        for arm, mode, case in cases:
                            if case == "native_matched":
                                realized = native_k[target]
                            elif case == "native_alternate":
                                realized = native_k[1 - target]
                            else:
                                realized = (physical if domain == "physical" else nominal)[
                                    "P" if case is None else case
                                ]
                            supplied = goal_bank[index, 1 - target] if mode == "alternate" else goal
                            row = {
                                "seed": seed,
                                "episode_id": episode,
                                "snapshot_offset": offset,
                                "target_id": target,
                                "domain": domain,
                                "arm": arm,
                                "mode": mode,
                                **score_effect(realized, goal),
                            }
                            supplied_score = score_effect(realized, supplied)
                            row.update(
                                {f"supplied_{key}": value for key, value in supplied_score.items()}
                            )
                            rows.append(row)
                for arm in ("K", "D"):
                    sensitivity.append(
                        {
                            "seed": seed,
                            "episode_id": episode,
                            "snapshot_offset": offset,
                            "arm": arm,
                            "common_goal_action_l2_change": float(
                                np.linalg.norm(
                                    generated[f"{arm}_{seed}_physical"][index, 0]
                                    - generated[f"{arm}_{seed}_physical"][index, 1]
                                )
                            ),
                        }
                    )
            print(
                json.dumps(
                    {"scored_snapshot": index + 1, "physical_chunk_replays": len(replay_rows)}
                ),
                flush=True,
            )
    finally:
        env.close()
        write_csv(root / "audit/acquisition_targets.csv", rows)
        write_csv(root / "audit/acquisition_replays.csv", replay_rows)
        write_csv(root / "audit/acquisition_action_sensitivity.csv", sensitivity)
    evidence = acquisition_evidence(rows)
    common = {}
    for seed in (101, 202):
        common[str(seed)] = {
            arm: summarize_scores(
                [
                    r
                    for r in rows
                    if r["domain"] == "physical"
                    and r["seed"] == seed
                    and r["arm"] == arm
                    and r["mode"] == ("P" if arm == "P" else "matched")
                ]
            )
            for arm in ("P", "K", "D")
        }
    evidence.update(
        status="complete" if metadata["common_checkpoint_updates"] == 1000 else "partial",
        checkpoint_updates=metadata["common_checkpoint_updates"],
        common_physical_goal=common,
        unique_generations=metadata["generation_count"],
        physical_chunk_replays=len(replay_rows),
        source_episode_ids=manifest["source_episode_ids"],
        replay_wall_seconds=time.monotonic() - started,
        independent_training_seeds=2,
        uncertainty_limit=(
            "Two training seeds, with repeated snapshots/targets from four source episodes; "
            "no equivalence or training-seed significance claim"
        ),
    )
    (root / "audit/acquisition_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--phase", choices=["generate", "replay"], required=True)
    args = parser.parse_args()
    (generate if args.phase == "generate" else replay_and_score)(
        Path.cwd(), args.artifacts.resolve()
    )
