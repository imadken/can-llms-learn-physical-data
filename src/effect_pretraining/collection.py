"""The fixed 100-attempt numerical program; no demonstrations or task feedback."""

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from effect_pretraining.integrity import sha256_file, verify_release
from effect_pretraining.simulation import (
    ContactRecorder,
    capture_snapshot,
    contact_geometry,
    contact_setup,
    controller_map,
    excitation_actions,
    numerical_coordinates,
    robot_pose,
    robot_state,
    save_snapshot,
)


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_id: int
    split: str
    intended_reset: str
    status: str
    retained_transitions: int
    actual_contact_steps: int
    setup_steps: int
    warmup_steps: int
    seconds: float
    invalid_reason: str | None = None
    geometry: dict | None = None


def split_name(episode_id: int) -> str:
    return "train" if episode_id < 80 else "development" if episode_id < 90 else "evaluation"


def collect(root: Path, artifacts: Path) -> None:
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    protected = verify_release(root)
    config = json.loads((root / "configs/preliminary_locked.json").read_text())
    artifacts.mkdir(parents=True, exist_ok=True)
    if (artifacts / "collection_run.json").exists():
        raise FileExistsError("The fixed collection must not be silently rerun")
    for name in ("episodes", "snapshots", "models"):
        (artifacts / name).mkdir(exist_ok=True)
    metadata = {
        "started_utc": datetime.now(UTC).isoformat(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "protected_hashes": protected,
        "config_sha256": sha256_file(root / "configs/preliminary_locked.json"),
        "rng_derivation": "SeedSequence([42026, episode_id]); native reset seed 42026+episode_id",
        "warmup": {"steps": 10, "action": [0, 0, 0, 0, 0, 0, -1]},
        "contact_observation": "any robot-object contact across every physics substep",
    }
    (artifacts / "collection_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    scene = (
        Path(get_libero_path("bddl_files"))
        / "libero_goal"
        / (config["collector"]["scene"] + ".bddl")
    )
    env = ControlEnv(bddl_file_name=str(scene), has_offscreen_renderer=False, use_camera_obs=False)
    records = []
    started = time.monotonic()
    try:
        mapping = controller_map(env)
        coordinates = numerical_coordinates(env)
        (artifacts / "numerical_interface.json").write_text(
            json.dumps(
                {
                    "controller_map": mapping.model_dump(),
                    "coordinates": coordinates.model_dump(),
                    "control_frequency_hz": env.env.control_freq,
                    "physics_timestep_seconds": env.env.model_timestep,
                },
                indent=2,
            )
            + "\n"
        )
        for episode in range(100):
            attempt_start = time.monotonic()
            state_rows, positions, rotations, contact_rows, pairs, goal_rows = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            step_seconds, executed = [], []
            setup = {"setup_steps": 0, "executed_setup_actions": []}
            geometry = None
            reason = None
            warmup_count = 0
            try:
                np.random.seed(42026 + episode)
                # One native reset attempt; the public wrapper retries randomization indefinitely.
                env.env.reset()
                for _ in range(10):
                    env.step([0, 0, 0, 0, 0, 0, -1])
                    warmup_count += 1
                if episode % 2:
                    geometry = contact_geometry(env)
                    setup = contact_setup(env, geometry, mapping)
                model_xml = env.sim.model.get_xml()
                (artifacts / "models" / f"episode_{episode:03d}.xml").write_text(model_xml)
                position, rotation = robot_pose(env)
                rng = np.random.default_rng(np.random.SeedSequence([42026, episode]))
                actions, boundary = excitation_actions(position, rotation, 200, rng, mapping)
                recorder = ContactRecorder(env)
                for offset in range(201):
                    state = robot_state(env)
                    position, rotation = robot_pose(env)
                    if not np.isfinite(state).all() or not np.isfinite(env.sim.data.qvel).all():
                        raise FloatingPointError("Non-finite simulator state")
                    state_rows.append(state)
                    positions.append(position)
                    rotations.append(rotation)
                    if episode >= 90 and offset in config["preliminary_1"]["snapshot_offsets"]:
                        save_snapshot(
                            artifacts / "snapshots" / f"ep{episode:03d}_t{offset:03d}.npz",
                            capture_snapshot(env),
                        )
                    if offset == 200:
                        break
                    step_start = time.monotonic()
                    _, contact_pairs = recorder.step(actions[offset])
                    step_seconds.append(time.monotonic() - step_start)
                    executed.append(actions[offset])
                    contact_rows.append(bool(contact_pairs))
                    pairs.append(contact_pairs)
                    controller = env.robots[0].controller
                    goal_rows.append(np.r_[controller.goal_pos, controller.goal_ori.ravel()])
                np.savez_compressed(
                    artifacts / "episodes" / f"episode_{episode:03d}.npz",
                    states=np.array(state_rows),
                    positions=np.array(positions),
                    rotations=np.array(rotations),
                    actions=np.array(executed),
                    robot_object_contact=np.array(contact_rows),
                    boundary_reflections=boundary,
                    controller_goals=np.array(goal_rows),
                    step_wall_seconds=np.array(step_seconds),
                    setup_actions=np.asarray(setup["executed_setup_actions"]).reshape(-1, 7),
                )
                (artifacts / "episodes" / f"episode_{episode:03d}_contacts.json").write_text(
                    json.dumps(pairs)
                )
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                # Preserve whatever this attempt produced; it is never replaced.
                np.savez_compressed(
                    artifacts / "episodes" / f"episode_{episode:03d}_invalid.npz",
                    states=np.array(state_rows),
                    actions=np.array(executed),
                    positions=np.array(positions),
                    rotations=np.array(rotations),
                    robot_object_contact=np.array(contact_rows),
                )
            record = EpisodeRecord(
                episode_id=episode,
                split=split_name(episode),
                intended_reset="contact_attempt" if episode % 2 else "native_free_motion_attempt",
                status="invalid" if reason else "complete",
                retained_transitions=len(executed),
                actual_contact_steps=sum(contact_rows),
                setup_steps=setup["setup_steps"],
                warmup_steps=warmup_count,
                seconds=time.monotonic() - attempt_start,
                invalid_reason=reason,
                geometry=geometry,
            )
            records.append(record.model_dump())
            (artifacts / "episode_records.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record.model_dump(exclude={"geometry"})), flush=True)
    finally:
        metadata["wall_seconds"] = time.monotonic() - started
        metadata["finished_utc"] = datetime.now(UTC).isoformat()
        (artifacts / "collection_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    collect(Path.cwd(), args.artifacts.resolve())
