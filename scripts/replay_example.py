"""Manual review reproduction of an existing snapshot; no automatic campaign execution."""

import argparse
import json
from pathlib import Path

import numpy as np

from effect_pretraining.simulation import capture_snapshot, load_snapshot, restore_snapshot


def replay(directory: Path) -> dict:
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    snapshot_path = next(directory.glob("ep*_t000.npz"))
    model_path = next(directory.glob("episode_*.xml"))
    actions_path = next(directory.glob("*_actions.npy"))
    expected_path = next(directory.glob("*_repeat0.npz"))
    bddl = (
        Path(get_libero_path("bddl_files"))
        / "libero_goal/push_the_plate_to_the_front_of_the_stove.bddl"
    )
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False)
    try:
        env.reset_from_xml_string(model_path.read_text())
        restore_snapshot(env, load_snapshot(snapshot_path))
        trace = []
        for action in np.load(actions_path, allow_pickle=False):
            env.step(action)
            trace.append(capture_snapshot(env).integration_state)
        with np.load(expected_path, allow_pickle=False) as expected:
            target = expected["integration_states"]
            actual = np.array(trace)
            return {
                "control_actions": len(actual),
                "integration_bit_identical": np.array_equal(actual, target),
                "integration_max_abs_difference": float(np.abs(actual - target).max()),
            }
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.example_dir), indent=2))
