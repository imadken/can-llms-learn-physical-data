"""Run explicitly against existing LIBERO dependencies before collection."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from effect_pretraining.coordinates import nominal_effect
from effect_pretraining.simulation import (
    ContactRecorder,
    capture_snapshot,
    contact_geometry,
    controller_map,
    excitation_actions,
    load_snapshot,
    numerical_coordinates,
    restore_snapshot,
    robot_pose,
    robot_state,
    save_snapshot,
)


def test_source_mapping_and_serialized_complete_replay(tmp_path):
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from robosuite.utils.control_utils import set_goal_orientation, set_goal_position

    task = (
        Path(get_libero_path("bddl_files"))
        / "libero_goal/push_the_plate_to_the_front_of_the_stove.bddl"
    )
    env = ControlEnv(bddl_file_name=str(task), has_offscreen_renderer=False, use_camera_obs=False)
    try:
        np.random.seed(842026)  # Separate engineering stream, outside the fixed corpus.
        env.env.reset()
        for _ in range(10):
            env.step([0, 0, 0, 0, 0, 0, -1])
        mapping = controller_map(env)
        coordinates = numerical_coordinates(env)
        np.testing.assert_allclose(coordinates.gripper_absolute_limits, [0.04, 0.04])
        position, rotation = robot_pose(env)
        action = np.array([1.1, 0.2, -0.3, 0.4, -0.5, 0.6, 1])
        scaled = env.robots[0].controller.scale_action(action[:6])
        np.testing.assert_allclose(mapping.scale(action[:6]), scaled)
        expected_pos = set_goal_position(scaled[:3], position)
        expected_ori = set_goal_orientation(scaled[3:], rotation)
        nominal = nominal_effect(position, rotation, action[None], mapping)
        np.testing.assert_allclose(nominal[:3], expected_pos - position)
        np.testing.assert_allclose(
            Rotation.from_rotvec(nominal[3:]).as_matrix(), rotation.T @ expected_ori, atol=5e-8
        )
        geometry = contact_geometry(env)
        assert geometry["table_support"]
        state = robot_state(env)
        np.testing.assert_allclose(coordinates.state(coordinates.state(state), inverse=True), state)
        snapshot = capture_snapshot(env)
        path = tmp_path / "complete_snapshot.npz"
        save_snapshot(path, snapshot)
        recovered = load_snapshot(path)
        actions, _ = excitation_actions(
            position, rotation, 16, np.random.default_rng(852026), mapping
        )
        traces = []
        recorder = ContactRecorder(env)
        for source in (snapshot, recovered):
            restore_snapshot(env, source)
            trace = []
            for command in actions:
                recorder.step(command)
                trace.append(capture_snapshot(env).integration_state)
            traces.append(np.array(trace))
        np.testing.assert_array_equal(traces[0], traces[1])
        report = {
            "serialized_full_snapshot_replay": "bit_identical",
            "physics_steps_compared": 16,
            "max_abs_integration_state_difference": float(np.abs(traces[0] - traces[1]).max()),
            "mapping_source_comparison": "passed",
            "numerical_inverse_transform": "passed",
            "contact_geometry_resolved": geometry,
        }
        Path("audit/simulation_correctness.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()
