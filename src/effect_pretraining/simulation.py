"""Source-resolved LIBERO collection and complete same-model numerical snapshots."""

import copy
import itertools
import json
import random
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from robosuite.utils.transform_utils import quat2axisangle

from effect_pretraining.coordinates import ControllerMap, NumericalCoordinates, ideal_pose_step


class Snapshot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    integration_state: NDArray
    controller: dict
    gripper_action: NDArray
    environment: dict
    observables: dict
    robot_buffers: dict
    numpy_rng: tuple
    python_rng: tuple


def robot_pose(env) -> tuple[NDArray, NDArray]:
    site = env.robots[0].eef_site_id
    return env.sim.data.site_xpos[site].copy(), env.sim.data.site_xmat[site].reshape(3, 3).copy()


def robot_state(env) -> NDArray:
    obs = env.env._get_observations(force_update=True)
    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"].copy()),
            obs["robot0_gripper_qpos"],
        ]
    )


def controller_map(env) -> ControllerMap:
    controller = env.robots[0].controller
    if not controller.use_delta or controller.impedance_mode != "fixed":
        raise ValueError("Expected fixed-impedance relative Cartesian OSC")
    if controller.orientation_limits is not None:
        raise ValueError("Non-null orientation limits require a source-exact composition adapter")
    if controller.interpolator_ori is not None or controller.interpolator_pos is not None:
        raise ValueError("Unexpected native controller interpolator")
    return ControllerMap(
        input_min=tuple(np.broadcast_to(controller.input_min, (6,))),
        input_max=tuple(np.broadcast_to(controller.input_max, (6,))),
        output_min=tuple(controller.output_min),
        output_max=tuple(controller.output_max),
        position_limits=None
        if controller.position_limits is None
        else controller.position_limits.tolist(),
    )


def numerical_coordinates(env) -> NumericalCoordinates:
    joints = env.robots[0].gripper.joints
    ranges = [env.sim.model.jnt_range[env.sim.model.joint_name2id(name)] for name in joints]
    low, high = env.env.action_spec
    return NumericalCoordinates(
        gripper_absolute_limits=tuple(float(np.abs(bounds).max()) for bounds in ranges),
        action_min=tuple(low),
        action_max=tuple(high),
    )


def capture_snapshot(env) -> Snapshot:
    physics = np.empty(
        mujoco.mj_stateSize(env.sim.model._model, mujoco.mjtState.mjSTATE_INTEGRATION)
    )
    mujoco.mj_getState(
        env.sim.model._model, env.sim.data._data, physics, mujoco.mjtState.mjSTATE_INTEGRATION
    )
    controller = {
        key: copy.deepcopy(value)
        for key, value in vars(env.robots[0].controller).items()
        if key != "sim"
    }
    observables = {
        name: {
            key: copy.deepcopy(value)
            for key, value in vars(observable).items()
            if not callable(value)
        }
        for name, observable in env.env._observables.items()
    }
    buffers = {
        name: copy.deepcopy(value)
        for name, value in vars(env.robots[0]).items()
        if name.startswith("recent_") or name == "torques"
    }
    return Snapshot(
        integration_state=physics,
        controller=controller,
        gripper_action=env.robots[0].gripper.current_action.copy(),
        environment={
            key: copy.deepcopy(getattr(env.env, key))
            for key in ("cur_time", "timestep", "done", "_obs_cache")
        },
        observables=observables,
        robot_buffers=buffers,
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
    )


def restore_snapshot(env, snapshot: Snapshot) -> None:
    mujoco.mj_setState(
        env.sim.model._model,
        env.sim.data._data,
        snapshot.integration_state,
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    env.sim.forward()
    # forward() recomputes derived values; restore integration/warm-start fields afterward.
    mujoco.mj_setState(
        env.sim.model._model,
        env.sim.data._data,
        snapshot.integration_state,
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    for key, value in snapshot.controller.items():
        setattr(env.robots[0].controller, key, copy.deepcopy(value))
    env.robots[0].gripper.current_action = snapshot.gripper_action.copy()
    for key, value in snapshot.environment.items():
        setattr(env.env, key, copy.deepcopy(value))
    for name, values in snapshot.observables.items():
        for key, value in values.items():
            setattr(env.env._observables[name], key, copy.deepcopy(value))
    for key, value in snapshot.robot_buffers.items():
        setattr(env.robots[0], key, copy.deepcopy(value))
    np.random.set_state(snapshot.numpy_rng)
    random.setstate(snapshot.python_rng)


def excitation_actions(
    position: NDArray,
    rotation: NDArray,
    count: int,
    rng: np.random.Generator,
    mapping: ControllerMap,
) -> tuple[NDArray, NDArray]:
    """Materialize actions using ideal poses only, before executing any of them."""
    lower = position + np.array([-0.12, -0.12, -0.05])
    upper = position + np.array([0.12, 0.12, 0.12])
    if mapping.position_limits is not None:
        lower = np.maximum(lower, mapping.position_limits[0])
        upper = np.minimum(upper, mapping.position_limits[1])
    ideal_position, ideal_rotation = position.copy(), rotation.copy()
    filtered = np.zeros(6)
    actions, boundary = np.zeros((count, 7)), np.zeros((count, 3), dtype=bool)
    for offset in range(count):
        if offset % 4 == 0:
            sample = rng.uniform(-1, 1, 6) * np.array([0.2] * 3 + [0.1] * 3)
            filtered = 0.5 * filtered + 0.5 * sample
        action = np.r_[filtered.copy(), -1 if (offset // 32) % 2 == 0 else 1]
        next_position, _ = ideal_pose_step(ideal_position, ideal_rotation, action, mapping)
        reflection = (next_position < lower) | (next_position > upper)
        action[:3][reflection] *= -1
        boundary[offset] = reflection
        ideal_position, ideal_rotation = ideal_pose_step(
            ideal_position, ideal_rotation, action, mapping
        )
        actions[offset] = action
    return actions, boundary


def actual_robot_object_contact(env) -> bool:
    model, data = env.sim.model, env.sim.data
    for index in range(data.ncon):
        contact = data.contact[index]
        names = [model.geom_id2name(int(geom)) or "" for geom in (contact.geom1, contact.geom2)]
        is_robot = [name.startswith(("robot0_", "gripper0_")) for name in names]
        if is_robot[0] == is_robot[1]:
            continue
        other = names[1 if is_robot[0] else 0]
        if any(other.startswith(obj.naming_prefix) for obj in env.env.objects):
            return True
    return False


def contact_geometry(env) -> dict:
    """Resolve collision-box corners and closed-pad midpoint from compiled pinned assets."""
    plate = env.env.objects_dict["plate_1"]
    model, data = env.sim.model, env.sim.data
    corners = []
    for name in plate.contact_geoms:
        index = model.geom_name2id(name)
        if int(model.geom_type[index]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError(f"Unsupported plate collision primitive: {name}")
        local = np.array(list(itertools.product([-1, 1], repeat=3))) * model.geom_size[index]
        world = local @ data.geom_xmat[index].reshape(3, 3).T + data.geom_xpos[index]
        corners.extend(world)
    corners = np.asarray(corners)
    initial_position, initial_rotation = robot_pose(env)
    saved = capture_snapshot(env)
    # Evaluate a kinematic asset configuration, then restore it before physical collection.
    for joint_name in env.robots[0].gripper.joints:
        joint_id = model.joint_name2id(joint_name)
        closed = np.clip(0.0, *model.jnt_range[joint_id])
        data.qpos[model.jnt_qposadr[joint_id]] = closed
    env.sim.forward()
    important = env.robots[0].gripper.important_geoms
    pad_names = important["left_fingerpad"] + important["right_fingerpad"]
    pad_midpoint = np.mean([data.geom_xpos[model.geom_name2id(name)] for name in pad_names], axis=0)
    offset_local = initial_rotation.T @ (pad_midpoint - initial_position)
    restore_snapshot(env, saved)
    low, high = corners.min(axis=0), corners.max(axis=0)
    edge = np.array([low[0], (low[1] + high[1]) / 2, (low[2] + high[2]) / 2])
    support = []
    for index in range(model.ngeom):
        name = model.geom_id2name(index) or ""
        if "table" in name and int(model.geom_type[index]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            extent_z = np.abs(data.geom_xmat[index].reshape(3, 3)[2]) @ model.geom_size[index]
            support.append({"name": name, "top_z_m": float(data.geom_xpos[index, 2] + extent_z)})
    if not support:
        raise ValueError("Cannot resolve table support height from declared collision geometry")
    return {
        "plate_collision_min_m": low.tolist(),
        "plate_collision_max_m": high.tolist(),
        "negative_x_edge_m": edge.tolist(),
        "closed_pad_offset_local_m": offset_local.tolist(),
        "closed_gripper_joint_configuration_m": [0.0, 0.0],
        "finger_pad_geoms": pad_names,
        "table_support": support,
    }


def contact_setup(env, geometry: dict, mapping: ControllerMap) -> dict:
    position, rotation = robot_pose(env)
    edge = np.array(geometry["negative_x_edge_m"])
    offset_world = rotation @ np.array(geometry["closed_pad_offset_local_m"])
    pad_target = edge + np.array([0, 0, 0.01])
    targets = [pad_target + np.array([0, 0, 0.1]) - offset_world, pad_target - offset_world]
    stage, steps = 0, 0
    executed = []
    # 1 mm is a fixed waypoint-arrival tolerance, not an outcome-dependent contact rule.
    for _ in range(80):
        position, _ = robot_pose(env)
        if np.linalg.norm(targets[stage] - position) <= 0.001:
            stage += 1
            if stage == len(targets):
                break
        delta = targets[stage] - position
        normalized = (
            2
            * (delta - mapping.output_min[:3])
            / (np.asarray(mapping.output_max[:3]) - mapping.output_min[:3])
            - 1
        )
        action = np.r_[np.clip(normalized, -0.2, 0.2), 0, 0, 0, 1]
        env.step(action)
        executed.append(action.tolist())
        steps += 1
    return {
        "setup_steps": steps,
        "setup_waypoints_completed": stage,
        "waypoint_arrival_tolerance_m": 0.001,
        "executed_setup_actions": executed,
    }


def save_snapshot(path: Path, snapshot: Snapshot) -> None:
    """Store internal numeric structures as JSON plus arrays, without executable pickle."""
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"ndarray": key}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, tuple):
            return {"tuple": [encode(x) for x in value]}
        if isinstance(value, list):
            return [encode(x) for x in value]
        if isinstance(value, dict):
            return {key: encode(val) for key, val in value.items()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Robosuite ring/delta buffers contain only numeric runtime fields.
        if type(value).__module__ == "robosuite.utils.buffers":
            return {"buffer_class": type(value).__name__, "fields": encode(vars(value))}
        raise TypeError(f"Unresolved full snapshot field: {type(value)}")

    metadata = encode(snapshot.model_dump())
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata)), **arrays)


def load_snapshot(path: Path) -> Snapshot:
    from robosuite.utils.buffers import DeltaBuffer, RingBuffer

    buffer_types = {cls.__name__: cls for cls in (DeltaBuffer, RingBuffer)}
    with np.load(path, allow_pickle=False) as payload:

        def decode(value):
            if isinstance(value, list):
                return [decode(item) for item in value]
            if not isinstance(value, dict):
                return value
            if set(value) == {"ndarray"}:
                return payload[value["ndarray"]].copy()
            if set(value) == {"tuple"}:
                return tuple(decode(item) for item in value["tuple"])
            if set(value) == {"buffer_class", "fields"}:
                cls = buffer_types[value["buffer_class"]]
                result = cls.__new__(cls)
                result.__dict__.update(decode(value["fields"]))
                return result
            return {key: decode(item) for key, item in value.items()}

        return Snapshot.model_validate(decode(json.loads(str(payload["metadata"]))))


class ContactRecorder:
    """Observe every physics substep, including contacts absent at the control endpoint."""

    def __init__(self, env):
        self.env = env
        self.current_pairs: set[tuple[str, str]] = set()
        self.original_step = env.sim.step
        env.sim.step = self._step

    def _step(self):
        result = self.original_step()
        model, data = self.env.sim.model, self.env.sim.data
        for index in range(data.ncon):
            contact = data.contact[index]
            names = tuple(
                model.geom_id2name(int(geom)) or "" for geom in (contact.geom1, contact.geom2)
            )
            is_robot = [name.startswith(("robot0_", "gripper0_")) for name in names]
            if is_robot[0] == is_robot[1]:
                continue
            other = names[1 if is_robot[0] else 0]
            if any(other.startswith(obj.naming_prefix) for obj in self.env.env.objects):
                self.current_pairs.add(tuple(sorted(names)))
        return result

    def step(self, action):
        self.current_pairs.clear()
        result = self.env.step(action)
        return result, sorted(self.current_pairs)
