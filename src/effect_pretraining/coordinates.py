"""Physical coordinates and ideal OSC composition; no simulator state access."""

from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial.transform import Rotation

Vector6 = Annotated[tuple[float, ...], Field(min_length=6, max_length=6)]


class ControllerMap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_min: Vector6
    input_max: Vector6
    output_min: Vector6
    output_max: Vector6
    position_limits: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    orientation_limits: None = None
    rotation_frame: str = "world_left_multiply"

    @model_validator(mode="after")
    def validate_ranges(self) -> "ControllerMap":
        if np.any(np.asarray(self.input_max) <= np.asarray(self.input_min)):
            raise ValueError("Controller input bounds must be ordered")
        if self.rotation_frame != "world_left_multiply":
            raise ValueError("This implementation requires source-verified world rotations")
        return self

    def scale(self, command: NDArray) -> NDArray:
        command = np.asarray(command, dtype=np.float64)
        if command.shape[-1] != 6 or not np.isfinite(command).all():
            raise ValueError("Expected six finite OSC commands")
        lo, hi = np.asarray(self.input_min), np.asarray(self.input_max)
        out_lo, out_hi = np.asarray(self.output_min), np.asarray(self.output_max)
        return out_lo + (np.clip(command, lo, hi) - lo) * (out_hi - out_lo) / (hi - lo)


def ideal_pose_step(
    position: NDArray, rotation: NDArray, action: NDArray, mapping: ControllerMap
) -> tuple[NDArray, NDArray]:
    scaled = mapping.scale(np.asarray(action)[:6])
    next_position = np.asarray(position) + scaled[:3]
    if mapping.position_limits is not None:
        next_position = np.clip(next_position, *np.asarray(mapping.position_limits))
    next_rotation = Rotation.from_rotvec(scaled[3:]).as_matrix() @ rotation
    return next_position, next_rotation


def endpoint_effect(
    initial_position: NDArray,
    initial_rotation: NDArray,
    final_position: NDArray,
    final_rotation: NDArray,
) -> NDArray:
    return np.concatenate(
        [
            final_position - initial_position,
            Rotation.from_matrix(initial_rotation.T @ final_rotation).as_rotvec(),
        ]
    )


def nominal_effect(
    initial_position: NDArray,
    initial_rotation: NDArray,
    actions: NDArray,
    mapping: ControllerMap,
) -> NDArray:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("Actions must be a finite H by 7 array, including gripper commands")
    position, rotation = initial_position.copy(), initial_rotation.copy()
    for action in actions:
        position, rotation = ideal_pose_step(position, rotation, action, mapping)
    return endpoint_effect(initial_position, initial_rotation, position, rotation)


def effect_discrepancy(nominal: NDArray, measured: NDArray) -> NDArray:
    """Translation difference and Log(Q_K^T Q_D), with physical angular semantics."""
    nominal, measured = np.asarray(nominal), np.asarray(measured)
    qk = Rotation.from_rotvec(nominal[..., 3:])
    qd = Rotation.from_rotvec(measured[..., 3:])
    return np.concatenate(
        [measured[..., :3] - nominal[..., :3], (qk.inv() * qd).as_rotvec()], axis=-1
    )


class NumericalCoordinates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gripper_absolute_limits: tuple[float, float]
    action_min: Annotated[tuple[float, ...], Field(min_length=7, max_length=7)]
    action_max: Annotated[tuple[float, ...], Field(min_length=7, max_length=7)]

    @model_validator(mode="after")
    def validate_scales(self) -> "NumericalCoordinates":
        if not np.all(np.asarray(self.gripper_absolute_limits) > 0):
            raise ValueError("Gripper physical scales must be positive")
        if not np.all(np.asarray(self.action_max) > np.asarray(self.action_min)):
            raise ValueError("Action range must have positive width")
        return self

    def state(self, raw: NDArray, *, inverse: bool = False) -> NDArray:
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape[-1] != 8 or not np.isfinite(raw).all():
            raise ValueError("Expected eight finite native state coordinates")
        scale = np.array([1, 1, 1, np.pi, np.pi, np.pi, *self.gripper_absolute_limits])
        return raw * scale if inverse else raw / scale

    def action(self, raw: NDArray, *, inverse: bool = False) -> NDArray:
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape[-1] != 7 or not np.isfinite(raw).all():
            raise ValueError("Expected seven finite action coordinates")
        lo, hi = np.asarray(self.action_min), np.asarray(self.action_max)
        return lo + (raw + 1) * (hi - lo) / 2 if inverse else 2 * (raw - lo) / (hi - lo) - 1
