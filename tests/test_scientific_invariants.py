from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from effect_pretraining.coordinates import (
    ControllerMap,
    NumericalCoordinates,
    effect_discrepancy,
    nominal_effect,
)
from effect_pretraining.effect_scaling import (
    fit_effect_transform,
    initialize_projector,
    reference_indices,
)
from effect_pretraining.integrity import verify_release
from effect_pretraining.native_flow import attention_mask, native_flow_target, physical_action_loss
from effect_pretraining.schedule import common_acquisition_round, learning_rate


@pytest.fixture
def mapping():
    return ControllerMap(
        input_min=(-1,) * 6,
        input_max=(1,) * 6,
        output_min=(-0.05,) * 3 + (-0.5,) * 3,
        output_max=(0.05,) * 3 + (0.5,) * 3,
        position_limits=None,
    )


def test_release_is_unchanged():
    assert len(verify_release(Path(__file__).resolve().parents[1])) == 6


def test_ideal_composition_clips_commands_and_keeps_initial_frame(mapping):
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    actions = np.zeros((2, 7))
    actions[0, [0, 3]] = [2, 0.4]
    actions[1, [1, 4]] = [0.2, 0.6]
    effect = nominal_effect(np.zeros(3), rotation, actions, mapping)
    np.testing.assert_allclose(effect[:3], [0.05, 0.01, 0], atol=1e-15)
    expected = rotation.T @ Rotation.from_rotvec([0, 0.3, 0]).as_matrix()
    expected = expected @ Rotation.from_rotvec([0.2, 0, 0]).as_matrix() @ rotation
    np.testing.assert_allclose(Rotation.from_rotvec(effect[3:]).as_matrix(), expected, atol=1e-14)
    actions[:, 6] = [1, -1]
    np.testing.assert_array_equal(effect, nominal_effect(np.zeros(3), rotation, actions, mapping))


def test_rotation_discrepancy_is_geodesic_not_vector_subtraction():
    k = np.array([[0, 0, 0, np.pi - 0.01, 0, 0]])
    d = np.array([[0, 0, 0, -np.pi + 0.01, 0, 0]])
    assert np.linalg.norm(effect_discrepancy(k, d)[0, 3:]) == pytest.approx(0.02)


def test_physical_transform_round_trip_and_native_gripper_retention():
    coordinates = NumericalCoordinates(
        gripper_absolute_limits=(0.04, 0.04),
        action_min=(-2,) * 7,
        action_max=(2,) * 7,
    )
    state = np.array([0.1, 0.2, 0.3, 1, 2, 3, 0.02, -0.03])
    np.testing.assert_allclose(coordinates.state(coordinates.state(state), inverse=True), state)
    action = np.linspace(-2, 2, 7)
    np.testing.assert_allclose(coordinates.action(coordinates.action(action), inverse=True), action)
    assert coordinates.state(state)[6] == 0.5


def test_effect_statistics_reject_eval_and_use_physical_floors():
    nominal = np.array([[0, 0, 0, 0, 0, 0], [0.004, 0, 0, 0.1, 0, 0]])
    transform = fit_effect_transform(nominal, [(0, 0), (1, 0)])
    np.testing.assert_allclose(transform.mean, nominal.mean(axis=0))
    np.testing.assert_allclose(
        transform.scale, [0.002, 0.001, 0.001, 0.05, np.pi / 180, np.pi / 180]
    )
    measured = nominal + 0.1
    assert transform.apply(measured)[0, 1] == pytest.approx(100)
    with pytest.raises(ValueError, match="TRAIN"):
        fit_effect_transform(nominal, [(0, 0), (90, 0)])
    assert len(np.unique(reference_indices(1000))) == 512
    assert reference_indices(1000)[-1] == 999


def test_initial_projector_rms_and_pairing_preserve_rng():
    torch.manual_seed(17)
    state = torch.randn(32, 24) * 3
    effect = torch.randn(32, 6)
    rng = torch.get_rng_state().clone()
    first, record = initialize_projector(state, effect, 9101)
    assert torch.equal(rng, torch.get_rng_state())
    second, other = initialize_projector(state, effect, 9101)
    assert record == other
    assert torch.equal(first.weight, second.weight)
    assert record["effect_rms_after"] == pytest.approx(record["state_rms"], rel=2e-7)
    assert torch.count_nonzero(first.bias) == 0


def test_masked_effect_has_no_direct_or_state_relay_path():
    padding = torch.tensor([[True, False, True, True]])
    blocks = torch.tensor([[1, 1, 1, 1]])
    mask = attention_mask(padding, blocks)
    assert not mask[..., 1].any()  # No query can read the masked effect key.
    assert not mask[:, 1, :].any()
    active = attention_mask(torch.ones_like(padding), blocks)
    assert not active[0, 0, 1]  # The preceding state cannot relay even an active effect.


def test_native_flow_sign_and_padded_coordinates_do_not_affect_loss():
    actions = torch.ones(2, 16, 32)
    noise = actions * 3
    noisy, velocity = native_flow_target(actions, noise, torch.tensor([0.0, 1.0]))
    assert torch.equal(noisy[0], actions[0])
    assert torch.equal(noisy[1], noise[1])
    assert (velocity == 2).all()
    prediction = velocity.clone()
    prediction[..., 7:] = 10000
    assert physical_action_loss(prediction, velocity) == 0


def test_fixed_schedule_and_common_round():
    assert learning_rate(1) == pytest.approx(1e-6)
    assert learning_rate(100) == pytest.approx(1e-4)
    assert learning_rate(1000) == pytest.approx(2.5e-6)
    assert learning_rate(5000, baseline=True) == pytest.approx(2.5e-6)
    counts = {(arm, seed): 250 for arm in ("P", "K", "D") for seed in (101, 202)}
    counts[("P", 101)] = 500
    assert common_acquisition_round(counts) == 250
    with pytest.raises(ValueError):
        common_acquisition_round({("P", 101): 250})
