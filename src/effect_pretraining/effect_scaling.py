"""The single nominal-TRAIN affine transform and paired projector calibration."""

import hashlib
import json
from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from torch import nn


class EffectTransform(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mean: tuple[float, ...]
    population_std: tuple[float, ...]
    scale: tuple[float, ...]
    training_window_count: int
    training_window_ids_sha256: str
    nominal_effects_sha256: str

    def apply(self, effects: NDArray) -> NDArray:
        return (np.asarray(effects, dtype=np.float64) - np.asarray(self.mean)) / np.asarray(
            self.scale
        )


def fit_effect_transform(
    nominal_effects: NDArray, window_ids: Sequence[tuple[int, int]]
) -> EffectTransform:
    effects = np.asarray(nominal_effects, dtype="<f8")
    if effects.shape != (len(window_ids), 6) or not len(window_ids):
        raise ValueError("Every complete training window needs exactly one six-dimensional effect")
    if not np.isfinite(effects).all():
        raise ValueError("Non-finite nominal effects are a correctness fault")
    if list(window_ids) != sorted(set(window_ids)) or any(
        not 0 <= ep <= 79 for ep, _ in window_ids
    ):
        raise ValueError("Only distinct canonical TRAIN episode/start-offset pairs are allowed")
    mean, std = effects.mean(axis=0), effects.std(axis=0, ddof=0)
    floors = np.array([0.001] * 3 + [np.pi / 180] * 3)
    ids_bytes = json.dumps(list(window_ids), separators=(",", ":")).encode()
    return EffectTransform(
        mean=tuple(mean),
        population_std=tuple(std),
        scale=tuple(np.maximum(std, floors)),
        training_window_count=len(window_ids),
        training_window_ids_sha256=hashlib.sha256(ids_bytes).hexdigest(),
        nominal_effects_sha256=hashlib.sha256(effects.tobytes()).hexdigest(),
    )


def reference_indices(window_count: int) -> NDArray:
    if window_count < 1:
        raise ValueError("Calibration requires complete training windows")
    return np.linspace(0, window_count - 1, min(512, window_count), dtype=np.int64)


def tensor_hash(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    return hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest()


def initialize_projector(
    frozen_state_tokens: torch.Tensor,
    transformed_nominal_effects: torch.Tensor,
    projector_seed: int,
) -> tuple[nn.Linear, dict]:
    if len(frozen_state_tokens) != len(transformed_nominal_effects):
        raise ValueError("Calibration state/effect windows must be paired")
    if not 1 <= len(frozen_state_tokens) <= 512:
        raise ValueError("Use the prescribed training-only reference subset")
    width = frozen_state_tokens.shape[-1]
    generator = torch.Generator(device="cpu").manual_seed(projector_seed)
    # fork_rng avoids changing the native/data/noise streams during module construction.
    with torch.random.fork_rng(devices=[]):
        projector = nn.Linear(6, width, dtype=torch.float32)
    nn.init.xavier_uniform_(projector.weight, gain=1.0, generator=generator)
    nn.init.zeros_(projector.bias)
    with torch.no_grad():
        state = frozen_state_tokens.detach().float().cpu()
        nominal = transformed_nominal_effects.detach().float().cpu()
        effect = projector(nominal)
        state_rms = state.double().square().mean().sqrt().item()
        effect_rms = effect.double().square().mean().sqrt().item()
        if not np.isfinite([state_rms, effect_rms]).all():
            raise ValueError("Non-finite projector calibration is a correctness fault")
        is_degenerate = min(state_rms, effect_rms) < 1e-12
        alpha = 1.0 if is_degenerate else state_rms / effect_rms
        projector.weight.mul_(alpha)
        post_rms = projector(nominal).double().square().mean().sqrt().item()
    return projector, {
        "projector_seed": projector_seed,
        "state_rms": state_rms,
        "effect_rms_before": effect_rms,
        "alpha": alpha,
        "effect_rms_after": post_rms,
        "status": "unresolved_degenerate" if is_degenerate else "calibrated",
        "weight_sha256": tensor_hash(projector.weight),
        "bias_sha256": tensor_hash(projector.bias),
    }
