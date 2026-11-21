"""Disposable native-model correctness checks; these are not acquisition fits."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from effect_pretraining.native_flow import NumericalFlow, load_native_policy, pad_native


def parameter_digest(parameters) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check_native(checkpoint: Path, vlm_assets: Path, output: Path) -> dict:
    torch.manual_seed(942026)
    policy = load_native_policy(checkpoint, vlm_assets)
    native = policy.model
    projector = nn.Linear(6, native.state_proj.out_features)
    model = NumericalFlow(native, projector).eval()
    frozen = [(name, p) for name, p in native.named_parameters() if not p.requires_grad]
    frozen_before = parameter_digest(frozen)
    trainable = [(name, p) for name, p in native.named_parameters() if p.requires_grad]
    before = parameter_digest(trainable)
    state = torch.tensor([[0.1, 0.1, 1.1, 0.9, 0.01, 0.02, 0.5, -0.5]])
    effect = torch.tensor([[0.5, -0.3, 0.4, 0.1, 0.2, -0.1]])
    actions = torch.randn(1, 16, 7) * 0.1
    noise = native.sample_noise((1, 16, native.config.max_action_dim), device="cpu")
    time = native.sample_time(1, device="cpu")
    loss = model(state, effect, torch.tensor([False]), actions, noise=noise, time=time)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite native training path")
    loss.backward()
    gradient_norm = projector.weight.grad.norm().item()
    if not gradient_norm > 0:
        raise ValueError("Effect projector gradients do not traverse the frozen decoder")
    if any(parameter.grad is not None for _, parameter in frozen):
        raise ValueError("Frozen parameters received gradients")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    optimizer.step()
    for values in optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            if name in values and values[name].dtype != torch.float32:
                raise ValueError("Adam moment buffers must remain float32")
    frozen_after = parameter_digest(frozen)
    if frozen_before != frozen_after or before == parameter_digest(trainable):
        raise ValueError("Frozen/trainable parameter integrity check failed")
    masked = torch.tensor([True])
    first = model.generate(state, effect, masked, noise)
    changed = model.generate(state, effect * -7 + 10, masked, noise)
    if not torch.equal(first, changed):
        raise ValueError("An attention-masked effect changed generated actions")
    transferred = model.transfer_weights()
    if any("effect" in key or "state_proj" in key or ".vlm." in key for key in transferred):
        raise ValueError("Transfer includes a temporary or frozen module")
    # Remove the temporary wrapper and exercise the ordinary inference entry point.
    del optimizer, model, projector
    native.eval()
    language = torch.tensor([[1]], dtype=torch.int64)
    language_mask = torch.tensor([[True]])
    with torch.no_grad():
        ordinary = native.sample_actions(
            [],
            [],
            language,
            language_mask,
            pad_native(state, native.config.max_state_dim),
            noise=noise.clone(),
        )
    if ordinary.shape != (1, 16, 32) or not torch.isfinite(ordinary).all():
        raise ValueError("Ordinary post-cleanup action inference failed")
    report = {
        "status": "passed",
        "device": "cpu",
        "data": "synthetic_engineering_inputs_only",
        "production_updates": 0,
        "trainable_parameters_and_adam_moments_float32": True,
        "disposable_optimizer_steps": 1,
        "effect_projector_gradient_norm": gradient_norm,
        "frozen_parameter_sha256_before": frozen_before,
        "frozen_parameter_sha256_after": frozen_after,
        "masked_generated_actions_bit_identical": True,
        "ordinary_post_cleanup_inference": "finite_H16_native_shape",
        "ordinary_inference_limit": (
            "image-free language/state smoke; actual image profiling remains required"
        ),
        "action_generation_parameter_count": sum(p.numel() for _, p in trainable),
        "transfer_keys": sorted(transferred),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("audit/native_correctness.json"))
    args = parser.parse_args()
    result = check_native(args.checkpoint, args.vlm_assets, args.output)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "transfer_keys"}, indent=2
        )
    )
