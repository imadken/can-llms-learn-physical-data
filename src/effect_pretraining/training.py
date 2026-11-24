"""Fixed production fits, admitted only from complete measured profiles."""

import gc
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

from effect_pretraining.checks import parameter_digest
from effect_pretraining.integrity import sha256_file, verify_release
from effect_pretraining.native_flow import NumericalFlow, load_native_policy
from effect_pretraining.profiling import optimizer_for
from effect_pretraining.resources import accelerator_job, read_budget
from effect_pretraining.schedule import learning_rate
from effect_pretraining.training_data import paired_update


def load_arrays(artifacts: Path) -> dict:
    with np.load(artifacts / "training_arrays.npz", allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def numerical_model(paths: dict, artifacts: Path, seed: int):
    policy = load_native_policy(Path(paths["checkpoint"]), Path(paths["vlm_assets"]))
    projector = nn.Linear(6, policy.model.state_proj.out_features)
    projector.load_state_dict(
        load_file(str(artifacts / f"initial_projector_seed{seed}.safetensors"))
    )
    return NumericalFlow(policy.model, projector).eval()


def trainable_state(model) -> dict:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value for name, value in model.state_dict().items() if name in names}


def save_checkpoint(path: Path, model, optimizer, update: int, metadata: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "trainable": trainable_state(model),
            "optimizer": optimizer.state_dict(),
            "completed_updates": update,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "completed_updates": update,
    }


def restore_checkpoint(path: Path, model, optimizer, expected_sha256: str) -> int:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = set(trainable_state(model))
    if set(payload["trainable"]) != expected:
        raise ValueError("Checkpoint trainable parameter set changed")
    model.load_state_dict(payload["trainable"], strict=False)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["completed_updates"]


def profiles_and_projection(root: Path, artifacts: Path) -> dict:
    profiles = json.loads((root / "audit/profiling.json").read_text())
    if profiles["status"] != "complete":
        raise ValueError(
            "Both prescribed profiles and development timing must pass before production"
        )
    numerical = profiles["paths"]["numerical"]["mean_update_seconds"]
    multimodal = profiles["paths"]["multimodal"]["mean_update_seconds"]
    # All evaluation cases are bounded at their maximum allowed action counts.
    with np.load(artifacts / "episodes/episode_080.npz", allow_pickle=False) as saved:
        control_step = float(saved["step_wall_seconds"].mean())
    acquisition_evaluation = (
        576 * profiles["numerical_chunk_seconds"] + 448 * 16 * control_step + 180
    )
    sample = profiles["task_rollout_sample"]
    baseline_rollouts = 200 * 300 / 8 * sample["seconds"]
    result = {
        "numerical_update_seconds": numerical,
        "multimodal_update_seconds": multimodal,
        "acquisition_full_fit_seconds": 6000 * numerical,
        "common_250_round_seconds": 1500 * numerical,
        "acquisition_evaluation_seconds": acquisition_evaluation,
        "baseline_fit_seconds": 5000 * multimodal,
        "baseline_evaluation_seconds_at_max_horizon": baseline_rollouts,
        "contingency_fraction": 0.25,
        "checkpoint_report_reserve_seconds": 600,
    }
    (root / "audit/resource_projection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def acquisition_rounds(root: Path, artifacts: Path) -> dict:
    protected = verify_release(root)
    projection = profiles_and_projection(root, artifacts)
    resolved = json.loads((root / "resolved_environment.json").read_text())
    paths = resolved["runtime_paths"]
    arrays_sha = sha256_file(artifacts / "training_arrays.npz")
    if arrays_sha != resolved["projector_calibration"]["training_arrays_sha256"]:
        raise ValueError("Frozen numerical training arrays changed after calibration")
    arrays = load_arrays(artifacts)
    status_path = root / "audit/acquisition_training.json"
    if status_path.exists():
        raise FileExistsError(
            "A production campaign requires explicit recovery, never a silent restart"
        )
    status = {"status": "not_started", "common_completed_updates": 0, "fits": {}, "rounds": []}
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for seed in (101, 202):
        for arm in ("P", "K", "D"):
            status["fits"][f"{arm}_{seed}"] = {"completed_updates": 0, "checkpoints": []}
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    for target in (250, 500, 750, 1000):
        budget = read_budget(artifacts / "resource_budget.json")
        reserve = projection["acquisition_evaluation_seconds"] * 1.25 + 600
        if not budget.permits(datetime.now(UTC), projection["common_250_round_seconds"], reserve):
            status["status"] = "partial" if status["common_completed_updates"] else "deferred"
            status["reason"] = (
                "measured common round plus evaluation/checkpoint reserve exceeds remaining cap"
            )
            break
        with accelerator_job(artifacts, f"acquisition_common_round_{target}"):
            for seed in (101, 202):
                for arm in ("P", "K", "D"):
                    key = f"{arm}_{seed}"
                    fit_status = status["fits"][key]
                    model = numerical_model(paths, artifacts, seed).cuda()
                    torch.cuda.reset_peak_memory_stats()
                    optimizer = optimizer_for(model)
                    initial_update = 0
                    if fit_status["checkpoints"]:
                        previous = fit_status["checkpoints"][-1]
                        initial_update = restore_checkpoint(
                            Path(previous["path"]), model, optimizer, previous["sha256"]
                        )
                    frozen = [
                        (name, p) for name, p in model.named_parameters() if not p.requires_grad
                    ]
                    frozen_before = parameter_digest(frozen)
                    if initial_update == 0:
                        initial_sha = parameter_digest(
                            [(name, p) for name, p in model.named_parameters() if p.requires_grad]
                        )
                        for other in ("P", "K", "D"):
                            other_sha = status["fits"][f"{other}_{seed}"].get(
                                "initialization_sha256"
                            )
                            if other_sha is not None and initial_sha != other_sha:
                                raise ValueError(
                                    "Paired native/projector initialization differs across arms"
                                )
                        fit_status["initialization_sha256"] = initial_sha
                        status_path.write_text(json.dumps(status, indent=2) + "\n")
                    meta = {
                        "arm": arm,
                        "seed": seed,
                        "code_commit": code_sha,
                        "protected_hashes": protected,
                        "effective_batch": 128,
                        "microbatch": 16,
                        "accumulation": 8,
                        "projector_stream": 800000 + seed,
                        "training_arrays_sha256": arrays_sha,
                        "initialization_sha256": fit_status["initialization_sha256"],
                        "projector_file_sha256": resolved["projector_calibration"]["seeds"][
                            str(seed)
                        ]["file_sha256"],
                        "evaluation_bank_sha256": sha256_file(
                            root / "audit/acquisition_bank_manifest.json"
                        ),
                        "pristine_checkpoint_revision": "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6",
                    }
                    log_path = root / "audit" / f"acquisition_{key}.jsonl"
                    for update in range(initial_update + 1, target + 1):
                        started = time.monotonic()
                        optimizer.zero_grad(set_to_none=True)
                        rate = learning_rate(update)
                        for group in optimizer.param_groups:
                            group["lr"] = rate
                        batch = paired_update(arrays, model.native, seed, update)
                        effects = (
                            torch.zeros_like(batch.nominal_z)
                            if arm == "P"
                            else batch.nominal_z
                            if arm == "K"
                            else batch.measured_z
                        )
                        loss_value = 0.0
                        for start in range(0, 128, 16):
                            stop = start + 16
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                loss = model(
                                    batch.states[start:stop].cuda(),
                                    effects[start:stop].cuda(),
                                    batch.masked[start:stop].cuda(),
                                    batch.actions[start:stop].cuda(),
                                    noise=batch.noise[start:stop].cuda(),
                                    time=batch.time[start:stop].cuda(),
                                )
                            if not torch.isfinite(loss):
                                raise FloatingPointError(
                                    f"Non-finite production loss {key}/{update}"
                                )
                            (loss / 8).backward()
                            loss_value += loss.detach().item() / 8
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                        if not torch.isfinite(norm):
                            raise FloatingPointError(
                                f"Non-finite production gradients {key}/{update}"
                            )
                        optimizer.step()
                        torch.cuda.synchronize()
                        stream_digest = hashlib.sha256()
                        for tensor in (batch.actions, batch.noise, batch.time, batch.masked):
                            stream_digest.update(tensor.numpy().tobytes())
                        record = {
                            "update": update,
                            "loss": loss_value,
                            "learning_rate": rate,
                            "gradient_norm_before_clip": float(norm),
                            "seconds": time.monotonic() - started,
                            "window_indices_sha256": hashlib.sha256(
                                batch.indices.tobytes()
                            ).hexdigest(),
                            "action_noise_time_mask_sha256": stream_digest.hexdigest(),
                        }
                        with log_path.open("a") as stream:
                            stream.write(json.dumps(record) + "\n")
                        if update % 25 == 0:
                            print(json.dumps({"fit": key, **record}), flush=True)
                    frozen_after = parameter_digest(frozen)
                    if frozen_before != frozen_after:
                        raise ValueError(f"Frozen parameters changed in {key}")
                    meta.update(
                        frozen_sha256_before=frozen_before,
                        frozen_sha256_after=frozen_after,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    )
                    checkpoint = save_checkpoint(
                        artifacts / "checkpoints" / f"{key}_u{target:04d}.pt",
                        model,
                        optimizer,
                        target,
                        meta,
                    )
                    fit_status["completed_updates"] = target
                    fit_status["checkpoints"].append(checkpoint)
                    status_path.write_text(json.dumps(status, indent=2) + "\n")
                    del optimizer, model, batch, effects, frozen
                    gc.collect()
                    torch.cuda.empty_cache()
            status["common_completed_updates"] = target
            status["rounds"].append(target)
            status["status"] = "complete" if target == 1000 else "partial"
            status_path.write_text(json.dumps(status, indent=2) + "\n")
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    acquisition_rounds(Path.cwd(), args.artifacts.resolve())
