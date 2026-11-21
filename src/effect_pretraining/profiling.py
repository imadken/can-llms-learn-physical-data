"""The two prescribed disposable 5+20-update profiles, with fixed effective batches."""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

from effect_pretraining.coordinates import ControllerMap, NumericalCoordinates, nominal_effect
from effect_pretraining.deployment import (
    generate_ordinary,
    observation_batch,
    official_initial_states,
)
from effect_pretraining.effect_scaling import EffectTransform
from effect_pretraining.integrity import verify_release
from effect_pretraining.native_flow import NumericalFlow, load_native_policy, pad_native
from effect_pretraining.resources import accelerator_job
from effect_pretraining.training_data import DemonstrationSampler, paired_update


def optimizer_for(model):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )


def prepare_baseline(policy):
    for parameter in policy.model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in policy.model.named_parameters():
        if name.startswith(("vlm_with_expert.lm_expert.", "state_proj.", "action_")):
            parameter.requires_grad_(True)
    # Two native LIBERO cameras; no invented image from the checkpoint's unused third slot.
    policy.config.input_features = {
        name: feature
        for name, feature in policy.config.input_features.items()
        if name != "observation.images.camera3"
    }
    policy.model.eval()
    return policy


def multimodal_loss(policy, batch: dict, noise: torch.Tensor, time: torch.Tensor):
    images, image_masks = policy.prepare_images(batch)
    language, language_masks = policy.prepare_language(batch)
    losses = policy.model(
        images,
        image_masks,
        language,
        language_masks,
        pad_native(batch["observation.state"], 32),
        pad_native(batch["action"], 32),
        noise=noise,
        time=time,
    )
    return losses[..., :7].mean()


def profile(root: Path, artifacts: Path) -> None:
    verify_release(root)
    result_path = root / "audit/profiling.json"
    if result_path.exists():
        raise FileExistsError("Do not silently repeat prescribed profiles")
    resolved = json.loads((root / "resolved_environment.json").read_text())
    paths = resolved["runtime_paths"]
    report = {
        "status": "started",
        "precision": "bf16_autocast_float32_parameters_and_optimizer",
        "paths": {},
        "production_updates": 0,
    }
    interface = json.loads((artifacts / "numerical_interface.json").read_text())
    coordinates = NumericalCoordinates.model_validate(interface["coordinates"])
    with np.load(artifacts / "training_arrays.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    try:
        with accelerator_job(artifacts, "required_disposable_profiles"):
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["capability"] = torch.cuda.get_device_capability(0)
            report["torch_architectures"] = torch.cuda.get_arch_list()
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("The locked BF16 precision is unavailable")
            for path_name, effective_batch, microbatch in [
                ("numerical", 128, 16),
                ("multimodal", 32, 1),
            ]:
                policy = load_native_policy(Path(paths["checkpoint"]), Path(paths["vlm_assets"]))
                if path_name == "numerical":
                    projector = nn.Linear(6, policy.model.state_proj.out_features)
                    projector.load_state_dict(
                        load_file(str(artifacts / "initial_projector_seed101.safetensors"))
                    )
                    model = NumericalFlow(policy.model, projector).cuda().eval()
                    sampler = None
                else:
                    policy = prepare_baseline(policy).cuda()
                    model = policy
                    sampler = DemonstrationSampler(
                        root / "audit/demonstration_split_manifest.json", coordinates
                    )
                optimizer = optimizer_for(model)
                measured, all_updates = [], []
                torch.cuda.reset_peak_memory_stats()
                for update in range(25):
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    if path_name == "numerical":
                        batch = paired_update(arrays, policy.model, 90101, update)
                    else:
                        batch = sampler.sample(90303, update, effective_batch)
                        with torch.random.fork_rng(devices=[]):
                            torch.manual_seed(90303000 + update)
                            noise = policy.model.sample_noise((effective_batch, 16, 32), "cpu")
                            flow_time = policy.model.sample_time(effective_batch, "cpu")
                    loss_value = 0.0
                    for start in range(0, effective_batch, microbatch):
                        stop = start + microbatch
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            if path_name == "numerical":
                                loss = model(
                                    batch.states[start:stop].cuda(),
                                    batch.nominal_z[start:stop].cuda(),
                                    batch.masked[start:stop].cuda(),
                                    batch.actions[start:stop].cuda(),
                                    noise=batch.noise[start:stop].cuda(),
                                    time=batch.time[start:stop].cuda(),
                                )
                            else:
                                micro = {
                                    key: value[start:stop].cuda()
                                    if isinstance(value, torch.Tensor)
                                    else value[start:stop]
                                    for key, value in batch.items()
                                    if key != "window_ids"
                                }
                                loss = multimodal_loss(
                                    policy,
                                    micro,
                                    noise[start:stop].cuda(),
                                    flow_time[start:stop].cuda(),
                                )
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"Non-finite {path_name} profile loss at update {update}"
                            )
                        (loss * (microbatch / effective_batch)).backward()
                        loss_value += loss.detach().float().item() * microbatch / effective_batch
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f"Non-finite {path_name} profile gradient")
                    optimizer.step()
                    torch.cuda.synchronize()
                    seconds = time.monotonic() - started
                    all_updates.append(
                        {
                            "update": update,
                            "phase": "warmup" if update < 5 else "measured",
                            "seconds": seconds,
                            "loss": loss_value,
                        }
                    )
                    if update >= 5:
                        measured.append(seconds)
                    print(json.dumps({"path": path_name, **all_updates[-1]}), flush=True)
                report["paths"][path_name] = {
                    "effective_batch": effective_batch,
                    "microbatch": microbatch,
                    "accumulation": effective_batch // microbatch,
                    "warmup_updates": 5,
                    "measured_updates": 20,
                    "mean_update_seconds": float(np.mean(measured)),
                    "max_update_seconds": float(np.max(measured)),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "updates": all_updates,
                }
                if path_name == "numerical":
                    with np.load(
                        artifacts / "episodes/episode_080.npz", allow_pickle=False
                    ) as development:
                        state = torch.tensor(
                            coordinates.state(development["states"][0])[None],
                            dtype=torch.float32,
                            device="cuda",
                        )
                        mapping = ControllerMap.model_validate(interface["controller_map"])
                        nominal = nominal_effect(
                            development["positions"][0],
                            development["rotations"][0],
                            development["actions"][:16],
                            mapping,
                        )
                    transform = EffectTransform.model_validate(
                        {
                            key: value
                            for key, value in resolved["effect_transform"].items()
                            if key in EffectTransform.model_fields
                        }
                    )
                    effect = torch.tensor(
                        transform.apply(nominal)[None], dtype=torch.float32, device="cuda"
                    )
                    generation_noise = batch.noise[:1].cuda()
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        generated = model.generate(
                            state, effect, torch.tensor([False], device="cuda"), generation_noise
                        )
                    torch.cuda.synchronize()
                    if not torch.isfinite(generated).all():
                        raise FloatingPointError("Non-finite BF16 numerical chunk generation")
                    report["numerical_chunk_seconds"] = time.monotonic() - started
                    report["numerical_chunk_development_source"] = {"episode_id": 80, "offset": 0}
                else:
                    from libero.libero import benchmark, get_libero_path
                    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

                    suite = benchmark.get_benchmark_dict()["libero_goal"]()
                    task = suite.tasks[0]
                    bddl = (
                        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                    )
                    env = OffScreenRenderEnv(
                        bddl_file_name=str(bddl), camera_heights=128, camera_widths=128
                    )
                    try:
                        env.reset()
                        entries = official_initial_states(suite, 0)
                        raw = env.set_init_state(entries[20])
                        for _ in range(10):
                            raw, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                        torch.cuda.synchronize()
                        started = time.monotonic()
                        generation_times = []
                        for _replan in range(2):
                            generation_start = time.monotonic()
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                chunk = generate_ordinary(
                                    policy, observation_batch(raw, coordinates, task.language)
                                )
                            torch.cuda.synchronize()
                            generation_times.append(time.monotonic() - generation_start)
                            commands = chunk[0, :4].float().cpu().numpy()
                            if not np.isfinite(commands).all():
                                raise FloatingPointError("Non-finite multimodal profile generation")
                            for action in commands:
                                raw, _, _, _ = env.step(
                                    np.clip(
                                        coordinates.action(action, inverse=True),
                                        coordinates.action_min,
                                        coordinates.action_max,
                                    )
                                )
                        report["task_rollout_sample"] = {
                            "task_id": 0,
                            "official_initial_state_index": 20,
                            "control_actions": 8,
                            "replans": 2,
                            "complete_task_rollout": False,
                            "seconds": time.monotonic() - started,
                            "mean_replan_generation_seconds": float(np.mean(generation_times)),
                            "outcomes_scored": False,
                        }
                    finally:
                        env.close()
                if sampler is not None:
                    sampler.close()
                del optimizer, model, policy, batch
                if path_name == "numerical":
                    del projector, state, effect, generation_noise, generated
                gc.collect()
                torch.cuda.empty_cache()
            report["status"] = "complete"
    except BaseException as error:
        report["status"] = "fault"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    profile(Path.cwd(), args.artifacts.resolve())
