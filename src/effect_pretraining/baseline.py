"""The single admitted S fit and fixed ten-task exploratory evaluation."""

import gc
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from effect_pretraining.checks import parameter_digest
from effect_pretraining.coordinates import NumericalCoordinates
from effect_pretraining.deployment import (
    generate_ordinary,
    observation_batch,
    official_initial_states,
)
from effect_pretraining.integrity import sha256_file, verify_release
from effect_pretraining.native_flow import load_native_policy
from effect_pretraining.profiling import multimodal_loss, optimizer_for, prepare_baseline
from effect_pretraining.residuals import write_csv
from effect_pretraining.resources import BudgetExpired, accelerator_job, read_budget
from effect_pretraining.schedule import learning_rate
from effect_pretraining.simulation import capture_snapshot, save_snapshot
from effect_pretraining.training import profiles_and_projection, save_checkpoint
from effect_pretraining.training_data import DemonstrationSampler


def fit_baseline(root: Path, artifacts: Path) -> dict:
    protected = verify_release(root)
    result_path = root / "audit/baseline_training.json"
    if result_path.exists():
        raise FileExistsError("The single S fit must not be silently repeated")
    projection = profiles_and_projection(root, artifacts)
    budget = read_budget(artifacts / "resource_budget.json")
    required = (
        projection["baseline_fit_seconds"]
        + projection["baseline_evaluation_seconds_at_max_horizon"]
    )
    if not budget.permits(datetime.now(UTC), required, 600):
        status = {
            "status": "deferred",
            "completed_updates": 0,
            "training_seed": None,
            "reason": (
                "fixed schedule and maximum-horizon evaluation with "
                "25% contingency exceed remaining cap"
            ),
        }
        result_path.write_text(json.dumps(status, indent=2) + "\n")
        return status
    resolved = json.loads((root / "resolved_environment.json").read_text())
    interface = json.loads((artifacts / "numerical_interface.json").read_text())
    coordinates = NumericalCoordinates.model_validate(interface["coordinates"])
    microbatch = json.loads((root / "audit/profiling.json").read_text())["paths"]["multimodal"][
        "microbatch"
    ]
    paths = resolved["runtime_paths"]
    status = {
        "status": "running",
        "completed_updates": 0,
        "training_seed": 303,
        "effective_batch": 32,
        "microbatch": microbatch,
        "accumulation": 32 // microbatch,
    }
    result_path.write_text(json.dumps(status, indent=2) + "\n")
    metadata = {
        "arm": "S",
        "seed": 303,
        "protected_hashes": protected,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "demonstration_split_sha256": sha256_file(root / "audit/demonstration_split_manifest.json"),
        "schedule_updates": 5000,
        "warmup_updates": 500,
        **status,
    }
    with accelerator_job(artifacts, "single_S_multitask_fit"):
        policy = prepare_baseline(
            load_native_policy(Path(paths["checkpoint"]), Path(paths["vlm_assets"]))
        ).cuda()
        optimizer = optimizer_for(policy)
        frozen = [(name, p) for name, p in policy.named_parameters() if not p.requires_grad]
        frozen_before = parameter_digest(frozen)
        sampler = DemonstrationSampler(
            root / "audit/demonstration_split_manifest.json", coordinates
        )
        torch.cuda.reset_peak_memory_stats()
        try:
            for update in range(1, 5001):
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                rate = learning_rate(update, baseline=True)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                batch = sampler.sample(303, update, 32)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(
                        int(np.random.SeedSequence([303, 32000, update]).generate_state(1)[0])
                    )
                    noise = policy.model.sample_noise((32, 16, 32), "cpu")
                    flow_time = policy.model.sample_time(32, "cpu")
                loss_value = 0.0
                for start in range(0, 32, microbatch):
                    stop = start + microbatch
                    micro = {
                        key: value[start:stop].cuda()
                        if isinstance(value, torch.Tensor)
                        else value[start:stop]
                        for key, value in batch.items()
                        if key != "window_ids"
                    }
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = multimodal_loss(
                            policy, micro, noise[start:stop].cuda(), flow_time[start:stop].cuda()
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite S loss at {update}")
                    (loss * (microbatch / 32)).backward()
                    loss_value += loss.detach().item() * microbatch / 32
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite S gradient")
                optimizer.step()
                torch.cuda.synchronize()
                status["completed_updates"] = update
                record = {
                    "update": update,
                    "loss": loss_value,
                    "learning_rate": rate,
                    "seconds": time.monotonic() - started,
                    "gradient_norm_before_clip": float(norm),
                    "window_ids": batch["window_ids"],
                }
                with (root / "audit/baseline_loss.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                if update % 25 == 0:
                    result_path.write_text(json.dumps(status, indent=2) + "\n")
                    print(
                        json.dumps(
                            {key: value for key, value in record.items() if key != "window_ids"}
                        ),
                        flush=True,
                    )
            status["status"] = "complete"
        except BudgetExpired:
            status["status"] = "partial"
            status["reason"] = "hard resource deadline"
        except BaseException as error:
            status["status"] = "fault"
            status["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            sampler.close()
            frozen_after = parameter_digest(frozen)
            if frozen_before != frozen_after:
                raise ValueError("Frozen VLM parameters changed during S")
            metadata.update(
                frozen_sha256_before=frozen_before,
                frozen_sha256_after=frozen_after,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            if status["completed_updates"]:
                status["checkpoint"] = save_checkpoint(
                    artifacts / "checkpoints" / f"S_303_u{status['completed_updates']:04d}.pt",
                    policy,
                    optimizer,
                    status["completed_updates"],
                    metadata,
                )
            result_path.write_text(json.dumps(status, indent=2) + "\n")
        del policy, optimizer, frozen
        gc.collect()
        torch.cuda.empty_cache()
    return status


def wilson_interval(successes: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [float(max(0, center - margin)), float(min(1, center + margin))]


def evaluate_baseline(root: Path, artifacts: Path) -> dict:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    training = json.loads((root / "audit/baseline_training.json").read_text())
    if training["status"] == "deferred":
        return training
    if (root / "audit/baseline_summary.json").exists():
        raise FileExistsError("Do not repeat the fixed S evaluation bank")
    checkpoint = training["checkpoint"]
    if sha256_file(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("S checkpoint hash mismatch")
    resolved = json.loads((root / "resolved_environment.json").read_text())
    paths = resolved["runtime_paths"]
    interface = json.loads((artifacts / "numerical_interface.json").read_text())
    coordinates = NumericalCoordinates.model_validate(interface["coordinates"])
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    rows = []
    output_dir = artifacts / "S_rollouts"
    output_dir.mkdir(exist_ok=True)
    started = time.monotonic()
    with accelerator_job(artifacts, "fixed_S_200_task_rollouts"):
        policy = (
            prepare_baseline(
                load_native_policy(Path(paths["checkpoint"]), Path(paths["vlm_assets"]))
            )
            .cuda()
            .eval()
        )
        payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
        policy.load_state_dict(payload["trainable"], strict=False)
        del payload
        for task_id, task in enumerate(suite.tasks):
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl), camera_heights=128, camera_widths=128, hard_reset=True
            )
            try:
                entries = official_initial_states(suite, task_id)
                for entry_id in range(20):
                    rollout_start = time.monotonic()
                    actions, traces, success, invalid, replans = [], [], False, None, 0
                    clipped = 0
                    budget_expired = False
                    fatal_error = None
                    try:
                        np.random.seed(83026 + 100 * task_id + entry_id)
                        env.env.reset()
                        raw = env.set_init_state(entries[entry_id])
                        for _ in range(10):
                            raw, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                        save_snapshot(
                            output_dir / f"task{task_id:02d}_entry{entry_id:02d}_initial.npz",
                            capture_snapshot(env),
                        )
                        while len(actions) < 300 and not success:
                            stream = int(
                                np.random.SeedSequence(
                                    [303, 82026, task_id, entry_id, replans]
                                ).generate_state(1)[0]
                            )
                            generator = torch.Generator().manual_seed(stream)
                            noise = torch.randn(1, 16, 32, generator=generator).cuda()
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                chunk = generate_ordinary(
                                    policy,
                                    observation_batch(raw, coordinates, task.language),
                                    noise=noise,
                                )
                            commands = coordinates.action(
                                chunk[0, :4].float().cpu().numpy(), inverse=True
                            )
                            if not np.isfinite(commands).all():
                                raise FloatingPointError("Non-finite S generated actions")
                            executed = np.clip(
                                commands, coordinates.action_min, coordinates.action_max
                            )
                            clipped += int((commands != executed).sum())
                            for action in executed:
                                raw, _, _, _ = env.step(action)
                                actions.append(action)
                                traces.append(capture_snapshot(env).integration_state)
                                if not np.isfinite(traces[-1]).all():
                                    raise FloatingPointError("Non-finite S simulation state")
                                success = bool(env.check_success())
                                if success or len(actions) == 300:
                                    break
                            replans += 1
                    except BudgetExpired:
                        invalid = "BudgetExpired: hard resource deadline"
                        budget_expired = True
                    except Exception as error:
                        invalid = f"{type(error).__name__}: {error}"
                        from robosuite.utils.errors import RandomizationError

                        if not isinstance(error, RandomizationError):
                            fatal_error = error
                    np.savez_compressed(
                        output_dir / f"task{task_id:02d}_entry{entry_id:02d}.npz",
                        actions=np.array(actions),
                        integration_states=np.array(traces),
                    )
                    row = {
                        "task_id": task_id,
                        "task_name": task.name,
                        "initial_state_index": entry_id,
                        "success": success if invalid is None else None,
                        "invalid_reason": invalid,
                        "control_actions": len(actions),
                        "replans": replans,
                        "seconds": time.monotonic() - rollout_start,
                        "clipped_action_coordinates": clipped,
                    }
                    rows.append(row)
                    write_csv(root / "audit/baseline_episodes.csv", rows)
                    print(json.dumps(row), flush=True)
                    if budget_expired:
                        raise BudgetExpired(
                            "S evaluation stopped at the hard cap; partial episode retained"
                        )
                    if fatal_error is not None:
                        raise fatal_error
            finally:
                env.close()
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    tasks = []
    for task_id, task in enumerate(suite.tasks):
        selected = [row for row in rows if row["task_id"] == task_id]
        valid = [row for row in selected if row["invalid_reason"] is None]
        successes = sum(bool(row["success"]) for row in valid)
        tasks.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "attempts": len(selected),
                "valid_episodes": len(valid),
                "invalid_episodes": len(selected) - len(valid),
                "successes": successes,
                "success_rate_valid": successes / len(valid) if valid else None,
                "wilson95_valid_episodes": wilson_interval(successes, len(valid)),
            }
        )
    valid = [row for row in rows if row["invalid_reason"] is None]
    successes = sum(bool(row["success"]) for row in valid)
    summary = {
        "status": "complete" if len(rows) == 200 else "partial",
        "checkpoint_updates": training["completed_updates"],
        "training_seed": 303,
        "independent_training_seeds": 1,
        "tasks": tasks,
        "total_attempts": len(rows),
        "valid_episodes": len(valid),
        "successes": successes,
        "equal_task_mean_valid": (
            float(np.mean([item["success_rate_valid"] for item in tasks]))
            if all(item["valid_episodes"] for item in tasks)
            else None
        ),
        "pooled_wilson95_valid_episodes": wilson_interval(successes, len(valid)),
        "wall_seconds": time.monotonic() - started,
        "uncertainty_limit": (
            "Descriptive episode intervals conditional on one trained seed; "
            "no training-seed uncertainty or benchmark-ceiling claim"
        ),
    }
    (root / "audit/baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
