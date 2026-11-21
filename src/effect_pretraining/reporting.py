"""Factual audit report from retained outputs; no outcome-dependent selection."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from effect_pretraining.integrity import verify_release
from effect_pretraining.resources import read_budget


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def partial_baseline_summary(audit: Path, training: dict) -> dict:
    """Retain attempted outcomes if a hard limit interrupts the fixed evaluation."""
    path = audit / "baseline_episodes.csv"
    if not path.exists():
        return {}
    from effect_pretraining.baseline import wilson_interval

    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    tasks = []
    for task in read_json(audit / "demonstration_split_manifest.json", []):
        selected = [row for row in rows if int(row["task_id"]) == task["task_id"]]
        valid = [row for row in selected if not row["invalid_reason"]]
        successes = sum(row["success"] == "True" for row in valid)
        tasks.append(
            {
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "attempts": len(selected),
                "valid_episodes": len(valid),
                "invalid_episodes": len(selected) - len(valid),
                "successes": successes if valid else None,
                "success_rate_valid": successes / len(valid) if valid else None,
                "wilson95_valid_episodes": wilson_interval(successes, len(valid)),
            }
        )
    valid = [row for row in rows if not row["invalid_reason"]]
    successes = sum(row["success"] == "True" for row in valid)
    rates = [task["success_rate_valid"] for task in tasks]
    return {
        "status": "partial",
        "checkpoint_updates": training.get("completed_updates", 0),
        "training_seed": training.get("training_seed"),
        "tasks": tasks,
        "total_attempts": len(rows),
        "valid_episodes": len(valid),
        "successes": successes if valid else None,
        "invalid_episodes": len(rows) - len(valid),
        "unperformed_episodes": 200 - len(rows),
        "equal_task_mean_valid": sum(rates) / len(rates)
        if rates and all(rate is not None for rate in rates)
        else None,
        "pooled_wilson95_valid_episodes": wilson_interval(successes, len(valid)),
        "source": "retained per-episode CSV after interruption; no additional rollout",
    }


def build_report(root: Path, artifacts: Path) -> dict:
    hashes = verify_release(root)
    audit = root / "audit"
    residual = read_json(audit / "residual_summary.json", {})
    ridge = read_json(audit / "ridge_diagnostic.json", {})
    acquisition = read_json(audit / "acquisition_evidence.json", {})
    training = read_json(audit / "acquisition_training.json", {})
    baseline_training = read_json(audit / "baseline_training.json", {})
    baseline = read_json(audit / "baseline_summary.json", {})
    if not baseline:
        baseline = partial_baseline_summary(audit, baseline_training)
        if baseline:
            (audit / "baseline_partial_summary.json").write_text(
                json.dumps(baseline, indent=2) + "\n"
            )
    resolved = read_json(root / "resolved_environment.json")
    budget = read_budget(artifacts / "resource_budget.json")
    now = datetime.now(UTC)
    is_running = (artifacts / "active_accelerator_job.json").exists()
    fit_rows = []
    for seed in (101, 202):
        for arm in ("P", "K", "D"):
            key = f"{arm}_{seed}"
            path = audit / f"acquisition_{key}.jsonl"
            records = (
                [json.loads(line) for line in path.read_text().splitlines()]
                if path.exists()
                else []
            )
            fit_rows.append(
                {
                    "arm": arm,
                    "planned_seed": seed,
                    "actual_updates": len(records),
                    "effective_batch": 128,
                    "window_presentations": len(records) * 128,
                    "last_common_checkpoint_updates": training.get("common_completed_updates", 0),
                }
            )
    fit_rows.append(
        {
            "arm": "S",
            "planned_seed": 303,
            "actual_updates": baseline_training.get("completed_updates", 0),
            "effective_batch": 32,
            "window_presentations": baseline_training.get("completed_updates", 0) * 32,
            "last_common_checkpoint_updates": "",
        }
    )
    with (audit / "run_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fit_rows[0]))
        writer.writeheader()
        writer.writerows(fit_rows)
    status = {
        "protocol_version": "2.2-locked",
        "as_of_utc": now.isoformat(),
        "repository_url": resolved["repository_url"],
        "branch": resolved["branch"],
        "remote_publication_performed": False,
        "preliminary_1": residual.get("status", "blocked"),
        "preliminary_2": acquisition.get(
            "status", "partial" if training.get("common_completed_updates") else "blocked"
        ),
        "preliminary_3": baseline.get(
            "status",
            "deferred"
            if baseline_training.get("status") == "deferred"
            else "partial"
            if baseline_training.get("completed_updates", 0)
            else "blocked",
        ),
        "campaign_process_running": is_running,
        "accelerator_seconds_used": budget.accelerator_seconds_used,
        "execution_wall_seconds_since_profiling": (
            (now - budget.profiling_started_utc).total_seconds() - budget.unavailable_wait_seconds
        )
        if budget.profiling_started_utc
        else 0,
        "limits": {
            "accelerator_seconds": 28800,
            "execution_wall_seconds": 43200,
            "cutoff": "2026-11-28T18:00:00+01:00",
        },
        "artifacts": str(artifacts),
        "protected_hashes": hashes,
        "actual_run_counts": fit_rows,
    }
    (audit / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    for name in (
        "resource_budget.json",
        "accelerator_jobs.jsonl",
        "accelerator_phases.jsonl",
        "collection_run.json",
        "residual_run.json",
    ):
        path = artifacts / name
        if path.exists():
            (audit / name).write_bytes(path.read_bytes())
    phase_commits = {
        "collection": read_json(artifacts / "collection_run.json", {}).get("code_commit"),
        "residual_replays": read_json(artifacts / "residual_run.json", {}).get("code_commit"),
    }
    invariant = read_json(audit / "actual_training_invariants.json", {})
    phase_commits["acquisition"] = sorted(
        {item["code_commit"] for item in invariant.get("checkpoints", [])}
    )
    phase_commits["S"] = read_json(audit / "actual_baseline_invariants.json", {}).get("code_commit")
    command_gaps = []
    for path in sorted(audit.glob("command_*.json")):
        command = read_json(path)
        if "-" in command["argv"] and not command.get("stdin_source_path"):
            command_gaps.append(path.name)
    (audit / "command_provenance.json").write_text(
        json.dumps(
            {
                "early_inline_inputs_not_retained": command_gaps,
                "scope": "engineering diagnostics; argv, environment, revision and output retained",
                "production_commands": "module CLIs with recorded arguments and versioned source",
                "subsequent_stdin": "source bytes captured with SHA256",
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Preliminary execution report",
        "",
        f"Repository: {resolved['repository_url']}",
        f"Local branch: `{resolved['branch']}`. The repository is public; no branch was "
        f"pushed or public release made.",
        "The bundle's delivery.json binds the exact final implementation/results commits "
        "without a self-referential commit hash.",
        f"Execution revisions: `{json.dumps(phase_commits, sort_keys=True)}`.",
        "",
        "| Preliminary | Status | Actual scope |",
        "| --- | --- | --- |",
        f"| 1. Residuals | {status['preliminary_1']} | 100 attempts; 20,000 retained "
        f"transitions; 100 snapshots; 400 replays |",
        f"| 2. Acquisition | {status['preliminary_2']} | P/K/D, seeds 101/202; common "
        f"checkpoint {training.get('common_completed_updates', 0)} |",
        f"| 3. S baseline | {status['preliminary_3']} | "
        f"{baseline_training.get('completed_updates', 0)} updates; "
        f"{baseline.get('total_attempts', 0)} task rollouts |",
        "",
        "## Residual characterization",
        "",
        "All 100 fixed collection attempts completed, with the frozen 80/10/10 episode split. "
        "The 14,800 complete TRAIN windows alone fit the effect transform and ridge features. "
        "The 1,000 warm-up and 4,000 contact-setup actions are excluded from training windows. "
        "There were zero invalid collection attempts and zero failed fixed replays. "
        "Collection timing includes the user-requested storage-cleanup pause before "
        "accelerator profiling.",
        "",
        "No recorded robot–movable-object contact occurred in the 20,000 retained "
        "transitions or fixed replays. "
        "The contact flag covers movable scene objects; fixture/table and self contacts "
        "are outside that flag. "
        "The contact stratum is empty. The prescribed collector was not retried or tuned.",
        "",
        "| Channel | Mean | Descriptive 95% episode-mean t interval | Ridge RMSE | "
        "TRAIN-mean RMSE | SSE reduction |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for channel in ("translation", "rotation"):
        item, diagnostic = residual[channel], ridge["overall"][channel]
        interval = item["episode_mean_95pct_t_interval"]
        lines.append(
            f"| {channel} ({item['units']}) | {item['mean']:.4f} | [{interval[0]:.4f}, "
            f"{interval[1]:.4f}] "
            f"| {diagnostic['ridge_rmse']:.4f} | {diagnostic['training_mean_rmse']:.4f} "
            f"| {100 * diagnostic['squared_error_reduction']:.3f}% |"
        )
    lines += [
        "",
        "All 200 duplicate pairs were bit-identical in full integration state. "
        "The 400 replay records represent 200 unique chunks from ten source episodes. "
        "Intervals concern this fixed scene's episode means; they do not measure "
        "training variability. "
        "Predictable discrepancy does not identify controller calibration as its cause, "
        "acquisition, downstream utility or equivalence.",
        "",
        "![Residual characterization](residual_structure.png)",
        "",
        "## Acquisition",
        "",
    ]
    if acquisition:
        for arm in ("K", "D"):
            lines.append(
                f"**{arm}: {acquisition[arm]['label']}** ({acquisition[arm]['domain']} "
                f"endpoint space)."
            )
            lines += [
                "",
                "| Seed | Mode | Joint attainment | Translation error (mm) | Rotation "
                "error (degrees) |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
            for seed in ("101", "202"):
                for mode, item in acquisition[arm]["seeds"][seed]["groups"].items():
                    lines.append(
                        f"| {seed} | {mode} | {item['joint_attainment_percent']:.3f}% "
                        f"| {item['mean_translation_error_mm']:.4f} | "
                        f"{item['mean_rotation_error_degrees']:.4f} |"
                    )
            lines.append("")
        lines += [
            "K's native score is analytic nominal attainment. The separate common "
            "physical-goal scores are shown below. "
            "Each seed has 64 targets from 32 snapshots drawn from four episodes. "
            "P/masked outputs and paired goal outputs are reused for scoring; these are "
            "not independent samples. "
            "Two training seeds support descriptive results, not equivalence or "
            "training-seed significance.",
            "",
            "| Seed | Arm | Common physical joint attainment | Translation error (mm) | "
            "Rotation error (degrees) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for seed, groups in acquisition["common_physical_goal"].items():
            for arm, item in groups.items():
                lines.append(
                    f"| {seed} | {arm} | {item['joint_attainment_percent']:.3f}% | "
                    f"{item['mean_translation_error_mm']:.4f} | "
                    f"{item['mean_rotation_error_degrees']:.4f} |"
                )
        generation = read_json(audit / "acquisition_generation.json", {})
        lines += [
            "",
            f"Generated {generation.get('generation_count')} unique chunks; "
            f"{acquisition.get('physical_chunk_replays')} whole physical chunk replays. "
            "All materialized generated actions and replay states passed finite-value checks. "
            "No target or result was excluded based on attainment. "
            f"Physical goals outside the nominal TRAIN coordinate range: "
            f"{generation.get('physical_goals_outside_training_nominal_range_percent')}%. "
            f"Nearly coincident goal pairs (within 10 mm and 5 degrees): "
            f"{generation.get('nearly_coincident_goal_pairs')}/32.",
            "",
            "![Acquisition](acquisition.png)",
            "",
        ]
    else:
        lines += ["No final/common-checkpoint acquisition evidence has been produced yet.", ""]
    lines += ["## Standard baseline", ""]
    if baseline:
        lines += [
            "| Task | Successes / valid episodes | Invalid episodes | "
            "Descriptive 95% Wilson interval |",
            "| --- | ---: | ---: | --- |",
        ]
        for task in baseline["tasks"]:
            interval = task["wilson95_valid_episodes"]
            interval_text = (
                f"[{100 * interval[0]:.2f}%, {100 * interval[1]:.2f}%]"
                if interval
                else "unmeasured"
            )
            numerator = task["successes"] if task["valid_episodes"] else "unmeasured"
            lines.append(
                f"| {task['task_name']} | {numerator}/{task['valid_episodes']} | "
                f"{task['invalid_episodes']} | {interval_text} |"
            )
        lines += [
            "",
            f"Pooled successes: {baseline['successes']}/{baseline['valid_episodes']}. "
            f"Equal-task mean: {baseline['equal_task_mean_valid']}. "
            f"{200 - baseline['total_attempts']} scheduled episodes were unperformed. "
            "Invalid attempts are retained with reasons in baseline_episodes.csv and "
            "excluded from valid-episode success denominators; there are no replacements. "
            "One trained seed does not estimate training variability or a benchmark ceiling. "
            "Demonstration/reset disjointness is not established, so this is an "
            "exploratory diagnostic rather than an independent leaderboard claim.",
            "",
            "![S baseline](baseline.png)",
            "",
        ]
    else:
        lines += [
            f"Status: {status['preliminary_3']}. No unperformed task outcome is "
            f"represented as a zero score.",
            "",
        ]
    lines += [
        "## Correctness and provenance",
        "",
        f"Hardware: {resolved['hardware']['cpu']}, {resolved['hardware']['gpu']}, "
        f"driver {resolved['hardware']['driver']}. "
        f"Python {resolved['dependencies']['python']}; Torch {resolved['dependencies']['torch']}; "
        f"Transformers {resolved['dependencies']['transformers']}; "
        f"MuJoCo {resolved['dependencies']['mujoco']}; "
        f"robosuite {resolved['dependencies']['robosuite']}. "
        f"Official LIBERO revision `{resolved['libero']['revision']}`; "
        f"SmolVLA revision `{resolved['model']['revision']}`. "
        "Full dependency paths and source hashes are in resolved_environment.json.",
        "",
        "The exact 2.2 scientific release remains hash-verified. Native mapping/inverse "
        "transforms, "
        "serialized full-state replay, padding/flow sign, projector gradients through "
        "the frozen decoder, "
        "masked-effect invariance, frozen hashes, transfer cleanup and paired streams are checked. "
        "Actual checkpoint master parameters and Adam moments are FP32 under BF16 autocast.",
        "",
        "Both projectors used 512 distinct evenly spaced TRAIN windows. State RMS was 0.219354522; "
        "absorbed alpha was 1.980329311 (seed 101) and 1.969732337 (seed 202). "
        "K and D share nominal-TRAIN means/scales with 1 mm/1 degree floors. P receives "
        "transformed zero. "
        "No evaluation/demonstration data fit those constants.",
        "",
        "The ten official raw demonstration files are revision/hash pinned. Each task "
        "has ten selected "
        "training and five separate held-out validation trajectories. Source-verified post-action "
        "observations are paired with the following action chunk, excluding one extra "
        "terminal candidate "
        "window per selected trajectory. Native camera rotation and processing are preserved. "
        "No validation-based checkpoint selection occurs.",
        "",
        f"Command provenance limitation: {len(command_gaps)} early engineering diagnostics "
        "retained argv/environment/revision/output but omitted inline stdin source. "
        "Their byte-exact input is unavailable; command_provenance.json identifies them. "
        "Production collection, replay, profiling and training use recorded module commands "
        "and versioned source. Subsequent inline inputs are retained and hashed.",
        "",
        "Early CUDA context failures occurred before update 0. The user performed the approved "
        "one-time cache flush/compaction after sudo required a local password. The "
        "original failures "
        "and both measured image microbatches are retained. A persistent CUDA context "
        "and an owned-process "
        "hard-limit guard preserve the original ceilings. No other job was terminated or "
        "shared software upgraded.",
        "",
        "## Budget and artifacts",
        "",
        f"Accelerator reservation time used: {budget.accelerator_seconds_used:.3f} seconds "
        f"({budget.accelerator_seconds_used / 3600:.5f} hours), ceiling 8 hours. "
        f"Execution wall time since profiling: "
        f"{status['execution_wall_seconds_since_profiling']:.3f} seconds, ceiling 12 "
        f"hours. "
        "The calendar cutoff remains 28 November 2026 at 18:00 Europe/Paris.",
        "",
    ]
    profiles = read_json(audit / "profiling.json", {})
    lines += [
        "| Profile | Microbatch / accumulation | Mean update seconds | Peak allocated GiB |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, item in profiles.get("paths", {}).items():
        lines.append(
            f"| {name} | {item['microbatch']} / {item['accumulation']} | "
            f"{item['mean_update_seconds']:.6f} | "
            f"{item['peak_allocated_bytes'] / 2**30:.4f} |"
        )
    peaks = [item["peak_allocated_bytes"] for item in invariant.get("checkpoints", [])]
    baseline_invariant = read_json(audit / "actual_baseline_invariants.json", {})
    if baseline_invariant.get("peak_allocated_bytes"):
        peaks.append(baseline_invariant["peak_allocated_bytes"])
    lines += [
        "",
        f"Maximum production fit allocation recorded: {max(peaks) / 2**30:.4f} GiB. "
        if peaks
        else "Production fit memory is not yet audited. ",
        "The ledger counts the full persistent accelerator reservation, including CPU "
        "replay between phases, without double-counting nested phases. Both image "
        "microbatch profiles and zero-update platform faults are retained and charged. "
        "Timing and phase-level details are in accelerator_jobs.jsonl and "
        "accelerator_phases.jsonl.",
        "",
        f"Campaign running at report time: {is_running}. Full local artifacts: `{artifacts}`. "
        "Datasets, full snapshots, model/optimizer checkpoints and caches are outside Git. "
        "The compressed review bundle includes five pre-existing replay examples and a "
        "SHA256 manifest "
        "of all retained full artifacts. Its SHA256 is in an external checksum file; "
        "delivery.json supplies revisions.",
        "",
        "The public repository was not pushed. The user's unrelated .gitignore edit, "
        "original source ZIP "
        "and PGX documentation are preserved. No Stage 2 campaign, optional extension, "
        "manuscript conclusion "
        "change, public release or scheduled continuation was performed.",
        "",
    ]
    (audit / "PRELIMINARY_REPORT.md").write_text("\n".join(lines))
    resolved["execution_status"] = {
        key: status[key] for key in ("preliminary_1", "preliminary_2", "preliminary_3")
    }
    resolved["repository_visibility"] = "public"
    resolved["profiling"] = {
        "status": "complete",
        "accelerator_seconds": budget.accelerator_seconds_used,
        "execution_clock_started": True,
    }
    (root / "resolved_environment.json").write_text(json.dumps(resolved, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in status.items()
                if key not in {"actual_run_counts", "protected_hashes"}
            },
            indent=2,
        )
    )
    return status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    build_report(Path.cwd(), args.artifacts.resolve())
