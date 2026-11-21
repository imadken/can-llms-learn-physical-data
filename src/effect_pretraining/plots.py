"""Recompute the exploratory residual figure and descriptive episode summaries."""

import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from effect_pretraining.residuals import write_csv


def build_residual_figure(root: Path, artifacts: Path) -> dict:
    audit = root / "audit"
    with (audit / "residual_replays.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    unique = [row for row in rows if row["repeat"] == "0"]
    ridge = json.loads((audit / "ridge_diagnostic.json").read_text())
    episodes = json.loads((artifacts / "episode_records.json").read_text())
    duplicates = list(csv.DictReader((audit / "duplicate_variability.csv").open()))
    episode_rows = []
    for episode in sorted({int(row["episode_id"]) for row in unique}):
        selected = [row for row in unique if int(row["episode_id"]) == episode]
        episode_rows.append(
            {
                "episode_id": episode,
                "unique_reference_chunks": len(selected),
                "intended_reset": selected[0]["intended_reset"],
                "mean_translation_residual_mm": float(
                    np.mean([float(x["translation_residual_mm"]) for x in selected])
                ),
                "mean_rotation_residual_degrees": float(
                    np.mean([float(x["rotation_residual_degrees"]) for x in selected])
                ),
            }
        )
    write_csv(audit / "residual_episode_summary.csv", episode_rows)
    summary = {
        "status": "complete",
        "episode_attempts": len(episodes),
        "retained_transitions": sum(x["retained_transitions"] for x in episodes),
        "warmup_control_steps": sum(x["warmup_steps"] for x in episodes),
        "contact_setup_control_steps": sum(x["setup_steps"] for x in episodes),
        "invalid_collection_attempts": sum(x["status"] != "complete" for x in episodes),
        "actual_robot_object_contact_steps": sum(x["actual_contact_steps"] for x in episodes),
        "reference_snapshots": len({(x["episode_id"], x["snapshot_offset"]) for x in rows}),
        "reference_replays": len(rows),
        "unique_reference_chunks": len(unique),
        "evaluation_source_episodes": len(episode_rows),
        "neural_training_seeds": 0,
        "ridge_predictors": 1,
        "duplicate_integration_max_abs_difference": max(
            float(x["integration_max_abs_difference"]) for x in duplicates
        ),
        "contact_stratum": "empty; contact-conditional residual estimates unavailable",
        "uncertainty": (
            "95% t intervals over the ten episode means, with nine degrees of freedom; "
            "descriptive fixed-scene sampling uncertainty; "
            "training variability and transfer uncertainty remain unmeasured"
        ),
    }
    for channel, key, episode_key in [
        ("translation", "translation_residual_mm", "mean_translation_residual_mm"),
        ("rotation", "rotation_residual_degrees", "mean_rotation_residual_degrees"),
    ]:
        values = np.array([float(row[key]) for row in unique])
        per_episode = np.array([row[episode_key] for row in episode_rows])
        margin = (
            t.ppf(0.975, len(per_episode) - 1) * per_episode.std(ddof=1) / np.sqrt(len(per_episode))
        )
        summary[channel] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.1)),
            "p90": float(np.quantile(values, 0.9)),
            "episode_mean_95pct_t_interval": [
                float(per_episode.mean() - margin),
                float(per_episode.mean() + margin),
            ],
            "units": "mm" if channel == "translation" else "degrees",
        }
    summary["interpretation_limits"] = [
        "No actual contact was achieved by the fixed collector; no contact contrast is estimable.",
        (
            "Residuals include ideal-map/controller tracking differences; "
            "the ridge does not identify their cause."
        ),
        (
            "A residual observation does not establish acquisition, downstream utility, "
            "equivalence or an RSS decision."
        ),
    ]
    (audit / "residual_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    translation_axis, structure_axis = axes
    rotation_axis = translation_axis.twinx()
    source_rows = []
    categories = ["native_free_motion_attempt", "contact_attempt"]
    for channel, key, axis, color, shift in [
        ("translation", "translation_residual_mm", translation_axis, "#146b9c", -0.08),
        ("rotation", "rotation_residual_degrees", rotation_axis, "#d26726", 0.08),
    ]:
        for index, category in enumerate(categories):
            values = np.array(
                [float(row[key]) for row in unique if row["intended_reset"] == category]
            )
            low, median, high = np.quantile(values, [0.1, 0.5, 0.9])
            axis.errorbar(
                index + shift,
                median,
                yerr=[[median - low], [high - median]],
                fmt="o",
                capsize=5,
                color=color,
                label=channel if index == 0 else None,
            )
            source_rows.append(
                {
                    "panel": "residual",
                    "channel": channel,
                    "category": category,
                    "p10": low,
                    "median": median,
                    "p90": high,
                    "normalized_sse": "",
                }
            )
    translation_axis.set_xticks([0, 1], ["Native reset", "Contact attempt"])
    translation_axis.set_ylabel("Translation residual (mm)", color="#146b9c")
    rotation_axis.set_ylabel("Rotation residual (degrees)", color="#d26726")
    translation_axis.set_title("Residuals: median and 10–90% range")
    translation_axis.set_xlim(-0.4, 1.4)
    translation_axis.text(
        0.5,
        0.03,
        "0 observed robot–object contacts",
        ha="center",
        transform=translation_axis.transAxes,
        fontsize=9,
    )
    for index, (label, item) in enumerate(ridge["overall"].items()):
        baseline_sse = item["training_mean_rmse"] ** 2
        bars = [1.0, item["zero_rmse"] ** 2 / baseline_sse, item["ridge_rmse"] ** 2 / baseline_sse]
        for offset, value, color, name in zip(
            [-0.22, 0, 0.22],
            bars,
            ["#a5adb6", "#636c76", "#16817a"],
            ["TRAIN mean", "Zero residual", "Fixed ridge"],
            strict=True,
        ):
            structure_axis.bar(
                index + offset, value, width=0.2, color=color, label=name if index == 0 else None
            )
            source_rows.append(
                {
                    "panel": "structure",
                    "channel": label,
                    "category": name,
                    "p10": "",
                    "median": "",
                    "p90": "",
                    "normalized_sse": value,
                }
            )
    structure_axis.set_xticks([0, 1], ["Translation", "Rotation"])
    structure_axis.set_ylabel("Held-out SSE / TRAIN-mean SSE")
    structure_axis.set_title("One ridge fit; no hyperparameter search")
    structure_axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Exploratory fixed plate scene · 10 evaluation episodes · 200 unique chunks", fontsize=11
    )
    figure.savefig(audit / "residual_structure.png", dpi=180)
    figure.savefig(audit / "residual_structure.pdf")
    plt.close(figure)
    write_csv(audit / "residual_plot_source.csv", source_rows)
    print(json.dumps(summary, indent=2))
    return summary


def build_acquisition_figure(root: Path) -> None:
    audit = root / "audit"
    evidence = json.loads((audit / "acquisition_evidence.json").read_text())
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    source = []
    modes = ["matched", "masked", "P", "alternate"]
    for axis, arm, title in [
        (axes[0], "K", "Native K: nominal endpoints"),
        (axes[1], "D", "Native D: measured endpoints"),
    ]:
        for seed, color, marker in [(101, "#146b9c", "o"), (202, "#d26726", "s")]:
            groups = evidence[arm]["seeds"][str(seed)]["groups"]
            values = [groups[mode]["joint_attainment_percent"] for mode in modes]
            axis.plot(range(4), values, marker=marker, color=color, label=f"Seed {seed}")
            for mode in modes:
                source.append(
                    {
                        "panel": f"native_{arm}",
                        "seed": seed,
                        "arm": arm,
                        "mode": mode,
                        **groups[mode],
                    }
                )
        axis.set_xticks(range(4), ["Matched", "Masked", "P", "Alternate"], rotation=25)
        axis.set_title(title)
    for seed, color, marker in [(101, "#146b9c", "o"), (202, "#d26726", "s")]:
        groups = evidence["common_physical_goal"][str(seed)]
        axes[2].plot(
            range(3),
            [groups[arm]["joint_attainment_percent"] for arm in ("P", "K", "D")],
            marker=marker,
            color=color,
            label=f"Seed {seed}",
        )
        for arm in ("P", "K", "D"):
            source.append(
                {
                    "panel": "common_physical",
                    "seed": seed,
                    "arm": arm,
                    "mode": "matched" if arm != "P" else "P",
                    **groups[arm],
                }
            )
    axes[2].set_xticks(range(3), ["P", "K", "D"])
    axes[2].set_title("Common measured physical goals")
    for axis in axes:
        axis.set_ylim(-2, 102)
        axis.set_ylabel("Joint attainment (%)")
        axis.grid(axis="y", alpha=0.2)
    axes[2].legend(frameon=False)
    figure.suptitle(
        f"Exploratory acquisition · common checkpoint u{evidence['checkpoint_updates']} "
        "· 10 mm and 5°",
        fontsize=12,
    )
    figure.supxlabel(
        "Each seed shown separately; 64 targets from 32 snapshots / 4 episodes; "
        "no seed-significance claim",
        fontsize=9,
    )
    figure.savefig(audit / "acquisition.png", dpi=180)
    figure.savefig(audit / "acquisition.pdf")
    plt.close(figure)
    write_csv(audit / "acquisition_plot_source.csv", source)


def build_baseline_figure(root: Path) -> None:
    audit = root / "audit"
    summary = json.loads((audit / "baseline_summary.json").read_text())
    tasks = summary["tasks"]
    figure, axis = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    values, lower, upper = [], [], []
    for task in tasks:
        value = task["success_rate_valid"]
        interval = task["wilson95_valid_episodes"]
        values.append(100 * value if value is not None else np.nan)
        lower.append(100 * (value - interval[0]) if interval else 0)
        upper.append(100 * (interval[1] - value) if interval else 0)
    axis.barh(range(len(tasks)), values, xerr=[lower, upper], capsize=3, color="#16817a")
    axis.set_yticks(
        range(len(tasks)),
        [textwrap.fill(task["task_name"].replace("_", " "), 44) for task in tasks],
    )
    axis.invert_yaxis()
    axis.set_xlim(0, 105)
    axis.set_xlabel("Success (%) · descriptive 95% Wilson episode intervals")
    axis.set_title(f"S exploratory baseline · seed 303 · {summary['checkpoint_updates']} updates")
    for index, task in enumerate(tasks):
        axis.text(
            101, index, f"{task['successes']}/{task['valid_episodes']}", va="center", fontsize=9
        )
    figure.supxlabel(
        "One trained seed; intervals do not estimate training variability or a benchmark ceiling",
        fontsize=9,
    )
    figure.savefig(audit / "baseline.png", dpi=180)
    figure.savefig(audit / "baseline.pdf")
    plt.close(figure)
    rows = []
    for task in tasks:
        interval = task["wilson95_valid_episodes"] or [None, None]
        rows.append(
            {key: value for key, value in task.items() if key != "wilson95_valid_episodes"}
            | {"wilson95_low": interval[0], "wilson95_high": interval[1]}
        )
    write_csv(audit / "baseline_plot_source.csv", rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--component", choices=["residual", "neural", "all"], default="residual")
    args = parser.parse_args()
    if args.component in ("residual", "all"):
        build_residual_figure(Path.cwd(), args.artifacts.resolve())
    if args.component in ("neural", "all"):
        if Path("audit/acquisition_evidence.json").exists():
            build_acquisition_figure(Path.cwd())
        if Path("audit/baseline_summary.json").exists():
            build_baseline_figure(Path.cwd())
