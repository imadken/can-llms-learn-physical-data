"""Resolve and freeze the required raw demonstrations without observing task scores."""

import concurrent.futures
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import h5py
import numpy as np

from effect_pretraining.integrity import sha256_file


def download_official_goal(root: Path, destination: Path) -> list[dict]:
    source = json.loads((root / "audit/demonstration_source.json").read_text())
    if len(source["files"]) != 10:
        raise ValueError("The complete ten-task raw suite is required")
    required = sum(item["size"] for item in source["files"])
    if shutil.disk_usage(root).free < required + 15_000_000_000:
        raise OSError("Insufficient storage for demonstrations plus resumable experiment artifacts")
    destination.mkdir(parents=True, exist_ok=True)

    def download(item):
        path = destination / Path(item["rfilename"]).name
        expected = item["lfs"]["sha256"]
        if path.exists():
            if sha256_file(path) != expected:
                raise ValueError(f"Existing demonstration hash mismatch: {path}")
        else:
            url = (
                f"https://huggingface.co/datasets/{source['repository']}/resolve/"
                f"{source['revision']}/{item['rfilename']}"
            )
            temporary = path.with_suffix(path.suffix + ".part")
            digest = hashlib.sha256()
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("wb") as stream,
            ):
                while chunk := response.read(4 * 1024 * 1024):
                    digest.update(chunk)
                    stream.write(chunk)
            if temporary.stat().st_size != item["size"] or digest.hexdigest() != expected:
                raise ValueError(f"Downloaded demonstration failed size/hash validation: {path}")
            temporary.replace(path)
        record = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": expected,
        }
        print(json.dumps(record), flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, source["files"]))
    (root / "audit/demonstration_artifacts.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def freeze_trajectories(root: Path, destination: Path) -> list[dict]:
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    records = []
    for task_id, task in enumerate(suite.tasks):
        path = destination / f"{task.name}_demo.hdf5"
        with h5py.File(path, "r") as payload:
            identifiers = sorted(payload["data"].keys())
            if len(identifiers) < 15:
                raise ValueError(
                    f"Required 10 train/5 validation trajectories unavailable: {task.name}"
                )
            rng = np.random.default_rng(np.random.SeedSequence([72026, task_id]))
            ordering = [identifiers[i] for i in rng.permutation(len(identifiers))]
            training, validation = ordering[:10], ordering[10:15]
            selected = []
            for split, names in [("train", training), ("validation", validation)]:
                for name in names:
                    group = payload["data"][name]
                    required = [
                        "actions",
                        "obs/ee_pos",
                        "obs/ee_ori",
                        "obs/gripper_states",
                        "obs/agentview_rgb",
                        "obs/eye_in_hand_rgb",
                    ]
                    for key in required:
                        if key not in group:
                            raise ValueError(
                                f"Missing native demonstration field {task.name}/{name}/{key}"
                            )
                    count = len(group["actions"])
                    if count < 16 or group["actions"].shape[1] != 7:
                        raise ValueError(f"Invalid complete action windows: {task.name}/{name}")
                    if group["obs/ee_pos"].shape != (count, 3):
                        raise ValueError("Invalid EEF demonstration position shape")
                    selected.append(
                        {
                            "trajectory_id": name,
                            "split": split,
                            "action_count": count,
                            "complete_window_count": count - 15,
                        }
                    )
            initial_file = (
                Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            )
            records.append(
                {
                    "task_id": task_id,
                    "task_name": task.name,
                    "language": task.language,
                    "demonstration_path": str(path.resolve()),
                    "demonstration_sha256": sha256_file(path),
                    "available_trajectories": len(identifiers),
                    "selected_trajectories": selected,
                    "initial_state_path": str(initial_file),
                    "initial_state_sha256": sha256_file(initial_file),
                    "evaluation_initial_state_indices": list(range(20)),
                    "rollout_limit_control_steps": 300,
                    "warmup_steps": 10,
                    "provenance_limit": "official exploratory resets; demonstration/reset disjointness not assumed",
                }
            )
    (root / "audit/demonstration_split_manifest.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    download_official_goal(Path.cwd(), args.destination)
    freeze_trajectories(Path.cwd(), args.destination)
