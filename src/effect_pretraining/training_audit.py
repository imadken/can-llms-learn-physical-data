"""Audit actual paired streams and checkpoint metadata without additional fits."""

import json
from pathlib import Path

import torch

from effect_pretraining.effect_scaling import tensor_hash
from effect_pretraining.integrity import verify_release


def audit(root: Path, *, require_final: bool = False) -> dict:
    verify_release(root)
    status = json.loads((root / "audit/acquisition_training.json").read_text())
    resolved = json.loads((root / "resolved_environment.json").read_text())
    result = {
        "common_checkpoint_updates": status["common_completed_updates"],
        "seeds": {},
        "checkpoints": [],
    }
    for seed in (101, 202):
        logs = {}
        initializations = []
        for arm in ("P", "K", "D"):
            key = f"{arm}_{seed}"
            path = root / "audit" / f"acquisition_{key}.jsonl"
            rows = (
                [json.loads(line) for line in path.read_text().splitlines()]
                if path.exists()
                else []
            )
            if [row["update"] for row in rows] != list(range(1, len(rows) + 1)) or len(rows) > 1000:
                raise ValueError(f"Missing, repeated or excess updates: {key}")
            if require_final and len(rows) != 1000:
                raise ValueError("The final audit requires exactly 1000 updates in all six fits")
            logs[arm] = rows
            fit = status["fits"][key]
            if fit.get("initialization_sha256"):
                initializations.append(fit["initialization_sha256"])
            for checkpoint in fit["checkpoints"]:
                # Mmap exposes dtype/shape/metadata without reading full optimizer storages.
                payload = torch.load(
                    checkpoint["path"], map_location="cpu", weights_only=True, mmap=True
                )
                meta = payload["metadata"]
                if meta["frozen_sha256_before"] != meta["frozen_sha256_after"]:
                    raise ValueError(f"Frozen parameters changed: {key}")
                if any(value.dtype != torch.float32 for value in payload["trainable"].values()):
                    raise ValueError("A trainable master parameter is not float32")
                for values in payload["optimizer"]["state"].values():
                    if (
                        values["exp_avg"].dtype != torch.float32
                        or values["exp_avg_sq"].dtype != torch.float32
                    ):
                        raise ValueError("Adam moment precision changed")
                current = tensor_hash(payload["trainable"]["effect_projector.weight"])
                initial = resolved["projector_calibration"]["seeds"][str(seed)]["weight_sha256"]
                result["checkpoints"].append(
                    {
                        "fit": key,
                        "updates": checkpoint["completed_updates"],
                        "path": checkpoint["path"],
                        "code_commit": meta["code_commit"],
                        "peak_allocated_bytes": meta["peak_allocated_bytes"],
                        "frozen_sha256": meta["frozen_sha256_after"],
                        "float32_trainable_masters_and_adam_moments": True,
                        "effect_projector_weight_changed": current != initial,
                    }
                )
                del payload
        if len(set(initializations)) > 1:
            raise ValueError("Paired initialization hashes differ")
        common = min(len(rows) for rows in logs.values())
        for index in range(common):
            for name in ("window_indices_sha256", "action_noise_time_mask_sha256"):
                if len({logs[arm][index][name] for arm in ("P", "K", "D")}) != 1:
                    raise ValueError(f"Paired {name} differs at seed {seed}, update {index + 1}")
        result["seeds"][str(seed)] = {
            "paired_updates_checked": common,
            "initialization_sha256": initializations[0] if initializations else None,
            "logged_update_counts": {arm: len(rows) for arm, rows in logs.items()},
        }
    result["status"] = "passed_final" if require_final else "passed_available_prefix"
    (root / "audit/actual_training_invariants.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key != "checkpoints"}, indent=2)
    )
    return result


def audit_baseline(root: Path, *, require_final: bool = False) -> dict:
    """Inspect the one S fit without generating actions or updating weights."""
    status = json.loads((root / "audit/baseline_training.json").read_text())
    log = root / "audit/baseline_loss.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    if [row["update"] for row in rows] != list(range(1, len(rows) + 1)) or len(rows) > 5000:
        raise ValueError("Missing, repeated or excess S updates")
    if require_final and (len(rows) != 5000 or status["completed_updates"] != 5000):
        raise ValueError("The final S audit requires exactly 5000 completed updates")
    result = {"logged_update_count": len(rows), "training_status": status["status"]}
    if "checkpoint" in status:
        payload = torch.load(
            status["checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True
        )
        if payload["completed_updates"] != status["completed_updates"]:
            raise ValueError("S checkpoint update count differs from the training log")
        meta = payload["metadata"]
        if meta["frozen_sha256_before"] != meta["frozen_sha256_after"]:
            raise ValueError("Frozen VLM parameters changed during S")
        if any(value.dtype != torch.float32 for value in payload["trainable"].values()):
            raise ValueError("An S trainable master parameter is not float32")
        for values in payload["optimizer"]["state"].values():
            if any(values[key].dtype != torch.float32 for key in ("exp_avg", "exp_avg_sq")):
                raise ValueError("S Adam moment precision changed")
        result.update(
            completed_updates=payload["completed_updates"],
            code_commit=meta["code_commit"],
            peak_allocated_bytes=meta["peak_allocated_bytes"],
            frozen_sha256=meta["frozen_sha256_after"],
            float32_trainable_masters_and_adam_moments=True,
            metadata_status_note=(
                "The metadata copy retains fit-start status; the checkpoint's top-level "
                "completed_updates and baseline_training.json record actual completion."
            ),
        )
        del payload
    elif require_final:
        raise ValueError("A completed S checkpoint is required")
    result["status"] = "passed_final" if require_final else "passed_available_prefix"
    (root / "audit/actual_baseline_invariants.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    (audit_baseline if args.baseline else audit)(Path.cwd(), require_final=args.require_final)
