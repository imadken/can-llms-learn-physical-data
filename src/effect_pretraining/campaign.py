"""Keep one CUDA context alive across the already specified bounded campaign."""

import json
import time
from pathlib import Path

import torch

from effect_pretraining.acquisition import generate, replay_and_score
from effect_pretraining.baseline import evaluate_baseline, fit_baseline
from effect_pretraining.integrity import verify_release
from effect_pretraining.resources import accelerator_job
from effect_pretraining.training import acquisition_rounds


def run(root: Path, artifacts: Path, recover_pre_update_fault: bool = False) -> None:
    verify_release(root)
    status_path = root / "audit/campaign_status.json"
    status = {"status": "running", "stage": "context_initialization", "context_attempts": []}
    with accelerator_job(artifacts, "persistent_bounded_stage1_campaign"):
        try:
            # These are context-initialization retries, with no model or optimizer updates.
            for attempt in range(3):
                try:
                    torch.cuda.init()
                    value = torch.ones(1, device="cuda")
                    torch.cuda.synchronize()
                    del value
                    status["context_attempts"].append({"attempt": attempt + 1, "status": "ready"})
                    break
                except torch.AcceleratorError as error:
                    status["context_attempts"].append({"attempt": attempt + 1, "error": str(error)})
                    if attempt == 2:
                        raise
                    time.sleep(1)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            prior_path = root / "audit/acquisition_training.json"
            if prior_path.exists():
                prior = json.loads(prior_path.read_text())
                if prior["common_completed_updates"] == 1000:
                    acquisition = prior
                elif (
                    recover_pre_update_fault
                    and all(fit["completed_updates"] == 0 for fit in prior["fits"].values())
                    and not any((root / "audit").glob("acquisition_*_*.jsonl"))
                ):
                    history = root / "audit/acquisition_pre_update_fault.json"
                    if history.exists():
                        raise FileExistsError("A prior pre-update recovery requires explicit audit")
                    prior["status"] = "blocked_before_update_0"
                    history.write_text(json.dumps(prior, indent=2) + "\n")
                    prior_path.unlink()
                    acquisition = acquisition_rounds(root, artifacts)
                else:
                    raise RuntimeError(
                        "Existing partial acquisition needs an audited checkpoint recovery"
                    )
            else:
                acquisition = acquisition_rounds(root, artifacts)
            status["stage"] = "acquisition_evaluation"
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            if acquisition["common_completed_updates"]:
                if not (root / "audit/acquisition_generation.json").exists():
                    generate(root, artifacts)
                if not (root / "audit/acquisition_evidence.json").exists():
                    # CUDA stays reserved while the materialized chunks are replayed on CPU.
                    with accelerator_job(artifacts, "fixed_acquisition_complete_chunk_replays"):
                        replay_and_score(root, artifacts)
            status["stage"] = "S_baseline"
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            baseline_path = root / "audit/baseline_training.json"
            baseline = (
                json.loads(baseline_path.read_text())
                if baseline_path.exists()
                else fit_baseline(root, artifacts)
            )
            if baseline["status"] in ("complete", "partial"):
                if not (root / "audit/baseline_summary.json").exists():
                    evaluate_baseline(root, artifacts)
            status.update(status="complete_bounded_program", stage="finished")
        except BaseException as error:
            status.update(status="interrupted_or_fault", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--recover-pre-update-fault", action="store_true")
    args = parser.parse_args()
    run(Path.cwd(), args.artifacts.resolve(), args.recover_pre_update_fault)
