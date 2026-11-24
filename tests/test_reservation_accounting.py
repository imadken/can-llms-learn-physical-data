import json

import pytest

from effect_pretraining.resources import BudgetExpired, accelerator_job, read_budget


def test_nested_phases_count_one_continuous_reservation(tmp_path):
    with accelerator_job(tmp_path, "parent"):
        with accelerator_job(tmp_path, "nested"):
            live = read_budget(tmp_path / "resource_budget.json")
            assert live.accelerator_seconds_used > 0
    jobs = [
        json.loads(line) for line in (tmp_path / "accelerator_jobs.jsonl").read_text().splitlines()
    ]
    phases = [
        json.loads(line)
        for line in (tmp_path / "accelerator_phases.jsonl").read_text().splitlines()
    ]
    assert len(jobs) == len(phases) == 1
    assert phases[0]["included_in_parent_reservation"]
    assert (
        read_budget(tmp_path / "resource_budget.json").accelerator_seconds_used
        == jobs[0]["elapsed_job_seconds"]
    )
    assert not (tmp_path / "active_accelerator_job.json").exists()


def test_new_nested_phase_cannot_start_after_checkpoint_reserve_is_reached(tmp_path):
    with accelerator_job(tmp_path, "parent"):
        path = tmp_path / "resource_budget.json"
        budget = json.loads(path.read_text())
        budget["accelerator_seconds_used"] = budget["accelerator_limit_seconds"] - 170
        path.write_text(json.dumps(budget))
        with pytest.raises(BudgetExpired, match="checkpoint/report reserve"):
            with accelerator_job(tmp_path, "too_late"):
                raise AssertionError("The phase must not start")
