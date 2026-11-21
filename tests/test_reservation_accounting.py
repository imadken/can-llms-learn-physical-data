import json

from effect_pretraining.resources import accelerator_job, read_budget


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
