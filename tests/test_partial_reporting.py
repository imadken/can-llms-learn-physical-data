import csv
import json

from effect_pretraining.reporting import partial_baseline_summary


def test_partial_evaluation_preserves_invalid_and_unperformed_outcomes(tmp_path):
    tasks = [{"task_id": index, "task_name": f"task_{index}"} for index in range(10)]
    (tmp_path / "demonstration_split_manifest.json").write_text(json.dumps(tasks))
    rows = [
        {"task_id": 0, "success": True, "invalid_reason": ""},
        {"task_id": 0, "success": False, "invalid_reason": ""},
        {"task_id": 1, "success": "", "invalid_reason": "BudgetExpired"},
    ]
    with (tmp_path / "baseline_episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = partial_baseline_summary(tmp_path, {"completed_updates": 5000, "training_seed": 303})
    assert result["status"] == "partial"
    assert result["total_attempts"] == 3
    assert result["valid_episodes"] == 2
    assert result["invalid_episodes"] == 1
    assert result["unperformed_episodes"] == 197
    assert result["successes"] == 1
    assert result["tasks"][0]["success_rate_valid"] == 0.5
    assert result["tasks"][1]["successes"] is None
    assert result["tasks"][2]["success_rate_valid"] is None
    assert result["tasks"][2]["wilson95_valid_episodes"] is None
    assert result["equal_task_mean_valid"] is None


def test_no_evaluation_file_does_not_create_outcomes(tmp_path):
    assert partial_baseline_summary(tmp_path, {"status": "deferred"}) == {}
