import numpy as np

from effect_pretraining.acquisition import acquisition_evidence, score_effect


def records_for_seed(seed, is_positive):
    rows = []
    for arm, domain in [("K", "nominal"), ("D", "physical")]:
        for mode in ("matched", "masked", "alternate", "P"):
            for index in range(10):
                rows.append(
                    {
                        "seed": seed,
                        "domain": domain,
                        "arm": "P" if mode == "P" else arm,
                        "mode": mode,
                        "joint_attained": bool(is_positive and mode == "matched" and index == 0),
                        "translation_error_mm": 1 if mode == "matched" else 2,
                        "rotation_error_degrees": 1,
                    }
                )
    return rows


def test_native_evidence_uses_ten_points_both_controls_both_seeds_and_target_following():
    rows = records_for_seed(101, True) + records_for_seed(202, True)
    result = acquisition_evidence(rows)
    assert all(
        result[arm]["label"] == "positive exploratory acquisition evidence" for arm in ("K", "D")
    )
    for row in rows:
        if row["seed"] == 202 and row["mode"] == "matched":
            row["joint_attained"] = False
    result = acquisition_evidence(rows)
    assert all(result[arm]["label"] == "mixed exploratory evidence" for arm in ("K", "D"))
    for row in rows:
        row["translation_error_mm"] = 2
    result = acquisition_evidence(rows)
    assert all(
        result[arm]["label"] == "acquisition not demonstrated within this pilot budget"
        for arm in ("K", "D")
    )


def test_attainment_preserves_separate_physical_tolerances():
    assert score_effect(np.array([0.01, 0, 0, 0, 0, 0]), np.zeros(6))["joint_attained"]
    assert not score_effect(np.array([0.010001, 0, 0, 0, 0, 0]), np.zeros(6))["joint_attained"]
