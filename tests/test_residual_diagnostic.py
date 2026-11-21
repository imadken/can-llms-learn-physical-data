import numpy as np

from effect_pretraining.residuals import fit_ridge, ridge_prediction, summarize_ridge


def test_fixed_ridge_objective_has_unpenalized_intercept_and_correct_penalty():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(31, 4))
    features[:, 3] = 5  # Constant features must remain well-defined.
    targets = rng.normal(size=(31, 6)) + np.arange(6)
    fit = fit_ridge(features, targets)
    prediction = ridge_prediction(fit, features)
    standardized = (features - fit["feature_mean"]) / fit["feature_scale"]
    error = prediction - targets
    gradient = 2 * standardized.T @ error / targets.size + 0.002 * fit["coefficients"]
    np.testing.assert_allclose(gradient, 0, atol=1e-14)
    np.testing.assert_allclose(error.mean(axis=0), 0, atol=1e-14)
    assert np.isfinite(prediction).all()


def test_empty_contact_stratum_is_undefined_without_nan_or_invented_estimate():
    summary = summarize_ridge(np.empty((0, 6)), np.empty((0, 6)), np.zeros(6))
    for channel in summary.values():
        assert channel["count"] == 0
        assert channel["squared_error_reduction"] is None
        assert channel["ridge_rmse"] is None
