"""Tests for forecasting_metrics.py.

This suite checks:
- `compute_oos_r2` matches the standard out-of-sample R² definition that uses
  the sample mean as the baseline.
- `evaluate_forecasting_accuracy` returns a sensible summary table and a
  per-date Spearman rank-correlation DataFrame.
- Input validation behaves as expected (missing columns, bad index, etc.).
- `diebold_mariano_test` produces the expected loss differential and DM statistic
  in a small, fully deterministic example.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasting_metrics import (
    compute_oos_r2,
    diebold_mariano_test,
    evaluate_forecasting_accuracy,
)


@pytest.fixture
def sample_panel():
    """Small (ticker, date) panel with two model forecasts and realized returns."""
    index = pd.MultiIndex.from_product(
        [["AAA", "BBB"], pd.to_datetime(["2020-01-01", "2020-02-01"])],
        names=["ticker", "date"],
    )
    return pd.DataFrame(
        {
            "model_a": [0.09, 0.22, -0.12, 0.04],
            "model_b": [0.20, 0.00, -0.05, -0.30],
            "realized": [0.10, 0.20, -0.10, 0.05],
        },
        index=index,
    )


def test_compute_oos_r2_simple_case():
    # Simple sanity check against a manual R² computation.
    y_true = [1.0, 2.0, 3.0]
    y_pred = [0.8, 2.2, 3.1]
    r2 = compute_oos_r2(y_true, y_pred)

    # R² = 1 - SSE/SST where baseline is mean(y_true).
    errors = np.array(y_true) - np.array(y_pred)
    sse = np.sum(errors**2)
    baseline = np.mean(y_true)
    sst = np.sum((np.array(y_true) - baseline) ** 2)
    expected = 1 - sse / sst

    assert np.isclose(r2, expected)


def test_evaluate_forecasting_accuracy_summary(sample_panel):
    # Evaluate both models and request "realized" as the outcome column.
    summary, spearman = evaluate_forecasting_accuracy(sample_panel, realized_col="realized")

    # Summary table should have one row per model and the expected columns.
    assert set(summary.index) == {"model_a", "model_b"}
    assert set(summary.columns) == {"oos_r2", "mae", "mse", "spearman_rank_corr"}

    # ---- Manual metric reconstruction for model_a ----
    # (Aligned is already clean in this synthetic test, but we follow the same logic.)
    aligned_a = sample_panel[["model_a", "realized"]]

    # Errors are prediction minus realized.
    errors_a = aligned_a["model_a"] - aligned_a["realized"]
    expected_mae_a = errors_a.abs().mean()
    expected_mse_a = (errors_a**2).mean()

    # Cross-sectional OOS R² baseline uses the per-date mean realized return.
    sse_a = ((aligned_a["realized"] - aligned_a["model_a"]) ** 2).sum()
    ybar_t = aligned_a["realized"].groupby(level="date").transform("mean")
    sst_a = ((aligned_a["realized"] - ybar_t) ** 2).sum()
    expected_r2_a = 1 - sse_a / sst_a if sst_a != 0 else np.nan

    assert np.isclose(summary.loc["model_a", "mae"], expected_mae_a)
    assert np.isclose(summary.loc["model_a", "mse"], expected_mse_a)
    assert np.isclose(summary.loc["model_a", "oos_r2"], expected_r2_a)

    # ---- Spearman rank correlation (per date), then average across dates ----
    daily_corrs = []
    for _, group in aligned_a.groupby(level="date"):
        # Spearman can be computed by correlating percentile ranks.
        pred_ranks = group["model_a"].rank(pct=True)
        realized_ranks = group["realized"].rank(pct=True)
        daily_corrs.append(pred_ranks.corr(realized_ranks))
    expected_spearman = np.mean(daily_corrs)

    assert np.isclose(summary.loc["model_a", "spearman_rank_corr"], expected_spearman)

    # ---- Per-date Spearman output shape ----
    # We should get a DataFrame indexed by date with one column per model.
    assert spearman.index.is_monotonic_increasing
    assert list(spearman.columns) == sorted(summary.index)


def test_evaluate_forecasting_accuracy_subset_models(sample_panel):
    # Restrict the evaluation to a subset of model columns.
    summary, spearman = evaluate_forecasting_accuracy(
        sample_panel, realized_col="realized", model_cols=["model_b"]
    )

    assert list(summary.index) == ["model_b"]
    assert list(spearman.columns) == ["model_b"]


def test_evaluate_forecasting_accuracy_missing_realized(sample_panel):
    # Missing the realized column should raise a KeyError.
    with pytest.raises(KeyError):
        evaluate_forecasting_accuracy(sample_panel.drop(columns=["realized"]))


def test_evaluate_forecasting_accuracy_invalid_index(sample_panel):
    # The API requires a ('ticker', 'date') MultiIndex.
    bad_index_df = sample_panel.reset_index()
    with pytest.raises(ValueError):
        evaluate_forecasting_accuracy(bad_index_df, realized_col="realized")


def test_diebold_mariano_test(sample_panel):
    # DM test compares per-date mean forecast losses between model pairs.
    results = diebold_mariano_test(sample_panel, realized_col="realized")

    # Output index should identify the model pair.
    assert list(results.index.names) == ["model_1", "model_2"]
    assert ("model_a", "model_b") in results.index

    # ---- Manual DM inputs for the same panel ----
    # Default loss is squared error.
    se_a = (sample_panel["model_a"] - sample_panel["realized"]) ** 2
    se_b = (sample_panel["model_b"] - sample_panel["realized"]) ** 2

    # Loss differential is averaged across tickers within each date.
    diff_by_date = (se_a - se_b).groupby(level="date").mean().sort_index()

    # For horizon=1, Newey-West lag = 0, so var(mean) is just gamma0 / n.
    mean_diff = diff_by_date.mean()
    demeaned = diff_by_date - mean_diff
    gamma0 = (demeaned @ demeaned) / len(diff_by_date)
    var_mean = gamma0 / len(diff_by_date)
    expected_dm = mean_diff / np.sqrt(var_mean)

    assert np.isclose(results.loc[("model_a", "model_b"), "mean_loss_diff"], mean_diff)
    assert np.isclose(results.loc[("model_a", "model_b"), "dm_stat"], expected_dm)
    assert results.loc[("model_a", "model_b"), "periods"] == 2