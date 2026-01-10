"""Tests for forecasting_metrics.py.

This suite checks:
- `compute_oos_r2` matches the standard out-of-sample R² definition using the
  sample mean as the baseline forecast.
- `evaluate_forecasting_accuracy` returns a correctly-shaped summary table plus
  a per-date Spearman rank-correlation time series for each model.
- Validation fails loudly when required structure is missing (MultiIndex, realized column).
- `diebold_mariano_test` matches a hand-constructed deterministic example, so we
  know the loss differential aggregation and DM statistic are wired correctly.
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
    """Small (ticker, date) panel with two model forecasts and realized returns.

    We keep the dataset tiny and fully deterministic so tests are transparent and
    failures are easy to debug (no randomness, no large sample effects).
    """
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
    # Use a minimal example where we can compute SSE and SST by hand.
    y_true = [1.0, 2.0, 3.0]
    y_pred = [0.8, 2.2, 3.1]
    r2 = compute_oos_r2(y_true, y_pred)

    # OOS R² uses the unconditional mean of y_true as the baseline forecast.
    errors = np.array(y_true) - np.array(y_pred)
    sse = np.sum(errors**2)
    baseline = np.mean(y_true)
    sst = np.sum((np.array(y_true) - baseline) ** 2)
    expected = 1 - sse / sst

    assert np.isclose(r2, expected)


def test_evaluate_forecasting_accuracy_summary(sample_panel):
    # Run the full evaluation to ensure we get both the scalar summary and the time series output.
    summary, spearman = evaluate_forecasting_accuracy(sample_panel, realized_col="realized")

    # The summary should contain one row per model column and a stable metric schema.
    assert set(summary.index) == {"model_a", "model_b"}
    assert set(summary.columns) == {"oos_r2", "mae", "mse", "spearman_rank_corr"}

    # ---- Manual metric reconstruction for model_a ----
    # We reconstruct the same ingredients so this test detects logic regressions (not just shape issues).
    aligned_a = sample_panel[["model_a", "realized"]]

    # MAE/MSE are computed on pooled asset-level errors.
    errors_a = aligned_a["model_a"] - aligned_a["realized"]
    expected_mae_a = errors_a.abs().mean()
    expected_mse_a = (errors_a**2).mean()

    # Cross-sectional OOS R² uses a per-date mean realized return as the baseline.
    sse_a = ((aligned_a["realized"] - aligned_a["model_a"]) ** 2).sum()
    ybar_t = aligned_a["realized"].groupby(level="date").transform("mean")
    sst_a = ((aligned_a["realized"] - ybar_t) ** 2).sum()
    expected_r2_a = 1 - sse_a / sst_a if sst_a != 0 else np.nan

    assert np.isclose(summary.loc["model_a", "mae"], expected_mae_a)
    assert np.isclose(summary.loc["model_a", "mse"], expected_mse_a)
    assert np.isclose(summary.loc["model_a", "oos_r2"], expected_r2_a)

    # ---- Spearman rank correlation: compute within each cross-section, then average over dates ----
    daily_corrs = []
    for _, group in aligned_a.groupby(level="date"):
        # Spearman can be implemented as Pearson correlation of within-date ranks.
        pred_ranks = group["model_a"].rank(pct=True)
        realized_ranks = group["realized"].rank(pct=True)
        daily_corrs.append(pred_ranks.corr(realized_ranks))
    expected_spearman = np.mean(daily_corrs)

    assert np.isclose(summary.loc["model_a", "spearman_rank_corr"], expected_spearman)

    # ---- Per-date Spearman output contract ----
    # The time series is useful for diagnostics, so we assert its index/columns are predictable.
    assert spearman.index.is_monotonic_increasing
    assert list(spearman.columns) == sorted(summary.index)


def test_evaluate_forecasting_accuracy_subset_models(sample_panel):
    # Allow callers to evaluate a subset without accidentally leaking in extra columns.
    summary, spearman = evaluate_forecasting_accuracy(
        sample_panel, realized_col="realized", model_cols=["model_b"]
    )

    assert list(summary.index) == ["model_b"]
    assert list(spearman.columns) == ["model_b"]


def test_evaluate_forecasting_accuracy_missing_realized(sample_panel):
    # The realized outcome column is required; missing it should fail clearly.
    with pytest.raises(KeyError):
        evaluate_forecasting_accuracy(sample_panel.drop(columns=["realized"]))


def test_evaluate_forecasting_accuracy_invalid_index(sample_panel):
    # The implementation relies on grouping by ticker/date, so it must reject non-MultiIndex inputs.
    bad_index_df = sample_panel.reset_index()
    with pytest.raises(ValueError):
        evaluate_forecasting_accuracy(bad_index_df, realized_col="realized")


def test_diebold_mariano_test(sample_panel):
    # DM test compares average forecast losses across time, so we validate it on a tiny deterministic panel.
    results = diebold_mariano_test(sample_panel, realized_col="realized")

    # The output index identifies which two models are being compared in each row.
    assert list(results.index.names) == ["model_1", "model_2"]
    assert ("model_a", "model_b") in results.index

    # ---- Manual DM construction for the same panel ----
    # Use squared error (the default loss) so the calculation is fully deterministic.
    se_a = (sample_panel["model_a"] - sample_panel["realized"]) ** 2
    se_b = (sample_panel["model_b"] - sample_panel["realized"]) ** 2

    # DM works on one observation per date: the cross-sectional mean loss differential.
    diff_by_date = (se_a - se_b).groupby(level="date").mean().sort_index()

    # With horizon=1, the Newey–West lag is 0, so var(mean) reduces to gamma0 / n.
    mean_diff = diff_by_date.mean()
    demeaned = diff_by_date - mean_diff
    gamma0 = (demeaned @ demeaned) / len(diff_by_date)
    var_mean = gamma0 / len(diff_by_date)
    expected_dm = mean_diff / np.sqrt(var_mean)

    assert np.isclose(results.loc[("model_a", "model_b"), "mean_loss_diff"], mean_diff)
    assert np.isclose(results.loc[("model_a", "model_b"), "dm_stat"], expected_dm)
    assert results.loc[("model_a", "model_b"), "periods"] == 2