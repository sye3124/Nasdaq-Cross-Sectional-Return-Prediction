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
    y_true = [1.0, 2.0, 3.0]
    y_pred = [0.8, 2.2, 3.1]
    r2 = compute_oos_r2(y_true, y_pred)

    errors = np.array(y_true) - np.array(y_pred)
    sse = np.sum(errors**2)
    baseline = np.mean(y_true)
    sst = np.sum((np.array(y_true) - baseline) ** 2)
    expected = 1 - sse / sst

    assert np.isclose(r2, expected)


def test_evaluate_forecasting_accuracy_summary(sample_panel):
    summary, spearman = evaluate_forecasting_accuracy(
        sample_panel, realized_col="realized"
    )

    assert set(summary.index) == {"model_a", "model_b"}
    assert set(summary.columns) == {"oos_r2", "mae", "mse", "spearman_rank_corr"}

    # Manual calculations for model_a
    aligned_a = sample_panel[["model_a", "realized"]]
    errors_a = aligned_a["model_a"] - aligned_a["realized"]
    expected_mae_a = errors_a.abs().mean()
    expected_mse_a = (errors_a**2).mean()
    baseline_a = aligned_a["realized"].mean()
    sse_a = ((aligned_a["realized"] - aligned_a["model_a"]) ** 2).sum()
    sst_a = ((aligned_a["realized"] - baseline_a) ** 2).sum()
    expected_r2_a = 1 - sse_a / sst_a

    assert np.isclose(summary.loc["model_a", "mae"], expected_mae_a)
    assert np.isclose(summary.loc["model_a", "mse"], expected_mse_a)
    assert np.isclose(summary.loc["model_a", "oos_r2"], expected_r2_a)

    # Cross-sectional Spearman correlations
    daily_corrs = []
    for _, group in aligned_a.groupby(level="date"):
        pred_ranks = group["model_a"].rank(pct=True)
        realized_ranks = group["realized"].rank(pct=True)
        daily_corrs.append(pred_ranks.corr(realized_ranks))
    expected_spearman = np.mean(daily_corrs)

    assert np.isclose(summary.loc["model_a", "spearman_rank_corr"], expected_spearman)

    # Check per-date Spearman output shape and ordering
    assert spearman.index.is_monotonic_increasing
    assert list(spearman.columns) == sorted(summary.index)


def test_evaluate_forecasting_accuracy_subset_models(sample_panel):
    summary, spearman = evaluate_forecasting_accuracy(
        sample_panel, realized_col="realized", model_cols=["model_b"]
    )

    assert list(summary.index) == ["model_b"]
    assert list(spearman.columns) == ["model_b"]


def test_evaluate_forecasting_accuracy_missing_realized(sample_panel):
    with pytest.raises(KeyError):
        evaluate_forecasting_accuracy(sample_panel.drop(columns=["realized"]))


def test_evaluate_forecasting_accuracy_invalid_index(sample_panel):
    bad_index_df = sample_panel.reset_index()
    with pytest.raises(ValueError):
        evaluate_forecasting_accuracy(bad_index_df, realized_col="realized")

def test_diebold_mariano_test(sample_panel):
    results = diebold_mariano_test(sample_panel, realized_col="realized")

    assert list(results.index.names) == ["model_1", "model_2"]
    assert ("model_a", "model_b") in results.index

    # Manual squared-error loss differentials averaged by date
    diff_by_date = pd.Series(
        {
            pd.to_datetime("2020-01-01"): 0.00025 - 0.025,
            pd.to_datetime("2020-02-01"): 0.00025 - 0.0625,
        }
    )
    mean_diff = diff_by_date.mean()
    demeaned = diff_by_date - mean_diff
    gamma0 = (demeaned @ demeaned) / len(diff_by_date)
    var_mean = gamma0 / len(diff_by_date)
    expected_dm = mean_diff / np.sqrt(var_mean)

    assert np.isclose(results.loc[("model_a", "model_b"), "mean_loss_diff"], mean_diff)
    assert np.isclose(results.loc[("model_a", "model_b"), "dm_stat"], expected_dm)
    assert results.loc[("model_a", "model_b"), "periods"] == 2
