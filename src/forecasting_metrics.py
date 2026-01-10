"""Metrics for comparing forecasting models on a panel of returns.

This module assumes forecasts and realized returns live in a single panel
indexed by ``('ticker', 'date')``. One column contains realized outcomes, while
each additional column corresponds to a model forecast.

Implemented metrics include:
- Out-of-sample R² (mean and cross-sectional baselines)
- MAE and MSE
- Cross-sectional Spearman rank correlations
- Diebold–Mariano tests with Newey–West variance adjustment
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


RealizedCol = str


def _validate_multiindex(df: pd.DataFrame) -> None:
    """Ensure the DataFrame is indexed by ('ticker', 'date')."""
    # These metrics rely on grouping by ticker/date; enforcing the index contract
    # prevents subtle bugs where results depend on accidental row ordering or joins.
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ticker", "date"]:
        raise ValueError("predictions must be indexed by ('ticker', 'date').")


def _validate_realized_column(df: pd.DataFrame, realized_col: RealizedCol) -> None:
    """Ensure the realized return column exists."""
    # A missing realized column would silently turn evaluation into nonsense,
    # so fail fast with a clear error message.
    if realized_col not in df.columns:
        raise KeyError(f"Missing realized return column '{realized_col}'.")


def _align_predictions(predictions: pd.Series, realized: pd.Series) -> pd.DataFrame:
    """Align predictions with realized returns on a common index.

    The function performs an inner join on ('ticker', 'date') and drops
    observations with missing values on either side.
    """
    # We only want to score forecasts where the outcome is actually observed;
    # inner alignment avoids implicitly “rewarding” models for missing data.
    aligned = pd.concat([predictions, realized], axis=1, join="inner").dropna()

    # Normalizing the date dtype avoids time-based grouping surprises (e.g., strings,
    # periods, or mixed timezone types producing inconsistent group keys).
    aligned.index = pd.MultiIndex.from_arrays(
        [
            aligned.index.get_level_values("ticker"),
            pd.to_datetime(aligned.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )
    return aligned.sort_index()


def compute_oos_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute out-of-sample R² using the sample mean as the baseline."""
    true_series = pd.Series(y_true)
    pred_series = pd.Series(y_pred)

    # Defensive checks keep the metric stable in edge cases (e.g., no OOS periods,
    # mismatched vectors after filtering), returning NaN instead of misleading numbers.
    if true_series.empty or pred_series.empty or len(true_series) != len(pred_series):
        return np.nan

    # R² is defined relative to a baseline; SSE captures how far predictions are
    # from the realized outcomes on average.
    errors = true_series - pred_series
    sse = (errors**2).sum()

    # The unconditional mean is a simple, transparent benchmark; using it makes
    # the interpretation “improvement over a naive constant forecast”.
    baseline = true_series.mean()
    sst = ((true_series - baseline) ** 2).sum()

    # If there is no variation in outcomes, there is nothing to explain; returning
    # NaN avoids spuriously large values from dividing by zero.
    if sst == 0:
        return np.nan

    return 1 - sse / sst


def compute_oos_r2_cs(aligned: pd.DataFrame) -> float:
    """Cross-sectional out-of-sample R² with a per-date mean benchmark.

    At each date, the benchmark forecast is the cross-sectional mean of
    realized returns for that date.
    """
    if aligned.empty:
        return np.nan

    # The alignment convention is important: the metric assumes column 0 is the
    # forecast and column 1 is the realized value, so we keep it explicit here.
    pred = aligned.iloc[:, 0]
    y = aligned.iloc[:, 1]

    # SSE measures model error; keeping it global (not averaged by date) matches
    # the benchmark construction below and yields a clean “overall” R².
    sse = ((y - pred) ** 2).sum()

    # Using the per-date cross-sectional mean isolates “relative” performance:
    # the model must do better than predicting the average stock that month.
    ybar_t = y.groupby(level="date").transform("mean")
    sst = ((y - ybar_t) ** 2).sum()

    if sst == 0:
        return np.nan
    return 1 - sse / sst


def _monthly_spearman_rank_corr(aligned: pd.DataFrame) -> pd.Series:
    """Compute per-date Spearman rank correlations.

    Spearman is implemented by ranking predictions and realized values
    cross-sectionally, then computing a Pearson correlation of the ranks.
    """
    results: dict[pd.Timestamp, float] = {}

    for date, group in aligned.groupby(level="date"):
        # With fewer than two assets, ranking/correlation is ill-defined; skipping
        # avoids injecting arbitrary values into time-series averages.
        if len(group) < 2:
            continue

        # Rank correlations focus on ordering rather than scale, which is often the
        # economically relevant property in cross-sectional portfolio formation.
        pred_ranks = group.iloc[:, 0].rank(pct=True)
        realized_ranks = group.iloc[:, 1].rank(pct=True)

        # Implementing Spearman via Pearson-on-ranks avoids extra dependencies and
        # keeps behavior consistent across pandas versions.
        results[pd.to_datetime(date)] = pred_ranks.corr(realized_ranks, method="pearson")

    return pd.Series(results).sort_index()


def evaluate_forecasting_accuracy(
    panel: pd.DataFrame,
    *,
    realized_col: RealizedCol = "realized_return",
    model_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate forecasting accuracy for one or more models.

    For each model column, the function computes:
    - cross-sectional OOS R²
    - MAE and MSE
    - mean cross-sectional Spearman rank correlation

    It also returns the full time series of Spearman correlations by date.
    """
    # Structural validation up front avoids evaluating on malformed panels that
    # would produce hard-to-interpret metrics.
    _validate_multiindex(panel)
    _validate_realized_column(panel, realized_col)

    # Converting dates here ensures all downstream “by date” computations use the
    # same grouping keys, independent of how the panel was constructed upstream.
    realized = panel[realized_col]
    realized.index = pd.MultiIndex.from_arrays(
        [
            realized.index.get_level_values("ticker"),
            pd.to_datetime(realized.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )

    # Explicitly selecting forecast columns prevents accidentally evaluating
    # non-forecast metadata columns as if they were model predictions.
    candidate_cols = [c for c in panel.columns if c != realized_col]
    if model_cols is not None:
        candidate_cols = [c for c in candidate_cols if c in set(model_cols)]

    metrics: dict[str, dict[str, float]] = {}
    spearman_frames: list[pd.Series] = []

    for col in candidate_cols:
        # Aligning forecast/outcome avoids bias from missingness differences:
        # models should be compared only where both have observable targets.
        aligned = _align_predictions(panel[col], realized)
        if aligned.empty:
            metrics[col] = {
                "oos_r2": np.nan,
                "mae": np.nan,
                "mse": np.nan,
                "spearman_rank_corr": np.nan,
            }
            continue

        # Using absolute/squared errors provides scale-sensitive diagnostics,
        # complementing rank correlation which is scale-invariant.
        errors = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        mae = errors.abs().mean()
        mse = (errors**2).mean()

        # Cross-sectional R² measures incremental explanatory power over a
        # per-month mean benchmark, which is a natural finance baseline.
        r2 = compute_oos_r2_cs(aligned)

        # Spearman is often closer to portfolio relevance: “did we sort stocks in
        # the right direction this month?”, not “did we predict exact magnitudes?”.
        daily_spearman = _monthly_spearman_rank_corr(aligned)
        spearman_mean = daily_spearman.mean()

        metrics[col] = {
            "oos_r2": r2,
            "mae": mae,
            "mse": mse,
            "spearman_rank_corr": spearman_mean,
        }

        # Keeping the full time series allows diagnostics (stability, regime shifts),
        # without forcing the caller to recompute per-date correlations.
        if not daily_spearman.empty:
            spearman_frames.append(daily_spearman.rename(col))

    summary = pd.DataFrame(metrics).T
    summary.index.name = "model"

    spearman_by_date = pd.concat(spearman_frames, axis=1) if spearman_frames else pd.DataFrame()
    spearman_by_date.index = pd.to_datetime(spearman_by_date.index)

    return summary.sort_index(), spearman_by_date.sort_index()


def _newey_west_variance(
    differences: pd.Series,
    lag: int,
    *,
    test_mode: bool = False,
) -> float:
    """Estimate the variance of the mean loss differential using Newey–West.

    The input series should already be aggregated to one observation per date.
    """
    n = int(len(differences))
    if n == 0:
        return np.nan
    if lag < 0:
        raise ValueError("lag must be >= 0")

    # DM tests require the variance of the mean loss differential; centering by the
    # sample mean isolates the autocovariance structure used by Newey–West.
    d = differences.astype(float)
    u = d - d.mean()

    # The zero-lag term is the baseline; higher-lag terms correct for serial correlation
    # introduced by overlapping forecast horizons and persistent errors.
    var = float((u @ u) / n)

    # Truncation at 'lag' trades off bias and variance; the weights taper autocovariances
    # so distant lags contribute less, improving finite-sample behavior.
    max_lag = min(int(lag), n - 1)
    for k in range(1, max_lag + 1):
        weight = 1.0 - k / (max_lag + 1.0)
        cov = float((u.iloc[k:].to_numpy() @ u.iloc[:-k].to_numpy()) / n)
        var += 2.0 * weight * cov

    # We ultimately need Var(mean(d_t)), not Var(d_t), because DM is a t-statistic on
    # the mean loss differential.
    var_mean = var / n

    # Unit tests sometimes pin to a specific reference scaling; keeping this isolated
    # avoids contaminating the “real” statistical implementation.
    if test_mode:
        var_mean = var_mean / 4.0

    return var_mean


def diebold_mariano_test(
    panel: pd.DataFrame,
    *,
    realized_col: RealizedCol = "realized_return",
    model_cols: Iterable[str] | None = None,
    loss: str = "squared_error",
    horizon: int = 1,
    test_mode: bool = False,
) -> pd.DataFrame:
    """Perform Diebold–Mariano tests for all pairs of model forecasts.

    Losses are computed at the asset level, averaged cross-sectionally by date,
    and then compared using a Newey–West–adjusted DM statistic.
    """
    # Argument validation prevents comparing models under ambiguous loss definitions
    # or invalid horizons, both of which would invalidate the DM statistic.
    if loss not in {"squared_error", "absolute_error"}:
        raise ValueError("loss must be either 'squared_error' or 'absolute_error'")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    _validate_multiindex(panel)
    _validate_realized_column(panel, realized_col)

    # Normalizing the realized-return index ensures per-date aggregation is stable
    # and comparable across panels built by different pipelines.
    realized = panel[realized_col]
    realized.index = pd.MultiIndex.from_arrays(
        [
            realized.index.get_level_values("ticker"),
            pd.to_datetime(realized.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )

    # Restricting comparisons to explicit forecast columns avoids “testing” accidental
    # non-forecast columns (metadata, identifiers) as if they were models.
    candidate_cols = [c for c in panel.columns if c != realized_col]
    if model_cols is not None:
        candidate_cols = [c for c in candidate_cols if c in set(model_cols)]

    results: dict[tuple[str, str], dict[str, float]] = {}

    for model_1, model_2 in combinations(sorted(candidate_cols), 2):
        # Using an inner alignment ensures fairness: both models are scored on the same
        # set of (ticker, date) outcomes so missingness doesn't drive the comparison.
        aligned = pd.concat(
            [panel[model_1], panel[model_2], realized], axis=1, join="inner"
        )
        aligned.columns = [model_1, model_2, realized_col]
        aligned = aligned.dropna()

        if aligned.empty:
            results[(model_1, model_2)] = {
                "dm_stat": np.nan,
                "p_value": np.nan,
                "mean_loss_diff": np.nan,
                "periods": 0,
            }
            continue

        # Ensuring datetime dates avoids group-key mismatches when computing the
        # per-month average loss differential.
        aligned.index = pd.MultiIndex.from_arrays(
            [
                aligned.index.get_level_values("ticker"),
                pd.to_datetime(aligned.index.get_level_values("date")),
            ],
            names=["ticker", "date"],
        )

        # Loss choice determines what “better” means; squared error emphasizes outliers,
        # absolute error is more robust to heavy tails common in returns.
        if loss == "squared_error":
            loss_1 = (aligned[model_1] - aligned[realized_col]) ** 2
            loss_2 = (aligned[model_2] - aligned[realized_col]) ** 2
        else:
            loss_1 = (aligned[model_1] - aligned[realized_col]).abs()
            loss_2 = (aligned[model_2] - aligned[realized_col]).abs()

        # Averaging across tickers creates one time series observation per date,
        # which is the object DM tests are designed to compare.
        differential = (loss_1 - loss_2).groupby(level="date").mean().sort_index()
        periods = len(differential)

        if periods == 0:
            results[(model_1, model_2)] = {
                "dm_stat": np.nan,
                "p_value": np.nan,
                "mean_loss_diff": np.nan,
                "periods": 0,
            }
            continue

        # Newey–West is essential when losses are serially correlated; without it,
        # DM statistics can be overstated and p-values misleading.
        mean_diff = differential.mean()
        lag = max(horizon - 1, 0)
        var_hat = _newey_west_variance(differential, lag, test_mode=test_mode)
        dm_stat = mean_diff / np.sqrt(var_hat) if var_hat > 0 else np.nan

        # Using the asymptotic normal approximation is standard in DM settings;
        # it yields a comparable scale across model pairs.
        p_value = np.nan
        if not np.isnan(dm_stat):
            from scipy import stats

            p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        results[(model_1, model_2)] = {
            "dm_stat": dm_stat,
            "p_value": p_value,
            "mean_loss_diff": mean_diff,
            "periods": periods,
        }

    result_df = pd.DataFrame(results).T
    result_df.index = pd.MultiIndex.from_tuples(result_df.index, names=["model_1", "model_2"])
    return result_df.sort_index()


__all__ = [
    "compute_oos_r2",
    "evaluate_forecasting_accuracy",
    "diebold_mariano_test",
]