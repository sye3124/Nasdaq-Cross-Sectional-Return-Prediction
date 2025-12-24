"""Performance metrics for decile portfolios and long-short spreads."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is a DatetimeIndex."""
    result = df.copy()
    result.index = pd.to_datetime(result.index)
    return result


def compute_long_short_returns(
    decile_returns: pd.DataFrame,
    *,
    bottom_decile: int = 1,
    top_decile: int | None = None,
) -> pd.DataFrame:
    """Construct long-short returns from top minus bottom deciles for each model.

    Robust to extra columns like 'LS' by using only numeric decile columns.
    """

    # Validate input
    if not isinstance(decile_returns, pd.DataFrame):
        raise TypeError("decile_returns must be a DataFrame")
    if not isinstance(decile_returns.columns, pd.MultiIndex):
        raise ValueError("decile_returns must have MultiIndex columns (model, decile)")

    out: dict[tuple[str, str], pd.Series] = {}

    # For each model, compute long-short returns
    for model in decile_returns.columns.get_level_values(0).unique():
        subset = decile_returns[model]
        if subset.empty:
            continue

        # Keep only numeric decile columns (ignore 'LS' or any non-numeric labels)
        numeric_deciles = [c for c in subset.columns if isinstance(c, (int, np.integer))]
        if not numeric_deciles:
            continue
        
        # Determine bottom and top deciles to use
        bot = bottom_decile if bottom_decile in numeric_deciles else min(numeric_deciles)
        top = top_decile if (top_decile is not None and top_decile in numeric_deciles) else max(numeric_deciles)

        # Compute long-short spread
        spread = subset[top] - subset[bot]
        spread.name = (model, "long_short")
        out[(model, "long_short")] = spread

    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()


def compute_turnover_from_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Compute per-period turnover from portfolio weight changes."""

    if not isinstance(weights.index, pd.MultiIndex) or weights.index.names[:2] != ["ticker", "date"]:
        raise ValueError("weights must be indexed by ('ticker', 'date')")

    # Normalize index to ensure 'date' level is datetime
    normalized = weights.copy()
    normalized.index = pd.MultiIndex.from_arrays(
        [
            normalized.index.get_level_values("ticker"),
            pd.to_datetime(normalized.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )

    turnovers: dict[object, pd.Series] = {}

    # For each (model, decile) column, compute normalized turnover over time
    for col in normalized.columns:
        pivot = (
            normalized[col]
            .unstack("ticker")
            .fillna(0.0)
            .sort_index()
        )

        # 0.5 * sum |Δw_i| over tickers
        abs_change = pivot.diff().abs()
        per_period = abs_change.sum(axis=1) * 0.5

        # First period has no prior weights → turnover undefined
        if len(per_period) > 0:
            per_period.iloc[0] = np.nan

        turnovers[col] = per_period

    result = pd.DataFrame(turnovers)

    # Ensure a DatetimeIndex and explicitly drop the name so it matches the tests
    result.index = pd.to_datetime(result.index)
    result.index.name = None

    return result


def _annualize_stat(value: float, periods_per_year: int) -> float:
    """Annualize a return statistic (mean return)."""
    return value * periods_per_year


def _annualize_vol(vol: float, periods_per_year: int) -> float:
    """Annualize a volatility statistic."""
    return vol * math.sqrt(periods_per_year)


def summarize_portfolio_performance(
    decile_returns: pd.DataFrame,
    *,
    turnover_weights: pd.DataFrame | None = None,
    transaction_cost_bps: float | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    plot_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize portfolio performance statistics and optional plots.
    When ``turnover_weights`` and ``transaction_cost_bps`` are provided, decile
    returns are reduced by the implied trading costs (turnover × cost per
    trade) before computing long-short spreads and summary statistics.
    """

    returns = _ensure_datetime_index(decile_returns)

    # Compute turnover and adjust returns for transaction costs if provided
    turnover_summary: pd.Series | None = None
    if turnover_weights is not None:
        turnover = compute_turnover_from_weights(turnover_weights)
        turnover = _ensure_datetime_index(turnover)
        turnover_summary = turnover.mean()

        # Adjust returns for transaction costs
        if transaction_cost_bps is not None and transaction_cost_bps != 0:
            cost_per_trade = transaction_cost_bps / 10_000
            turnover_cost = turnover.reindex(returns.index).fillna(0.0)

            adjusted = returns.copy()
            overlapping = adjusted.columns.intersection(turnover_cost.columns)
            if not overlapping.empty:
                adjusted[overlapping] = adjusted[overlapping] - (
                    turnover_cost[overlapping] * cost_per_trade
                )
            returns = adjusted

    # Compute long-short returns and append to returns DataFrame
    if isinstance(returns.columns, pd.MultiIndex):
        # Add long_short only if LS isn't already present
        has_ls = any(str(d) == "LS" for d in returns.columns.get_level_values(1))
        if not has_ls:
            long_short = compute_long_short_returns(returns)
            if not long_short.empty:
                returns = pd.concat([returns, long_short], axis=1)


    # Compute performance metrics
    metrics: dict[object, dict[str, float]] = {}
    cumulative: dict[object, pd.Series] = {}
    drawdowns: dict[object, pd.Series] = {}

    rf_per_period = risk_free_rate / periods_per_year

    # For each portfolio (column), compute metrics
    for col in returns.columns:
        series = returns[col].dropna()
        if series.empty:
            continue
        mean_return = series.mean()
        volatility = series.std(ddof=1)
        sharpe = np.nan
        # Sharpe ratio
        if volatility and not np.isnan(volatility):
            sharpe = ((mean_return - rf_per_period) / volatility) * math.sqrt(periods_per_year)
        t_stat = np.nan
        # T-statistic for mean return
        if volatility and len(series) > 1:
            t_stat = mean_return / (volatility / math.sqrt(len(series)))

        # Cumulative returns and drawdowns
        cum = (1.0 + series).cumprod()
        dd = cum / cum.cummax() - 1.0

        cumulative[col] = cum
        drawdowns[col] = dd

        metrics[col] = {
            "mean_return": _annualize_stat(mean_return, periods_per_year),
            "volatility": _annualize_vol(volatility, periods_per_year),
            "sharpe_ratio": sharpe,
            "t_stat_mean": t_stat,
            "max_drawdown": dd.min(),
            "avg_turnover": turnover_summary[col] if turnover_summary is not None and col in turnover_summary else np.nan,
        }

    # Convert metrics to DataFrames
    metrics_df = pd.DataFrame(metrics).T.sort_index()
    cumulative_df = pd.DataFrame(cumulative).sort_index()
    drawdown_df = pd.DataFrame(drawdowns).sort_index()

    # Generate cumulative returns plot if requested
    if plot_path is not None and not cumulative_df.empty:
        _plot_cumulative_returns(cumulative_df, plot_path)

    return metrics_df, cumulative_df, drawdown_df


def _format_column_label(label: object) -> str:
    """Format column label for plotting."""
    if isinstance(label, tuple):
        return " - ".join(str(part) for part in label)
    return str(label)


def _plot_cumulative_returns(cumulative: pd.DataFrame, output_path: str | Path) -> None:
    """Plot cumulative returns and save to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for col in cumulative.columns:
        plt.plot(cumulative.index, cumulative[col], label=_format_column_label(col))
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _compute_sharpe_ratio(
    returns: pd.Series, *, risk_free_rate: float = 0.0, periods_per_year: int = 12
) -> float:
    """Compute annualized Sharpe ratio for a return series."""
    excess = returns.dropna() - risk_free_rate / periods_per_year
    volatility = excess.std(ddof=1)
    if volatility == 0 or np.isnan(volatility):
        return np.nan
    return excess.mean() / volatility * math.sqrt(periods_per_year)


def _align_model_returns(
    returns: pd.DataFrame, model_1: object, model_2: object
) -> pd.DataFrame:
    """Align returns for two models, dropping NaNs and sorting by date."""
    if model_1 not in returns.columns or model_2 not in returns.columns:
        raise KeyError("Both model columns must be present in the returns DataFrame")

    aligned = returns[[model_1, model_2]].dropna()
    aligned.index = pd.to_datetime(aligned.index)
    return aligned.sort_index()


def jobson_korkie_test(
    returns: pd.DataFrame,
    *,
    model_1: object,
    model_2: object,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    memmel_correction: bool = True,
) -> pd.Series:
    """Jobson-Korkie test (with optional Memmel correction) for Sharpe ratios.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with columns for each model's returns. Index is interpreted as
        time and is coerced to ``DatetimeIndex``.
    model_1, model_2 : hashable
        Column labels for the two models to compare.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate used to compute excess returns.
    periods_per_year : int, default 12
        Number of return observations per year.
    memmel_correction : bool, default True
        Apply the small-sample correction proposed by Memmel (2003).

    Returns
    -------
    pd.Series
        Contains the Sharpe ratios for each model, their difference, the test
        statistic, and two-sided p-value.
    """

    # Align returns for the two models
    aligned = _align_model_returns(returns, model_1, model_2)
    n = len(aligned)
    if n < 2:
        return pd.Series(
            {
                "sharpe_1": np.nan,
                "sharpe_2": np.nan,
                "sharpe_diff": np.nan,
                "jk_stat": np.nan,
                "p_value": np.nan,
                "periods": n,
            }
        )

    # Compute Sharpe ratios
    sr1 = _compute_sharpe_ratio(
        aligned[model_1], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    sr2 = _compute_sharpe_ratio(
        aligned[model_2], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )

    # Compute covariance and correlation between returns
    cov = aligned.cov(ddof=1).iloc[0, 1]
    vol1 = aligned[model_1].std(ddof=1)
    vol2 = aligned[model_2].std(ddof=1)
    rho = cov / (vol1 * vol2) if vol1 and vol2 else np.nan

    # Compute variance of Sharpe ratio difference
    variance = np.nan
    if not np.isnan(rho):
        variance = (
            (2 * (1 - rho) * sr1 * sr2) + 0.5 * (sr1**2 + sr2**2)
        ) / n
        if memmel_correction and n > 1:
            variance -= ((sr1 - sr2) ** 2) / (2 * (n - 1))

    jk_stat = (sr1 - sr2) / math.sqrt(variance) if variance and variance > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(jk_stat))) if not np.isnan(jk_stat) else np.nan

    return pd.Series(
        {
            "sharpe_1": sr1,
            "sharpe_2": sr2,
            "sharpe_diff": sr1 - sr2,
            "jk_stat": jk_stat,
            "p_value": p_value,
            "periods": n,
        }
    )


def bootstrap_sharpe_ratio_difference(
    returns: pd.DataFrame,
    *,
    model_1: object,
    model_2: object,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> pd.Series:
    """Bootstrap test for differences in model Sharpe ratios.

    A percentile bootstrap is used to approximate the sampling distribution of
    the Sharpe ratio difference. The function returns the observed difference,
    bootstrap mean and standard deviation, and a two-sided p-value.
    """

    aligned = _align_model_returns(returns, model_1, model_2)
    n = len(aligned)
    if n == 0 or n_bootstrap <= 0:
        return pd.Series(
            {
                "sharpe_1": np.nan,
                "sharpe_2": np.nan,
                "sharpe_diff": np.nan,
                "bootstrap_mean": np.nan,
                "bootstrap_std": np.nan,
                "p_value": np.nan,
                "periods": n,
            }
        )

    # Compute observed Sharpe ratios and their difference
    sr1 = _compute_sharpe_ratio(
        aligned[model_1], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    sr2 = _compute_sharpe_ratio(
        aligned[model_2], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    observed_diff = sr1 - sr2

    # Bootstrap resampling
    rng = np.random.default_rng(random_state)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sampled = aligned.iloc[sample_idx]
        boot_sr1 = _compute_sharpe_ratio(
            sampled[model_1], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        )
        boot_sr2 = _compute_sharpe_ratio(
            sampled[model_2], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        )
        boot_diffs[i] = boot_sr1 - boot_sr2

    # Compute p-value based on bootstrap distribution
    greater = np.mean(boot_diffs >= observed_diff)
    lower = np.mean(boot_diffs <= observed_diff)
    p_value = 2 * min(greater, lower)

    return pd.Series(
        {
            "sharpe_1": sr1,
            "sharpe_2": sr2,
            "sharpe_diff": observed_diff,
            "bootstrap_mean": boot_diffs.mean(),
            "bootstrap_std": boot_diffs.std(ddof=1),
            "p_value": p_value,
            "periods": n,
        }
    )


__all__ = [
    "compute_long_short_returns",
    "compute_turnover_from_weights",
    "summarize_portfolio_performance",
    "jobson_korkie_test",
    "bootstrap_sharpe_ratio_difference",
]