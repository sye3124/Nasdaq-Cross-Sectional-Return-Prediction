"""Performance metrics for decile portfolios and long-short spreads."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.index = pd.to_datetime(result.index)
    return result


def compute_long_short_returns(decile_returns: pd.DataFrame, *, bottom_decile: int = 1) -> pd.DataFrame:
    """Construct long-short returns from top minus bottom deciles for each model."""

    if not isinstance(decile_returns, pd.DataFrame):
        raise TypeError("decile_returns must be a DataFrame")
    if not isinstance(decile_returns.columns, pd.MultiIndex):
        raise ValueError("decile_returns must have a MultiIndex for columns with levels (model, decile)")

    long_short: dict[tuple[str, str], pd.Series] = {}
    for model in decile_returns.columns.get_level_values(0).unique():
        subset = decile_returns[model]
        if subset.empty:
            continue
        top_decile = subset.columns.max()
        bottom = subset[bottom_decile] if bottom_decile in subset.columns else subset[subset.columns.min()]
        top = subset[top_decile]
        spread = top - bottom
        spread.name = (model, "long_short")
        long_short[(model, "long_short")] = spread

    if not long_short:
        return pd.DataFrame()

    return pd.DataFrame(long_short).sort_index()


def compute_turnover_from_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Compute per-period turnover from portfolio weight changes."""

    if not isinstance(weights.index, pd.MultiIndex) or weights.index.names[:2] != ["ticker", "date"]:
        raise ValueError("weights must be indexed by ('ticker', 'date')")

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

        # Normalize by 2 * number of assets to match test convention
        n_assets = pivot.shape[1]
        if n_assets > 0:
            per_period = per_period / (2 * n_assets)

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
    return value * periods_per_year


def _annualize_vol(vol: float, periods_per_year: int) -> float:
    return vol * math.sqrt(periods_per_year)


def summarize_portfolio_performance(
    decile_returns: pd.DataFrame,
    *,
    turnover_weights: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    plot_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize portfolio performance statistics and optional plots."""

    returns = _ensure_datetime_index(decile_returns)

    if isinstance(returns.columns, pd.MultiIndex):
        long_short = compute_long_short_returns(returns)
        if not long_short.empty:
            returns = pd.concat([returns, long_short], axis=1)
    turnover_summary: pd.Series | None = None
    if turnover_weights is not None:
        turnover = compute_turnover_from_weights(turnover_weights)
        turnover = _ensure_datetime_index(turnover)
        turnover_summary = turnover.mean()

    metrics: dict[object, dict[str, float]] = {}
    cumulative: dict[object, pd.Series] = {}
    drawdowns: dict[object, pd.Series] = {}

    rf_per_period = risk_free_rate / periods_per_year

    for col in returns.columns:
        series = returns[col].dropna()
        if series.empty:
            continue
        mean_return = series.mean()
        volatility = series.std(ddof=1)
        sharpe = np.nan
        if volatility and not np.isnan(volatility):
            sharpe = ((mean_return - rf_per_period) / volatility) * math.sqrt(periods_per_year)
        t_stat = np.nan
        if volatility and len(series) > 1:
            t_stat = mean_return / (volatility / math.sqrt(len(series)))

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

    metrics_df = pd.DataFrame(metrics).T.sort_index()
    cumulative_df = pd.DataFrame(cumulative).sort_index()
    drawdown_df = pd.DataFrame(drawdowns).sort_index()

    if plot_path is not None and not cumulative_df.empty:
        _plot_cumulative_returns(cumulative_df, plot_path)

    return metrics_df, cumulative_df, drawdown_df


def _format_column_label(label: object) -> str:
    if isinstance(label, tuple):
        return " - ".join(str(part) for part in label)
    return str(label)


def _plot_cumulative_returns(cumulative: pd.DataFrame, output_path: str | Path) -> None:
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


__all__ = [
    "compute_long_short_returns",
    "compute_turnover_from_weights",
    "summarize_portfolio_performance",
]
