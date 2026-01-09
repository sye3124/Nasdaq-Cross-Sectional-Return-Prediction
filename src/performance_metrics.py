"""Performance reporting for decile portfolios and long/short spreads.

This module focuses on the “portfolio layer” of the pipeline: given time series
of decile portfolio returns (often produced from model ranks), it can:

- compute top-minus-bottom long/short returns per model
- compute turnover from portfolio weights
- apply simple transaction-cost adjustments (bps × turnover)
- summarize performance (annualized mean, vol, Sharpe, drawdowns, turnover)
- optionally plot cumulative returns
- compare Sharpe ratios across models (Jobson–Korkie / bootstrap)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_sort_decile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort MultiIndex columns (model, decile) in a way that won't break selection.

    We sort:
    - model level alphabetically
    - numeric deciles ascending
    - everything else (e.g. "LS"/"long_short") after numeric deciles

    This avoids a handful of pandas edge cases where partially sorted MultiIndex
    columns can trigger KeyErrors during slicing / plotting.
    """
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        return df

    cols = list(df.columns)
    models = sorted(set(m for m, _ in cols))

    ordered_cols: list[tuple[object, object]] = []
    for m in models:
        level2 = [d for mm, d in cols if mm == m]
        numeric = sorted([d for d in level2 if isinstance(d, (int, np.integer))])
        other = [d for d in level2 if not isinstance(d, (int, np.integer))]
        ordered_cols.extend([(m, d) for d in numeric + other])

    return df.loc[:, ordered_cols]


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the index coerced to DatetimeIndex."""
    result = df.copy()
    result.index = pd.to_datetime(result.index)
    return result


def compute_long_short_returns(
    decile_returns: pd.DataFrame,
    *,
    bottom_decile: int = 1,
    top_decile: int | None = None,
    cap: float | None = None,
) -> pd.DataFrame:
    """Build top-minus-bottom long/short returns for each model.

    The input is expected to have columns like (model, 1), (model, 2), ... where
    the second level is the decile number. Non-numeric second-level entries are
    ignored.

    Parameters
    ----------
    decile_returns
        DataFrame indexed by date with MultiIndex columns (model, decile).
    bottom_decile
        Which decile to treat as "short" (default: 1).
    top_decile
        Which decile to treat as "long". If None, uses the largest decile found.
    cap
        Optional cap applied to the long/short spread each period (winsor-like
        clipping), useful for stabilizing plots.

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex columns (model, "long_short"). Empty if inputs
        are empty or no valid deciles are found.
    """
    if not isinstance(decile_returns, pd.DataFrame):
        raise TypeError("decile_returns must be a DataFrame")
    if decile_returns.empty:
        return pd.DataFrame()

    # Make sure columns are truly a 2-level MultiIndex; accept list-of-tuples too.
    cols = decile_returns.columns
    if not isinstance(cols, pd.MultiIndex):
        if len(cols) > 0 and all(isinstance(c, tuple) and len(c) == 2 for c in cols):
            decile_returns = decile_returns.copy()
            decile_returns.columns = pd.MultiIndex.from_tuples(cols)
        else:
            raise ValueError("decile_returns must have MultiIndex columns (model, decile)")

    out: dict[tuple[object, str], pd.Series] = {}

    model_level = decile_returns.columns.get_level_values(0)
    decile_level = decile_returns.columns.get_level_values(1)

    for model in pd.unique(model_level):
        numeric_deciles = sorted(
            {d for d in decile_level[model_level == model] if isinstance(d, (int, np.integer))}
        )
        if not numeric_deciles:
            continue

        bot = bottom_decile if bottom_decile in numeric_deciles else numeric_deciles[0]
        top = (
            top_decile
            if (top_decile is not None and top_decile in numeric_deciles)
            else numeric_deciles[-1]
        )

        col_bot = (model, bot)
        col_top = (model, top)
        if col_bot not in decile_returns.columns or col_top not in decile_returns.columns:
            continue

        spread = decile_returns[col_top] - decile_returns[col_bot]

        if cap is not None:
            spread = spread.clip(lower=-cap, upper=cap)

        spread.name = (model, "long_short")
        out[(model, "long_short")] = spread

    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()


def compute_turnover_from_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Compute turnover from a panel of portfolio weights.

    Turnover is computed per period as:

        0.5 * sum_i |w_i,t - w_i,t-1|

    The factor 0.5 avoids double-counting buys and sells for a fully-invested
    portfolio.
    """
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

    for col in normalized.columns:
        pivot = normalized[col].unstack("ticker").fillna(0.0).sort_index()
        abs_change = pivot.diff().abs()
        per_period = abs_change.sum(axis=1) * 0.5

        if len(per_period) > 0:
            per_period.iloc[0] = np.nan

        turnovers[col] = per_period

    result = pd.DataFrame(turnovers)
    result.index = pd.to_datetime(result.index)
    result.index.name = None
    return result


def _annualize_stat(value: float, periods_per_year: int) -> float:
    """Annualize a per-period mean return."""
    return value * periods_per_year


def _annualize_vol(vol: float, periods_per_year: int) -> float:
    """Annualize a per-period volatility."""
    return vol * math.sqrt(periods_per_year)


def summarize_portfolio_performance(
    decile_returns: pd.DataFrame,
    *,
    turnover_weights: pd.DataFrame | None = None,
    transaction_cost_bps: float | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    plot_path: str | Path | None = None,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute summary stats, cumulative performance, and drawdowns.

    If turnover weights and a transaction-cost setting are provided, returns are
    reduced by:

        cost_t = turnover_t * (bps / 10_000)

    Costs are applied:
    - directly to the provided decile portfolios (matching columns), and
    - to derived long/short portfolios using a simple proxy:
        turnover(LS) ≈ turnover(top) + turnover(bottom)
    """
    returns = _ensure_datetime_index(decile_returns)

    if start_date is not None:
        returns = returns.loc[start_date:].copy()

    returns = _safe_sort_decile_columns(returns)
    returns.columns = pd.MultiIndex.from_tuples(list(returns.columns))

    if isinstance(returns.columns, pd.MultiIndex):
        returns = returns.sort_index(axis=1)

    turnover_summary: pd.Series | None = None

    if turnover_weights is not None:
        turnover = compute_turnover_from_weights(turnover_weights)
        turnover = _ensure_datetime_index(turnover)
        turnover_summary = turnover.mean()

        if transaction_cost_bps is not None and transaction_cost_bps != 0:
            cost_per_trade = transaction_cost_bps / 10_000
            turnover_cost = turnover.reindex(returns.index).fillna(0.0)

            if isinstance(returns.columns, pd.MultiIndex) and isinstance(turnover_cost.columns, pd.MultiIndex):
                direct_cols = returns.columns.intersection(turnover_cost.columns)
                cost_df = (
                    turnover_cost[direct_cols].copy()
                    if len(direct_cols) > 0
                    else pd.DataFrame(index=returns.index)
                )
                cost_df = cost_df.reindex(columns=returns.columns).fillna(0.0)
            else:
                overlapping = returns.columns.intersection(turnover_cost.columns)
                if overlapping.empty:
                    raise ValueError(
                        "No overlapping columns between returns and turnover_cost. "
                        f"returns cols example: {list(returns.columns)[:5]} | "
                        f"turnover cols example: {list(turnover_cost.columns)[:5]}"
                    )
                cost_df = turnover_cost.reindex(columns=overlapping).fillna(0.0)
                cost_df = cost_df.reindex(columns=returns.columns).fillna(0.0)

            returns = returns - cost_df * cost_per_trade

    if isinstance(returns.columns, pd.MultiIndex):
        lvl1 = returns.columns.get_level_values(1)
        has_ls = any(str(x).lower() in {"ls", "long_short"} for x in lvl1)

        if not has_ls:
            long_short = compute_long_short_returns(returns)

            if not long_short.empty:
                if not isinstance(long_short.columns, pd.MultiIndex):
                    if all(isinstance(c, tuple) and len(c) == 2 for c in long_short.columns):
                        long_short = long_short.copy()
                        long_short.columns = pd.MultiIndex.from_tuples(long_short.columns)
                if isinstance(long_short.columns, pd.MultiIndex):
                    long_short = long_short.sort_index(axis=1)

                returns = pd.concat([returns, long_short], axis=1)
                returns = returns.sort_index(axis=1)

                if (
                    turnover_weights is not None
                    and transaction_cost_bps is not None
                    and transaction_cost_bps != 0
                ):
                    cost_per_trade = transaction_cost_bps / 10_000

                    turnover = compute_turnover_from_weights(turnover_weights)
                    turnover = _ensure_datetime_index(turnover)
                    turnover_cost = turnover.reindex(returns.index).fillna(0.0)

                    if isinstance(turnover_cost.columns, pd.MultiIndex):
                        for (model, tag) in long_short.columns:
                            model_deciles = [
                                d
                                for (m, d) in turnover_cost.columns
                                if m == model and isinstance(d, (int, np.integer))
                            ]
                            if not model_deciles:
                                continue

                            bottom = int(min(model_deciles))
                            top = int(max(model_deciles))

                            if (model, top) in turnover_cost.columns and (model, bottom) in turnover_cost.columns:
                                ls_turnover = turnover_cost[(model, top)] + turnover_cost[(model, bottom)]
                                returns[(model, tag)] = returns[(model, tag)] - (ls_turnover * cost_per_trade)

    metrics: dict[object, dict[str, float]] = {}
    cumulative: dict[object, pd.Series] = {}
    drawdowns: dict[object, pd.Series] = {}

    rf_per_period = risk_free_rate / periods_per_year

    PLOT_RET_CLIP = (-0.9, 2.0)
    returns = returns.clip(lower=PLOT_RET_CLIP[0], upper=PLOT_RET_CLIP[1])

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

        series = series.where(series > -0.999, np.nan).dropna()
        if series.empty:
            continue

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
            "avg_turnover": (
                turnover_summary[col] if turnover_summary is not None and col in turnover_summary else np.nan
            ),
        }

    metrics_df = pd.DataFrame(metrics).T.sort_index()
    cumulative_df = pd.DataFrame(cumulative).sort_index()
    drawdown_df = pd.DataFrame(drawdowns).sort_index()

    if plot_path is not None and not cumulative_df.empty:
        _plot_cumulative_returns(cumulative_df, plot_path)

    return metrics_df, cumulative_df, drawdown_df


def _format_column_label(label: object) -> str:
    """Turn a (model, decile) tuple into a readable legend label."""
    if isinstance(label, tuple):
        return " - ".join(str(part) for part in label)
    return str(label)


def _plot_cumulative_returns(cumulative: pd.DataFrame, output_path: str | Path) -> None:
    """Plot cumulative return curves and save them to disk.

    Uses iloc iteration intentionally to avoid pandas MultiIndex lookup quirks.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for i, col_label in enumerate(list(cumulative.columns)):
        series = cumulative.iloc[:, i]
        plt.plot(cumulative.index, series, label=_format_column_label(col_label))

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
    """Compute the annualized Sharpe ratio of a return series."""
    excess = returns.dropna() - risk_free_rate / periods_per_year
    volatility = excess.std(ddof=1)
    if volatility == 0 or np.isnan(volatility):
        return np.nan
    return excess.mean() / volatility * math.sqrt(periods_per_year)


def _align_model_returns(returns: pd.DataFrame, model_1: object, model_2: object) -> pd.DataFrame:
    """Align two return series on a common date index (dropping missing rows)."""
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
    """Jobson–Korkie test (optionally with Memmel correction) for Sharpe ratios."""
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

    sr1 = _compute_sharpe_ratio(
        aligned[model_1], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    sr2 = _compute_sharpe_ratio(
        aligned[model_2], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )

    cov = aligned.cov(ddof=1).iloc[0, 1]
    vol1 = aligned[model_1].std(ddof=1)
    vol2 = aligned[model_2].std(ddof=1)
    rho = cov / (vol1 * vol2) if vol1 and vol2 else np.nan

    variance = np.nan
    if not np.isnan(rho):
        variance = ((2 * (1 - rho) * sr1 * sr2) + 0.5 * (sr1**2 + sr2**2)) / n
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
    """Bootstrap a p-value for the Sharpe ratio difference between two models."""
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

    sr1 = _compute_sharpe_ratio(
        aligned[model_1], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    sr2 = _compute_sharpe_ratio(
        aligned[model_2], risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    observed_diff = sr1 - sr2

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