"""Reporting and plotting helpers for model comparison.

This module ties together the evaluation pieces of the project. It focuses on:

1) Summarizing portfolio return panels (mean/vol/Sharpe, drawdowns, cumulative curves)
2) Producing “significance tables” (Diebold–Mariano for forecast losses, Jobson–Korkie for Sharpes)
3) Saving a small set of plots that stay stable even when pandas MultiIndex columns
   are slightly messy (Index-of-tuples, non-lexsorted columns, etc.)

Most helpers are intentionally defensive: they coerce indices to datetime, reorder
MultiIndex columns into a safe layout, and avoid brittle partial slicing like ``df[model]``.
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # kept for backwards compatibility / optional use

from forecasting_metrics import diebold_mariano_test
import performance_metrics as pm
from performance_metrics import jobson_korkie_test, summarize_portfolio_performance


def _format_plot_title(
    plot_type: str,
    *,
    subject: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    log_scale: bool | None = None,
    extra: str | None = None,
) -> str:
    """Build a consistent plot title string.

    Format (with optional parts omitted when not provided):

        "<plot_type> — <subject> (from YYYY-MM-DD, log scale, <extra>)"
    """
    title = plot_type.strip()
    if subject:
        title += f" — {subject.strip()}"

    # Collect the parenthetical qualifiers (date range, scale, extra notes).
    parts: list[str] = []
    if start_date is not None:
        parts.append(f"from {pd.to_datetime(start_date).date()}")
    if log_scale is not None:
        parts.append("log scale" if log_scale else "linear scale")
    if extra:
        parts.append(extra.strip())

    if parts:
        title += f" ({', '.join(parts)})"

    return title


def _format_y_label(*, log_scale: bool) -> str:
    """Y-axis label helper for wealth index plots."""
    return "Growth of $1 (log)" if log_scale else "Growth of $1"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the index coerced to datetime."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)  # important for consistent slicing and plotting
    return out


def _ensure_model_decile_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns are a true MultiIndex (model, decile) when they look like tuples."""
    if df is None or df.empty:
        return df

    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        return df

    # If the columns are tuples like ("ridge", 1), ("ridge", 2), ... convert them properly.
    if len(cols) > 0 and all(isinstance(c, tuple) and len(c) == 2 for c in cols):
        out = df.copy()
        out.columns = pd.MultiIndex.from_tuples(cols, names=[None, None])
        return out

    # If it doesn't look like (model, decile), leave it alone.
    return df


def _reorder_decile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder (model, decile) columns into a stable, selection-safe order.

    Order rules:
    - model level sorted (as strings)
    - numeric deciles sorted ascending
    - non-numeric entries (like "LS") appended after the numeric deciles

    This sidesteps pandas MultiIndex slicing corner cases.
    """
    df = _ensure_model_decile_multiindex(df)
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return df

    models = list(pd.unique(df.columns.get_level_values(0)))

    ordered_cols: list[tuple[object, object]] = []
    for m in sorted(models, key=lambda x: str(x)):
        # All columns for this model
        sub = [c for c in df.columns if c[0] == m]

        # Numeric deciles first (1..N)
        numeric = sorted(
            [c for c in sub if isinstance(c[1], (int, np.integer))],
            key=lambda x: int(x[1]),
        )

        # Then extras such as ("model", "LS")
        other = [c for c in sub if not isinstance(c[1], (int, np.integer))]
        other = sorted(other, key=lambda x: str(x[1]))

        ordered_cols.extend(numeric + other)

    # Return a column-reordered view (keeps data unchanged).
    return df.loc[:, ordered_cols]


def _plot_cumulative(cumulative: pd.DataFrame, output_path: Path) -> None:
    """Save a simple cumulative wealth plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot each column as its own line (stringifying labels to avoid tuple legend weirdness).
    for col in cumulative.columns:
        plt.plot(cumulative.index, cumulative[col], label=str(col))

    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # tight_layout can warn on some backends; don't let it crash the run
    try:
        plt.tight_layout()
    except Exception:
        pass

    plt.savefig(output_path)
    plt.close()


def _summarize_returns_basic(
    returns: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute a lightweight performance summary for a return panel.

    This is intentionally conservative and robust:
    - coerces index to datetime
    - sorts columns if MultiIndex
    - avoids wealth math on returns <= -100% (which break cumprod/log plots)
    - does not rely on partial MultiIndex selection like ``df[model]``
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame")
    if returns.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    out = returns.copy()
    out = _ensure_datetime_index(out).sort_index()  # stable time axis is critical

    # Keep MultiIndex columns in a stable order for downstream access/plotting.
    if isinstance(out.columns, pd.MultiIndex):
        out = out.sort_index(axis=1)

    rf_per_period = risk_free_rate / periods_per_year  # convert annual rf to per-period rf

    metrics: dict[object, dict[str, float]] = {}
    cumulative: dict[object, pd.Series] = {}
    drawdowns: dict[object, pd.Series] = {}

    for col in list(out.columns):
        s = out[col].dropna()
        if s.empty:
            continue

        mean_r = float(s.mean())
        vol = float(s.std(ddof=1))  # sample std (common in performance reporting)

        # Annualized Sharpe ratio using per-period rf.
        sharpe = np.nan
        if vol > 0 and not np.isnan(vol):
            sharpe = ((mean_r - rf_per_period) / vol) * math.sqrt(periods_per_year)

        # Simple t-stat for the mean (IID approximation).
        t_stat = np.nan
        if vol > 0 and len(s) > 1:
            t_stat = mean_r / (vol / math.sqrt(len(s)))

        # Wealth process is undefined if any return is <= -100%.
        # We drop such returns for cumprod/drawdown math, but still report mean/vol above.
        s_wealth = s.where(s > -0.999).dropna()
        if s_wealth.empty:
            metrics[col] = {
                "mean_return": mean_r * periods_per_year,
                "volatility": vol * math.sqrt(periods_per_year),
                "sharpe_ratio": sharpe,
                "t_stat_mean": t_stat,
                "max_drawdown": np.nan,
            }
            continue

        cum = (1.0 + s_wealth).cumprod()          # growth of $1
        dd = cum / cum.cummax() - 1.0            # drawdown relative to running peak

        metrics[col] = {
            "mean_return": mean_r * periods_per_year,
            "volatility": vol * math.sqrt(periods_per_year),
            "sharpe_ratio": sharpe,
            "t_stat_mean": t_stat,
            "max_drawdown": float(dd.min()),
        }
        cumulative[col] = cum
        drawdowns[col] = dd

    metrics_df = pd.DataFrame(metrics).T.sort_index()
    cumulative_df = pd.DataFrame(cumulative).sort_index()
    drawdown_df = pd.DataFrame(drawdowns).sort_index()

    return metrics_df, cumulative_df, drawdown_df


def compute_dm_significance_table(
    prediction_panel: pd.DataFrame,
    *,
    realized_col: str = "realized_return",
    model_cols: Iterable[str] | None = None,
    loss: str = "squared_error",
    horizon: int = 1,
) -> pd.DataFrame:
    """Convenience wrapper around the Diebold–Mariano test implementation."""
    return diebold_mariano_test(
        prediction_panel,
        realized_col=realized_col,
        model_cols=model_cols,
        loss=loss,
        horizon=horizon,
    )


def compute_sharpe_significance_table(
    returns: pd.DataFrame,
    *,
    models: Sequence[object] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    memmel_correction: bool = True,
) -> pd.DataFrame:
    """Compute Jobson–Korkie Sharpe ratio tests for all model pairs."""
    usable = list(models) if models is not None else list(returns.columns)
    results: dict[tuple[object, object], pd.Series] = {}

    for m1, m2 in combinations(usable, 2):
        # Each test returns a Series with Sharpes, the JK stat, and a p-value.
        jk = jobson_korkie_test(
            returns,
            model_1=m1,
            model_2=m2,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            memmel_correction=memmel_correction,
        )
        results[(m1, m2)] = jk

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=["model_1", "model_2"])
    return out.sort_index()


def generate_comparison_report(
    *,
    decile_returns: pd.DataFrame,
    prediction_panel: pd.DataFrame | None = None,
    portfolio_weights: pd.DataFrame | None = None,
    factor_loadings: pd.DataFrame | None = None,
    feature_panel: pd.DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
    transaction_cost_bps: float | None = None,  # kept for API compatibility (ignored here)
    realized_col: str = "realized_return",
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    rolling_sharpe_window: int = 12,  # kept for API compatibility (ignored here)
    output_dir: str | Path = "reports",
    make_all_cumulative_plot: bool = False,
    make_ls_only_plot: bool = False,
) -> dict[str, pd.DataFrame | Path | None]:
    """Generate a compact “model comparison” report.

    The report includes:
    - basic performance metrics on the provided decile return columns
    - optional cumulative return plots
    - an optional Diebold–Mariano table if a prediction panel is provided
    - an optional Sharpe significance table computed on long/short spreads
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Make columns safe for any downstream selection/plotting.
    decile_returns = _reorder_decile_columns(decile_returns)

    # Basic performance on whatever columns were passed in (deciles and/or LS).
    metrics, cumulative, drawdowns = _summarize_returns_basic(
        decile_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    cum_plot: Path | None = None

    # Optional: plot all curves (can be crowded).
    if make_all_cumulative_plot and not cumulative.empty:
        cum_plot = output_path / "cumulative_returns.png"
        _plot_cumulative(cumulative, cum_plot)

    # Optional: plot only long/short lines (usually the plot you actually want).
    if make_ls_only_plot:
        ls = pm.compute_long_short_returns(decile_returns)
        if ls is not None and not ls.empty:
            ls = _ensure_datetime_index(ls).sort_index()

            # Guard against returns <= -100% before cumprod/log plots.
            ls = ls.where(ls > -0.999)

            cum_ls = (1.0 + ls).cumprod()
            ls_plot = output_path / "cumulative_returns_long_short.png"
            _plot_cumulative(cum_ls, ls_plot)
            cum_plot = ls_plot

    dm_table: pd.DataFrame | None = None
    if prediction_panel is not None:
        # DM compares predictive accuracy model-by-model.
        dm_table = compute_dm_significance_table(prediction_panel, realized_col=realized_col)

    sharpe_table: pd.DataFrame | None = None
    long_short = pm.compute_long_short_returns(decile_returns)
    if long_short is not None and not long_short.empty:
        # Sharpe tests are only meaningful on the long/short strategy returns.
        sharpe_table = compute_sharpe_significance_table(
            long_short,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    return {
        "performance_metrics": metrics,
        "cumulative_returns": cumulative,
        "drawdowns": drawdowns,
        "cumulative_plot": cum_plot,
        "dm_table": dm_table,
        "sharpe_table": sharpe_table,
        "rolling_sharpe": None,
        "rolling_sharpe_plot": None,
        "factor_exposures": None,
        "factor_exposures_plot": None,
        "feature_importances": None,
        "feature_importance_plot": None,
    }


def clip_realized_returns(
    predictions: pd.DataFrame,
    *,
    realized_col: str = "realized_return",
    lower: float = -0.9,
    upper: float = 2.0,
) -> pd.DataFrame:
    """Clip extreme realized returns to keep diagnostics/plots from blowing up."""
    out = predictions.copy()
    if realized_col in out.columns:
        # Clip only realized returns; predictions stay untouched.
        out[realized_col] = out[realized_col].clip(lower=lower, upper=upper)
        out = out.dropna(subset=[realized_col])  # drop rows where realized is missing
    return out


def plot_decile_cumulative_for_model(
    decile_returns: pd.DataFrame,
    *,
    model: str,
    n_deciles: int = 10,
    start_date: str | None = None,
    log_scale: bool = True,
    output_path: str | Path,
    clip_floor: float = -0.999,
) -> Path:
    """Plot cumulative wealth curves for one model’s decile portfolios."""
    if not isinstance(decile_returns, pd.DataFrame):
        raise TypeError("decile_returns must be a DataFrame")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = decile_returns.copy()

    # Convert tuple-like columns into a MultiIndex so sorting/selection is reliable.
    if not isinstance(df.columns, pd.MultiIndex):
        if len(df.columns) > 0 and all(isinstance(c, tuple) and len(c) == 2 for c in df.columns):
            df.columns = pd.MultiIndex.from_tuples(list(df.columns))
    if isinstance(df.columns, pd.MultiIndex):
        df = df.sort_index(axis=1)

    # Only plot deciles that actually exist for this model.
    cols = [(model, d) for d in range(1, n_deciles + 1) if (model, d) in df.columns]
    if not cols:
        raise KeyError(
            f"No decile columns found for model={model!r}. "
            f"Example columns: {list(df.columns)[:10]}"
        )

    sub = df.loc[:, cols].dropna(how="all")

    # Keep time axis clean and optionally restrict sample.
    sub = _ensure_datetime_index(sub).sort_index()
    if start_date is not None:
        sub = sub.loc[pd.to_datetime(start_date) :]

    # Guard: returns <= -100% break wealth math (and log plots).
    sub = sub.where(sub > clip_floor)

    cum = (1.0 + sub).cumprod()

    plt.figure(figsize=(10, 6))
    for c in cum.columns:
        plt.plot(cum.index, cum[c], label=str(c))

    plt.title(
        _format_plot_title(
            "Cumulative Returns",
            subject=f"{model} deciles (1–{n_deciles})",
            start_date=start_date,
            log_scale=log_scale,
        )
    )
    plt.xlabel("Date")
    if log_scale:
        plt.yscale("log")
    plt.ylabel(_format_y_label(log_scale=log_scale))

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_long_short_cumulative(
    decile_returns: pd.DataFrame,
    *,
    start_date: str = "1995-01-31",
    log_scale: bool = True,
    output_path: str | Path,
    bottom_decile: int = 1,
    top_decile: int | None = None,
    clip_floor: float = -0.999,
) -> Path:
    """Plot cumulative performance of top-minus-bottom spreads."""
    if not isinstance(decile_returns, pd.DataFrame):
        raise TypeError("decile_returns must be a DataFrame")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = decile_returns.copy()

    # Ensure MultiIndex columns for consistent long/short extraction.
    if not isinstance(df.columns, pd.MultiIndex):
        if len(df.columns) > 0 and all(isinstance(c, tuple) and len(c) == 2 for c in df.columns):
            df.columns = pd.MultiIndex.from_tuples(list(df.columns))
    if isinstance(df.columns, pd.MultiIndex):
        df = df.sort_index(axis=1)

    # Compute long/short return panel (model, "long_short") columns.
    ls = pm.compute_long_short_returns(df, bottom_decile=bottom_decile, top_decile=top_decile)
    if ls is None or ls.empty:
        raise ValueError("compute_long_short_returns produced an empty DataFrame")

    ls = _ensure_datetime_index(ls).sort_index()
    ls = ls.loc[pd.to_datetime(start_date) :]
    ls = ls.where(ls > clip_floor)  # guard for wealth math

    cum = (1.0 + ls).cumprod()

    plt.figure(figsize=(9, 5))
    for c in cum.columns:
        plt.plot(cum.index, cum[c], label=str(c))

    plt.title(
        _format_plot_title(
            "Long–Short Cumulative Returns",
            subject="top decile − bottom decile",
            start_date=start_date,
            log_scale=log_scale,
        )
    )
    plt.xlabel("Date")
    if log_scale:
        plt.yscale("log")
    plt.ylabel(_format_y_label(log_scale=log_scale))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def transaction_cost_stress_test(
    decile_returns: pd.DataFrame,
    decile_weights: pd.DataFrame,
    *,
    start_date: str = "1995-01-31",
    cost_bps_list: list[int] = [0, 10, 25, 50],
    periods_per_year: int = 12,
    output_dir: str | Path,
    save_plot: bool = True,
    plot_ls_models: Sequence[str] | None = None,
) -> dict[str, object]:
    """Evaluate long/short performance under different transaction cost settings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, pd.DataFrame]] = {}
    rows: list[pd.Series] = []

    for bps in cost_bps_list:
        # summarize_portfolio_performance applies turnover-based costs internally
        metrics, cumulative, drawdowns = summarize_portfolio_performance(
            decile_returns,
            turnover_weights=decile_weights,
            transaction_cost_bps=bps,
            periods_per_year=periods_per_year,
            plot_path=None,
        )

        # Restrict to the requested reporting window for plots/tables
        cumulative_post = cumulative.loc[start_date:].copy()
        drawdowns_post = drawdowns.loc[start_date:].copy()

        # Pull out LS rows (index is typically (model, "LS"))
        ls_idx = [idx for idx in metrics.index if isinstance(idx, tuple) and str(idx[1]) == "LS"]
        for idx in ls_idx:
            r = metrics.loc[idx].copy()
            r["tc_bps"] = bps
            r["portfolio"] = str(idx)
            rows.append(r)

        results[bps] = {"metrics": metrics, "cumulative": cumulative_post, "drawdowns": drawdowns_post}

    # One tidy table (easy to sort/filter/export)
    tc_summary = pd.DataFrame(rows).reset_index(drop=True)
    tc_summary = tc_summary[
        ["tc_bps", "portfolio", "mean_return", "volatility", "sharpe_ratio", "t_stat_mean", "max_drawdown"]
    ].sort_values(["tc_bps", "sharpe_ratio"], ascending=[True, False])

    plot_path: Path | None = None
    if save_plot:
        suffix = "all_models" if not plot_ls_models else "_vs_".join([str(m) for m in plot_ls_models])
        plot_path = output_dir / f"ls_with_costs_log__{suffix}.png"

        plt.figure(figsize=(10, 6))

        for bps in cost_bps_list:
            cum = results[bps]["cumulative"]

            # LS columns look like (model, "LS") when using summarize_portfolio_performance.
            ls_cols = [c for c in cum.columns if isinstance(c, tuple) and str(c[1]) == "LS"]

            # Optional filter so the plot doesn't get overcrowded.
            if plot_ls_models is not None:
                wanted = set(map(str, plot_ls_models))
                ls_cols = [c for c in ls_cols if str(c[0]) in wanted]

            for col in ls_cols:
                plt.plot(cum.index, cum[col], label=f"{col[0]} LS ({bps} bps)")

        plt.yscale("log")
        plt.title(
            _format_plot_title(
                "Long–Short Cumulative Returns",
                subject="with transaction costs",
                start_date=start_date,
                log_scale=True,
                extra=f"costs: {', '.join(map(str, cost_bps_list))} bps",
            )
        )
        plt.ylabel(_format_y_label(log_scale=True))
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    return {"tc_summary": tc_summary, "results": results, "plot_path": plot_path}


def run_plot_suite(
    *,
    predictions: pd.DataFrame,
    decile_returns: pd.DataFrame,
    decile_weights: pd.DataFrame,
    realized_col: str = "realized_return",
    model_cols: Sequence[str] | None = None,
    clip_lower: float = -0.9,
    clip_upper: float = 2.0,
    start_date: str = "1995-01-31",
    cost_bps_list: list[int] = [0, 10, 25, 50],
    example_model: str | None = None,
    output_dir: str | Path = "reports",
) -> dict[str, object]:
    """Run the “standard plot suite” and write outputs into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clip realized returns only (predictions untouched) to keep plots stable.
    preds_clean = clip_realized_returns(
        predictions, realized_col=realized_col, lower=clip_lower, upper=clip_upper
    )

    # Auto-detect model columns (numeric forecasts) when not provided.
    if model_cols is None:
        model_cols = [c for c in preds_clean.columns if c != realized_col]
        model_cols = [c for c in model_cols if preds_clean[c].dtype.kind in "if"]

    # Choose an example model for the decile plot.
    if example_model is None:
        example_model = model_cols[0] if len(model_cols) else None

    # Caller may already pass “clean” decile inputs; we keep them as-is here.
    decile_returns_clean = decile_returns.copy()
    decile_weights_clean = decile_weights.copy()

    paths: dict[str, Path | None] = {}

    if example_model is not None:
        paths["decile_cum_plot"] = plot_decile_cumulative_for_model(
            decile_returns_clean,
            model=str(example_model),
            start_date=start_date,
            output_path=output_dir / f"cumulative_returns_{example_model}.png",
        )

    # Multi-model LS plot.
    paths["long_short_plot"] = plot_long_short_cumulative(
        decile_returns_clean,
        start_date=start_date,
        output_path=output_dir / "ls_cumulative_log__all_models.png",
    )

    # Stress test cost assumptions; also produces a plot by default.
    tc_out = transaction_cost_stress_test(
        decile_returns_clean,
        decile_weights_clean,
        start_date=start_date,
        cost_bps_list=list(cost_bps_list),
        output_dir=output_dir,
        save_plot=True,
        plot_ls_models=["ols_pred", "ridge_rank"],  # keep plot readable by default
    )

    # Persist the summary table as CSV (handy for quick review / spreadsheets).
    tc_summary: pd.DataFrame = tc_out["tc_summary"]
    tc_path = output_dir / "tc_summary.csv"
    tc_summary.to_csv(tc_path, index=False)

    paths["tc_plot"] = tc_out["plot_path"]
    paths["tc_summary_csv"] = tc_path

    return {
        "predictions_clean": preds_clean,
        "model_cols": list(model_cols),
        "example_model": example_model,
        "paths": paths,
        "tc_summary": tc_summary,
        "tc_results": tc_out["results"],
    }


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a DataFrame index to datetime and sort."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _sanitize_decile_returns_for_ls(decile_returns: pd.DataFrame) -> pd.DataFrame:
    """Prepare a decile-return frame for long/short computation."""
    if not isinstance(decile_returns, pd.DataFrame) or decile_returns.empty:
        return decile_returns

    out = decile_returns.copy()
    cols = out.columns

    # Convert Index-of-tuples -> MultiIndex when possible.
    if not isinstance(cols, pd.MultiIndex):
        if len(cols) > 0 and all(isinstance(c, tuple) and len(c) == 2 for c in cols):
            out.columns = pd.MultiIndex.from_tuples(cols)
        else:
            return out

    # Keep only numeric deciles (drop "LS" so top/bottom deciles are unambiguous).
    keep_cols = [c for c in out.columns if isinstance(c[1], (int, np.integer))]
    out = out.loc[:, keep_cols].copy()

    # Stable sort so top/bottom selection behaves consistently.
    return out.sort_index(axis=1)


def _extract_ls_series(ls_df: pd.DataFrame, model_name: str) -> pd.Series:
    """Extract one model’s long/short series from a long/short return DataFrame."""
    if ls_df.empty:
        return pd.Series(dtype=float)

    # Preferred format: MultiIndex columns (model, "long_short") or (model, "LS")
    if isinstance(ls_df.columns, pd.MultiIndex) and ls_df.columns.nlevels >= 2:
        lvl0 = ls_df.columns.get_level_values(0)
        lvl1 = ls_df.columns.get_level_values(1).astype(str)

        mask = (lvl0 == model_name) & (lvl1.isin(["long_short", "LS"]))
        cols = ls_df.columns[mask]
        if len(cols) == 0:
            return pd.Series(dtype=float)

        col = cols[0]  # if multiple, just pick the first
        s = ls_df[col].copy()
        s.name = str(col)
        return s

    # Fallback if columns are single-level strings.
    for c in [model_name, f"{model_name}_long_short", f"{model_name}_LS"]:
        if c in ls_df.columns:
            s = ls_df[c].copy()
            s.name = c
            return s

    return pd.Series(dtype=float)


def plot_long_short_rank_vs_raw(
    *,
    decile_returns_raw: pd.DataFrame,
    decile_returns_rank: pd.DataFrame,
    raw_model: str = "ridge_pred",
    rank_model: str = "ridge_rank",
    start_date: str | pd.Timestamp | None = "1995-01-31",
    log_scale: bool = True,
    output_path: str | Path = "reports/ls_rank_vs_raw.png",
    title: str | None = None,
    bottom_decile: int = 1,
    top_decile: int | None = None,
    clip_floor: float = -0.999,
) -> dict[str, pd.Series | pd.DataFrame | Path]:
    """Plot long/short cumulative returns for a rank signal vs a raw prediction."""
    if not isinstance(decile_returns_raw, pd.DataFrame) or not isinstance(decile_returns_rank, pd.DataFrame):
        raise TypeError("decile_returns_raw and decile_returns_rank must both be DataFrames")

    # Helper: convert Index-of-tuples to MultiIndex and sort (prevents KeyErrors).
    def _ensure_mi_sorted(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not isinstance(out.columns, pd.MultiIndex):
            if len(out.columns) > 0 and all(isinstance(c, tuple) and len(c) == 2 for c in out.columns):
                out.columns = pd.MultiIndex.from_tuples(list(out.columns))
        if isinstance(out.columns, pd.MultiIndex):
            out = out.sort_index(axis=1)
        return out

    raw_clean = _ensure_mi_sorted(decile_returns_raw)
    rank_clean = _ensure_mi_sorted(decile_returns_rank)

    # Compute long/short for each set of decile returns.
    ls_raw_df = pm.compute_long_short_returns(raw_clean, bottom_decile=bottom_decile, top_decile=top_decile)
    ls_rank_df = pm.compute_long_short_returns(rank_clean, bottom_decile=bottom_decile, top_decile=top_decile)

    if ls_raw_df is None or ls_raw_df.empty:
        raise ValueError(f"RAW long-short is empty for raw_model={raw_model}")
    if ls_rank_df is None or ls_rank_df.empty:
        raise ValueError(f"RANK long-short is empty for rank_model={rank_model}")

    # Standardize indices before alignment.
    ls_raw_df = _ensure_datetime_index(ls_raw_df).sort_index()
    ls_rank_df = _ensure_datetime_index(ls_rank_df).sort_index()

    raw_key = (raw_model, "long_short")
    rank_key = (rank_model, "long_short")

    if raw_key not in ls_raw_df.columns:
        raise KeyError(f"Could not find RAW column {raw_key} in ls_raw_df.columns")
    if rank_key not in ls_rank_df.columns:
        raise KeyError(f"Could not find RANK column {rank_key} in ls_rank_df.columns")

    # Rename for a clean legend.
    s_raw = ls_raw_df[raw_key].rename("RAW")
    s_rank = ls_rank_df[rank_key].rename("RANK")

    # Restrict to the desired sample window.
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        s_raw = s_raw.loc[start_date:]
        s_rank = s_rank.loc[start_date:]

    # Align and guard wealth math.
    ls = pd.concat([s_raw, s_rank], axis=1).sort_index()
    ls = ls.where(ls > clip_floor).dropna(how="all")
    cum = (1.0 + ls).cumprod()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    if "RAW" in cum.columns:
        plt.plot(cum.index, cum["RAW"], label=f"{raw_model} (raw)")
    if "RANK" in cum.columns:
        plt.plot(cum.index, cum["RANK"], label=f"{rank_model} (rank)")

    if log_scale:
        plt.yscale("log")

    if title is None:
        title = _format_plot_title(
            "Long–Short Cumulative Returns",
            subject=f"{raw_model} (raw) vs {rank_model} (rank)",
            start_date=start_date,
            log_scale=log_scale,
        )

    plt.title(title)
    plt.ylabel(_format_y_label(log_scale=log_scale))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {"ls_raw": s_raw, "ls_rank": s_rank, "ls": ls, "cum": cum, "output_path": output_path}


__all__ = [
    "generate_comparison_report",
    "plot_decile_cumulative_for_model",
    "plot_long_short_cumulative",
    "transaction_cost_stress_test",
    "clip_realized_returns",
    "run_plot_suite",
    "compute_dm_significance_table",
    "compute_sharpe_significance_table",
    "plot_long_short_rank_vs_raw",
]