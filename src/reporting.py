import math
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd

# Use non-interactive backend for saving plots
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Custom metric utilities
from forecasting_metrics import diebold_mariano_test
from performance_metrics import (
    compute_long_short_returns,
    jobson_korkie_test,
    summarize_portfolio_performance,
)


def _format_column_label(label: object) -> str:
    # Convert tuple labels to readable strings
    if isinstance(label, tuple):
        return " - ".join(str(part) for part in label)
    return str(label)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure index is a datetime index for time-series operations
    result = df.copy()
    result.index = pd.to_datetime(result.index)
    return result


def _compute_rolling_sharpe(
    returns: pd.DataFrame,
    *,
    window: int,
    risk_free_rate: float,
    periods_per_year: int,
) -> pd.DataFrame:
    # Compute rolling Sharpe ratio over the specified window
    if window <= 0:
        raise ValueError("window must be positive")

    rf_per_period = risk_free_rate / periods_per_year

    def _calc(series: pd.Series) -> float:
        # Sharpe = mean(excess) / std(excess) annualized
        excess = series - rf_per_period
        volatility = excess.std(ddof=1)
        if volatility == 0 or np.isnan(volatility):
            return np.nan
        return excess.mean() / volatility * math.sqrt(periods_per_year)

    ordered = _ensure_datetime_index(returns)
    return ordered.sort_index().rolling(window=window).apply(_calc, raw=False)


def _plot_lines(df: pd.DataFrame, *, output_path: Path, title: str, ylabel: str) -> None:
    # Generic line plot helper
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=_format_column_label(col))
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compute_model_performance_table(
    decile_returns: pd.DataFrame,
    *,
    turnover_weights: pd.DataFrame | None = None,
    transaction_cost_bps: float | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Summarize performance tables and plots for decile portfolios."""

    plot_path = None
    if output_dir is not None:
        # Output location for cumulative return plot
        output_dir = Path(output_dir)
        plot_path = output_dir / "cumulative_returns.png"

    # Compute standard performance statistics
    metrics, cumulative, drawdowns = summarize_portfolio_performance(
        decile_returns,
        turnover_weights=turnover_weights,
        transaction_cost_bps=transaction_cost_bps,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        plot_path=plot_path,
    )

    return {
        "metrics": metrics,
        "cumulative": cumulative,
        "drawdowns": drawdowns,
        "cumulative_plot": plot_path,
    }


def compute_dm_significance_table(
    prediction_panel: pd.DataFrame,
    *,
    realized_col: str = "realized_return",
    model_cols: Iterable[str] | None = None,
    loss: str = "squared_error",
    horizon: int = 1,
) -> pd.DataFrame:
    """Run Diebold-Mariano tests comparing forecast accuracy across models."""
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
    """Run Jobson-Korkie (with optional Memmel correction) pairwise Sharpe tests."""

    usable_models = list(models) if models is not None else list(returns.columns)
    results: dict[tuple[object, object], pd.Series] = {}

    for model_1, model_2 in combinations(sorted(usable_models), 2):
        # Compute statistical test comparing Sharpe ratios
        stats = jobson_korkie_test(
            returns,
            model_1=model_1,
            model_2=model_2,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            memmel_correction=memmel_correction,
        )
        results[(model_1, model_2)] = stats

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results).T
    result_df.index = pd.MultiIndex.from_tuples(result_df.index, names=["model_1", "model_2"])
    return result_df.sort_index()


def plot_rolling_sharpe(
    returns: pd.DataFrame,
    *,
    window: int = 12,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    output_path: str | Path,
) -> pd.DataFrame:
    """Compute and save rolling Sharpe ratio plot."""
    rolling = _compute_rolling_sharpe(
        returns,
        window=window,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    _plot_lines(rolling, output_path=Path(output_path), title="Rolling Sharpe", ylabel="Sharpe Ratio")
    return rolling


def compute_portfolio_factor_exposures(
    weights: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    *,
    factor_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate factor exposures for each portfolio based on its weights."""

    # Ensure loadings index is (ticker, date)
    loadings = factor_loadings.copy()
    loadings.index = pd.MultiIndex.from_arrays(
        [
            loadings.index.get_level_values("ticker"),
            pd.to_datetime(loadings.index.get_level_values("date")),
        ],
        names=["ticker", "date"],
    )

    # Auto-select factors named beta_*
    candidate_factors = factor_cols or [col for col in loadings.columns if col.startswith("beta_")]
    candidate_factors = [c for c in candidate_factors if c in loadings.columns]

    frames: list[pd.DataFrame] = []
    for col in weights.columns:
        # Each column is a portfolio's weights over time
        series = weights[col].rename("weight")

        # Align weights with factor loadings for each ticker/date
        aligned = loadings.join(series, how="inner")
        if aligned.empty:
            continue

        # Weighted factor exposure for each date
        weighted = aligned[candidate_factors].multiply(aligned["weight"], axis=0)
        exposure = weighted.groupby(level="date").sum()

        # MultiIndex columns: (portfolio, factor)
        exposure.columns = pd.MultiIndex.from_product([[col], candidate_factors])
        frames.append(exposure)

    if not frames:
        # Return empty structure if nothing matched
        empty_cols = pd.MultiIndex.from_arrays([[], []], names=[None, "factor"])
        return pd.DataFrame(columns=empty_cols)

    return pd.concat(frames, axis=1).sort_index()


def plot_factor_exposures(exposures: pd.DataFrame, *, output_path: str | Path) -> None:
    """Plot time-series of factor exposures for each portfolio."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    if isinstance(exposures.columns, pd.MultiIndex):
        # Separate curves for each (portfolio, factor)
        for portfolio in exposures.columns.get_level_values(0).unique():
            for factor in exposures[portfolio].columns:
                label = f"{_format_column_label(portfolio)} - {factor}"
                plt.plot(exposures.index, exposures[portfolio][factor], label=label)
    else:
        # Simple case
        for col in exposures.columns:
            plt.plot(exposures.index, exposures[col], label=_format_column_label(col))

    plt.title("Portfolio Factor Exposures")
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compute_feature_importance(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "next_return",
    model_factory=None,
) -> pd.Series:
    """Estimate feature importance using a tree-based model (default: RandomForest)."""

    df = panel.copy()
    required = list(feature_cols) + [target_col]

    # Drop rows missing features or target
    df = df.dropna(subset=required)
    if df.empty:
        return pd.Series(dtype=float)

    # Default model factory if none provided
    factory = model_factory or (
        lambda: RandomForestRegressor(
            n_estimators=300,
            random_state=0,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=1,
        )
    )
    model = factory()
    model.fit(df[feature_cols].to_numpy(), df[target_col].to_numpy())

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.Series(dtype=float)

    return pd.Series(importances, index=feature_cols).sort_values(ascending=False)


def plot_feature_importance(importances: pd.Series, *, output_path: str | Path) -> None:
    """Plot horizontal bar chart of feature importances."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sorted_imp = importances.sort_values(ascending=True)
    sorted_imp.plot(kind="barh")
    plt.title("Feature Importance (Tree-Based Model)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_comparison_report(
    *,
    decile_returns: pd.DataFrame,
    prediction_panel: pd.DataFrame | None = None,
    portfolio_weights: pd.DataFrame | None = None,
    factor_loadings: pd.DataFrame | None = None,
    feature_panel: pd.DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
    transaction_cost_bps: float | None = None,
    realized_col: str = "realized_return",
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
    rolling_sharpe_window: int = 12,
    output_dir: str | Path = "reports",
) -> dict[str, pd.DataFrame | Path | None]:
    """Produce tables and plots combining all comparison utilities."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Performance metrics for decile portfolios
    performance = compute_model_performance_table(
        decile_returns,
        turnover_weights=portfolio_weights,
        transaction_cost_bps=transaction_cost_bps,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        output_dir=output_path,
    )

    # Diebold–Mariano test results
    dm_table = None
    if prediction_panel is not None:
        dm_table = compute_dm_significance_table(
            prediction_panel,
            realized_col=realized_col,
        )

    # Compute long-short spreads and test Sharpe significance
    sharpe_table = None
    long_short = compute_long_short_returns(decile_returns)
    if not long_short.empty:
        sharpe_table = compute_sharpe_significance_table(
            long_short,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        rolling_sharpe_path = output_path / "rolling_sharpe.png"
        rolling_sharpe = plot_rolling_sharpe(
            long_short,
            window=rolling_sharpe_window,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            output_path=rolling_sharpe_path,
        )
    else:
        rolling_sharpe_path = None
        rolling_sharpe = pd.DataFrame()

    # Factor exposures (if available)
    exposure_table = None
    exposure_plot = None
    if portfolio_weights is not None and factor_loadings is not None:
        exposure_table = compute_portfolio_factor_exposures(portfolio_weights, factor_loadings)
        exposure_plot = output_path / "factor_exposures.png"
        if not exposure_table.empty:
            plot_factor_exposures(exposure_table, output_path=exposure_plot)

    # Feature importance (if provided)
    feature_importances = None
    feature_plot = None
    if feature_panel is not None and feature_cols is not None:
        feature_importances = compute_feature_importance(feature_panel, feature_cols)
        feature_plot = output_path / "feature_importance.png"
        if not feature_importances.empty:
            plot_feature_importance(feature_importances, output_path=feature_plot)

    return {
        "performance_metrics": performance["metrics"],
        "cumulative_returns": performance["cumulative"],
        "drawdowns": performance["drawdowns"],
        "cumulative_plot": performance["cumulative_plot"],
        "dm_table": dm_table,
        "sharpe_table": sharpe_table,
        "rolling_sharpe": rolling_sharpe,
        "rolling_sharpe_plot": rolling_sharpe_path,
        "factor_exposures": exposure_table,
        "factor_exposures_plot": exposure_plot,
        "feature_importances": feature_importances,
        "feature_importance_plot": feature_plot,
    }


__all__ = [
    "compute_model_performance_table",
    "compute_dm_significance_table",
    "compute_sharpe_significance_table",
    "compute_portfolio_factor_exposures",
    "compute_feature_importance",
    "plot_feature_importance",
    "plot_factor_exposures",
    "plot_rolling_sharpe",
    "generate_comparison_report",
]
