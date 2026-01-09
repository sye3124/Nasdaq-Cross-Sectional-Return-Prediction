"""Portfolio construction helpers for decile-sorted strategies.

Given a panel indexed by ``('ticker', 'date')`` that contains one or more model
signals (predictions, ranks, scores, etc.) and realized returns, this module can:

- assign each ticker to a decile within each date based on a model signal
- compute decile portfolio returns (equal-weighted or value-weighted)
- compute per-ticker portfolio weights for each decile (for turnover/costs)

Outputs are organized as DataFrames with MultiIndex columns (model, decile),
where decile is usually 1..N plus an optional "LS" (top-minus-bottom spread).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _validate_multiindex(df: pd.DataFrame, name: str) -> None:
    """Ensure an input frame is indexed by ('ticker', 'date')."""
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ticker", "date"]:
        raise ValueError(f"{name} must use a MultiIndex with levels ('ticker', 'date').")


def _assign_deciles(values: pd.Series, n_deciles: int) -> pd.Series:
    """Assign decile labels 1..n_deciles within a cross-section.

    Uses quantile binning when possible. If the cross-section doesn't have
    enough unique values to support qcut cleanly (common when signals are
    discrete or heavily tied), we fall back to binning on average ranks.
    """
    x = values.astype(float)

    # If there are too many ties, qcut can error or create empty bins.
    if x.nunique(dropna=True) < n_deciles:
        ranks = x.rank(method="average")
        return pd.cut(ranks, bins=n_deciles, labels=False, include_lowest=True) + 1

    # Rank first so ties are deterministically broken before quantile binning.
    return pd.qcut(x.rank(method="first"), q=n_deciles, labels=False) + 1


def compute_decile_portfolio_returns(
    panel: pd.DataFrame,
    *,
    model_cols: Sequence[str] | None = None,
    return_col: str = "realized_return",
    weight_col: str | None = None,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Form decile portfolios for each model and compute their returns.

    Parameters
    ----------
    panel
        Panel indexed by ``('ticker', 'date')``. Must include the model signal
        columns and ``return_col``. If ``weight_col`` is provided, value-weighted
        returns are computed using that column.
    model_cols
        Which columns to treat as model signals. If None, uses all columns except
        ``return_col`` and (optionally) ``weight_col``.
    return_col
        Column containing realized returns to aggregate within each decile.
    weight_col
        Optional column to use for value-weighting (e.g. market cap or dollar volume).
        If None, portfolios are equal-weighted within each decile.
    n_deciles
        Number of buckets to form (default: 10).

    Returns
    -------
    pd.DataFrame
        Indexed by date with MultiIndex columns (model, decile). Includes an "LS"
        column per model equal to top-minus-bottom when both legs exist.
    """
    _validate_multiindex(panel, "panel")
    if n_deciles <= 0:
        raise ValueError("n_deciles must be positive.")

    # If the caller doesn't specify model columns, infer them defensively.
    if model_cols is None:
        excluded: set[str] = set()
        if weight_col is not None:
            excluded.add(weight_col)
        excluded.add(return_col)
        model_cols = [col for col in panel.columns if col not in excluded]

    results: dict[str, pd.DataFrame] = {}

    for model in model_cols:
        if model not in panel.columns:
            raise KeyError(f"Missing model column '{model}'.")

        # Work with only the columns we need for this model.
        subset_cols: Iterable[str] = [model, return_col]
        if weight_col is not None:
            subset_cols = list(subset_cols) + [weight_col]

        # We can’t assign deciles without the model signal.
        df_model = panel[list(subset_cols)].dropna(subset=[model])
        if df_model.empty:
            continue

        # Normalize index types (esp. date) so groupby/align behaves consistently.
        df_model = df_model.copy()
        df_model.index = pd.MultiIndex.from_arrays(
            [
                df_model.index.get_level_values("ticker"),
                pd.to_datetime(df_model.index.get_level_values("date")),
            ],
            names=["ticker", "date"],
        )

        # Assign deciles within each date based on the model signal.
        df_model["decile"] = df_model.groupby(level="date")[model].transform(
            _assign_deciles, n_deciles
        )

        # Always require realized returns for return aggregation.
        df_model = df_model.dropna(subset=[return_col])
        if df_model.empty:
            continue

        gkey = [pd.Grouper(level="date"), "decile"]

        if weight_col is None:
            # Equal-weighted: mean return per (date, decile).
            returns = df_model.groupby(gkey)[return_col].mean()
        else:
            # Value-weighted: sum(w * r) / sum(w) per (date, decile).
            w = df_model[weight_col].astype(float).fillna(0.0)
            r = df_model[return_col].astype(float)

            tmp = pd.DataFrame(
                {"wr": w * r, "w": w, "decile": df_model["decile"]},
                index=df_model.index,
            )

            num = tmp.groupby(gkey)["wr"].sum()
            den = tmp.groupby(gkey)["w"].sum()

            # Avoid division by zero in periods where all weights are zero.
            returns = num / den.replace(0.0, np.nan)

        # Reshape to date x decile.
        returns = returns.unstack("decile")

        # Add long/short spread when both legs exist.
        if 1 in returns.columns and n_deciles in returns.columns:
            returns["LS"] = returns[n_deciles] - returns[1]
        else:
            returns["LS"] = np.nan

        results[model] = returns

    if not results:
        empty_index = pd.Index([], name="date")
        empty_cols = pd.MultiIndex.from_arrays([[], []], names=[None, None])
        return pd.DataFrame(index=empty_index, columns=empty_cols)

    # Combine model blocks into MultiIndex columns (model, decile).
    combined = pd.concat(results, axis=1)
    combined = combined.sort_index(axis=1, level=[0, 1])
    combined.columns = combined.columns.set_names([None, None])

    return combined.sort_index()


def compute_decile_portfolio_weights(
    panel: pd.DataFrame,
    *,
    model_cols: Sequence[str] | None = None,
    weight_col: str | None = None,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Compute per-ticker weights for each (model, decile) portfolio.

    The output is designed for turnover and transaction-cost calculations: it is
    indexed by (ticker, date) and contains MultiIndex columns (model, decile)
    with weights that sum to 1 within each (date, model, decile).

    Parameters
    ----------
    panel
        Panel indexed by ``('ticker', 'date')`` containing model signals and an
        optional weighting variable.
    model_cols
        Which columns to treat as model signals. If None, uses all columns except
        ``weight_col`` (when provided).
    weight_col
        Optional column to use for value-weighting. If None, weights are equal
        within each decile.
    n_deciles
        Number of buckets to form (default: 10).

    Returns
    -------
    pd.DataFrame
        Weights indexed by ``('ticker', 'date')`` with columns (model, decile).
    """
    _validate_multiindex(panel, "panel")
    if n_deciles <= 0:
        raise ValueError("n_deciles must be positive.")

    if model_cols is None:
        excluded: set[str] = set()
        if weight_col is not None:
            excluded.add(weight_col)
        model_cols = [col for col in panel.columns if col not in excluded]

    results: dict[str, pd.DataFrame] = {}

    for model in model_cols:
        if model not in panel.columns:
            raise KeyError(f"Missing model column '{model}'.")

        subset_cols: Iterable[str] = [model]
        if weight_col is not None:
            subset_cols = list(subset_cols) + [weight_col]

        df_model = panel[list(subset_cols)].dropna(subset=[model])
        if df_model.empty:
            continue

        df_model = df_model.copy()
        df_model.index = pd.MultiIndex.from_arrays(
            [
                df_model.index.get_level_values("ticker"),
                pd.to_datetime(df_model.index.get_level_values("date")),
            ],
            names=["ticker", "date"],
        )

        # Decile assignment is driven by the model signal (not by weights).
        df_model["decile"] = df_model.groupby(level="date")[model].transform(
            _assign_deciles, n_deciles
        )

        if weight_col is None:
            # Equal weights within each (date, decile).
            df_model["weight"] = df_model.groupby([pd.Grouper(level="date"), "decile"])[model].transform(
                lambda s: 1.0 / len(s)
            )
        else:
            if weight_col not in df_model.columns:
                raise KeyError(f"Missing weight column '{weight_col}'.")

            def _normalize(group: pd.Series) -> pd.Series:
                total = group.sum()
                if total == 0:
                    # If all weights are zero, fall back to equal-weight.
                    return pd.Series(1.0 / len(group), index=group.index)
                return group / total

            # Normalize weights within each (date, decile).
            df_model["weight"] = df_model.groupby([pd.Grouper(level="date"), "decile"])[weight_col].transform(
                _normalize
            )

        # Pivot to a wide format: (ticker, date) x decile, then attach model level via concat.
        weights = (
            df_model[["weight", "decile"]]
            .reset_index()
            .pivot_table(index=["ticker", "date"], columns="decile", values="weight", fill_value=0.0)
            .sort_index()
        )

        results[model] = weights

    if not results:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        empty_cols = pd.MultiIndex.from_arrays([[], []], names=[None, None])
        return pd.DataFrame(index=empty_index, columns=empty_cols)

    combined = pd.concat(results, axis=1)
    combined = combined.sort_index(axis=1, level=[0, 1])
    combined.columns = combined.columns.set_names([None, None])

    return combined.sort_index()


__all__ = [
    "compute_decile_portfolio_returns", 
    "compute_decile_portfolio_weights"
]