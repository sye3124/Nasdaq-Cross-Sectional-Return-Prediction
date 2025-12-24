"""Portfolio construction helpers for decile-sorted strategies."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _validate_multiindex(df: pd.DataFrame, name: str) -> None:
    """Validate that the DataFrame uses a MultiIndex with levels ('ticker', 'date')."""
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ticker", "date"]:
        raise ValueError(f"{name} must use a MultiIndex with levels ('ticker', 'date').")


def _assign_deciles(values: pd.Series, n_deciles: int) -> pd.Series:
    """Assign decile ranks (1 to n_deciles) to the input values."""
    x = values.astype(float)
    # If there are fewer unique values than deciles, use rank-based binning
    if x.nunique(dropna=True) < n_deciles:
        ranks = x.rank(method="average")
        return pd.cut(ranks, bins=n_deciles, labels=False, include_lowest=True) + 1

    # Standard quantile-based binning
    return pd.qcut(x.rank(method="first"), q=n_deciles, labels=False) + 1


def compute_decile_portfolio_returns(
    panel: pd.DataFrame,
    *,
    model_cols: Sequence[str] | None = None,
    return_col: str = "realized_return",
    weight_col: str | None = None,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Form decile portfolios and compute their next-period returns."""

    # Validate inputs
    _validate_multiindex(panel, "panel")
    if n_deciles <= 0:
        raise ValueError("n_deciles must be positive.")

    # Determine model columns
    if model_cols is None:
        excluded = {return_col}
        if weight_col is not None:
            excluded.add(weight_col)
        model_cols = [col for col in panel.columns if col not in excluded]

    results: dict[str, pd.DataFrame] = {}

    # Process each model column
    for model in model_cols:
        if model not in panel.columns:
            raise KeyError(f"Missing model column '{model}'.")

        # Subset relevant columns and drop rows with missing model signals
        subset_cols: Iterable[str] = [model, return_col]
        if weight_col is not None:
            subset_cols = list(subset_cols) + [weight_col]

        # Prepare DataFrame for the current model
        df_model = panel[list(subset_cols)].dropna(subset=[model])
        if df_model.empty:
            continue
        
        # Reindex to ensure correct MultiIndex structure
        df_model = df_model.copy()
        df_model.index = pd.MultiIndex.from_arrays(
            [
                df_model.index.get_level_values("ticker"),
                pd.to_datetime(df_model.index.get_level_values("date")),
            ],
            names=["ticker", "date"],
        )

        # Assign deciles within each date based on the model signal
        df_model["decile"] = df_model.groupby(level="date")[model].transform(
            _assign_deciles, n_deciles
        )

        # Compute returns for each (date, decile)
        gkey = [pd.Grouper(level="date"), "decile"]

        # Always drop missing realized returns (matches old behavior)
        df_model = df_model.dropna(subset=[return_col])
        if df_model.empty:
            continue

        if weight_col is None:
            # Equal-weight: mean return within each (date, decile)
            returns = df_model.groupby(gkey)[return_col].mean()
        else:
            # Value-weight: sum(w * r) / sum(w) within each (date, decile)
            w = df_model[weight_col].astype(float).fillna(0.0)
            r = df_model[return_col].astype(float)

            tmp = pd.DataFrame(
                {"wr": w * r, "w": w, "decile": df_model["decile"]},
                index=df_model.index,
            )

            num = tmp.groupby(gkey)["wr"].sum()
            den = tmp.groupby(gkey)["w"].sum()

            # Avoid division by zero -> NaN
            returns = num / den.replace(0.0, np.nan)

        # Reshape to have deciles as columns
        returns = returns.unstack("decile")
        # Long-short spread only when both bottom and top deciles are present
        if 1 in returns.columns and n_deciles in returns.columns:
            returns["LS"] = returns[n_deciles] - returns[1]
        else:
            returns["LS"] = np.nan
        results[model] = returns

    # Combine results from all models
    if not results:
        empty_index = pd.Index([], name="date")
        empty_cols = pd.MultiIndex.from_arrays([[], []], names=[None, None])
        return pd.DataFrame(index=empty_index, columns=empty_cols)

    combined = pd.concat(results, axis=1)
    # Sort columns by model and decile
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
    """Return per-ticker portfolio weights for each decile and model."""

    # Validate inputs
    _validate_multiindex(panel, "panel")
    if n_deciles <= 0:
        raise ValueError("n_deciles must be positive.")

    # Determine model columns
    if model_cols is None:
        excluded = set()
        if weight_col is not None:
            excluded.add(weight_col)
        model_cols = [col for col in panel.columns if col not in excluded]

    results: dict[str, pd.DataFrame] = {}

    # Process each model column
    for model in model_cols:
        if model not in panel.columns:
            raise KeyError(f"Missing model column '{model}'.")

        # Subset relevant columns and drop rows with missing model signals
        subset_cols: Iterable[str] = [model]
        if weight_col is not None:
            subset_cols = list(subset_cols) + [weight_col]

        # Prepare DataFrame for the current model
        df_model = panel[list(subset_cols)].dropna(subset=[model])
        if df_model.empty:
            continue
        
        # Reindex to ensure correct MultiIndex structure
        df_model = df_model.copy()
        df_model.index = pd.MultiIndex.from_arrays(
            [
                df_model.index.get_level_values("ticker"),
                pd.to_datetime(df_model.index.get_level_values("date")),
            ],
            names=["ticker", "date"],
        )

        # Assign deciles within each date based on the model signal
        df_model["decile"] = df_model.groupby(level="date")[model].transform(
            _assign_deciles, n_deciles
        )

        # Compute weights for each (date, decile)
        if weight_col is None:
            df_model["weight"] = df_model.groupby([pd.Grouper(level="date"), "decile"])[model].transform(
                lambda s: 1.0 / len(s)
            )
        else:
            if weight_col not in df_model.columns:
                raise KeyError(f"Missing weight column '{weight_col}'.")

            def _normalize(group: pd.Series) -> pd.Series:
                total = group.sum()
                if total == 0:
                    return pd.Series(1.0 / len(group), index=group.index)
                return group / total

            df_model["weight"] = df_model.groupby([pd.Grouper(level="date"), "decile"])[weight_col].transform(
                _normalize
            )

        # Reshape to have deciles as columns
        weights = (
            df_model[["weight", "decile"]]
            .reset_index()
            .pivot_table(index=["ticker", "date"], columns="decile", values="weight", fill_value=0.0)
            .sort_index()
        )

        results[model] = weights

    # Combine results from all models
    if not results:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        empty_cols = pd.MultiIndex.from_arrays([[], []], names=[None, None])
        return pd.DataFrame(index=empty_index, columns=empty_cols)

    combined = pd.concat(results, axis=1)
    combined = combined.sort_index(axis=1, level=[0, 1])
    combined.columns = combined.columns.set_names([None, None])

    return combined.sort_index()

__all__ = ["compute_decile_portfolio_returns", "compute_decile_portfolio_weights"]
