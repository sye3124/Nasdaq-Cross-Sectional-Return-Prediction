"""Portfolio construction helpers for decile-sorted strategies."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _validate_multiindex(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ticker", "date"]:
        raise ValueError(f"{name} must use a MultiIndex with levels ('ticker', 'date').")


def _assign_deciles(values: pd.Series, n_deciles: int) -> pd.Series:
    ranks = values.rank(pct=True, method="average")
    deciles = np.ceil(ranks * n_deciles).astype(int)
    deciles = deciles.clip(1, n_deciles)
    return deciles


def _compute_group_return(group: pd.DataFrame, return_col: str, weight_col: str | None) -> float | np.nan:
    usable = group.dropna(subset=[return_col])
    if usable.empty:
        return np.nan

    if weight_col is None:
        weights = pd.Series(1.0 / len(usable), index=usable.index)
    else:
        if weight_col not in usable.columns:
            raise KeyError(f"Missing weight column '{weight_col}'.")
        raw_weights = usable[weight_col].astype(float)
        total = raw_weights.sum()
        if total == 0:
            weights = pd.Series(1.0 / len(usable), index=usable.index)
        else:
            weights = raw_weights / total

    return float((weights * usable[return_col]).sum())


def compute_decile_portfolio_returns(
    panel: pd.DataFrame,
    *,
    model_cols: Sequence[str] | None = None,
    return_col: str = "realized_return",
    weight_col: str | None = None,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Form decile portfolios and compute their next-period returns."""

    _validate_multiindex(panel, "panel")
    if n_deciles <= 0:
        raise ValueError("n_deciles must be positive.")

    if model_cols is None:
        excluded = {return_col}
        if weight_col is not None:
            excluded.add(weight_col)
        model_cols = [col for col in panel.columns if col not in excluded]

    results: dict[str, pd.DataFrame] = {}

    for model in model_cols:
        if model not in panel.columns:
            raise KeyError(f"Missing model column '{model}'.")

        subset_cols: Iterable[str] = [model, return_col]
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

        # Assign deciles within each date based on the model signal
        df_model["decile"] = df_model.groupby(level="date")[model].transform(
            _assign_deciles, n_deciles
        )

        # Only pass return/weight columns into the groupby apply to avoid pandas warnings
        group_cols = [return_col]
        if weight_col is not None:
            group_cols.append(weight_col)

        grouped = df_model.groupby([pd.Grouper(level="date"), "decile"])[group_cols]
        returns = grouped.apply(
            lambda g: _compute_group_return(
                g,
                return_col=return_col,
                weight_col=weight_col,
            )
        )
        returns = returns.unstack("decile")
        results[model] = returns

    if not results:
        empty_index = pd.Index([], name="date")
        empty_cols = pd.MultiIndex.from_arrays([[], []], names=[None, None])
        return pd.DataFrame(index=empty_index, columns=empty_cols)

    combined = pd.concat(results, axis=1)
    # Sort by (model, decile) but keep column level names as None to match tests
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

    _validate_multiindex(panel, "panel")
    if n_deciles <= 0:
        raise ValueError("n_deciles must be positive.")

    if model_cols is None:
        excluded = set()
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

        df_model["decile"] = df_model.groupby(level="date")[model].transform(
            _assign_deciles, n_deciles
        )

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

__all__ = ["compute_decile_portfolio_returns", "compute_decile_portfolio_weights"]
