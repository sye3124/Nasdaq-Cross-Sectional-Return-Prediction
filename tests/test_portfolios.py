"""Tests for portfolios.py.

This test module validates `compute_decile_portfolio_returns`, which takes a
(ticker, date) panel containing model signals and realized returns, then builds
decile-sorted portfolio return series.

What is tested
--------------
- Decile assignment:
    Assigns tickers to within-date deciles using quantile binning when possible,
    and falls back to rank-based binning when the cross-section has too few
    unique values (common with small samples or tied signals).

- Return aggregation:
    Computes portfolio returns per (date, decile) using either:
      * equal weights (simple mean), or
      * value weights (sum(w * r) / sum(w)) when `weight_col` is provided.

- Output layout:
    Returns a DataFrame indexed by date with MultiIndex columns (model, decile),
    where decile is typically 1..N plus an optional "LS" column.

- Long–short spread:
    Adds an "LS" column per model equal to top minus bottom only when both
    extreme deciles (1 and N) exist for that model/date; otherwise the spread is NaN.
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.portfolios import compute_decile_portfolio_returns


def _make_panel() -> pd.DataFrame:
    """Create a tiny two-month panel with a monotone signal and realized returns."""
    index = pd.MultiIndex.from_product(
        [["A", "B", "C", "D"], pd.to_datetime(["2020-01-31", "2020-02-29"])],
        names=["ticker", "date"],
    )
    data = {
        # Signal increases within the first month and decreases within the second month.
        "model_a": [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1],
        # Realized returns used to compute portfolio performance.
        "realized_return": [0.01, 0.02, 0.03, 0.04, -0.01, 0.0, 0.01, 0.02],
    }
    return pd.DataFrame(data, index=index)


def test_equal_weight_deciles():
    panel = _make_panel()

    # Ask for 10 deciles even though we only have 4 names per date; the implementation
    # falls back to rank-based binning when unique values < n_deciles.
    result = compute_decile_portfolio_returns(
        panel,
        model_cols=["model_a"],
        return_col="realized_return",
        n_deciles=10,
    )

    # Build expected output using the same fallback decile assignment logic:
    # 1) compute ranks
    # 2) bin ranks into 10 bins with pd.cut
    expected_rows = {}
    for date, g in panel.groupby(level="date"):
        s = g["model_a"].astype(float)
        ranks = s.rank(method="average")

        # With only 4 values, this produces a sparse set of deciles in {1..10}.
        dec = pd.cut(ranks, bins=10, labels=False, include_lowest=True) + 1  # 1..10

        tmp = pd.DataFrame(
            {
                "decile": dec.to_numpy(),
                "ret": g["realized_return"].to_numpy(),
            },
            index=g.index,
        )

        # Equal-weighted return in each decile is just the mean of the members.
        by_dec = tmp.groupby("decile")["ret"].mean()
        expected_rows[pd.to_datetime(date)] = by_dec

    expected = pd.DataFrame(expected_rows).T.sort_index()
    expected.index.name = "date"

    # The function adds LS = top - bottom only if both are present; otherwise NaN.
    expected["LS"] = expected.get(10) - expected.get(1)

    # The function returns MultiIndex columns: (model, decile_or_LS).
    expected.columns = pd.MultiIndex.from_product([["model_a"], expected.columns])
    expected.columns = expected.columns.set_names([None, None])

    # Match the function's column sorting by (model, decile).
    expected = expected.sort_index(axis=1, level=[0, 1])

    pdt.assert_frame_equal(result, expected)


def test_value_weighted_deciles():
    # Single date, three tickers. All signals are tied, so rank-binning will place
    # everything into a single decile bucket.
    index = pd.MultiIndex.from_product(
        [["A", "B", "C"], pd.to_datetime(["2020-03-31"])],
        names=["ticker", "date"],
    )
    panel = pd.DataFrame(
        {
            "model_b": [0.5, 0.5, 0.5],            # tied signal
            "realized_return": [0.1, 0.0, -0.05],  # returns to be value-weighted
            "mktcap": [1.0, 3.0, 6.0],             # weights
        },
        index=index,
    )

    result = compute_decile_portfolio_returns(
        panel,
        model_cols=["model_b"],
        return_col="realized_return",
        weight_col="mktcap",
        n_deciles=10,
    )

    # Basic shape checks: one date, MultiIndex columns, and "date" as the index name.
    assert result.index.name == "date"
    assert result.shape[0] == 1

    # Identify which numeric decile bucket exists for this date (it should be exactly one).
    numeric_deciles = [
        c for c in result.columns
        if c[0] == "model_b" and isinstance(c[1], (int, np.integer))
    ]
    assert len(numeric_deciles) == 1

    # Value-weighted return = sum(w*r)/sum(w) = (1*0.1 + 3*0.0 + 6*-0.05) / (1+3+6)
    #                           = (0.1 + 0 - 0.3) / 10 = -0.02
    assert np.isclose(result.loc[pd.to_datetime("2020-03-31"), numeric_deciles[0]], -0.02)

    # We also expect an LS column, but it's NaN because decile 1 and/or 10 don't exist here.
    expected = pd.DataFrame(
        {
            ("model_b", 5): [-0.02],   # the specific decile produced by the rank-binning in this case
            ("model_b", "LS"): [np.nan],
        },
        index=pd.to_datetime(["2020-03-31"]),
    )
    expected.index.name = "date"

    pdt.assert_frame_equal(result, expected)