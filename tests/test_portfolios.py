import numpy as np
import pandas as pd
import pandas.testing as pdt

from portfolios import compute_decile_portfolio_returns


def _make_panel():
    index = pd.MultiIndex.from_product(
        [["A", "B", "C", "D"], pd.to_datetime(["2020-01-31", "2020-02-29"])], names=["ticker", "date"]
    )
    data = {
        "model_a": [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1],
        "realized_return": [0.01, 0.02, 0.03, 0.04, -0.01, 0.0, 0.01, 0.02],
    }
    return pd.DataFrame(data, index=index)


def test_equal_weight_deciles():
    panel = _make_panel()

    result = compute_decile_portfolio_returns(
        panel,
        model_cols=["model_a"],
        return_col="realized_return",
        n_deciles=10,
    )

    # Build expected using the same decile assignment rule used by the module:
    # ranks -> pd.cut into n_deciles bins (because nunique < n_deciles here).
    expected_rows = {}
    for date, g in panel.groupby(level="date"):
        s = g["model_a"].astype(float)
        ranks = s.rank(method="average")
        dec = pd.cut(ranks, bins=10, labels=False, include_lowest=True) + 1  # 1..10

        tmp = pd.DataFrame(
            {
                "decile": dec.to_numpy(),
                "ret": g["realized_return"].to_numpy(),
            },
            index=g.index,
        )

        by_dec = tmp.groupby("decile")["ret"].mean()
        expected_rows[pd.to_datetime(date)] = by_dec

    expected = pd.DataFrame(expected_rows).T.sort_index()
    expected.index.name = "date"

    # Add LS column if both 1 and 10 exist; otherwise NaN (matches updated module behavior)
    expected["LS"] = expected.get(10) - expected.get(1)

    # Convert to MultiIndex columns (model, decile)
    expected.columns = pd.MultiIndex.from_product([["model_a"], expected.columns])
    expected.columns = expected.columns.set_names([None, None])

    # Sort columns like the function does
    expected = expected.sort_index(axis=1, level=[0, 1])

    pdt.assert_frame_equal(result, expected)


def test_value_weighted_deciles():
    index = pd.MultiIndex.from_product(
        [["A", "B", "C"], pd.to_datetime(["2020-03-31"])], names=["ticker", "date"]
    )
    panel = pd.DataFrame(
        {
            "model_b": [0.5, 0.5, 0.5],
            "realized_return": [0.1, 0.0, -0.05],
            "mktcap": [1.0, 3.0, 6.0],
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

    # Expect exactly one decile column with the value-weighted return -0.02,
    # plus an LS column (likely NaN if decile 1 or 10 is missing).
    assert result.index.name == "date"
    assert result.shape[0] == 1

    # find the numeric decile columns
    numeric_deciles = [c for c in result.columns if c[0] == "model_b" and isinstance(c[1], (int, np.integer))]
    assert len(numeric_deciles) == 1
    assert np.isclose(result.loc[pd.to_datetime("2020-03-31"), numeric_deciles[0]], -0.02)

    expected = pd.DataFrame(
        {
            ("model_b", 5): [-0.02],
            ("model_b", "LS"): [np.nan],
        },
        index=pd.to_datetime(["2020-03-31"]),
    )
    expected.index.name = "date"

    pdt.assert_frame_equal(result, expected)