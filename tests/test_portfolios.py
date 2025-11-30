import pandas as pd
import pandas.testing as pdt

from src.portfolios import compute_decile_portfolio_returns


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

    expected = pd.DataFrame(
        {
            ("model_a", 3): [0.01, 0.02],
            ("model_a", 5): [0.01, 0.02],
            ("model_a", 8): [0.03, 0.0],
            ("model_a", 10): [-0.01, 0.04],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )
    expected.index.name = "date"

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

    expected = pd.DataFrame({("model_b", 7): [-0.02]}, index=pd.to_datetime(["2020-03-31"]))
    expected.index.name = "date"

    pdt.assert_frame_equal(result, expected)