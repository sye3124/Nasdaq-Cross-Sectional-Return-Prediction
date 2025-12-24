
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import factor_exposures as fe

def test_compute_factor_exposures_lagged_betas():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")

    factors = pd.DataFrame(
        {
            "MKT": [0.01, 0.02, 0.03, 0.04, 0.05],
            "SMB": 0.0,
            "HML": 0.0,
            "RF": 0.0,
        },
        index=dates,
    )

    # Excess return is exactly 2 * MKT, so the estimated beta_MKT should be 2.0
    returns = pd.DataFrame(
    {
        "RET": (2 * factors["MKT"]).values,
    },
    index=pd.MultiIndex.from_product([["AAA"], dates], names=["ticker", "date"]),
    )

    config = fe.ExposureConfig(min_months=3, max_months=5)
    result = fe.compute_factor_exposures(returns, factors, config=config)

    # Only the last two rows should have lagged exposures available (after shift)
    non_na = result.dropna()
    assert list(non_na.index.get_level_values("date")) == [dates[3], dates[4]]
    pd.testing.assert_series_equal(
        non_na["beta_MKT"],
        pd.Series([2.0, 2.0], index=pd.MultiIndex.from_product([["AAA"], dates[3:]], names=["ticker", "date"])),
        check_names=False,
        check_exact=False,
        rtol=1e-6,
    )
    assert (non_na["beta_SMB"].abs() < 1e-8).all()
    assert (non_na["beta_HML"].abs() < 1e-8).all()


def test_compute_factor_exposures_not_enough_history_returns_empty():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    factors = pd.DataFrame(
        {"MKT": 0.0, "SMB": 0.0, "HML": 0.0, "RF": 0.0}, index=dates
    )
    returns = pd.DataFrame(
        {"RET": 0.0},
        index=pd.MultiIndex.from_product([["AAA"], dates], names=["ticker", "date"]),
    )

    config = fe.ExposureConfig(min_months=4, max_months=5)
    result = fe.compute_factor_exposures(returns, factors, config=config)

    assert result.empty
    assert list(result.columns) == ["alpha_ff3", "beta_MKT", "beta_SMB", "beta_HML"]


def test_compute_factor_exposures_requires_multiindex():
    factors = pd.DataFrame({"MKT": [], "SMB": [], "HML": [], "RF": []})
    returns = pd.DataFrame({"RET": []})

    with pytest.raises(ValueError):
        fe.compute_factor_exposures(returns, factors)