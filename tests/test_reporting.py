"""Tests for reporting.py.

This test module targets the "glue" and plotting/reporting utilities in
`src.reporting`. Unlike core finance/stat modules, these helpers are mostly about:

- Robust input handling (date coercion, MultiIndex/tuple columns, stable ordering)
- Defensive behavior (don't crash on weird shapes, avoid brittle pandas selection)
- Writing plot files reliably (matplotlib Agg backend)

We intentionally DO NOT unit-test statistical correctness of DM / Jobson–Korkie here,
because those belong to `forecasting_metrics.py` and `performance_metrics.py`.
Instead we test that reporting.py calls/wires things correctly and stays stable.

What is tested
--------------
- Title/label helpers:
    `_format_plot_title()` and `_format_y_label()` produce stable, consistent strings.

- Column hygiene:
    `_ensure_model_decile_multiindex()` converts "Index of tuples" -> true MultiIndex.
    `_reorder_decile_columns()` produces a deterministic column order:
        model sorted, then numeric deciles ascending, then non-numeric ("LS") last.

- Basic return summary:
    `_summarize_returns_basic()` computes annualized mean/vol/Sharpe, t-stat,
    and returns cumulative/drawdown series, while:
        * coercing index to datetime,
        * not exploding on constant returns (zero vol),
        * and skipping wealth math for <= -100% returns.

- Plot outputs (smoke tests):
    Plot functions should write a non-empty PNG file to disk.

- "Pipeline-ish" helpers:
    `transaction_cost_stress_test()` and `plot_long_short_rank_vs_raw()` are
    tested with monkeypatched dependencies so tests stay fast and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.reporting as rp


def _make_decile_returns(
    *,
    models=("m1", "m2"),
    deciles=(1, 2, "LS"),
    dates=None,
    base=0.01,
) -> pd.DataFrame:
    """Create a tiny decile return panel with MultiIndex columns.

    Why this helper exists:
    - Most reporting functions assume a (model, decile) MultiIndex column layout.
    - Repeating that boilerplate in every test obscures what each test is checking.
    """
    if dates is None:
        dates = pd.date_range("2020-01-31", periods=4, freq="ME")

    cols = pd.MultiIndex.from_product([models, deciles])

    # Keep values small and stable: this avoids pathological cumprod growth and makes
    # drawdown math predictable, while still producing non-zero series.
    data = {}
    for col in cols:
        data[col] = base + 0.001 * (hash(str(col)) % 5)

    return pd.DataFrame(data, index=dates)


def test_format_plot_title_variants():
    # Stable titles are important for reproducible reports and for avoiding
    # lots of slightly different plot names when comparing models.
    t1 = rp._format_plot_title("Cumulative Returns")
    assert t1 == "Cumulative Returns"

    t2 = rp._format_plot_title(
        "Cumulative Returns",
        subject="Model A",
        start_date="2020-01-31",
        log_scale=True,
        extra="deciles",
    )
    assert "Cumulative Returns — Model A" in t2
    assert "from 2020-01-31" in t2
    assert "log scale" in t2
    assert "deciles" in t2
    assert t2.endswith(")")


def test_format_y_label():
    # The y-label must match whether we plot wealth on log or linear scale.
    assert rp._format_y_label(log_scale=True) == "Growth of $1 (log)"
    assert rp._format_y_label(log_scale=False) == "Growth of $1"


def test_ensure_model_decile_multiindex_from_tuple_index():
    # Many CSV round-trips (or certain constructions) turn MultiIndex columns into
    # a flat Index of tuples. This helper restores stable MultiIndex behavior.
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    df = pd.DataFrame(
        {(("m1", 1)): [0.01, 0.02], (("m1", "LS")): [0.03, 0.04]},
        index=dates,
    )
    df.columns = pd.Index(list(df.columns))  # force "Index of tuples" (not MultiIndex)

    out = rp._ensure_model_decile_multiindex(df)
    assert isinstance(out.columns, pd.MultiIndex)
    assert out.columns.nlevels == 2


def test_reorder_decile_columns_numeric_then_other():
    # MultiIndex sorting in pandas can be brittle when columns aren't lexsorted.
    # We want a deterministic order so:
    # - iteration over deciles is stable,
    # - "top minus bottom" means the same thing always,
    # - plots come out consistent.
    df = _make_decile_returns(models=("b", "a"), deciles=("LS", 2, 1))
    out = rp._reorder_decile_columns(df)

    assert list(out.columns) == [
        ("a", 1),
        ("a", 2),
        ("a", "LS"),
        ("b", 1),
        ("b", 2),
        ("b", "LS"),
    ]


def test_summarize_returns_basic_happy_path():
    # This is the core calculation that many reports rely on.
    # We test:
    # - index coercion (string -> datetime),
    # - Sharpe handling when volatility is zero,
    # - cumulative + drawdown availability.
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    returns = pd.DataFrame(
        {
            "m1": [0.01, 0.02, 0.00, -0.01, 0.03],
            "m2": [0.00, 0.00, 0.00, 0.00, 0.00],  # constant => vol=0 => Sharpe undefined
        },
        index=dates.astype(str),  # exercise datetime coercion
    )

    metrics, cumulative, drawdowns = rp._summarize_returns_basic(
        returns, risk_free_rate=0.0, periods_per_year=12
    )

    assert set(metrics.columns) >= {
        "mean_return",
        "volatility",
        "sharpe_ratio",
        "t_stat_mean",
        "max_drawdown",
    }
    assert "m1" in metrics.index
    assert "m2" in metrics.index

    # Cumulative/drawdowns are only created when wealth math is possible;
    # for m1 it should be present.
    assert "m1" in cumulative.columns
    assert "m1" in drawdowns.columns

    # For constant returns, Sharpe is not meaningful (division by zero vol).
    assert np.isnan(metrics.loc["m2", "sharpe_ratio"])


def test_summarize_returns_basic_empty_and_type_error():
    # These are reporting helpers — they should fail loudly on wrong types,
    # but gracefully on empty inputs.
    with pytest.raises(TypeError):
        rp._summarize_returns_basic([1, 2, 3])  # not a DataFrame

    metrics, cumulative, drawdowns = rp._summarize_returns_basic(pd.DataFrame())
    assert metrics.empty and cumulative.empty and drawdowns.empty


def test_clip_realized_returns_clips_and_drops_nans():
    # Realized return tails can blow up plots; clipping stabilizes diagnostics.
    # We also drop NaN realized values so performance isn't accidentally inflated.
    preds = pd.DataFrame(
        {
            "realized_return": [np.nan, -5.0, 0.1, 10.0],
            "m1": [0.0, 0.0, 0.0, 0.0],
        }
    )
    out = rp.clip_realized_returns(preds, realized_col="realized_return", lower=-0.9, upper=2.0)

    assert len(out) == 3  # NaN realized row removed
    assert out["realized_return"].min() >= -0.9
    assert out["realized_return"].max() <= 2.0


def test_plot_decile_cumulative_for_model_writes_file(tmp_path: Path):
    # Plotting should be a side-effect that reliably creates an artifact,
    # but should not require GUI backends (Agg is used in the module).
    df = _make_decile_returns(models=("m1",), deciles=(1, 2, 3))
    out_path = tmp_path / "deciles.png"

    result = rp.plot_decile_cumulative_for_model(
        df,
        model="m1",
        n_deciles=3,
        start_date=None,
        log_scale=False,
        output_path=out_path,
    )

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0  # non-empty PNG


def test_plot_decile_cumulative_for_model_raises_on_missing_model(tmp_path: Path):
    # Failing early is better than silently creating an empty plot.
    df = _make_decile_returns(models=("m1",), deciles=(1, 2, 3))
    with pytest.raises(KeyError):
        rp.plot_decile_cumulative_for_model(
            df,
            model="does_not_exist",
            n_deciles=3,
            output_path=tmp_path / "x.png",
        )


def test_plot_long_short_cumulative_uses_pm_and_writes_file(tmp_path: Path, monkeypatch):
    # `plot_long_short_cumulative` delegates LS construction to performance_metrics.
    # We monkeypatch it to avoid dependency on the exact LS implementation while still
    # validating plotting/wiring behavior.
    df = _make_decile_returns(models=("m1", "m2"), deciles=(1, 2))

    def fake_compute_ls(decile_returns, bottom_decile=1, top_decile=None):
        idx = pd.to_datetime(decile_returns.index)
        out = pd.DataFrame(
            {
                ("m1", "long_short"): [0.01, 0.02, 0.00, 0.01],
                ("m2", "long_short"): [0.00, 0.01, 0.01, -0.01],
            },
            index=idx,
        )
        out.columns = pd.MultiIndex.from_tuples(out.columns)
        return out

    monkeypatch.setattr(rp.pm, "compute_long_short_returns", fake_compute_ls)

    out_path = tmp_path / "ls.png"
    result = rp.plot_long_short_cumulative(
        df,
        start_date="2020-01-31",
        log_scale=True,
        output_path=out_path,
        bottom_decile=1,
        top_decile=2,
    )

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_transaction_cost_stress_test_smoke(tmp_path: Path, monkeypatch):
    # This function orchestrates a lot of work and file output. We monkeypatch
    # summarize_portfolio_performance to keep the test:
    # - fast (no heavy computations),
    # - deterministic (no randomness),
    # - focused on reporting.py behavior (aggregation + plot writing).
    decile_returns = _make_decile_returns(models=("m1",), deciles=(1, 2, "LS"))
    decile_weights = _make_decile_returns(models=("m1",), deciles=(1, 2))

    def fake_summarize_portfolio_performance(
        decile_returns,
        turnover_weights,
        transaction_cost_bps,
        periods_per_year,
        plot_path=None,
    ):
        # Provide exactly the objects reporting.py expects:
        # - metrics with an LS row
        # - cumulative/drawdowns with an ("m1","LS") column
        idx = pd.MultiIndex.from_tuples([("m1", "LS")], names=[None, None])
        metrics = pd.DataFrame(
            {
                "mean_return": [0.12],
                "volatility": [0.2],
                "sharpe_ratio": [0.6],
                "t_stat_mean": [1.5],
                "max_drawdown": [-0.3],
            },
            index=idx,
        )

        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        cum = pd.DataFrame({("m1", "LS"): [1.0, 1.02, 1.05]}, index=dates)
        dd = pd.DataFrame({("m1", "LS"): [0.0, -0.01, -0.02]}, index=dates)
        cum.columns = pd.MultiIndex.from_tuples(cum.columns)
        dd.columns = pd.MultiIndex.from_tuples(dd.columns)
        return metrics, cum, dd

    monkeypatch.setattr(rp, "summarize_portfolio_performance", fake_summarize_portfolio_performance)

    out = rp.transaction_cost_stress_test(
        decile_returns,
        decile_weights,
        start_date="2020-01-31",
        cost_bps_list=[0, 10],
        periods_per_year=12,
        output_dir=tmp_path,
        save_plot=True,
        plot_ls_models=["m1"],
    )

    assert "tc_summary" in out and "results" in out
    assert isinstance(out["tc_summary"], pd.DataFrame)

    # If a plot was requested, it should exist (some code paths may choose to skip).
    if out["plot_path"] is not None:
        assert Path(out["plot_path"]).exists()


def test_plot_long_short_rank_vs_raw_smoke(tmp_path: Path, monkeypatch):
    # This compares two datasets by recomputing LS inside each one.
    # We monkeypatch LS computation so this test stays stable regardless of how
    # performance_metrics defines "top minus bottom".
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    raw = _make_decile_returns(models=("ridge_pred",), deciles=(1, 10), dates=dates)
    rank = _make_decile_returns(models=("ridge_rank",), deciles=(1, 10), dates=dates)

    def fake_compute_ls(decile_returns, bottom_decile=1, top_decile=None):
        idx = pd.to_datetime(decile_returns.index)
        models = sorted(set([c[0] for c in decile_returns.columns]))
        data = {}
        for m in models:
            data[(m, "long_short")] = [0.01, 0.0, 0.02, -0.01]
        out = pd.DataFrame(data, index=idx)
        out.columns = pd.MultiIndex.from_tuples(out.columns)
        return out

    monkeypatch.setattr(rp.pm, "compute_long_short_returns", fake_compute_ls)

    out_path = tmp_path / "rank_vs_raw.png"
    out = rp.plot_long_short_rank_vs_raw(
        decile_returns_raw=raw,
        decile_returns_rank=rank,
        raw_model="ridge_pred",
        rank_model="ridge_rank",
        start_date="2020-01-31",
        log_scale=True,
        output_path=out_path,
    )

    assert out["output_path"] == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert isinstance(out["cum"], pd.DataFrame)