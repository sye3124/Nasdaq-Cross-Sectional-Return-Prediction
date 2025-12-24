# Nasdaq-Cross-Sectional-Return-Prediction
Testing Predictability and Portfolio Performance in Nasdaq Stocks

## Rolling factor exposures
Use `src/factor_exposures.py` to estimate Fama-French three-factor loadings for each stock. The helper uses a 36–60 month rolling window, runs an OLS regression of excess returns on `MKT`, `SMB`, and `HML`, and lags the resulting betas so they are observable at time *t*.

## Cross-sectional characteristics
`src/features.py` constructs a `features` database from price/volume histories. It implements common asset-pricing signals using only market data:
- Size (log price × volume using monthly close/average volume)
- Value proxies (price-to-12M moving average or long-term reversal)
- Momentum (cumulative return months *t-12* to *t-2*)
- Volatility (annualized realized volatility from daily returns)
- Investment (change in log average 12M volume)
- Profitability (12M rolling Sharpe or return-over-volatility)

## Comparison tables and plots
`src/reporting.py` stitches together the forecasting, portfolio, and performance helpers to generate common evaluation artifacts:
- Model performance tables with cumulative-return plots via `compute_model_performance_table`
- Diebold–Mariano and Sharpe-significance tables via `compute_dm_significance_table` and `compute_sharpe_significance_table`
- Rolling Sharpe, factor-exposure, and feature-importance plots via `plot_rolling_sharpe`, `plot_factor_exposures`, and `plot_feature_importance`

The convenience wrapper `generate_comparison_report` accepts decile returns (and optional predictions, weights, factor loadings, and feature panels) and writes plots to a target `output_dir`.