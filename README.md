# Nasdaq Cross-Sectional Return Prediction  
**Testing Predictability and Portfolio Performance in Nasdaq Stocks**

---

## Project Overview

This project studies **cross-sectional return predictability** within the Nasdaq stock universe.  
Rather than forecasting the time-series returns of individual assets, the objective is to determine whether **differences in firm characteristics and factor exposures across stocks** contain systematic information about **differences in expected returns**.

The project evaluates whether such information can be exploited to form portfolios with economically and statistically meaningful out-of-sample performance, while controlling for overfitting, turnover, and transaction costs.

---

## Data and Universe

The dataset consists of Nasdaq-listed equities with historical price and volume data. Returns and firm-level signals are constructed at the **monthly frequency**, which aligns with standard practice in the empirical asset pricing literature.

All computations are performed in a strictly **out-of-sample** setting using rolling or expanding windows, ensuring that no future information is used in portfolio construction.

---

## Rolling Factor Exposures

The module `src/factor_exposures.py` estimates **Fama–French three-factor loadings** for each stock.

- Excess returns are regressed on the market (`MKT`), size (`SMB`), and value (`HML`) factors
- Rolling estimation windows of 36–60 months are used
- Estimated betas are **lagged by one period** to ensure they are observable at time *t*

These factor exposures are later included as additional cross-sectional predictors.

---

## Cross-Sectional Characteristics

Firm characteristics are constructed in `src/features.py` using only price and volume information.  
The implemented signals are standard in the asset pricing literature and include:

- **Size**: log(price × volume)
- **Value proxy**: price relative to a long-term moving average
- **Momentum**: cumulative returns from *t–12* to *t–2*
- **Volatility**: annualized realized volatility from daily returns
- **Investment**: changes in log average trading volume
- **Profitability proxy**: rolling return-over-volatility measures

All features are lagged appropriately to avoid look-ahead bias.

---

## Cross-Sectional Prediction Models

The project compares several cross-sectional prediction approaches:

- **Linear models**: Ordinary Least Squares (OLS)
- **Regularized models**: Ridge, Lasso, Elastic Net
- **Nonlinear models**: Random Forests

Each model predicts either the **level** or the **cross-sectional ranking** of next-period returns, not individual asset time-series returns.

---

## Portfolio Formation

Predicted signals are converted into **cross-sectional decile portfolios** at each date:

- Stocks are sorted into deciles based on predicted returns or ranks
- Equal-weighted decile portfolios are formed
- Long–short portfolios are constructed as **top-minus-bottom deciles**

Performance evaluation includes:

- Mean returns
- Volatility
- Sharpe ratios
- Maximum drawdowns
- Portfolio turnover
- Transaction cost sensitivity

---

## Model Comparison and Statistical Testing

Formal statistical evaluation is a core component of the project. Implemented tests include:

- **Diebold–Mariano tests** for predictive accuracy differences
- **Sharpe ratio significance tests** (Jobson–Korkie with Memmel correction)
- Out-of-sample portfolio performance comparisons

This avoids informal or purely visual comparisons and ensures rigorous inference.

---

## Rank-Based Signals vs Raw Predictions

In addition to using raw model predictions, the project evaluates **cross-sectional rank-transformed signals**.

For example, model predictions are converted into standardized cross-sectional ranks before portfolio formation. Long–short performance from rank-based strategies is directly compared to performance from raw prediction strategies.

This analysis helps assess whether models primarily add value through **relative ordering** rather than absolute return forecasts.

---

## Reporting and Outputs

The module `src/reporting.py` consolidates forecasting, portfolio construction, and evaluation into reproducible outputs.

Key artifacts include:

- Cumulative return plots for selected models
- Long–short cumulative performance plots (log scale)
- Rank-based versus raw-signal comparison plots
- Transaction cost stress-test summaries

The main convenience wrapper `run_plot_suite` produces a minimal and interpretable set of figures used in the final analysis.

---

## Data Loading and Memory Management

To facilitate reproducibility while controlling memory usage, the data-loading pipeline supports subsampling through a global parameter `P_SAMPLE`.

By default:

```python
P_SAMPLE = 1