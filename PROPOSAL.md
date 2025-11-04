# From Predictability to Profitability: Cross-Sectional Stock Selection on Nasdaq

## Project Category
*Business & Finance Tools* | *Data Analysis & Visualization*

---

### Problem Statement and Motivation  
The weak form of the Efficient Market Hypothesis (EMH) asserts that all information contained in past prices and trading volumes is fully reflected in current prices. If this hypothesis holds, then historical data should contain no predictive power for future returns, rendering most technical trading strategies futile.  

However, empirical research in behavioral finance and quantitative investing suggests that limited and temporary inefficiencies can exist, especially across different stocks at a given time (cross-sectionally). This project will **test whether cross-sectional predictability** in Nasdaq stocks can be exploited to generate economically meaningful signals and portfolio performance.  

Instead of forecasting a single stock’s price, the focus will be on predicting **which stocks are likely to outperform their peers** using lagged market-based features. This approach connects machine learning predictions directly to investment outcomes and tests the weak-form efficiency hypothesis.  

---

### Planned Approach and Technologies  

1. **Data Collection and Preparation**  
   - Retrieve daily or weekly historical data (Open, High, Low, Close, Volume) for the Nasdaq-100 using the `yfinance` API.  
   - Compute derived features: multi-horizon momentum (1w, 1m, 6m, 12m), volatility, turnover, mean reversion indicators, and moving-average gaps.  
   - Standardize features cross-sectionally at each time step to prevent scale distortions.  
   - Split the dataset temporally: train on pre-2015 data, test on 2015–2024 (and other cutoffs for robustness).  

2. **Modeling and Portfolio Simulation**  
   - Train multiple supervised models using `scikit-learn`:  
     - **Linear, Lasso, Ridge Regression** for interpretability.  
     - **Random Forest / Gradient Boosting** for non-linear effects.  
     - Optional **Neural Network (MLP)** using TensorFlow for complex interactions.  
   - Each period, use model predictions to **rank stocks** and form a **long–short, sector-neutral portfolio**:  
     - Long top decile, short bottom decile.  
     - Apply equal or volatility-scaled weights.  
     - Deduct transaction costs (e.g., 10–25 bps per turnover side).  

3. **Evaluation and Interpretation**  
   - Statistical metrics: Information Coefficient (Spearman rank correlation between predicted and realized returns).  
   - Economic metrics: portfolio **Sharpe ratio**, **CAGR**, **max drawdown**, **hit rate**.  
   - Factor-based performance: regress portfolio returns on **Fama–French + Momentum factors** to estimate residual alpha.  
   - Visualizations with `matplotlib` and `seaborn` (feature importances, rolling IC, cumulative returns).  

---

### Expected Challenges and Mitigation  
- **High noise and low signal:** Use cross-sectional z-scoring, regularization (Lasso/Ridge), and ensemble methods.  
- **Overfitting risk:** Implement walk-forward validation and out-of-sample testing.  
- **Computational load:** Parallelize per-date modeling using `joblib` or `Numba`.  
- **Data leakage:** Strict temporal splits and validation pipelines.  

---

### Success Criteria  
- Consistent **positive Information Coefficient** and **Sharpe ratio** out-of-sample after costs.  
- Statistically significant **alpha** relative to common risk factors.  
- Reproducible, modular, and well-documented code with clear interpretation of economic results.  

---

### Stretch Goals  
- **Regime analysis**: does predictability vary during high vs. low volatility periods?  
- Ensemble blending of models for **robustness**.  
- **Interactive dashboard** (Streamlit) to visualize live portfolio and model diagnostics.  