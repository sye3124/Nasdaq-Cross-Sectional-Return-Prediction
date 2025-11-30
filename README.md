 # Nasdaq-Cross-Sectional-Return-Prediction
 Testing Predictability and Portfolio Performance in Nasdaq Stocks

## Rolling factor exposures

Use `src/factor_exposures.py` to estimate Fama-French three-factor loadings for each stock. The helper uses a 36–60 month rolling window, runs an OLS regression of excess returns on `MKT`, `SMB`, and `HML`, and lags the resulting betas so they are observable at time *t*.