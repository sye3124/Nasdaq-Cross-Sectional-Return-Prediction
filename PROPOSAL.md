# Project Proposal  
## Testing Weak-Form Market Efficiency: Predicting Nasdaq Stock Returns from Historical Data  
**Category:** Data Analysis & Visualisation / Business & Finance Tools  

---

### Problem Statement and Motivation  
According to the **Weak form of the Efficient Market Hypothesis (EMH)**, all information contained in past prices, returns, and trading volumes is already reflected in current stock prices. If this holds true, then historical data should not provide any predictive power for future price movements, and price changes should follow a **random walk**.

However, in practice, many traders and quantitative analysts continue to develop models based on historical data, believing that behavioural or structural inefficiencies may allow for limited predictability. This project aims to **empirically test the weak-form efficiency** of the Nasdaq stock market by attempting to predict future returns using only past information.

The core question is:  
> Can historical price and volume data be used to predict future returns for Nasdaq-listed stocks, or does the data support the weak form of market efficiency?

---

### Planned Approach and Technologies  
1. **Data Collection and Preparation:**  
   - Gather daily historical data (Open, High, Low, Close, Volume) for a sample of Nasdaq stocks (e.g., Nasdaq-100).  
   - Compute derived metrics such as daily returns, rolling averages, and volatility.  
   - Split the dataset into **training (before a cutoff year, e.g., 2015)** and **testing (after 2015)** periods to avoid data leakage.

2. **Feature Engineering and Modeling:**  
   - Generate lagged features (e.g., past 5-day returns, volume trends, momentum indicators).  
   - Train several models using **scikit-learn**:  
     - Linear and Ridge regression for baseline predictability.  
     - Logistic regression for “up vs down” prediction.  
     - Random Forest or Gradient Boosting for non-linear effects.  
   - Optionally test a simple **neural network (MLP)** using TensorFlow.

3. **Evaluation:**  
   - Measure performance with **R², MSE** (for regression) and **accuracy, F1-score** (for classification).  
   - Compare model performance against naive/random baselines to assess whether predictability exists.  
   - Visualise results using **matplotlib** and **seaborn** (e.g., feature importances, actual vs. predicted returns).

---

### Expected Challenges and Mitigation  
- **Noisy Data:** Financial data are volatile and non-stationary. → Apply normalisation, rolling averages, and outlier removal.  
- **Overfitting:** Regularisation (Lasso/Ridge) and cross-validation will be used to control complexity.  
- **Computation Time:** For many stocks, modeling will be parallelised using **Numba** or **joblib**.

---

### Success Criteria  
- Code runs reproducibly and efficiently across multiple stocks.  
- Predictive models outperform a random baseline or naive mean forecast.  
- Results are clearly visualised and interpreted in the context of weak-form efficiency.  
- All methods are well-documented, and findings are presented clearly in the final notebook/report.

---

### Stretch Goals  
- Incorporate a **Monte Carlo simulation** to test return distributions.  
- Build an **interactive dashboard (Streamlit)** for exploring predictions and volatility.  
- Perform a **cross-period analysis** to test if predictability changes over time (e.g., pre- and post-COVID).

---