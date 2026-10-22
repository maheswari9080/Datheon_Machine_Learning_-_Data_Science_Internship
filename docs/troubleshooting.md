# Troubleshooting & Diagnostic Guide

This document catalogs common operational anomalies, edge cases, and their verified resolutions.

---

## 1. Remote Dataset Download Timeouts
- **Symptom**: Network drops or `urllib.error.URLError` when fetching remote CSVs.
- **Resolution**: All fetch functions in `src/data_utils.py` utilize user-agent headers and local persistent caching (`data/raw/`). Once downloaded, subsequent executions read directly from local disk with zero network dependency.
- **Fallback Support**: For Task 4 where the primary GitHub URL was inactive (HTTP 404), the pipeline automatically validates and leverages the verified benchmark review corpus.

---

## 2. TotalCharges Whitespace Conversion in Telco Churn
- **Symptom**: `ValueError: could not convert string to float: ' '`
- **Cause**: In the original Telco Churn dataset, 11 records with `tenure=0` contain single space characters in the `TotalCharges` column.
- **Resolution**: `pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')` converts spaces to `NaN`, followed by `.fillna(0.0)` which accurately reflects zero accumulated charges for new customers.

---

## 3. Lasso Coordinate Descent Convergence Warnings
- **Symptom**: `ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations`
- **Cause**: Unscaled target values in California housing (~$200,000) create large duality gaps under very small $\alpha$ penalties.
- **Resolution**: Setting `max_iter=3000` and tuning $\alpha$ through `GridSearchCV` yields stable convergence around $\alpha = 100.0$.

---

## 4. Matplotlib Headless Execution on Servers / Terminals
- **Symptom**: `_tkinter.TclError: no display name and no $DISPLAY environment variable`
- **Resolution**: `src/plotting.py` forces `matplotlib.use('Agg')` before importing `pyplot`, ensuring artifact generation runs smoothly without requiring an active graphical desktop.
