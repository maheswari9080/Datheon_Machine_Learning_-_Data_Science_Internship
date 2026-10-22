# Environment Setup & Reproduction Guide

This guide details the procedure for establishing a clean, reproducible Python environment to run the Applied Machine Learning Studio portfolio suite.

---

## 1. Prerequisites
- **Python**: Version 3.10 to 3.12 (Tested on Python 3.12.10).
- **Git**: Version 2.40+
- **Hardware**: Standard x86_64 or ARM CPU; no dedicated GPU required.

---

## 2. Clone & Environment Initialization

```bash
# Clone the repository
git clone git@github.com:maheswari9080/Datheon_Machine_Learning_-_Data_Science_Internship.git
cd Datheon_Machine_Learning_-_Data_Science_Internship

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Automated Test Suite Execution

To execute all 18 unit tests covering data acquisition, schema checking, text preprocessing, and inference validation:

```bash
pytest tests/ -v
```

Expected output:
```
============================== 18 passed in ~2.5s ==============================
```

---

## 4. Running the Jupyter Notebooks

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and open any of the four project notebooks:
1. `task1_iris_classification.ipynb`
2. `task2_housing_regression.ipynb`
3. `task3_customer_churn.ipynb`
4. `task4_sentiment_analysis.ipynb`

All notebooks run sequentially from top to bottom (`Cell` -> `Run All`).

---

## 5. Viewing the Central Portfolio Portal

Open `index.html` in any modern web browser:
```bash
# On Windows:
Start-Process index.html
# On macOS:
open index.html
# On Linux:
xdg-open index.html
```
