# Applied Machine Learning Studio
### Datheon Machine Learning & Data Science Internship Suite

![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-brightgreen)
![Tests Passing](https://img.shields.io/badge/Tests-18%2F18%20Passing-success)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-teal)

Applied Machine Learning Studio is an enterprise-grade data science portfolio featuring four independent machine learning projects. The repository demonstrates best practices in exploratory data analysis, data leakage prevention, feature engineering, regularized regression, class-imbalanced classification, natural language processing, and production inference engineering.

---

## Portfolio Projects Summary

| Task | Domain | Benchmark Dataset | Compared Algorithms | Primary Metrics | Status |
|---|---|---|---|---|---|
| **Task 1** | Multi-Class Benchmark Classification | Iris Dataset (150 samples) | Logistic Regression vs. Decision Tree | **Accuracy: 96.67%**<br>Macro F1: 0.9666 | Verified |
| **Task 2** | Real Estate Price Prediction | California Housing (20,640 records) | OLS vs. Ridge vs. Lasso (GridSearchCV) | **R²: 0.6353**<br>RMSE: $69,127 | Verified |
| **Task 3** | Customer Churn Classification | Telco Churn (7,043 customers) | Random Forest vs. XGBoost (Class Weighted) | **ROC-AUC: 0.8425**<br>Recall: 76.74% | Verified |
| **Task 4** | NLP Sentiment Classification | Customer Reviews (3,000 samples) | Multinomial Naive Bayes vs. Logistic Regression | **Accuracy: 85.83%**<br>ROC-AUC: 0.9346 | Verified |

---

## Key Highlights & Architectural Strengths

- **Leak-Free Preprocessing**: All scalers, imputers, encoders, and TF-IDF vectorizers are encapsulated within Scikit-learn `Pipeline` and `ColumnTransformer` constructs, fitted strictly on training data.
- **Cost-Sensitive Learning**: Task 3 leverages balanced class weights and scale factors to overcome severe class imbalance, driving minority-class churn capture above 76.7% recall.
- **Domain Feature Engineering**: Task 2 extracts domain housing ratios (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`) via custom transformer classes.
- **Production Inference Interfaces**: Every project provides a hardened inference function in `src/predictions.py` complete with schema checking and boundary validation.
- **Editorial Web Portal**: Includes a responsive analytics dashboard in `index.html` presenting real benchmark results, visual diagnostic plots, and interactive links.

---

## Repository Architecture

```
.
├── index.html                   # Central editorial analytics portfolio dashboard
├── README.md                    # Portfolio documentation & benchmark summary
├── requirements.txt             # Locked project dependencies
├── .gitignore                   # Clean version control exclusions
├── notebooks/
│   ├── task1_iris_classification.ipynb       # Multi-class benchmark notebook
│   ├── task2_housing_regression.ipynb        # California housing regression notebook
│   ├── task3_customer_churn.ipynb            # Cost-sensitive churn classification notebook
│   └── task4_sentiment_analysis.ipynb        # NLP sentiment analysis notebook
├── src/
│   ├── __init__.py              # Core package declaration
│   ├── data_utils.py            # Caching, schema validation, text sanitation
│   ├── metrics.py               # Standardized multi-task evaluation engine
│   ├── plotting.py              # Publication-grade visualizers
│   ├── predictions.py           # Production inference helper interfaces
│   ├── task1_pipeline.py        # Task 1 training and pipeline logic
│   ├── task2_pipeline.py        # Task 2 feature engineering & ColumnTransformer
│   ├── task3_pipeline.py        # Task 3 class-weighted churn pipelines
│   └── task4_pipeline.py        # Task 4 TF-IDF sentiment pipeline
├── data/
│   ├── raw/                     # Cached benchmark datasets (Iris, Housing, Churn, Sentiment)
│   └── processed/
├── reports/
│   ├── classification/          # Task 1 confusion matrices, boundary plots, JSON metrics
│   ├── regression/              # Task 2 residual diagnostics, parity plots, JSON metrics
│   ├── churn/                   # Task 3 ROC curves, PR curves, feature importance charts
│   └── sentiment/               # Task 4 sentiment ROC curves, comparison charts
├── tests/
│   ├── __init__.py
│   ├── test_data_utils.py       # Tests for caching and text cleaning
│   ├── test_classification.py   # Tests for Task 1 metrics and predictors
│   ├── test_regression.py       # Tests for Task 2 pipelines and estimators
│   ├── test_churn.py            # Tests for Task 3 threshold tuning
│   └── test_sentiment.py        # Tests for Task 4 sentiment inference
└── docs/
    ├── setup.md                 # Local reproduction and setup instructions
    ├── methodology.md           # In-depth architectural & statistical rationale
    └── troubleshooting.md       # Operational debugging and common resolutions
```

---

## Quick Start & Reproduction

### 1. Installation
```bash
git clone git@github.com:maheswari9080/Datheon_Machine_Learning_-_Data_Science_Internship.git
cd Datheon_Machine_Learning_-_Data_Science_Internship

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
pytest tests/ -v
```

### 3. Open Interactive Notebooks
```bash
jupyter notebook notebooks/
```

### 4. Open Central Dashboard
Double click `index.html` or run:
```bash
# On Windows
Start-Process index.html
```

---

## Project Methodology Summary

### Task 1: Multi-Class Benchmark Classification (Iris)
- Stratified 80:20 partitioning ensures balanced multi-class proportions across Setosa, Versicolor, and Virginica.
- Decision Tree achieves **96.67% accuracy** on the test holdout, outperforming Logistic Regression (93.33%) by separating non-linear boundary clusters.
- Includes 2D decision boundary visualization on discriminatory petal dimensions.

### Task 2: Real Estate Housing Price Prediction (California Housing)
- Median imputation for missing `total_bedrooms` records and one-hot encoding for `ocean_proximity`.
- 5-Fold cross-validation GridSearchCV optimizes regularization parameter $\alpha$.
- Explains residual variance homoscedasticity and evaluates predictive limitations ($R^2 \approx 0.6353$).

### Task 3: Customer Churn Classification (Telco Churn)
- Addresses a 73.5% vs. 26.5% class imbalance using cost-sensitive weighting (`class_weight='balanced'`, `scale_pos_weight=2.76`).
- Both Random Forest and XGBoost exceed the target benchmark with **ROC-AUC > 0.84**.
- Top attrition predictors: Month-to-month contract terms, short account tenure, and absence of online tech support.

### Task 4: NLP Sentiment Classification on Customer Reviews
- Sanitizes raw review text by stripping HTML tags, lowercasing, and normalizing punctuation.
- Sublinear TF-IDF scaling (`max_features=5000`) with unigrams and bigrams.
- Logistic Regression achieves **85.83% accuracy** and **0.9346 ROC-AUC**, outperforming Multinomial Naive Bayes (84.83%).

---

## License & Attribution
Developed for the **Datheon Machine Learning & Data Science Internship** program.
All code and visual assets are licensed under the MIT License.
