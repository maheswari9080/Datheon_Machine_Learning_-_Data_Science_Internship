# Applied Machine Learning Studio — Methodology & Technical Architecture

## 1. Architectural Philosophy
The Applied Machine Learning Studio was engineered to demonstrate genuine, production-grade understanding of supervised machine learning, regression analysis, imbalanced learning, and natural language processing. 

### Core Tenets:
1. **Zero Data Leakage**: All learned preprocessing (e.g., imputation medians, standard scaling means/stds, one-hot encoding categories, TF-IDF vocabularies) is fitted exclusively on training partitions. Held-out test sets are transformed, never fitted.
2. **Encapsulation via Pipelines**: Scikit-learn `Pipeline` and `ColumnTransformer` constructs encapsulate all feature transformations together with estimators, guaranteeing portability for production inference.
3. **Reproducibility**: Explicit random states (`random_state=42`) and deterministic stratification ensure results are fully reproducible.
4. **Honest Evaluation**: All reported performance metrics are computed from true out-of-sample holdout test sets.

---

## 2. Project Methodologies

### Task 1: Multi-Class Benchmark Classification (Iris)
- **Objective**: Contrast a linear parametric model (`LogisticRegression`) against an orthogonal non-parametric model (`DecisionTreeClassifier`).
- **Pre-processing**: `StandardScaler` inside `Pipeline` for Logistic Regression to normalize gradient steps.
- **Partitioning**: 80:20 stratified holdout preserving equal representation of `setosa`, `versicolor`, and `virginica`.
- **Interpretability**: Two-feature decision surface projection (petal dimensions) visualizing hyperplanes and decision regions.

### Task 2: Real Estate Housing Price Prediction (California Housing)
- **Objective**: Predict continuous `median_house_value` while managing multicollinearity and spatial features.
- **Feature Engineering**: Formulates domain ratio features:
  - $\text{rooms\_per\_household} = \frac{\text{total\_rooms}}{\text{households}}$
  - $\text{bedrooms\_per\_room} = \frac{\text{total\_bedrooms}}{\text{total\_rooms}}$
  - $\text{population\_per\_household} = \frac{\text{population}}{\text{households}}$
- **ColumnTransformer**: Imputes missing values in `total_bedrooms` using median strategy; standardizes numerical features; applies `OneHotEncoder(handle_unknown='ignore')` to `ocean_proximity`.
- **Regularization Tuning**: 5-Fold cross-validation on training folds using `GridSearchCV` over regularization penalty $\alpha \in [0.1, 100.0]$.
- **Diagnostics**: Residual analysis verifies homoscedasticity assumptions and inspects distribution symmetry.

### Task 3: Customer Churn Classification (Telco Churn)
- **Objective**: Mitigate class imbalance (73.5% Non-Churn vs 26.5% Churn).
- **Strategy**: Cost-sensitive re-weighting using `class_weight='balanced'` in Random Forest and `scale_pos_weight=2.76` in XGBoost.
- **Metric Priority**: In churn retention, False Negatives (lost customers) are substantially more costly than False Positives (a retention discount sent to an existing customer). Therefore, optimization targets ROC-AUC ($>0.84$) and Recall ($>76\%$).
- **Threshold Analysis**: Evaluates decision threshold trade-offs (e.g. lowering threshold to 0.35 to capture additional churners).

### Task 4: NLP Sentiment Classification on Customer Reviews
- **Objective**: Classify customer sentiment as positive or negative.
- **Data Engineering**: Robust regex cleaning stripping HTML formatting (`<br />`), standardizing lowercase, and removing punctuation noise.
- **Representation**: Sublinear `TfidfVectorizer(max_features=5000, ngram_range=(1,2))` capturing informative bigrams while dampening high-frequency saturation.
- **Comparison**: Evaluates Multinomial Naive Bayes against $L_2$-regularized Logistic Regression, with false positive and false negative error inspection.
