"""
Task 3: Predictive Customer Churn Pipeline & Imbalance Modeling
Implements robust preprocessing for Telco Churn and model wrappers for Random Forest and XGBoost.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_utils import split_data


def prepare_churn_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load Telco Churn data, cast TotalCharges, drop customerID, and create stratified 80:20 split.
    """
    df = pd.read_csv("data/raw/telco_churn.csv")

    # Safe numeric conversion for TotalCharges (handle whitespace blanks)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    # For new tenure=0 customers, TotalCharges is typically 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Drop non-predictive identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Target binary conversion (1 for Yes, 0 for No)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = split_data(
        df, target_col="Churn", test_size=0.2, stratify=True, random_state=42
    )
    return X_train, X_test, y_train, y_test


def build_churn_preprocessor() -> ColumnTransformer:
    """
    Construct ColumnTransformer for numerical scaling and categorical one-hot encoding.
    """
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ]
    )
    return preprocessor


def build_churn_pipelines(imbalance_ratio: float = 2.76) -> dict:
    """
    Construct Random Forest and XGBoost pipelines addressing class imbalance.

    imbalance_ratio is negative_samples / positive_samples (~73.5 / 26.5 ~= 2.76)
    """
    preprocessor = build_churn_preprocessor()

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "clf",
            XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                scale_pos_weight=imbalance_ratio,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            ),
        ),
    ])

    return {"RandomForest": rf_pipeline, "XGBoost": xgb_pipeline}
