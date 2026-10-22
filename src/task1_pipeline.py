"""
Task 1: Multi-Class Benchmark Classification Execution Script & Pipelines
Prepares stratified train/test data, fits Logistic Regression and Decision Tree models, and outputs metrics.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_utils import split_data


def build_iris_pipelines():
    """Build and return Logistic Regression and Decision Tree pipelines."""
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=200, random_state=42)),
    ])

    dt_pipeline = Pipeline([
        ("classifier", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ])

    return {"LogisticRegression": lr_pipeline, "DecisionTree": dt_pipeline}


def prepare_iris_data():
    """Load and partition Iris dataset into stratified 80:20 train/test sets."""
    df = pd.read_csv("data/raw/iris.csv")
    X_train, X_test, y_train, y_test = split_data(df, target_col="species", test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
