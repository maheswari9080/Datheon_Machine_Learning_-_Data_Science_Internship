"""
Task 2: Real Estate Housing Price Prediction Pipelines
Implements feature engineering, ColumnTransformer preprocessing, and regression estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_utils import split_data


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that engineers ratio features:
    - rooms_per_household = total_rooms / households
    - bedrooms_per_room = total_bedrooms / total_rooms
    - population_per_household = population / households
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        total_rooms = X_out["total_rooms"]
        total_bedrooms = X_out["total_bedrooms"]
        population = X_out["population"]
        households = np.maximum(X_out["households"], 1)

        X_out["rooms_per_household"] = total_rooms / households
        X_out["bedrooms_per_room"] = total_bedrooms / np.maximum(total_rooms, 1)
        X_out["population_per_household"] = population / households
        return X_out


def build_housing_preprocessor() -> ColumnTransformer:
    """
    Construct leak-proof ColumnTransformer handling numerical imputation/scaling and categorical encoding.
    """
    numeric_features = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
    ]

    categorical_features = ["ocean_proximity"]

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


def build_housing_pipeline(estimator: BaseEstimator) -> Pipeline:
    """
    Assemble the full composite pipeline: Feature Engineering -> ColumnTransformer -> Estimator.
    """
    return Pipeline([
        ("feature_engineer", HousingFeatureEngineer()),
        ("preprocessor", build_housing_preprocessor()),
        ("model", estimator),
    ])


def prepare_housing_data():
    """Load California housing dataset and perform 80:20 split before preprocessing."""
    df = pd.read_csv("data/raw/housing.csv")
    X_train, X_test, y_train, y_test = split_data(
        df, target_col="median_house_value", test_size=0.2, stratify=False, random_state=42
    )
    return X_train, X_test, y_train, y_test
