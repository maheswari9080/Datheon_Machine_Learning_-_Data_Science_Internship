"""
Unit tests for regression pipeline and housing price estimation helpers
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import compute_regression_metrics
from src.predictions import predict_housing_price


class MockHousingPipeline:
    """Mock pipeline mimicking ColumnTransformer + Regression behavior."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=1.0)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        numeric_cols = ["longitude", "latitude", "housing_median_age", "total_rooms",
                        "total_bedrooms", "population", "households", "median_income",
                        "rooms_per_household", "bedrooms_per_room", "population_per_household"]
        X_num = X[numeric_cols].values
        self.scaler.fit(X_num)
        self.model.fit(self.scaler.transform(X_num), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        numeric_cols = ["longitude", "latitude", "housing_median_age", "total_rooms",
                        "total_bedrooms", "population", "households", "median_income",
                        "rooms_per_household", "bedrooms_per_room", "population_per_household"]
        X_num = X[numeric_cols].values
        X_scaled = self.scaler.transform(X_num)
        return self.model.predict(X_scaled)


@pytest.fixture
def sample_housing_pipeline():
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "longitude": np.random.uniform(-124, -114, n),
        "latitude": np.random.uniform(32, 42, n),
        "housing_median_age": np.random.uniform(5, 50, n),
        "total_rooms": np.random.uniform(500, 5000, n),
        "total_bedrooms": np.random.uniform(100, 1000, n),
        "population": np.random.uniform(200, 3000, n),
        "households": np.random.uniform(100, 1000, n),
        "median_income": np.random.uniform(1.5, 10.0, n),
        "ocean_proximity": ["NEAR BAY"] * n,
    })
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    y = df["median_income"] * 40000 + np.random.normal(0, 10000, n)

    pipeline = MockHousingPipeline()
    pipeline.fit(df, y)
    return pipeline


def test_compute_regression_metrics():
    y_true = np.array([200000, 300000, 400000, 250000])
    y_pred = np.array([210000, 290000, 410000, 240000])

    metrics = compute_regression_metrics(y_true, y_pred, n_features=2)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "adjusted_r2" in metrics
    assert metrics["rmse"] == 10000.0
    assert metrics["mae"] == 10000.0


def test_predict_housing_price_valid(sample_housing_pipeline):
    record = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    }
    result = predict_housing_price(sample_housing_pipeline, record)
    assert "predicted_median_house_value" in result
    assert "formatted_price_usd" in result
    assert "engineered_features" in result
    assert result["predicted_median_house_value"] > 0


def test_predict_housing_price_missing_field(sample_housing_pipeline):
    record = {
        "longitude": -122.23,
        "latitude": 37.88,
        # missing median_income
    }
    with pytest.raises(KeyError):
        predict_housing_price(sample_housing_pipeline, record)
