"""
Unit tests for customer churn pipeline and inference helpers
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.predictions import predict_churn


class MockChurnPipeline:
    """Mock pipeline that outputs calibrated churn probabilities."""
    def __init__(self, prob=0.65):
        self.prob = prob

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([[1.0 - self.prob, self.prob]])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(["Yes" if self.prob >= 0.5 else "No"])


def test_predict_churn_high_risk():
    pipeline = MockChurnPipeline(prob=0.82)
    sample_customer = {
        "customerID": "1234-TEST",
        "tenure": 3,
        "MonthlyCharges": 89.5,
        "TotalCharges": 268.5,
        "Contract": "Month-to-month",
    }
    result = predict_churn(pipeline, sample_customer, threshold=0.5)
    assert result["churn_prediction"] == "Yes"
    assert result["churn_probability"] == 0.82
    assert result["risk_tier"] == "High"


def test_predict_churn_threshold_adjustment():
    pipeline = MockChurnPipeline(prob=0.45)
    sample_customer = {"tenure": 12, "MonthlyCharges": 65.0}

    # At standard 0.5 threshold -> No
    res_default = predict_churn(pipeline, sample_customer, threshold=0.5)
    assert res_default["churn_prediction"] == "No"

    # At sensitive 0.4 threshold -> Yes
    res_sensitive = predict_churn(pipeline, sample_customer, threshold=0.4)
    assert res_sensitive["churn_prediction"] == "Yes"


def test_predict_churn_invalid_threshold():
    pipeline = MockChurnPipeline(prob=0.5)
    with pytest.raises(ValueError):
        predict_churn(pipeline, {"tenure": 5}, threshold=1.5)
