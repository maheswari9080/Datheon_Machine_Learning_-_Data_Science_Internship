"""
Unit tests for classification pipeline and prediction helpers
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.metrics import compute_classification_metrics
from src.predictions import predict_iris


@pytest.fixture
def mock_iris_models():
    np.random.seed(42)
    # Generate mock 3-class data
    X = np.random.uniform(1.0, 7.0, size=(60, 4))
    y = np.array(["setosa"] * 20 + ["versicolor"] * 20 + ["virginica"] * 20)

    lr = LogisticRegression(max_iter=100)
    lr.fit(X, y)

    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X, y)

    return {"lr": lr, "dt": dt, "X": X, "y": y}


def test_compute_classification_metrics(mock_iris_models):
    lr = mock_iris_models["lr"]
    X = mock_iris_models["X"]
    y = mock_iris_models["y"]

    y_pred = lr.predict(X)
    metrics = compute_classification_metrics(y, y_pred)

    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_predict_iris_valid(mock_iris_models):
    lr = mock_iris_models["lr"]
    res = predict_iris(lr, 5.1, 3.5, 1.4, 0.2)
    assert "predicted_species" in res
    assert res["predicted_species"] in ["setosa", "versicolor", "virginica"]
    assert "class_probabilities" in res


def test_predict_iris_invalid_values(mock_iris_models):
    lr = mock_iris_models["lr"]
    # Negative value
    with pytest.raises(ValueError):
        predict_iris(lr, -5.1, 3.5, 1.4, 0.2)

    # Extreme unrealistic value
    with pytest.raises(ValueError):
        predict_iris(lr, 100.0, 3.5, 1.4, 0.2)

    # String input
    with pytest.raises(TypeError):
        predict_iris(lr, "five", 3.5, 1.4, 0.2)
