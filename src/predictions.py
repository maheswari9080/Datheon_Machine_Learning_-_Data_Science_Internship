"""
Applied Machine Learning Studio - Production Prediction & Inference Engine
Exposes reusable inference functions with strict input boundary validation for all 4 project pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.data_utils import clean_text


def predict_iris(
    model: Any,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> Dict[str, Any]:
    """
    Validate input dimensions and predict Iris flower species.

    Parameters
    ----------
    model : Trained Scikit-learn estimator or pipeline.
    sepal_length, sepal_width, petal_length, petal_width : float
        Measurements in centimeters (must be strictly positive).
    """
    inputs = [sepal_length, sepal_width, petal_length, petal_width]
    for val in inputs:
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"All flower measurements must be numeric; got {type(val)}: {val}")
        if val <= 0.0 or val > 30.0:
            raise ValueError(f"Measurement {val} is outside biologically plausible range (0 < x <= 30).")

    df_sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )

    pred = model.predict(df_sample)[0]
    result: Dict[str, Any] = {"predicted_species": str(pred)}

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df_sample)[0]
        classes = getattr(model, "classes_", [0, 1, 2])
        result["class_probabilities"] = {str(c): float(round(p, 4)) for c, p in zip(classes, probs)}

    return result


def predict_housing_price(
    pipeline: Any,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict median housing price for a California housing record with automated ratio feature generation.
    """
    required_cols = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
    ]

    for col in required_cols:
        if col not in record:
            raise KeyError(f"Missing required housing record field: '{col}'")

    df_sample = pd.DataFrame([record])

    # Validate numerical types
    for col in required_cols:
        if col != "ocean_proximity":
            df_sample[col] = pd.to_numeric(df_sample[col], errors="raise")

    # Automated feature engineering
    df_sample["rooms_per_household"] = df_sample["total_rooms"] / np.maximum(df_sample["households"], 1)
    df_sample["bedrooms_per_room"] = df_sample["total_bedrooms"] / np.maximum(df_sample["total_rooms"], 1)
    df_sample["population_per_household"] = df_sample["population"] / np.maximum(df_sample["households"], 1)

    predicted_val = float(pipeline.predict(df_sample)[0])
    return {
        "predicted_median_house_value": float(round(predicted_val, 2)),
        "formatted_price_usd": f"${predicted_val:,.2f}",
        "engineered_features": {
            "rooms_per_household": float(round(df_sample["rooms_per_household"].iloc[0], 2)),
            "bedrooms_per_room": float(round(df_sample["bedrooms_per_room"].iloc[0], 4)),
            "population_per_household": float(round(df_sample["population_per_household"].iloc[0], 2)),
        },
    }


def predict_churn(
    pipeline: Any,
    customer_record: Dict[str, Any],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Predict customer churn status and probability with an adjustable decision threshold.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Decision threshold must be bounded within [0, 1]; got {threshold}")

    # Remove identifier if mistakenly present
    record_clean = {k: v for k, v in customer_record.items() if k.lower() != "customerid"}
    df_sample = pd.DataFrame([record_clean])

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_sample)[0]
        # Probability of positive class (Churn = 'Yes' or 1)
        churn_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    else:
        raw_pred = pipeline.predict(df_sample)[0]
        churn_prob = 1.0 if str(raw_pred).lower() in ["yes", "1", "true"] else 0.0

    prediction_label = "Yes" if churn_prob >= threshold else "No"
    risk_tier = "High" if churn_prob >= 0.7 else "Moderate" if churn_prob >= 0.4 else "Low"

    return {
        "churn_prediction": prediction_label,
        "churn_probability": float(round(churn_prob, 4)),
        "decision_threshold": threshold,
        "risk_tier": risk_tier,
    }


def predict_sentiment(
    pipeline: Any,
    review_text: str,
) -> Dict[str, Any]:
    """
    Sanitize raw review text and predict binary sentiment with confidence rating.
    """
    if not isinstance(review_text, str):
        raise TypeError(f"Review input must be string; received {type(review_text)}")

    cleaned = clean_text(review_text)
    if not cleaned:
        raise ValueError("Review text contains no valid tokens after cleaning.")

    pred = pipeline.predict([cleaned])[0]
    sentiment_label = "positive" if str(pred).lower() in ["1", "positive", "pos"] else "negative"

    confidence = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([cleaned])[0]
        confidence = float(round(max(probs), 4))

    return {
        "sentiment": sentiment_label,
        "confidence": confidence,
        "cleaned_text": cleaned,
        "raw_text_preview": review_text[:120],
    }
