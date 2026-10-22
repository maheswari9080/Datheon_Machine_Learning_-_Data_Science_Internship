"""
Unit tests for NLP sentiment classification pipeline and inference helpers
"""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.predictions import predict_sentiment


@pytest.fixture
def mock_sentiment_pipeline():
    texts = [
        "absolutely fantastic and wonderful experience",
        "loved the product quality and fast delivery",
        "terrible waste of money awful broken",
        "worst customer service disappointed and angry",
    ]
    labels = [1, 1, 0, 0]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression()),
    ])
    pipe.fit(texts, labels)
    return pipe


def test_predict_sentiment_positive(mock_sentiment_pipeline):
    res = predict_sentiment(mock_sentiment_pipeline, "Fantastic and wonderful, very happy!")
    assert res["sentiment"] in ["positive", "negative"]
    assert "cleaned_text" in res
    assert "confidence" in res
    assert res["confidence"] is not None


def test_predict_sentiment_empty_input(mock_sentiment_pipeline):
    with pytest.raises(ValueError):
        predict_sentiment(mock_sentiment_pipeline, "   !!!   ")


def test_predict_sentiment_non_string(mock_sentiment_pipeline):
    with pytest.raises(TypeError):
        predict_sentiment(mock_sentiment_pipeline, 12345)
