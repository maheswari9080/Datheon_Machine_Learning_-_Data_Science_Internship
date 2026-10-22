"""
Task 4: NLP Sentiment Classification Pipelines
Handles text cleaning, vocabulary extraction with TF-IDF, and Naive Bayes / Logistic Regression models.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data_utils import clean_text, split_data


def prepare_sentiment_data() -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load sentiment dataset, clean raw text (strip HTML, lowercase, punctuation),
    and partition into stratified 80:20 train and test sets.
    """
    df = pd.read_csv("data/raw/sentiment_reviews.csv")

    # Clean text reviews
    df["clean_review"] = df["review"].apply(clean_text)

    # Filter any empty records post-cleaning
    df = df[df["clean_review"].str.len() > 0].copy()

    # Binary sentiment target: 1 for positive, 0 for negative
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

    X_train, X_test, y_train, y_test = split_data(
        df[["clean_review", "label"]], target_col="label", test_size=0.2, stratify=True, random_state=42
    )

    return X_train["clean_review"], X_test["clean_review"], y_train, y_test


def build_sentiment_pipelines() -> Dict[str, Pipeline]:
    """
    Construct TF-IDF pipelines for Multinomial Naive Bayes and Logistic Regression.
    Feature dimension capped at 5000 tokens with unigrams and bigrams.
    """
    nb_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        ("clf", MultinomialNB(alpha=1.0)),
    ])

    lr_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        ("clf", LogisticRegression(max_iter=300, random_state=42)),
    ])

    return {"MultinomialNB": nb_pipeline, "LogisticRegression": lr_pipeline}
