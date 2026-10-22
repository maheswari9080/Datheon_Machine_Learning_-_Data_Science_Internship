"""
Unit tests for src/data_utils.py
"""

import os
import tempfile
import pandas as pd
import pytest

from src.data_utils import clean_text, fetch_dataset, inspect_dataset, split_data, _validate_schema


def test_clean_text_basic():
    raw = "<b>Outstanding!</b> The delivery was fast & prompt.<br /><br />"
    cleaned = clean_text(raw)
    assert "<b>" not in cleaned
    assert "<br />" not in cleaned
    assert "outstanding the delivery was fast prompt" == cleaned


def test_clean_text_edge_cases():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == "12345"


def test_split_data_classification():
    df = pd.DataFrame({
        "feat1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feat2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "label": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
    })
    X_train, X_test, y_train, y_test = split_data(df, "label", test_size=0.2, random_state=42)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert "label" not in X_train.columns
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_schema_validation_success_and_failure():
    df = pd.DataFrame({"col_a": [1], "col_b": [2]})
    # Should pass without error
    _validate_schema(df, ["col_a", "col_b"], source="test")

    # Should raise ValueError for missing column
    with pytest.raises(ValueError) as excinfo:
        _validate_schema(df, ["col_a", "missing_col"], source="test")
    assert "missing_col" in str(excinfo.value)


def test_inspect_dataset():
    df = pd.DataFrame({
        "num": [1, 2, None, 4],
        "cat": ["x", "y", "x", "x"],
    })
    report = inspect_dataset(df)
    assert report["num_rows"] == 4
    assert report["num_columns"] == 2
    assert report["missing_counts"]["num"] == 1
    assert report["duplicate_rows"] == 0


def test_fetch_dataset_caching():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "test_cache.csv")
        sample_df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        sample_df.to_csv(cache_path, index=False)

        # Should load directly from cache without hitting URL
        loaded = fetch_dataset("http://invalid.url.example.com", cache_path, expected_columns=["x", "y"])
        assert loaded.shape == (2, 2)
        assert list(loaded.columns) == ["x", "y"]
