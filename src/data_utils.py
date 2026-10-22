"""
Applied Machine Learning Studio - Data Engineering Utilities
Provides robust data acquisition, caching, schema validation, text sanitation, and data splitting.
"""

from __future__ import annotations

import io
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_dataset(
    url: str,
    local_path: str,
    expected_columns: Optional[List[str]] = None,
    fallback_url: Optional[str] = None,
    force_download: bool = False,
    sep: str = ",",
) -> pd.DataFrame:
    """
    Acquire a dataset from a URL with persistent disk caching and schema validation.

    Parameters
    ----------
    url : str
        Primary remote dataset URL.
    local_path : str
        Local file path for caching.
    expected_columns : Optional[List[str]]
        Required columns that must be present in the DataFrame.
    fallback_url : Optional[str]
        Secondary URL attempted if primary retrieval fails.
    force_download : bool
        If True, ignore local cache and re-download.
    sep : str
        Delimiter string for parsing CSV data.

    Returns
    -------
    pd.DataFrame
        Loaded and verified DataFrame.
    """
    if os.path.exists(local_path) and not force_download:
        logger.info(f"Loading cached dataset from: {local_path}")
        df = pd.read_csv(local_path, sep=sep)
        _validate_schema(df, expected_columns, source=local_path)
        return df

    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    df = _download_with_fallback(url, fallback_url=fallback_url, sep=sep)
    _validate_schema(df, expected_columns, source=url)

    df.to_csv(local_path, index=False)
    logger.info(f"Dataset successfully cached to: {local_path}")
    return df


def _download_with_fallback(
    url: str, fallback_url: Optional[str] = None, sep: str = ","
) -> pd.DataFrame:
    """Helper to download CSV data with user-agent headers and fallback support."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Applied-ML-Studio/1.0"}

    for candidate_url in [url, fallback_url]:
        if not candidate_url:
            continue
        try:
            logger.info(f"Attempting download from: {candidate_url}")
            req = urllib.request.Request(candidate_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as response:
                content = response.read()
                df = pd.read_csv(io.BytesIO(content), sep=sep)
                logger.info(f"Successfully retrieved dataset from: {candidate_url}")
                return df
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, Exception) as exc:
            logger.warning(f"Download failed for {candidate_url}: {exc}")

    raise RuntimeError(f"Failed to acquire dataset from primary ({url}) and fallback ({fallback_url})")


def _validate_schema(
    df: pd.DataFrame, expected_columns: Optional[List[str]], source: str
) -> None:
    """Validate that expected columns are present in the loaded dataset."""
    if expected_columns:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Schema validation failure for {source}. Missing columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Produce structured summary statistics and integrity checks for a dataset.
    """
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = {k: round((v / len(df)) * 100, 2) for k, v in missing_counts.items()}
    duplicates = int(df.duplicated().sum())

    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "missing_counts": missing_counts,
        "missing_percentages": missing_pct,
        "duplicate_rows": duplicates,
    }


def clean_text(
    text: Any,
    remove_html: bool = True,
    lower: bool = True,
    remove_punctuation: bool = True,
    remove_extra_spaces: bool = True,
) -> str:
    """
    Sanitize raw review text for natural language processing pipelines.

    Parameters
    ----------
    text : Any
        Input text or object convertible to string.
    remove_html : bool
        Strip HTML tags such as <br />, <div>, etc.
    lower : bool
        Convert characters to lowercase.
    remove_punctuation : bool
        Remove non-alphanumeric punctuation marks.
    remove_extra_spaces : bool
        Collapse multi-space sequences into single spaces.
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text)

    if remove_html:
        text = re.sub(r"<[^>]+>", " ", text)

    if lower:
        text = text.lower()

    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)

    if remove_extra_spaces:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Partition features and target into reproducible train and test subsets.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify_target = None
    if stratify:
        # Only stratify if target is discrete/classification
        if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 20:
            logger.info("Continuous target detected; stratification disabled.")
            stratify_target = None
        else:
            stratify_target = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_target
    )

    return X_train, X_test, y_train, y_test
