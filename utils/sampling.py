"""
Smart sampling utilities for variety-maximizing value selection.

These utilities help create representative samples of column values
for LLM context without overwhelming token budgets.
"""

from typing import Any

import pandas as pd
import numpy as np


def sample_column_values(
    series: pd.Series,
    max_samples: int = 5,
    include_extremes: bool = True,
) -> list[Any]:
    """
    Sample values from a column with variety maximization.

    For numeric columns: includes min, max, median if include_extremes=True.
    For categorical: prioritizes diverse categories.
    For datetime: includes earliest, latest, and spread.

    Args:
        series: Pandas Series to sample from
        max_samples: Maximum number of samples to return
        include_extremes: Whether to include min/max for ordered types

    Returns:
        List of sampled values (JSON-serializable)
    """
    # Drop nulls for sampling
    non_null = series.dropna()

    if len(non_null) == 0:
        return []

    unique_values = non_null.unique()

    # If we have fewer unique values than max_samples, return all
    if len(unique_values) <= max_samples:
        return [_serialize_value(v) for v in unique_values[:max_samples]]

    dtype = series.dtype

    # Numeric columns: include extremes and spread
    if pd.api.types.is_numeric_dtype(dtype):
        return _sample_numeric(non_null, max_samples, include_extremes)

    # Datetime columns: include extremes and spread
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return _sample_datetime(non_null, max_samples, include_extremes)

    # Categorical/string: sample for variety
    return _sample_categorical(non_null, max_samples)


def _sample_numeric(
    series: pd.Series,
    max_samples: int,
    include_extremes: bool,
) -> list[Any]:
    """Sample numeric values with extremes and spread."""
    samples = []

    if include_extremes and max_samples >= 2:
        samples.append(series.min())
        samples.append(series.max())
        max_samples -= 2

    if max_samples > 0:
        # Add median if we have room
        median = series.median()
        if median not in samples:
            samples.append(median)
            max_samples -= 1

    if max_samples > 0:
        # Sample from quantiles for spread
        remaining = series[~series.isin(samples)]
        if len(remaining) > 0:
            quantiles = [0.25, 0.75, 0.1, 0.9]
            for q in quantiles:
                if max_samples <= 0:
                    break
                val = remaining.quantile(q)
                if val not in samples:
                    samples.append(val)
                    max_samples -= 1

    return [_serialize_value(v) for v in samples]


def _sample_datetime(
    series: pd.Series,
    max_samples: int,
    include_extremes: bool,
) -> list[Any]:
    """Sample datetime values with extremes and spread."""
    samples = []

    if include_extremes and max_samples >= 2:
        samples.append(series.min())
        samples.append(series.max())
        max_samples -= 2

    if max_samples > 0:
        # Add middle date
        sorted_dates = series.sort_values()
        mid_idx = len(sorted_dates) // 2
        mid_date = sorted_dates.iloc[mid_idx]
        if mid_date not in samples:
            samples.append(mid_date)
            max_samples -= 1

    if max_samples > 0:
        # Sample spread dates
        remaining = series[~series.isin(samples)].sort_values()
        if len(remaining) > 0:
            indices = np.linspace(0, len(remaining) - 1, max_samples + 2, dtype=int)[1:-1]
            for idx in indices:
                if max_samples <= 0:
                    break
                val = remaining.iloc[idx]
                if val not in samples:
                    samples.append(val)
                    max_samples -= 1

    return [_serialize_value(v) for v in samples]


def _sample_categorical(
    series: pd.Series,
    max_samples: int,
) -> list[Any]:
    """Sample categorical values for variety."""
    # Get value counts and sample diverse values
    value_counts = series.value_counts()

    samples = []

    # Include most common
    if len(value_counts) > 0:
        samples.append(value_counts.index[0])

    # Include least common
    if len(value_counts) > 1 and max_samples >= 2:
        samples.append(value_counts.index[-1])

    # Fill remaining with spread
    remaining_indices = [
        i for i in range(len(value_counts))
        if value_counts.index[i] not in samples
    ]

    if remaining_indices and len(samples) < max_samples:
        # Sample evenly spaced indices
        n_needed = max_samples - len(samples)
        step = max(1, len(remaining_indices) // (n_needed + 1))
        for i in range(0, len(remaining_indices), step):
            if len(samples) >= max_samples:
                break
            samples.append(value_counts.index[remaining_indices[i]])

    return [_serialize_value(v) for v in samples[:max_samples]]


def _serialize_value(value: Any) -> Any:
    """Convert value to JSON-serializable format."""
    if pd.isna(value):
        return None

    # Handle pandas Timestamp
    if hasattr(value, "isoformat"):
        return value.isoformat()

    # Handle numpy types
    if hasattr(value, "item"):
        return value.item()

    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        return value.tolist()

    return value
