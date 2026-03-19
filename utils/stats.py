"""
Column statistics computation.

Computes summary statistics for DataFrame columns to provide
metadata to the LLM without exposing raw data.
"""

from typing import Any

import pandas as pd
import numpy as np

from ..core.view import ColumnStats
from .sampling import sample_column_values


def compute_column_stats(
    df: pd.DataFrame,
    max_sample_size: int = 5,
) -> dict[str, ColumnStats]:
    """
    Compute statistics for all columns in a DataFrame.

    For each column, computes:
        - dtype: Data type
        - unique: Count of unique values
        - nulls: Count of null values
        - total: Total row count
        - min/max: For numeric and datetime columns
        - mean: For numeric columns
        - sample: Representative sample values

    Args:
        df: DataFrame to analyze
        max_sample_size: Maximum sample values per column

    Returns:
        Dict mapping column name to ColumnStats
    """
    stats = {}

    for col in df.columns:
        series = df[col]
        stats[col] = compute_single_column_stats(series, max_sample_size)

    return stats


def compute_single_column_stats(
    series: pd.Series,
    max_sample_size: int = 5,
) -> ColumnStats:
    """
    Compute statistics for a single column.

    Args:
        series: Pandas Series
        max_sample_size: Maximum sample values

    Returns:
        ColumnStats object
    """
    dtype = _get_dtype_string(series.dtype)
    unique = series.nunique()
    nulls = int(series.isna().sum())
    total = len(series)

    # Compute min/max/mean for appropriate types
    min_val = None
    max_val = None
    mean_val = None

    non_null = series.dropna()

    if len(non_null) > 0:
        if pd.api.types.is_numeric_dtype(series.dtype):
            min_val = _safe_value(non_null.min())
            max_val = _safe_value(non_null.max())
            mean_val = _safe_value(non_null.mean())

        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            min_val = _safe_value(non_null.min())
            max_val = _safe_value(non_null.max())

    # Get sample values
    sample = sample_column_values(series, max_sample_size)

    return ColumnStats(
        name=series.name or "",
        dtype=dtype,
        unique=unique,
        nulls=nulls,
        total=total,
        min=min_val,
        max=max_val,
        mean=mean_val,
        sample=sample,
    )


def _get_dtype_string(dtype: Any) -> str:
    """Convert pandas dtype to readable string."""
    dtype_str = str(dtype)

    # Simplify common types
    if "int" in dtype_str:
        return "integer"
    if "float" in dtype_str:
        return "float"
    if "datetime" in dtype_str:
        return "datetime"
    if "date" in dtype_str:
        return "date"
    if "bool" in dtype_str:
        return "boolean"
    if "object" in dtype_str or "string" in dtype_str:
        return "string"
    if "category" in dtype_str:
        return "category"

    return dtype_str


def _safe_value(value: Any) -> Any:
    """Convert value to safe JSON-serializable format."""
    if pd.isna(value):
        return None

    # Handle pandas Timestamp
    if hasattr(value, "isoformat"):
        return value.isoformat()

    # Handle numpy types
    if hasattr(value, "item"):
        return value.item()

    # Handle infinity
    if isinstance(value, float):
        if np.isinf(value):
            return None
        if np.isnan(value):
            return None

    return value


def summarize_dataframe(df: pd.DataFrame, include_stats: bool = True) -> str:
    """
    Generate human-readable summary of a DataFrame.

    Args:
        df: DataFrame to summarize
        include_stats: Whether to include column statistics

    Returns:
        Formatted summary string
    """
    lines = [f"DataFrame: {len(df)} rows x {len(df.columns)} columns"]

    if include_stats:
        stats = compute_column_stats(df)
        lines.append("\nColumns:")

        for name, col_stats in stats.items():
            type_info = col_stats.dtype
            if col_stats.nulls > 0:
                null_pct = (col_stats.nulls / col_stats.total) * 100
                type_info += f", {null_pct:.1f}% null"

            lines.append(f"  - {name} ({type_info})")

            if col_stats.min is not None:
                range_str = f"range: {col_stats.min} to {col_stats.max}"
                if col_stats.mean is not None:
                    range_str += f", mean: {col_stats.mean:.2f}"
                lines.append(f"      {range_str}")

            if col_stats.sample:
                sample_str = ", ".join(str(v) for v in col_stats.sample[:3])
                lines.append(f"      sample: [{sample_str}]")

    return "\n".join(lines)
