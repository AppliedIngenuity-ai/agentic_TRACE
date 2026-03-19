"""
Tool result types for structured success/error/warning handling.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ToolResult:
    """
    Structured result from tool execution.

    Tools return this to indicate success/failure with optional warnings.
    The orchestrator uses this to decide whether to save views and what
    to report back to the LLM.

    Attributes:
        success: Whether the tool execution succeeded
        view_name: Name of the created view (if any)
        dataframe: The resulting DataFrame (if any)
        metadata: Additional metadata about the result
        warnings: List of warning messages (success can still have warnings)
        error: Error message if success=False
        error_type: Error category for programmatic handling

    Example success with warnings:
        >>> result = ToolResult(
        ...     success=True,
        ...     view_name="filtered_stocks",
        ...     dataframe=df,
        ...     warnings=["No data found for ticker XYZ"]
        ... )

    Example error:
        >>> result = ToolResult(
        ...     success=False,
        ...     error="Invalid date range: start > end",
        ...     error_type="VALIDATION_ERROR"
        ... )
    """

    success: bool

    # On success - view/data created
    view_name: str | None = None
    dataframe: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Warnings (can occur even on success)
    warnings: list[str] = field(default_factory=list)

    # On failure
    error: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize result for LLM response.

        On error: Returns {"error": ..., "error_type": ...}
        On success: Returns view info, row count, metadata, and any warnings
        """
        if not self.success:
            return {
                "error": self.error,
                "error_type": self.error_type,
            }

        result: dict[str, Any] = {}

        if self.view_name:
            result["view"] = self.view_name

        if self.dataframe is not None:
            result["rows"] = len(self.dataframe)
            result["columns"] = list(self.dataframe.columns)

        # Include additional metadata
        result.update(self.metadata)

        if self.warnings:
            result["warnings"] = self.warnings

        return result

    def __bool__(self) -> bool:
        """Allow using result in boolean context: if result: ..."""
        return self.success


# Common error types for consistency
class ErrorType:
    """Standard error type constants."""

    VALIDATION_ERROR = "VALIDATION_ERROR"  # Bad input parameters
    DATA_ERROR = "DATA_ERROR"              # Data not found or invalid
    EXPRESSION_ERROR = "EXPRESSION_ERROR"  # Invalid expression syntax
    VIEW_ERROR = "VIEW_ERROR"              # View doesn't exist or invalid
    COLUMN_ERROR = "COLUMN_ERROR"          # Column doesn't exist
    TYPE_ERROR = "TYPE_ERROR"              # Type mismatch
    PERMISSION_ERROR = "PERMISSION_ERROR"  # Not allowed
    INTERNAL_ERROR = "INTERNAL_ERROR"      # Unexpected error
