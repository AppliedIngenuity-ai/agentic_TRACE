"""
Filter view tool - filters rows in an existing view.
"""

import warnings
from typing import TYPE_CHECKING

import pandas as pd

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session


class FilterViewTool(BaseTool):
    """
    Filter rows in an existing view.

    Creates a new view containing only rows that match the filter condition.
    Supports comparison operators and logical expressions.
    """

    name = "filter_view"
    category = ToolCategory.TRANSFORM

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="source_view",
                param_type="string",
                description="Name of the view to filter",
                required=True,
            ),
            ToolParameter(
                name="output_view",
                param_type="string",
                description="Name for the filtered view",
                required=True,
            ),
            ToolParameter(
                name="condition",
                param_type="string",
                description="Filter condition (e.g., 'price > 100', 'ticker == \"AAPL\"')",
                required=True,
            ),
            ToolParameter(
                name="keep_columns",
                param_type="array",
                description="Optional: Columns to keep in output. If specified, only these columns are included. Default: all columns.",
                required=False,
                items_type="string",
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "source_view", "output_view", "condition"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        source_view = kwargs["source_view"]
        output_view = kwargs["output_view"]
        condition = kwargs["condition"]
        keep_columns = kwargs.get("keep_columns")

        # Validate source view exists
        if error := self.validate_view_exists(session, source_view):
            return self.error(error, ErrorType.VIEW_ERROR)

        # Check output doesn't exist
        if output_view in session.views:
            return self.error(
                f"View '{output_view}' already exists",
                ErrorType.VIEW_ERROR,
            )

        # Get source DataFrame
        df = session.get_dataframe_required(source_view)

        # Validate and apply filter
        try:
            # Suppress FutureWarning about datetime isin with string values
            # (user-provided conditions can't easily be pre-converted)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The behavior of 'isin' with dtype=datetime64",
                    category=FutureWarning,
                )
                filtered_df = df.query(condition)
        except Exception as e:
            return self.error(
                f"Invalid filter condition: {str(e)}",
                ErrorType.EXPRESSION_ERROR,
            )

        # Check if any rows match
        result_warnings = []
        if len(filtered_df) == 0:
            result_warnings.append(f"Filter '{condition}' matched no rows")

        # Filter columns if specified
        if keep_columns:
            missing = [c for c in keep_columns if c not in filtered_df.columns]
            if missing:
                available = list(filtered_df.columns)
                return self.error(
                    f"Columns not found: {missing}. Available: {available}",
                    ErrorType.COLUMN_ERROR,
                )
            filtered_df = filtered_df[keep_columns]

        return self.success(
            view_name=output_view,
            df=filtered_df,
            warnings=result_warnings,
            source_rows=len(df),
            filtered_rows=len(filtered_df),
            condition=condition,
        )

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata
        cond = m.get("condition", "?")
        src_rows = m.get("source_rows", "?")
        filtered = m.get("filtered_rows", "?")
        msg = f"Filtered where {cond} \u2192 '{result.view_name}' ({filtered} of {src_rows} rows)"
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        return (
            "Filter rows in a view based on a condition. "
            "Condition uses pandas query syntax: 'column > value', "
            "'column == \"string\"', 'column.isin([1,2,3])', etc."
        )
