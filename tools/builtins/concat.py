"""
Concat tool - stack views vertically.
"""

from typing import TYPE_CHECKING

import pandas as pd

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session


class ConcatTool(BaseTool):
    """
    Concatenate multiple views vertically.

    Stacks views on top of each other. Views should have compatible columns.
    """

    name = "concat"
    category = ToolCategory.TRANSFORM

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="views",
                param_type="array",
                description="List of view names to concatenate",
                required=True,
                items_type="string",
            ),
            ToolParameter(
                name="output_view",
                param_type="string",
                description="Name for the concatenated view",
                required=True,
            ),
            ToolParameter(
                name="ignore_index",
                param_type="boolean",
                description="Reset index after concatenation (default: True)",
                required=False,
                default=True,
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "views", "output_view"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        views = kwargs["views"]
        output_view = kwargs["output_view"]
        ignore_index = kwargs.get("ignore_index", True)

        # Validate we have at least 2 views
        if len(views) < 2:
            return self.error(
                "Need at least 2 views to concatenate",
                ErrorType.VALIDATION_ERROR,
            )

        # Check output doesn't exist
        if output_view in session.views:
            return self.error(
                f"View '{output_view}' already exists",
                ErrorType.VIEW_ERROR,
            )

        # Validate all views exist
        for view_name in views:
            if error := self.validate_view_exists(session, view_name):
                return self.error(error, ErrorType.VIEW_ERROR)

        # Get DataFrames
        dfs = [session.get_dataframe_required(v) for v in views]

        # Check column compatibility
        first_cols = set(dfs[0].columns)
        warnings = []
        for i, df in enumerate(dfs[1:], 2):
            other_cols = set(df.columns)
            if first_cols != other_cols:
                missing = first_cols - other_cols
                extra = other_cols - first_cols
                warnings.append(
                    f"View {views[i-1]} has different columns: "
                    f"missing={list(missing)}, extra={list(extra)}"
                )

        # Concatenate
        try:
            result_df = pd.concat(dfs, ignore_index=ignore_index)
        except Exception as e:
            return self.error(
                f"Concatenation failed: {str(e)}",
                ErrorType.DATA_ERROR,
            )

        return self.success(
            view_name=output_view,
            df=result_df,
            warnings=warnings,
            source_views=views,
            total_rows=len(result_df),
        )

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata
        views = m.get("source_views", [])
        total = m.get("total_rows", "?")
        msg = f"Concatenated {views} \u2192 '{result.view_name}' ({total} rows)"
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        return (
            "Concatenate multiple views vertically (stack rows). "
            "Views should have compatible columns."
        )
