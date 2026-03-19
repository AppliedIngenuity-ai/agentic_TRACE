"""
Pivot tool - pivot table transformation.
"""

from typing import TYPE_CHECKING

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session


class PivotTool(BaseTool):
    """
    Create a pivot table from a view.

    Reshapes data by creating columns from unique values of a specified column.
    """

    name = "pivot"
    category = ToolCategory.TRANSFORM

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="source_view",
                param_type="string",
                description="Name of the view to pivot",
                required=True,
            ),
            ToolParameter(
                name="output_view",
                param_type="string",
                description="Name for the pivoted view",
                required=True,
            ),
            ToolParameter(
                name="index",
                param_type="array",
                description="Columns to use as row index",
                required=True,
                items_type="string",
            ),
            ToolParameter(
                name="columns",
                param_type="string",
                description="Column whose unique values become new columns",
                required=True,
            ),
            ToolParameter(
                name="values",
                param_type="string",
                description="Column whose values fill the pivot table",
                required=True,
            ),
            ToolParameter(
                name="aggfunc",
                param_type="string",
                description="Aggregation function if duplicates exist (default: mean)",
                required=False,
                default="mean",
                enum=["sum", "mean", "count", "min", "max", "first", "last"],
            ),
            ToolParameter(
                name="finalize",
                param_type="boolean",
                description="Set finalize=true to return this view. You can return one, many, or none if there is no valid solution.",
                required=False,
                default=False,
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "source_view", "output_view", "index", "columns", "values"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        source_view = kwargs["source_view"]
        output_view = kwargs["output_view"]
        index = kwargs["index"]
        columns = kwargs["columns"]
        values = kwargs["values"]
        aggfunc = kwargs.get("aggfunc", "mean")

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

        # Validate columns exist
        all_cols = index + [columns, values]
        if error := self.validate_columns_exist(df, *all_cols):
            return self.error(error, ErrorType.COLUMN_ERROR)

        # Perform pivot
        try:
            pivoted = df.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc,
            ).reset_index()

            # Flatten column names if multi-level
            if hasattr(pivoted.columns, 'levels'):
                pivoted.columns = [
                    '_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else col
                    for col in pivoted.columns
                ]

        except Exception as e:
            return self.error(
                f"Pivot failed: {str(e)}",
                ErrorType.DATA_ERROR,
            )

        finalize = kwargs.get("finalize", False)

        return self.success(
            view_name=output_view,
            df=pivoted,
            rows=len(pivoted),
            result_columns=list(pivoted.columns),
            source_rows=len(df),
            result_rows=len(pivoted),
            num_columns=len(pivoted.columns),
            index_col=index,
            columns_param=columns,
            values_col=values,
            aggfunc=aggfunc,
            finalize=finalize if finalize else None,
        )

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata
        rows = m.get("result_rows", "?")
        cols = m.get("num_columns", "?")
        idx = m.get("index_col", "?")
        col_param = m.get("columns_param", "?")
        vals = m.get("values_col", "?")
        agg = m.get("aggfunc", "mean")
        msg = f"Pivoted index={idx}, columns={col_param}, values={vals}, aggfunc={agg} \u2192 '{result.view_name}' ({rows} rows, {cols} cols)"
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        return (
            "Create a pivot table. Specify index (rows), columns (becomes column headers), "
            "and values (cell values). Use aggfunc for duplicate handling."
        )
