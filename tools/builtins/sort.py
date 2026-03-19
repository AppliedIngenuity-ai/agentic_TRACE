"""
Sort tool - sort and limit rows.
"""

from typing import TYPE_CHECKING

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session


class SortTool(BaseTool):
    """
    Sort rows in a view and optionally limit results.

    Creates a new view with rows sorted by specified columns.
    Supports both global limit and per-partition limit for "top N per group" patterns.
    """

    name = "sort"
    category = ToolCategory.TRANSFORM

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="source_view",
                param_type="string",
                description="Name of the view to sort",
                required=True,
            ),
            ToolParameter(
                name="output_view",
                param_type="string",
                description="Name for the sorted view",
                required=True,
            ),
            ToolParameter(
                name="by",
                param_type="array",
                description="Columns to sort by",
                required=True,
                items_type="string",
            ),
            ToolParameter(
                name="ascending",
                param_type="boolean",
                description="Sort ascending (default: True, set False for descending)",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="limit",
                param_type="integer",
                description="Global limit on number of rows (use limit_per_partition for top N per group)",
                required=False,
            ),
            ToolParameter(
                name="partition_by",
                param_type="string",
                description="Column to partition by for per-group limits",
                required=False,
            ),
            ToolParameter(
                name="limit_per_partition",
                param_type="integer",
                description="Number of rows to keep per partition (requires partition_by)",
                required=False,
            ),
            ToolParameter(
                name="finalize",
                param_type="boolean",
                description="Mark this view as final output",
                required=False,
                default=False,
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "source_view", "output_view", "by"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        source_view = kwargs["source_view"]
        output_view = kwargs["output_view"]
        by = kwargs["by"]
        ascending = kwargs.get("ascending", True)
        limit = kwargs.get("limit")
        partition_by = kwargs.get("partition_by")
        limit_per_partition = kwargs.get("limit_per_partition")
        finalize = kwargs.get("finalize", False)

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

        # Validate sort columns exist
        if error := self.validate_columns_exist(df, *by):
            return self.error(error, ErrorType.COLUMN_ERROR)

        # Validate partition column if specified
        if partition_by:
            if error := self.validate_columns_exist(df, partition_by):
                return self.error(error, ErrorType.COLUMN_ERROR)

        # Check for potential limit vs limit_per_partition mistake
        warnings = []
        if limit and not partition_by and not limit_per_partition:
            # If data has multiple rows per group but limit is small
            # Try to detect if user probably meant limit_per_partition
            candidates = session.domain_config.partition_candidates or ['id', 'category', 'group']
            for col in candidates:
                if col in df.columns:
                    n_groups = df[col].nunique()
                    n_rows = len(df)
                    if n_rows > 2 * n_groups and limit <= n_groups:
                        warnings.append(
                            f"View has {n_rows} rows across {n_groups} {col}s but limit={limit}. "
                            f"Did you mean limit_per_partition={limit} with partition_by='{col}'?"
                        )
                    break

        # Sort
        try:
            sorted_df = df.sort_values(by=by, ascending=ascending)

            # Handle partitioned limit
            if partition_by and limit_per_partition:
                # Group by partition, take top N from each
                sorted_df = (
                    sorted_df.groupby(partition_by, group_keys=False)
                    .head(limit_per_partition)
                )
            elif limit:
                # Global limit
                sorted_df = sorted_df.head(limit)

        except Exception as e:
            return self.error(
                f"Sort failed: {str(e)}",
                ErrorType.DATA_ERROR,
            )

        # Create result
        result = self.success(
            view_name=output_view,
            df=sorted_df,
            sorted_by=by,
            ascending=ascending,
            limited_to=limit,
            partition_by=partition_by,
            limit_per_partition=limit_per_partition,
            warnings=warnings if warnings else None,
        )

        # Handle finalize flag - orchestrator will finalize if requested
        if finalize:
            result.metadata["finalize"] = True

        return result

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata
        rows = len(result.dataframe) if result.dataframe is not None else "?"
        by = m.get("sorted_by", "?")
        asc = m.get("ascending", True)
        direction = "ASC" if asc else "DESC"
        msg = f"Sorted by {by} {direction}"
        if m.get("limited_to"):
            msg += f", limit {m['limited_to']}"
        if m.get("partition_by"):
            msg += f", partition by {m['partition_by']}"
        if m.get("limit_per_partition"):
            msg += f", {m['limit_per_partition']} per partition"
        msg += f" \u2192 '{result.view_name}' ({rows} rows)"
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        return (
            "Sort rows by columns and optionally limit results. "
            "Use limit for global top N. Use limit_per_partition with partition_by "
            "for top N per group. "
            "Set finalize=True to mark as final output."
        )
