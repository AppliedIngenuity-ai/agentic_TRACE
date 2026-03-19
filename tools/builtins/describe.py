"""
Describe tool - inspect view or data source schema and statistics.
"""

from typing import TYPE_CHECKING

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType
from ...utils.stats import compute_column_stats

if TYPE_CHECKING:
    from ...core.session import Session
    from ...data.sources import DataSource


class DescribeTool(BaseTool):
    """
    Describe a view's or data source's schema and statistics.

    Returns detailed information about columns, types, and value distributions.
    Use to understand the structure and content of a view or data source.
    """

    name = "describe"
    category = ToolCategory.EXPLORE

    def __init__(self, data_sources: list["DataSource"] | None = None):
        """
        Initialize with available data sources.

        Args:
            data_sources: List of data sources (optional)
        """
        super().__init__()
        self.data_sources = data_sources or []
        self._source_map: dict[str, "DataSource"] = {}

    def set_data_sources(self, sources: list["DataSource"]) -> None:
        """Set available data sources."""
        self.data_sources = sources
        self._source_map = {s.name: s for s in sources}

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="view_name",
                param_type="string",
                description="Name of the view OR data source to describe",
                required=True,
            ),
            ToolParameter(
                name="columns",
                param_type="array",
                description="Specific columns to describe (optional, default: all)",
                required=False,
                items_type="string",
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "view_name"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        name = kwargs["view_name"]
        columns = kwargs.get("columns")
        is_data_source = False

        # Check if it's a view first
        if name in session.views:
            df = session.get_dataframe_required(name)
        # Then check if it's a data source
        elif name in self._source_map:
            is_data_source = True
            source = self._source_map[name]
            try:
                # Sample data from the source (limit to 1000 rows for stats)
                df = source.sample(limit=1000)
            except Exception as e:
                return self.error(
                    f"Failed to sample data source '{name}': {str(e)}",
                    ErrorType.DATA_ERROR,
                )
        else:
            # Neither view nor data source found
            available_views = list(session.views.keys())
            available_sources = list(self._source_map.keys())
            return self.error(
                f"'{name}' not found. Available views: {available_views}. "
                f"Available data sources: {available_sources}",
                ErrorType.VIEW_ERROR,
            )

        # Filter columns if specified
        if columns:
            if error := self.validate_columns_exist(df, *columns):
                return self.error(error, ErrorType.COLUMN_ERROR)
            df = df[columns]

        # Compute statistics
        stats = compute_column_stats(df)

        # Format for response
        column_info = {
            col_name: col_stats.to_dict()
            for col_name, col_stats in stats.items()
        }

        result_metadata = {
            "rows": len(df),
            "columns": column_info,
        }

        if is_data_source:
            result_metadata["is_data_source"] = True
            result_metadata["note"] = "Statistics based on sample of up to 1000 rows"

        return self.success(
            view_name=name,
            **result_metadata,
        )

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata
        name = result.view_name or "?"
        rows = m.get("rows", "?")
        col_info = m.get("columns", {})
        n_cols = len(col_info) if isinstance(col_info, dict) else "?"
        col_names = list(col_info.keys()) if isinstance(col_info, dict) else []
        msg = f"Described '{name}': {rows} rows, {n_cols} columns"
        if col_names:
            preview = col_names[:5]
            if len(col_names) > 5:
                preview.append("...")
            msg += f" {preview}"
        if m.get("is_data_source"):
            msg += " (data source, sampled)"
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        return (
            "Get schema and statistics for a view OR data source. "
            "Shows column types, unique counts, null counts, min/max, and sample values."
        )
