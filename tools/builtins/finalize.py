"""
Finalize tool - mark a view as final output.
"""

from typing import TYPE_CHECKING

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session


class FinalizeTool(BaseTool):
    """
    Mark a view as final output.

    Signals that a view contains the requested result and should be
    returned to the user. Multiple views can be finalized for complex queries.
    """

    name = "finalize"
    category = ToolCategory.OUTPUT

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="view_name",
                param_type="string",
                description="Name of the view to finalize",
                required=True,
            ),
            ToolParameter(
                name="summary",
                param_type="string",
                description="Brief summary of what this view contains",
                required=False,
            ),
            ToolParameter(
                name="keep_columns",
                param_type="array",
                description="Optional: Columns to keep in final output. If specified, only these columns are included. Default: all columns.",
                required=False,
                items_type="string",
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "view_name"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        view_name = kwargs["view_name"]
        summary = kwargs.get("summary", "")
        keep_columns = kwargs.get("keep_columns")

        # Validate view exists
        if error := self.validate_view_exists(session, view_name):
            return self.error(error, ErrorType.VIEW_ERROR)

        # Get view info
        df = session.get_dataframe_required(view_name)
        meta = session.get_view_required(view_name)

        # Filter columns if specified
        if keep_columns:
            missing = [c for c in keep_columns if c not in df.columns]
            if missing:
                available = list(df.columns)
                return self.error(
                    f"Columns not found: {missing}. Available: {available}",
                    ErrorType.COLUMN_ERROR,
                )
            df = df[keep_columns]
            # Update the view with filtered columns
            session.update_view(view_name, df, operation="finalize (column filtered)")

        # Mark as finalized
        session.finalize_view(view_name)

        return self.success(
            view_name=view_name,
            df=df,  # Include df so orchestrator can access it
            finalized=True,
            rows=len(df),
            columns=list(df.columns),
            summary=summary,
        )

    def summarize(self, result: ToolResult) -> str:
        """Custom summary for finalize tool."""
        if not result.success:
            return f"Error: {result.error}"

        view = result.view_name
        rows = result.metadata.get("rows", "?")
        summary = result.metadata.get("summary", "")

        msg = f"Finalized view '{view}' ({rows} rows)"
        if summary:
            msg += f": {summary}"
        return msg

    def tool_description(self) -> str:
        return (
            "Mark a view as final output to return to the user. "
            "Add a summary to describe what the view contains."
        )
