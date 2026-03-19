"""
Apply window tool - window/rolling functions.
"""

from typing import TYPE_CHECKING

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session


class ApplyWindowTool(BaseTool):
    """
    Apply window functions to a view.

    Creates a new view with rolling/window calculations on specified columns.
    Supports rolling mean, sum, min, max, std, and cumulative functions.
    """

    name = "apply_window"
    category = ToolCategory.TRANSFORM

    ALLOWED_FUNCTIONS = {
        # Rolling functions
        "rolling_mean": ("rolling", "mean"),
        "rolling_sum": ("rolling", "sum"),
        "rolling_min": ("rolling", "min"),
        "rolling_max": ("rolling", "max"),
        "rolling_std": ("rolling", "std"),
        "rolling_var": ("rolling", "var"),
        "rolling_count": ("rolling", "count"),
        # Cumulative functions
        "cumsum": ("cumulative", "cumsum"),
        "cummin": ("cumulative", "cummin"),
        "cummax": ("cumulative", "cummax"),
        "cumprod": ("cumulative", "cumprod"),
        # Shift functions
        "lag": ("shift", None),
        "lead": ("shift", None),
        # Rank functions
        "rank": ("rank", None),
        "pct_change": ("pct_change", None),
    }

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="source_view",
                param_type="string",
                description="Name of the view to apply window to",
                required=True,
            ),
            ToolParameter(
                name="output_view",
                param_type="string",
                description="Name for the output view",
                required=True,
            ),
            ToolParameter(
                name="column",
                param_type="string",
                description="Column to apply window function to",
                required=True,
            ),
            ToolParameter(
                name="function",
                param_type="string",
                description="Window function to apply",
                required=True,
                enum=list(self.ALLOWED_FUNCTIONS.keys()),
            ),
            ToolParameter(
                name="output_column",
                param_type="string",
                description="Name for the result column",
                required=True,
            ),
            ToolParameter(
                name="window",
                param_type="integer",
                description="Window size for rolling functions",
                required=False,
            ),
            ToolParameter(
                name="partition_by",
                param_type="array",
                description="Columns to partition/group by before applying window",
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="order_by",
                param_type="string",
                description="Column to order by before applying window",
                required=False,
            ),
            ToolParameter(
                name="periods",
                param_type="integer",
                description="Number of periods for lag/lead (default: 1)",
                required=False,
                default=1,
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "source_view", "output_view", "column", "function", "output_column"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        source_view = kwargs["source_view"]
        output_view = kwargs["output_view"]
        column = kwargs["column"]
        function = kwargs["function"]
        output_column = kwargs["output_column"]
        window = kwargs.get("window")
        partition_by = kwargs.get("partition_by")
        order_by = kwargs.get("order_by")
        periods = kwargs.get("periods", 1)

        # Validate function
        if function not in self.ALLOWED_FUNCTIONS:
            return self.error(
                f"Unknown function: '{function}'. Allowed: {list(self.ALLOWED_FUNCTIONS.keys())}",
                ErrorType.VALIDATION_ERROR,
            )

        # Validate window for rolling functions
        func_type = self.ALLOWED_FUNCTIONS[function][0]
        if func_type == "rolling" and not window:
            return self.error(
                f"Window size required for rolling function '{function}'",
                ErrorType.VALIDATION_ERROR,
            )

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
        df = session.get_dataframe_required(source_view).copy()

        # Validate column exists
        if error := self.validate_columns_exist(df, column):
            return self.error(error, ErrorType.COLUMN_ERROR)

        # Validate partition_by columns if specified
        if partition_by:
            if error := self.validate_columns_exist(df, *partition_by):
                return self.error(error, ErrorType.COLUMN_ERROR)

        # Validate order_by column if specified
        if order_by:
            if error := self.validate_columns_exist(df, order_by):
                return self.error(error, ErrorType.COLUMN_ERROR)

        # Apply window function
        try:
            result_series = self._apply_window(
                df, column, function, window, partition_by, order_by, periods
            )
            df[output_column] = result_series
        except Exception as e:
            return self.error(
                f"Window function failed: {str(e)}",
                ErrorType.DATA_ERROR,
            )

        return self.success(
            view_name=output_view,
            df=df,
            function=function,
            input_column=column,
            output_column=output_column,
        )

    def _apply_window(self, df, column, function, window, partition_by, order_by, periods):
        """Apply the window function to the DataFrame."""
        func_type, agg_func = self.ALLOWED_FUNCTIONS[function]

        # Sort if order_by specified
        if order_by:
            if partition_by:
                df = df.sort_values(partition_by + [order_by])
            else:
                df = df.sort_values(order_by)

        series = df[column]

        if partition_by:
            grouped = df.groupby(partition_by)[column]

            if func_type == "rolling":
                return grouped.transform(lambda x: getattr(x.rolling(window), agg_func)())
            elif func_type == "cumulative":
                # Direct cumulative methods (cumsum, cummin, cummax, cumprod)
                return grouped.transform(lambda x: getattr(x, agg_func)())
            elif func_type == "shift":
                shift_periods = periods if function == "lag" else -periods
                return grouped.transform(lambda x: x.shift(shift_periods))
            elif func_type == "rank":
                return grouped.transform(lambda x: x.rank())
            elif func_type == "pct_change":
                return grouped.transform(lambda x: x.pct_change(periods=periods))
        else:
            if func_type == "rolling":
                return getattr(series.rolling(window), agg_func)()
            elif func_type == "cumulative":
                # Direct cumulative methods (cumsum, cummin, cummax, cumprod)
                return getattr(series, agg_func)()
            elif func_type == "shift":
                shift_periods = periods if function == "lag" else -periods
                return series.shift(shift_periods)
            elif func_type == "rank":
                return series.rank()
            elif func_type == "pct_change":
                return series.pct_change(periods=periods)

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata
        rows = len(result.dataframe) if result.dataframe is not None else "?"
        func = m.get("function", "?")
        in_col = m.get("input_column", "?")
        out_col = m.get("output_column", "?")
        msg = f"Applied {func}({in_col}) \u2192 '{result.view_name}'.{out_col} ({rows} rows)"
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        return (
            "Apply window functions: rolling_mean, rolling_sum, cumsum, cumprod, lag, lead, rank, pct_change. "
            "Use partition_by for per-group calculations, order_by for ordering."
        )
