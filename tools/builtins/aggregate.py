"""
Aggregate tool - group aggregation and window functions.

Combines group-by aggregation (sum, mean, count, ...) with window/rolling
functions (rolling_mean, cumsum, lag, rank, ...) in a single tool.
"""

from typing import TYPE_CHECKING

import pandas as pd

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session
    from ...data.sources import DataSource


class AggregateTool(BaseTool):
    """
    Group-by aggregation and window functions.

    Two modes based on which parameters are provided:

    1. **Group aggregation** (aggregations is set):
       Groups by specified columns and computes aggregate functions.
       aggregations={"close": "mean", "volume": "sum"}

    2. **Window function** (function is set):
       Applies rolling, cumulative, shift, or rank functions.
       function="rolling_mean", column="close", window=20
    """

    name = "aggregate"
    category = ToolCategory.TRANSFORM

    AGG_FUNCTIONS = [
        "sum", "mean", "count", "min", "max", "std", "var",
        "first", "last", "nunique",
    ]

    WINDOW_FUNCTIONS = {
        # Rolling
        "rolling_mean": ("rolling", "mean"),
        "rolling_sum": ("rolling", "sum"),
        "rolling_min": ("rolling", "min"),
        "rolling_max": ("rolling", "max"),
        "rolling_std": ("rolling", "std"),
        "rolling_var": ("rolling", "var"),
        "rolling_count": ("rolling", "count"),
        # Cumulative
        "cumsum": ("cumulative", "cumsum"),
        "cummin": ("cumulative", "cummin"),
        "cummax": ("cumulative", "cummax"),
        "cumprod": ("cumulative", "cumprod"),
        # Shift
        "lag": ("shift", None),
        "lead": ("shift", None),
        # Rank/pct
        "rank": ("rank", None),
        "pct_change": ("pct_change", None),
        # Transform (broadcast group aggregate to all rows — no row reduction)
        "first": ("transform", "first"),
        "last": ("transform", "last"),
        "mean": ("transform", "mean"),
        "std": ("transform", "std"),
        "min": ("transform", "min"),
        "max": ("transform", "max"),
        "sum": ("transform", "sum"),
    }

    def __init__(self):
        super().__init__()
        self._db_source: "DataSource | None" = None

    def set_db_source(self, source: "DataSource") -> None:
        """Set DB source for named equation lookups."""
        self._db_source = source

    # ------------------------------------------------------------------
    # Overridable hint methods — subclass to provide domain-specific examples.
    # Base returns generic text; domain subclasses (e.g. StockAggregateTool)
    # override with concrete, LLM-friendly examples.
    # ------------------------------------------------------------------

    def _hint_mode_required(self) -> str:
        """Hint appended to 'must specify aggregations or function' error."""
        return ""

    def _hint_low_row_warning(self) -> str:
        """Hint when group aggregation produced very few rows."""
        return ""

    def _hint_partition_by(self) -> str:
        """Example text for partition_by parameter description."""
        return ""

    def _hint_tool_examples(self) -> str:
        """Domain-specific examples appended to tool_description()."""
        return ""

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="source_view",
                param_type="string",
                description="Source view to operate on",
                required=True,
            ),
            ToolParameter(
                name="output_view",
                param_type="string",
                description="Name for the output view (default: source_view name, auto-incremented if taken)",
                required=False,
            ),
            # -- Group aggregation params --
            ToolParameter(
                name="group_by",
                param_type="array",
                description=(
                    "Columns to group by (for aggregation mode). "
                    "Use [] for global aggregate."
                ),
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="aggregations",
                param_type="object",
                description=(
                    "Column-to-function map for group aggregation. "
                    "Output columns are always named col_func: "
                    "{'col_a': 'mean'} -> col_a_mean. "
                    "{'col_a': ['first', 'last']} -> col_a_first, col_a_last. "
                    f"Functions: {', '.join(self.AGG_FUNCTIONS)}. "
                    "IMPORTANT: first/last are order-dependent — use sort_by to control row ordering. "
                    "sort_by='date' + 'first' = earliest row; sort_by='date' + 'last' = latest row. "
                    "WARNING: do NOT use sort_by='-date' with 'last' — that gives the EARLIEST row (oldest), not the latest. "
                    "To get both start and end in one call use {'col': ['first', 'last']} with sort_by= — this produces col_first (earliest) and col_last (latest) in one step with no join needed. "
                    "To rename auto-generated output columns, use rename={'col_last': 'new_name'}."
                ),
                required=False,
            ),
            ToolParameter(
                name="sort_by",
                param_type="string",
                description=(
                    "Column to sort by before aggregating (default: 'date' if it exists). "
                    "Required for order-dependent functions like first/last. "
                    "Prefix with '-' for descending (e.g., '-date' for newest first)."
                ),
                required=False,
            ),
            # -- Per-row function params (keeps all rows, adds a column) --
            ToolParameter(
                name="function",
                param_type="string",
                description=(
                    "Function to apply per row (keeps all rows, adds a new column). "
                    "Must be used together with column= and output_column=. "
                    f"Built-in: {', '.join(self.WINDOW_FUNCTIONS.keys())}. "
                    "mean/std/min/max/sum broadcast the group aggregate to all rows (useful for normalization). "
                    "Or a named equation from the equations table (e.g., MOVING_AVERAGE)."
                ),
                required=False,
            ),
            ToolParameter(
                name="column",
                param_type="string",
                description=(
                    "Column to operate on. Must be a numeric column from the source view. "
                    "Required when using function=. "
                    "Set to '' if using aggregations= instead."
                ),
                required=True,
            ),
            ToolParameter(
                name="output_column",
                param_type="string",
                description="Result column name (used with function=)",
                required=False,
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
                description=(
                    "Columns to partition by when using function= "
                    "(e.g., per-entity calculation)"
                ),
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="order_by",
                param_type="string",
                description="Column to sort by before applying function=",
                required=False,
            ),
            ToolParameter(
                name="periods",
                param_type="integer",
                description="Periods for lag/lead/pct_change (default: 1)",
                required=False,
                default=1,
            ),
            # -- Shared params --
            ToolParameter(
                name="keep_columns",
                param_type="array",
                description="Columns to keep in output (default: all)",
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="rename",
                param_type="object",
                description=(
                    "Rename output columns: {old_name: new_name}. "
                    "Applied after aggregation. Useful to clean up auto-generated names like "
                    "'pct_increase_last' → 'pct_increase': rename={'pct_increase_last': 'pct_increase'}."
                ),
                required=False,
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
        # Unify sort_by and order_by — LLMs use them interchangeably.
        # Ensure both are set so group mode (uses sort_by) and window mode
        # (uses order_by) both work regardless of which the LLM passed.
        sort_by = kwargs.get("sort_by")
        order_by = kwargs.get("order_by")
        if sort_by and not order_by:
            # Strip direction prefix for order_by (order_by is always ascending)
            kwargs["order_by"] = sort_by.strip().lstrip("-") if isinstance(sort_by, str) else sort_by
        elif order_by and not sort_by:
            kwargs["sort_by"] = order_by

        if not kwargs.get("output_view"):
            kwargs["output_view"] = kwargs.get("source_view", "output")
        if error := self.validate_required(kwargs, "source_view", "output_view"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        aggregations = kwargs.get("aggregations") or None  # treat {} as not provided
        function = kwargs.get("function")

        # Determine mode
        if aggregations and function:
            return self.error(
                "Specify either 'aggregations' or 'function', not both.",
                ErrorType.VALIDATION_ERROR,
            )
        if aggregations:
            return self._execute_group(session, kwargs)
        if function:
            return self._execute_window(session, kwargs)

        # Neither specified — need one or the other
        window_fns = list(self.WINDOW_FUNCTIONS.keys())
        msg = (
            "Missing required parameter: specify 'aggregations' or 'function' (with column= and output_column=). "
            f"Available functions: {', '.join(window_fns)}."
        )
        hint = self._hint_mode_required()
        if hint:
            msg += f" {hint}"
        return self.error(msg, ErrorType.VALIDATION_ERROR)

    # ── Group aggregation mode ──────────────────────────────────────────

    def _execute_group(self, session: "Session", kwargs: dict) -> ToolResult:
        source_view = kwargs["source_view"]
        output_view = kwargs["output_view"]
        group_by = kwargs.get("group_by", [])
        aggregations = kwargs["aggregations"]
        keep_columns = kwargs.get("keep_columns")
        sort_by = kwargs.get("sort_by")

        if error := self.validate_view_exists(session, source_view):
            return self.error(error, ErrorType.VIEW_ERROR)

        # Auto-rename
        original_name = output_view
        renamed = False
        if output_view in session.views:
            output_view = self._generate_unique_name(output_view, session)
            renamed = True

        df = session.get_dataframe_required(source_view)

        # Validate group_by
        if group_by:
            if error := self.validate_columns_exist(df, *group_by):
                return self.error(error, ErrorType.COLUMN_ERROR)

        # Validate aggregation columns and functions
        warnings = []

        # Sort data before aggregation (critical for first/last)
        has_order_dependent = any(
            f in ("first", "last")
            for func in aggregations.values()
            for f in (func if isinstance(func, list) else [func])
        )
        if sort_by:
            # Strip surrounding quotes the LLM sometimes adds (e.g. '"date"' -> 'date')
            sort_by = sort_by.strip().strip("\"'")
            descending = sort_by.startswith("-")
            sort_col = sort_by.lstrip("-")
            if sort_col in df.columns:
                df = df.sort_values(sort_col, ascending=not descending)
                # Warn about the common mistake: sort descending + 'last' = oldest row (not newest)
                if descending and has_order_dependent:
                    uses_last = any(
                        f == "last"
                        for func in aggregations.values()
                        for f in (func if isinstance(func, list) else [func])
                    )
                    if uses_last:
                        warnings.append(
                            f"sort_by='{sort_by}' (descending) with 'last' aggregation returns the OLDEST row "
                            f"(last row after descending sort = smallest {sort_col} value). "
                            f"To get the LATEST row use sort_by='{sort_col}' (ascending) with 'last', "
                            f"or sort_by='{sort_by}' (descending) with 'first'."
                        )
            else:
                return self.error(
                    f"sort_by column '{sort_col}' not found. Available: {list(df.columns)}",
                    ErrorType.COLUMN_ERROR,
                )
        elif has_order_dependent:
            # Auto-detect date column as default sort
            for candidate in ("date", "Date", "timestamp", "Timestamp"):
                if candidate in df.columns:
                    df = df.sort_values(candidate)
                    warnings.append(f"Auto-sorted by '{candidate}' for first/last aggregation. Use sort_by to control ordering.")
                    break

        for col, func in aggregations.items():
            if col not in df.columns:
                return self.error(
                    f"Column '{col}' not found. Available: {list(df.columns)}",
                    ErrorType.COLUMN_ERROR,
                )
            funcs = func if isinstance(func, list) else [func]
            for f in funcs:
                if f not in self.AGG_FUNCTIONS:
                    return self.error(
                        f"Unknown function: '{f}'. Allowed: {self.AGG_FUNCTIONS}",
                        ErrorType.VALIDATION_ERROR,
                    )

        if renamed:
            warnings.append(f"View '{original_name}' already exists, renamed to '{output_view}'")

        try:
            if group_by:
                result_df = df.groupby(group_by, as_index=False).agg(aggregations)
            else:
                # Global aggregate — manual for pandas 2.x compat
                result_data = {}
                for col, func in aggregations.items():
                    funcs = func if isinstance(func, list) else [func]
                    for f in funcs:
                        if f == "first":
                            val = df[col].iloc[0]
                        elif f == "last":
                            val = df[col].iloc[-1]
                        else:
                            val = df[col].agg(f)
                        col_name = f"{col}_{f}"
                        result_data[col_name] = [val]
                result_df = pd.DataFrame(result_data)

            if isinstance(result_df.columns, pd.MultiIndex):
                result_df.columns = ['_'.join(col).strip('_') for col in result_df.columns]

            # Rename single-function columns to col_func for consistent naming
            for col, func in aggregations.items():
                if isinstance(func, str) and col in result_df.columns:
                    result_df = result_df.rename(columns={col: f"{col}_{func}"})

        except Exception as e:
            return self.error(f"Aggregation failed: {e}", ErrorType.DATA_ERROR)

        if result_df.empty:
            return self.error("Aggregation produced no results", ErrorType.DATA_ERROR)

        if keep_columns:
            missing = [c for c in keep_columns if c not in result_df.columns]
            if missing:
                return self.error(
                    f"Columns not found: {missing}. Available: {list(result_df.columns)}",
                    ErrorType.COLUMN_ERROR,
                )
            result_df = result_df[keep_columns]

        rename = kwargs.get("rename")
        if rename:
            missing = [k for k in rename if k not in result_df.columns]
            if missing:
                return self.error(
                    f"rename: columns not found: {missing}. Available: {list(result_df.columns)}",
                    ErrorType.COLUMN_ERROR,
                )
            result_df = result_df.rename(columns=rename)

        finalize = kwargs.get("finalize", False)

        # Hint: if the result is tiny relative to source, the LLM likely wants to join it back
        # or should have used window mode instead
        if len(result_df) <= 5 and len(df) > len(result_df):
            result_cols = [c for c in result_df.columns if c not in (group_by or [])]
            msg = (
                f"Result has {len(result_df)} row(s) reduced from {len(df)}. "
                f"Output columns ({result_cols}) are in '{output_view}', NOT the source view."
            )
            hint = self._hint_low_row_warning()
            if hint:
                msg += f" {hint}"
            warnings.append(msg)

        return self.success(
            view_name=output_view,
            df=result_df,
            rows=len(result_df),
            result_columns=list(result_df.columns),
            groups=len(result_df),
            source_rows=len(df),
            aggregations=aggregations,
            group_by=group_by,
            finalize=finalize if finalize else None,
            warnings=warnings if warnings else None,
        )

    # ── Window function mode ────────────────────────────────────────────

    def _execute_window(self, session: "Session", kwargs: dict) -> ToolResult:
        source_view = kwargs["source_view"]
        output_view = kwargs["output_view"]
        function = kwargs["function"]
        column = kwargs.get("column") or None  # treat '' as None
        output_column = kwargs.get("output_column")
        window = kwargs.get("window")
        partition_by = kwargs.get("partition_by")
        order_by = kwargs.get("order_by")
        periods = kwargs.get("periods", 1)

        # sort_by controls rank direction: sort_by='-close' → rank 1 = highest close
        sort_by = kwargs.get("sort_by", "")
        if isinstance(sort_by, str) and sort_by.strip().startswith("-"):
            rank_ascending = False   # rank 1 = largest value
        else:
            rank_ascending = True    # rank 1 = smallest value (default)

        # Normalize comma-separated function (e.g. "cumsum,cumsum" -> "cumsum")
        if isinstance(function, str) and "," in function:
            fn_parts = [f.strip() for f in function.split(",") if f.strip()]
            unique_fns = list(dict.fromkeys(fn_parts))
            if len(unique_fns) == 1:
                function = unique_fns[0]
            else:
                return self.error(
                    f"Only one function can be specified at a time. Got: {fn_parts}. "
                    "To apply different functions, call aggregate separately for each.",
                    ErrorType.VALIDATION_ERROR,
                )

        # Normalize comma-separated column into a list (e.g. "pv_sum,volume_sum")
        if isinstance(column, str) and "," in column:
            column = [c.strip() for c in column.split(",") if c.strip()]

        # Resolve named equation if ALL_CAPS
        resolved_function = function
        if function.isupper():
            resolved = self._resolve_equation(function)
            if isinstance(resolved, ToolResult):
                return resolved
            # Override with resolved values
            resolved_function = resolved["function"]
            if not column and resolved.get("default_column"):
                column = resolved["default_column"]
            if not window and resolved.get("default_window"):
                window = resolved["default_window"]
            if not order_by and resolved.get("default_sort_by"):
                order_by = resolved["default_sort_by"]
            if not output_column:
                output_column = function.lower()

        # Validate function name
        if resolved_function not in self.WINDOW_FUNCTIONS:
            return self.error(
                f"Unknown function: '{resolved_function}'. "
                f"Available: {list(self.WINDOW_FUNCTIONS.keys())}",
                ErrorType.VALIDATION_ERROR,
            )

        if not column:
            df_check = session.get_dataframe(kwargs["source_view"])
            available = list(df_check.columns) if df_check is not None else []

            # Check if output_column name hints at intended column(s)
            suggestions = []
            if isinstance(output_column, str) and df_check is not None:
                suggestions = [c for c in available if c in output_column and len(c) > 1]
                suggestions.sort(key=len, reverse=True)

            hint = ""
            if suggestions:
                examples = ", ".join(f"column='{c}'" for c in suggestions)
                hint = f" Did you intend any of: {examples}? If so, re-call with that column specified."

            return self.error(
                f"'column' is required when using function='{resolved_function}'.{hint} "
                f"Available columns: {available}",
                ErrorType.VALIDATION_ERROR,
            )
        if not output_column and not isinstance(column, list):
            output_column = f"{column}_{resolved_function}"

        func_type = self.WINDOW_FUNCTIONS[resolved_function][0]
        if func_type == "rolling" and not window:
            return self.error(
                f"'window' size required for rolling function '{resolved_function}'",
                ErrorType.VALIDATION_ERROR,
            )

        if error := self.validate_view_exists(session, source_view):
            return self.error(error, ErrorType.VIEW_ERROR)

        # Auto-rename
        original_name = output_view
        renamed = False
        if output_view in session.views:
            output_view = self._generate_unique_name(output_view, session)
            renamed = True

        df = session.get_dataframe_required(source_view).copy()

        cols_to_validate = column if isinstance(column, list) else [column]
        if error := self.validate_columns_exist(df, *cols_to_validate):
            return self.error(error, ErrorType.COLUMN_ERROR)
        if partition_by:
            if error := self.validate_columns_exist(df, *partition_by):
                return self.error(error, ErrorType.COLUMN_ERROR)
        if order_by:
            if error := self.validate_columns_exist(df, order_by):
                return self.error(error, ErrorType.COLUMN_ERROR)

        warnings = []
        if renamed:
            warnings.append(f"View '{original_name}' already exists, renamed to '{output_view}'")


        # Auto-detect missing partition_by using domain config
        if not partition_by:
            candidates = session.domain_config.partition_candidates or []
            for cand in candidates:
                if cand in df.columns and df[cand].nunique() > 1:
                    label = session.domain_config.partition_label
                    warnings.append(
                        f"Column '{cand}' has {df[cand].nunique()} unique values but "
                        f"partition_by is not set. Results may mix groups ({label}). "
                        f"Consider: partition_by=['{cand}']"
                    )
                    break

        try:
            if isinstance(column, list):
                for col in column:
                    col_out = f"{col}_{resolved_function}"
                    df[col_out] = self._apply_window(
                        df, col, resolved_function, window, partition_by, order_by, periods, rank_ascending
                    )
                output_column = ", ".join(f"{c}_{resolved_function}" for c in column)
            else:
                result_series = self._apply_window(
                    df, column, resolved_function, window, partition_by, order_by, periods, rank_ascending
                )
                df[output_column] = result_series
        except Exception as e:
            return self.error(f"Window function failed: {e}", ErrorType.DATA_ERROR)

        rename = kwargs.get("rename")
        if rename:
            missing = [k for k in rename if k not in df.columns]
            if missing:
                return self.error(
                    f"rename: columns not found: {missing}. Available: {list(df.columns)}",
                    ErrorType.COLUMN_ERROR,
                )
            df = df.rename(columns=rename)

        finalize = kwargs.get("finalize", False)

        return self.success(
            view_name=output_view,
            df=df,
            rows=len(df),
            result_columns=list(df.columns),
            function=function,
            resolved_function=resolved_function if resolved_function != function else None,
            input_column=column,
            output_column=output_column,
            finalize=finalize if finalize else None,
            warnings=warnings if warnings else None,
        )

    def _apply_window(self, df, column, function, window, partition_by, order_by, periods, rank_ascending=True):
        """Apply window function to DataFrame column."""
        _, agg_func = self.WINDOW_FUNCTIONS[function]

        if order_by:
            sort_cols = (partition_by or []) + [order_by]
            df.sort_values(sort_cols, inplace=True)

        series = df[column]

        if partition_by:
            grouped = df.groupby(partition_by)[column]

            if self.WINDOW_FUNCTIONS[function][0] == "rolling":
                return grouped.transform(lambda x: getattr(x.rolling(window), agg_func)())
            elif self.WINDOW_FUNCTIONS[function][0] == "cumulative":
                return grouped.transform(lambda x: getattr(x, agg_func)())
            elif self.WINDOW_FUNCTIONS[function][0] == "shift":
                shift_periods = periods if function == "lag" else -periods
                return grouped.transform(lambda x: x.shift(shift_periods))
            elif self.WINDOW_FUNCTIONS[function][0] == "rank":
                return grouped.transform(lambda x: x.rank(ascending=rank_ascending))
            elif self.WINDOW_FUNCTIONS[function][0] == "pct_change":
                return grouped.transform(lambda x: x.pct_change(periods=periods))
            elif self.WINDOW_FUNCTIONS[function][0] == "transform":
                return grouped.transform(agg_func)
        else:
            func_type = self.WINDOW_FUNCTIONS[function][0]
            if func_type == "rolling":
                return getattr(series.rolling(window), agg_func)()
            elif func_type == "cumulative":
                return getattr(series, agg_func)()
            elif func_type == "shift":
                shift_periods = periods if function == "lag" else -periods
                return series.shift(shift_periods)
            elif func_type == "rank":
                return series.rank(ascending=rank_ascending)
            elif func_type == "pct_change":
                return series.pct_change(periods=periods)
            elif func_type == "transform":
                if agg_func == "first":
                    val = series.iloc[0]
                elif agg_func == "last":
                    val = series.iloc[-1]
                else:
                    val = getattr(series, agg_func)()
                return pd.Series(val, index=series.index)

    def _resolve_equation(self, name: str) -> "dict | ToolResult":
        """Resolve a named equation from the equations table."""
        if not self._db_source:
            # No DB source — just return error
            return self.error(
                f"Named equation '{name}' cannot be resolved (no equation database configured). "
                f"Use a built-in function: {list(self.WINDOW_FUNCTIONS.keys())}",
                ErrorType.VALIDATION_ERROR,
            )

        try:
            import json
            sql = f"SELECT type, rule FROM equations WHERE name = '{name}'"
            df = self._db_source.query(sql)

            if df.empty:
                return self.error(
                    f"Equation '{name}' not found. "
                    f"Built-in functions: {list(self.WINDOW_FUNCTIONS.keys())}",
                    ErrorType.DATA_ERROR,
                )

            row = df.iloc[0]
            eq_type = row["type"]

            if eq_type not in ("window", "aggregate"):
                return self.error(
                    f"Equation '{name}' is type '{eq_type}', not window/aggregate. "
                    "Use create_view(add_columns=...) for column equations.",
                    ErrorType.VALIDATION_ERROR,
                )

            rule = row["rule"]
            if isinstance(rule, str):
                rule = json.loads(rule)

            # Map equation pattern to built-in function name
            pattern = rule.get("pattern", "")
            func_name = rule.get("function", "")

            if pattern == "rolling":
                resolved_func = f"rolling_{func_name}" if func_name else "rolling_mean"
            elif pattern == "cumulative":
                resolved_func = f"cum{func_name}" if func_name else "cumsum"
            elif pattern == "ewm":
                # EWM maps to rolling for our purposes — but not directly supported
                # Fallback to rolling_mean as approximation
                resolved_func = "rolling_mean"
            elif pattern == "shift":
                resolved_func = "lag"
            elif pattern == "pct_change":
                resolved_func = "pct_change"
            else:
                return self.error(
                    f"Equation '{name}' has unsupported pattern '{pattern}'",
                    ErrorType.VALIDATION_ERROR,
                )

            # Extract defaults from parameters
            params = rule.get("parameters", [])
            defaults = {}
            for p in params:
                if p.get("default") is not None:
                    if p["name"] == "column":
                        defaults["default_column"] = p["default"]
                    elif p["name"] in ("window", "span"):
                        defaults["default_window"] = p["default"]
                    elif p["name"] == "sort_by":
                        defaults["default_sort_by"] = p["default"]

            # Equations can also specify sort_by at the rule level
            if rule.get("sort_by"):
                defaults["default_sort_by"] = rule["sort_by"]

            return {"function": resolved_func, **defaults}

        except Exception as e:
            if isinstance(e, (KeyError,)):
                raise
            return self.error(f"Equation lookup failed: {e}", ErrorType.DATA_ERROR)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _generate_unique_name(self, base_name: str, session: "Session") -> str:
        counter = 2
        new_name = f"{base_name}_{counter}"
        while new_name in session.views:
            counter += 1
            new_name = f"{base_name}_{counter}"
        return new_name

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata

        # Window mode
        if m.get("function"):
            rows = len(result.dataframe) if result.dataframe is not None else "?"
            func = m.get("function", "?")
            in_col = m.get("input_column", "?")
            out_col = m.get("output_column", "?")
            msg = f"Applied {func}({in_col}) → '{result.view_name}'.{out_col} ({rows} rows)"
            parts = [msg]
            if result.warnings:
                parts.append(f"Warnings: {'; '.join(result.warnings)}")
            return " | ".join(parts)

        # Group mode
        groups = m.get("groups", "?")
        src_rows = m.get("source_rows", "?")
        aggs = m.get("aggregations", {})
        group_by = m.get("group_by", [])
        agg_parts = [f"{col}={func}" for col, func in aggs.items()]
        agg_str = ", ".join(agg_parts) if agg_parts else "?"
        if group_by:
            msg = f"Aggregated by {group_by}, {agg_str} → '{result.view_name}' ({groups} groups from {src_rows} rows)"
        else:
            msg = f"Aggregated (global) {agg_str} → '{result.view_name}' ({groups} rows from {src_rows} rows)"
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        base = (
            "Aggregate data. Two modes — use ONE, not both: "
            "(1) aggregations={...}: REDUCES rows to one per group. Output columns named col_func (e.g. {'value':'mean'} → value_mean). "
            "(2) function=: KEEPS ALL ROWS, adds a new column. Requires column= and output_column=. "
            "Do not pass aggregations={} when using function=."
        )
        examples = self._hint_tool_examples()
        if examples:
            base += f" {examples}"
        return base
