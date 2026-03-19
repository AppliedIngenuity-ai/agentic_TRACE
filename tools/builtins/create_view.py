"""
Create view tool - creates a new view from a data source or existing view.

Supports filtering, sorting, computed columns, and finalization in a single call.
"""

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session
    from ...data.sources import DataSource


class CreateViewTool(BaseTool):
    """
    Create a new view from a data source.

    Creates a view by querying a data source with optional filters.
    The view is added to the session and can be further transformed.
    """

    name = "create_view"
    category = ToolCategory.TRANSFORM

    def __init__(self, data_sources: list["DataSource"] | None = None):
        """
        Initialize with available data sources.

        Args:
            data_sources: List of data sources to query
        """
        super().__init__()
        self.data_sources = data_sources or []
        self._source_map: dict[str, "DataSource"] = {}

    # ------------------------------------------------------------------
    # Overridable hint methods — subclass to provide domain-specific examples.
    # ------------------------------------------------------------------

    def _hint_window_example(self, kind: str) -> str:
        """Domain-specific example for window function hints.

        Args:
            kind: One of "first_last", "lag_lead", "generic", "shift_keyword"
        """
        return ""

    def set_data_sources(self, sources: list["DataSource"]) -> None:
        """Set available data sources."""
        self.data_sources = sources
        self._source_map = {s.name: s for s in sources}

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="view_name",
                param_type="string",
                description="Name for the new view",
                required=True,
            ),
            ToolParameter(
                name="source",
                param_type="string",
                description="Data source or existing view name to query from",
                required=True,
            ),
            ToolParameter(
                name="columns",
                param_type="array",
                description="Optional: Columns to keep in output. If specified, only these columns are included. Default: all columns. Use to filter/select specific columns.",
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="filters",
                param_type="object",
                description="EQUALITY and IN filters only — {\"ticker\": \"AAPL\"} matches rows where ticker is exactly 'AAPL'; {\"ticker\": [\"AAPL\",\"MSFT\"]} matches a list. Do NOT use filters for date or numeric ranges — use where= instead: where=\"date >= '2021-01-01'\" or where=\"close > 100\".",
                required=False,
            ),
            ToolParameter(
                name="where",
                param_type="string",
                description="Filter expression for date ranges and comparisons. Examples: \"date >= '2021-01-01'\", \"close > 100\", \"date >= '2021-01-01' and date <= '2024-12-31'\", \"ticker != 'AAPL'\". Do not include the word WHERE.",
                required=False,
            ),
            ToolParameter(
                name="sort_by",
                param_type="string",
                description="Column to sort by before applying limit. Prefix with '-' for descending (e.g., 'date' for oldest first, '-date' for newest first).",
                required=False,
            ),
            ToolParameter(
                name="limit",
                param_type="integer",
                description="Maximum rows to return (combine with sort_by to get first/last N rows)",
                required=False,
            ),
            ToolParameter(
                name="add_columns",
                param_type="object",
                description=(
                    "Computed columns to add: {column_name: expression}. "
                    "Supports arithmetic (e.g., '(close - open) / open * 100'), "
                    "SQL functions (CORR, SQRT, ABS, ROUND, FLOOR, CEIL, CAST, COALESCE, CASE WHEN), "
                    "and named equations (e.g., 'DAILY_RETURN_PCT')."
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
        # Auto-generate view_name if missing
        if not kwargs.get("view_name") and kwargs.get("source"):
            source = kwargs["source"]
            base = source.replace("stock_prices", "view").replace(" ", "_")
            # Make unique
            name = base
            counter = 1
            while name in session.views:
                counter += 1
                name = f"{base}_{counter}"
            kwargs["view_name"] = name

        # Validate required params
        if error := self.validate_required(kwargs, "view_name", "source"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        view_name = kwargs["view_name"]
        source_name = kwargs["source"]
        columns = kwargs.get("columns")
        filters = kwargs.get("filters", {})
        where_clause = kwargs.get("where")
        sort_by = kwargs.get("sort_by")
        if sort_by:
            sort_by = sort_by.strip().strip("\"'")
        limit = kwargs.get("limit")

        # Parse string-valued dicts (LLMs often serialize objects as strings)
        if isinstance(filters, str):
            try:
                import ast
                filters = ast.literal_eval(filters)
                kwargs["filters"] = filters
            except (ValueError, SyntaxError):
                filters = {}
        add_columns_raw = kwargs.get("add_columns")
        if isinstance(add_columns_raw, str):
            try:
                import ast
                kwargs["add_columns"] = ast.literal_eval(add_columns_raw)
            except (ValueError, SyntaxError):
                pass

        # Parse where clause if LLM sent it as a dict (e.g., "{'date': '>= 2025-01-01'}")
        if where_clause and isinstance(where_clause, str):
            try:
                import ast
                parsed = ast.literal_eval(where_clause)
                if isinstance(parsed, dict):
                    # Convert dict to SQL WHERE clause
                    parts = []
                    for col, condition in parsed.items():
                        cond_str = str(condition).strip()
                        # Extract operator and value
                        import re as _re
                        m = _re.match(r'^(>=|<=|!=|>|<|=)\s*(.+)$', cond_str)
                        if m:
                            op, val = m.group(1), m.group(2).strip()
                            # Quote non-numeric values
                            if not val.replace('.', '', 1).lstrip('-').isdigit():
                                val = f"'{val}'"
                            parts.append(f"{col} {op} {val}")
                        else:
                            parts.append(f"{col} = '{cond_str}'")
                    where_clause = " AND ".join(parts)
                    kwargs["where"] = where_clause
            except (ValueError, SyntaxError):
                pass  # It's a normal string clause, use as-is

        original_name = view_name
        renamed = False
        if view_name in session.views:
            suggested = self._generate_unique_name(view_name, session)
            return self.error(
                f"VIEW NOT CREATED: View '{view_name}' already exists. "
                f"Tool call failed, not executed — try again with a different view_name "
                f"(e.g. '{suggested}'), or use the existing view directly.",
                ErrorType.VALIDATION_ERROR,
            )

        # Check if source is a data source or an existing view
        source = self._source_map.get(source_name)
        from_view = False

        if source:
            # Source is a data source - use SQL query
            # When add_columns is present, SELECT * and defer column selection
            # (requested columns may not exist in the source — they get created by add_columns)
            sql_columns = columns
            if columns and kwargs.get("add_columns"):
                sql_columns = None  # SELECT * — column filtering happens after add_columns
            try:
                sql = self._build_query(source, sql_columns, filters, where_clause, sort_by, limit)
                df = source.query(sql)
            except Exception as e:
                return self.error(
                    f"Query failed: {str(e)}",
                    ErrorType.DATA_ERROR,
                )
        elif source_name in session.views:
            # Source is an existing view - operate on DataFrame directly
            from_view = True
            df = session.get_dataframe(source_name).copy()

            # Column selection is deferred until after where/filters/sort/add_columns
            # so that where clauses can reference columns not in the final output.

            # Apply filters (skip any column already covered by where_clause)
            if filters and where_clause:
                where_lower = where_clause.lower()
                filters = {k: v for k, v in filters.items()
                           if self._parse_filter_key(k)[0].lower() not in where_lower}
            if filters:
                for key, val in filters.items():
                    col, op = self._parse_filter_key(key)
                    if col not in df.columns:
                        continue
                    if isinstance(val, (list, tuple)):
                        if op == "!=":
                            df = df[~df[col].isin(val)]
                        else:
                            df = df[df[col].isin(val)]
                    elif isinstance(val, str) and re.match(r'^\d{4}-\d{2}-\d{2},\d{4}-\d{2}-\d{2}$', val.strip()):
                        start, end = val.strip().split(",")
                        df = df[(df[col] >= start) & (df[col] <= end)]
                    elif op == "=":
                        df = df[df[col] == val]
                    elif op == "!=":
                        df = df[df[col] != val]
                    elif op == ">":
                        df = df[df[col] > val]
                    elif op == ">=":
                        df = df[df[col] >= val]
                    elif op == "<":
                        df = df[df[col] < val]
                    elif op == "<=":
                        df = df[df[col] <= val]

            # Apply where clause using pandas query
            if where_clause:
                # Handle SQL LIKE patterns directly (no eval — safe string matching)
                df, like_applied = self._apply_sql_like(df, where_clause)
                if not like_applied:
                    try:
                        df = df.query(where_clause)
                    except Exception as e:
                        # Try SQL→pandas conversions as fallback
                        converted = self._convert_sql_in_to_pandas(where_clause)
                        converted = self._convert_sql_eq_to_pandas(converted)
                        converted = self._convert_sql_logic_to_pandas(converted)
                        if converted != where_clause:
                            try:
                                df = df.query(converted)
                            except Exception:
                                return self.error(
                                    f"Where clause failed: {str(e)}",
                                    ErrorType.DATA_ERROR,
                                )
                        else:
                            return self.error(
                                f"Where clause failed: {str(e)}",
                                ErrorType.DATA_ERROR,
                            )


            # Apply sorting
            if sort_by:
                descending = sort_by.startswith("-")
                col_name = sort_by.lstrip("-")
                if col_name not in df.columns:
                    return self.error(
                        f"Sort column '{col_name}' not found. Available: {list(df.columns)}",
                        ErrorType.COLUMN_ERROR,
                    )
                df = df.sort_values(col_name, ascending=not descending)

            # Apply limit
            if limit:
                df = df.head(limit)
        else:
            available_sources = list(self._source_map.keys())
            available_views = list(session.views.keys())
            return self.error(
                f"Source '{source_name}' not found. Data sources: {available_sources}. Views: {available_views}",
                ErrorType.DATA_ERROR,
            )

        if df.empty:
            return self.error(
                f"No data returned from source '{source_name}' with given filters",
                ErrorType.DATA_ERROR,
            )

        # Check for missing filter values (data-dependent warning)
        warnings = []

        missing_values = self._check_filter_coverage(df, filters)
        if missing_values:
            for col, missing in missing_values.items():
                warnings.append(f"Filter value(s) not found in '{col}': {missing}")

        # Apply add_columns (computed columns)
        add_columns = kwargs.get("add_columns")
        added_cols = []
        if add_columns and isinstance(add_columns, dict):
            for col_name, expression in add_columns.items():
                result = self._add_computed_column(df, col_name, expression)
                if isinstance(result, ToolResult):
                    return result  # Error
                df = result
                added_cols.append(col_name)

        # Apply column selection last — after where/filters/sort/add_columns
        # so that where clauses can reference columns not in the final output.
        if columns:
            missing_cols = [c for c in columns if c not in df.columns]
            if missing_cols:
                return self.error(
                    f"Columns not found: {missing_cols}. Available: {list(df.columns)}",
                    ErrorType.COLUMN_ERROR,
                )
            df = df[columns]

        # Handle finalize flag
        finalize = kwargs.get("finalize", False)

        # Return success - session.add_view is called by orchestrator
        return self.success(
            view_name=view_name,
            df=df,
            source=source_name,
            from_view=from_view,
            rows=len(df),
            result_columns=list(df.columns),
            filters_applied=bool(filters or where_clause),
            where_clause=where_clause,
            sort_by=sort_by,
            limit=limit,
            filters=filters if filters else None,
            selected_columns=columns,
            added_columns=added_cols if added_cols else None,
            finalize=finalize if finalize else None,
            warnings=warnings if warnings else None,
        )

    @staticmethod
    def _convert_sql_eq_to_pandas(where_clause: str) -> str:
        """Convert bare SQL = equality operator to pandas == operator.

        Only replaces = that is NOT already part of >=, <=, !=, ==.
        Handles spaces around the operator (e.g. 'high = high_max' and 'high=high_max').
        Does not touch string literals because the only = signs inside quoted values
        (e.g. "ticker = 'A=B'") are after a quote char, which is not a word char,
        but since we only target = flanked by word chars or spaces this is safe in practice.

        Examples:
            "high = high_max"      -> "high == high_max"
            "high=high_max"        -> "high==high_max"
            "date >= '2021-01-01'" -> unchanged  (>= is protected)
            "close != 0"           -> unchanged  (!= is protected)
            "rank == 1"            -> unchanged  (already ==)
        """
        # Match = that is not preceded by <, >, !, = and not followed by =
        return re.sub(r'(?<![<>!=])=(?!=)', '==', where_clause)

    @staticmethod
    def _convert_sql_in_to_pandas(where_clause: str) -> str:
        """Convert SQL IN (...) / NOT IN (...) syntax to pandas in [...] syntax.

        e.g. "ticker IN ('NVDA', 'AMD')"  →  "ticker in ['NVDA', 'AMD']"
             "ticker NOT IN ('A')"         →  "ticker not in ['A']"
        Only applied as a fallback when pandas .query() raises a syntax error.
        """
        pattern = re.compile(r'(\w+)\s+(NOT\s+IN|IN)\s*\(([^)]+)\)', re.IGNORECASE)

        def _replace(m):
            col = m.group(1).strip()
            not_kw = "not in" if m.group(2).upper().startswith("NOT") else "in"
            inner = m.group(3)
            return f"{col} {not_kw} [{inner}]"

        return pattern.sub(_replace, where_clause)

    @staticmethod
    def _convert_sql_logic_to_pandas(where_clause: str) -> str:
        """Convert SQL AND/OR keywords to Python and/or for pandas .query().

        Only replaces whole-word uppercase AND/OR outside of quoted strings.
        e.g. "industry == 'A' OR industry == 'B'" → "industry == 'A' or industry == 'B'"
        """
        result = re.sub(r'\bAND\b', 'and', where_clause)
        result = re.sub(r'\bOR\b', 'or', result)
        return result

    @staticmethod
    def _apply_sql_like(df, where_clause: str):
        """Apply SQL LIKE patterns as direct pandas filtering (no eval).

        Handles patterns like: column LIKE '%value%' with OR/AND connectors.
        Returns (filtered_df, True) if LIKE patterns were found and applied,
        or (df, False) if no LIKE patterns detected.

        Security: values are used only as literal string matches via
        str.contains(literal, regex=False) — no eval or code execution.
        """
        like_pattern = re.compile(
            r"(\w+)\s+LIKE\s+'([^']*)'",
            re.IGNORECASE,
        )
        matches = list(like_pattern.finditer(where_clause))
        if not matches:
            return df, False

        # Check if all conditions are OR-connected or AND-connected
        # Strip out LIKE clauses to see what connectors remain
        stripped = like_pattern.sub('', where_clause).strip()
        # Remove leading/trailing connectors
        stripped = re.sub(r'^\s*(and|or)\s*', '', stripped, flags=re.IGNORECASE)
        stripped = re.sub(r'\s*(and|or)\s*$', '', stripped, flags=re.IGNORECASE)

        # If there's leftover text beyond connectors, this is a mixed clause we can't handle
        remaining = re.sub(r'\b(and|or)\b', '', stripped, flags=re.IGNORECASE).strip()
        if remaining:
            return df, False

        # Determine connector (default OR if ambiguous)
        use_or = 'OR' in where_clause.upper() or len(matches) == 1

        # Build mask from each LIKE condition
        mask = None
        for m in matches:
            col = m.group(1).strip()
            pattern_val = m.group(2)
            if col not in df.columns:
                continue
            # Strip SQL wildcards and use as literal substring match
            inner = pattern_val.strip('%')
            if not inner:
                continue
            # Safe: regex=False ensures no regex injection, case=False for case-insensitive
            condition = df[col].astype(str).str.contains(inner, case=False, na=False, regex=False)
            if mask is None:
                mask = condition
            elif use_or:
                mask = mask | condition
            else:
                mask = mask & condition

        if mask is not None:
            return df[mask], True
        return df, False

    def _generate_unique_name(self, base_name: str, session: "Session") -> str:
        """Generate a unique view name by appending a number."""
        counter = 2
        new_name = f"{base_name}_{counter}"
        while new_name in session.views:
            counter += 1
            new_name = f"{base_name}_{counter}"
        return new_name

    def _check_filter_coverage(
        self,
        df,
        filters: dict[str, Any],
    ) -> dict[str, list[Any]]:
        """
        Check if all filter values are present in the result.

        Returns dict of column -> list of missing values.
        Only checks equality filters (not comparison operators).
        """
        missing = {}

        for key, val in filters.items():
            col, op = self._parse_filter_key(key)

            # Only check equality filters for missing values
            if op != "=":
                continue

            if col not in df.columns:
                continue

            if isinstance(val, (list, tuple)):
                # Check each value in the list
                found_values = set(df[col].dropna().unique())
                missing_vals = [v for v in val if v not in found_values]
                if missing_vals:
                    missing[col] = missing_vals
            elif val is not None:
                # Single value filter — coerce val to column dtype before comparing
                # (e.g. date columns are datetime objects, filter value may be a string)
                col_vals = df[col].dropna()
                found = val in col_vals.values
                if not found:
                    try:
                        coerced = col_vals.dtype.type(val)
                        found = coerced in col_vals.values
                    except Exception:
                        try:
                            coerced_series = pd.Series([val])
                            found = pd.to_datetime(coerced_series).iloc[0] in col_vals.values
                        except Exception:
                            pass
                if not found:
                    missing[col] = [val]

        return missing

    # Supported comparison operators in filter keys
    FILTER_OPERATORS = [">=", "<=", "!=", "<>", ">", "<"]

    def _parse_filter_key(self, key: str) -> tuple[str, str]:
        """
        Parse a filter key to extract column name and operator.

        Examples:
            "ticker" -> ("ticker", "=")
            "date>=" -> ("date", ">=")
            "price<" -> ("price", "<")
        """
        for op in self.FILTER_OPERATORS:
            if key.endswith(op):
                return key[:-len(op)], op
        return key, "="

    def _build_query(
        self,
        source: "DataSource",
        columns: list[str] | None,
        filters: dict[str, Any],
        where_clause: str | None,
        sort_by: str | None,
        limit: int | None,
    ) -> str:
        """Build SQL query from parameters."""
        table = getattr(source, "table", source.name)

        # SELECT clause
        if columns:
            select = ", ".join(columns)
        else:
            select = "*"

        # WHERE clause
        conditions = []

        # If where_clause covers a column already in filters, drop the filter to avoid
        # conflicting conditions (e.g. filters={"date":"2025-03-02"} + where="date >= '2025-03-02'"
        # would AND to a single-day equality, ignoring the range).
        if filters and where_clause:
            where_lower = where_clause.lower()
            filters = {k: v for k, v in filters.items()
                       if self._parse_filter_key(k)[0].lower() not in where_lower}

        if filters:
            for key, val in filters.items():
                col, op = self._parse_filter_key(key)

                if isinstance(val, str):
                    # Check if value starts with an operator (e.g., ">= 2025-01-01")
                    op_match = re.match(r'^(>=|<=|!=|>|<)\s*(.+)$', val.strip())
                    if op_match:
                        extracted_op = op_match.group(1)
                        extracted_val = op_match.group(2).strip()
                        # Strip surrounding quotes the LLM may have added (e.g., "'2025-01-01'" -> "2025-01-01")
                        if len(extracted_val) >= 2 and extracted_val[0] in ("'", '"') and extracted_val[0] == extracted_val[-1]:
                            extracted_val = extracted_val[1:-1]
                        extracted_val = extracted_val.replace("'", "''")
                        # Only leave unquoted if purely numeric
                        try:
                            float(extracted_val)
                            conditions.append(f"{col} {extracted_op} {extracted_val}")
                        except ValueError:
                            conditions.append(f"{col} {extracted_op} '{extracted_val}'")
                    elif re.match(r'^\d{4}-\d{2}-\d{2},\d{4}-\d{2}-\d{2}$', val.strip()):
                        # Range shorthand "YYYY-MM-DD,YYYY-MM-DD" -> >= start AND <= end
                        start, end = val.strip().split(",")
                        conditions.append(f"{col} >= '{start}' AND {col} <= '{end}'")
                    else:
                        escaped = val.replace("'", "''")
                        conditions.append(f"{col} {op} '{escaped}'")
                elif isinstance(val, (list, tuple)):
                    # IN clause (only makes sense with = operator)
                    vals = ", ".join(
                        f"'{v}'" if isinstance(v, str) else str(v)
                        for v in val
                    )
                    if op == "!=":
                        conditions.append(f"{col} NOT IN ({vals})")
                    else:
                        conditions.append(f"{col} IN ({vals})")
                elif val is None:
                    if op == "!=":
                        conditions.append(f"{col} IS NOT NULL")
                    else:
                        conditions.append(f"{col} IS NULL")
                else:
                    conditions.append(f"{col} {op} {val}")

        if where_clause:
            conditions.append(f"({where_clause})")

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        # ORDER BY clause
        order_clause = ""
        if sort_by:
            if sort_by.startswith("-"):
                order_clause = f"ORDER BY {sort_by[1:]} DESC"
            else:
                order_clause = f"ORDER BY {sort_by} ASC"

        # LIMIT clause
        limit_clause = f"LIMIT {limit}" if limit else ""

        return f"SELECT {select} FROM {table} {where} {order_clause} {limit_clause}"

    def _add_computed_column(
        self, df: pd.DataFrame, col_name: str, expression: str
    ) -> "pd.DataFrame | ToolResult":
        """Add a computed column using df.eval() or DuckDB for SQL expressions.

        Validates the expression (same approach as AddColumnTool), then computes.
        Returns modified DataFrame on success, ToolResult error on failure.
        """
        # Route SQL CASE WHEN expressions through DuckDB instead of df.eval()
        if re.search(r'\bCASE\b', expression, re.IGNORECASE) and re.search(r'\bWHEN\b', expression, re.IGNORECASE):
            return self._add_column_via_duckdb(df, col_name, expression)

        # Route known SQL function calls through DuckDB (not supported by df.eval()).
        # Explicit whitelist — do NOT expand to arbitrary functions (DuckDB can read local files).
        _duckdb_sql_functions = {
            # String
            'substr', 'substring', 'length', 'len', 'upper', 'lower', 'trim',
            'ltrim', 'rtrim', 'replace', 'concat', 'lpad', 'rpad', 'regexp_extract',
            # Date/time
            'strftime', 'strptime', 'date_trunc', 'date_part', 'date_diff',
            'year', 'month', 'day', 'hour', 'minute', 'second', 'epoch',
            'to_timestamp', 'make_date',
            # Math
            'floor', 'ceil', 'ceiling', 'mod', 'power', 'pow', 'sqrt', 'abs',
            'ln', 'log', 'log2', 'log10', 'exp', 'sign', 'round',
            # Statistics
            'corr', 'stddev', 'stddev_pop', 'stddev_samp', 'variance', 'var_pop', 'var_samp',
            # Type conversion
            'cast', 'try_cast',
            # Conditional
            'coalesce', 'nullif', 'ifnull',
        }
        # Aliases for common function names LLMs may use
        _function_aliases = {
            'correlation': 'corr',
            'stdev': 'stddev',
            'square_root': 'sqrt',
            'absolute': 'abs',
            'date_subtract': 'date_diff',
        }
        _func_call = re.search(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(', expression)
        if _func_call:
            func_lower = _func_call.group(1).lower()
            # Rewrite aliases to canonical DuckDB names
            if func_lower in _function_aliases:
                original = _func_call.group(1)
                canonical = _function_aliases[func_lower]
                expression = re.sub(r'\b' + original + r'\b', canonical, expression, flags=re.IGNORECASE)
                func_lower = canonical
            if func_lower in _duckdb_sql_functions:
                return self._add_column_via_duckdb(df, col_name, expression)

        # Validate expression — whitelist allowed characters
        # Includes [] and quotes for list/string operations (e.g., `ticker.isin(["AAPL"])`)
        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            "_+-*/()%. []'\"~&|<>=!,@"
        )
        if not all(c in allowed_chars for c in expression):
            disallowed = [c for c in expression if c not in allowed_chars]
            return self.error(
                f"Expression '{expression}' contains disallowed characters: {disallowed}",
                ErrorType.EXPRESSION_ERROR,
            )

        # Check that referenced columns exist
        # Strip string literals first so words inside quotes (e.g. 'NASDAQ 100') are not
        # mistaken for column references.
        expression_no_strings = re.sub(r"'[^']*'", "''", expression)
        expression_no_strings = re.sub(r'"[^"]*"', '""', expression_no_strings)
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression_no_strings)
        builtins = {'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None',
                     'abs', 'round', 'min', 'max', 'sqrt'}
        indicator_keywords = {'ema', 'sma', 'rsi', 'macd', 'bollinger', 'atr', 'moving', 'rolling'}
        window_keywords = {'shift', 'rolling', 'cumsum', 'pct_change', 'rank', 'lag', 'lead', 'ewm', 'iloc'}
        aggregate_keywords = {'first', 'last', 'mean', 'std', 'var', 'sum', 'count', 'min', 'max', 'nunique'}

        # Reject SQL window function syntax: OVER (...) or named window functions
        _has_over = re.search(r'\bOVER\s*\(', expression, re.IGNORECASE)
        _has_lag_lead = re.search(r'\b(LAG|LEAD)\s*\(', expression, re.IGNORECASE)
        _has_first_last = re.search(r'\b(FIRST_VALUE|LAST_VALUE)\s*\(', expression, re.IGNORECASE)
        if _has_over or _has_lag_lead or _has_first_last:
            if _has_first_last or (_has_over and re.search(r'\b(FIRST_VALUE|LAST_VALUE|FIRST|LAST)\b', expression, re.IGNORECASE)):
                hint = (
                    "Use aggregate() with function='first' or function='last' — these broadcast the "
                    "group's first/last value to every row (equivalent to FIRST_VALUE/LAST_VALUE OVER PARTITION BY). "
                    "column= accepts a list to do multiple columns in one call."
                )
                domain_hint = self._hint_window_example("first_last")
                if domain_hint:
                    hint += f" {domain_hint}"
            elif _has_lag_lead:
                hint = "Use aggregate() with function='lag' or function='lead'."
                domain_hint = self._hint_window_example("lag_lead")
                if domain_hint:
                    hint += f" {domain_hint}"
            else:
                hint = (
                    "Use aggregate() window functions instead: first/last (broadcast group value), "
                    "lag/lead (shift), pct_change, rolling_mean, rank."
                )
                domain_hint = self._hint_window_example("generic")
                if domain_hint:
                    hint += f" {domain_hint}"
            return self.error(
                f"Expression uses window syntax (OVER/PARTITION BY) which is not supported in add_columns. {hint}",
                ErrorType.EXPRESSION_ERROR,
            )

        # Check for pandas method calls (e.g., close.shift(20)) — suggest aggregate tool
        if re.search(r'\.\b(shift|rolling|cumsum|pct_change|rank|ewm|iloc|diff)\b', expression):
            return self.error(
                f"Expression '{expression}' uses a window operation (shift/rolling/pct_change/rank). "
                f"add_columns only supports simple arithmetic (e.g., 'close - open'). "
                f"For shift/rolling/pct_change, use aggregate(function='pct_change', ...) or "
                f"compute_indicator() instead.",
                ErrorType.EXPRESSION_ERROR,
            )

        for word in words:
            if word.lower() not in builtins and word not in df.columns:
                # Suggest compute_indicator if the missing column looks like a technical indicator
                word_lower = word.lower()
                hint = ""
                if any(kw in word_lower for kw in indicator_keywords):
                    hint = (
                        f" HINT: '{word}' looks like a technical indicator. "
                        f"First call create_view() to get base data, then "
                        f"compute_indicator() to add the indicator column, "
                        f"then create_view(source=<that_view>) with add_columns to transform it."
                    )
                elif word_lower in window_keywords:
                    hint = f" HINT: '{word}' is a window/shift operation. Use aggregate(function='{word_lower}', ...) to add the result as a new column."
                    domain_hint = self._hint_window_example("shift_keyword")
                    if domain_hint:
                        hint += f" {domain_hint}"
                elif word_lower in aggregate_keywords:
                    hint = (
                        f" HINT: '{word}' is an aggregate function, not a column. "
                        f"Use aggregate(function='{word_lower}', column='<col>', ...) instead. "
                        f"add_columns only supports arithmetic expressions (e.g., 'col_a - col_b')."
                    )
                return self.error(
                    f"Unknown column '{word}' in expression '{expression}'. "
                    f"Available: {list(df.columns)}.{hint}",
                    ErrorType.EXPRESSION_ERROR,
                )

        try:
            df[col_name] = df.eval(expression)
        except Exception as e:
            return self.error(
                f"Failed to compute '{col_name}' = '{expression}': {e}. "
                f"Available columns: {list(df.columns)}",
                ErrorType.EXPRESSION_ERROR,
            )

        # Normalize result type: bool/int → float64 for consistency
        col_dtype = df[col_name].dtype
        if pd.api.types.is_bool_dtype(col_dtype):
            df[col_name] = df[col_name].astype("float64")
        elif pd.api.types.is_integer_dtype(col_dtype):
            df[col_name] = df[col_name].astype("float64")

        # Check for infinite values (division by zero)
        if pd.api.types.is_float_dtype(df[col_name].dtype):
            inf_count = np.isinf(df[col_name]).sum()
            if inf_count > 0:
                return self.error(
                    f"Expression '{expression}' produced {inf_count} infinite values "
                    "(likely division by zero).",
                    ErrorType.EXPRESSION_ERROR,
                )

        return df

    # SQL reserved words that conflict with column names (e.g., ticker 'ON')
    _SQL_RESERVED = {
        'on', 'in', 'is', 'as', 'by', 'or', 'and', 'not', 'all', 'any',
        'set', 'key', 'to', 'do', 'no', 'if', 'at', 'of', 'for', 'end',
        'select', 'from', 'where', 'order', 'group', 'having', 'join',
        'left', 'right', 'inner', 'outer', 'cross', 'full', 'union',
        'insert', 'update', 'delete', 'create', 'drop', 'alter', 'table',
        'index', 'between', 'like', 'null', 'true', 'false', 'case', 'when',
        'then', 'else', 'with', 'over', 'partition', 'row', 'rows',
    }

    def _quote_reserved_columns(self, expression: str, df: pd.DataFrame) -> str:
        """Quote column names in expression that are SQL reserved words."""
        for col in df.columns:
            if col.lower() in self._SQL_RESERVED:
                # Replace bare column name with quoted version, preserving word boundaries
                expression = re.sub(
                    r'\b' + re.escape(col) + r'\b',
                    f'"{col}"',
                    expression,
                )
        return expression

    def _add_column_via_duckdb(
        self, df: pd.DataFrame, col_name: str, expression: str
    ) -> "pd.DataFrame | ToolResult":
        """Evaluate a SQL expression (e.g. CASE WHEN) via DuckDB and add as a new column."""
        try:
            import duckdb
            # Quote column names that clash with SQL reserved words (e.g., ticker 'ON')
            expression = self._quote_reserved_columns(expression, df)
            conn = duckdb.connect()
            conn.register("_df", df)
            sql = f'SELECT *, ({expression}) AS "{col_name}" FROM _df'
            try:
                result_df = conn.execute(sql).df()
            except duckdb.BinderException as e:
                if "must appear in the GROUP BY clause" in str(e):
                    # Expression contains aggregate functions (CORR, etc.) —
                    # add OVER () to each aggregate call to broadcast results to all rows
                    _agg_funcs = r'\b(CORR|STDDEV|STDDEV_POP|STDDEV_SAMP|VARIANCE|VAR_POP|VAR_SAMP|AVG|SUM|COUNT|MIN|MAX)\s*\('
                    windowed = re.sub(
                        _agg_funcs + r'([^)]*)\)',
                        lambda m: f'{m.group(1)}({m.group(2)}) OVER ()',
                        expression,
                        flags=re.IGNORECASE,
                    )
                    sql = f'SELECT *, ({windowed}) AS "{col_name}" FROM _df'
                    result_df = conn.execute(sql).df()
                else:
                    raise
            conn.close()
            # Normalize new column type: bool/int → float64
            if col_name in result_df.columns:
                col_dtype = result_df[col_name].dtype
                if pd.api.types.is_bool_dtype(col_dtype):
                    result_df[col_name] = result_df[col_name].astype("float64")
                elif pd.api.types.is_integer_dtype(col_dtype):
                    result_df[col_name] = result_df[col_name].astype("float64")
            return result_df
        except Exception as e:
            return self.error(
                f"Failed to compute '{col_name}' = '{expression}': {e}. "
                f"Available columns: {list(df.columns)}",
                ErrorType.EXPRESSION_ERROR,
            )

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata
        rows = len(result.dataframe) if result.dataframe is not None else "?"
        source = m.get("source", "?")
        if m.get("from_view"):
            msg = f"Created '{result.view_name}' ({rows} rows) from view '{source}'"
        else:
            msg = f"Created '{result.view_name}' ({rows} rows) from {source}"
        # Append key parameters
        details = []
        tf = m.get("ticker_filter")
        if tf:
            tickers = tf.get("tickers", [])
            if tf.get("group"):
                details.append(f"group={tf['group']} ({', '.join(tickers)})")
            elif tickers:
                details.append(f"tickers=[{', '.join(tickers)}]")
        if m.get("filters"):
            for k, v in m["filters"].items():
                if k != "ticker":  # ticker already shown above
                    details.append(f"{k}={v}")
        if m.get("where_clause"):
            details.append(f"where {m['where_clause']}")
        if m.get("sort_by"):
            details.append(f"sorted by {m['sort_by']}")
        if m.get("limit"):
            details.append(f"limit {m['limit']}")
        if m.get("selected_columns"):
            details.append(f"columns={m['selected_columns']}")
        if details:
            msg += ", " + ", ".join(details)
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        return (
            "Create a new view from a data source OR existing view. "
            "filters= is for EXACT MATCHES only: {\"ticker\": \"AAPL\"} or {\"ticker\": [\"AAPL\",\"MSFT\"]} for lists. "
            "For date ranges and numeric comparisons ALWAYS use where=: where=\"date >= '2021-01-01'\", where=\"close > 100 AND volume > 1000000\". "
            "Never use filters for dates or numbers — it will match only that exact value (and warn if not found). "
            "Supports: columns (select), sort_by + limit (top/bottom N), "
            "add_columns ({name: expression} for computed columns), finalize=true to mark as final output."
        )
