"""
Stock-specific create view tool with group, ticker, and equation support.

Extends the generic CreateViewTool to add:
- group: Lookup tickers from groups table (e.g., "FANG" -> [META, AMZN, NFLX, GOOGL])
- tickers: Comma-separated ticker list
- tickers_from: Extract tickers from another view
- Named equation resolution for add_columns expressions (e.g., "DAILY_RETURN_PCT")
"""

import json
import re
from typing import TYPE_CHECKING, Any

import pandas as pd

from agentic_TRACE.tools.builtins.create_view import CreateViewTool
from agentic_TRACE.tools.base import ToolParameter
from agentic_TRACE.tools.result import ToolResult, ErrorType

if TYPE_CHECKING:
    from agentic_TRACE.core.session import Session
    from agentic_TRACE.data.sources import DataSource


class StockCreateViewTool(CreateViewTool):
    """
    Create a view with stock-specific ticker filtering.

    Extends CreateViewTool to support:
    - group: Filter by stock group (looks up tickers from groups table)
    - tickers: Filter by comma-separated ticker list
    - tickers_from: Filter by tickers from another view

    These parameters add a ticker IN (...) filter to the query.
    """

    name = "create_view"  # Same name - replaces the generic one

    def __init__(self, data_sources: list["DataSource"] | None = None):
        super().__init__(data_sources)
        self._db_source: "DataSource" | None = None

    def _hint_window_example(self, kind: str) -> str:
        """Stock-specific window function examples."""
        if kind == "first_last":
            return (
                "Example: aggregate(function='first', column=['ma30','vwap'], "
                "partition_by=['ticker'], sort_by='date') adds ma30_first and vwap_first to every row; "
                "then create_view(add_columns={'rel_ma30': '(ma30 - ma30_first) / ma30_first * 100'}) normalizes."
            )
        if kind == "lag_lead":
            return (
                "Example: aggregate(function='lag', column='close', partition_by=['ticker'], "
                "order_by='date', output_column='prev_close') adds the previous row's close to every row."
            )
        if kind == "generic":
            return (
                "Example: aggregate(function='first', column='close', partition_by=['ticker'], "
                "sort_by='date', output_column='first_close')."
            )
        if kind == "shift_keyword":
            return "Example: aggregate(function='lag', column='close', partition_by=['ticker'], order_by='date')."
        return ""

    def set_db_source(self, source: "DataSource") -> None:
        """Set the database source for group lookups."""
        self._db_source = source

    def parameters(self) -> list[ToolParameter]:
        """Extend base parameters with stock-specific ones."""
        base_params = super().parameters()

        # Add stock-specific parameters after 'source'
        stock_params = [
            ToolParameter(
                name="group",
                param_type="string",
                description="Stock group name to filter by (e.g., 'FANG', 'MAG7'). Looks up tickers from groups table.",
                required=False,
            ),
            ToolParameter(
                name="tickers",
                param_type="string",
                description="Comma-separated ticker list (e.g., 'AAPL,NVDA,AMD')",
                required=False,
            ),
            ToolParameter(
                name="tickers_from",
                param_type="string",
                description="View name to extract tickers from (uses the 'ticker' column)",
                required=False,
            ),
        ]

        # Insert after 'source' parameter
        result = []
        for param in base_params:
            result.append(param)
            if param.name == "source":
                result.extend(stock_params)

        return result

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        """Execute with ticker resolution before calling base implementation."""
        group = kwargs.pop("group", None)
        tickers = kwargs.pop("tickers", None)
        tickers_from = kwargs.pop("tickers_from", None)

        ticker_source_info = None

        # Handle group parameter - lookup tickers from groups table
        if group:
            result = self._lookup_group(group)
            if isinstance(result, ToolResult):
                return result  # Error result
            ticker_list = result
            ticker_source_info = {"group": group, "tickers": ticker_list}

        # Handle tickers parameter - parse comma-separated string
        elif tickers:
            ticker_list = [t.strip().upper() for t in tickers.split(",")]
            ticker_source_info = {"tickers": ticker_list}

        # Handle tickers_from parameter - extract from view
        elif tickers_from:
            result = self._extract_tickers_from_view(session, tickers_from)
            if isinstance(result, ToolResult):
                return result  # Error result
            ticker_list = result
            ticker_source_info = {"from_view": tickers_from, "tickers": ticker_list}

        else:
            ticker_list = None

        # Add ticker filter if we resolved any tickers
        if ticker_list:
            # Parse filters if LLM sent as string
            filters = kwargs.get("filters") or {}
            if isinstance(filters, str):
                try:
                    import ast
                    filters = ast.literal_eval(filters)
                except (ValueError, SyntaxError):
                    filters = {}
            # Remove redundant group/tickers keys — already resolved to ticker list
            filters.pop("group", None)
            filters.pop("group_name", None)
            filters.pop("tickers", None)
            filters["ticker"] = ticker_list
            kwargs["filters"] = filters

            # If LLM pointed source at 'groups' table, redirect to stock_prices
            source = kwargs.get("source", "")
            if source.lower() in ("groups", "group"):
                # Find stock prices source
                for s in self.data_sources:
                    if "price" in s.name.lower() or "stock" in s.name.lower():
                        kwargs["source"] = s.name
                        break

        # Call base implementation
        result = super().execute(session, **kwargs)

        # Add ticker source info to metadata
        if ticker_source_info and result.success:
            result.metadata["ticker_filter"] = ticker_source_info

        # Warn if querying stock_prices with no ticker filter.
        # Only applies when the source is an actual data source (not an existing view —
        # views are already filtered, so the warning would be a false positive).
        if result.success and not ticker_list and not result.metadata.get("from_view"):
            source_name = kwargs.get("source", "")
            has_ticker_filter = bool(
                kwargs.get("filters", {}).get("ticker")
                or (kwargs.get("where") and "ticker" in str(kwargs.get("where", "")))
            )
            is_price_source = "price" in source_name.lower() or "stock" in source_name.lower()
            if is_price_source and not has_ticker_filter:
                result.warnings.append(
                    "No ticker or group filter — result contains all tickers. "
                    "If a specific group or ticker was intended, re-run with group='...' or tickers='...'."
                )

        return result

    def _lookup_group(self, group_name: str) -> list[str] | ToolResult:
        """Look up tickers for a group name from the groups table."""
        if not self._db_source:
            return self.error(
                "No database source configured for group lookups",
                ErrorType.CONFIGURATION_ERROR,
            )

        try:
            # Case-insensitive lookup
            sql = f"SELECT DISTINCT ticker FROM groups WHERE UPPER(group_name) = UPPER('{group_name}')"
            df = self._db_source.query(sql)

            if df.empty:
                # Get available groups for hint
                available_sql = "SELECT DISTINCT group_name FROM groups ORDER BY group_name"
                try:
                    available_df = self._db_source.query(available_sql)
                    available = available_df["group_name"].tolist()
                    hint = f"Available groups: {available}"
                except Exception:
                    hint = "Use search(source='groups') to find valid group names"

                return self.error(
                    f"Group '{group_name}' not found. {hint}",
                    ErrorType.DATA_ERROR,
                )

            return df["ticker"].tolist()

        except Exception as e:
            return self.error(
                f"Group lookup failed: {str(e)}",
                ErrorType.DATA_ERROR,
            )

    def _extract_tickers_from_view(
        self, session: "Session", view_name: str
    ) -> list[str] | ToolResult:
        """Extract unique tickers from a view's ticker column."""
        df = session.get_dataframe(view_name)

        if df is None:
            available = list(session.views.keys())
            return self.error(
                f"View '{view_name}' not found. Available: {available}",
                ErrorType.VIEW_ERROR,
            )

        if "ticker" not in df.columns:
            return self.error(
                f"View '{view_name}' has no 'ticker' column. Columns: {list(df.columns)}",
                ErrorType.VALIDATION_ERROR,
            )

        return df["ticker"].unique().tolist()

    def _add_computed_column(
        self, df: pd.DataFrame, col_name: str, expression: str
    ) -> pd.DataFrame | ToolResult:
        """Override to resolve named equations before computing."""
        if self._is_named_equation(expression):
            resolution = self._resolve_equation(expression, list(df.columns))
            if isinstance(resolution, ToolResult):
                return resolution
            expression = resolution["formula"]

        return super()._add_computed_column(df, col_name, expression)

    def _is_named_equation(self, expression: str) -> bool:
        """Check if expression looks like a named equation (ALL_CAPS identifier)."""
        return bool(re.match(r'^[A-Z][A-Z0-9_]+$', expression))

    def _resolve_equation(
        self, name: str, available_columns: list[str]
    ) -> dict | ToolResult:
        """Resolve a named equation from the equations table."""
        if not self._db_source:
            return self.error(
                "No database source configured for equation lookups",
                ErrorType.CONFIGURATION_ERROR,
            )

        try:
            sql = f"SELECT type, description, rule FROM equations WHERE name = '{name}'"
            eq_df = self._db_source.query(sql)

            if eq_df.empty:
                try:
                    available_sql = "SELECT name, type FROM equations WHERE type = 'column' ORDER BY name"
                    available_df = self._db_source.query(available_sql)
                    column_eqs = available_df["name"].tolist()
                    hint = f"Column equations: {column_eqs[:10]}{'...' if len(column_eqs) > 10 else ''}"
                except Exception:
                    hint = "Use explore(source='equations') to find available equations"

                return self.error(
                    f"Equation '{name}' not found. {hint}",
                    ErrorType.DATA_ERROR,
                )

            row = eq_df.iloc[0]
            eq_type = row["type"]

            if eq_type != "column":
                tool_hint = {
                    "window": "Use aggregate tool with function parameter for window equations.",
                    "aggregate": "Use aggregate tool for aggregate equations.",
                }.get(eq_type, f"Unknown equation type: {eq_type}")

                return self.error(
                    f"'{name}' is type '{eq_type}', cannot use with add_columns. {tool_hint}",
                    ErrorType.VALIDATION_ERROR,
                )

            rule = row["rule"]
            if isinstance(rule, str):
                rule = json.loads(rule)

            formula = rule.get("expression", "")
            required_columns = rule.get("required_columns", [])

            if available_columns:
                missing = [c for c in required_columns if c not in available_columns]
                if missing:
                    return self.error(
                        f"'{name}' requires columns {required_columns}, missing: {missing}. "
                        f"Available: {available_columns}",
                        ErrorType.VALIDATION_ERROR,
                    )

            return {"formula": formula, "description": row.get("description", "")}

        except Exception as e:
            return self.error(
                f"Equation lookup failed: {str(e)}",
                ErrorType.DATA_ERROR,
            )

    def tool_description(self) -> str:
        return (
            "Create a view from a data source or existing view. "
            "Filter by group (e.g., 'FANG', 'MAG7'), tickers (comma-separated), "
            "or tickers_from (another view). "
            "Use add_columns with named equations (e.g., 'DAILY_RETURN_PCT') or "
            "arithmetic expressions. Use finalize=true on your last tool call."
        )
