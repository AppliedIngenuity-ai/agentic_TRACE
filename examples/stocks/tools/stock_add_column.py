"""
Stock-specific add column tool with equation support.

Extends the generic AddColumnTool to support named equations from the equations table.
Named equations are ALL_CAPS identifiers like "DAILY_RETURN_PCT" that resolve to formulas.
"""

import json
import re
from typing import TYPE_CHECKING

import pandas as pd

from agentic_TRACE.tools.builtins.add_column import AddColumnTool
from agentic_TRACE.tools.base import ToolParameter
from agentic_TRACE.tools.result import ToolResult, ErrorType

if TYPE_CHECKING:
    from agentic_TRACE.core.session import Session
    from agentic_TRACE.data.sources import DataSource


class StockAddColumnTool(AddColumnTool):
    """
    Add a computed column with support for named equations.

    Extends AddColumnTool to support:
    - equation: Named equation from equations table (e.g., "DAILY_RETURN_PCT")

    Named equations are resolved to their formulas before computing.
    Only "column" type equations are supported (not "window" or "aggregate").
    """

    name = "add_column"  # Same name - replaces the generic one

    def __init__(self):
        super().__init__()
        self._db_source: "DataSource" | None = None

    def set_db_source(self, source: "DataSource") -> None:
        """Set the database source for equation lookups."""
        self._db_source = source

    def parameters(self) -> list[ToolParameter]:
        """Extend base parameters with equation option."""
        base_params = super().parameters()

        # Modify expression description and add equation parameter
        result = []
        for param in base_params:
            if param.name == "expression":
                # Update description to mention equations
                result.append(ToolParameter(
                    name="expression",
                    param_type="string",
                    description=(
                        "Expression to compute. Can be arithmetic (e.g., 'close - open', 'price * 1.1') "
                        "or a named equation (e.g., 'DAILY_RETURN_PCT'). "
                        "Named equations are looked up from the equations table."
                    ),
                    required=True,
                ))
            else:
                result.append(param)

        return result

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        """Execute with equation resolution before calling base implementation."""
        expression = kwargs.get("expression", "")
        source_view = kwargs.get("source_view")

        # Check if expression looks like a named equation (ALL_CAPS with underscores)
        if self._is_named_equation(expression):
            # Get available columns from source view
            if source_view and source_view in session.views:
                df = session.get_dataframe_required(source_view)
                available_columns = list(df.columns)
            else:
                available_columns = []

            # Resolve the equation
            resolution = self._resolve_equation(expression, available_columns)
            if isinstance(resolution, ToolResult):
                return resolution  # Error result

            # Replace expression with resolved formula
            kwargs["expression"] = resolution["formula"]

            # Call base implementation
            result = super().execute(session, **kwargs)

            # Add equation resolution info to metadata
            if result.success:
                result.metadata["equation"] = {
                    "name": expression,
                    "formula": resolution["formula"],
                    "description": resolution.get("description"),
                }

            return result

        # Not a named equation - call base implementation directly
        return super().execute(session, **kwargs)

    def _is_named_equation(self, expression: str) -> bool:
        """Check if expression looks like a named equation (ALL_CAPS identifier)."""
        return bool(re.match(r'^[A-Z][A-Z0-9_]+$', expression))

    def _resolve_equation(
        self, name: str, available_columns: list[str]
    ) -> dict | ToolResult:
        """
        Resolve a named equation from the equations table.

        Returns dict with formula/description or ToolResult error.
        """
        if not self._db_source:
            return self.error(
                "No database source configured for equation lookups",
                ErrorType.CONFIGURATION_ERROR,
            )

        try:
            sql = f"SELECT type, description, rule FROM equations WHERE name = '{name}'"
            df = self._db_source.query(sql)

            if df.empty:
                # Get available equations for hint
                try:
                    available_sql = "SELECT name, type FROM equations ORDER BY name"
                    available_df = self._db_source.query(available_sql)
                    column_eqs = available_df[available_df["type"] == "column"]["name"].tolist()
                    hint = f"Column equations: {column_eqs[:10]}{'...' if len(column_eqs) > 10 else ''}"
                except Exception:
                    hint = "Use search(source='equations') to find available equations"

                return self.error(
                    f"Equation '{name}' not found. {hint}",
                    ErrorType.DATA_ERROR,
                )

            row = df.iloc[0]
            eq_type = row["type"]
            description = row.get("description", "")

            # Parse rule JSON
            rule = row["rule"]
            if isinstance(rule, str):
                rule = json.loads(rule)

            # Check type - only 'column' type can be used in add_column
            if eq_type != "column":
                tool_hint = {
                    "window": "Use apply_window tool for window equations (rolling, lag, rank).",
                    "aggregate": "Use aggregate tool for aggregate equations.",
                }.get(eq_type, f"Unknown equation type: {eq_type}")

                return self.error(
                    f"'{name}' is type '{eq_type}', cannot use with add_column. {tool_hint}",
                    ErrorType.VALIDATION_ERROR,
                )

            # Get the expression formula
            formula = rule.get("expression", "")
            required_columns = rule.get("required_columns", [])

            # Check required columns exist
            if available_columns:
                missing = [c for c in required_columns if c not in available_columns]
                if missing:
                    return self.error(
                        f"'{name}' requires columns {required_columns}, missing: {missing}. "
                        f"Available: {available_columns}",
                        ErrorType.VALIDATION_ERROR,
                    )

            return {
                "formula": formula,
                "description": description,
                "required_columns": required_columns,
            }

        except Exception as e:
            return self.error(
                f"Equation lookup failed: {str(e)}",
                ErrorType.DATA_ERROR,
            )

    def tool_description(self) -> str:
        return (
            "Add a computed column to a view. "
            "Expression can be arithmetic ('close - open', '(high + low) / 2') "
            "or a named equation ('DAILY_RETURN_PCT', 'SPREAD'). "
            "Named equations are resolved from the equations table."
        )
