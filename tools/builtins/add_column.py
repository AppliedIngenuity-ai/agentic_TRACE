"""
Add column tool - adds computed columns to views.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session


class AddColumnTool(BaseTool):
    """
    Add a computed column to a view.

    Creates a new view with an additional column computed from an expression.
    Supports arithmetic operations on existing columns.
    """

    name = "add_column"
    category = ToolCategory.TRANSFORM

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="source_view",
                param_type="string",
                description="Name of the view to add column to",
                required=True,
            ),
            ToolParameter(
                name="output_view",
                param_type="string",
                description="Name for the output view",
                required=True,
            ),
            ToolParameter(
                name="column_name",
                param_type="string",
                description="Name for the new column",
                required=True,
            ),
            ToolParameter(
                name="expression",
                param_type="string",
                description="Expression to compute (e.g., 'close - open', 'price * quantity')",
                required=True,
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "source_view", "output_view", "column_name", "expression"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        source_view = kwargs["source_view"]
        output_view = kwargs["output_view"]
        column_name = kwargs["column_name"]
        expression = kwargs["expression"]

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

        # Validate expression
        validation_error = self._validate_expression(expression, df)
        if validation_error:
            return self.error(validation_error, ErrorType.EXPRESSION_ERROR)

        # Compute column
        try:
            df[column_name] = df.eval(expression)
        except Exception as e:
            return self.error(
                f"Failed to compute expression: {str(e)}",
                ErrorType.EXPRESSION_ERROR,
            )

        # Check for infinite values which indicate division by zero
        if df[column_name].dtype in [np.float64, np.float32, float]:
            inf_count = np.isinf(df[column_name]).sum()
            if inf_count > 0:
                return self.error(
                    f"Expression '{expression}' produced {inf_count} infinite values "
                    f"(likely division by zero). Check your expression for zero denominators.",
                    ErrorType.EXPRESSION_ERROR,
                )

        # Generate hint about new column characteristics
        hint = None
        if df[column_name].dtype in [np.float64, np.float32, float]:
            col_max = df[column_name].abs().max()
            if col_max > 1 and ("pct" in column_name.lower() or "percent" in column_name.lower()
                                or "return" in column_name.lower() or "pct" in expression.lower()
                                or "* 100" in expression):
                hint = (
                    f"Column '{column_name}' contains percentage values (e.g., 2.5 means 2.5%). "
                    f"To convert to decimal ratio: {column_name} / 100"
                )

        return self.success(
            view_name=output_view,
            df=df,
            column_added=column_name,
            expression=expression,
            hint=hint,
        )

    def _validate_expression(self, expression: str, df: pd.DataFrame) -> str | None:
        """
        Validate expression for safety and correctness.

        Returns error message if invalid, None if valid.
        """
        # Basic safety checks - whitelist allowed characters
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-*/()%. ")
        if not all(c in allowed_chars for c in expression):
            disallowed = [c for c in expression if c not in allowed_chars]
            return f"Expression contains disallowed characters: {disallowed}"

        # Check that referenced columns exist
        # Simple heuristic: words that aren't operators/numbers are column names
        import re
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
        operators = {'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None', 'abs', 'round', 'min', 'max', 'sqrt'}

        for word in words:
            if word not in operators and word not in df.columns:
                return f"Unknown column: '{word}'. Available: {list(df.columns)}"

        return None

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata
        rows = len(result.dataframe) if result.dataframe is not None else "?"
        col = m.get("column_added", "?")
        expr = m.get("expression", "?")
        msg = f"Added column '{col}' = {expr} \u2192 '{result.view_name}' ({rows} rows)"
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        if m.get("hint"):
            parts.append(f"NOTE: {m['hint']}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        return (
            "Add a computed column to a view. "
            "Expression supports arithmetic: 'close - open', 'price * 1.1', "
            "'(high + low) / 2'. References existing column names."
        )
