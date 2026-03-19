"""
Base tool class and related types.

All tools inherit from BaseTool and implement execute().
Tools can override tool_description(), summarize(), and parameters()
to customize behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from .result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ..core.session import Session


class ToolCategory(Enum):
    """
    Tool categories for organization and prompt generation.

    EXPLORE: Tools that help understand data (search, describe, resolve temporal expressions)
    TRANSFORM: Tools that create/modify views (create_view, filter, aggregate, join, etc.)
    OUTPUT: Tools that produce artifacts without returning views (chart, export)
    """
    EXPLORE = "explore"
    TRANSFORM = "transform"
    OUTPUT = "output"


@dataclass
class ToolParameter:
    """
    Definition of a single tool parameter.

    Used to generate OpenAI function schemas and validate inputs.
    """
    name: str
    param_type: Literal["string", "integer", "number", "boolean", "array", "object"]
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None
    items_type: str | None = None  # For array types

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function parameter schema."""
        schema: dict[str, Any] = {
            "type": self.param_type,
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum

        if self.param_type == "array" and self.items_type:
            schema["items"] = {"type": self.items_type}

        return schema


@dataclass
class ToolDefinition:
    """
    Complete tool definition for registration and schema generation.
    """
    name: str
    description: str
    parameters: list[ToolParameter]
    category: ToolCategory = ToolCategory.TRANSFORM

    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_openai_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Subclasses must implement:
        - execute(session, **kwargs) -> ToolResult
        - parameters() -> list[ToolParameter]

    Subclasses may override:
        - tool_description() -> str  (default: uses class docstring)
        - summarize(result, df) -> str  (default: generic summary)

    Class attributes:
        - name: str - Tool name for registration (required)
        - category: ToolCategory - Tool category (default: TRANSFORM)

    Example:
        >>> class MyTool(BaseTool):
        ...     name = "my_tool"
        ...     category = ToolCategory.TRANSFORM
        ...
        ...     def parameters(self) -> list[ToolParameter]:
        ...         return [ToolParameter("input", "string", "Input value")]
        ...
        ...     def execute(self, session, **kwargs) -> ToolResult:
        ...         # Do work...
        ...         return self.success(view_name="result", df=result_df)
    """

    # Class attributes - must be set by subclass
    name: str
    category: ToolCategory = ToolCategory.TRANSFORM

    def __init__(self):
        """Initialize tool. Subclasses can accept configuration here."""
        if not hasattr(self, 'name') or not self.name:
            raise ValueError(f"{self.__class__.__name__} must define 'name' attribute")

    def definition(self) -> ToolDefinition:
        """
        Get complete tool definition.

        Override this for full control over the definition.
        Default implementation uses tool_description() and parameters().
        """
        return ToolDefinition(
            name=self.name,
            description=self.tool_description(),
            parameters=self.parameters(),
            category=self.category,
        )

    def tool_description(self) -> str:
        """
        Get tool description for LLM prompt.

        Override to customize. Default uses class docstring.
        """
        return self.__doc__ or f"Execute the {self.name} tool"

    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """
        Define accepted parameters.

        Returns:
            List of ToolParameter definitions
        """
        pass

    @abstractmethod
    def execute(self, session: "Session", **kwargs) -> ToolResult:
        """
        Execute the tool.

        Args:
            session: Current session state
            **kwargs: Tool parameters

        Returns:
            ToolResult indicating success/failure
        """
        pass

    def summarize(self, result: ToolResult) -> str:
        """
        Generate human-readable summary of the result.

        Override for custom summarization logic.
        Default provides generic summary based on result contents.

        Args:
            result: The ToolResult from execute()

        Returns:
            Human-readable summary string
        """
        if not result.success:
            return f"Error ({result.error_type}): {result.error}"

        parts = []

        if result.view_name:
            row_count = len(result.dataframe) if result.dataframe is not None else "?"
            parts.append(f"Created view '{result.view_name}' with {row_count} rows")

        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")

        if result.metadata:
            # Include key metadata items (skip bulky/internal fields and None values)
            _skip = {"columns", "column_stats", "hint", "column_origins",
                     "chart_base64", "results", "equation", "warnings"}
            meta_items = [f"{k}={v}" for k, v in result.metadata.items()
                         if k not in _skip and v is not None]
            if meta_items:
                parts.append(f"Metadata: {', '.join(meta_items)}")

            # Show hint prominently at the end
            if "hint" in result.metadata and result.metadata["hint"]:
                parts.append(f"NOTE: {result.metadata['hint']}")

        return " | ".join(parts) if parts else "Tool executed successfully"

    # ─── Helper methods for subclasses ───

    def success(
        self,
        view_name: str | None = None,
        df: pd.DataFrame | None = None,
        warnings: list[str] | None = None,
        **extra_metadata: Any,
    ) -> ToolResult:
        """
        Create a successful result.

        Args:
            view_name: Name of the created view
            df: The resulting DataFrame
            warnings: Optional list of warning messages
            **extra_metadata: Additional metadata to include

        Returns:
            ToolResult with success=True
        """
        return ToolResult(
            success=True,
            view_name=view_name,
            dataframe=df,
            metadata=extra_metadata,
            warnings=warnings or [],
        )

    def error(
        self,
        message: str,
        error_type: str = ErrorType.INTERNAL_ERROR,
    ) -> ToolResult:
        """
        Create an error result.

        Args:
            message: Error message
            error_type: Error category (use ErrorType constants)

        Returns:
            ToolResult with success=False
        """
        return ToolResult(
            success=False,
            error=message,
            error_type=error_type,
        )

    # ─── Validation helpers ───

    def validate_required(
        self,
        kwargs: dict[str, Any],
        *required_params: str,
    ) -> str | None:
        """
        Check that required parameters are present.

        Args:
            kwargs: The kwargs passed to execute()
            *required_params: Names of required parameters

        Returns:
            Error message if validation fails, None if valid
        """
        missing = [p for p in required_params if p not in kwargs or kwargs[p] is None]
        if missing:
            return f"Missing required parameters: {', '.join(missing)}"
        return None

    def validate_view_exists(
        self,
        session: "Session",
        view_name: str,
    ) -> str | None:
        """
        Check that a view exists in the session.

        If not found in session views, auto-loads from the global source registry
        (so tools like aggregate/join can reference data sources directly by name).

        Args:
            session: Current session
            view_name: Name of the view to check

        Returns:
            Error message if view doesn't exist, None if valid
        """
        if view_name not in session.views:
            # Try auto-loading from the global source registry
            try:
                from ..data.sources import get_global_source_registry
                source = get_global_source_registry().get(view_name)
                if source is not None:
                    df = source.query(f"SELECT * FROM {view_name}")
                    session.add_view(
                        view_name, df,
                        source=view_name,
                        operation="auto_load",
                    )
                    return None
            except Exception:
                pass
            available = list(session.views.keys())
            return f"View '{view_name}' not found. Available views: {available}"
        return None

    def validate_columns_exist(
        self,
        df: pd.DataFrame,
        *columns: str,
    ) -> str | None:
        """
        Check that columns exist in a DataFrame.

        Args:
            df: DataFrame to check
            *columns: Column names to verify

        Returns:
            Error message if any column missing, None if valid
        """
        missing = [c for c in columns if c not in df.columns]
        if missing:
            available = list(df.columns)
            return f"Columns not found: {missing}. Available: {available}"
        return None
