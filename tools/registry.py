"""
Tool registry with decorator-based registration.

Tools can be registered via decorator or explicit registration.
The registry handles tool discovery, dispatch, and schema generation.
"""

from typing import Any, Callable, Type

from .base import BaseTool, ToolCategory, ToolDefinition


class ToolRegistry:
    """
    Registry for tool instances.

    Handles:
        - Tool registration (decorator or explicit)
        - Tool lookup by name
        - Category-based filtering
        - OpenAI function schema generation

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(SearchTool())
        >>> registry.register(CreateViewTool(db_connection=db))
        >>>
        >>> # Get tool by name
        >>> tool = registry.get("search")
        >>>
        >>> # Get all tools as OpenAI functions
        >>> functions = registry.to_openai_functions()
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._metadata_extractor: Callable | None = None

    def register(self, tool: BaseTool) -> BaseTool:
        """
        Register a tool instance.

        Args:
            tool: Tool instance to register

        Returns:
            The registered tool (for chaining)

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' already registered. "
                f"Use replace=True to override."
            )
        self._tools[tool.name] = tool
        return tool

    def register_or_replace(self, tool: BaseTool) -> BaseTool:
        """
        Register a tool, replacing existing if present.

        Args:
            tool: Tool instance to register

        Returns:
            The registered tool
        """
        self._tools[tool.name] = tool
        return tool

    def unregister(self, tool_name: str) -> BaseTool | None:
        """
        Remove a tool from the registry.

        Args:
            tool_name: Name of tool to remove

        Returns:
            The removed tool, or None if not found
        """
        return self._tools.pop(tool_name, None)

    def get(self, tool_name: str) -> BaseTool | None:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)

    def get_required(self, tool_name: str) -> BaseTool:
        """
        Get a tool by name, raising if not found.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance

        Raises:
            KeyError: If tool not found
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            available = list(self._tools.keys())
            raise KeyError(f"Tool '{tool_name}' not found. Available: {available}")
        return tool

    def list_tools(self, category: ToolCategory | None = None) -> list[BaseTool]:
        """
        List all registered tools, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of matching tools
        """
        tools = list(self._tools.values())
        if category is not None:
            tools = [t for t in tools if t.category == category]
        return tools

    def list_names(self, category: ToolCategory | None = None) -> list[str]:
        """
        List names of registered tools.

        Args:
            category: Optional category filter

        Returns:
            List of tool names
        """
        return [t.name for t in self.list_tools(category)]

    def get_definitions(
        self,
        category: ToolCategory | None = None,
    ) -> list[ToolDefinition]:
        """
        Get tool definitions, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of ToolDefinition objects
        """
        return [t.definition() for t in self.list_tools(category)]

    def to_openai_functions(
        self,
        category: ToolCategory | None = None,
    ) -> list[dict[str, Any]]:
        """
        Convert registered tools to OpenAI function calling format.

        Args:
            category: Optional category filter

        Returns:
            List of OpenAI function schemas
        """
        return [d.to_openai_function() for d in self.get_definitions(category)]

    def get_tools_description(self) -> str:
        """
        Generate human-readable description of all tools for system prompt.

        Returns:
            Formatted string describing all tools
        """
        lines = []

        for category in ToolCategory:
            tools = self.list_tools(category)
            if not tools:
                continue

            lines.append(f"\n### {category.value.title()} Tools\n")

            for tool in tools:
                defn = tool.definition()
                lines.append(f"**{defn.name}**: {defn.description}")

                # List parameters
                if defn.parameters:
                    lines.append("  Parameters:")
                    for param in defn.parameters:
                        req = "(required)" if param.required else "(optional)"
                        lines.append(f"    - {param.name} ({param.param_type}) {req}: {param.description}")

                lines.append("")

        return "\n".join(lines)

    def set_metadata_extractor(
        self,
        extractor: Callable[[Any, dict], dict],
    ) -> None:
        """
        Set custom metadata extractor for views.

        The extractor receives (dataframe, base_metadata) and returns
        extended metadata dict.

        Args:
            extractor: Callable that extends base metadata
        """
        self._metadata_extractor = extractor

    def get_metadata_extractor(self) -> Callable | None:
        """Get the custom metadata extractor if set."""
        return self._metadata_extractor

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def __iter__(self):
        return iter(self._tools.values())


# Global registry for decorator-based registration
_global_registry: ToolRegistry | None = None


def get_global_registry() -> ToolRegistry:
    """
    Get the global tool registry.

    Creates one if it doesn't exist.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def set_global_registry(registry: ToolRegistry) -> None:
    """
    Set the global tool registry.

    Useful for testing or custom configurations.
    """
    global _global_registry
    _global_registry = registry


def register_tool(
    cls: Type[BaseTool] | None = None,
    *,
    registry: ToolRegistry | None = None,
) -> Type[BaseTool] | Callable[[Type[BaseTool]], Type[BaseTool]]:
    """
    Decorator to register a tool class.

    Can be used with or without parentheses:

        @register_tool
        class MyTool(BaseTool):
            ...

        @register_tool(registry=custom_registry)
        class MyTool(BaseTool):
            ...

    Note: This registers an instance created with no arguments.
    For tools that require configuration, use explicit registration:

        registry.register(MyTool(config=value))

    Args:
        cls: The tool class (when used without parentheses)
        registry: Optional registry to use (default: global)

    Returns:
        Decorated class or decorator function
    """
    target_registry = registry or get_global_registry()

    def decorator(tool_cls: Type[BaseTool]) -> Type[BaseTool]:
        # Instantiate and register
        instance = tool_cls()
        target_registry.register(instance)
        return tool_cls

    if cls is not None:
        # Called without parentheses: @register_tool
        return decorator(cls)

    # Called with parentheses: @register_tool() or @register_tool(registry=...)
    return decorator
