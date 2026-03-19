"""
Tool configuration for stocks example.

Registers 8 tools for the stock market analysis agent:
- explore: Search sources, describe views, list all, ask user
- create_view: Create/filter/sort views with computed columns (StockCreateViewTool adds group/tickers)
- aggregate: Group aggregation and window functions (with named equation support)
- compute_indicator: Technical indicators (SMA, EMA, RSI, etc.)
- join: Join or stack views
- pivot: Pivot tables
- chart: Visualizations
- resolve_time: Date range resolution (domain-specific)
"""

from agentic_TRACE.tools.registry import ToolRegistry
from agentic_TRACE.tools.builtins import (
    ExploreTool,
    JoinTool,
    PivotTool,
    ChartTool,
)

# Stock-specific tools (subclass the generic ones, or domain-specific)
from agentic_TRACE.examples.stocks.tools import (
    ResolveTimeTool,
    StockCreateViewTool,
    StockAggregateTool,
    ComputeIndicatorTool,
)


def create_stock_registry(data_sources: list = None) -> ToolRegistry:
    """
    Create a tool registry configured for stock analysis.

    Args:
        data_sources: List of DataSource instances for tools that need data access

    Returns:
        Configured ToolRegistry with 8 tools
    """
    registry = ToolRegistry()

    # Find primary DB source for lookups (groups, equations tables)
    db_source = None
    stock_source = None
    if data_sources:
        for source in data_sources:
            if db_source is None:
                db_source = source
            if "price" in source.name.lower() or "stock" in source.name.lower():
                stock_source = source

    # 1. Explore tool (search + describe + ask_human)
    explore_tool = ExploreTool()
    if data_sources:
        explore_tool.set_data_sources(data_sources)
    registry.register(explore_tool)

    # 2. Stock-specific create_view (with group/tickers + equation support for add_columns)
    create_view_tool = StockCreateViewTool()
    if data_sources:
        create_view_tool.set_data_sources(data_sources)
    if db_source:
        create_view_tool.set_db_source(db_source)
    registry.register(create_view_tool)

    # 3. Stock aggregate tool (group aggregation + window functions + equation lookup + stock hints)
    aggregate_tool = StockAggregateTool()
    if db_source:
        aggregate_tool.set_db_source(db_source)
    registry.register(aggregate_tool)

    # 4. Compute indicator (technical indicators)
    registry.register(ComputeIndicatorTool())

    # 5. Join tool (join + stack)
    join_tool = JoinTool()
    if data_sources:
        join_tool.set_data_sources(data_sources)
    registry.register(join_tool)

    # 6. Pivot tool
    registry.register(PivotTool())

    # 7. Chart tool
    registry.register(ChartTool())

    # 8. Resolve time (domain-specific date range resolution)
    resolve_time = ResolveTimeTool()
    if stock_source:
        resolve_time.set_data_source(stock_source)
    elif db_source:
        resolve_time.set_data_source(db_source)
    registry.register(resolve_time)

    return registry


# Default registry for simple usage
_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """Get the default stock tool registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = create_stock_registry()
    return _default_registry
