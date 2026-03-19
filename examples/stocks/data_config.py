"""
Data source configuration for stocks example.

Registers the DuckDB data sources for stock market analysis.
"""

from pathlib import Path

from agentic_TRACE.data.sources import SourceRegistry
from agentic_TRACE.data.adapters.duckdb import DuckDBSource, DuckDBConnectionPool


def create_source_registry(db_path: str | Path) -> SourceRegistry:
    """
    Create a source registry with stock market data sources.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        Configured SourceRegistry
    """
    registry = SourceRegistry()

    # Use connection pool for shared connection
    pool = DuckDBConnectionPool(db_path)

    # Stock prices table
    registry.register(DuckDBSource(
        name="stock_prices",
        db_path=db_path,
        table="stock_prices",
        description=(
            "Daily OHLCV data for US equities. "
            "Columns: ticker, date, open, high, low, close, volume"
        ),
    ))

    # Companies table
    registry.register(DuckDBSource(
        name="companies",
        db_path=db_path,
        table="companies",
        description=(
            "Company metadata. "
            "Columns: ticker, name, sector, industry"
        ),
    ))

    # Groups table (for named groups like FANG, MAG7)
    registry.register(DuckDBSource(
        name="groups",
        db_path=db_path,
        table="groups",
        description=(
            "Named stock groups. "
            "Columns: group_name, ticker. "
            "Example groups: FANG, MAG7, semiconductors"
        ),
    ))

    # Equations table (pre-defined named computations)
    registry.register(DuckDBSource(
        name="equations",
        db_path=db_path,
        table="equations",
        description=(
            "37 pre-defined computations and technical indicators. "
            "Popular: RSI, MACD, BOLLINGER_PCT_B, ATR, VOLUME_RATIO, EMA, MOVING_AVERAGE, DAILY_RETURN_PCT. "
            "Search by keywords (e.g., 'momentum', 'volatility') to discover more."
        ),
    ))

    return registry


# Default registry
_default_registry: SourceRegistry | None = None
_default_db_path: Path | None = None


def get_default_registry(db_path: str | Path | None = None) -> SourceRegistry:
    """
    Get the default source registry.

    Args:
        db_path: Path to database. If None, uses default location.

    Returns:
        Configured SourceRegistry
    """
    global _default_registry, _default_db_path

    if db_path is None:
        # Default to data/market.duckdb relative to examples/stocks
        db_path = Path(__file__).parent / "data" / "market.duckdb"

        # If not found there, try the original location
        if not db_path.exists():
            original_path = Path(__file__).parent.parent.parent / "data" / "market.duckdb"
            if original_path.exists():
                db_path = original_path

    db_path = Path(db_path)

    # Return cached if same path
    if _default_registry is not None and _default_db_path == db_path:
        return _default_registry

    _default_registry = create_source_registry(db_path)
    _default_db_path = db_path

    return _default_registry
