"""
Data source abstractions.

DataSource provides a uniform interface for accessing tabular data from
various backends (DuckDB, PostgreSQL, Spark, APIs, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ColumnSchema:
    """Schema information for a single column."""
    name: str
    dtype: str
    description: str = ""
    nullable: bool = True


@dataclass
class TableSchema:
    """Schema information for a table/source."""
    columns: list[ColumnSchema] = field(default_factory=list)

    def to_description(self) -> str:
        """Generate human-readable schema description."""
        lines = []
        for col in self.columns:
            nullable = "nullable" if col.nullable else "not null"
            desc = f" - {col.description}" if col.description else ""
            lines.append(f"  - {col.name} ({col.dtype}, {nullable}){desc}")
        return "\n".join(lines)


class DataSource(ABC):
    """
    Abstract base class for data sources.

    Subclasses implement access to specific backends (DuckDB, Postgres, etc.)
    while providing a uniform interface for querying and schema inspection.

    Attributes:
        name: Logical name for this source (e.g., "stock_prices")
        description: Human-readable description for LLM prompts
        schema: Column schema information

    Example:
        >>> source = DuckDBSource(
        ...     name="stock_prices",
        ...     db_path="data/market.duckdb",
        ...     table="stock_prices",
        ...     description="Daily OHLCV data for US equities"
        ... )
        >>> df = source.query("SELECT * FROM stock_prices WHERE ticker = 'AAPL'")
    """

    name: str
    description: str
    schema: TableSchema | None

    def __init__(
        self,
        name: str,
        description: str = "",
        schema: TableSchema | None = None,
    ):
        self.name = name
        self.description = description
        self.schema = schema

    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.

        Args:
            sql: SQL query string

        Returns:
            Query results as DataFrame
        """
        pass

    @abstractmethod
    def get_table_names(self) -> list[str]:
        """
        Get available table names.

        Returns:
            List of table names
        """
        pass

    def get_schema_description(self) -> str:
        """
        Generate human-readable schema description for system prompt.

        Override for custom formatting.

        Returns:
            Formatted schema description
        """
        lines = [f"**{self.name}**"]

        if self.description:
            lines.append(f"  {self.description}")

        if self.schema:
            lines.append("  Columns:")
            lines.append(self.schema.to_description())

        return "\n".join(lines)

    def sample(self, limit: int = 5) -> pd.DataFrame:
        """
        Get sample rows from the source.

        Default implementation uses LIMIT clause; override if needed.

        Args:
            limit: Number of rows to sample

        Returns:
            Sample DataFrame
        """
        return self.query(f"SELECT * FROM {self.name} LIMIT {limit}")


class SourceRegistry:
    """
    Registry for data sources.

    Manages available data sources and provides access for tools.
    Sources are registered by name and can be retrieved for querying.

    Example:
        >>> registry = SourceRegistry()
        >>> registry.register(DuckDBSource(name="prices", ...))
        >>> registry.register(PostgresSource(name="companies", ...))
        >>>
        >>> # Get source by name
        >>> prices = registry.get("prices")
        >>> df = prices.query("SELECT * FROM prices WHERE date > '2024-01-01'")
        >>>
        >>> # Generate descriptions for system prompt
        >>> prompt_text = registry.get_all_descriptions()
    """

    def __init__(self):
        self._sources: dict[str, DataSource] = {}

    def register(self, source: DataSource) -> DataSource:
        """
        Register a data source.

        Args:
            source: DataSource instance to register

        Returns:
            The registered source

        Raises:
            ValueError: If source with same name already registered
        """
        if source.name in self._sources:
            raise ValueError(f"Source '{source.name}' already registered")
        self._sources[source.name] = source
        return source

    def register_or_replace(self, source: DataSource) -> DataSource:
        """
        Register a source, replacing existing if present.

        Args:
            source: DataSource instance

        Returns:
            The registered source
        """
        self._sources[source.name] = source
        return source

    def get(self, name: str) -> DataSource | None:
        """
        Get a source by name.

        Args:
            name: Source name

        Returns:
            DataSource or None if not found
        """
        return self._sources.get(name)

    def get_required(self, name: str) -> DataSource:
        """
        Get a source by name, raising if not found.

        Args:
            name: Source name

        Returns:
            DataSource

        Raises:
            KeyError: If source not found
        """
        source = self._sources.get(name)
        if source is None:
            available = list(self._sources.keys())
            raise KeyError(f"Source '{name}' not found. Available: {available}")
        return source

    def list_sources(self) -> list[DataSource]:
        """Get all registered sources."""
        return list(self._sources.values())

    def list_names(self) -> list[str]:
        """Get names of all registered sources."""
        return list(self._sources.keys())

    def get_all_descriptions(self) -> str:
        """
        Generate descriptions of all sources for system prompt.

        Returns:
            Formatted string describing all data sources
        """
        if not self._sources:
            return "No data sources available."

        lines = ["## Available Data Sources\n"]
        for source in self._sources.values():
            lines.append(source.get_schema_description())
            lines.append("")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, name: str) -> bool:
        return name in self._sources

    def __iter__(self):
        return iter(self._sources.values())


# Global source registry
_global_source_registry: SourceRegistry | None = None


def get_global_source_registry() -> SourceRegistry:
    """Get the global source registry, creating if needed."""
    global _global_source_registry
    if _global_source_registry is None:
        _global_source_registry = SourceRegistry()
    return _global_source_registry


def set_global_source_registry(registry: SourceRegistry) -> None:
    """Set the global source registry."""
    global _global_source_registry
    _global_source_registry = registry
