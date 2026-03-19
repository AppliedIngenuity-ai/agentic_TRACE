"""
DuckDB data source adapter.
"""

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore

from ..sources import DataSource, TableSchema, ColumnSchema


class DuckDBSource(DataSource):
    """
    Data source backed by DuckDB.

    Provides access to tables in a DuckDB database file or in-memory database.

    Example:
        >>> source = DuckDBSource(
        ...     name="stock_prices",
        ...     db_path="data/market.duckdb",
        ...     table="stock_prices",
        ...     description="Daily OHLCV data for US equities"
        ... )
        >>> df = source.query("SELECT * FROM stock_prices WHERE ticker = 'AAPL'")

    Attributes:
        name: Logical name for this source
        db_path: Path to DuckDB database file (or ":memory:")
        table: Primary table name for this source
        description: Human-readable description
    """

    def __init__(
        self,
        name: str,
        db_path: str | Path,
        table: str | None = None,
        description: str = "",
        schema: TableSchema | None = None,
        read_only: bool = True,
    ):
        """
        Initialize DuckDB source.

        Args:
            name: Logical name for this source
            db_path: Path to database file or ":memory:"
            table: Primary table name (defaults to name)
            description: Description for prompts
            schema: Optional explicit schema
            read_only: Open database read-only (default True)
        """
        if duckdb is None:
            raise ImportError("duckdb is required. Install with: pip install duckdb")

        super().__init__(name=name, description=description, schema=schema)

        self.db_path = str(db_path)
        self.table = table or name
        self.read_only = read_only
        self._connection: Any = None

    def _get_connection(self) -> Any:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = duckdb.connect(self.db_path, read_only=self.read_only)
        return self._connection

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results.

        Args:
            sql: SQL query string

        Returns:
            Query results as DataFrame with normalized types:
            - All integer columns → float64 (avoids int/float mismatch in comparisons)
            - Date columns → datetime64 (timezone-naive)
            - String columns → object (str)
        """
        conn = self._get_connection()
        df = conn.execute(sql).fetchdf()
        return self._normalize_dtypes(df)

    @staticmethod
    def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame column types to canonical set.

        Ensures consistent types throughout the pipeline:
        - int → float64 (prevents int/float comparison issues)
        - timezone-aware datetime → timezone-naive datetime64
        """
        for col in df.columns:
            dtype = df[col].dtype
            # Convert integer types to float64
            if pd.api.types.is_integer_dtype(dtype):
                df[col] = df[col].astype("float64")
            # Strip timezone from datetime columns
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                if hasattr(df[col].dt, "tz") and df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
        return df

    def get_table_names(self) -> list[str]:
        """Get list of tables in the database."""
        conn = self._get_connection()
        result = conn.execute("SHOW TABLES").fetchdf()
        return result["name"].tolist() if "name" in result.columns else []

    def get_schema_description(self) -> str:
        """Generate schema description, auto-detecting if not provided."""
        if self.schema:
            return super().get_schema_description()

        # Auto-detect schema from database
        lines = [f"**{self.name}** (table: `{self.table}`)"]

        if self.description:
            lines.append(f"  {self.description}")

        try:
            conn = self._get_connection()
            schema_df = conn.execute(f"DESCRIBE {self.table}").fetchdf()

            lines.append("  Columns:")
            for _, row in schema_df.iterrows():
                col_name = row.get("column_name", row.get("Field", ""))
                col_type = row.get("column_type", row.get("Type", ""))
                nullable = row.get("null", "YES")
                null_str = "nullable" if nullable == "YES" else "not null"
                lines.append(f"    - {col_name} ({col_type}, {null_str})")

        except Exception:
            lines.append("  (Schema not available)")

        return "\n".join(lines)

    def sample(self, limit: int = 5) -> pd.DataFrame:
        """Get sample rows from the primary table."""
        return self.query(f"SELECT * FROM {self.table} LIMIT {limit}")

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __del__(self):
        """Clean up connection on deletion."""
        self.close()


class DuckDBConnectionPool:
    """
    Connection pool for sharing DuckDB connections across sources.

    Useful when multiple sources use the same database file.

    Example:
        >>> pool = DuckDBConnectionPool("data/market.duckdb")
        >>> prices = DuckDBSourcePooled(pool, name="prices", table="stock_prices")
        >>> companies = DuckDBSourcePooled(pool, name="companies", table="companies")
    """

    def __init__(self, db_path: str | Path, read_only: bool = True):
        if duckdb is None:
            raise ImportError("duckdb is required")

        self.db_path = str(db_path)
        self.read_only = read_only
        self._connection: Any = None

    def get_connection(self) -> Any:
        """Get shared connection."""
        if self._connection is None:
            self._connection = duckdb.connect(self.db_path, read_only=self.read_only)
        return self._connection

    def close(self) -> None:
        """Close the shared connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class DuckDBSourcePooled(DataSource):
    """
    DuckDB source that uses a shared connection pool.

    Use this when multiple sources share the same database.
    """

    def __init__(
        self,
        pool: DuckDBConnectionPool,
        name: str,
        table: str | None = None,
        description: str = "",
        schema: TableSchema | None = None,
    ):
        super().__init__(name=name, description=description, schema=schema)
        self.pool = pool
        self.table = table or name

    def query(self, sql: str) -> pd.DataFrame:
        """Execute query using pooled connection."""
        conn = self.pool.get_connection()
        return conn.execute(sql).fetchdf()

    def get_table_names(self) -> list[str]:
        """Get list of tables in the database."""
        conn = self.pool.get_connection()
        result = conn.execute("SHOW TABLES").fetchdf()
        return result["name"].tolist() if "name" in result.columns else []

    def get_schema_description(self) -> str:
        """Generate schema description."""
        if self.schema:
            return super().get_schema_description()

        lines = [f"**{self.name}** (table: `{self.table}`)"]

        if self.description:
            lines.append(f"  {self.description}")

        try:
            conn = self.pool.get_connection()
            schema_df = conn.execute(f"DESCRIBE {self.table}").fetchdf()

            lines.append("  Columns:")
            for _, row in schema_df.iterrows():
                col_name = row.get("column_name", row.get("Field", ""))
                col_type = row.get("column_type", row.get("Type", ""))
                lines.append(f"    - {col_name} ({col_type})")

        except Exception:
            lines.append("  (Schema not available)")

        return "\n".join(lines)
