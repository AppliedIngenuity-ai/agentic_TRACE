"""
View and column metadata types.

ViewMetadata tracks information about views (DataFrames) in the session.
ColumnStats holds statistics about individual columns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ColumnStats:
    """
    Statistics for a single column.

    Provides summary information without exposing raw data to the LLM.
    Helps the agent understand data characteristics for decision making.

    Attributes:
        name: Column name
        dtype: Data type (string representation)
        unique: Count of unique values
        nulls: Count of null/missing values
        total: Total count of values
        min: Minimum value (for numeric/date columns)
        max: Maximum value (for numeric/date columns)
        mean: Mean value (for numeric columns)
        sample: Sample of distinct values (variety-maximizing)
    """

    name: str
    dtype: str
    unique: int
    nulls: int
    total: int
    min: Any = None
    max: Any = None
    mean: float | None = None
    sample: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON/LLM output."""
        result = {
            "dtype": self.dtype,
            "unique": self.unique,
            "nulls": self.nulls,
            "total": self.total,
        }

        if self.min is not None:
            result["min"] = self._serialize_value(self.min)
        if self.max is not None:
            result["max"] = self._serialize_value(self.max)
        if self.mean is not None:
            result["mean"] = round(self.mean, 4) if isinstance(self.mean, float) else self.mean
        if self.sample:
            result["sample"] = [self._serialize_value(v) for v in self.sample]

        return result

    def _serialize_value(self, value: Any) -> Any:
        """Convert value to JSON-serializable format."""
        # Handle pandas Timestamp, numpy types, etc.
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if hasattr(value, "item"):  # numpy scalar
            return value.item()
        return value

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ColumnStats":
        """Create from dictionary."""
        return cls(
            name=name,
            dtype=data.get("dtype", "unknown"),
            unique=data.get("unique", 0),
            nulls=data.get("nulls", 0),
            total=data.get("total", 0),
            min=data.get("min"),
            max=data.get("max"),
            mean=data.get("mean"),
            sample=data.get("sample", []),
        )


@dataclass
class ViewMetadata:
    """
    Metadata for a view (DataFrame) in the session.

    Tracks view identity, lineage, and statistics without holding the data itself.
    The actual DataFrame is stored separately in Session.dataframes.

    Attributes:
        name: View name (unique within session)
        rows: Number of rows
        columns: Dict of column name -> ColumnStats
        source: Source description (table name, parent view, etc.)
        created_at: Timestamp of creation
        operation: Description of operation that created this view
        parent_views: Names of parent views (for lineage tracking)
        is_finalized: Whether this view is marked as final output
    """

    name: str
    rows: int
    columns: dict[str, ColumnStats]
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    operation: str = ""
    parent_views: list[str] = field(default_factory=list)
    is_finalized: bool = False

    # Extension point for domain-specific metadata
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON/LLM output."""
        return {
            "name": self.name,
            "rows": self.rows,
            "columns": {name: stats.to_dict() for name, stats in self.columns.items()},
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "operation": self.operation,
            "parent_views": self.parent_views,
            "is_finalized": self.is_finalized,
            **self.extra,
        }

    def to_summary(self) -> str:
        """Generate concise summary for LLM context."""
        col_types = ", ".join(f"{n}({s.dtype})" for n, s in list(self.columns.items())[:5])
        if len(self.columns) > 5:
            col_types += f", ... (+{len(self.columns) - 5} more)"

        parts = [f"'{self.name}': {self.rows} rows, columns=[{col_types}]"]

        if self.source:
            parts.append(f"source={self.source}")

        return " | ".join(parts)

    def get_column_names(self) -> list[str]:
        """Get list of column names."""
        return list(self.columns.keys())

    def get_column_types(self) -> dict[str, str]:
        """Get mapping of column names to types."""
        return {name: stats.dtype for name, stats in self.columns.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ViewMetadata":
        """Create from dictionary."""
        columns = {
            name: ColumnStats.from_dict(name, stats_data)
            for name, stats_data in data.get("columns", {}).items()
        }

        return cls(
            name=data["name"],
            rows=data.get("rows", 0),
            columns=columns,
            source=data.get("source", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            operation=data.get("operation", ""),
            parent_views=data.get("parent_views", []),
            is_finalized=data.get("is_finalized", False),
            extra={k: v for k, v in data.items() if k not in {
                "name", "rows", "columns", "source", "created_at",
                "operation", "parent_views", "is_finalized"
            }},
        )
