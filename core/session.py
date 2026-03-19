"""
Session state management.

Session holds all state for a single agent run:
- Views (DataFrames) and their metadata
- Execution log
- Finalized outputs
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from .view import ViewMetadata, ColumnStats
from ..utils.stats import compute_column_stats


@dataclass
class DomainConfig:
    """
    Domain-specific configuration for lineage, scope detection, and tool hints.

    Core uses generic defaults; domain examples (e.g., stocks) override
    with domain-specific column names, labels, and examples.

    Attributes:
        entity_columns: Column names to try for entity detection
                        (e.g., ["ticker", "symbol"] for stocks)
        entity_label: Display label for entities (e.g., "Tickers", "Users")
        time_columns: Column names to try for time dimension detection
        time_label: Display label for time range
        extra_detail_keys: Extra tool_args keys to include in lineage details
                           (e.g., ["tickers"] for stocks create_view)
        partition_candidates: Column names for auto-detecting missing partition_by
        partition_label: Label for partition context (e.g., "per-stock", "per-user")
        partition_example: Example for partition_by in descriptions
                           (e.g., "['ticker']")
    """
    entity_columns: list[str] = field(default_factory=list)
    entity_label: str = "Entities"
    time_columns: list[str] = field(
        default_factory=lambda: ["date", "Date", "DATE", "timestamp", "Timestamp"]
    )
    time_label: str = "Date range"
    extra_detail_keys: list[str] = field(default_factory=list)
    partition_candidates: list[str] = field(default_factory=list)
    partition_label: str = "per-entity"
    partition_example: str = "['id']"


@dataclass
class StepLog:
    """
    Log entry for a single execution step.

    Records what happened at each step for debugging and replay.
    """

    step_number: int
    timestamp: datetime
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    summary: str = ""
    error: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "step": self.step_number,
            "timestamp": self.timestamp.isoformat(),
            "tool": self.tool_name,
            "args": self.tool_args,
            "result": self.result,
            "summary": self.summary,
            "error": self.error,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "duration_ms": self.duration_ms,
        }


class Session:
    """
    Container for all state during an agent run.

    The Session tracks:
        - views: Metadata about each view (name -> ViewMetadata)
        - dataframes: Actual DataFrames (name -> pd.DataFrame)
        - steps: Execution log (list of StepLog)
        - finalized: Names of views marked as final output

    Views and DataFrames are stored separately so metadata can be
    serialized without including raw data.

    Example:
        >>> session = Session()
        >>> session.add_view("prices", df, source="stock_prices table")
        >>> session.get_view("prices")  # Returns ViewMetadata
        >>> session.get_dataframe("prices")  # Returns DataFrame
    """

    def __init__(
        self,
        metadata_extractor: Callable[[pd.DataFrame, dict], dict] | None = None,
        domain_config: DomainConfig | None = None,
    ):
        """
        Initialize a new session.

        Args:
            metadata_extractor: Optional function to extend view metadata.
                               Receives (df, base_metadata) -> extended_metadata
            domain_config: Domain-specific configuration for lineage and tools.
                          Defaults to generic (no entity detection).
        """
        self._views: dict[str, ViewMetadata] = {}
        self._dataframes: dict[str, pd.DataFrame] = {}
        self._steps: list[StepLog] = []
        self._finalized: list[str] = []
        self._human_inputs: list[dict[str, str]] = []  # [{question, response}, ...]
        self._metadata_extractor = metadata_extractor
        self.domain_config = domain_config or DomainConfig()
        self._created_at = datetime.now()

        # Session identification (set by orchestrator)
        self.session_id: str = ""
        self.query: str = ""

    @property
    def views(self) -> dict[str, ViewMetadata]:
        """Get all view metadata."""
        return self._views

    @property
    def dataframes(self) -> dict[str, pd.DataFrame]:
        """Get all DataFrames."""
        return self._dataframes

    @property
    def steps(self) -> list[StepLog]:
        """Get execution log."""
        return self._steps

    @property
    def finalized(self) -> list[str]:
        """Get names of finalized views."""
        return self._finalized

    @property
    def human_inputs(self) -> list[dict[str, str]]:
        """Get all human Q&A pairs recorded during this session."""
        return self._human_inputs

    def record_human_input(self, question: str, response: str) -> None:
        """Record a human clarification (question + answer) for use in validation."""
        self._human_inputs.append({"question": question, "response": response})

    def add_view(
        self,
        name: str,
        df: pd.DataFrame,
        source: str = "",
        operation: str = "",
        parent_views: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ViewMetadata:
        """
        Add a view (DataFrame) to the session.

        Computes column statistics and creates ViewMetadata automatically.

        Args:
            name: Unique view name
            df: The DataFrame
            source: Source description (e.g., table name)
            operation: Description of operation that created this
            parent_views: Names of parent views (for lineage)
            extra_metadata: Additional domain-specific metadata

        Returns:
            The created ViewMetadata

        Raises:
            ValueError: If view name already exists
        """
        if name in self._views:
            raise ValueError(f"View '{name}' already exists. Use update_view() to modify.")

        # Compute column statistics
        columns = compute_column_stats(df)

        # Create base metadata
        metadata = ViewMetadata(
            name=name,
            rows=len(df),
            columns=columns,
            source=source,
            operation=operation,
            parent_views=parent_views or [],
            extra=extra_metadata or {},
        )

        # Apply custom metadata extractor if set
        if self._metadata_extractor:
            extended = self._metadata_extractor(df, metadata.to_dict())
            # Merge extended fields into extra
            for key, value in extended.items():
                if key not in {"name", "rows", "columns", "source", "created_at",
                              "operation", "parent_views", "is_finalized"}:
                    metadata.extra[key] = value

        self._views[name] = metadata
        self._dataframes[name] = df.copy()

        return metadata

    def update_view(
        self,
        name: str,
        df: pd.DataFrame,
        operation: str = "",
    ) -> ViewMetadata:
        """
        Update an existing view with new DataFrame.

        Args:
            name: View name (must exist)
            df: New DataFrame
            operation: Description of update operation

        Returns:
            Updated ViewMetadata

        Raises:
            KeyError: If view doesn't exist
        """
        if name not in self._views:
            raise KeyError(f"View '{name}' not found")

        old_meta = self._views[name]

        # Compute new statistics
        columns = compute_column_stats(df)

        # Create updated metadata
        metadata = ViewMetadata(
            name=name,
            rows=len(df),
            columns=columns,
            source=old_meta.source,
            operation=operation or old_meta.operation,
            parent_views=old_meta.parent_views,
            extra=old_meta.extra.copy(),
        )

        # Apply custom metadata extractor if set
        if self._metadata_extractor:
            extended = self._metadata_extractor(df, metadata.to_dict())
            for key, value in extended.items():
                if key not in {"name", "rows", "columns", "source", "created_at",
                              "operation", "parent_views", "is_finalized"}:
                    metadata.extra[key] = value

        self._views[name] = metadata
        self._dataframes[name] = df.copy()

        return metadata

    def get_view(self, name: str) -> ViewMetadata | None:
        """Get view metadata by name."""
        return self._views.get(name)

    def get_view_required(self, name: str) -> ViewMetadata:
        """Get view metadata, raising if not found."""
        if name not in self._views:
            available = list(self._views.keys())
            raise KeyError(f"View '{name}' not found. Available: {available}")
        return self._views[name]

    def get_dataframe(self, name: str) -> pd.DataFrame | None:
        """Get DataFrame by view name."""
        return self._dataframes.get(name)

    def get_dataframe_required(self, name: str) -> pd.DataFrame:
        """Get DataFrame, raising if not found."""
        if name not in self._dataframes:
            available = list(self._dataframes.keys())
            raise KeyError(f"View '{name}' not found. Available: {available}")
        return self._dataframes[name]

    def finalize_view(self, name: str) -> None:
        """
        Mark a view as final output.

        Args:
            name: View name to finalize

        Raises:
            KeyError: If view doesn't exist
        """
        if name not in self._views:
            raise KeyError(f"View '{name}' not found")

        self._views[name].is_finalized = True
        if name not in self._finalized:
            self._finalized.append(name)

    def unfinalize_all(self) -> None:
        """Un-finalize all views (used by post-finalize reopen)."""
        for name in self._finalized:
            if name in self._views:
                self._views[name].is_finalized = False
        self._finalized.clear()

    def log_step(
        self,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        summary: str = "",
        error: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: int = 0,
    ) -> StepLog:
        """
        Log an execution step.

        Args:
            tool_name: Name of tool executed
            tool_args: Arguments passed to tool
            result: Tool result dict
            summary: Human-readable summary
            error: Error message if failed
            tokens_in: Input tokens used
            tokens_out: Output tokens used
            duration_ms: Execution duration

        Returns:
            The created StepLog
        """
        step = StepLog(
            step_number=len(self._steps) + 1,
            timestamp=datetime.now(),
            tool_name=tool_name,
            tool_args=tool_args,
            result=result,
            summary=summary,
            error=error,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
        )
        self._steps.append(step)
        return step

    def get_views_summary(self) -> str:
        """
        Generate summary of all views for LLM context.

        Returns:
            Formatted string listing all views and their key stats
        """
        if not self._views:
            return "No views created yet."

        lines = ["Current views:"]
        for name, meta in self._views.items():
            status = " [FINAL]" if meta.is_finalized else ""
            lines.append(f"  - {meta.to_summary()}{status}")

        return "\n".join(lines)

    def get_finalized_views(self) -> list[tuple[str, pd.DataFrame, ViewMetadata]]:
        """
        Get all finalized views with their data.

        Returns:
            List of (name, dataframe, metadata) tuples
        """
        return [
            (name, self._dataframes[name], self._views[name])
            for name in self._finalized
            if name in self._dataframes
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize session state for logging/replay.

        Note: Does not include raw DataFrames.
        """
        return {
            "session_id": self.session_id,
            "query": self.query,
            "created_at": self._created_at.isoformat(),
            "views": {name: meta.to_dict() for name, meta in self._views.items()},
            "views_created": list(self._views.keys()),
            "steps": [step.to_dict() for step in self._steps],
            "total_steps": len(self._steps),
            "finalized": self._finalized,
            "human_inputs": self._human_inputs,
        }

    def __len__(self) -> int:
        """Number of views in session."""
        return len(self._views)

    def __contains__(self, view_name: str) -> bool:
        """Check if view exists."""
        return view_name in self._views
