"""Core framework components: session management, view metadata, lineage."""

from .session import Session
from .view import ViewMetadata, ColumnStats
from .lineage import get_chart_annotation, get_provenance, build_column_lineage, build_data_scope

__all__ = [
    "Session", "ViewMetadata", "ColumnStats",
    "get_chart_annotation", "get_provenance", "build_column_lineage", "build_data_scope",
]
