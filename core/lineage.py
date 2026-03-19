"""
Column lineage tracking and provenance generation.

Traces how each column in a view was derived by walking the view DAG
(parent_views) and inspecting tool metadata stored in ViewMetadata.extra.

Two public outputs:
  - get_chart_annotation(): compact 3-4 line string for embedding in chart PNGs
  - get_provenance(): detailed multi-line derivation chain for provenance files

All functions are best-effort: they return partial results on errors rather
than raising exceptions, so lineage never blocks chart or finalize operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .session import Session


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ColumnLineage:
    """Lineage for a single column in a view."""
    column: str
    operation: str          # "source", "computed", "aggregated", "indicator", "window", "passthrough", "joined", "pivoted", "unknown"
    expression: str | None = None
    source_view: str | None = None
    source_column: str | None = None
    details: dict = field(default_factory=dict)
    ancestors: list[ColumnLineage] = field(default_factory=list)


@dataclass
class DataScope:
    """Summary of what data a view contains."""
    entities: list[str] | None = None
    entity_label: str = "Entities"
    date_range: tuple[str, str] | None = None
    time_label: str = "Date range"
    row_count: int = 0


# ---------------------------------------------------------------------------
# Internal: recursive column tracer
# ---------------------------------------------------------------------------

def _get_tool_info(session: Session, view_name: str) -> tuple[str, dict, dict, list[str]]:
    """Extract tool_name, tool_args, extra metadata, and parent_views from a view."""
    meta = session.get_view(view_name)
    if meta is None:
        return "", {}, {}, []
    extra = meta.extra or {}
    tool_name = extra.get("tool_name", meta.operation or "")
    tool_args = extra.get("tool_args", {})
    parents = meta.parent_views or []
    return tool_name, tool_args, extra, parents


def build_column_lineage(
    session: Session,
    view_name: str,
    column_name: str,
    _visited: set | None = None,
) -> ColumnLineage:
    """
    Recursively trace how a column was derived.

    Walks the view DAG using parent_views and tool metadata to determine
    whether the column was computed, aggregated, joined, etc.
    """
    if _visited is None:
        _visited = set()

    # Prevent infinite loops (should never happen but safety first)
    key = (view_name, column_name)
    if key in _visited:
        return ColumnLineage(column=column_name, operation="circular_ref")
    _visited.add(key)

    tool_name, tool_args, extra, parents = _get_tool_info(session, view_name)

    # No metadata at all — terminal
    if not tool_name:
        return ColumnLineage(column=column_name, operation="source", source_view=view_name)

    # --- create_view (terminal if from data source, passthrough if from view) ---
    if tool_name == "create_view":
        if extra.get("from_view") and parents:
            return _passthrough(session, parents[0], column_name, _visited)
        # From data source — terminal
        return ColumnLineage(
            column=column_name,
            operation="source",
            source_view=view_name,
            details=_create_view_details(tool_args, session.domain_config.extra_detail_keys),
        )

    # --- add_column ---
    if tool_name == "add_column":
        col_added = extra.get("column_added")
        if column_name == col_added:
            expr = extra.get("expression", tool_args.get("expression", ""))
            ancestor_lineages = []
            if parents:
                refs = _extract_column_refs(expr, session, parents[0])
                for ref in refs:
                    ancestor_lineages.append(
                        build_column_lineage(session, parents[0], ref, _visited)
                    )
            return ColumnLineage(
                column=column_name,
                operation="computed",
                expression=expr,
                source_view=parents[0] if parents else None,
                ancestors=ancestor_lineages,
            )
        # Not the added column — passthrough
        if parents:
            return _passthrough(session, parents[0], column_name, _visited)

    # --- aggregate ---
    if tool_name == "aggregate":
        group_by = tool_args.get("group_by", [])
        aggregations = tool_args.get("aggregations", {})
        if column_name in (group_by or []):
            # Group key — passthrough from parent
            if parents:
                return _passthrough(session, parents[0], column_name, _visited)
        # Check if column is an aggregated result
        if column_name in aggregations:
            agg_func = aggregations[column_name]
            ancestor = None
            if parents:
                # The aggregated column name is also the source column name
                ancestor = build_column_lineage(session, parents[0], column_name, _visited)
            return ColumnLineage(
                column=column_name,
                operation="aggregated",
                expression=f"{agg_func}({column_name})",
                source_view=parents[0] if parents else None,
                details={"function": agg_func, "group_by": group_by},
                ancestors=[ancestor] if ancestor else [],
            )
        # Might be a renamed agg column — check tool_args
        if parents:
            return _passthrough(session, parents[0], column_name, _visited)

    # --- join ---
    if tool_name == "join":
        column_origins = extra.get("column_origins", {})
        if column_name in column_origins:
            origin = column_origins[column_name]
            src_view = origin.get("view", "")
            src_col = origin.get("column", column_name)
            ancestor = build_column_lineage(session, src_view, src_col, _visited) if src_view else None
            return ColumnLineage(
                column=column_name,
                operation="joined",
                source_view=src_view,
                source_column=src_col,
                details={"join_on": tool_args.get("on", []),
                         "left_on": tool_args.get("left_on"),
                         "right_on": tool_args.get("right_on"),
                         "how": tool_args.get("how", "inner")},
                ancestors=[ancestor] if ancestor else [],
            )
        # Column not in origins — try matching by checking both parents
        for parent in parents:
            parent_meta = session.get_view(parent)
            if parent_meta and column_name in parent_meta.columns:
                ancestor = build_column_lineage(session, parent, column_name, _visited)
                return ColumnLineage(
                    column=column_name,
                    operation="joined",
                    source_view=parent,
                    source_column=column_name,
                    ancestors=[ancestor],
                )

    # --- compute_indicator ---
    if tool_name == "compute_indicator":
        out_col = extra.get("output_column", tool_args.get("output_column", ""))
        if column_name == out_col:
            indicator = extra.get("indicator", tool_args.get("indicator", ""))
            period = extra.get("period", tool_args.get("period", ""))
            input_col = tool_args.get("column", "")
            ancestor = None
            if parents and input_col:
                ancestor = build_column_lineage(session, parents[0], input_col, _visited)
            return ColumnLineage(
                column=column_name,
                operation="indicator",
                expression=f"{indicator}({input_col}, {period})" if period else f"{indicator}({input_col})",
                source_view=parents[0] if parents else None,
                details={"indicator": indicator, "period": period, "input_column": input_col},
                ancestors=[ancestor] if ancestor else [],
            )
        if parents:
            return _passthrough(session, parents[0], column_name, _visited)

    # --- apply_window ---
    if tool_name == "apply_window":
        out_col = extra.get("output_column", tool_args.get("output_column", ""))
        if column_name == out_col:
            func = extra.get("function", tool_args.get("function", ""))
            input_col = extra.get("input_column", tool_args.get("column", ""))
            window = tool_args.get("window", "")
            partition = tool_args.get("partition_by", [])
            ancestor = None
            if parents and input_col:
                ancestor = build_column_lineage(session, parents[0], input_col, _visited)
            return ColumnLineage(
                column=column_name,
                operation="window",
                expression=f"{func}({input_col}, window={window})" if window else f"{func}({input_col})",
                source_view=parents[0] if parents else None,
                details={"function": func, "input_column": input_col, "window": window, "partition_by": partition},
                ancestors=[ancestor] if ancestor else [],
            )
        if parents:
            return _passthrough(session, parents[0], column_name, _visited)

    # --- sort, filter_view, concat — passthrough ---
    if tool_name in ("sort", "filter_view"):
        if parents:
            return _passthrough(session, parents[0], column_name, _visited)

    if tool_name == "concat":
        # Columns pass through from first parent
        if parents:
            return _passthrough(session, parents[0], column_name, _visited)

    # --- pivot ---
    if tool_name == "pivot":
        index_col = tool_args.get("index")
        values_col = tool_args.get("values")
        agg_func = tool_args.get("aggfunc", "mean")
        if column_name == index_col and parents:
            return _passthrough(session, parents[0], column_name, _visited)
        # Pivot-generated columns are values aggregated by column header
        return ColumnLineage(
            column=column_name,
            operation="pivoted",
            expression=f"{agg_func}({values_col})" if values_col else None,
            source_view=parents[0] if parents else None,
            details={"index": index_col, "values": values_col, "aggfunc": agg_func,
                     "columns_param": tool_args.get("columns")},
        )

    # --- fallback: try passthrough to first parent ---
    if parents:
        return _passthrough(session, parents[0], column_name, _visited)

    return ColumnLineage(column=column_name, operation="unknown", source_view=view_name)


def _passthrough(session: Session, parent_view: str, column_name: str, visited: set) -> ColumnLineage:
    """Column passes through unchanged from a parent view."""
    ancestor = build_column_lineage(session, parent_view, column_name, visited)
    return ColumnLineage(
        column=column_name,
        operation="passthrough",
        source_view=parent_view,
        source_column=column_name,
        ancestors=[ancestor],
    )


def _create_view_details(tool_args: dict, extra_keys: list[str] | None = None) -> dict:
    """Extract relevant create_view details for lineage.

    Args:
        tool_args: Tool arguments dict.
        extra_keys: Additional domain-specific keys to extract from tool_args
                    (e.g., ["tickers"] for stocks). Configured via DomainConfig.
    """
    details = {}
    # Standard keys (domain-agnostic)
    for key in ("source", "where", "filters", "sort_by", "limit", "columns"):
        val = tool_args.get(key)
        if val:
            details["data_source" if key == "source" else key] = val
    # Domain-specific extra keys
    for key in (extra_keys or []):
        val = tool_args.get(key)
        if val:
            details[key] = val
    return details


def _extract_column_refs(expression: str, session: Session, parent_view: str) -> list[str]:
    """
    Extract column name references from an expression string.

    Uses the parent view's actual columns to match — only returns names
    that exist in the parent DataFrame.
    """
    parent_meta = session.get_view(parent_view)
    if not parent_meta:
        return []
    parent_cols = set(parent_meta.columns.keys())

    # Find all word-like tokens in expression and intersect with actual columns
    tokens = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expression))
    return sorted(tokens & parent_cols)


# ---------------------------------------------------------------------------
# Data scope (computed from actual DataFrame, not lineage)
# ---------------------------------------------------------------------------

def build_data_scope(session: Session, view_name: str) -> DataScope:
    """Compute data scope from the actual DataFrame using session's DomainConfig."""
    df = session.get_dataframe(view_name)
    if df is None:
        return DataScope()

    config = session.domain_config
    scope = DataScope(
        row_count=len(df),
        entity_label=config.entity_label,
        time_label=config.time_label,
    )

    # Entity detection (e.g., tickers, user IDs, product SKUs)
    for col in config.entity_columns:
        if col in df.columns:
            entities = sorted(df[col].dropna().unique().tolist())
            scope.entities = [str(e) for e in entities]
            break

    # Time range detection
    for col in config.time_columns:
        if col in df.columns:
            dates = df[col].dropna()
            if len(dates) > 0:
                scope.date_range = (str(dates.min()), str(dates.max()))
            break

    return scope


# ---------------------------------------------------------------------------
# Lineage flattening / formatting
# ---------------------------------------------------------------------------

def _flatten_lineage(lineage: ColumnLineage, depth: int = 0) -> str:
    """Flatten a ColumnLineage tree into a readable string."""
    if lineage.operation == "source":
        src = lineage.details.get("data_source", lineage.source_view or "?")
        return f"{lineage.column} [from {src}]"

    if lineage.operation == "passthrough":
        # Skip passthrough nodes — just recurse to ancestor
        if lineage.ancestors:
            return _flatten_lineage(lineage.ancestors[0], depth)
        return f"{lineage.column}"

    if lineage.operation == "computed":
        return lineage.expression or lineage.column

    if lineage.operation == "aggregated":
        return lineage.expression or lineage.column

    if lineage.operation == "indicator":
        return lineage.expression or lineage.column

    if lineage.operation == "window":
        return lineage.expression or lineage.column

    if lineage.operation == "joined":
        src = lineage.source_view or "?"
        col = lineage.source_column or lineage.column
        if lineage.ancestors:
            inner = _flatten_lineage(lineage.ancestors[0], depth + 1)
            if inner != col:
                return f"{inner} [via {src}]"
        return f"{col} [from {src}]"

    if lineage.operation == "pivoted":
        return lineage.expression or lineage.column

    return lineage.column


def _detailed_lineage(lineage: ColumnLineage, indent: int = 0) -> list[str]:
    """Build detailed provenance lines for a column."""
    prefix = "  " * indent
    lines = []

    if lineage.operation == "source":
        src = lineage.details.get("data_source", lineage.source_view or "?")
        detail = f"{prefix}{lineage.column}: source column from '{src}'"
        extras = []
        if lineage.details.get("where"):
            extras.append(f"where: {lineage.details['where']}")
        # Include any domain-specific detail keys (e.g., tickers, filters)
        for key in ("filters", "sort_by", "limit", "columns"):
            if lineage.details.get(key):
                extras.append(f"{key}: {lineage.details[key]}")
        # Domain extras from DomainConfig.extra_detail_keys are stored as
        # regular dict entries — include any non-standard keys
        _standard = {"data_source", "where", "filters", "sort_by", "limit", "columns"}
        for key, val in lineage.details.items():
            if key not in _standard and val:
                extras.append(f"{key}: {val}")
        if extras:
            detail += f" ({', '.join(extras)})"
        lines.append(detail)
        return lines

    if lineage.operation == "passthrough":
        if lineage.ancestors:
            return _detailed_lineage(lineage.ancestors[0], indent)
        lines.append(f"{prefix}{lineage.column}: passthrough")
        return lines

    if lineage.operation == "computed":
        lines.append(f"{prefix}{lineage.column} = {lineage.expression}")
        for anc in lineage.ancestors:
            lines.extend(_detailed_lineage(anc, indent + 1))
        return lines

    if lineage.operation == "aggregated":
        func = lineage.details.get("function", "?")
        group = lineage.details.get("group_by", [])
        lines.append(f"{prefix}{lineage.column} = {func}(), grouped by {group}")
        for anc in lineage.ancestors:
            lines.extend(_detailed_lineage(anc, indent + 1))
        return lines

    if lineage.operation == "indicator":
        lines.append(f"{prefix}{lineage.column} = {lineage.expression}")
        for anc in lineage.ancestors:
            lines.extend(_detailed_lineage(anc, indent + 1))
        return lines

    if lineage.operation == "window":
        lines.append(f"{prefix}{lineage.column} = {lineage.expression}")
        partition = lineage.details.get("partition_by")
        if partition:
            lines.append(f"{prefix}  partition_by: {partition}")
        for anc in lineage.ancestors:
            lines.extend(_detailed_lineage(anc, indent + 1))
        return lines

    if lineage.operation == "joined":
        src = lineage.source_view or "?"
        col = lineage.source_column or lineage.column
        join_on = lineage.details.get("join_on") or lineage.details.get("left_on")
        how = lineage.details.get("how", "inner")
        lines.append(f"{prefix}{lineage.column}: from '{src}'.{col} ({how} join on {join_on})")
        for anc in lineage.ancestors:
            lines.extend(_detailed_lineage(anc, indent + 1))
        return lines

    if lineage.operation == "pivoted":
        lines.append(f"{prefix}{lineage.column} = pivot({lineage.expression})")
        cols_param = lineage.details.get("columns_param")
        if cols_param:
            lines.append(f"{prefix}  pivot columns: {cols_param}")
        return lines

    lines.append(f"{prefix}{lineage.column}: {lineage.operation}")
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_view_chain(session: Session, view_name: str, _visited: set | None = None) -> list[str]:
    """
    Build the view transformation chain by recursively walking parent_views.

    Returns a list of view names from source to current view.
    Example: ['stock_prices', 'all_returns', 'filtered_returns']
    """
    if _visited is None:
        _visited = set()

    if view_name in _visited:
        return [view_name]
    _visited.add(view_name)

    meta = session.get_view(view_name)
    if not meta:
        return [view_name]

    parents = meta.parent_views
    if not parents:
        # Leaf node (source view)
        return [view_name]

    # For single parent, recurse
    if len(parents) == 1:
        parent_chain = _build_view_chain(session, parents[0], _visited)
        return parent_chain + [view_name]

    # Multiple parents (join, concat) — show all branches
    # Format: parent1 + parent2 → current_view
    # Just return current view and indicate merge
    return [f"({' + '.join(parents)})", view_name]


def get_chart_annotation(
    session: Session,
    view_name: str,
    x: str,
    y: str,
    color_by: str | None = None,
) -> str:
    """
    Generate a compact annotation for embedding in chart images.

    Returns 3-5 lines summarizing:
      Y: <formula or column lineage>
      X: <column>
      Views: <transformation chain>
      Data: <entities> | <date range> | <row count>
    """
    try:
        lines = []

        # Y axis lineage
        y_lineage = build_column_lineage(session, view_name, y)
        y_flat = _flatten_lineage(y_lineage)
        lines.append(f"Y: {y_flat}")

        # X axis (usually just the column name)
        x_lineage = build_column_lineage(session, view_name, x)
        x_flat = _flatten_lineage(x_lineage)
        if x_flat != x:
            lines.append(f"X: {x_flat}")
        else:
            lines.append(f"X: {x}")

        # View transformation chain
        view_chain = _build_view_chain(session, view_name)
        if len(view_chain) > 1:
            # Show transformation path
            chain_str = " → ".join(view_chain)
            lines.append(f"Views: {chain_str}")
        elif view_chain:
            # Single view (direct from source)
            lines.append(f"View: {view_chain[0]}")

        # Data scope
        scope = build_data_scope(session, view_name)
        scope_parts = []
        if scope.entities:
            entity_str = (
                ", ".join(scope.entities) if len(scope.entities) <= 5
                else f"{len(scope.entities)} {scope.entity_label.lower()}"
            )
            scope_parts.append(entity_str)
        if scope.date_range:
            scope_parts.append(f"{scope.date_range[0]} \u2192 {scope.date_range[1]}")
        scope_parts.append(f"{scope.row_count} rows")
        lines.append(f"Data: {' | '.join(scope_parts)}")

        return "\n".join(lines)
    except Exception:
        return ""


def get_provenance(
    session: Session,
    view_name: str,
    x: str | None = None,
    y: str | None = None,
    color_by: str | None = None,
) -> str:
    """
    Generate detailed provenance for a view.

    Returns a multi-line string with full derivation chain for x and y columns
    (if provided), plus data scope.
    """
    try:
        lines = [f"Provenance for view: {view_name}", ""]

        if y:
            y_lineage = build_column_lineage(session, view_name, y)
            lines.append(f"Y-axis: {y}")
            lines.extend(_detailed_lineage(y_lineage, indent=1))
            lines.append("")

        if x:
            x_lineage = build_column_lineage(session, view_name, x)
            lines.append(f"X-axis: {x}")
            lines.extend(_detailed_lineage(x_lineage, indent=1))
            lines.append("")

        if color_by:
            cb_lineage = build_column_lineage(session, view_name, color_by)
            lines.append(f"Color: {color_by}")
            lines.extend(_detailed_lineage(cb_lineage, indent=1))
            lines.append("")

        # Data scope
        scope = build_data_scope(session, view_name)
        lines.append("Data scope:")
        if scope.entities:
            lines.append(f"  {scope.entity_label}: {', '.join(scope.entities)}")
        if scope.date_range:
            lines.append(f"  {scope.time_label}: {scope.date_range[0]} \u2192 {scope.date_range[1]}")
        lines.append(f"  Rows: {scope.row_count}")

        # View chain (all ancestors)
        lines.append("")
        lines.append("View lineage:")
        _append_view_chain(session, view_name, lines, indent=1, visited=set())

        return "\n".join(lines)
    except Exception:
        return f"Provenance unavailable for {view_name}"


def _append_view_chain(
    session: Session,
    view_name: str,
    lines: list[str],
    indent: int,
    visited: set,
) -> None:
    """Recursively append view lineage chain."""
    if view_name in visited:
        return
    visited.add(view_name)
    prefix = "  " * indent

    tool_name, tool_args, extra, parents = _get_tool_info(session, view_name)
    op_desc = _describe_operation(tool_name, tool_args, extra)
    lines.append(f"{prefix}{view_name}: {op_desc}")

    for parent in parents:
        _append_view_chain(session, parent, lines, indent + 1, visited)


def _describe_operation(tool_name: str, tool_args: dict, extra: dict) -> str:
    """One-line description of what a tool operation did."""
    if not tool_name:
        return "unknown"

    if tool_name == "create_view":
        src = tool_args.get("source", "?")
        parts = [f"create_view(source={src})"]
        # Include any non-standard args (domain-specific like tickers, group, etc.)
        _standard_cv = {"source", "view_name", "columns", "filters", "where",
                        "sort_by", "limit", "add_columns", "output_view"}
        for key, val in tool_args.items():
            if key not in _standard_cv and val:
                parts.append(f"{key}={val}")
        if tool_args.get("where"):
            parts.append(f"where=\"{tool_args['where']}\"")
        if tool_args.get("sort_by"):
            parts.append(f"sort_by={tool_args['sort_by']}")
        if tool_args.get("limit"):
            parts.append(f"limit={tool_args['limit']}")
        return ", ".join(parts)

    if tool_name == "add_column":
        col = extra.get("column_added", tool_args.get("column_name", "?"))
        expr = extra.get("expression", tool_args.get("expression", "?"))
        return f"add_column({col} = {expr})"

    if tool_name == "aggregate":
        aggs = tool_args.get("aggregations", {})
        group = tool_args.get("group_by", [])
        return f"aggregate({aggs}, group_by={group})"

    if tool_name == "join":
        how = tool_args.get("how", "inner")
        on = tool_args.get("on", tool_args.get("left_on", "?"))
        left = tool_args.get("left_view", "?")
        right = tool_args.get("right_view", "?")
        return f"{how}_join({left}, {right}, on={on})"

    if tool_name == "compute_indicator":
        ind = extra.get("indicator", tool_args.get("indicator", "?"))
        period = extra.get("period", tool_args.get("period", ""))
        col = tool_args.get("column", "?")
        return f"compute_indicator({ind}, column={col}, period={period})"

    if tool_name == "apply_window":
        func = extra.get("function", tool_args.get("function", "?"))
        col = tool_args.get("column", "?")
        window = tool_args.get("window", "?")
        return f"apply_window({func}, column={col}, window={window})"

    if tool_name == "sort":
        by = tool_args.get("by", "?")
        return f"sort(by={by})"

    if tool_name == "filter_view":
        cond = extra.get("condition", tool_args.get("condition", "?"))
        return f"filter(where=\"{cond}\")"

    if tool_name == "concat":
        views = extra.get("source_views", tool_args.get("views", []))
        return f"concat({views})"

    if tool_name == "pivot":
        idx = tool_args.get("index", "?")
        cols = tool_args.get("columns", "?")
        vals = tool_args.get("values", "?")
        return f"pivot(index={idx}, columns={cols}, values={vals})"

    if tool_name == "finalize":
        return "finalize"

    return tool_name
