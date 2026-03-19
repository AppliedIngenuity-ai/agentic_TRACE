"""
Join tool - join or stack views.
"""

from typing import TYPE_CHECKING

import pandas as pd

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session
    from ...data.sources import DataSource


class JoinTool(BaseTool):
    """
    Join or stack views.

    mode="join" (default): Join two views on specified columns (inner, left, right, outer).
    mode="stack": Vertically concatenate multiple views (stack rows).
    """

    name = "join"
    category = ToolCategory.TRANSFORM

    ALLOWED_JOINS = ["inner", "left", "right", "outer", "cross"]

    def __init__(self):
        super().__init__()
        self._source_map: dict[str, "DataSource"] = {}

    def set_data_sources(self, sources: list["DataSource"]) -> None:
        self._source_map = {s.name: s for s in sources}

    def _get_df(self, session: "Session", name: str) -> pd.DataFrame | None:
        """Get DataFrame from session view, falling back to data source."""
        if name in session.views:
            return session.get_dataframe_required(name)
        if name in self._source_map:
            src = self._source_map[name]
            table = getattr(src, "table", src.name)
            return src.query(f"SELECT * FROM {table}")
        return None

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="mode",
                param_type="string",
                description="'join' (default) to join two views, or 'stack' to concatenate views vertically",
                required=False,
                default="join",
                enum=["join", "stack"],
            ),
            ToolParameter(
                name="left_view",
                param_type="string",
                description="Name of the left view (join mode)",
                required=False,
            ),
            ToolParameter(
                name="right_view",
                param_type="string",
                description="Name of the right view (join mode)",
                required=False,
            ),
            ToolParameter(
                name="views",
                param_type="array",
                description="List of view names to stack (stack mode)",
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="output_view",
                param_type="string",
                description="Name for the output view",
                required=True,
            ),
            ToolParameter(
                name="on",
                param_type="array",
                description="Columns to join on (must exist in both views)",
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="left_on",
                param_type="array",
                description="Columns from left view to join on",
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="right_on",
                param_type="array",
                description="Columns from right view to join on",
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="how",
                param_type="string",
                description="Join type: inner, left, right, outer",
                required=False,
                default="inner",
                enum=["inner", "left", "right", "outer"],
            ),
            ToolParameter(
                name="finalize",
                param_type="boolean",
                description="Set finalize=true to return this view. You can return one, many, or none if there is no valid solution.",
                required=False,
                default=False,
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        mode = kwargs.get("mode", "join")

        if mode == "stack":
            return self._execute_stack(session, kwargs)
        return self._execute_join(session, kwargs)

    # ── Stack mode (concat) ──────────────────────────────────────────────

    def _execute_stack(self, session: "Session", kwargs: dict) -> ToolResult:
        if error := self.validate_required(kwargs, "views", "output_view"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        views = kwargs["views"]
        output_view = kwargs["output_view"]

        if len(views) < 2:
            return self.error(
                "Need at least 2 views to stack",
                ErrorType.VALIDATION_ERROR,
            )

        if output_view in session.views:
            return self.error(
                f"View '{output_view}' already exists",
                ErrorType.VIEW_ERROR,
            )

        for view_name in views:
            if error := self.validate_view_exists(session, view_name):
                return self.error(error, ErrorType.VIEW_ERROR)

        dfs = [session.get_dataframe_required(v) for v in views]

        # Check column compatibility
        first_cols = set(dfs[0].columns)
        warnings = []
        for i, df in enumerate(dfs[1:], 2):
            other_cols = set(df.columns)
            if first_cols != other_cols:
                missing = first_cols - other_cols
                extra = other_cols - first_cols
                warnings.append(
                    f"View {views[i-1]} has different columns: "
                    f"missing={list(missing)}, extra={list(extra)}"
                )

        try:
            result_df = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            return self.error(
                f"Stack failed: {str(e)}",
                ErrorType.DATA_ERROR,
            )

        finalize = kwargs.get("finalize", False)

        return self.success(
            view_name=output_view,
            df=result_df,
            mode="stack",
            source_views=views,
            rows=len(result_df),
            result_columns=list(result_df.columns),
            total_rows=len(result_df),
            warnings=warnings if warnings else None,
            finalize=finalize if finalize else None,
        )

    # ── Join mode ────────────────────────────────────────────────────────

    def _execute_join(self, session: "Session", kwargs: dict) -> ToolResult:
        # Infer left_view from most recently created view if not provided
        self._inferred_left_view = False
        if not kwargs.get("left_view") and kwargs.get("right_view") and session.views:
            inferred = list(session.views.keys())[-1]
            # Don't infer the same view as right_view
            if inferred == kwargs["right_view"]:
                candidates = [v for v in session.views if v != kwargs["right_view"]]
                inferred = candidates[-1] if candidates else None
            if inferred:
                kwargs["left_view"] = inferred
                self._inferred_left_view = True

        if error := self.validate_required(kwargs, "left_view", "right_view", "output_view"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        left_view = kwargs["left_view"]
        right_view = kwargs["right_view"]
        output_view = kwargs["output_view"]
        on = kwargs.get("on")
        left_on = kwargs.get("left_on")
        right_on = kwargs.get("right_on")
        how = kwargs.get("how", "inner")

        # If only left_on or right_on is given, default the other to the same columns
        if left_on and not right_on:
            right_on = left_on
        elif right_on and not left_on:
            left_on = right_on

        # Validate join type
        if how not in self.ALLOWED_JOINS:
            return self.error(
                f"Invalid join type: '{how}'. Allowed: {self.ALLOWED_JOINS}",
                ErrorType.VALIDATION_ERROR,
            )

        # Resolve views (fall back to data sources if not in session)
        left_df = self._get_df(session, left_view)
        right_df = self._get_df(session, right_view)

        if left_df is None:
            available = list(session.views.keys()) + list(self._source_map.keys())
            return self.error(
                f"View '{left_view}' not found. Available: {available}",
                ErrorType.VIEW_ERROR,
            )
        if right_df is None:
            available = list(session.views.keys()) + list(self._source_map.keys())
            return self.error(
                f"View '{right_view}' not found. Available: {available}",
                ErrorType.VIEW_ERROR,
            )

        # Auto-rename if output view already exists
        original_name = output_view
        renamed = False
        if output_view in session.views:
            output_view = self._generate_unique_name(output_view, session)
            renamed = True

        # Validate join columns specified (check for empty arrays too)
        # Cross join doesn't need keys — it produces every combination of rows
        if how != "cross" and not on and not (left_on and right_on):
            common_cols = sorted(set(left_df.columns) & set(right_df.columns))
            if len(common_cols) == 1:
                # Exactly one common column — auto-join on it with a warning
                on = common_cols
                warnings.append(
                    f"'on' was not specified — auto-joined on common column '{common_cols[0]}'. "
                    f"Specify on= explicitly if a different join key was intended."
                )
            else:
                suggestion = f" Common columns that could be used: {common_cols}" if common_cols else ""
                return self.error(
                    f"Must specify 'on' or both 'left_on' and 'right_on'.{suggestion}",
                    ErrorType.VALIDATION_ERROR,
                )

        # Validate join columns exist
        if on:
            if error := self.validate_columns_exist(left_df, *on):
                return self.error(f"Left view: {error}", ErrorType.COLUMN_ERROR)
            if error := self.validate_columns_exist(right_df, *on):
                return self.error(f"Right view: {error}", ErrorType.COLUMN_ERROR)
        else:
            if error := self.validate_columns_exist(left_df, *left_on):
                return self.error(f"Left view: {error}", ErrorType.COLUMN_ERROR)
            if error := self.validate_columns_exist(right_df, *right_on):
                return self.error(f"Right view: {error}", ErrorType.COLUMN_ERROR)

        # Check for key intersection before joining (for inner joins)
        warnings = []
        if self._inferred_left_view:
            warnings.append(
                f"left_view was not specified — inferred '{left_view}' "
                f"({len(left_df)} rows, columns: {list(left_df.columns)}). "
                f"If this is not the intended left view, re-run with left_view='...' explicitly."
            )
        if renamed:
            warnings.append(f"View '{original_name}' already exists, renamed to '{output_view}'")

        if how == "inner":
            intersection_info = self._check_key_intersection(
                left_df, right_df, on, left_on, right_on
            )
            if intersection_info["intersection_count"] == 0:
                return self.error(
                    f"No matching keys between views. "
                    f"Left has {intersection_info['left_unique']} unique keys, "
                    f"right has {intersection_info['right_unique']} unique keys, "
                    f"but no intersection. Check that join columns have matching values.",
                    ErrorType.DATA_ERROR,
                )

        # Perform join - use view names as suffixes for colliding columns
        suffixes = (f"_{left_view}", f"_{right_view}")
        try:
            if how == "cross":
                result_df = left_df.merge(right_df, how="cross", suffixes=suffixes)
            elif on:
                result_df = left_df.merge(right_df, on=on, how=how, suffixes=suffixes)
            else:
                result_df = left_df.merge(
                    right_df, left_on=left_on, right_on=right_on, how=how, suffixes=suffixes,
                )
        except Exception as e:
            return self.error(
                f"Join failed: {str(e)}",
                ErrorType.DATA_ERROR,
            )

        # Check for empty result (even for non-inner joins this might be unexpected)
        if result_df.empty and how == "inner":
            return self.error(
                f"Inner join produced no results. Views have no matching keys.",
                ErrorType.DATA_ERROR,
            )

        left_suffix, right_suffix = suffixes

        # Clean up join key columns for left_on/right_on joins
        # Join keys are redundant on the right side (same values by definition)
        dropped_cols = []
        if not on and left_on and right_on:
            for l_col, r_col in zip(left_on, right_on):
                if l_col == r_col:
                    # Same column name: pandas created suffixed copies
                    # Drop right-side copy, rename left-side back to original
                    left_suffixed = f"{l_col}{left_suffix}"
                    right_suffixed = f"{r_col}{right_suffix}"
                    if left_suffixed in result_df.columns and right_suffixed in result_df.columns:
                        result_df = result_df.drop(columns=[right_suffixed])
                        result_df = result_df.rename(columns={left_suffixed: l_col})
                        dropped_cols.append(right_suffixed)
                else:
                    # Different column names: right join column is redundant, drop it
                    if r_col in result_df.columns:
                        result_df = result_df.drop(columns=[r_col])
                        dropped_cols.append(r_col)

        # Detect remaining columns that were renamed due to non-key collisions
        renamed_cols = {}
        for col in result_df.columns:
            if col.endswith(left_suffix):
                base = col[:-len(left_suffix)]
                partner = f"{base}{right_suffix}"
                if partner in result_df.columns:
                    renamed_cols[base] = (col, partner)

        hint = None
        if renamed_cols:
            parts = [f"{base} -> {lc}, {rc}" for base, (lc, rc) in renamed_cols.items()]
            hint = f"Columns renamed (name collision): {'; '.join(parts)}. Use the suffixed names in subsequent operations."

        if dropped_cols:
            drop_msg = f"Dropped redundant right-side join columns: {dropped_cols}"
            if warnings:
                warnings.append(drop_msg)
            else:
                warnings = [drop_msg]

        # Build column_origins: map each output column to (source_view, original_column)
        join_keys = set(on) if on else set()
        column_origins = {}
        for col in result_df.columns:
            if col in join_keys:
                # Join key exists in both views
                column_origins[col] = {"view": left_view, "column": col}
            elif col.endswith(left_suffix):
                base = col[:-len(left_suffix)]
                column_origins[col] = {"view": left_view, "column": base}
            elif col.endswith(right_suffix):
                base = col[:-len(right_suffix)]
                column_origins[col] = {"view": right_view, "column": base}
            elif col in left_df.columns and col not in right_df.columns:
                column_origins[col] = {"view": left_view, "column": col}
            elif col in right_df.columns and col not in left_df.columns:
                column_origins[col] = {"view": right_view, "column": col}
            else:
                # Ambiguous or both — default to left
                column_origins[col] = {"view": left_view, "column": col}

        finalize = kwargs.get("finalize", False)

        return self.success(
            view_name=output_view,
            df=result_df,
            mode="join",
            join_type=how,
            left_view_name=left_view,
            right_view_name=right_view,
            join_on=on,
            join_left_on=left_on,
            join_right_on=right_on,
            left_rows=len(left_df),
            right_rows=len(right_df),
            rows=len(result_df),
            result_columns=list(result_df.columns),
            result_rows=len(result_df),
            column_origins=column_origins,
            hint=hint,
            warnings=warnings if warnings else None,
            finalize=finalize if finalize else None,
        )

    def _generate_unique_name(self, base_name: str, session: "Session") -> str:
        """Generate a unique view name by appending a number."""
        counter = 2
        new_name = f"{base_name}_{counter}"
        while new_name in session.views:
            counter += 1
            new_name = f"{base_name}_{counter}"
        return new_name

    def _check_key_intersection(
        self,
        left_df,
        right_df,
        on: list[str] | None,
        left_on: list[str] | None,
        right_on: list[str] | None,
    ) -> dict:
        """Check the intersection of join keys between two DataFrames."""
        if on:
            left_keys = set(tuple(row) for row in left_df[on].drop_duplicates().values)
            right_keys = set(tuple(row) for row in right_df[on].drop_duplicates().values)
        else:
            left_keys = set(tuple(row) for row in left_df[left_on].drop_duplicates().values)
            right_keys = set(tuple(row) for row in right_df[right_on].drop_duplicates().values)

        intersection = left_keys & right_keys

        return {
            "left_unique": len(left_keys),
            "right_unique": len(right_keys),
            "intersection_count": len(intersection),
        }

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata

        # Stack mode
        if m.get("mode") == "stack":
            views = m.get("source_views", [])
            total = m.get("total_rows", "?")
            msg = f"Stacked {views} \u2192 '{result.view_name}' ({total} rows)"
            parts = [msg]
            if result.warnings:
                parts.append(f"Warnings: {'; '.join(result.warnings)}")
            return " | ".join(parts)

        # Join mode
        how = m.get("join_type", "inner").capitalize()
        left_rows = m.get("left_rows", "?")
        right_rows = m.get("right_rows", "?")
        result_rows = m.get("result_rows", "?")
        left_view = m.get("left_view_name", "?")
        right_view = m.get("right_view_name", "?")
        on = m.get("join_on")
        left_on = m.get("join_left_on")
        right_on = m.get("join_right_on")
        if on:
            key_str = str(on)
        elif left_on and right_on:
            key_str = f"{left_on}={right_on}"
        else:
            key_str = "?"
        msg = f"{how} joined '{left_view}' ({left_rows}) \u00d7 '{right_view}' ({right_rows}) on {key_str} \u2192 '{result.view_name}' ({result_rows} rows)"
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        if m.get("hint"):
            parts.append(f"NOTE: {m['hint']}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        return (
            "Join or stack views. mode='join' (default): join two views on columns "
            "(on, left_on/right_on, how: inner/left/right/outer). "
            "left_view and right_view can be session views OR data source names (e.g., 'companies'). "
            "mode='stack': vertically concatenate multiple views."
        )
