"""
Explore tool - unified search, describe, list, and ask-human.

Combines the functionality of SearchTool, DescribeTool, and AskHumanTool into
a single tool with smart routing based on which parameters are provided.
"""

from typing import TYPE_CHECKING, Any
import re
import unicodedata

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType
from ...utils.stats import compute_column_stats

if TYPE_CHECKING:
    from ...core.session import Session
    from ...data.sources import DataSource


class ExploreTool(BaseTool):
    """
    Explore data sources, views, and ask humans for clarification.

    Behavior depends on which parameters are provided:
    - question: Ask the human user for clarification
    - view: Describe an existing view (schema, stats)
    - source + query: Search inside a specific data source
    - query (no source): Search FOR matching sources/views by name, description, columns
    - (no params): List all data sources and active views
    """

    name = "explore"
    category = ToolCategory.EXPLORE

    def __init__(self, data_sources: list["DataSource"] | None = None):
        super().__init__()
        self.data_sources = data_sources or []
        self._source_map: dict[str, "DataSource"] = {}
        self._human_input_available = True  # Set by orchestrator

    def set_data_sources(self, sources: list["DataSource"]) -> None:
        """Set data sources for search and describe."""
        self.data_sources = sources
        self._source_map = {s.name: s for s in sources}

    def set_human_input_available(self, available: bool) -> None:
        """Set whether human input is available (e.g., web UI vs batch)."""
        self._human_input_available = available

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                param_type="string",
                description=(
                    "Search term (case-insensitive substring match). "
                    "With source: searches data values inside that source. "
                    "Without source: searches all source names, descriptions, columns, and data values."
                ),
                required=False,
            ),
            ToolParameter(
                name="source",
                param_type="string",
                description="Data source name to search inside",
                required=False,
            ),
            ToolParameter(
                name="view",
                param_type="string",
                description="View name to describe (schema and statistics)",
                required=False,
            ),
            ToolParameter(
                name="question",
                param_type="string",
                description="Question to ask the human user for clarification",
                required=False,
            ),
            ToolParameter(
                name="options",
                param_type="array",
                description="Options for human question (user can always type free-text)",
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="mode",
                param_type="string",
                description="Search mode: keyword (default, case-insensitive substring match) or exact (full phrase match)",
                required=False,
                default="keyword",
                enum=["keyword", "exact"],
            ),
            ToolParameter(
                name="columns",
                param_type="array",
                description="Columns to search in or describe",
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="limit",
                param_type="integer",
                description="Max search results (default: 20)",
                required=False,
                default=20,
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        question = kwargs.get("question")
        view = kwargs.get("view")
        source = kwargs.get("source")
        query = kwargs.get("query")

        # Route 1: Ask human — "question" always means ask the user
        if question:
            return self._ask_human(kwargs)

        # Route 2: Describe a view
        if view:
            return self._describe_view(session, view, kwargs.get("columns"))

        # Route 3: Search inside a specific source
        if source and query:
            return self._search_in_source(source, query, kwargs)

        # Route 3b: Browse a source (source given, no query — return rows directly)
        if source:
            return self._browse_source(source, kwargs)

        # Route 4: Search FOR sources/views by keyword
        if query:
            return self._search_for_sources(session, query, kwargs)

        # Route 5: List all sources and views
        return self._list_all(session)

    # ── Route 1: Ask human ──────────────────────────────────────────────

    def _ask_human(self, kwargs: dict) -> ToolResult:
        question = kwargs["question"]
        options = kwargs.get("options", [])

        if not self._human_input_available:
            return ToolResult(
                success=True,
                metadata={
                    "human_unavailable": True,
                    "question": question,
                    "options": options,
                    "message": (
                        "Human input is not available in this mode. "
                        "The question has been noted. Proceed with your best judgment."
                    ),
                },
            )

        return ToolResult(
            success=True,
            metadata={
                "needs_human_input": True,
                "question": question,
                "options": options,
                "context": kwargs.get("context", ""),
            },
        )

    # ── Route 2: Describe view ──────────────────────────────────────────

    def _describe_view(
        self, session: "Session", view_name: str, columns: list[str] | None
    ) -> ToolResult:
        is_data_source = False

        if view_name in session.views:
            df = session.get_dataframe_required(view_name)
        elif view_name in self._source_map:
            is_data_source = True
            source = self._source_map[view_name]
            try:
                df = source.sample(limit=1000)
            except Exception as e:
                return self.error(
                    f"Failed to sample data source '{view_name}': {e}",
                    ErrorType.DATA_ERROR,
                )
        else:
            available_views = list(session.views.keys())
            available_sources = list(self._source_map.keys())
            return self.error(
                f"'{view_name}' not found. Views: {available_views}. "
                f"Data sources: {available_sources}",
                ErrorType.VIEW_ERROR,
            )

        if columns:
            if error := self.validate_columns_exist(df, *columns):
                return self.error(error, ErrorType.COLUMN_ERROR)
            df = df[columns]

        stats = compute_column_stats(df)
        column_info = {name: s.to_dict() for name, s in stats.items()}

        meta = {"rows": len(df), "columns": column_info}
        if is_data_source:
            meta["is_data_source"] = True
            meta["note"] = "Statistics based on sample of up to 1000 rows"

        return self.success(view_name=view_name, **meta)

    # ── Route 3: Search inside a source ─────────────────────────────────

    def _search_in_source(
        self, source_name: str, query: str, kwargs: dict
    ) -> ToolResult:
        if source_name not in self._source_map:
            available = list(self._source_map.keys())
            return self.error(
                f"Source '{source_name}' not found. Available: {available}",
                ErrorType.DATA_ERROR,
            )

        source = self._source_map[source_name]
        mode = kwargs.get("mode", "keyword")
        columns = kwargs.get("columns")
        limit = kwargs.get("limit", 20)

        result = self._search_single_source(source, query, mode, columns, limit)

        if not result or not result.get("matches"):
            return self.error(
                f"No results for '{query}' in {source_name}. "
                f"If the name looks misspelled or unclear, use explore(question='...') to ask the user what they meant.",
                ErrorType.DATA_ERROR,
            )

        return self.success(
            results=[{
                "source": source_name,
                "matches": result["matches"],
                "total_found": result["total_found"],
            }],
            query=query,
            mode=mode,
            total_matches=result["total_found"],
            sources_searched=1,
            warnings=result.get("warnings"),
        )

    # ── Route 3b: Browse a source ────────────────────────────────────────

    def _browse_source(self, source_name: str, kwargs: dict) -> ToolResult:
        """Return rows from a source directly, with optional column filter and limit."""
        if source_name not in self._source_map:
            available = list(self._source_map.keys())
            return self.error(
                f"Source '{source_name}' not found. Available: {available}",
                ErrorType.DATA_ERROR,
            )

        source = self._source_map[source_name]
        columns = kwargs.get("columns")
        limit = kwargs.get("limit", 50)
        table = getattr(source, "table", source_name)

        try:
            if columns:
                col_str = ", ".join(columns)
                df = source.query(f"SELECT {col_str} FROM {table} LIMIT {limit}")
            else:
                df = source.query(f"SELECT * FROM {table} LIMIT {limit}")
        except Exception as e:
            return self.error(f"Browse failed for '{source_name}': {e}", ErrorType.DATA_ERROR)

        if df.empty:
            return self.error(f"Source '{source_name}' returned no rows", ErrorType.DATA_ERROR)

        return self.success(
            results=[{
                "source": source_name,
                "matches": df.to_dict(orient="records"),
                "total_found": len(df),
            }],
            source=source_name,
            columns=list(df.columns),
            rows=len(df),
        )

    # ── Route 4: Search FOR sources/views ───────────────────────────────

    def _search_for_sources(self, session: "Session", query: str, kwargs: dict | None = None) -> ToolResult:
        """Search source names, descriptions, column names, and data content."""
        query_lower = query.lower()
        mode = (kwargs or {}).get("mode", "keyword")
        limit = (kwargs or {}).get("limit", 10)
        matches = []

        # Search data sources
        for source in self.data_sources:
            match_info = self._match_source(source, query_lower, mode, limit)
            if match_info:
                matches.append(match_info)

        # Search active views
        for view_name in session.views:
            df = session.get_dataframe(view_name)
            if df is None:
                continue
            match_info = self._match_view(view_name, df, query_lower)
            if match_info:
                matches.append(match_info)

        if not matches:
            # No matches — list all sources as fallback
            all_sources = [
                {"name": s.name, "description": s.description, "type": "source"}
                for s in self.data_sources
            ]
            all_views = [
                {"name": v, "type": "view", "rows": len(session.get_dataframe(v))}
                for v in session.views
                if session.get_dataframe(v) is not None
            ]
            return self.success(
                query=query,
                matches=[],
                all_sources=all_sources,
                all_views=all_views,
                message=f"No sources/views match '{query}'. Listed all available. If the name looks misspelled or unclear, use explore(question='...') to ask the user what they meant.",
            )

        return self.success(
            query=query,
            matches=matches,
            total_matches=len(matches),
        )

    @staticmethod
    def _query_clean(text: str) -> str:
        """Remove all non-letter/non-digit characters (Unicode-safe) for SQL LIKE safety.

        Keeps Unicode letters (e.g. accented, CJK), digits, and spaces.
        Strips everything else including quotes, %, _, ;, etc.
        """
        return "".join(
            c if unicodedata.category(c)[0] in ("L", "N") or c == " " else " "
            for c in text
        ).strip()

    def _match_source(self, source: "DataSource", query_lower: str, mode: str = "keyword", limit: int = 10) -> dict | None:
        """Check if a data source matches the query by name, description, columns, or data values."""
        name_match = query_lower in source.name.lower()
        desc_match = query_lower in source.description.lower() if source.description else False

        # Check column names
        matching_columns = []
        all_columns = []
        schema_df = None
        schema_error = None
        try:
            table = getattr(source, "table", source.name)
            schema_df = source.query(f"SELECT * FROM {table} LIMIT 1")
            all_columns = list(schema_df.columns)
            matching_columns = [c for c in all_columns if query_lower in c.lower()]
        except Exception as e:
            schema_error = f"Schema detection for '{source.name}' failed: {e}"

        # If no metadata match, search data values in text/string columns (detected by dtype).
        # keyword mode: token-based OR search, ranked by number of tokens matched.
        # exact mode: full-phrase substring match only.
        # _query_clean strips all non-letter/non-digit chars (Unicode-safe) for SQL safety.
        data_matches: list[dict] = []
        search_errors: list[str] = []
        if not name_match and not desc_match and not matching_columns and all_columns and schema_df is not None:
            try:
                table = getattr(source, "table", source.name)
                text_cols = [
                    c for c in all_columns
                    if schema_df[c].dtype == "object" or str(schema_df[c].dtype).startswith("str")
                ]

                if mode == "exact":
                    safe = self._query_clean(query_lower)
                    tokens = [safe] if safe else []
                else:
                    clean = self._query_clean(query_lower)
                    tokens = [t for t in clean.split() if len(t) > 2]
                    if not tokens:
                        safe = self._query_clean(query_lower)
                        tokens = [safe] if safe else []

                if tokens:
                    for col in text_cols[:5]:
                        if mode == "exact":
                            cond = f"LOWER(CAST({col} AS VARCHAR)) LIKE '%{tokens[0]}%'"
                        else:
                            cond = " OR ".join(
                                f"LOWER(CAST({col} AS VARCHAR)) LIKE '%{t}%'" for t in tokens
                            )
                        rows_df = source.query(f"SELECT * FROM {table} WHERE {cond} LIMIT {limit * 2}")
                        if not rows_df.empty:
                            if mode == "keyword" and len(tokens) > 1:
                                col_lower = rows_df[col].astype(str).str.lower()
                                rows_df = rows_df.copy()
                                rows_df["_score"] = sum(
                                    col_lower.str.contains(t, regex=False).astype(int)
                                    for t in tokens
                                )
                                rows_df = rows_df.sort_values("_score", ascending=False).head(limit)
                                rows_df = rows_df.drop(columns=["_score"])
                            else:
                                rows_df = rows_df.head(limit)
                            data_matches.extend(rows_df.to_dict("records"))
                            break
            except Exception as e:
                search_errors.append(f"Data search in '{source.name}' failed: {e}")

        # Collect all errors that may have prevented finding matches
        all_errors = []
        if schema_error:
            all_errors.append(schema_error)
        all_errors.extend(search_errors)

        if not name_match and not desc_match and not matching_columns and not data_matches:
            if all_errors:
                # Return error info so callers can surface it instead of silent "0 matches"
                return {
                    "name": source.name,
                    "type": "source",
                    "description": source.description or "",
                    "columns": all_columns,
                    "match_reasons": [],
                    "search_errors": all_errors,
                }
            return None

        info: dict[str, Any] = {
            "name": source.name,
            "type": "source",
            "description": source.description or "",
            "columns": all_columns,
        }

        reasons = []
        if name_match:
            reasons.append("name")
        if desc_match:
            reasons.append("description")
        if matching_columns:
            reasons.append(f"columns: {matching_columns}")
            # Quick content match count for columns that matched
            try:
                table = getattr(source, "table", source.name)
                for col in matching_columns[:3]:  # Limit to avoid slow queries
                    count_sql = f"SELECT COUNT(*) as cnt FROM {table} WHERE CAST({col} AS VARCHAR) LIKE '%{query_lower}%'"
                    count_df = source.query(count_sql)
                    if not count_df.empty:
                        info[f"{col}_content_matches"] = int(count_df.iloc[0]["cnt"])
            except Exception as e:
                all_errors.append(f"Content count for '{source.name}.{col}' failed: {e}")
        if data_matches:
            reasons.append(f"data values: {len(data_matches)} row(s) match")
            info["matching_rows"] = data_matches

        info["match_reasons"] = reasons
        if all_errors:
            info["search_errors"] = all_errors
        return info

    def _match_view(self, view_name: str, df, query_lower: str) -> dict | None:
        """Check if a view matches by name or column names."""
        name_match = query_lower in view_name.lower()
        matching_columns = [c for c in df.columns if query_lower in c.lower()]

        if not name_match and not matching_columns:
            return None

        info: dict[str, Any] = {
            "name": view_name,
            "type": "view",
            "rows": len(df),
            "columns": list(df.columns),
        }

        reasons = []
        if name_match:
            reasons.append("name")
        if matching_columns:
            reasons.append(f"columns: {matching_columns}")

        info["match_reasons"] = reasons
        return info

    # ── Route 5: List all ───────────────────────────────────────────────

    def _list_all(self, session: "Session") -> ToolResult:
        sources = []
        for source in self.data_sources:
            info: dict[str, Any] = {
                "name": source.name,
                "description": source.description or "",
            }
            try:
                table = getattr(source, "table", source.name)
                schema_df = source.query(f"SELECT * FROM {table} LIMIT 1")
                info["columns"] = list(schema_df.columns)
            except Exception:
                info["columns"] = []
            sources.append(info)

        views = []
        for view_name in session.views:
            df = session.get_dataframe(view_name)
            meta = session.get_view(view_name)
            view_info: dict[str, Any] = {
                "name": view_name,
                "rows": len(df) if df is not None else 0,
                "columns": list(df.columns) if df is not None else [],
            }
            if meta and meta.is_finalized:
                view_info["finalized"] = True
            views.append(view_info)

        return self.success(
            sources=sources,
            views=views,
            total_sources=len(sources),
            total_views=len(views),
        )

    # ── Shared search helper ────────────────────────────────────────────

    def _search_single_source(
        self,
        source: "DataSource",
        query: str,
        mode: str,
        columns: list[str] | None,
        limit: int,
    ) -> dict[str, Any] | None:
        """Search inside a single data source. Reuses SearchTool logic."""
        table = getattr(source, "table", source.name)

        try:
            schema_df = source.query(f"SELECT * FROM {table} LIMIT 1")
            if schema_df.empty:
                return None

            all_columns = list(schema_df.columns)

            if columns:
                missing = [c for c in columns if c not in all_columns]
                if missing:
                    return {"matches": [], "total_found": 0,
                            "warnings": [f"Columns not found: {missing}"]}
                search_columns = columns
            else:
                search_columns = [
                    col for col in all_columns
                    if schema_df[col].dtype == "object"
                    or str(schema_df[col].dtype).startswith("str")
                ]
                if not search_columns:
                    return None

        except Exception as e:
            return {"matches": [], "total_found": 0,
                    "warnings": [f"Schema detection failed: {e}"]}

        query_lower = query.lower()

        if mode == "exact":
            safe = self._query_clean(query_lower)
            or_conds = [f"LOWER(CAST({c} AS VARCHAR)) LIKE '%{safe}%'"
                        for c in search_columns]
            tokens = []  # Not used for ranking in exact mode
        else:
            # Token-based keyword search: split on non-letter/non-digit chars (Unicode-safe)
            clean = self._query_clean(query_lower)
            tokens = [t for t in clean.split() if len(t) > 2]
            if not tokens:
                tokens = [clean] if clean else [self._query_clean(query_lower)]
            or_conds = [
                f"LOWER(CAST({c} AS VARCHAR)) LIKE '%{t}%'"
                for c in search_columns
                for t in tokens
            ]

        sql = f"SELECT * FROM {table} WHERE {' OR '.join(or_conds)} LIMIT {limit * 2}"

        try:
            df = source.query(sql)
        except Exception as e:
            return {"matches": [], "total_found": 0,
                    "warnings": [f"Query failed: {e}"]}

        if df.empty:
            return {"matches": [], "total_found": 0}

        # Rank results by how many tokens match any search column (keyword mode only)
        if mode == "keyword" and len(tokens) > 1:
            def score_row(row):
                text = " ".join(str(row[c]).lower() for c in search_columns if c in row.index)
                return sum(t in text for t in tokens)
            df = df.copy()
            df["_score"] = df.apply(score_row, axis=1)
            df = df.sort_values("_score", ascending=False).head(limit).drop(columns=["_score"])
        else:
            df = df.head(limit)

        matches = []
        for _, row in df.iterrows():
            record = {}
            for col in row.index:
                val = row[col]
                if hasattr(val, 'isoformat'):
                    record[col] = val.isoformat()
                elif hasattr(val, 'item'):
                    record[col] = val.item()
                else:
                    record[col] = val
            matches.append(record)

        return {"matches": matches, "total_found": len(matches)}

    # ── Summarize / description ─────────────────────────────────────────

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata

        # Ask human result
        if m.get("needs_human_input"):
            q = m.get("question", "")
            resp = m.get("human_response", "")
            if resp:
                return f"Asked: '{q}' → Human: '{resp}'"
            return f"Waiting for human: '{q}'"
        if m.get("human_unavailable"):
            return f"Human unavailable — noted: '{m.get('question', '')}'"

        # Describe result
        if "columns" in m and isinstance(m["columns"], dict) and "rows" in m:
            name = result.view_name or "?"
            rows = m.get("rows", "?")
            n_cols = len(m["columns"])
            col_names = list(m["columns"].keys())[:5]
            return f"Described '{name}': {rows} rows, {n_cols} columns {col_names}"

        # Browse source result (source + rows, no query)
        if "source" in m and "rows" in m and "query" not in m:
            src = m["source"]
            rows = m["rows"]
            cols = m.get("columns", [])
            return f"Browsed '{src}': {rows} rows, columns: {cols}"

        # Search results
        if "results" in m and "query" in m:
            q = m["query"]
            results = m.get("results", [])
            total = m.get("total_matches", 0)
            return f"Search '{q}' in {m.get('sources_searched', '?')} source(s): {total} matches"

        if "query" in m and "matches" in m:
            q = m["query"]
            matches = m.get("matches", [])
            if isinstance(matches, list) and matches and isinstance(matches[0], dict) and "name" in matches[0]:
                names = [x["name"] for x in matches]
                return f"Search '{q}': found {names}"
            total = m.get("total_matches", len(matches))
            return f"Search '{q}': {total} matches"

        # List result
        if "sources" in m:
            n_src = m.get("total_sources", 0)
            n_view = m.get("total_views", 0)
            return f"Listed {n_src} sources, {n_view} views"

        return "Explore completed"

    def tool_description(self) -> str:
        return (
            "Explore data: search sources, describe views, list all, or ask the user. "
            "query='chip' finds sources/views with matching names or columns. "
            "source='companies' + query='apple' searches inside that source. "
            "view='my_view' describes its schema. "
            "question='Which ticker?' asks the user."
        )
