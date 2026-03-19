"""
Search tool for exploring data sources.

Supports:
- Keyword search (case-insensitive, word-boundary aware)
- Exact search (case-sensitive exact match)
- Column restriction
- Result ranking (exact matches first, then partial)
- Configurable substring matching
"""

from typing import TYPE_CHECKING, Any
import re

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session
    from ...data.sources import DataSource


class SearchTool(BaseTool):
    """
    Search for entities in data sources.

    Use this to explore what data is available before creating views.
    Supports keyword and exact matching with ranking.

    Search modes:
    - keyword (default): Case-insensitive, respects word boundaries on left by default.
      "apple" matches "Apple Inc" but NOT "Pineapple Corp".
      "chip" matches "Chips", "chipmaker" but NOT "microchip".
    - exact: Case-sensitive exact match only.

    Substring options (for keyword mode):
    - allow_left_substring: if True, matches like "microchip" for query "chip"
    - allow_right_substring: if True (default), matches like "chips" for query "chip"

    Results are ranked: exact matches > boundary matches > substring matches.

    Examples:
        search(query="apple", source="companies")  # finds "Apple Inc", NOT "Pineapple"
        search(query="chip")  # finds "Chip", "Chips", "chipmaker"
        search(query="chip", allow_left_substring=True)  # also finds "microchip"
        search(query="AAPL", mode="exact")  # exact ticker match only
        search(query="tech", columns=["sector", "industry"])  # restrict columns
    """

    name = "search"
    category = ToolCategory.EXPLORE

    def __init__(self, data_sources: list["DataSource"] | None = None):
        """
        Initialize search tool.

        Args:
            data_sources: List of data sources to search (optional)
        """
        super().__init__()
        self.data_sources = data_sources or []

    def set_data_sources(self, sources: list["DataSource"]) -> None:
        """Set data sources to search."""
        self.data_sources = sources

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                param_type="string",
                description="Search query (entity name, ticker, keyword)",
                required=True,
            ),
            ToolParameter(
                name="source",
                param_type="string",
                description="Specific source to search (optional, default: all)",
                required=False,
            ),
            ToolParameter(
                name="mode",
                param_type="string",
                description="Search mode: 'keyword' (default, case-insensitive) or 'exact' (case-sensitive)",
                required=False,
                default="keyword",
                enum=["keyword", "exact"],
            ),
            ToolParameter(
                name="columns",
                param_type="array",
                description="Specific columns to search (optional, default: all text columns)",
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="allow_left_substring",
                param_type="boolean",
                description="Allow matches where query appears mid-word (e.g., 'chip' in 'microchip'). Default: False",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="allow_right_substring",
                param_type="boolean",
                description="Allow matches where query is a prefix (e.g., 'chip' matches 'chips'). Default: True",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="limit",
                param_type="integer",
                description="Maximum results per source (default: 20)",
                required=False,
                default=20,
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "query"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        query = kwargs["query"]
        source_name = kwargs.get("source")
        mode = kwargs.get("mode", "keyword")
        columns = kwargs.get("columns")
        allow_left_substring = kwargs.get("allow_left_substring", False)
        allow_right_substring = kwargs.get("allow_right_substring", True)
        limit = kwargs.get("limit", 20)

        if not self.data_sources:
            return self.error(
                "No data sources configured for search",
                ErrorType.DATA_ERROR,
            )

        # Validate mode
        if mode not in ("keyword", "exact"):
            return self.error(
                f"Invalid search mode: '{mode}'. Use 'keyword' or 'exact'.",
                ErrorType.VALIDATION_ERROR,
            )

        # Filter sources if specified
        sources = self.data_sources
        if source_name:
            sources = [s for s in sources if s.name == source_name]
            if not sources:
                available = [s.name for s in self.data_sources]
                return self.error(
                    f"Source '{source_name}' not found. Available: {available}",
                    ErrorType.DATA_ERROR,
                )

        # Search each source
        all_results = []
        warnings = []

        for source in sources:
            try:
                source_results = self._search_source(
                    source, query, mode, columns, limit,
                    allow_left_substring, allow_right_substring
                )
                if source_results:
                    all_results.append({
                        "source": source.name,
                        "matches": source_results["matches"],
                        "total_found": source_results["total_found"],
                        "columns_searched": source_results["columns_searched"],
                    })
                    if source_results.get("warnings"):
                        warnings.extend(source_results["warnings"])
            except Exception as e:
                warnings.append(f"Error searching {source.name}: {str(e)}")

        if not all_results:
            return self.error(
                f"No results found for query: '{query}'",
                ErrorType.DATA_ERROR,
            )

        total_matches = sum(r["total_found"] for r in all_results)

        return self.success(
            results=all_results,
            query=query,
            mode=mode,
            total_matches=total_matches,
            sources_searched=len(sources),
            warnings=warnings if warnings else None,
        )

    def _search_source(
        self,
        source: "DataSource",
        query: str,
        mode: str,
        columns: list[str] | None,
        limit: int,
        allow_left_substring: bool,
        allow_right_substring: bool,
    ) -> dict[str, Any] | None:
        """
        Search a single data source.

        Returns dict with matches, total_found, columns_searched, warnings.
        """
        table = getattr(source, "table", source.name)

        # First, get schema to determine text columns
        try:
            schema_df = source.query(f"SELECT * FROM {table} LIMIT 1")
            if schema_df.empty:
                return None

            all_columns = list(schema_df.columns)

            # Determine which columns to search
            if columns:
                # Validate specified columns exist
                missing = [c for c in columns if c not in all_columns]
                if missing:
                    return {
                        "matches": [],
                        "total_found": 0,
                        "columns_searched": [],
                        "warnings": [f"Columns not found in {source.name}: {missing}"],
                    }
                search_columns = columns
            else:
                # Default: search all text/string columns
                search_columns = [
                    col for col in all_columns
                    if schema_df[col].dtype == "object" or str(schema_df[col].dtype).startswith("str")
                ]
                if not search_columns:
                    return None

        except Exception as e:
            return {
                "matches": [],
                "total_found": 0,
                "columns_searched": [],
                "warnings": [f"Schema detection failed: {str(e)}"],
            }

        # Build and execute search query
        if mode == "exact":
            results = self._exact_search(source, table, query, search_columns, limit)
        else:
            results = self._keyword_search(
                source, table, query, search_columns, limit,
                allow_left_substring, allow_right_substring
            )

        results["columns_searched"] = search_columns
        return results

    def _keyword_search(
        self,
        source: "DataSource",
        table: str,
        query: str,
        columns: list[str],
        limit: int,
        allow_left_substring: bool,
        allow_right_substring: bool,
    ) -> dict[str, Any]:
        """
        Keyword search with configurable substring matching.

        Results are ranked:
        1. Exact matches (case-insensitive): "chip" matches "chip" or "Chip"
        2. Boundary matches: "chip" matches "Chips" (if allow_right_substring)
        3. Substring matches: "chip" in "microchip" (if allow_left_substring)
        """
        query_lower = query.lower()
        escaped_query = query_lower.replace("'", "''")

        # Build SQL WHERE clause - fetch candidates broadly, filter in Python
        or_conditions = []
        for col in columns:
            or_conditions.append(f"LOWER(CAST({col} AS VARCHAR)) LIKE '%{escaped_query}%'")

        where_clause = " OR ".join(or_conditions)

        # Fetch candidates
        fetch_limit = limit * 5  # Fetch more since we'll filter
        sql = f"""
            SELECT * FROM {table}
            WHERE {where_clause}
            LIMIT {fetch_limit}
        """

        try:
            df = source.query(sql)
        except Exception as e:
            return {
                "matches": [],
                "total_found": 0,
                "warnings": [f"Query failed: {str(e)}"],
            }

        if df.empty:
            return {"matches": [], "total_found": 0}

        # Filter and rank results based on substring options
        ranked = self._filter_and_rank_results(
            df, columns, query_lower,
            allow_left_substring, allow_right_substring
        )

        # Limit results
        top_results = ranked[:limit]

        return {
            "matches": top_results,
            "total_found": len(ranked),
        }

    def _exact_search(
        self,
        source: "DataSource",
        table: str,
        query: str,
        columns: list[str],
        limit: int,
    ) -> dict[str, Any]:
        """Exact search: case-sensitive exact match only."""
        escaped_query = query.replace("'", "''")

        # Build WHERE clause for exact match
        or_conditions = []
        for col in columns:
            or_conditions.append(f"CAST({col} AS VARCHAR) = '{escaped_query}'")

        where_clause = " OR ".join(or_conditions)

        sql = f"""
            SELECT * FROM {table}
            WHERE {where_clause}
            LIMIT {limit}
        """

        try:
            df = source.query(sql)
        except Exception as e:
            return {
                "matches": [],
                "total_found": 0,
                "warnings": [f"Query failed: {str(e)}"],
            }

        if df.empty:
            return {"matches": [], "total_found": 0}

        matches = self._serialize_records(df.head(limit))

        return {
            "matches": matches,
            "total_found": len(df),
        }

    def _filter_and_rank_results(
        self,
        df,
        columns: list[str],
        query_lower: str,
        allow_left_substring: bool,
        allow_right_substring: bool,
    ) -> list[dict[str, Any]]:
        """
        Filter and rank search results by match quality.

        Ranking order:
        1. Exact match (case-insensitive): entire field equals query
        2. Boundary match: query at word boundary, optionally followed by more chars
        3. Substring match: query anywhere (only if allow_left_substring)
        """
        exact_matches = []
        boundary_matches = []
        substring_matches = []

        # Build regex patterns
        # Boundary pattern: start of string or after non-alphanumeric
        if allow_right_substring:
            # Allow any chars after the query
            boundary_pattern = re.compile(
                r'(^|[^a-zA-Z0-9])' + re.escape(query_lower),
                re.IGNORECASE
            )
        else:
            # Must match at boundary and end at boundary
            boundary_pattern = re.compile(
                r'(^|[^a-zA-Z0-9])' + re.escape(query_lower) + r'([^a-zA-Z0-9]|$)',
                re.IGNORECASE
            )

        for _, row in df.iterrows():
            record = self._serialize_row(row)
            match_type = None

            for col in columns:
                val = str(row.get(col, "")).lower()

                if val == query_lower:
                    # Exact match (entire field)
                    match_type = 1
                    break
                elif boundary_pattern.search(val):
                    # Boundary match
                    if match_type is None or match_type > 2:
                        match_type = 2
                elif allow_left_substring and query_lower in val:
                    # Substring match (only if allowed)
                    if match_type is None:
                        match_type = 3

            # Only include if we found a valid match
            if match_type == 1:
                exact_matches.append(record)
            elif match_type == 2:
                boundary_matches.append(record)
            elif match_type == 3:
                substring_matches.append(record)
            # If match_type is None, the row is filtered out

        # Combine in rank order
        return exact_matches + boundary_matches + substring_matches

    def _serialize_row(self, row) -> dict[str, Any]:
        """Serialize a DataFrame row to a JSON-safe dict."""
        record = {}
        for col in row.index:
            val = row[col]
            if hasattr(val, 'isoformat'):
                record[col] = val.isoformat()
            elif hasattr(val, 'item'):
                record[col] = val.item()
            else:
                record[col] = val
        return record

    def _serialize_records(self, df) -> list[dict[str, Any]]:
        """Serialize DataFrame to list of JSON-safe dicts."""
        return [self._serialize_row(row) for _, row in df.iterrows()]

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata
        query = m.get("query", "?")
        total = m.get("total_matches", 0)
        sources = m.get("sources_searched", "?")
        mode = m.get("mode", "keyword")
        match_word = "match" if total == 1 else "matches"
        msg = f"Searched '{query}' ({mode}): {total} {match_word} across {sources} source(s)"
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        return (
            "Search for entities in data sources. "
            "Modes: 'keyword' (case-insensitive, word-boundary aware) or 'exact'. "
            "By default, 'chip' matches 'Chip', 'Chips' but NOT 'microchip'. "
            "Set allow_left_substring=True to also match 'microchip'."
        )
