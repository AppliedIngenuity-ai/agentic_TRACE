"""
Integration tests — full LLM-powered queries against the stock database.

These test the exact queries that exposed bugs in the 2026-03-16 session:
- Housing stock query: explore routing, LIKE/OR, aggregate defaults, finalization
- Simple AAPL query: smoke test for basic single-ticker flow
- Semiconductor chart: README showcase query

Requires: LLM API key + market.duckdb database.
Skipped automatically if either is unavailable.

Run:
    pytest tests/test_integration.py -v -s
"""

import pytest

from tests.conftest import assert_successful_run


# Mark all tests in this module as integration (slow, needs API)
pytestmark = pytest.mark.integration


class TestHousingStockQuery:
    """The query that exposed most bugs on 2026-03-16.

    Previously caused: explore routing failure, LIKE/OR syntax errors,
    aggregate defaulting to wrong column, float equality failures,
    40+ iteration loops, LLM not stopping after finalization.
    """

    QUERY = (
        "when was the high and at what value as well as what was the percent change "
        "on that date from the start for each housing stock in 2025"
    )

    def test_completes_successfully(self, stock_orchestrator):
        result = stock_orchestrator.run(query=self.QUERY, generate_summary=True)
        # Allow some errors since the LLM may need to retry (e.g., missing column)
        # but should ultimately succeed
        assert_successful_run(result, max_iterations=25, max_errors=3)

    def test_output_has_expected_columns(self, stock_orchestrator):
        result = stock_orchestrator.run(query=self.QUERY, generate_summary=False)
        assert result.has_output()
        df = result.get_primary_dataframe()
        assert df is not None
        # Should have ticker and date at minimum
        cols_lower = [c.lower() for c in df.columns]
        assert "ticker" in cols_lower, f"Missing 'ticker' column. Got: {list(df.columns)}"
        assert "date" in cols_lower, f"Missing 'date' column. Got: {list(df.columns)}"

    def test_finds_homebuilder_tickers(self, stock_orchestrator):
        result = stock_orchestrator.run(query=self.QUERY, generate_summary=False)
        if not result.has_output():
            pytest.skip("No output — cannot verify tickers")
        df = result.get_primary_dataframe()
        tickers = set(df["ticker"].unique()) if "ticker" in df.columns else set()
        homebuilders = {"DHI", "LEN", "TOL", "PHM", "NVR"}
        # Should find at least some homebuilder tickers
        overlap = tickers & homebuilders
        assert len(overlap) >= 2, (
            f"Expected homebuilder tickers, got: {tickers}. "
            f"Overlap with {homebuilders}: {overlap}"
        )


class TestSimpleQuery:
    """Basic smoke test — single ticker, simple question."""

    def test_aapl_returns(self, stock_orchestrator):
        result = stock_orchestrator.run(
            query="Show me AAPL returns for Q1 2024",
            generate_summary=False,
        )
        assert_successful_run(result, max_iterations=15, max_errors=1)
        df = result.get_primary_dataframe()
        assert df is not None
        if "ticker" in df.columns:
            assert "AAPL" in df["ticker"].values, f"AAPL not in result: {df['ticker'].unique()}"


class TestSemiconductorNormalized:
    """The README showcase query."""

    QUERY = "Plot the semiconductor stocks, normalized to 1, for the last 90 trading days"

    def test_completes_with_output(self, stock_orchestrator):
        result = stock_orchestrator.run(query=self.QUERY, generate_summary=False)
        assert_successful_run(result, max_iterations=20, max_errors=2)


class TestMissingGroup:
    """Query for a group that doesn't exist — should not spin forever."""

    def test_nonexistent_group_doesnt_loop(self, stock_orchestrator):
        result = stock_orchestrator.run(
            query="Show me the FOOBAR stocks for 2025",
            generate_summary=False,
        )
        # May not succeed (group doesn't exist), but should NOT hit max_iterations
        steps = result.execution_stack.steps if result.execution_stack else []
        assert len(steps) < 20, (
            f"Query for nonexistent group used {len(steps)} steps — "
            f"should have stopped earlier"
        )


class TestAggregateColumnRequired:
    """Verify that aggregate without column= gives a useful error, not wrong defaults.

    This is a direct tool call test (no LLM), validating the fix from today's session.
    """

    def test_missing_column_with_hint(self, session, source_registry, tool_registry):
        """aggregate(function='max', output_column='high_val') should error with suggestion."""
        cv = tool_registry.get("create_view")
        agg = tool_registry.get("aggregate")

        # Create a view with known columns
        result = cv.execute(
            session,
            view_name="test_view",
            source="stock_prices",
            tickers="AAPL",
            where="date >= '2025-01-01' and date <= '2025-01-31'",
        )
        assert result.success, f"create_view failed: {result.error}"
        session.add_view("test_view", result.dataframe, result.metadata)

        # Call aggregate without column= but with output_column that hints at 'high'
        result = agg.execute(
            session,
            function="max",
            source_view="test_view",
            output_column="high_val",
            partition_by=["ticker"],
        )
        assert not result.success, "Expected error when column is missing"
        assert "column" in result.error.lower(), f"Error should mention column: {result.error}"
        assert "high" in result.error.lower(), (
            f"Error should suggest 'high' based on output_column='high_val': {result.error}"
        )

    def test_missing_column_explicit_works(self, session, source_registry, tool_registry):
        """aggregate(function='max', column='high', ...) should succeed."""
        cv = tool_registry.get("create_view")
        agg = tool_registry.get("aggregate")

        result = cv.execute(
            session,
            view_name="test_view2",
            source="stock_prices",
            tickers="AAPL",
            where="date >= '2025-01-01' and date <= '2025-01-31'",
        )
        assert result.success
        session.add_view("test_view2", result.dataframe, result.metadata)

        result = agg.execute(
            session,
            function="max",
            column="high",
            source_view="test_view2",
            output_column="high_max",
            partition_by=["ticker"],
        )
        assert result.success, f"aggregate with explicit column failed: {result.error}"


class TestAggregateToolSchema:
    """Verify the aggregate tool schema enforces column= as required.

    These tests protect against regressions — the column= requirement
    was added because small LLMs (Gemini Flash Lite) ignored description
    text and only followed the JSON schema required array.
    """

    def test_column_is_required_in_schema(self, tool_registry):
        """column must be in the required array of the aggregate tool schema."""
        schemas = tool_registry.to_openai_functions()
        agg_schema = next(s for s in schemas if s['function']['name'] == 'aggregate')
        required = agg_schema['function']['parameters']['required']
        assert 'column' in required, (
            f"'column' must be in required params. Got: {required}"
        )

    def test_function_description_mentions_column(self, tool_registry):
        """function= param description must mention column= requirement."""
        schemas = tool_registry.to_openai_functions()
        agg_schema = next(s for s in schemas if s['function']['name'] == 'aggregate')
        func_desc = agg_schema['function']['parameters']['properties']['function']['description']
        assert 'column=' in func_desc, (
            f"function= description must mention column=. Got: {func_desc[:100]}"
        )

    def test_tool_description_no_window_jargon(self, tool_registry):
        """Tool description should not use 'window function' jargon (LLMs don't understand it)."""
        schemas = tool_registry.to_openai_functions()
        agg_schema = next(s for s in schemas if s['function']['name'] == 'aggregate')
        desc = agg_schema['function']['description']
        # Check parameter descriptions too
        props = agg_schema['function']['parameters']['properties']
        all_descs = [desc] + [p['description'] for p in props.values()]
        for d in all_descs:
            assert 'window function' not in d.lower(), (
                f"Found 'window function' jargon in LLM-facing text: {d[:100]}"
            )


class TestAggregateModes:
    """Test aggregate in both modes with various column= values."""

    def _make_view(self, session, tool_registry):
        cv = tool_registry.get("create_view")
        result = cv.execute(
            session,
            view_name="test_data",
            source="stock_prices",
            tickers="AAPL",
            where="date >= '2025-01-01' and date <= '2025-01-31'",
        )
        assert result.success
        session.add_view("test_data", result.dataframe, result.metadata)

    def test_function_mode_with_column(self, session, tool_registry):
        """function= with column= should succeed."""
        self._make_view(session, tool_registry)
        agg = tool_registry.get("aggregate")
        result = agg.execute(
            session,
            function="first",
            column="close",
            source_view="test_data",
            output_column="start_price",
            sort_by="date",
            partition_by=["ticker"],
        )
        assert result.success, f"function mode with column failed: {result.error}"

    def test_function_mode_without_column_errors(self, session, tool_registry):
        """function= without column= should error."""
        self._make_view(session, tool_registry)
        agg = tool_registry.get("aggregate")
        result = agg.execute(
            session,
            function="first",
            source_view="test_data",
            output_column="start_price",
            sort_by="date",
            partition_by=["ticker"],
        )
        assert not result.success
        assert "column" in result.error.lower()

    def test_function_mode_with_empty_column_errors(self, session, tool_registry):
        """function= with column='' should error (same as missing)."""
        self._make_view(session, tool_registry)
        agg = tool_registry.get("aggregate")
        result = agg.execute(
            session,
            function="first",
            column="",
            source_view="test_data",
            output_column="start_price",
            sort_by="date",
            partition_by=["ticker"],
        )
        assert not result.success
        assert "column" in result.error.lower()

    def test_group_mode_with_column_ignored(self, session, tool_registry):
        """aggregations= with column= should succeed (column is ignored in group mode)."""
        self._make_view(session, tool_registry)
        agg = tool_registry.get("aggregate")
        result = agg.execute(
            session,
            aggregations={"close": "mean"},
            column="close",
            source_view="test_data",
            group_by=["ticker"],
        )
        assert result.success, f"group mode with column failed: {result.error}"

    def test_group_mode_with_empty_column(self, session, tool_registry):
        """aggregations= with column='' should succeed."""
        self._make_view(session, tool_registry)
        agg = tool_registry.get("aggregate")
        result = agg.execute(
            session,
            aggregations={"close": "mean"},
            column="",
            source_view="test_data",
            group_by=["ticker"],
        )
        assert result.success, f"group mode with empty column failed: {result.error}"

    def test_group_mode_without_column(self, session, tool_registry):
        """aggregations= without column= should succeed."""
        self._make_view(session, tool_registry)
        agg = tool_registry.get("aggregate")
        result = agg.execute(
            session,
            aggregations={"close": "mean"},
            source_view="test_data",
            group_by=["ticker"],
        )
        assert result.success, f"group mode without column failed: {result.error}"


class TestTypeNormalization:
    """Verify that DuckDB int columns come back as float64."""

    def test_volume_is_float64(self, source_registry):
        """DuckDB BIGINT (volume) should be normalized to float64."""
        import pandas as pd

        source = source_registry.get("stock_prices")
        df = source.query("SELECT volume FROM stock_prices LIMIT 5")
        assert pd.api.types.is_float_dtype(df["volume"].dtype), (
            f"volume should be float64, got {df['volume'].dtype}"
        )

    def test_all_numeric_columns_float64(self, source_registry):
        """All numeric columns from stock_prices should be float64."""
        import pandas as pd

        source = source_registry.get("stock_prices")
        df = source.query("SELECT * FROM stock_prices LIMIT 1")
        for col in ["open", "high", "low", "close", "volume", "adj_close"]:
            if col in df.columns:
                assert pd.api.types.is_float_dtype(df[col].dtype), (
                    f"{col} should be float64, got {df[col].dtype}"
                )


class TestSQLConversions:
    """Test SQL syntax conversions in where clauses — the exact patterns that failed."""

    def test_like_with_or(self, session, tool_registry):
        """The exact failing pattern from session 20260316_145424."""
        cv = tool_registry.get("create_view")

        # First create a companies view
        result = cv.execute(session, view_name="companies", source="companies")
        assert result.success
        session.add_view("companies", result.dataframe, result.metadata)

        # Now filter with LIKE + OR (previously failed)
        result = cv.execute(
            session,
            view_name="filtered",
            source="companies",
            where="industry LIKE '%Technology%' OR industry LIKE '%Software%'",
        )
        assert result.success, f"LIKE + OR failed: {result.error}"
        assert len(result.dataframe) > 0, "LIKE + OR returned empty result"

    def test_sql_and(self, session, tool_registry):
        cv = tool_registry.get("create_view")
        result = cv.execute(
            session,
            view_name="date_range",
            source="stock_prices",
            tickers="AAPL",
            where="date >= '2025-01-01' AND date <= '2025-03-01'",
        )
        assert result.success, f"SQL AND failed: {result.error}"

    def test_column_deferral(self, session, tool_registry):
        """where= can reference columns not in the columns= list."""
        cv = tool_registry.get("create_view")

        # Create base view
        result = cv.execute(
            session,
            view_name="base",
            source="stock_prices",
            tickers="AAPL",
            where="date >= '2025-01-01' and date <= '2025-01-31'",
        )
        assert result.success
        session.add_view("base", result.dataframe, result.metadata)

        # Select only ticker+date but filter on close (not in output columns)
        result = cv.execute(
            session,
            view_name="filtered",
            source="base",
            columns=["ticker", "date"],
            where="close > 200",
        )
        assert result.success, f"Column deferral failed: {result.error}"
        assert list(result.dataframe.columns) == ["ticker", "date"]
