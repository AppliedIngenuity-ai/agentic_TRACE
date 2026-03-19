"""
Shared fixtures for agentic_TRACE tests.

Unit tests use tool instances + Session directly.
Integration tests use a full orchestrator with LLM config.
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from agentic_TRACE.core.session import Session, DomainConfig
from agentic_TRACE.agent.orchestrator import AgentOrchestrator
from agentic_TRACE.config.llm import LLMConfig

from agentic_TRACE.examples.stocks.data_config import get_default_registry as get_source_registry
from agentic_TRACE.examples.stocks.tools_config import create_stock_registry
from agentic_TRACE.examples.stocks.prompt_config import configure_prompts
from agentic_TRACE.examples.stocks.hooks import pre_process, PostFinalizeValidator


# ── Paths ──────────────────────────────────────────────────────────────

STOCKS_DIR = Path(__file__).parent.parent / "examples" / "stocks"
DB_PATH = STOCKS_DIR / "data" / "market.duckdb"
CONFIGS_DIR = STOCKS_DIR / "configs"

# Default integration test config — use gemini flash lite (same model that exposed bugs)
DEFAULT_CONFIG = CONFIGS_DIR / "gemini_3_1_flash_lite.yaml"


# ── Unit test fixtures ─────────────────────────────────────────────────

@pytest.fixture
def session():
    """Fresh session for each test."""
    return Session()


@pytest.fixture
def source_registry():
    """Stock data sources (requires market.duckdb)."""
    if not DB_PATH.exists():
        pytest.skip(f"Database not found: {DB_PATH}")
    return get_source_registry(str(DB_PATH))


@pytest.fixture
def sources(source_registry):
    """List of DataSource instances."""
    return list(source_registry)


@pytest.fixture
def tool_registry(sources):
    """Stock tool registry with data sources configured."""
    return create_stock_registry(sources)


@pytest.fixture
def create_view_tool(tool_registry):
    return tool_registry.get("create_view")


@pytest.fixture
def aggregate_tool(tool_registry):
    return tool_registry.get("aggregate")


@pytest.fixture
def join_tool(tool_registry):
    return tool_registry.get("join")


@pytest.fixture
def explore_tool(tool_registry):
    return tool_registry.get("explore")


# ── Stock domain config ────────────────────────────────────────────────

STOCK_DOMAIN = DomainConfig(
    entity_columns=["ticker", "Ticker", "TICKER", "symbol", "Symbol"],
    entity_label="Tickers",
    time_columns=["date", "Date", "DATE", "timestamp", "Timestamp"],
    time_label="Date range",
    extra_detail_keys=["tickers", "group"],
    partition_candidates=["ticker", "symbol"],
    partition_label="per-stock",
    partition_example="['ticker']",
)


# ── Integration test fixtures ──────────────────────────────────────────

def _find_config() -> Path | None:
    """Find a usable LLM config for integration tests."""
    # Prefer gemini flash lite (matches the sessions that exposed bugs)
    if DEFAULT_CONFIG.exists():
        return DEFAULT_CONFIG
    # Fall back to any available config
    for config in CONFIGS_DIR.glob("*.yaml"):
        if "example" not in config.name:
            return config
    return None


@pytest.fixture
def stock_orchestrator():
    """Full stock orchestrator for integration tests.

    Skips if no LLM config or database available.
    """
    if not DB_PATH.exists():
        pytest.skip(f"Database not found: {DB_PATH}")

    config_path = _find_config()
    if config_path is None:
        pytest.skip("No LLM config found")

    llm_config = LLMConfig.from_yaml(str(config_path))

    source_registry = get_source_registry(str(DB_PATH))
    sources = list(source_registry)
    tool_registry = create_stock_registry(sources)
    prompt_builder = configure_prompts()

    post_validator = PostFinalizeValidator(
        source_registry=source_registry,
        llm_config=llm_config,
    )

    orchestrator = AgentOrchestrator(
        tool_registry=tool_registry,
        source_registry=source_registry,
        prompt_builder=prompt_builder,
        llm_config=llm_config,
        max_iterations=25,
        verbose=False,
        domain_config=STOCK_DOMAIN,
        pre_process=pre_process,
        post_finalize=post_validator,
    )

    return orchestrator


# ── Assertion helpers ──────────────────────────────────────────────────

def assert_successful_run(result, max_iterations=25, max_tokens=300_000, max_errors=0):
    """Common assertions for integration test results."""
    assert result.status == "success", (
        f"Expected success, got {result.status}: {result.error_message}"
    )
    assert result.has_output(), "No finalized output"
    assert result.total_tokens < max_tokens, (
        f"Token budget exceeded: {result.total_tokens}"
    )
    steps = result.execution_stack.steps if result.execution_stack else []
    assert len(steps) < max_iterations, (
        f"Too many steps: {len(steps)}"
    )
    errors = [s for s in steps if s.error]
    assert len(errors) <= max_errors, (
        f"Unexpected errors ({len(errors)}): {[s.error for s in errors]}"
    )
