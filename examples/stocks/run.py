#!/usr/bin/env python3
"""
Entry point for the stocks example.

Usage:
    python -m examples.stocks.run "Show me AAPL returns for Q1 2024"
    python -m examples.stocks.run -q "Compare FANG stocks last month" -v
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Ensure the repo's parent dir is on sys.path so 'agentic_TRACE' resolves as a package
_repo_parent = str(Path(__file__).resolve().parent.parent.parent.parent)
if _repo_parent not in sys.path:
    sys.path.insert(0, _repo_parent)

from agentic_TRACE.agent.orchestrator import AgentOrchestrator
from agentic_TRACE.config.llm import LLMConfig
from agentic_TRACE.core.session import DomainConfig
from agentic_TRACE.logs.manager import LogManager

from agentic_TRACE.examples.stocks.data_config import get_default_registry as get_source_registry
from agentic_TRACE.examples.stocks.tools_config import create_stock_registry
from agentic_TRACE.examples.stocks.prompt_config import configure_prompts


def sanitize_for_json(obj):
    """Recursively convert all dict keys to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # datetime, Timestamp, etc.
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    else:
        return obj


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stock market analysis agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m examples.stocks.run "Show me AAPL returns for Q1 2024"
    python -m examples.stocks.run -q "Top 5 stocks by volume yesterday" -v
    python -m examples.stocks.run -q "Compare FANG performance ytd" --max-iter 15
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to run",
    )
    parser.add_argument(
        "-q", "--query",
        dest="query_flag",
        help="Query to run (alternative to positional)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Verbose output (-v normal, -vv full untruncated logs)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Maximum iterations (default: 20)",
    )
    parser.add_argument(
        "--db",
        type=str,
        help="Path to DuckDB database",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        help="Save execution log",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./output",
        help="Directory for logs (default: ./output)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip generating LLM summary",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to LLM config YAML (default: uses localhost:8080)",
    )

    args = parser.parse_args()

    # Get query
    query = args.query or args.query_flag
    if not query:
        parser.print_help()
        print("\nError: Query is required")
        sys.exit(1)

    # Setup data sources
    source_registry = get_source_registry(args.db)
    sources = list(source_registry)

    # Setup tools
    tool_registry = create_stock_registry(sources)

    # Setup prompts
    prompt_builder = configure_prompts()

    # Setup logging
    log_manager = None
    if args.save_log:
        log_manager = LogManager(args.log_dir)

    # Load LLM config if specified
    llm_config = None
    if args.config:
        llm_config = LLMConfig.from_yaml(args.config)
        if args.verbose:
            print(f"LLM config: {llm_config.name} (provider={llm_config.provider})")

    # Stock-specific domain config for lineage labels and partition detection
    stock_domain = DomainConfig(
        entity_columns=["ticker", "Ticker", "TICKER", "symbol", "Symbol"],
        entity_label="Tickers",
        time_columns=["date", "Date", "DATE", "timestamp", "Timestamp"],
        time_label="Date range",
        extra_detail_keys=["tickers", "group"],
        partition_candidates=["ticker", "symbol"],
        partition_label="per-stock",
        partition_example="['ticker']",
    )

    # Create orchestrator
    orchestrator = AgentOrchestrator(
        tool_registry=tool_registry,
        source_registry=source_registry,
        prompt_builder=prompt_builder,
        log_manager=log_manager,
        llm_config=llm_config,
        max_iterations=args.max_iter,
        verbose=args.verbose,
        domain_config=stock_domain,
    )

    # Determine log path early so we can display it
    log_dir = Path(args.log_dir) if args.save_log else Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    if args.verbose:
        print(f"Data sources: {source_registry.list_names()}")
        print(f"Tools: {tool_registry.list_names()}")
        print(f"Log directory: {log_dir}")

    # Run query
    result = orchestrator.run(
        query=query,
        generate_summary=not args.no_summary,
    )

    # Output results
    print(f"\nStatus: {result.status}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Tokens: {result.total_tokens_in} in / {result.total_tokens_out} out")

    if result.views:
        print(f"\n{'='*50}")
        print("Results:")
        for view in result.views:
            print(f"\n--- {view.name} ({len(view.dataframe)} rows) ---")
            print(view.dataframe.to_string(max_rows=20))
    else:
        print("\nNo output views generated.")

    if result.final_message:
        print(f"\n{'='*50}")
        print("LLM message:")
        print(result.final_message)

    if result.summary:
        print(f"\n{'='*50}")
        print("Summary:")
        print(result.summary)

    if args.verbose:
        print(f"\n{'='*50}")
        print("Execution trace:")
        print(result.describe_execution())

    if result.error_message:
        print(f"\nError: {result.error_message}")
        sys.exit(1)

    # Auto-save results as CSV if successful
    if result.status == "success" and result.views:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        # Sanitize query for filename (first 30 chars, alphanumeric only)
        query_snippet = re.sub(r'[^a-zA-Z0-9]', '_', query[:30]).strip('_')
        csv_filename = f"results_{timestamp}_{query_snippet}.csv"

        df = result.views[0].dataframe
        csv_path = results_dir / csv_filename
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

    # Save log if requested or if verbose mode (matching original behavior)
    if args.save_log or args.verbose:

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_filename = f"session_{timestamp}_{result.session_id}.json"
        log_path = log_dir / log_filename

        # Build log data matching original format
        log_data = result.execution_stack.to_dict()
        log_data["status"] = result.status
        log_data["llm_message"] = result.final_message
        log_data["summary"] = result.summary
        log_data["total_tokens_in"] = result.total_tokens_in
        log_data["total_tokens_out"] = result.total_tokens_out
        log_data["duration_seconds"] = result.duration_seconds

        # Add data preview from final view
        if result.views:
            df = result.views[0].dataframe
            log_data["data_preview"] = {
                "rows_shown": min(100, len(df)),
                "total_rows": len(df),
                "data": df.head(100).to_dict(orient="records"),
            }

        # Sanitize for JSON (convert Timestamp keys to strings, etc.)
        log_data = sanitize_for_json(log_data)

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
