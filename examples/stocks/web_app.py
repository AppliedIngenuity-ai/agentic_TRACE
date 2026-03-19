#!/usr/bin/env python3
"""
Web UI for the stocks example.

Usage:
    python -m examples.stocks.web_app
    python -m examples.stocks.web_app --port 8080
    python -m examples.stocks.web_app --config-dir ./my_configs
"""

import argparse
import sys
from pathlib import Path

# Ensure the repo's parent dir is on sys.path so 'agentic_TRACE' resolves as a package
_repo_parent = str(Path(__file__).resolve().parent.parent.parent.parent)
if _repo_parent not in sys.path:
    sys.path.insert(0, _repo_parent)

from agentic_TRACE.agent.orchestrator import AgentOrchestrator
from agentic_TRACE.config.llm import LLMConfig, WebConfig
from agentic_TRACE.core.session import DomainConfig
from agentic_TRACE.web.server import AgentWebServer
from agentic_TRACE.web.stream import StreamOutput, StreamInput

from agentic_TRACE.examples.stocks.data_config import get_default_registry as get_source_registry
from agentic_TRACE.examples.stocks.tools_config import create_stock_registry
from agentic_TRACE.examples.stocks.prompt_config import configure_prompts
from agentic_TRACE.examples.stocks.hooks import pre_process, PostFinalizeValidator


_verbose_level = 0  # Set by main() from CLI args


def create_orchestrator(
    llm_config: LLMConfig | None,
    session_dir: Path,
    stream_output: StreamOutput,
    stream_input: StreamInput,
) -> AgentOrchestrator:
    """
    Factory function to create a stock analysis orchestrator.

    This is called by the web server for each new session.
    """
    # Setup data sources
    source_registry = get_source_registry()
    sources = list(source_registry)

    # Setup tools
    tool_registry = create_stock_registry(sources)

    # Setup prompts
    prompt_builder = configure_prompts()

    # Setup lifecycle hooks
    post_validator = PostFinalizeValidator(llm_config=llm_config)

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
    return AgentOrchestrator(
        tool_registry=tool_registry,
        source_registry=source_registry,
        prompt_builder=prompt_builder,
        llm_config=llm_config,
        session_dir=session_dir,
        stream_output=stream_output,
        stream_input=stream_input,
        max_iterations=50,
        verbose=_verbose_level,
        pre_process=pre_process,
        post_finalize=post_validator,
        domain_config=stock_domain,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stock Analysis Agent - Web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m examples.stocks.web_app
    python -m examples.stocks.web_app --port 8080
    python -m examples.stocks.web_app --host 0.0.0.0  # Allow external connections
    python -m examples.stocks.web_app --ssl-cert cert.pem --ssl-key key.pem  # HTTPS
        """,
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from config or 127.0.0.1, use 0.0.0.0 for external)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on (default: from config or 8000)",
    )
    parser.add_argument(
        "--ssl-cert",
        type=str,
        default=None,
        help="Path to SSL certificate file for HTTPS",
    )
    parser.add_argument(
        "--ssl-key",
        type=str,
        default=None,
        help="Path to SSL private key file for HTTPS",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a single LLM config YAML file (overrides config-dir, only this config is available)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory containing LLM config YAML files",
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Directory for session data",
    )
    parser.add_argument(
        "--web-config",
        type=str,
        default=None,
        help="Path to web_config.yaml",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (truncated logs)",
    )
    parser.add_argument(
        "-vv", "--very-verbose",
        action="store_true",
        help="Full untruncated verbose logs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode",
    )

    args = parser.parse_args()
    # Compute verbose level: 0, 1, or 2
    args.verbose_level = 2 if args.very_verbose else (1 if args.verbose else 0)

    # Determine directories
    example_dir = Path(__file__).parent
    config_dir = Path(args.config_dir) if args.config_dir else example_dir / "configs"
    session_dir = Path(args.session_dir) if args.session_dir else example_dir / "sessions"

    # Load or create web config
    if args.web_config:
        web_config = WebConfig.from_yaml(args.web_config)
    else:
        # Check for web_config.yaml in example dir
        web_config_path = example_dir / "web_config.yaml"
        if web_config_path.exists():
            web_config = WebConfig.from_yaml(web_config_path)
        else:
            # Create default config
            web_config = WebConfig(
                title="Stock Analysis Agent",
                poll_interval_ms=3000,
                session_dir=str(session_dir),
                config_dir=str(config_dir),
            )

    # Ensure config dir exists with at least one config
    config_dir.mkdir(parents=True, exist_ok=True)
    if not list(config_dir.glob("*.yaml")):
        # Create a default config
        default_config_path = config_dir / "default.yaml"
        default_config_path.write_text("""# Default LLM configuration
# Edit this file or add new .yaml files for different configs

name: "Default (Local)"
provider: openai
model: local-model
api_base: http://localhost:8080/v1
temperature: 0.0
max_tokens: 4096
""")
        print(f"Created default config at: {default_config_path}")
        print("Edit this file to configure your LLM provider.")

    # Set verbose level for orchestrator factory
    global _verbose_level
    _verbose_level = args.verbose_level

    # Create and run server
    server = AgentWebServer(
        orchestrator_factory=create_orchestrator,
        web_config=web_config,
        config_dir=config_dir,
        session_dir=session_dir,
    )

    # If --config specified, override discovered configs with just that one
    if args.config:
        config_path = Path(args.config)
        single_config = LLMConfig.from_yaml(config_path)
        config_key = config_path.stem
        server.llm_configs = {config_key: single_config}
        # Clear web_config filter so the single config is visible
        server.web_config.configs = [config_key]
        server.web_config.default_config = config_key
        print(f"Using single config: {single_config.name} ({config_path})")

    # Build SSL context from args if provided
    ssl_context = None
    if args.ssl_cert and args.ssl_key:
        ssl_context = (args.ssl_cert, args.ssl_key)

    server.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        ssl_context=ssl_context,
    )


if __name__ == "__main__":
    main()
