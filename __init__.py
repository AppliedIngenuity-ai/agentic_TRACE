"""
Agentic Data Framework

A generic framework for building LLM-powered data analysis agents.
Provides core abstractions for tools, sessions, data sources, and orchestration.
"""

__version__ = "0.1.0"

from .core.session import Session
from .core.view import ViewMetadata, ColumnStats
from .tools.base import BaseTool, ToolCategory, ToolParameter, ToolDefinition
from .tools.result import ToolResult
from .tools.registry import ToolRegistry, register_tool
from .data.sources import DataSource, SourceRegistry
from .agent.orchestrator import AgentOrchestrator
from .agent.result import AgentResult, ExecutionStack, ExecutionStep
from .config.prompts import PromptBuilder
from .config.llm import LLMConfig, LLMProvider, LLMResponse, get_provider
from .logs.manager import LogManager

__all__ = [
    # Core
    "Session",
    "ViewMetadata",
    "ColumnStats",
    # Tools
    "BaseTool",
    "ToolCategory",
    "ToolParameter",
    "ToolDefinition",
    "ToolResult",
    "ToolRegistry",
    "register_tool",
    # Data
    "DataSource",
    "SourceRegistry",
    # Agent
    "AgentOrchestrator",
    "AgentResult",
    "ExecutionStack",
    "ExecutionStep",
    # Config
    "PromptBuilder",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "get_provider",
    # Logs
    "LogManager",
]
