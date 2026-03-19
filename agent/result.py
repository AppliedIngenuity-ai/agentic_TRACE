"""
Agent execution result types.

AgentResult contains the complete output from an agent run:
- Finalized views (DataFrames)
- Execution stack (replayable trace)
- LLM messages and summary

Hook result types for pre-process and post-finalize hooks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..core.view import ViewMetadata


# ─── Hook result types ───


@dataclass
class PreProcessResult:
    """
    Result from a pre-process hook.

    Pre-process hooks run before the LLM starts and can inject context
    into the first user message (e.g., relevant recipes, examples).

    Attributes:
        context: Additional context to inject into the user message.
            Will be appended after the original query.
    """

    context: str | None = None


@dataclass
class PostFinalizeResult:
    """
    Result from a post-finalize validation hook.

    Post-finalize hooks run after the LLM calls finalize() and can
    validate the output, warn about issues, or force the LLM to retry.

    Attributes:
        action: What to do with the result.
            - "pass": All good, continue normally
            - "warn": Show warning to user in summary, session still succeeds
            - "reopen": Un-finalize, send warning to LLM, give it more iterations to fix
            - "fail": Mark session as failed with explanation
        message: Human-readable description of the issue (or None if pass)
        details: Optional structured details for logging/debugging
    """

    action: Literal["pass", "warn", "reopen", "fail"] = "pass"
    message: str | None = None
    details: dict[str, Any] | None = None
    summary: str | None = None  # Optional user-facing summary (avoids extra LLM call)


@dataclass
class ExecutionStep:
    """
    Single step in the execution trace.

    Records everything that happened at one step for debugging,
    auditing, and replay.
    """

    step_number: int
    timestamp: datetime

    # LLM interaction
    llm_input: str = ""  # Messages sent to LLM
    llm_output: str = ""  # LLM response text
    llm_reasoning: str = ""  # Any chain-of-thought output

    # Tool execution (if this step involved a tool call)
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    tool_summary: str = ""

    # Metrics
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0       # This step's duration
    elapsed_ms: int = 0        # Time since session start (cumulative)

    # Error info
    error: str | None = None

    def describe(self) -> str:
        """
        Generate human-readable description of this step.

        Returns:
            Formatted step description
        """
        elapsed_str = f" [{self.elapsed_ms / 1000:.1f}s]" if self.elapsed_ms else ""
        if self.tool_name:
            args_str = ", ".join(f"{k}={v!r}" for k, v in (self.tool_args or {}).items())
            result_str = self.tool_summary or str(self.tool_result)[:100]
            return f"Step {self.step_number}{elapsed_str}: {self.tool_name}({args_str}) → {result_str}"

        # Non-tool step (e.g., LLM reasoning)
        output_preview = self.llm_output[:100] + "..." if len(self.llm_output) > 100 else self.llm_output
        return f"Step {self.step_number}{elapsed_str}: {output_preview}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "step_number": self.step_number,
            "timestamp": self.timestamp.isoformat(),
            "llm_input": self.llm_input,
            "llm_output": self.llm_output,
            "llm_reasoning": self.llm_reasoning,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "tool_summary": self.tool_summary,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "duration_ms": self.duration_ms,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionStep":
        """Create from dictionary."""
        return cls(
            step_number=data["step_number"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            llm_input=data.get("llm_input", ""),
            llm_output=data.get("llm_output", ""),
            llm_reasoning=data.get("llm_reasoning", ""),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args"),
            tool_result=data.get("tool_result"),
            tool_summary=data.get("tool_summary", ""),
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
            duration_ms=data.get("duration_ms", 0),
            elapsed_ms=data.get("elapsed_ms", 0),
            error=data.get("error"),
        )


@dataclass
class ExecutionStack:
    """
    Complete execution trace for an agent run.

    Contains all steps taken, enabling:
    - Debugging: See exactly what happened
    - Auditing: Verify agent actions
    - Replay: Re-execute the same steps
    - Description: Generate human-readable summaries
    """

    steps: list[ExecutionStep] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    # Session metadata
    session_id: str = ""
    query: str = ""

    # Verbose debugging: full LLM messages at each iteration
    messages_snapshots: dict[int, list] | None = None

    def add_step(self, step: ExecutionStep) -> None:
        """Add a step to the stack."""
        self.steps.append(step)

    def describe(self) -> str:
        """
        Generate human-readable description of all steps.

        Returns:
            Multi-line description of execution
        """
        if not self.steps:
            return "No steps executed."

        lines = [f"Execution trace ({len(self.steps)} steps):"]
        for step in self.steps:
            lines.append(f"  {step.describe()}")

        return "\n".join(lines)

    def describe_step(self, step_number: int) -> str:
        """
        Get description of a specific step.

        Args:
            step_number: 1-indexed step number

        Returns:
            Step description or error message
        """
        if step_number < 1 or step_number > len(self.steps):
            return f"Invalid step number: {step_number}"

        return self.steps[step_number - 1].describe()

    def get_total_tokens(self) -> tuple[int, int]:
        """Get total (input_tokens, output_tokens) across all steps."""
        tokens_in = sum(s.tokens_in for s in self.steps)
        tokens_out = sum(s.tokens_out for s in self.steps)
        return tokens_in, tokens_out

    def get_total_duration_ms(self) -> int:
        """Get total execution time in milliseconds."""
        return sum(s.duration_ms for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        result = {
            "session_id": self.session_id,
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_steps": len(self.steps),
        }
        if self.messages_snapshots:
            result["messages_snapshots"] = self.messages_snapshots
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionStack":
        """Create from dictionary."""
        stack = cls(
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )
        stack.steps = [ExecutionStep.from_dict(s) for s in data.get("steps", [])]
        return stack


@dataclass
class FinalizedView:
    """
    A view that was marked as final output.

    Contains the actual DataFrame along with its metadata.
    """

    name: str
    dataframe: pd.DataFrame
    metadata: "ViewMetadata"

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata (not the dataframe)."""
        return {
            "name": self.name,
            "rows": len(self.dataframe),
            "columns": list(self.dataframe.columns),
            "metadata": self.metadata.to_dict(),
        }


@dataclass
class AgentResult:
    """
    Complete result from an agent run.

    Contains:
    - views: List of finalized output views
    - execution_stack: Complete execution trace
    - final_message: Last LLM message (not a summary)
    - summary: Optional LLM-generated summary
    - status: Overall result status
    - total_tokens: Token usage
    - duration_seconds: Total execution time
    """

    # Final outputs
    views: list[FinalizedView] = field(default_factory=list)

    # Execution trace
    execution_stack: ExecutionStack = field(default_factory=ExecutionStack)

    # LLM outputs
    final_message: str = ""  # Last LLM message
    summary: str | None = None  # Optional generated summary

    # Status
    status: Literal["success", "error", "max_iterations", "incomplete", "no_output"] = "success"
    error_message: str | None = None

    # Session identification
    session_id: str = ""

    # Metrics
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_tokens_reasoning: int = 0  # Thinking/reasoning tokens (Gemini, OpenAI o-series)
    duration_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens used (in + out)."""
        return self.total_tokens_in + self.total_tokens_out

    def has_output(self) -> bool:
        """Check if there are any finalized views."""
        return len(self.views) > 0

    def get_primary_view(self) -> FinalizedView | None:
        """Get the first finalized view (primary output)."""
        return self.views[0] if self.views else None

    def get_primary_dataframe(self) -> pd.DataFrame | None:
        """Get the DataFrame from the primary view."""
        view = self.get_primary_view()
        return view.dataframe if view else None

    def describe_execution(self) -> str:
        """Get human-readable execution description."""
        return self.execution_stack.describe()

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize result for storage.

        Note: Does not include DataFrames. Use save_views() for that.
        """
        return {
            "session_id": self.session_id,
            "views": [v.to_dict() for v in self.views],
            "execution_stack": self.execution_stack.to_dict(),
            "final_message": self.final_message,
            "summary": self.summary,
            "status": self.status,
            "error_message": self.error_message,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens_reasoning": self.total_tokens_reasoning,
            "duration_seconds": self.duration_seconds,
        }
