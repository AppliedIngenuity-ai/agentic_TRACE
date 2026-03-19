"""
Agent orchestrator - main execution loop.

The AgentOrchestrator coordinates:
- LLM interactions
- Tool execution
- Session state management
- Result collection
"""

import copy
import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..core.session import Session, DomainConfig
from ..tools.registry import ToolRegistry
from ..tools.result import ToolResult
from ..data.sources import SourceRegistry
from ..config.prompts import PromptBuilder
from ..logs.manager import LogManager
from ..web.stream import sanitize_for_json
from .result import AgentResult, ExecutionStack, ExecutionStep, FinalizedView, PreProcessResult, PostFinalizeResult

from ..config.llm import LLMConfig, LLMProvider, LLMResponse, get_provider
from ..web.stream import StreamOutput, StreamInput


class AgentOrchestrator:
    """
    Main agent orchestration loop.

    Coordinates the interaction between:
    - LLM (via pluggable LLMProvider: OpenAI, Anthropic, text-tool-calling, etc.)
    - Tools (registered in ToolRegistry)
    - Session state (views, dataframes, logs)
    - Data sources (via SourceRegistry)

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(SearchTool())
        >>> registry.register(CreateViewTool())
        >>>
        >>> orchestrator = AgentOrchestrator(
        ...     tool_registry=registry,
        ...     source_registry=sources,
        ... )
        >>> result = orchestrator.run("Show me AAPL returns for Q1 2024")
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        source_registry: SourceRegistry | None = None,
        prompt_builder: PromptBuilder | None = None,
        log_manager: LogManager | None = None,
        # LLM configuration (individual params or LLMConfig)
        llm_config: LLMConfig | None = None,
        llm_endpoint: str | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        temperature: float = 0.0,
        # Execution configuration
        max_iterations: int = 10,
        metadata_extractor: Callable | None = None,
        verbose: bool = False,
        # Web streaming (optional)
        stream_output: StreamOutput | None = None,
        stream_input: StreamInput | None = None,
        # Session directory for web mode
        session_dir: Path | str | None = None,
        # Lifecycle hooks
        pre_process: Callable | None = None,
        post_finalize: Callable | None = None,
        # Domain configuration
        domain_config: DomainConfig | None = None,
        # Error handling
        include_tool_docs_on_error: bool = False,
    ):
        """
        Initialize orchestrator.

        Args:
            tool_registry: Registry of available tools
            source_registry: Registry of data sources
            prompt_builder: Builder for system prompts
            log_manager: Manager for logs/artifacts
            llm_config: LLMConfig object (alternative to individual params)
            llm_endpoint: OpenAI-compatible API endpoint
            llm_model: Model name
            llm_api_key: API key
            temperature: LLM temperature
            max_iterations: Maximum tool call iterations
            metadata_extractor: Custom view metadata extractor
            verbose: Print step-by-step execution details
            stream_output: StreamOutput for web UI events
            stream_input: StreamInput for web UI human responses
            session_dir: Directory for session artifacts (web mode)
            pre_process: Hook called before LLM starts. Receives (query, session) and
                returns PreProcessResult with optional context to inject.
            post_finalize: Hook called after finalize(). Receives (session, finalized_views)
                and returns PostFinalizeResult with action (pass/warn/reopen/fail).
        """
        self.tool_registry = tool_registry
        self.verbose = verbose
        # -vv (verbose >= 2) disables log truncation
        self._log_max_chars = 0 if (isinstance(verbose, int) and verbose >= 2) else 10000
        self.source_registry = source_registry

        # Populate global registry so validate_view_exists can auto-load sources by name
        if source_registry is not None:
            from ..data.sources import set_global_source_registry
            set_global_source_registry(source_registry)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.log_manager = log_manager

        # LLM config - prefer LLMConfig object if provided
        if llm_config:
            self.llm_endpoint = llm_config.get_api_base() or os.getenv("LLM_ENDPOINT", "http://localhost:8080/v1")
            self.llm_model = llm_config.model
            self.llm_api_key = llm_config.get_api_key() or os.getenv("LLM_API_KEY", "not-needed")
            self.temperature = llm_config.temperature
            self.llm_config = llm_config
        else:
            self.llm_endpoint = llm_endpoint or os.getenv("LLM_ENDPOINT", "http://localhost:8080/v1")
            self.llm_model = llm_model or os.getenv("LLM_MODEL", "local-model")
            self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY", "not-needed")
            self.temperature = temperature
            self.llm_config = None

        self.max_iterations = max_iterations
        self.metadata_extractor = metadata_extractor
        self.domain_config = domain_config
        self.include_tool_docs_on_error = include_tool_docs_on_error

        # Web streaming
        self.stream_output = stream_output
        self.stream_input = stream_input
        self.session_dir = Path(session_dir) if session_dir else None

        # Lifecycle hooks
        self._pre_process_hook = pre_process
        self._post_finalize_hook = post_finalize

        # LLM provider (lazy init)
        self._provider: LLMProvider | None = None

        # Session log path (for incremental debug logging)
        self._log_path: Path | None = None

        # Stop signal (set via stop() to cancel a running session)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the running session to stop after the current iteration."""
        self._stop_event.set()

    def _init_log_file(self, session_id: str) -> None:
        """Initialize session log path for debugging."""
        if self.session_dir:
            self._log_path = self.session_dir / "debug.log"
            self._log(f"Session started: {session_id}")

    def _log(self, msg: str) -> None:
        """Log message to console (if verbose) and session log file (incremental/atomic)."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] {msg}\n"
        if self.verbose:
            print(line, end="")
        if self._log_path:
            # Append mode with immediate write - survives hangs
            with open(self._log_path, "a") as f:
                f.write(line)

    def _truncate_for_log(self, text: str, max_len: int | None = None) -> str:
        """Truncate text for logging: first 3K + last 7K if over max_len.

        Args:
            max_len: Maximum length before truncating. None or 0 = no truncation.
                     Default is 10000 unless self._log_max_chars is set.
        """
        if max_len is None:
            max_len = getattr(self, "_log_max_chars", 10000)
        if not text or max_len == 0 or len(text) <= max_len:
            return text
        head = text[:3000]
        tail = text[-(max_len - 3000):]
        return f"{head}\n... [{len(text)} chars total, showing first 3K + last {max_len - 3000} chars] ...\n{tail}"

    def _log_llm_request(self, messages: list[dict], iteration: int, tools: list[dict] | None = None) -> None:
        """Log full LLM request — all messages, no truncation."""
        self._log(f"=== LLM Request (iteration {iteration}, {len(messages)} messages) ===")

        # Log tool definitions on first iteration
        if tools and iteration == 1:
            tool_names = [t.get("function", {}).get("name", "?") for t in tools]
            self._log(f"  Tools ({len(tools)}): {tool_names}")
            self._log(f"  Tool definitions:\n{json.dumps(tools, indent=2, default=str)}")

        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "") or ""
            self._log(f"  [{i}] {role} ({len(content)} chars): {content}")
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    self._log(f"      tool_call: {fn.get('name')}({fn.get('arguments', '')})")
            if msg.get("role") == "tool":
                self._log(f"      tool_call_id: {msg.get('tool_call_id')}")

    def _log_llm_response(self, response: LLMResponse, iteration: int) -> None:
        """Log full LLM response — no truncation."""
        try:
            content = response.content or "(no content)"
            self._log(f"=== LLM Response (iteration {iteration}) ===")

            # For text-tool-calling providers, log the raw model output before parsing
            if response.raw:
                raw_msg = getattr(response.raw, "choices", [None])
                if raw_msg and raw_msg[0]:
                    raw_content = getattr(getattr(raw_msg[0], "message", None), "content", None)
                    if raw_content and raw_content != content:
                        self._log(f"  Raw output ({len(raw_content)} chars): {raw_content}")

            self._log(f"  Content ({len(content)} chars): {content}")

            if response.tool_calls:
                self._log(f"  Tool calls: {len(response.tool_calls)}")
                for tc in response.tool_calls:
                    self._log(f"    - {tc.name}({json.dumps(tc.arguments, default=str)})")
            else:
                self._log("  Tool calls: None (LLM stopping)")

            if response.tokens_in or response.tokens_out:
                reasoning_part = f", reasoning={response.tokens_reasoning}" if response.tokens_reasoning else ""
                cache_part = ""
                # Show cache info for Anthropic provider (has cache fields in raw response)
                if response.raw and hasattr(response.raw, "usage") and hasattr(response.raw.usage, "cache_creation_input_tokens"):
                    cache_part = f", cache_create={response.cache_created}, cache_read={response.cache_read}"
                self._log(f"  Tokens: in={response.tokens_in}, out={response.tokens_out}{reasoning_part}{cache_part}")

            if response.raw:
                raw_msg = getattr(response.raw, "choices", [None])
                if raw_msg and raw_msg[0]:
                    msg = getattr(raw_msg[0], "message", None)
                    if msg:
                        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                        if reasoning:
                            self._log(f"  Reasoning ({len(reasoning)} chars): {reasoning}")
                        refusal = getattr(msg, "refusal", None)
                        if refusal:
                            self._log(f"  Refusal: {refusal}")
        except Exception as e:
            self._log(f"  Error logging response: {e}")

    def _get_provider(self) -> LLMProvider:
        """Get or create LLM provider."""
        if self._provider is None:
            if self.llm_config:
                self._provider = get_provider(self.llm_config)
            else:
                # Fallback: create OpenAI provider from individual params
                from ..config.llm import OpenAIProvider
                self._provider = OpenAIProvider(
                    api_base=self.llm_endpoint,
                    api_key=self.llm_api_key,
                )
        return self._provider

    def _emit(self, event_type: str, **data: Any) -> None:
        """Emit a stream event if streaming is enabled."""
        if self.stream_output:
            self.stream_output.emit(event_type, **data)

    # ─── Lifecycle hooks ───

    def _run_pre_process(self, query: str, session: "Session") -> PreProcessResult | None:
        """
        Run pre-process hook before LLM starts.

        Override this method or pass a callable to __init__(pre_process=...).
        Returns PreProcessResult with optional context to inject, or None.
        """
        if self._pre_process_hook:
            try:
                result = self._pre_process_hook(query, session)
                has_context = result and result.context
                self._log(f"[pre_process] {'Injecting context: ' + result.context[:200] + '...' if has_context else 'No context injected'}")
                return result
            except Exception as e:
                tb = traceback.format_exc()
                self._log(f"[error] Pre-process hook failed: {e}")
                print(f"[ERROR] Pre-process hook failed: {e}\n{tb}")
                return None
        return None

    def _run_post_finalize(
        self, session: "Session", finalized_views: list[FinalizedView]
    ) -> PostFinalizeResult | None:
        """
        Run post-finalize validation hook.

        Override this method or pass a callable to __init__(post_finalize=...).
        Returns PostFinalizeResult with action (pass/warn/reopen/fail), or None.
        """
        if self._post_finalize_hook:
            try:
                result = self._post_finalize_hook(session, finalized_views)
                action = result.action if result else "pass"
                message = result.message if result else None
                self._log(f"[post_finalize] Action={action}: {message or 'PASS'}")
                return result
            except Exception as e:
                tb = traceback.format_exc()
                self._log(f"[error] Post-finalize hook failed: {e}")
                print(f"[ERROR] Post-finalize hook failed: {e}\n{tb}")
                return None
        return None

    def run(
        self,
        query: str,
        generate_summary: bool = True,
    ) -> AgentResult:
        """
        Run the agent on a query.

        Args:
            query: User's natural language query
            generate_summary: Whether to generate LLM summary at end

        Returns:
            AgentResult with views, execution stack, and messages
        """
        start_time = time.time()

        # Generate session ID (8 chars like original)
        session_id = str(uuid.uuid4())[:8]

        # Initialize session with metadata extractor
        session = Session(
            metadata_extractor=self.metadata_extractor,
            domain_config=self.domain_config,
        )
        session.session_id = session_id
        session.query = query

        # Initialize execution stack
        execution_stack = ExecutionStack(started_at=datetime.now())
        execution_stack.session_id = session_id
        execution_stack.query = query

        # Messages snapshots for verbose debugging
        messages_snapshots: dict[int, list] = {}

        # Verbose: print session header
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Session: {session_id}")
            print(f"Query: {query}")
            print(f"{'='*60}\n")

        # Initialize debug log file (writes to session_dir/debug.log)
        self._init_log_file(session_id)
        self._log(f"Query: {query}")

        # Reset session flags
        self._stop_event.clear()
        self._ask_why_sent = False
        self._continue_nudge_sent = False
        self._post_finalize_ran = False
        self._reopen_count = 0  # Track post_finalize reopens to prevent infinite loops
        self._last_charted_view = None  # Track which view was charted (for auto-finalize)
        self._chart_counter = 0  # Ensure unique chart filenames within a session
        self._text_clarification_routed = False  # Detect LLM asking question in text
        self._tool_call_counts: dict[str, int] = {}  # Track repeated identical tool calls
        self._loop_warned: set[str] = set()  # Keys already warned to avoid repeat warnings
        self._last_failed_key: str | None = None  # Last failed call key for consecutive-failure detection
        self._consecutive_errors: int = 0  # Count consecutive failed tool calls (any kind)

        # Emit session start event for web UI
        config_name = self.llm_config.name if self.llm_config else None
        self._emit("start", session_id=session_id, query=query, config=config_name)

        # Build system prompt
        system_prompt = self._build_system_prompt()

        # Run pre-process hook to get additional context
        user_message = query
        pre_start = time.time()
        pre_result = self._run_pre_process(query, session)
        if pre_result and pre_result.context:
            user_message = f"{query}\n\n{pre_result.context}"

        # Log pre-process as an execution step
        pre_duration_ms = int((time.time() - pre_start) * 1000)
        pre_elapsed_ms = int((time.time() - start_time) * 1000)
        pre_step = ExecutionStep(
            step_number=len(execution_stack.steps) + 1,
            timestamp=datetime.now(),
            tool_name="pre_process",
            tool_args={"query": query},
            tool_summary="Injected context" if (pre_result and pre_result.context) else "No context injected",
            duration_ms=pre_duration_ms,
            elapsed_ms=pre_elapsed_ms,
        )
        execution_stack.add_step(pre_step)
        self._emit(
            "pre_process",
            context=pre_result.context if (pre_result and pre_result.context) else None,
            duration_ms=pre_duration_ms,
            elapsed_ms=pre_elapsed_ms,
        )

        # Initialize messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Get tool definitions for function calling
        tools = self.tool_registry.to_openai_functions()

        # Filter tools if config specifies a subset
        if self.llm_config and self.llm_config.tools:
            allowed = set(self.llm_config.tools)
            tools = [t for t in tools if t.get("function", {}).get("name") in allowed]

        # Main loop
        final_message = ""
        total_tokens_in = 0
        total_tokens_out = 0
        total_tokens_reasoning = 0
        max_context_tokens = 0  # Track largest prompt sent

        for iteration in range(self.max_iterations):
            # Check for stop signal
            if self._stop_event.is_set():
                self._emit("complete", status="stopped")
                break

            # Verbose: print iteration header
            if self.verbose:
                print(f"--- Iteration {iteration + 1} ---")

            # Emit iteration event for web UI
            self._emit("iteration", n=iteration + 1)

            # Store messages snapshot for debugging
            messages_snapshots[iteration + 1] = copy.deepcopy(messages)
            step_start = time.time()

            # Inject ephemeral context reminder (removed after LLM call)
            context_reminder = self._build_context_reminder(query, messages, iteration + 1, session)
            if context_reminder:
                messages.append({"role": "user", "content": context_reminder})

            # Call LLM
            self._log(f"Calling LLM (iteration {iteration + 1})...")
            self._log_llm_request(messages, iteration + 1, tools=tools)
            try:
                response = self._call_llm(messages, tools)
            except Exception as e:
                tb = traceback.format_exc()
                self._log(f"[error] LLM call failed: {e}")
                self._log(f"[error] Traceback:\n{tb}")
                print(f"[ERROR] LLM call failed: {e}")
                print(f"[ERROR] Traceback:\n{tb}")
                return self._error_result(
                    str(e),
                    execution_stack,
                    session,
                    start_time,
                    total_tokens_in,
                    total_tokens_out,
                    total_tokens_reasoning,
                )

            # Remove ephemeral context reminder (not part of conversation history)
            if context_reminder and messages and messages[-1].get("content") == context_reminder:
                messages.pop()

            # Check stop signal immediately after LLM responds (before executing tools)
            if self._stop_event.is_set():
                self._emit("complete", status="stopped")
                break

            # Log response
            self._log_llm_response(response, iteration + 1)

            # Track tokens
            total_tokens_in += response.tokens_in
            total_tokens_out += response.tokens_out
            total_tokens_reasoning += response.tokens_reasoning
            if response.tokens_in > max_context_tokens:
                max_context_tokens = response.tokens_in

            # Emit per-iteration token event for web UI
            if response.tokens_in or response.tokens_out:
                self._emit(
                    "tokens",
                    iteration=iteration + 1,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    tokens_reasoning=response.tokens_reasoning,
                    total_tokens_in=total_tokens_in,
                    total_tokens_out=total_tokens_out,
                    total_tokens_reasoning=total_tokens_reasoning,
                )

            final_message = response.content or ""

            # Check if we have tool calls
            tool_calls = response.tool_calls if response.tool_calls else None

            if not tool_calls:
                # No tool calls - LLM stopped calling tools
                # Verbose: print LLM response
                if self.verbose:
                    content_preview = final_message[:400] + "..." if len(final_message) > 400 else final_message
                    print(f"  LLM response: {content_preview}")
                    print()

                # Emit LLM message for web UI
                if final_message:
                    self._emit("llm_message", content=final_message)

                step = ExecutionStep(
                    step_number=len(execution_stack.steps) + 1,
                    timestamp=datetime.now(),
                    llm_output=final_message,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    duration_ms=int((time.time() - step_start) * 1000),
                    elapsed_ms=int((time.time() - start_time) * 1000),
                )
                execution_stack.add_step(step)

                # Check completion state
                has_finalized = len(session.finalized) > 0
                has_views = len(session.views) > 0

                # Detect if LLM asked a clarifying question in text instead of using explore(question=...)
                if (not self._text_clarification_routed
                        and final_message
                        and self._looks_like_question(final_message)):
                    self._text_clarification_routed = True
                    self._log("LLM asked question as text - routing to human input")
                    self._emit("warning", message="LLM asked clarifying question as text - routing to human input")
                    fake_result = ToolResult(
                        success=True,
                        metadata={
                            "needs_human_input": True,
                            "question": final_message.strip(),
                            "options": [],
                        }
                    )
                    human_response = self._get_human_input(fake_result)
                    session.record_human_input(question=final_message.strip(), response=human_response)
                    messages.append({"role": "assistant", "content": final_message})
                    messages.append({"role": "user", "content": human_response})
                    continue

                if has_views and not has_finalized:
                    # Views exist but not finalized.
                    # If a chart was produced and LLM stopped, treat as natural completion —
                    # the chart is the final output, no nudge needed.
                    view_names = list(session.views.keys())
                    if self._last_charted_view and self._last_charted_view in view_names:
                        auto_view = self._last_charted_view
                    elif not self._continue_nudge_sent:
                        # No chart yet — nudge the model to continue
                        self._continue_nudge_sent = True
                        view_summaries = []
                        for vn in view_names:
                            vdf = session.get_dataframe(vn)
                            cols = list(vdf.columns) if vdf is not None else []
                            rows = len(vdf) if vdf is not None else 0
                            view_summaries.append(f"'{vn}' ({rows} rows, columns: {cols})")
                        views_str = "; ".join(view_summaries)

                        self._log("Views created but not finalized - nudging to continue")
                        self._emit("warning", message="LLM stopped mid-workflow - nudging to continue")
                        nudge = (
                            f"You created views but did not finalize. Current views: {views_str}. "
                            f"Continue working — call the next tool (e.g., compute_indicator, aggregate, chart). "
                            f"Set finalize=true on your last non-chart tool call when done. "
                            f"Do NOT repeat what you already did."
                        )
                        messages.append({"role": "assistant", "content": final_message})
                        messages.append({"role": "user", "content": nudge})
                        continue
                    else:
                        # Second stop without chart — auto-finalize last view
                        auto_view = view_names[-1]

                    self._log(f"Auto-finalizing view '{auto_view}'")
                    session.finalize_view(auto_view)

                    df = session.get_dataframe(auto_view)
                    row_count = len(df) if df is not None else 0
                    final_message = final_message or f"View '{auto_view}' with {row_count} rows."

                    # Run post-finalize validation
                    if not self._post_finalize_ran:
                        self._post_finalize_ran = True
                        post_start = time.time()
                        auto_finalized_views = [
                            FinalizedView(name=name, dataframe=fdf, metadata=meta)
                            for name, fdf, meta in session.get_finalized_views()
                        ]
                        post_result = self._run_post_finalize(session, auto_finalized_views)
                        post_action = post_result.action if post_result else "pass"
                        post_message = post_result.message if post_result else None
                        post_duration_ms = int((time.time() - post_start) * 1000)
                        post_elapsed_ms = int((time.time() - start_time) * 1000)
                        post_step = ExecutionStep(
                            step_number=len(execution_stack.steps) + 1,
                            timestamp=datetime.now(),
                            tool_name="post_finalize",
                            tool_args={"views": [v.name for v in auto_finalized_views]},
                            tool_summary=f"{post_action}: {post_message}" if post_message else post_action,
                            duration_ms=post_duration_ms,
                            elapsed_ms=post_elapsed_ms,
                            error=post_message if post_action in ("warn", "reopen", "fail") else None,
                        )
                        execution_stack.add_step(post_step)
                        self._emit(
                            "post_finalize", action=post_action, message=post_message,
                            duration_ms=post_duration_ms, elapsed_ms=post_elapsed_ms,
                        )
                        if post_result and post_result.action == "reopen":
                            self._reopen_count += 1
                            self._post_finalize_ran = False
                            session.unfinalize_all()
                            self._log(f"[post_finalize] Reopening (#{self._reopen_count}): {post_result.message}")
                            self._emit("warning", message=f"Validation issue: {post_result.message}")
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"VALIDATION WARNING: {post_result.message} "
                                    f"Please review and fix the issue, then finalize again."
                                ),
                            })
                            continue
                        elif post_result and post_result.action == "warn":
                            self._emit("warning", message=f"Validation warning: {post_result.message}")
                        elif post_result and post_result.action == "fail":
                            self._emit("error", message=f"Validation failed: {post_result.message}")

                    break

                elif not has_views and not has_finalized and iteration > 0 and not self._ask_why_sent:
                    # No views created at all — ask the LLM why
                    self._ask_why_sent = True
                    self._log("No views created - asking LLM why")
                    self._emit("warning", message="LLM stopped without creating views - asking why")
                    ask_why = (
                        "You stopped without creating any views. "
                        "If you need to ask the user for clarification, use explore(question='...', options=[...]) — do not ask in text. "
                        "If a tool call failed, try a different approach. "
                        "Reply with REASON: <explanation> ONLY if this query truly cannot be answered."
                    )
                    messages.append({"role": "assistant", "content": final_message})
                    messages.append({"role": "user", "content": ask_why})
                    continue

                else:
                    # Either: has_finalized (success), first iteration with no action,
                    # or already asked why (LLM replied with REASON: or text)
                    if self._ask_why_sent and final_message:
                        # LLM responded to ask-why — check for REASON: prefix
                        if "REASON:" in final_message.upper():
                            self._log(f"LLM provided reason: {final_message[:200]}")

                    # Run post-finalize when LLM stops with finalized views
                    if has_finalized and not self._post_finalize_ran:
                        self._post_finalize_ran = True
                        post_start = time.time()
                        pf_views = [
                            FinalizedView(name=name, dataframe=fdf, metadata=meta)
                            for name, fdf, meta in session.get_finalized_views()
                        ]
                        post_result = self._run_post_finalize(session, pf_views)
                        post_action = post_result.action if post_result else "pass"
                        post_message = post_result.message if post_result else None
                        post_duration_ms = int((time.time() - post_start) * 1000)
                        post_elapsed_ms = int((time.time() - start_time) * 1000)
                        post_step = ExecutionStep(
                            step_number=len(execution_stack.steps) + 1,
                            timestamp=datetime.now(),
                            tool_name="post_finalize",
                            tool_args={"views": [v.name for v in pf_views]},
                            tool_summary=f"{post_action}: {post_message}" if post_message else post_action,
                            duration_ms=post_duration_ms,
                            elapsed_ms=post_elapsed_ms,
                            error=post_message if post_action in ("warn", "reopen", "fail") else None,
                        )
                        execution_stack.add_step(post_step)
                        self._emit(
                            "post_finalize", action=post_action, message=post_message,
                            duration_ms=post_duration_ms, elapsed_ms=post_elapsed_ms,
                        )
                        if post_result and post_result.action == "reopen":
                            self._reopen_count += 1
                            self._post_finalize_ran = False
                            session.unfinalize_all()
                            self._log(f"[post_finalize] Reopening (#{self._reopen_count}): {post_result.message}")
                            self._emit("warning", message=f"Validation issue: {post_result.message}")
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"VALIDATION WARNING: {post_result.message} "
                                    f"Please review and fix the issue, then finalize again."
                                ),
                            })
                            continue
                        elif post_result and post_result.action == "warn":
                            self._emit("warning", message=f"Validation warning: {post_result.message}")
                        elif post_result and post_result.action == "fail":
                            self._emit("error", message=f"Validation failed: {post_result.message}")
                            final_message = f"Validation failed: {post_result.message}"
                            break

                    break

            # Process tool calls
            # Build assistant message dict with tool calls for message history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
            tool_call_dicts = []
            for tc in tool_calls:
                tc_dict: dict[str, Any] = {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, default=str)},
                }
                if tc.thought_signature is not None:
                    # Gemini requires thought_signature round-tripped at the tool call level
                    tc_dict["extra_content"] = {"google": {"thought_signature": tc.thought_signature}}
                tool_call_dicts.append(tc_dict)
            assistant_msg["tool_calls"] = tool_call_dicts
            messages.append(assistant_msg)

            for tool_call in tool_calls:
                tool_name = tool_call.name
                tool_args = tool_call.arguments

                # Verbose: print tool call
                if self.verbose:
                    print(f"  Tool: {tool_name}")
                    print(f"  Args: {json.dumps(tool_args, indent=2)}")

                # Emit tool call event for web UI
                self._log(f"  Tool call: {tool_name}({json.dumps(tool_args, default=str)[:300]})")
                self._emit("tool_call", name=tool_name, args=tool_args,
                          elapsed_ms=int((time.time() - start_time) * 1000))

                # If this exact call was already loop-warned, block it instead of executing.
                # This forces the LLM to use different arguments or stop.
                call_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, default=str)}"

                if call_key in self._loop_warned:
                    count = self._tool_call_counts.get(call_key, "?")
                    tool_result = ToolResult(
                        success=False,
                        error=(
                            f"BLOCKED: '{tool_name}' has been called with these exact arguments "
                            f"{count} times and a loop warning was already sent. "
                            "Use DIFFERENT arguments (e.g. add or change title=, column=, etc.) "
                            "or stop calling this tool and finalize what you have."
                        ),
                        error_type="LOOP_ERROR",
                    )
                else:
                    # Execute tool
                    tool_result = self._execute_tool(tool_name, session, tool_args)

                    # Track repeated identical tool calls.
                    # Successful calls always count. Failed calls only count when the
                    # immediately preceding call was the same failure (consecutive mistake).
                    if tool_result.success:
                        self._tool_call_counts[call_key] = self._tool_call_counts.get(call_key, 0) + 1
                        self._last_failed_key = None
                        self._consecutive_errors = 0
                    else:
                        if self._last_failed_key == call_key:
                            # Same failure twice in a row — count it
                            self._tool_call_counts[call_key] = self._tool_call_counts.get(call_key, 0) + 1
                        self._last_failed_key = call_key
                        self._consecutive_errors += 1

                # Handle ask_human tool - pause for user input
                if tool_result.success and tool_result.metadata.get("needs_human_input"):
                    human_response = self._get_human_input(tool_result)
                    tool_result.metadata["human_response"] = human_response
                    tool_result.metadata["instruction"] = (
                        f"The user answered: '{human_response}'. "
                        f"Use this value in your next tool call."
                    )
                    session.record_human_input(
                        question=tool_result.metadata.get("question", ""),
                        response=human_response,
                    )

                # Handle chart results - save to file and emit event for web UI
                if tool_result.success and tool_result.metadata.get("chart_base64"):
                    chart_filename = self._save_chart_artifact(tool_result, tool_args)
                    if chart_filename:
                        self._emit("chart", file=chart_filename)
                        # Replace base64 data with filename in metadata (so LLM doesn't get huge base64)
                        tool_result.metadata["chart_file"] = chart_filename
                        del tool_result.metadata["chart_base64"]
                        # Track charted view for auto-finalize fallback
                        self._last_charted_view = tool_args.get("view_name")
                        # Write chart provenance file alongside the PNG
                        self._save_provenance(
                            session, tool_args.get("view_name"),
                            x=tool_args.get("x"), y=tool_args.get("y"),
                            color_by=tool_args.get("color_by"),
                            output_dir="artifacts",
                            filename=chart_filename.replace(".png", "_provenance.txt"),
                        )

                # Emit tool result event for web UI
                rows = None
                if tool_result.dataframe is not None:
                    rows = len(tool_result.dataframe)
                tool = self.tool_registry.get(tool_name)
                summary = tool.summarize(tool_result) if tool else str(tool_result)
                if tool_result.success:
                    self._log(f"  Tool result: OK{f' ({rows} rows)' if rows else ''} - {summary[:200]}")
                else:
                    self._log(f"  Tool result: ERROR - {tool_result.error}")
                self._emit(
                    "tool_result",
                    name=tool_name,
                    success=tool_result.success,
                    summary=summary,
                    rows=rows,
                    error=tool_result.error,
                    elapsed_ms=int((time.time() - start_time) * 1000),
                )

                # Verbose: print result preview
                if self.verbose:
                    result_dict = tool_result.to_dict()
                    # Show row count prominently if available
                    rows_info = ""
                    if "rows" in result_dict:
                        rows_info = f" ({result_dict['rows']} rows)"
                    elif tool_result.dataframe is not None:
                        rows_info = f" ({len(tool_result.dataframe)} rows)"

                    result_str = str(result_dict)
                    result_preview = result_str[:400] + "..." if len(result_str) > 400 else result_str
                    print(f"  Result{rows_info}: {result_preview}")
                    print()

                # Get tool summary
                tool = self.tool_registry.get(tool_name)
                tool_summary = tool.summarize(tool_result) if tool else str(tool_result)

                # Create step log
                step = ExecutionStep(
                    step_number=len(execution_stack.steps) + 1,
                    timestamp=datetime.now(),
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=tool_result.to_dict(),
                    tool_summary=tool_summary,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    duration_ms=int((time.time() - step_start) * 1000),
                    elapsed_ms=int((time.time() - start_time) * 1000),
                    error=tool_result.error if not tool_result.success else None,
                )
                execution_stack.add_step(step)

                # Log to session
                session.log_step(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=tool_result.to_dict(),
                    summary=tool_summary,
                    error=tool_result.error,
                )

                # Build tool response message
                response_content = self._format_tool_response(
                    tool_result, session, tool_name=tool_name if not tool_result.success else None
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": response_content,
                })

            # Log session state after processing all tool calls
            view_names = list(session.views.keys())
            finalized_names = list(session.finalized)
            self._log(f"  Session state: views={view_names}, finalized={finalized_names}")

            # Detect repeated identical tool calls — warn once per key
            for call_key, count in self._tool_call_counts.items():
                if count >= 3 and call_key not in self._loop_warned:
                    self._loop_warned.add(call_key)
                    tool_label = call_key.split(":", 1)[0]
                    self._log(f"[loop] Repeated tool call detected: {tool_label} × {count}")
                    self._emit("warning", message=f"Loop detected: '{tool_label}' called {count} times with same args")
                    extra = ""
                    if tool_label == "chart":
                        extra = (
                            " If you need to change the chart title, add title='...' to your call. "
                            "Any further identical chart() calls will be BLOCKED."
                        )
                    else:
                        extra = " Any further identical calls will be BLOCKED."
                    messages.append({
                        "role": "user",
                        "content": (
                            f"WARNING: You have called '{tool_label}' with the same arguments {count} times "
                            f"and received the same result each time. Stop repeating — use the information you "
                            f"already have and make progress toward the goal.{extra}"
                        ),
                    })

            # Inject replan guidance after errors — tiered by severity.
            if self._consecutive_errors == 1:
                messages.append({
                    "role": "user",
                    "content": (
                        "NOTE: The previous tool call failed. Read the error carefully before retrying. "
                        "If the error includes a hint or suggests a different approach, strongly consider it. "
                        "Use explore(view='...') to inspect available columns if you are unsure."
                    ),
                })
            elif self._consecutive_errors == 2:
                self._log(f"[replan] {self._consecutive_errors} consecutive errors — nudging replan")
                messages.append({
                    "role": "user",
                    "content": (
                        f"WARNING: {self._consecutive_errors} consecutive tool calls have failed. "
                        f"Stop and reconsider your approach — the current strategy is not working. "
                        f"Review the error messages and try a different approach."
                    ),
                })
            elif self._consecutive_errors >= 3:
                self._log(f"[replan] {self._consecutive_errors} consecutive errors — strong replan")
                self._emit("warning", message=f"Consecutive errors: {self._consecutive_errors} — requesting replan")
                messages.append({
                    "role": "user",
                    "content": (
                        f"STOP AND REPLAN: The last {self._consecutive_errors} tool calls ALL failed. "
                        f"Retrying variations of the same approach will not work. "
                        f"You MUST try a fundamentally different strategy. "
                        f"Use explore(view='...') to check what columns are actually available."
                    ),
                })

            # Compact old messages to save context
            messages = self._compact_messages(messages)

        else:
            # Hit max iterations
            self._log(f"MAX ITERATIONS REACHED ({self.max_iterations})")
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"MAX ITERATIONS REACHED ({self.max_iterations})")
                print(f"{'='*60}")

            result = self._max_iterations_result(
                execution_stack,
                session,
                final_message,
                start_time,
                total_tokens_in,
                total_tokens_out,
                total_tokens_reasoning,
                messages_snapshots,
            )

            # Save results and emit completion event (same as normal completion)
            if self.stream_output and self.session_dir:
                results_dir = self.session_dir / "results"
                results_dir.mkdir(parents=True, exist_ok=True)

                for view in result.views:
                    csv_filename = f"{view.name}.csv"
                    csv_path = results_dir / csv_filename
                    view.dataframe.to_csv(csv_path, index=False)

                    # Write view provenance alongside CSV
                    self._save_provenance(
                        session, view.name,
                        output_dir="results",
                        filename=f"{view.name}_provenance.txt",
                    )

                    preview_rows = self.stream_output.preview_rows if self.stream_output else 10
                    preview = view.dataframe.head(preview_rows).to_dict(orient="records")
                    self._emit(
                        "result",
                        file=csv_filename,
                        rows=len(view.dataframe),
                        columns=list(view.dataframe.columns),
                        preview=preview,
                    )

            duration = time.time() - start_time
            reasoning_part = f", tokens_reasoning={total_tokens_reasoning}" if total_tokens_reasoning else ""
            self._log(f"Session complete: status=max_iterations, views={len(result.views)}, "
                      f"tokens_in={total_tokens_in}, tokens_out={total_tokens_out}{reasoning_part}, "
                      f"duration={duration:.1f}s, steps={len(execution_stack.steps)}")
            context_window = self.llm_config.context_window if self.llm_config else 0
            self._emit(
                "complete", status="max_iterations", summary=None,
                elapsed_ms=int(duration * 1000),
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                tokens_reasoning=total_tokens_reasoning,
                max_context_tokens=max_context_tokens,
                context_window=context_window,
            )

            # Write status.json for max_iterations case
            if self.session_dir:
                self._write_status_file("max_iterations", None, len(result.views))

            return result

        # Verbose: session complete
        if self.verbose:
            print(f"\n{'='*60}")
            print("SESSION COMPLETE")
            print(f"{'='*60}")

        # Build final result
        execution_stack.completed_at = datetime.now()
        execution_stack.messages_snapshots = messages_snapshots
        duration = time.time() - start_time

        # Collect finalized views
        finalized_views = [
            FinalizedView(
                name=name,
                dataframe=df,
                metadata=meta,
            )
            for name, df, meta in session.get_finalized_views()
        ]

        # Generate summary if requested
        summary = None
        if generate_summary and finalized_views:
            summary = self._generate_summary(session, final_message, messages)

        # Determine status with more detail
        if finalized_views:
            status = "success"
        elif len(session.views) > 0:
            # Views created but not finalized
            status = "incomplete"
        else:
            # No views at all
            status = "no_output"

        result = AgentResult(
            session_id=session_id,
            views=finalized_views,
            execution_stack=execution_stack,
            final_message=final_message,
            summary=summary,
            status=status,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
            total_tokens_reasoning=total_tokens_reasoning,
            duration_seconds=duration,
        )

        # Save if log manager configured
        if self.log_manager:
            self.log_manager.save_result(result)

        # Emit result events for web UI
        if self.stream_output and self.session_dir:
            results_dir = self.session_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            for view in finalized_views:
                # Save CSV for web download
                csv_filename = f"{view.name}.csv"
                csv_path = results_dir / csv_filename
                view.dataframe.to_csv(csv_path, index=False)

                # Write view provenance alongside CSV
                self._save_provenance(
                    session, view.name,
                    output_dir="results",
                    filename=f"{view.name}_provenance.txt",
                )

                # Emit result event with preview
                preview_rows = self.stream_output.preview_rows if self.stream_output else 10
                preview = view.dataframe.head(preview_rows).to_dict(orient="records")
                self._emit(
                    "result",
                    file=csv_filename,
                    rows=len(view.dataframe),
                    columns=list(view.dataframe.columns),
                    preview=preview,
                )

        # Emit completion event
        reasoning_part = f", tokens_reasoning={total_tokens_reasoning}" if total_tokens_reasoning else ""
        self._log(f"Session complete: status={status}, views={len(finalized_views)}, "
                  f"tokens_in={total_tokens_in}, tokens_out={total_tokens_out}{reasoning_part}, "
                  f"duration={duration:.1f}s, steps={len(execution_stack.steps)}")
        context_window = self.llm_config.context_window if self.llm_config else 0
        complete_data = {
            "status": status, "summary": summary,
            "elapsed_ms": int(duration * 1000),
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "tokens_reasoning": total_tokens_reasoning,
            "max_context_tokens": max_context_tokens,
            "context_window": context_window,
        }
        if final_message:
            complete_data["llm_message"] = final_message
        self._emit("complete", **complete_data)

        # Write status.json as definitive completion marker (atomic write)
        if self.session_dir:
            self._write_status_file(status, summary, len(finalized_views))

        return result

    def _write_status_file(self, status: str, summary: str | None, view_count: int) -> None:
        """Write status.json atomically as completion marker."""
        import json
        import tempfile

        status_data = {
            "status": status,
            "summary": summary,
            "view_count": view_count,
            "completed_at": time.time(),
        }

        status_path = self.session_dir / "status.json"
        # Atomic write: write to temp file, then rename
        try:
            fd, tmp_path = tempfile.mkstemp(dir=self.session_dir, suffix=".tmp")
            with os.fdopen(fd, 'w') as f:
                json.dump(status_data, f)
            os.rename(tmp_path, status_path)
        except Exception as e:
            self._log(f"Warning: Failed to write status.json: {e}")

    def _build_context_reminder(
        self,
        query: str,
        messages: list[dict],
        iteration: int,
        session: "Session",
    ) -> str | None:
        """Build an ephemeral context reminder appended to each LLM call.

        This message is injected before the LLM call and removed afterward,
        so it never accumulates in conversation history. It reinforces:
        - Today's date (so the LLM knows the current year)
        - The original user query (so constraints survive compaction)
        - The last tool call and result (so the LLM stays grounded)
        - Recent views with schema info

        Only injected from iteration 2 onward (iteration 1 has no prior context).
        """
        if iteration <= 1:
            return None

        from datetime import date as _date
        today = _date.today().isoformat()

        # Find last tool result in messages
        last_tool_info = ""
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > 500:
                    content = content[:500] + "..."
                # Find matching assistant tool_call to get the tool name
                tool_call_id = msg.get("tool_call_id")
                tool_name = "unknown"
                if tool_call_id:
                    for m in reversed(messages):
                        if m.get("role") == "assistant" and m.get("tool_calls"):
                            for tc in m["tool_calls"]:
                                if tc.get("id") == tool_call_id:
                                    tool_name = tc["function"]["name"]
                                    break
                            if tool_name != "unknown":
                                break
                last_tool_info = (
                    f"\n<last_tool_result>\n"
                    f"  <tool>{tool_name}</tool>\n"
                    f"  <result>{content}</result>\n"
                    f"</last_tool_result>"
                )
                break

        # Include last 3 views with name, columns, row count
        views_info = ""
        view_items = list(session.views.items())
        if view_items:
            recent = view_items[-3:]  # last 3
            view_lines = []
            for name, meta in recent:
                cols = [c for c in meta.columns]
                final_tag = ' finalized="true"' if meta.is_finalized else ""
                view_lines.append(f'  <view name="{name}" rows="{meta.rows}" columns="{", ".join(cols)}"{final_tag} />')
            views_info = "\n<recent_views>\n" + "\n".join(view_lines) + "\n</recent_views>"

        # If any views are finalized, nudge the LLM to wrap up
        finalized_nudge = ""
        if session.finalized:
            finalized_names = list(session.finalized)
            finalized_nudge = (
                f"\n<finalized_views>{', '.join(finalized_names)}</finalized_views>\n"
                f"<instruction>The following views have been finalized: {', '.join(finalized_names)}. "
                f"If these address the original query, respond with a text summary and STOP. "
                f"Only continue making tool calls if the query is not yet fully answered.</instruction>"
            )

        return (
            f"<context_reminder>\n"
            f"  <today>{today}</today>\n"
            f"  <original_query>{query}</original_query>{last_tool_info}{views_info}{finalized_nudge}\n"
            f"</context_reminder>"
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt with dynamic content.

        Tool descriptions are NOT included here because they are already
        sent as function definitions in the tools parameter. This avoids
        duplicating tool info and saves significant context tokens.
        """
        return self.prompt_builder.build(
            tool_registry=self.tool_registry,
            source_registry=self.source_registry,
            include_tools=False,
        )

    def _call_llm(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Call the LLM API with retry on transient errors."""
        provider = self._get_provider()
        n_tools = len(tools) if tools else 0
        self._log(f"  Sending request to {self.llm_endpoint} (model={self.llm_model}, msgs={len(messages)}, tools={n_tools})")

        # Let the provider format tools for its specific API
        formatted_tools = provider.format_tools(tools) if tools else []

        # Rate-limit delay (configurable per LLM config, useful for free-tier APIs)
        delay = self.llm_config.request_delay if self.llm_config else 0.0
        if delay > 0:
            self._log(f"  [rate-limit] sleeping {delay}s before request")
            time.sleep(delay)

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                t0 = time.time()
                result = provider.call(
                    messages=messages,
                    tools=formatted_tools,
                    model=self.llm_model,
                    temperature=self.temperature,
                    max_tokens=self.llm_config.max_tokens if self.llm_config else 4096,
                )
                self._log(f"  LLM responded in {time.time() - t0:.1f}s")
                return result
            except Exception as e:
                elapsed = time.time() - t0
                self._log(f"  LLM call failed after {elapsed:.1f}s: {e}")
                is_transient = "connection" in str(e).lower() or "timeout" in str(e).lower()
                if is_transient and attempt < max_retries:
                    wait = 2 ** attempt  # 1s, 2s
                    self._log(f"[retry] LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {wait}s...")
                    self._emit("warning", message=f"LLM connection issue, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                    # Reset provider in case connection is stale
                    provider.reset()
                    continue
                raise

    def _execute_tool(
        self,
        tool_name: str,
        session: Session,
        args: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool and handle errors."""
        tool = self.tool_registry.get(tool_name)

        if tool is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}",
                error_type="UNKNOWN_TOOL",
            )

        # Detect garbled/placeholder arguments (e.g. "..??" from uncertain models)
        args_str = str(args)
        if "??" in args_str:
            return ToolResult(
                success=False,
                error=(
                    "Tool arguments contain placeholder values ('??'). "
                    "Provide real values for all parameters."
                ),
                error_type="VALIDATION_ERROR",
            )

        try:
            result = tool.execute(session, **args)

            # If successful with a view, add to session
            if result.success and result.view_name and result.dataframe is not None:
                if result.view_name not in session.views:
                    parent_views = self._extract_parent_views(tool_name, args, result.metadata)
                    session.add_view(
                        result.view_name,
                        result.dataframe,
                        operation=tool_name,
                        parent_views=parent_views,
                        extra_metadata={
                            **result.metadata,
                            "tool_name": tool_name,
                            "tool_args": args,
                        },
                    )

                # Handle finalize flag from tool metadata
                if result.metadata.get("finalize"):
                    session.finalize_view(result.view_name)

            return result

        except Exception as e:
            # Log full traceback for debugging
            tb = traceback.format_exc()
            self._log(f"[error] Tool {tool_name} failed: {e}")
            self._log(f"[error] Traceback:\n{tb}")
            # Also print to console for visibility
            print(f"[ERROR] Tool {tool_name} failed: {e}")
            print(f"[ERROR] Traceback:\n{tb}")
            return ToolResult(
                success=False,
                error=str(e),
                error_type="EXECUTION_ERROR",
            )

    def _extract_parent_views(self, tool_name: str, args: dict, result_metadata: dict | None = None) -> list[str]:
        """Extract parent view names from tool args for lineage tracking."""
        parents = []
        # Single source view (compute_indicator, aggregate, pivot, chart)
        if "source_view" in args:
            parents.append(args["source_view"])
        # Join: two parent views
        if "left_view" in args:
            parents.append(args["left_view"])
        if "right_view" in args:
            parents.append(args["right_view"])
        # Concat: list of views
        if "views" in args and isinstance(args["views"], list):
            parents.extend(args["views"])
        # create_view: source is a parent view when from_view=True (determined by tool at runtime)
        if tool_name == "create_view" and result_metadata and result_metadata.get("from_view"):
            src = args.get("source", "")
            if src:
                parents.append(src)
        return parents

    def _format_tool_response(
        self, result: ToolResult, session: Session, tool_name: str | None = None,
    ) -> str:
        """Format tool result for LLM, including session state.

        Args:
            result: Tool execution result.
            session: Current session.
            tool_name: If set (on error), optionally re-include tool docs
                       so the LLM has immediate context for the retry.
        """
        # Sanitize to handle inf/nan values which are not valid JSON
        sanitized = sanitize_for_json(result.to_dict())
        parts = [json.dumps(sanitized, default=str, allow_nan=False)]

        # On error, optionally re-include the tool's schema for immediate LLM context.
        # This is one-shot — it's in this response only, not persisted in the system prompt.
        if tool_name and self.include_tool_docs_on_error:
            tool = self.tool_registry.get(tool_name)
            if tool:
                desc = tool.tool_description()
                param_lines = []
                for p in tool.parameters():
                    req = " (required)" if p.required else ""
                    param_lines.append(f"  {p.name}: {p.param_type}{req} — {p.description}")
                params_str = "\n".join(param_lines) if param_lines else "  (none)"
                parts.append(
                    f"\n\n--- Tool reference (for retry) ---\n"
                    f"{tool_name}: {desc}\n"
                    f"Parameters:\n{params_str}"
                )

        # Add current session state summary
        views_summary = session.get_views_summary()
        if views_summary:
            parts.append(f"\n\n{views_summary}")

        return "\n".join(parts)

    def _compact_messages(self, messages: list[dict], keep_recent: int = 5) -> list[dict]:
        """
        Compact old tool messages to save context.

        Keeps full detail for the most recent tool responses,
        compacts older ones to summaries.
        """
        # Find tool message indices
        tool_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "tool"
        ]

        if len(tool_indices) <= keep_recent:
            return messages

        # Compact older tool messages
        to_compact = tool_indices[:-keep_recent]

        # Compact assistant tool_call args for repeated calls that are being compacted.
        # When the LLM repeats the same tool N times, older assistant messages carry
        # the full arguments blob unnecessarily — replace with a short note.
        compacted_ids = {messages[idx].get("tool_call_id") for idx in to_compact}
        for msg in messages:
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            for tc in msg["tool_calls"]:
                if tc.get("id") not in compacted_ids:
                    continue
                tool_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                    call_key = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
                except Exception:
                    continue
                count = self._tool_call_counts.get(call_key, 0)
                if count > 1:
                    tc["function"]["arguments"] = json.dumps(
                        {"_note": f"repeated {count}x — args identical to earlier calls"}
                    )

        for idx in to_compact:
            content = messages[idx].get("content", "")
            # Keep just first line or summary
            if len(content) > 200:
                # Preserve human input responses — compact to just the answer so the
                # LLM doesn't lose track of what the user said.
                try:
                    first_json = content.split("\n\n")[0]
                    data = json.loads(first_json)
                    if data.get("needs_human_input") and "human_response" in data:
                        compact = {
                            "human_response": data["human_response"],
                            "instruction": data.get(
                                "instruction",
                                f"User answered: '{data['human_response']}'",
                            ),
                        }
                        messages[idx]["content"] = json.dumps(compact)
                        continue
                    # Build a clean compact summary instead of malformed truncated JSON
                    if "error" in data:
                        messages[idx]["content"] = f"[ERROR: {str(data['error'])[:150]}]"
                    else:
                        parts = []
                        if "view" in data:
                            rows = data.get("rows", "?")
                            parts.append(f"view={data['view']} ({rows} rows)")
                        if "chart_type" in data:
                            chart_info = f"chart({data['chart_type']})"
                            title = data.get("title")
                            if title:
                                chart_info += f" '{title}'"
                            chart_file = data.get("chart_file")
                            if chart_file:
                                chart_info += f" → {chart_file}"
                            parts.append(chart_info)
                        if "indicator" in data:
                            parts.append(f"indicator={data['indicator']}")
                        warnings_list = data.get("warnings") or []
                        for w in warnings_list[-2:]:
                            parts.append(f"WARNING: {str(w)[:120]}")
                        if parts:
                            messages[idx]["content"] = "[OK: " + " | ".join(parts) + "]"
                        else:
                            messages[idx]["content"] = content[:150] + " [compacted]"
                    continue
                except Exception:
                    pass
                messages[idx]["content"] = content[:150] + " [compacted]"

        return messages

    def _generate_summary(
        self,
        session: Session,
        final_message: str,
        messages: list[dict],
    ) -> str:
        """Generate LLM summary of the results."""
        # For now, use the final message as summary
        # Could add a separate summarization call here
        return final_message

    def _error_result(
        self,
        error: str,
        stack: ExecutionStack,
        session: Session,
        start_time: float,
        tokens_in: int,
        tokens_out: int,
        tokens_reasoning: int = 0,
    ) -> AgentResult:
        """Create error result."""
        duration = time.time() - start_time
        stack.completed_at = datetime.now()

        # Add error as an execution step so it appears in the execution log
        error_step = ExecutionStep(
            step_number=len(stack.steps) + 1,
            timestamp=datetime.now(),
            tool_name="error",
            tool_args={},
            tool_summary=error,
            duration_ms=0,
            elapsed_ms=int(duration * 1000),
            error=error,
        )
        stack.add_step(error_step)

        # Log completion details (same format as success/max_iterations)
        view_names = list(session.views.keys())
        finalized_names = list(session.finalized)
        self._log(f"Session error: {error}")
        self._log(f"  Session state at error: views={view_names}, finalized={finalized_names}")
        reasoning_part = f", tokens_reasoning={tokens_reasoning}" if tokens_reasoning else ""
        self._log(f"Session complete: status=error, views=0, "
                  f"tokens_in={tokens_in}, tokens_out={tokens_out}{reasoning_part}, "
                  f"duration={duration:.1f}s, steps={len(stack.steps)}")

        # Emit error event for web UI
        self._emit("error", message=error)
        context_window = self.llm_config.context_window if self.llm_config else 0
        self._emit(
            "complete", status="error",
            elapsed_ms=int(duration * 1000),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_reasoning=tokens_reasoning,
            max_context_tokens=0,
            context_window=context_window,
        )

        # Write status.json for error case
        if self.session_dir:
            self._write_status_file("error", error, 0)

        return AgentResult(
            session_id=stack.session_id,
            execution_stack=stack,
            status="error",
            error_message=error,
            total_tokens_in=tokens_in,
            total_tokens_out=tokens_out,
            total_tokens_reasoning=tokens_reasoning,
            duration_seconds=duration,
        )

    def _max_iterations_result(
        self,
        stack: ExecutionStack,
        session: Session,
        final_message: str,
        start_time: float,
        tokens_in: int,
        tokens_out: int,
        tokens_reasoning: int = 0,
        messages_snapshots: dict[int, list] | None = None,
    ) -> AgentResult:
        """Create max iterations result."""
        stack.completed_at = datetime.now()
        if messages_snapshots:
            stack.messages_snapshots = messages_snapshots

        # Still collect any finalized views
        finalized_views = [
            FinalizedView(name=name, dataframe=df, metadata=meta)
            for name, df, meta in session.get_finalized_views()
        ]

        return AgentResult(
            session_id=stack.session_id,
            views=finalized_views,
            execution_stack=stack,
            final_message=final_message,
            status="max_iterations",
            total_tokens_in=tokens_in,
            total_tokens_out=tokens_out,
            total_tokens_reasoning=tokens_reasoning,
            duration_seconds=time.time() - start_time,
        )

    def _looks_like_question(self, message: str) -> bool:
        """Detect if LLM response is a clarifying question instead of a tool call."""
        msg = message.strip()
        if msg.endswith("?"):
            return True
        lower = msg.lower()
        return any(phrase in lower for phrase in [
            "did you mean", "could you clarify", "please clarify",
            "what did you mean", "which group", "could you please",
        ])

    def _get_human_input(self, tool_result: ToolResult) -> str:
        """
        Prompt user for input when ask_human tool is called.

        In CLI mode: displays question and waits for terminal input.
        In web mode: emits needs_input event and polls StreamInput for response.

        The web mode polling uses time.sleep() between checks (default 0.5s),
        so it blocks the thread but is not CPU-intensive.

        Always includes a "Something else" option for free-text input.
        """
        question = tool_result.metadata.get("question", "")
        options = tool_result.metadata.get("options", [])
        context = tool_result.metadata.get("context", "")

        # Generate a unique request ID for this input request
        request_id = f"input_{uuid.uuid4().hex[:8]}"

        # Web mode: emit event and wait for response via StreamInput
        if self.stream_output and self.stream_input:
            self._emit(
                "needs_input",
                id=request_id,
                question=question,
                options=options,
                context=context,
            )

            # Wait for response - uses time.sleep() polling, blocks but not CPU-intensive
            # Default 5 minute timeout
            response = self.stream_input.wait_for_response(request_id, timeout=300)

            if response is None:
                response = ""  # Timeout - use empty response

            # Emit that we received the response
            self._emit("input_received", id=request_id, value=response)

            return response

        # CLI mode: use terminal input
        print("\n" + "=" * 60)
        print("AGENT NEEDS CLARIFICATION")
        print("=" * 60)

        if context:
            print(f"\nContext: {context}")

        print(f"\nQuestion: {question}\n")

        if options:
            print("Options:")
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}")
            print(f"  {len(options) + 1}. Something else (type your answer)")
            print()

            while True:
                try:
                    choice = input("Enter your choice (number or text): ").strip()

                    # Check if it's a number
                    if choice.isdigit():
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(options):
                            response = options[choice_num - 1]
                            break
                        elif choice_num == len(options) + 1:
                            response = input("Enter your answer: ").strip()
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(options) + 1}")
                    else:
                        # Treat any non-numeric input as free-text
                        response = choice
                        break
                except (ValueError, EOFError):
                    response = ""
                    break
        else:
            # No options - just get free text
            response = input("Your answer: ").strip()

        print("=" * 60 + "\n")
        return response

    def _save_chart_artifact(self, tool_result: ToolResult, tool_args: dict[str, Any]) -> str | None:
        """
        Save chart base64 data to artifacts directory.

        Returns the filename if saved successfully, None otherwise.
        """
        if not self.session_dir:
            return None

        import base64

        try:
            # Create artifacts directory
            artifacts_dir = self.session_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename from chart metadata
            chart_type = tool_result.metadata.get("chart_type", "chart")
            title = tool_result.metadata.get("title", "chart")
            # Sanitize title for filename
            safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
            safe_title = safe_title[:50]  # Limit length
            self._chart_counter += 1
            filename = f"{chart_type}_{safe_title}_{self._chart_counter}.png"

            # Decode and save
            chart_data = base64.b64decode(tool_result.metadata["chart_base64"])
            filepath = artifacts_dir / filename
            filepath.write_bytes(chart_data)

            return filename
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not save chart artifact: {e}")
            return None

    def _save_provenance(
        self,
        session: Session,
        view_name: str | None,
        x: str | None = None,
        y: str | None = None,
        color_by: str | None = None,
        output_dir: str = "results",
        filename: str | None = None,
    ) -> None:
        """Write a provenance file for a view. Best-effort — never raises."""
        if not self.session_dir or not view_name:
            return
        try:
            from ..core.lineage import get_provenance
            provenance = get_provenance(session, view_name, x=x, y=y, color_by=color_by)
            if not provenance:
                return
            out_dir = self.session_dir / output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (filename or f"{view_name}_provenance.txt")
            out_path.write_text(provenance, encoding="utf-8")
        except Exception:
            pass  # Provenance is best-effort
