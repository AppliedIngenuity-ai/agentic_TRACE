"""
Lifecycle hooks for the stocks example.

Pre-process: Injects relevant context before the LLM starts.
Post-finalize: Validates output data by reviewing query, steps, and results.
"""

import json
import numpy as np
import os
from typing import TYPE_CHECKING

from agentic_TRACE.agent.result import PreProcessResult, PostFinalizeResult

if TYPE_CHECKING:
    from agentic_TRACE.core.session import Session
    from agentic_TRACE.agent.result import FinalizedView


def pre_process(query: str, session: "Session") -> PreProcessResult:
    """
    Pre-process hook: analyze the query and inject relevant context.

    Override or extend this to inject task-specific guidance,
    related examples, or domain knowledge before the LLM starts.
    """
    return PreProcessResult(context=None)


class PostFinalizeValidator:
    """
    Post-finalize validator that reviews execution results.

    Examines the original query, execution steps, and output data
    to detect inconsistencies, errors, or suspicious patterns.

    Uses a fast LLM call to review a summary of the execution.
    Falls back to rule-based checks if no LLM is available.
    """

    def __init__(self, llm_config=None):
        """
        Initialize validator.

        Args:
            llm_config: LLMConfig for the review LLM call.
                If None, uses rule-based validation only.
        """
        self.llm_config = llm_config
        self._client = None

    def __call__(self, session: "Session", finalized_views: list["FinalizedView"]) -> PostFinalizeResult:
        """Run validation on finalized output."""
        # Collect rule-based issues first
        rule_issues = self._rule_based_checks(session, finalized_views)

        # Build review context for LLM-based review
        review_context = self._build_review_context(session, finalized_views)

        # Add contextual hints for the LLM reviewer
        hints = []
        unpartitioned = [
            s for s in session.steps
            if s.tool_name == "compute_indicator"
            and not s.error
            and not (s.tool_args or {}).get("partition_by")
        ]
        if unpartitioned:
            hints.append(
                "It looks like compute_indicator may have been called without partition_by. "
                "Please check whether the tool use was correct — specifically, verify that "
                "the user's request did not require per-stock calculations (e.g. EMA per ticker). "
                "If the data contains multiple tickers and partition_by was omitted, "
                "the indicator would have been computed over all rows combined rather than per stock."
            )

        # Try LLM-based review (also generates summary to avoid extra LLM call)
        llm_issues, summary = self._llm_review(review_context, hints=hints)

        # Combine issues
        all_issues = rule_issues + llm_issues

        if not all_issues:
            return PostFinalizeResult(action="pass", summary=summary)

        message = "Validation issues:\n" + "\n".join(f"  - {i}" for i in all_issues)

        # Reopen if the LLM validator found actionable issues, or if a chart was
        # requested but never produced. Rule-only issues (e.g. null column hints) stay as "warn".
        query_lower = session.query.lower().split()
        chart_requested = any(w in query_lower for w in self._PLOT_KEYWORDS)
        chart_produced = any(s for s in session.steps if s.tool_name == "chart" and not s.error)
        action = "reopen" if (llm_issues or (chart_requested and not chart_produced)) else "warn"

        return PostFinalizeResult(action=action, message=message, summary=summary)

    # Words that *might* indicate a visualization was requested (verb usage only)
    _PLOT_KEYWORDS = {"plot", "chart", "graph", "visualize"}

    def _rule_based_checks(self, session: "Session", finalized_views: list["FinalizedView"]) -> list[str]:
        """Run rule-based sanity checks on output data."""
        issues = []

        for view in finalized_views:
            df = view.dataframe
            if df is None or df.empty:
                continue

            for col in df.select_dtypes(include=[np.number]).columns:
                # Check for infinite values
                if df[col].dtype in [np.float64, np.float32, float]:
                    inf_count = int(np.isinf(df[col]).sum())
                    if inf_count > 0:
                        issues.append(
                            f"'{view.name}.{col}' has {inf_count} infinite values "
                            f"(likely division by zero)"
                        )

                # Check for all-null columns
                if df[col].isna().all():
                    issues.append(f"'{view.name}.{col}' is entirely null/NaN")

        # Soft hint: flag chart keywords found in query but no chart produced.
        # This is intentionally non-definitive — the LLM validator confirms.
        query_lower = session.query.lower().split()
        matched_keywords = [w for w in self._PLOT_KEYWORDS if w in query_lower]
        if matched_keywords:
            chart_steps = [s for s in session.steps if s.tool_name == "chart" and not s.error]
            if not chart_steps:
                kw_list = ", ".join(f'"{w}"' for w in matched_keywords)
                issues.append(
                    f"The keyword(s) {kw_list} appeared in the request — it is possible the "
                    f"user wanted a chart but none was produced. Confirm whether a "
                    f"visualization was actually required."
                )

        return issues

    def _build_review_context(self, session: "Session", finalized_views: list["FinalizedView"]) -> str:
        """Build a concise review context for LLM validation."""
        parts = []

        # Original query
        parts.append(f"QUERY: {session.query}")

        # Human clarifications (if any) — these override ambiguous terms in the query
        if session.human_inputs:
            clarifications = []
            for hi in session.human_inputs:
                clarifications.append(f"  Q: {hi['question']}\n  A: {hi['response']}")
            parts.append("USER CLARIFICATIONS (treat these as the user's true intent):\n" + "\n".join(clarifications))

        # Execution steps
        steps_summary = []
        for step in session.steps:
            if step.tool_name:
                limit = None if step.tool_name == "chart" else 200
                args_str = json.dumps(step.tool_args, default=str)[:limit] if step.tool_args else ""
                status = "ERROR" if step.error else "OK"
                steps_summary.append(f"  {step.step_number}. {step.tool_name}({args_str}) -> {status}")
                if step.summary:
                    steps_summary.append(f"     Result: {step.summary}")
                if step.error:
                    steps_summary.append(f"     Error: {step.error}")
        parts.append("STEPS:\n" + "\n".join(steps_summary))

        # Chart steps — do not truncate args; validator must see full params to avoid false positives
        chart_steps = [s for s in session.steps if s.tool_name == "chart"]
        if chart_steps:
            chart_lines = []
            for s in chart_steps:
                args_str = json.dumps(s.tool_args, default=str) if s.tool_args else ""
                status = "ERROR" if s.error else "OK"
                chart_lines.append(f"  chart({args_str}) -> {status}")
            parts.append("CHARTS:\n" + "\n".join(chart_lines))
        else:
            parts.append("CHARTS: none")

        # Result data samples and statistics
        for view in finalized_views:
            df = view.dataframe
            if df is None or df.empty:
                parts.append(f"VIEW '{view.name}': empty")
                continue

            parts.append(f"\nVIEW '{view.name}': {len(df)} rows, columns: {list(df.columns)}")

            # Unique values for low-cardinality categorical columns — prevents false "missing filter" diagnoses
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            for col in cat_cols:
                n_unique = df[col].nunique()
                if n_unique <= 30:
                    vals = sorted(df[col].dropna().unique().tolist())
                    parts.append(f"  UNIQUE {col} ({n_unique}): {vals}")

            # For small DataFrames, include all data
            if len(df) <= 30:
                parts.append(f"  ALL DATA:\n{df.to_string(max_rows=30, max_cols=15)}")
            else:
                # Sample: first 10, middle 10, last 10
                first = df.head(10)
                last = df.tail(10)
                middle = df.iloc[len(df)//4 : len(df)//4 + 10] if len(df) > 20 else df.iloc[5:15]
                parts.append(f"  FIRST 10 ROWS:\n{first.to_string(max_cols=15)}")
                parts.append(f"  MIDDLE SAMPLE:\n{middle.to_string(max_cols=15)}")
                parts.append(f"  LAST 10 ROWS:\n{last.to_string(max_cols=15)}")

                # Distributional summary for numeric columns
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    desc = numeric_df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
                    parts.append(f"  STATISTICS:\n{desc.to_string()}")

        return "\n".join(parts)

    def _llm_review(self, review_context: str, hints: list[str] | None = None) -> tuple[list[str], str | None]:
        """
        Use LLM to review execution and generate a user-facing summary.

        Returns (issues, summary) — both validation issues and a brief summary
        in a single LLM call, avoiding the need for a separate summary iteration.
        """
        if not self.llm_config:
            return [], None

        try:
            client = self._get_client()
            if not client:
                return [], None

            hints_block = ""
            if hints:
                hints_block = "\n\nADDITIONAL CHECKS:\n" + "\n".join(f"- {h}" for h in hints)

            prompt = (
                "You are a data validation reviewer. Do TWO things:\n\n"
                "1. VALIDATE — check the FINAL OUTPUT DATA for correctness:\n"
                "   - Does the output answer the user's query?\n"
                "   - Are the data values reasonable (no obvious math errors, no suspicious zeros/infinities)?\n"
                "   - Does the data match the user's filters (correct tickers, correct date range)?\n"
                "   - If a chart was requested, was one produced?\n\n"
                "DO NOT flag:\n"
                "   - Process efficiency (redundant steps, extra views) — only the final output matters\n"
                "   - Intermediate errors that were recovered from\n"
                "   - Style or naming issues (truncated view names, etc.)\n\n"
                "RULES:\n"
                "- Before claiming a filter is missing, check the UNIQUE values shown for each column. "
                "If the expected values are already present and no others, the filter was applied.\n"
                "- If the user specified a date or date range, confirm the output corresponds accordingly.\n"
                "- Only flag issues you can confirm from the data shown. Do not speculate.\n\n"
                "2. SUMMARIZE — one or two sentences for the user.\n\n"
                "Format:\nSUMMARY: <text>\nVALIDATION: PASS\n  or\nSUMMARY: <text>\nISSUE: <problem>\n\n"
                f"{review_context}{hints_block}"
            )

            response = client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()

            # Parse summary and issues
            summary = None
            issues = []
            for line in content.split("\n"):
                line = line.strip()
                if line.upper().startswith("SUMMARY:"):
                    summary = line[8:].strip()
                elif line.upper().startswith("ISSUE:"):
                    issues.append(line[6:].strip())
                elif line.upper().startswith("VALIDATION:") and "PASS" in line.upper():
                    continue  # Skip "VALIDATION: PASS" line
                elif line.upper() == "PASS":
                    continue

            return issues, summary

        except Exception as e:
            print(f"[WARN] Post-finalize LLM review failed: {e}")
            return [], None

    def _get_client(self):
        """Get or create OpenAI client for validation calls."""
        if self._client:
            return self._client

        if not self.llm_config:
            return None

        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.llm_config.get_api_base(),
                api_key=self.llm_config.get_api_key() or os.getenv("LLM_API_KEY", "not-needed"),
            )
            return self._client
        except Exception:
            return None
