"""
Stock-specific aggregate tool.

Subclasses the generic AggregateTool to provide stock-domain examples
in error messages and tool descriptions that help LLMs make correct
tool calls (partition_by=['ticker'], column='close', etc.).
"""

from agentic_TRACE.tools.builtins.aggregate import AggregateTool


class StockAggregateTool(AggregateTool):
    """Aggregate tool with stock-domain hints for LLM guidance."""

    def _hint_mode_required(self) -> str:
        return (
            "Example: function='pct_change', column='close', partition_by=['ticker'] "
            "for per-stock daily returns."
        )

    def _hint_low_row_warning(self) -> str:
        return (
            "If you need WHICH ROW had the max/min (e.g. the date of the highest close), "
            "use create_view(sort_by='-close', limit=1) instead — no aggregate needed. "
            "To keep all rows with the group stat (e.g. normalize by group max), "
            "use window mode: aggregate(function='max', column='close', partition_by=['ticker']) "
            "adds close_max to every row without reducing rows."
        )

    def _hint_tool_examples(self) -> str:
        return (
            "Normalization example: aggregate(function='mean', column='ema', "
            "partition_by=['ticker'], output_column='ema_mean') then "
            "create_view(add_columns={'z': '(ema - ema_mean) / ema_std'}). "
            "Use partition_by=['ticker'] to compute per-stock in window mode. "
            "Top-N per group pattern: aggregate(function='rank', column='close', "
            "partition_by=['ticker'], sort_by='-close', output_column='rank') then "
            "create_view(where='rank <= 5') keeps top 5 per ticker. "
            "sort_by='-close' makes rank 1 = highest close; "
            "sort_by='close' makes rank 1 = lowest close."
        )
