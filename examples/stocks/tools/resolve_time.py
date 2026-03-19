"""
Resolve time tool - domain-specific temporal resolution for financial data.

This is the domain-specific tool for the stocks example. It handles:
- Quarters: Q1_2024, Q2_2024, etc.
- Relative periods: ytd, last_month, last_year
- Trading days: last_20_trading_days, first_20_trading_days_of_Q1_2024
"""

import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import TYPE_CHECKING

from agentic_TRACE.tools.base import BaseTool, ToolCategory, ToolParameter
from agentic_TRACE.tools.result import ToolResult, ErrorType
from agentic_TRACE.data.sources import DataSource

if TYPE_CHECKING:
    from agentic_TRACE.core.session import Session


class ResolveTimeTool(BaseTool):
    """
    Resolve temporal expressions to concrete date ranges.

    Handles financial temporal expressions and counts actual trading days
    from the database (not approximated).
    """

    name = "resolve_time"
    category = ToolCategory.EXPLORE

    def __init__(self, data_source: DataSource | None = None, trading_days_table: str = "stock_prices"):
        """
        Initialize with data source for trading day lookups.

        Args:
            data_source: DataSource for querying trading days
            trading_days_table: Table name containing date column for trading days
        """
        super().__init__()
        self.data_source = data_source
        self.trading_days_table = trading_days_table

    def set_data_source(self, source: DataSource) -> None:
        """Set the data source for trading day queries."""
        self.data_source = source

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                param_type="string",
                description=(
                    "Named time period (not specific dates). Examples: "
                    "'Q2_2024', 'last_quarter', 'last_month', 'last_20_trading_days', "
                    "'first_10_trading_days_of_Q1_2024'"
                ),
                required=True,
            )
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "expression"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        expression = kwargs["expression"]
        actual_today = datetime.now()
        # Normalize: collapse spaces/hyphens to underscores, then apply aliases
        expr_clean = expression.strip().lower().replace(" ", "_").replace("-", "_")
        # Normalize "past" → "last" (e.g. "past year" → "last_year", "past 100 trading days" → "last_100_trading_days")
        if expr_clean.startswith("past_"):
            expr_clean = "last_" + expr_clean[5:]
        elif expr_clean == "past":
            expr_clean = "last_year"

        # Get effective "today" - use latest data date if today has no data
        effective_today, data_warning = self._get_effective_today(actual_today)

        # Try each resolver in order
        resolvers = [
            self._resolve_n_trading_days_of_period,  # first_20_trading_days_of_Q1_2024
            self._resolve_last_n_trading_days,       # last_20_trading_days
            self._resolve_quarter,                    # Q1_2024
            self._resolve_relative,                   # ytd, last_month
            self._resolve_year,                       # 2024
            self._resolve_n_months,                   # last_7_months
        ]

        for resolver in resolvers:
            result = resolver(expr_clean, effective_today)
            if result:
                # Add data warning if we're using a different date than actual today
                if data_warning:
                    result["data_warning"] = data_warning
                return self.success(**result)

        # Unknown expression
        return self.error(
            f"Unknown temporal expression: '{expression}'. "
            "resolve_time handles NAMED periods only (e.g. Q1_2024, last_quarter, last_4_quarters, last_month, last_20_trading_days). "
            "For specific dates, use them directly in create_view where= parameter "
            "(e.g. where=\"date >= '2025-01-01'\"). Do NOT call resolve_time for specific dates.",
            ErrorType.VALIDATION_ERROR,
        )

    def _resolve_quarter(self, expr: str, today: datetime) -> dict | None:
        """Resolve quarter expressions like Q1_2024, Q2_2024."""
        match = re.match(r'^q([1-4])_?(\d{4})$', expr)
        if not match:
            return None

        quarter = int(match.group(1))
        year = int(match.group(2))

        # Quarter start/end dates
        quarter_dates = {
            1: ((1, 1), (3, 31)),
            2: ((4, 1), (6, 30)),
            3: ((7, 1), (9, 30)),
            4: ((10, 1), (12, 31)),
        }

        (start_month, start_day), (end_month, end_day) = quarter_dates[quarter]

        start = datetime(year, start_month, start_day)
        end = datetime(year, end_month, end_day)

        # If end is in future, use today
        if end > today:
            end = today

        # Count actual trading days
        trading_days = self._count_trading_days(start, end)

        return {
            "expression": expr,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "trading_days": trading_days,
            "period": f"Q{quarter} {year}",
        }

    def _resolve_relative(self, expr: str, today: datetime) -> dict | None:
        """Resolve relative expressions like ytd, last_month, yesterday."""
        yesterday = today - timedelta(days=1)

        relative_periods = {
            "ytd": lambda: (datetime(today.year, 1, 1), today),
            "this_year": lambda: (datetime(today.year, 1, 1), today),
            "last_week": lambda: (today - timedelta(days=7), today),
            "last_month": lambda: (today - timedelta(days=30), today),
            "last_3_months": lambda: (today - timedelta(days=90), today),
            "last_6_months": lambda: (today - timedelta(days=180), today),
            "last_year": lambda: (today - timedelta(days=365), today),
            "last_2_years": lambda: (today - timedelta(days=730), today),
            "last_3_years": lambda: (today - timedelta(days=1095), today),
            "last_4_years": lambda: (today - timedelta(days=1460), today),
            "last_5_years": lambda: (today - timedelta(days=1825), today),
            "yesterday": lambda: (yesterday, yesterday),
            "today": lambda: (today, today),
        }

        # Check for last_N_years / last_N_quarters patterns if not in predefined
        if expr not in relative_periods:
            match = re.match(r'^last_(\d+)_years?$', expr)
            if match:
                n_years = int(match.group(1))
                start = today - timedelta(days=n_years * 365)
                trading_days = self._count_trading_days(start, today)
                return {
                    "expression": expr,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": today.strftime("%Y-%m-%d"),
                    "trading_days": trading_days,
                    "period": f"Last {n_years} years (rolling)",
                }

            # last_quarter / last_N_quarters — go back to start of quarter N quarters ago
            match = re.match(r'^last_(\d+)_quarters?$', expr)
            n_quarters = 1 if expr in ("last_quarter",) else (int(match.group(1)) if match else None)
            if n_quarters is not None or expr == "last_quarter":
                if n_quarters is None:
                    n_quarters = 1
                start = today - relativedelta(months=n_quarters * 3)
                # Snap start to beginning of its calendar quarter
                quarter_start_month = ((start.month - 1) // 3) * 3 + 1
                start = start.replace(month=quarter_start_month, day=1)
                trading_days = self._count_trading_days(start, today)
                label = "last quarter" if n_quarters == 1 else f"last {n_quarters} quarters"
                return {
                    "expression": expr,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": today.strftime("%Y-%m-%d"),
                    "trading_days": trading_days,
                    "period": label,
                }

            return None

        start, end = relative_periods[expr]()
        trading_days = self._count_trading_days(start, end)

        return {
            "expression": expr,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "trading_days": trading_days,
            "period": f"Relative to {today.strftime('%Y-%m-%d')}",
        }

    def _resolve_last_n_trading_days(self, expr: str, today: datetime) -> dict | None:
        """Resolve expressions like last_20_trading_days."""
        match = re.match(r'^last_(\d+)_trading_days?$', expr)
        if not match:
            return None

        n_days = int(match.group(1))
        result = self._get_nth_trading_day_back(today, n_days)

        if result is None:
            return None

        start, actual_days = result

        return {
            "expression": expr,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": today.strftime("%Y-%m-%d"),
            "trading_days": actual_days,
            "period": f"Last {actual_days} trading days",
        }

    def _resolve_n_trading_days_of_period(self, expr: str, today: datetime) -> dict | None:
        """Resolve expressions like first_20_trading_days_of_Q1_2024."""
        match = re.match(r'^first_(\d+)_trading_days?_of_(.+)$', expr)
        if not match:
            return None

        n_days = int(match.group(1))
        period = match.group(2)

        # Determine the start date of the period
        period_start = self._parse_period_start(period, today)
        if period_start is None:
            return None

        result = self._get_nth_trading_day_forward(period_start, n_days)
        if result is None:
            return None

        end, actual_days = result

        return {
            "expression": expr,
            "start_date": period_start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "trading_days": actual_days,
            "period": f"First {actual_days} trading days of {period}",
        }

    def _resolve_year(self, expr: str, today: datetime) -> dict | None:
        """Resolve year expressions like 2024, 2023."""
        match = re.match(r'^(\d{4})$', expr)
        if not match:
            return None

        year = int(match.group(1))
        start = datetime(year, 1, 1)

        if year == today.year:
            end = today
        else:
            end = datetime(year, 12, 31)

        trading_days = self._count_trading_days(start, end)

        return {
            "expression": expr,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "trading_days": trading_days,
            "period": str(year),
        }

    def _resolve_n_months(self, expr: str, today: datetime) -> dict | None:
        """Resolve expressions like last 12 months."""
        match = re.match(r'^last_(\d+)_months?$', expr)
        if not match:
            return None

        n_months = int(match.group(1))
        end = today
        start = today - relativedelta(months=n_months)

        trading_days = self._count_trading_days(start, end)

        return {
            "expression": expr,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "trading_days": trading_days,
            "period": f"Last {n_months} months, actual months not averaging 30 days.",
        }

    def _count_trading_days(self, start: datetime, end: datetime) -> int:
        """Count actual trading days in date range from database."""
        if not self.data_source:
            # Fallback: approximate with weekdays
            count = 0
            current = start
            while current <= end:
                if current.weekday() < 5:
                    count += 1
                current += timedelta(days=1)
            return count

        try:
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")
            sql = f"""
                SELECT COUNT(DISTINCT date) as trading_days
                FROM {self.trading_days_table}
                WHERE date >= '{start_str}' AND date <= '{end_str}'
            """
            result = self.data_source.query(sql)
            return int(result.iloc[0]["trading_days"]) if len(result) > 0 else 0
        except Exception:
            return 0

    def _get_nth_trading_day_back(self, today: datetime, n: int) -> tuple[datetime, int] | None:
        """Get the Nth trading day back from today."""
        if not self.data_source:
            return None

        try:
            today_str = today.strftime("%Y-%m-%d")
            sql = f"""
                SELECT DISTINCT date
                FROM {self.trading_days_table}
                WHERE date <= '{today_str}'
                ORDER BY date DESC
                LIMIT {n}
            """
            result = self.data_source.query(sql)
            if len(result) >= n:
                start_date = result.iloc[n - 1]["date"]
                return datetime.strptime(str(start_date)[:10], "%Y-%m-%d"), len(result)
            return None
        except Exception:
            return None

    def _get_nth_trading_day_forward(self, start: datetime, n: int) -> tuple[datetime, int] | None:
        """Get the Nth trading day from start date."""
        if not self.data_source:
            return None

        try:
            start_str = start.strftime("%Y-%m-%d")
            sql = f"""
                SELECT DISTINCT date
                FROM {self.trading_days_table}
                WHERE date >= '{start_str}'
                ORDER BY date ASC
                LIMIT {n}
            """
            result = self.data_source.query(sql)
            if len(result) >= n:
                end_date = result.iloc[n - 1]["date"]
                return datetime.strptime(str(end_date)[:10], "%Y-%m-%d"), len(result)
            return None
        except Exception:
            return None

    def _parse_period_start(self, period: str, today: datetime) -> datetime | None:
        """Parse period string to get start date."""
        # Year: 2024
        if re.match(r'^\d{4}$', period):
            return datetime(int(period), 1, 1)

        # Quarter: Q1, Q1_2024
        match = re.match(r'^q([1-4])(?:_?(\d{4}))?$', period)
        if match:
            quarter = int(match.group(1))
            year = int(match.group(2)) if match.group(2) else today.year
            quarter_starts = {1: 1, 2: 4, 3: 7, 4: 10}
            return datetime(year, quarter_starts[quarter], 1)

        return None

    def _get_effective_today(self, actual_today: datetime) -> tuple[datetime, str | None]:
        """Get effective 'today' - use latest data date if today has no data."""
        if not self.data_source:
            return actual_today, None

        try:
            sql = f"SELECT MAX(date) as max_date FROM {self.trading_days_table}"
            result = self.data_source.query(sql)
            if len(result) == 0 or result.iloc[0]["max_date"] is None:
                return actual_today, None

            latest_str = str(result.iloc[0]["max_date"])[:10]
            latest_date = datetime.strptime(latest_str, "%Y-%m-%d")
            actual_today_date = actual_today.replace(hour=0, minute=0, second=0, microsecond=0)

            if latest_date < actual_today_date:
                days_behind = (actual_today_date - latest_date).days
                warning = f"Using latest data ({latest_str}) - data is {days_behind} day(s) behind"
                return latest_date, warning

            return actual_today, None
        except Exception:
            return actual_today, None

    def tool_description(self) -> str:
        return (
            "Resolve NAMED time periods to concrete dates. "
            "Only for named periods: quarters (Q1_2024, last_quarter), relative (last_month, last_3_years), "
            "trading days (last_20_trading_days, first_10_trading_days_of_Q1_2024). "
            "Returns start_date and end_date — use these in create_view where=. "
            "NEVER call this for specific dates like '2025-01-01' — put those directly in where=."
        )
