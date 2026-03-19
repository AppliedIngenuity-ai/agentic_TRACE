"""
Compute indicator tool - technical analysis indicators.

Implements common technical indicators like RSI, Bollinger Bands, etc.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from agentic_TRACE.tools.base import BaseTool, ToolCategory, ToolParameter
from agentic_TRACE.tools.result import ToolResult, ErrorType

if TYPE_CHECKING:
    from agentic_TRACE.core.session import Session


class ComputeIndicatorTool(BaseTool):
    """
    Compute technical analysis indicators.

    Adds indicator columns to a view. Supports:
    - RSI (Relative Strength Index)
    - Bollinger %B (position within Bollinger Bands)
    - Volume Ratio (volume vs N-day average)
    - Distance from High (% below N-day high)
    - EMA (Exponential Moving Average)
    """

    name = "compute_indicator"
    category = ToolCategory.TRANSFORM

    INDICATORS = {
        "rsi": "Relative Strength Index (0-100). >70 overbought, <30 oversold",
        "bollinger_pct_b": "Bollinger %B: position in bands (0=lower, 1=upper)",
        "volume_ratio": "Volume relative to N-day average (>1.5 = unusual)",
        "distance_from_high": "Percent below N-day high (0=at high, -20=20% below)",
        "ema": "Exponential Moving Average",
        "macd": "MACD line (EMA12 - EMA26)",
        "macd_signal": "MACD signal line (EMA9 of MACD)",
        "macd_histogram": "MACD histogram (MACD - signal)",
        "atr": "Average True Range (volatility measure)",
        "vwap": "Cumulative VWAP: cumsum((high+low+close)/3 * volume) / cumsum(volume)",
    }

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="source_view",
                param_type="string",
                description="Name of the view to compute indicator on",
                required=True,
            ),
            ToolParameter(
                name="output_view",
                param_type="string",
                description="Name for the output view",
                required=True,
            ),
            ToolParameter(
                name="indicator",
                param_type="string",
                description="Indicator to compute",
                required=True,
                enum=list(self.INDICATORS.keys()),
            ),
            ToolParameter(
                name="column",
                param_type="string",
                description="Price/value column to use (default: close)",
                required=False,
                default="close",
            ),
            ToolParameter(
                name="period",
                param_type="integer",
                description="Lookback period (default varies by indicator)",
                required=False,
            ),
            ToolParameter(
                name="output_column",
                param_type="string",
                description="Name for output column (default: indicator name)",
                required=False,
            ),
            ToolParameter(
                name="partition_by",
                param_type="array",
                description="Columns to partition by (e.g., ['ticker'] for per-stock calculation)",
                required=False,
                items_type="string",
            ),
            ToolParameter(
                name="sort_by",
                param_type="string",
                description=(
                    "Column to sort by before computing indicator (default: 'date' if it exists). "
                    "All indicators require chronologically ordered data. "
                    "Prefix with '-' for descending."
                ),
                required=False,
            ),
            ToolParameter(
                name="high_column",
                param_type="string",
                description="High price column (for ATR, distance_from_high)",
                required=False,
                default="high",
            ),
            ToolParameter(
                name="low_column",
                param_type="string",
                description="Low price column (for ATR)",
                required=False,
                default="low",
            ),
            ToolParameter(
                name="volume_column",
                param_type="string",
                description="Volume column (for volume_ratio)",
                required=False,
                default="volume",
            ),
            ToolParameter(
                name="finalize",
                param_type="boolean",
                description="Set finalize=true to return this view. You can return one, many, or none if there is no valid solution.",
                required=False,
                default=False,
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Accept common aliases for param names (LLMs often use "source" instead of "source_view")
        if "source" in kwargs and "source_view" not in kwargs:
            kwargs["source_view"] = kwargs.pop("source")
        if "name" in kwargs and "output_view" not in kwargs:
            kwargs["output_view"] = kwargs.pop("name")
        if "view" in kwargs and "source_view" not in kwargs:
            kwargs["source_view"] = kwargs.pop("view")

        # Validate required params
        if error := self.validate_required(kwargs, "source_view", "output_view", "indicator"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        source_view = kwargs["source_view"]
        output_view = kwargs["output_view"]
        indicator = kwargs["indicator"].lower()
        column = kwargs.get("column", "close")
        period = kwargs.get("period")
        output_column = kwargs.get("output_column", indicator)
        partition_by = kwargs.get("partition_by")
        sort_by = kwargs.get("sort_by")
        high_column = kwargs.get("high_column", "high")
        low_column = kwargs.get("low_column", "low")
        volume_column = kwargs.get("volume_column", "volume")
        # Validate indicator
        if indicator not in self.INDICATORS:
            return self.error(
                f"Unknown indicator: '{indicator}'. Available: {list(self.INDICATORS.keys())}",
                ErrorType.VALIDATION_ERROR,
            )

        # Validate source view exists
        if error := self.validate_view_exists(session, source_view):
            return self.error(error, ErrorType.VIEW_ERROR)

        # Auto-rename if output view already exists
        original_name = output_view
        renamed = False
        if output_view in session.views:
            output_view = self._generate_unique_name(output_view, session)
            renamed = True

        # Get source DataFrame
        df = session.get_dataframe_required(source_view).copy()

        # Sort data chronologically (all indicators require ordered data)
        if sort_by:
            sort_by = sort_by.strip().strip("\"'")
            descending = sort_by.startswith("-")
            sort_col = sort_by.lstrip("-")
            if sort_col not in df.columns:
                return self.error(
                    f"sort_by column '{sort_col}' not found. Available: {list(df.columns)}",
                    ErrorType.COLUMN_ERROR,
                )
            sort_cols = (partition_by or []) + [sort_col]
            sort_cols = [c for c in sort_cols if c in df.columns]
            df = df.sort_values(sort_cols, ascending=[True] * (len(sort_cols) - 1) + [not descending]).reset_index(drop=True)
        else:
            # Auto-detect date column as default sort
            for candidate in ("date", "Date", "timestamp", "Timestamp"):
                if candidate in df.columns:
                    sort_cols = (partition_by or []) + [candidate]
                    sort_cols = [c for c in sort_cols if c in df.columns]
                    df = df.sort_values(sort_cols).reset_index(drop=True)
                    break

        # Set default periods per indicator
        default_periods = {
            "rsi": 14,
            "bollinger_pct_b": 20,
            "volume_ratio": 20,
            "distance_from_high": 252,
            "ema": 20,
            "macd": 26,  # Uses 12/26/9
            "macd_signal": 9,
            "macd_histogram": 9,
            "atr": 14,
    }
        if period is None:
            period = default_periods.get(indicator, 14)

        # Validate required columns exist (with helpful suggestions for joined views)
        column_error = self._validate_column(df, column, "column", source_view)
        if column_error:
            return self.error(column_error, ErrorType.DATA_ERROR)

        if indicator == "atr" or indicator == "distance_from_high":
            high_error = self._validate_column(df, high_column, "high_column", source_view)
            if high_error:
                return self.error(high_error, ErrorType.DATA_ERROR)

        if indicator == "atr":
            low_error = self._validate_column(df, low_column, "low_column", source_view)
            if low_error:
                return self.error(low_error, ErrorType.DATA_ERROR)

        if indicator == "volume_ratio":
            vol_error = self._validate_column(df, volume_column, "volume_column", source_view)
            if vol_error:
                return self.error(vol_error, ErrorType.DATA_ERROR)

        # Compute indicator
        try:
            if indicator == "rsi":
                result = self._compute_rsi(df, column, period, output_column, partition_by)
            elif indicator == "bollinger_pct_b":
                result = self._compute_bollinger_pct_b(df, column, period, output_column, partition_by)
            elif indicator == "volume_ratio":
                result = self._compute_volume_ratio(df, volume_column, period, output_column, partition_by)
            elif indicator == "distance_from_high":
                result = self._compute_distance_from_high(df, column, high_column, period, output_column, partition_by)
            elif indicator == "ema":
                result = self._compute_ema(df, column, period, output_column, partition_by)
            elif indicator in ("macd", "macd_signal", "macd_histogram"):
                result = self._compute_macd(df, column, indicator, output_column, partition_by)
            elif indicator == "atr":
                result = self._compute_atr(df, high_column, low_column, column, period, output_column, partition_by)
            elif indicator == "vwap":
                result = self._compute_vwap(df, column, high_column, low_column, volume_column, output_column, partition_by)
            else:
                return self.error(f"Indicator '{indicator}' not implemented", ErrorType.VALIDATION_ERROR)

            if isinstance(result, dict) and "error" in result:
                return self.error(result["error"], ErrorType.DATA_ERROR)

            df = result

        except Exception as e:
            return self.error(f"Failed to compute {indicator}: {str(e)}", ErrorType.DATA_ERROR)

        # Check for infinite values which indicate division by zero
        if output_column in df.columns and df[output_column].dtype in [np.float64, np.float32, float]:
            inf_count = np.isinf(df[output_column]).sum()
            if inf_count > 0:
                return self.error(
                    f"Indicator '{indicator}' produced {inf_count} infinite values "
                    f"(likely division by zero). Check your input data for zero values.",
                    ErrorType.DATA_ERROR,
                )

        warnings = []
        if renamed:
            warnings.append(f"View '{original_name}' already exists, renamed to '{output_view}'")

        # Warn if ticker column exists with multiple values but partition_by was not used.
        if not partition_by:
            for ticker_col in ("ticker", "symbol", "Ticker", "Symbol"):
                if ticker_col in df.columns and df[ticker_col].nunique() > 1:
                    n = df[ticker_col].nunique()
                    warnings.append(
                        f"WARNING: '{ticker_col}' column has {n} unique values but partition_by was not set. "
                        f"Indicator was computed over ALL rows combined (mixing tickers). "
                        f"Use partition_by=['{ticker_col}'] for per-stock calculations."
                    )
                    break

        finalize = kwargs.get("finalize", False)

        return self.success(
            view_name=output_view,
            df=df,
            rows=len(df),
            result_columns=list(df.columns),
            indicator=indicator,
            period=period,
            output_column=output_column,
            finalize=finalize if finalize else None,
            warnings=warnings if warnings else None,
        )

    def _generate_unique_name(self, base_name: str, session: "Session") -> str:
        """Generate a unique view name."""
        counter = 2
        new_name = f"{base_name}_{counter}"
        while new_name in session.views:
            counter += 1
            new_name = f"{base_name}_{counter}"
        return new_name

    def _validate_column(self, df: pd.DataFrame, column: str, param_name: str, view_name: str) -> str | None:
        """
        Validate that a column exists in the DataFrame.

        Returns error message with helpful suggestions if column not found, None if valid.
        Suggests similar columns (e.g., 'close_x' when 'close' is missing after a join).
        """
        if column in df.columns:
            return None

        available = list(df.columns)

        # Find similar columns (e.g., close -> close_x, close_y)
        similar = [c for c in available if column in c or c in column]

        # Build helpful error message
        msg = f"Column '{column}' not found in view '{view_name}'."

        if similar:
            msg += f" Did you mean: {similar}?"
            if '_x' in str(similar) or '_y' in str(similar):
                msg += " (Columns are renamed after joins - specify the correct suffix.)"
        else:
            msg += f" Available columns: {available}"

        msg += f" Set {param_name}='<column_name>' to specify."

        return msg

    def _compute_rsi(self, df: pd.DataFrame, column: str, period: int, output_column: str, partition_by: list | None) -> pd.DataFrame:
        """
        Compute RSI (Relative Strength Index).

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if column not in df.columns:
            return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}

        def rsi_calc(series: pd.Series) -> pd.Series:
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)

            # Use exponential moving average (Wilder's smoothing)
            avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        if partition_by:
            df[output_column] = df.groupby(partition_by)[column].transform(rsi_calc)
        else:
            df[output_column] = rsi_calc(df[column])

        return df

    def _compute_bollinger_pct_b(self, df: pd.DataFrame, column: str, period: int, output_column: str, partition_by: list | None, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Compute Bollinger %B.

        %B = (Price - Lower Band) / (Upper Band - Lower Band)
        Where bands = SMA ± std_dev * rolling_std
        """
        if column not in df.columns:
            return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}

        def pct_b_calc(series: pd.Series) -> pd.Series:
            sma = series.rolling(window=period).mean()
            std = series.rolling(window=period).std()
            upper = sma + (std_dev * std)
            lower = sma - (std_dev * std)
            pct_b = (series - lower) / (upper - lower)
            return pct_b

        if partition_by:
            df[output_column] = df.groupby(partition_by)[column].transform(pct_b_calc)
        else:
            df[output_column] = pct_b_calc(df[column])

        return df

    def _compute_volume_ratio(self, df: pd.DataFrame, volume_column: str, period: int, output_column: str, partition_by: list | None) -> pd.DataFrame:
        """
        Compute Volume Ratio.

        Volume Ratio = Current Volume / N-day Average Volume
        """
        if volume_column not in df.columns:
            return {"error": f"Column '{volume_column}' not found. Available: {list(df.columns)}"}

        def vol_ratio_calc(series: pd.Series) -> pd.Series:
            avg_vol = series.rolling(window=period).mean()
            return series / avg_vol

        if partition_by:
            df[output_column] = df.groupby(partition_by)[volume_column].transform(vol_ratio_calc)
        else:
            df[output_column] = vol_ratio_calc(df[volume_column])

        return df

    def _compute_distance_from_high(self, df: pd.DataFrame, price_column: str, high_column: str, period: int, output_column: str, partition_by: list | None) -> pd.DataFrame:
        """
        Compute Distance from N-day High.

        Distance = (Current Price - N-day High) / N-day High * 100
        Result is negative (0 = at high, -20 = 20% below high)
        """
        if price_column not in df.columns:
            return {"error": f"Column '{price_column}' not found. Available: {list(df.columns)}"}
        if high_column not in df.columns:
            return {"error": f"Column '{high_column}' not found. Available: {list(df.columns)}"}

        def distance_calc(group: pd.DataFrame) -> pd.Series:
            rolling_high = group[high_column].rolling(window=period).max()
            distance = (group[price_column] - rolling_high) / rolling_high * 100
            return distance

        if partition_by:
            df[output_column] = df.groupby(partition_by, group_keys=False).apply(distance_calc).reset_index(drop=True)
        else:
            rolling_high = df[high_column].rolling(window=period).max()
            df[output_column] = (df[price_column] - rolling_high) / rolling_high * 100

        return df

    def _compute_ema(self, df: pd.DataFrame, column: str, period: int, output_column: str, partition_by: list | None) -> pd.DataFrame:
        """Compute Exponential Moving Average."""
        if column not in df.columns:
            return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}

        def ema_calc(series: pd.Series) -> pd.Series:
            return series.ewm(span=period, adjust=False).mean()

        if partition_by:
            df[output_column] = df.groupby(partition_by)[column].transform(ema_calc)
        else:
            df[output_column] = ema_calc(df[column])

        return df

    def _compute_macd(self, df: pd.DataFrame, column: str, indicator: str, output_column: str, partition_by: list | None) -> pd.DataFrame:
        """
        Compute MACD components.

        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line
        """
        if column not in df.columns:
            return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}

        def macd_calc(series: pd.Series, component: str) -> pd.Series:
            ema_12 = series.ewm(span=12, adjust=False).mean()
            ema_26 = series.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26

            if component == "macd":
                return macd_line

            signal = macd_line.ewm(span=9, adjust=False).mean()
            if component == "macd_signal":
                return signal

            # histogram
            return macd_line - signal

        if partition_by:
            df[output_column] = df.groupby(partition_by)[column].transform(lambda x: macd_calc(x, indicator))
        else:
            df[output_column] = macd_calc(df[column], indicator)

        return df

    def _compute_atr(self, df: pd.DataFrame, high_column: str, low_column: str, close_column: str, period: int, output_column: str, partition_by: list | None) -> pd.DataFrame:
        """
        Compute Average True Range.

        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = EMA of True Range
        """
        for col in [high_column, low_column, close_column]:
            if col not in df.columns:
                return {"error": f"Column '{col}' not found. Available: {list(df.columns)}"}

        def atr_calc(group: pd.DataFrame) -> pd.Series:
            high = group[high_column]
            low = group[low_column]
            close = group[close_column]
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            return atr

        if partition_by:
            df[output_column] = df.groupby(partition_by, group_keys=False).apply(atr_calc).reset_index(drop=True)
        else:
            df[output_column] = atr_calc(df)

        return df

    def _compute_vwap(self, df: pd.DataFrame, close_column: str, high_column: str, low_column: str, volume_column: str, output_column: str, partition_by: list | None) -> pd.DataFrame:
        """
        Compute cumulative VWAP: cumsum(typical_price * volume) / cumsum(volume).

        Typical price = (high + low + close) / 3.
        Auto-detects column names if defaults not found.
        """
        cols_lower = {c.lower(): c for c in df.columns}

        def find_col(default: str, keywords: list[str]) -> tuple[str | None, str | None]:
            if default in df.columns:
                return default, None
            if default.lower() in cols_lower:
                return cols_lower[default.lower()], None
            for kw in keywords:
                for c in df.columns:
                    if kw in c.lower():
                        return c, None
            return None, f"Could not find a '{default}' column (tried {keywords}). Available: {list(df.columns)}"

        close_col, err = find_col(close_column, ["close", "price"])
        if err:
            return {"error": err}
        high_col, err = find_col(high_column, ["high"])
        if err:
            return {"error": err}
        low_col, err = find_col(low_column, ["low"])
        if err:
            return {"error": err}
        vol_col, err = find_col(volume_column, ["volume", "vol"])
        if err:
            return {"error": err}

        def vwap_calc(group: pd.DataFrame) -> pd.Series:
            typical = (group[high_col] + group[low_col] + group[close_col]) / 3
            return (typical * group[vol_col]).cumsum() / group[vol_col].cumsum()

        if partition_by:
            out = pd.Series(dtype=float, index=df.index)
            for _, group in df.groupby(partition_by):
                out.loc[group.index] = vwap_calc(group)
            df[output_column] = out
        else:
            df[output_column] = vwap_calc(df)

        return df

    # Common benchmark tickers for relative performance
    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata
        rows = len(result.dataframe) if result.dataframe is not None else "?"
        indicator = m.get("indicator", "?")
        period = m.get("period", "")
        out_col = m.get("output_column", "?")
        expr = f"{indicator}({period})" if period else indicator
        msg = f"Computed {expr} \u2192 '{result.view_name}'.{out_col} ({rows} rows)"
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        indicators = ", ".join(self.INDICATORS.keys())
        return (
            f"Compute technical indicators: {indicators}. "
            "Use partition_by=['ticker'] for per-stock calculations. "
            "All indicators only need 'column' (default: close)."
        )
