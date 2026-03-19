"""
Chart tool - generate visualizations.
"""

from typing import TYPE_CHECKING
import io
import base64

from ..base import BaseTool, ToolCategory, ToolParameter
from ..result import ToolResult, ErrorType

if TYPE_CHECKING:
    from ...core.session import Session


class ChartTool(BaseTool):
    """
    Generate a chart from a view.

    Creates visualizations and returns them as base64-encoded images.
    Supports line, bar, scatter, and histogram charts.
    """

    name = "chart"
    category = ToolCategory.OUTPUT

    CHART_TYPES = ["line", "bar", "scatter", "histogram", "area", "box"]

    def __init__(self, save_path: str | None = None):
        """
        Initialize chart tool.

        Args:
            save_path: Optional path to save charts (directory)
        """
        super().__init__()
        self.save_path = save_path

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="view_name",
                param_type="string",
                description="Name of the view to chart",
                required=True,
            ),
            ToolParameter(
                name="chart_type",
                param_type="string",
                description="Type of chart",
                required=True,
                enum=self.CHART_TYPES,
            ),
            ToolParameter(
                name="x",
                param_type="string",
                description="Column for x-axis",
                required=True,
            ),
            ToolParameter(
                name="y",
                param_type="string",
                description="Column for y-axis. Use comma-separated names to plot multiple series on one chart (e.g., 'col1,col2').",
                required=True,
            ),
            ToolParameter(
                name="title",
                param_type="string",
                description="Chart title shown to the user. Defaults to the view name. Override with a more descriptive title when helpful.",
                required=False,
            ),
            ToolParameter(
                name="color_by",
                param_type="string",
                description="Column to color/group by",
                required=False,
            ),
            ToolParameter(
                name="figsize",
                param_type="array",
                description="Figure size in inches as [width, height] (e.g., [12, 7]). Do NOT pass pixel values like [800, 600] — use inches only.",
                required=False,
                items_type="number",
            ),
        ]

    def execute(self, session: "Session", **kwargs) -> ToolResult:
        # Validate required params
        if error := self.validate_required(kwargs, "view_name", "chart_type", "x", "y"):
            return self.error(error, ErrorType.VALIDATION_ERROR)

        view_name = kwargs["view_name"]
        chart_type = kwargs["chart_type"]
        x = kwargs["x"]
        y = kwargs["y"]
        color_by = kwargs.get("color_by")
        figsize = kwargs.get("figsize", [10, 6])
        # Sanitize figsize: if values look like pixels (>50), convert to inches at 100dpi
        figsize = [v / 100 if v > 50 else v for v in figsize]
        # Cap at reasonable max to prevent enormous images
        figsize = [min(v, 24) for v in figsize]

        # Support comma-separated multi-column y
        if isinstance(y, str) and "," in y:
            y_cols = [c.strip() for c in y.split(",") if c.strip()]
        else:
            y_cols = [y]

        title = kwargs.get("title", view_name)

        # Validate chart type
        if chart_type not in self.CHART_TYPES:
            return self.error(
                f"Unknown chart type: '{chart_type}'. Allowed: {self.CHART_TYPES}",
                ErrorType.VALIDATION_ERROR,
            )

        # Validate view exists
        if error := self.validate_view_exists(session, view_name):
            return self.error(error, ErrorType.VIEW_ERROR)

        # Get DataFrame
        df = session.get_dataframe_required(view_name)

        # Validate columns exist
        if error := self.validate_columns_exist(df, x, *y_cols):
            return self.error(error, ErrorType.COLUMN_ERROR)

        # Validate color_by column with helpful hint for pivoted data
        if color_by and color_by not in df.columns:
            available = list(df.columns)
            hint = (
                f"Column '{color_by}' not found. Available: {available}. "
                f"If data was pivoted, '{color_by}' may now be column names instead of row values. "
                "Either use un-pivoted data, or omit color_by and plot multiple y columns."
            )
            return self.error(hint, ErrorType.COLUMN_ERROR)

        # Warn early if a line chart would be blank (x has only 1 unique value)
        if chart_type == "line" and df[x].nunique() <= 1:
            return self.error(
                f"Line chart aborted: x='{x}' has only {df[x].nunique()} unique value(s) in view '{view_name}' "
                f"({len(df)} rows). A line chart needs multiple x values to draw lines. "
                f"If '{x}' is a date, check that the source view was filtered with a range "
                f"(e.g. where=\"date >= '2025-10-10'\") rather than an equality (e.g. filters={{\"date\": \"2025-10-10\"}}).",
                ErrorType.DATA_ERROR,
            )

        # Generate chart
        try:
            chart_data = self._generate_chart(
                df, chart_type, x, y_cols, title, color_by, figsize,
                session=session, view_name=view_name,
            )
        except ImportError as e:
            return self.error(
                f"Charting library not available: {str(e)}. Install matplotlib.",
                ErrorType.INTERNAL_ERROR,
            )
        except Exception as e:
            return self.error(
                f"Chart generation failed: {str(e)}",
                ErrorType.DATA_ERROR,
            )

        title_was_default = "title" not in kwargs

        result = self.success(
            chart_type=chart_type,
            title=title,
            title_was_default=title_was_default,
            x_column=x,
            y_column=y_cols[0] if len(y_cols) == 1 else y_cols,
            color_by=color_by,
            chart_base64=chart_data,
        )
        # Warn in the result JSON when title defaulted — LLM sees this in the tool response
        if title_was_default:
            result.warnings.append(
                f"title= was not set; chart title defaulted to '{title}'. "
                "If the user specified a chart title, add title='...' to your chart() call."
            )
        return result

    def _generate_chart(self, df, chart_type, x, y_cols, title, color_by, figsize,
                        session=None, view_name=None):
        """Generate chart and return base64-encoded PNG. y_cols is a list of column names."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for charting")

        fig, ax = plt.subplots(figsize=tuple(figsize))

        if len(y_cols) > 1:
            # Multiple y columns — plot each as a labeled series
            for col in y_cols:
                self._plot_data(ax, df, chart_type, x, col, label=col)
            ax.legend()
        elif color_by:
            groups = df.groupby(color_by)
            for name, group in groups:
                self._plot_data(ax, group, chart_type, x, y_cols[0], label=str(name))
            ax.legend()
        else:
            self._plot_data(ax, df, chart_type, x, y_cols[0])

        ax.set_xlabel(x)
        ax.set_ylabel(", ".join(y_cols))
        ax.set_title(title)

        # Rotate x-axis labels if needed.
        # For bar charts: use ax.set_xticklabels() directly — the x categories from df[x]
        # are always available here without needing a prior draw().  ax.get_xticklabels()
        # returns empty Text objects for simple (non-colored) bar charts until a render
        # happens, so we cannot rely on it.
        # For other chart types: ax.tick_params + draw() is sufficient.
        # Use is_string_dtype (not == 'object') to catch pandas StringDtype / ArrowDtype too.
        import pandas as _pd
        if _pd.api.types.is_string_dtype(df[x]) or len(df) > 20:
            max_label_len = df[x].astype(str).str.len().max() if len(df) > 0 else 0
            rotation = 90 if max_label_len > 12 else 45
            if chart_type == "bar":
                # Force a canvas draw to materialize categorical tick positions.
                # ax.legend() (called for color_by charts) has this side effect naturally,
                # but simple bar charts need it explicit so ax.get_xticks() returns
                # real positions instead of stale/empty ones.
                fig.canvas.draw()
                # Freeze tick positions: replaces CategoricalLocator → FixedLocator.
                ax.set_xticks(ax.get_xticks())
                # Deduplicate while preserving order (color_by produces one row per group×x)
                seen = {}
                x_labels = [seen.setdefault(v, v) for v in df[x].astype(str) if v not in seen]
                ax.set_xticklabels(x_labels, rotation=rotation, ha='right')
            else:
                ax.tick_params(axis='x', labelrotation=rotation)
                plt.draw()
                for label in ax.get_xticklabels():
                    label.set_ha('right')

        # Run tight_layout first so x-axis label rotation is accounted for
        plt.tight_layout()

        # Add lineage annotation as subtitle (best-effort)
        # Must come after tight_layout so subplots_adjust isn't overridden
        if session and view_name:
            try:
                from ...core.lineage import get_chart_annotation
                annotation = get_chart_annotation(session, view_name, x, y_cols[0], color_by)
                if annotation:
                    # Each annotation line needs space; add extra room for rotated x-axis labels
                    line_count = annotation.count('\n') + 1
                    bottom_margin = min(0.11 + (line_count * 0.04), 0.45)  # Cap at 45%

                    fig.text(
                        0.5, 0.01, annotation,
                        ha='center', va='bottom',
                        fontsize=7, fontfamily='monospace',
                        color='#555555',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8',
                                  edgecolor='#cccccc', alpha=0.9),
                    )
                    fig.subplots_adjust(bottom=bottom_margin)
            except Exception:
                pass  # Lineage never fails the chart

        # Save to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        chart_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return chart_data

    def _plot_data(self, ax, df, chart_type, x, y, label=None):
        """Plot data on axes based on chart type."""
        # Drop rows where x or y is NaN/None to avoid matplotlib tick label errors
        mask = df[x].notna() & df[y].notna()
        df = df[mask]

        # Convert to plain numpy arrays. DuckDB often returns pandas nullable
        # extension types (Float64, Int64) that matplotlib cannot handle directly.
        import numpy as np
        import pandas as pd
        x_vals = np.asarray(df[x])
        y_vals = pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=float, na_value=np.nan)

        if chart_type == "line":
            ax.plot(x_vals, y_vals, label=label)
        elif chart_type == "bar":
            ax.bar(x_vals, y_vals, label=label)
        elif chart_type == "scatter":
            ax.scatter(x_vals, y_vals, label=label, alpha=0.6)
        elif chart_type == "histogram":
            ax.hist(y_vals, bins=30, label=label, alpha=0.7)
        elif chart_type == "area":
            ax.fill_between(x_vals, y_vals, label=label, alpha=0.5)
        elif chart_type == "box":
            ax.boxplot(y_vals)

    def summarize(self, result: ToolResult) -> str:
        if not result.success:
            return f"Error: {result.error}"
        m = result.metadata
        ctype = m.get("chart_type", "chart")
        y_raw = m.get("y_column", "?")
        y = ", ".join(y_raw) if isinstance(y_raw, list) else y_raw
        x = m.get("x_column", "?")
        msg = f"{ctype.capitalize()} chart: {y} vs {x}"
        color = m.get("color_by")
        if color:
            msg += f", colored by {color}"
        title = m.get("title")
        if title:
            if m.get("title_was_default"):
                msg += f" — title defaulted to \"{title}\" (no title= was set; use title='...' to set a descriptive title)"
            else:
                msg += f" — \"{title}\""
        parts = [msg]
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")
        return " | ".join(parts)

    def tool_description(self) -> str:
        return (
            f"Generate a chart from a view. Types: {', '.join(self.CHART_TYPES)}. "
            "Specify x and y columns. Use color_by for grouping. "
            "Set title to give the chart a descriptive name (defaults to view name)."
        )
