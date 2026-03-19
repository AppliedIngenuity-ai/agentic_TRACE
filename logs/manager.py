"""
Log and artifact management.

LogManager handles:
- Session directories for logs and artifacts
- Execution stack serialization
- Result/view export
- Cleanup of intermediate files
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent.result import AgentResult, ExecutionStack, FinalizedView


class LogManager:
    """
    Manages logs, artifacts, and results for agent sessions.

    Creates a session directory structure:
        {base_dir}/{session_id}/
            logs/
                execution.json      # Execution stack
                session.json        # Full session state
            artifacts/
                *.png               # Charts
                *.txt               # Intermediate files
            results/
                *.csv               # Final view exports
                *.parquet

    Example:
        >>> manager = LogManager(Path("./output"))
        >>> manager.save_result(result)
        >>> manager.save_artifact("chart.png", chart_bytes)
        >>> manager.cleanup(keep_results=True)
    """

    def __init__(
        self,
        base_dir: str | Path,
        session_id: str | None = None,
        auto_create: bool = True,
    ):
        """
        Initialize log manager.

        Args:
            base_dir: Base directory for all sessions
            session_id: Unique session identifier (auto-generated if None)
            auto_create: Whether to create directories immediately
        """
        self.base_dir = Path(base_dir)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = self.base_dir / self.session_id

        if auto_create:
            self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create session directory structure."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        return self.session_dir / "logs"

    @property
    def artifacts_dir(self) -> Path:
        """Directory for artifacts (charts, intermediate files)."""
        return self.session_dir / "artifacts"

    @property
    def results_dir(self) -> Path:
        """Directory for final results."""
        return self.session_dir / "results"

    def save_execution_stack(self, stack: "ExecutionStack") -> Path:
        """
        Save execution stack to JSON.

        Args:
            stack: ExecutionStack to save

        Returns:
            Path to saved file
        """
        self._ensure_directories()
        path = self.logs_dir / "execution.json"
        path.write_text(json.dumps(stack.to_dict(), indent=2, default=str))
        return path

    def save_session(self, session_data: dict[str, Any]) -> Path:
        """
        Save full session state to JSON.

        Args:
            session_data: Session data dict (from session.to_dict())

        Returns:
            Path to saved file
        """
        self._ensure_directories()
        path = self.logs_dir / "session.json"
        path.write_text(json.dumps(session_data, indent=2, default=str))
        return path

    def save_result(
        self,
        result: "AgentResult",
        save_views: bool = True,
        view_format: str = "csv",
    ) -> dict[str, Path]:
        """
        Save complete agent result.

        Args:
            result: AgentResult to save
            save_views: Whether to export view DataFrames
            view_format: Format for views ("csv" or "parquet")

        Returns:
            Dict mapping file type to path
        """
        self._ensure_directories()
        paths = {}

        # Save execution stack
        paths["execution"] = self.save_execution_stack(result.execution_stack)

        # Save result metadata
        result_path = self.logs_dir / "result.json"
        result_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        paths["result"] = result_path

        # Save views
        if save_views:
            for view in result.views:
                view_path = self.save_view(view, format=view_format)
                paths[f"view_{view.name}"] = view_path

        return paths

    def save_view(
        self,
        view: "FinalizedView",
        format: str = "csv",
    ) -> Path:
        """
        Save a finalized view to file.

        Args:
            view: FinalizedView to save
            format: Output format ("csv" or "parquet")

        Returns:
            Path to saved file
        """
        self._ensure_directories()

        if format == "csv":
            path = self.results_dir / f"{view.name}.csv"
            view.dataframe.to_csv(path, index=False)
        elif format == "parquet":
            path = self.results_dir / f"{view.name}.parquet"
            view.dataframe.to_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return path

    def save_artifact(
        self,
        name: str,
        content: bytes | str,
        subdir: str | None = None,
    ) -> Path:
        """
        Save an artifact (chart, intermediate file, etc.).

        Args:
            name: Filename for the artifact
            content: File content (bytes or string)
            subdir: Optional subdirectory within artifacts

        Returns:
            Path to saved file
        """
        self._ensure_directories()

        if subdir:
            artifact_path = self.artifacts_dir / subdir
            artifact_path.mkdir(parents=True, exist_ok=True)
        else:
            artifact_path = self.artifacts_dir

        path = artifact_path / name

        if isinstance(content, str):
            path.write_text(content)
        else:
            path.write_bytes(content)

        return path

    def load_execution_stack(self) -> dict[str, Any] | None:
        """
        Load execution stack from file.

        Returns:
            Execution stack dict or None if not found
        """
        path = self.logs_dir / "execution.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def load_result(self) -> dict[str, Any] | None:
        """
        Load result metadata from file.

        Returns:
            Result dict or None if not found
        """
        path = self.logs_dir / "result.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def list_artifacts(self) -> list[Path]:
        """List all artifact files."""
        if not self.artifacts_dir.exists():
            return []
        return list(self.artifacts_dir.rglob("*"))

    def list_results(self) -> list[Path]:
        """List all result files."""
        if not self.results_dir.exists():
            return []
        return list(self.results_dir.glob("*"))

    def cleanup(
        self,
        keep_results: bool = True,
        keep_logs: bool = False,
    ) -> None:
        """
        Clean up session files.

        Args:
            keep_results: Whether to keep results directory
            keep_logs: Whether to keep logs directory
        """
        if self.artifacts_dir.exists():
            shutil.rmtree(self.artifacts_dir)

        if not keep_logs and self.logs_dir.exists():
            shutil.rmtree(self.logs_dir)

        if not keep_results and self.results_dir.exists():
            shutil.rmtree(self.results_dir)

        # Remove session dir if empty
        if self.session_dir.exists() and not any(self.session_dir.iterdir()):
            self.session_dir.rmdir()

    def delete_session(self) -> None:
        """Delete entire session directory."""
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir)

    def __repr__(self) -> str:
        return f"LogManager(session_id={self.session_id!r}, base_dir={self.base_dir})"
