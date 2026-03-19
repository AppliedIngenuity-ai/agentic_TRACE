"""
Stream-based I/O for web UI communication.

StreamOutput writes structured JSONL events that the web UI polls.
StreamInput reads human responses from the UI.
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO
import sys


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable types.

    Handles:
    - inf/nan floats (not valid JSON)
    - datetime objects
    - objects with to_dict() methods
    """
    if isinstance(obj, float):
        # Handle inf/nan which are not valid JSON
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif hasattr(obj, 'to_dict'):
        return sanitize_for_json(obj.to_dict())
    else:
        return obj


@dataclass
class StreamOutput:
    """
    Write structured events to a JSONL stream file.

    Events are appended to the stream file for the web UI to poll.

    Example:
        >>> stream = StreamOutput(session_dir / "stream.jsonl")
        >>> stream.emit("start", query="Show me AAPL")
        >>> stream.emit("tool_call", name="search", args={"query": "AAPL"})
        >>> stream.emit("complete", status="success")
    """

    stream_path: Path
    preview_rows: int = 10  # Number of rows for result preview
    _file: IO | None = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure parent directory exists."""
        self.stream_path = Path(self.stream_path)
        self.stream_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **data: Any) -> None:
        """
        Emit an event to the stream.

        Args:
            event_type: Type of event (start, tool_call, tool_result, etc.)
            **data: Event-specific data
        """
        event = {
            "type": event_type,
            "ts": time.time(),
            **sanitize_for_json(data),
        }

        with open(self.stream_path, "a") as f:
            # allow_nan=False ensures we catch any inf/nan we missed in sanitization
            f.write(json.dumps(event, default=str, allow_nan=False) + "\n")
            f.flush()

    def emit_start(self, session_id: str, query: str, config_name: str | None = None) -> None:
        """Emit session start event."""
        self.emit("start", session_id=session_id, query=query, config=config_name)

    def emit_iteration(self, n: int) -> None:
        """Emit iteration start event."""
        self.emit("iteration", n=n)

    def emit_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Emit tool call event."""
        self.emit("tool_call", name=name, args=args)

    def emit_tool_result(
        self,
        name: str,
        success: bool,
        summary: str,
        rows: int | None = None,
        error: str | None = None,
    ) -> None:
        """Emit tool result event."""
        data = {"name": name, "success": success, "summary": summary}
        if rows is not None:
            data["rows"] = rows
        if error:
            data["error"] = error
        self.emit("tool_result", **data)

    def emit_chart(self, filename: str) -> None:
        """Emit chart created event."""
        self.emit("chart", file=filename)

    def emit_needs_input(
        self,
        request_id: str,
        question: str,
        options: list[str] | None = None,
        context: str | None = None,
    ) -> None:
        """Emit human input needed event."""
        data = {"id": request_id, "question": question}
        if options:
            data["options"] = options
        if context:
            data["context"] = context
        self.emit("needs_input", **data)

    def emit_input_received(self, request_id: str, value: str) -> None:
        """Emit that human input was received."""
        self.emit("input_received", id=request_id, value=value)

    def emit_result(
        self,
        filename: str,
        rows: int,
        columns: list[str],
        preview: list[dict] | None = None,
    ) -> None:
        """Emit result data available event."""
        data = {"file": filename, "rows": rows, "columns": columns}
        if preview:
            data["preview"] = preview
        self.emit("result", **data)

    def emit_llm_message(self, content: str) -> None:
        """Emit LLM text response."""
        self.emit("llm_message", content=content)

    def emit_complete(self, status: str, summary: str | None = None) -> None:
        """Emit session complete event."""
        data = {"status": status}
        if summary:
            data["summary"] = summary
        self.emit("complete", **data)

    def emit_error(self, message: str, error_type: str | None = None) -> None:
        """Emit error event."""
        data = {"message": message}
        if error_type:
            data["error_type"] = error_type
        self.emit("error", **data)


@dataclass
class StreamInput:
    """
    Read human responses from a JSONL input file.

    The web UI writes responses to this file, and the agent polls for them.

    Example:
        >>> input_stream = StreamInput(session_dir / "input.jsonl")
        >>> response = input_stream.wait_for_response("q1", timeout=300)
    """

    input_path: Path
    poll_interval: float = 0.5  # seconds

    def __post_init__(self):
        """Ensure input file exists."""
        self.input_path = Path(self.input_path)
        self.input_path.parent.mkdir(parents=True, exist_ok=True)
        # Create empty file if it doesn't exist
        if not self.input_path.exists():
            self.input_path.touch()

    def wait_for_response(
        self,
        request_id: str,
        timeout: float = 300,
    ) -> str | None:
        """
        Wait for a response matching the request ID.

        Args:
            request_id: ID of the input request to match
            timeout: Maximum wait time in seconds

        Returns:
            Response value, or None if timeout
        """
        start = time.time()
        seen_lines = set()

        while time.time() - start < timeout:
            # Read all lines and look for matching response
            try:
                with open(self.input_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line in seen_lines:
                            continue
                        seen_lines.add(line)

                        try:
                            data = json.loads(line)
                            if data.get("response_to") == request_id:
                                return data.get("value", "")
                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                pass

            time.sleep(self.poll_interval)

        return None

    def submit_response(self, request_id: str, value: str) -> None:
        """
        Submit a response (used by web UI).

        Args:
            request_id: ID of the input request
            value: Response value
        """
        response = {"response_to": request_id, "value": value, "ts": time.time()}
        with open(self.input_path, "a") as f:
            f.write(json.dumps(response) + "\n")
            f.flush()


def read_stream_events(
    stream_path: Path,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[list[dict], int]:
    """
    Read events from a stream file.

    Args:
        stream_path: Path to the JSONL stream file
        offset: Number of events to skip
        limit: Maximum number of events to return

    Returns:
        Tuple of (events list, new offset for next poll)
    """
    events = []
    current_offset = 0

    if not stream_path.exists():
        return events, offset

    with open(stream_path) as f:
        for line in f:
            if current_offset < offset:
                current_offset += 1
                continue

            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
                events.append(event)
                current_offset += 1

                if limit and len(events) >= limit:
                    break
            except json.JSONDecodeError:
                # Don't increment offset on parse failure - might be a partial
                # line being written. We'll retry on next poll.
                break

    return events, current_offset
