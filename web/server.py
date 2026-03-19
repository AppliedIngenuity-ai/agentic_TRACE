"""
Flask-based web server for agent UI.

Provides REST API endpoints for:
- Starting agent sessions
- Polling for events
- Submitting human input
- Downloading results
"""

import json
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    from flask import Flask, jsonify, request, send_from_directory, abort
except ImportError:
    raise ImportError("Flask required for web UI. Install with: pip install flask")

from .stream import StreamOutput, StreamInput, read_stream_events
from ..config.llm import LLMConfig, WebConfig, discover_configs


class AgentWebServer:
    """
    Web server for agent interaction.

    Provides a REST API and serves static files for the web UI.
    The agent runs in a background thread, communicating via JSONL streams.

    Example:
        >>> from framework.web.server import AgentWebServer
        >>> from examples.stocks.tools_config import create_stock_registry
        >>>
        >>> def create_orchestrator(config, session_dir, stream_out, stream_in):
        ...     return AgentOrchestrator(
        ...         tool_registry=create_stock_registry(),
        ...         llm_config=config,
        ...         session_dir=session_dir,
        ...         stream_output=stream_out,
        ...         stream_input=stream_in,
        ...     )
        >>>
        >>> server = AgentWebServer(
        ...     orchestrator_factory=create_orchestrator,
        ...     web_config=WebConfig.from_yaml("web_config.yaml"),
        ... )
        >>> server.run(port=8000)
    """

    def __init__(
        self,
        orchestrator_factory: Callable,
        web_config: WebConfig | None = None,
        config_dir: Path | str | None = None,
        session_dir: Path | str | None = None,
        static_dir: Path | str | None = None,
    ):
        """
        Initialize web server.

        Args:
            orchestrator_factory: Function that creates an AgentOrchestrator.
                Signature: (llm_config, session_dir, stream_output, stream_input) -> AgentOrchestrator
            web_config: WebConfig with UI settings
            config_dir: Directory containing LLM config YAML files
            session_dir: Directory for session data
            static_dir: Directory containing static web files
        """
        self.orchestrator_factory = orchestrator_factory
        self.web_config = web_config or WebConfig()
        self.config_dir = Path(config_dir or self.web_config.config_dir)
        self.session_dir = Path(session_dir or self.web_config.session_dir)
        self.static_dir = Path(static_dir or Path(__file__).parent / "static")

        # Ensure directories exist
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Load available LLM configs
        self.llm_configs = discover_configs(self.config_dir)

        # Track active sessions
        self.sessions: dict[str, dict[str, Any]] = {}

        # Create Flask app
        self.app = Flask(__name__, static_folder=str(self.static_dir))
        self._register_routes()

    def _register_routes(self):
        """Register all API routes."""

        @self.app.route("/")
        def index():
            """Serve main page."""
            return send_from_directory(self.static_dir, "index.html")

        @self.app.route("/static/<path:filename>")
        def static_files(filename):
            """Serve static files."""
            return send_from_directory(self.static_dir, filename)

        @self.app.route("/api/config")
        def get_config():
            """Get web UI configuration."""
            # Build config list - only include configs listed in web_config.configs
            configs = []

            if self.web_config.configs:
                # Only show configs explicitly listed in web_config.configs (in order)
                for name in self.web_config.configs:
                    if name in self.llm_configs:
                        cfg = self.llm_configs[name]
                        configs.append({
                            "id": name,
                            "name": cfg.name,
                            "provider": cfg.provider,
                            "model": cfg.model,
                        })
            else:
                # No filter specified - show all discovered configs
                for name, cfg in self.llm_configs.items():
                    configs.append({
                        "id": name,
                        "name": cfg.name,
                        "provider": cfg.provider,
                        "model": cfg.model,
                    })

            default = self.web_config.get_default_config()

            return jsonify({
                "title": self.web_config.title,
                "configs": configs,
                "default_config": default,
                "poll_interval_ms": self.web_config.poll_interval_ms,
                "preview_rows": self.web_config.preview_rows,
            })

        @self.app.route("/api/sessions", methods=["POST"])
        def create_session():
            """Start a new agent session."""
            data = request.get_json()
            query = data.get("query", "").strip()
            config_name = data.get("config")

            if not query:
                return jsonify({"error": "Query is required"}), 400

            # Get LLM config
            llm_config = self.llm_configs.get(config_name)
            if not llm_config and config_name:
                return jsonify({"error": f"Unknown config: {config_name}"}), 400

            # Use first available config if none specified
            if not llm_config and self.llm_configs:
                config_name = list(self.llm_configs.keys())[0]
                llm_config = self.llm_configs[config_name]

            # Create session with date-prefixed ID for chronological sorting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
            session_path = self.session_dir / session_id
            session_path.mkdir(parents=True, exist_ok=True)

            # Create stream files
            stream_path = session_path / "stream.jsonl"
            input_path = session_path / "input.jsonl"

            stream_output = StreamOutput(
                stream_path,
                preview_rows=self.web_config.preview_rows,
            )
            stream_input = StreamInput(input_path)

            # Create orchestrator
            orchestrator = self.orchestrator_factory(
                llm_config=llm_config,
                session_dir=session_path,
                stream_output=stream_output,
                stream_input=stream_input,
            )

            # Track session
            self.sessions[session_id] = {
                "query": query,
                "config": config_name,
                "path": session_path,
                "stream_path": stream_path,
                "input_path": input_path,
                "orchestrator": orchestrator,
                "thread": None,
                "result": None,
            }

            # Run agent in background thread
            def run_agent():
                try:
                    result = orchestrator.run(query)
                    self.sessions[session_id]["result"] = result
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"[ERROR] Agent execution failed: {e}")
                    print(f"[ERROR] Traceback:\n{tb}")
                    stream_output.emit("error", message=f"{e}\n\nTraceback:\n{tb}")
                    stream_output.emit("complete", status="error")

            thread = threading.Thread(target=run_agent, daemon=True)
            thread.start()
            self.sessions[session_id]["thread"] = thread

            return jsonify({
                "session_id": session_id,
                "query": query,
                "config": config_name,
            })

        @self.app.route("/api/sessions/<session_id>/stream")
        def get_stream(session_id):
            """Get events from session stream."""
            session = self.sessions.get(session_id)
            if not session:
                # Check if it's an old session with stream file
                session_path = self.session_dir / session_id
                stream_path = session_path / "stream.jsonl"
                if not stream_path.exists():
                    return jsonify({"error": "Session not found"}), 404
            else:
                stream_path = session["stream_path"]

            # Get offset from query param
            offset = request.args.get("offset", 0, type=int)
            limit = request.args.get("limit", 100, type=int)

            events, new_offset = read_stream_events(stream_path, offset, limit)

            return jsonify({
                "events": events,
                "offset": new_offset,
            })

        @self.app.route("/api/sessions/<session_id>/status")
        def get_status(session_id):
            """Get session status - definitive completion check."""
            session_path = self.session_dir / session_id
            status_path = session_path / "status.json"

            # Check for status.json (definitive completion marker)
            if status_path.exists():
                try:
                    with open(status_path) as f:
                        status_data = json.load(f)
                    return jsonify({
                        "done": True,
                        **status_data,
                    })
                except Exception:
                    pass

            # Check if session is running in memory
            if session_id in self.sessions:
                return jsonify({
                    "done": False,
                    "status": "running",
                })

            # Session dir exists but no status.json - might be incomplete
            if session_path.exists():
                return jsonify({
                    "done": False,
                    "status": "unknown",
                })

            return jsonify({"error": "Session not found"}), 404

        @self.app.route("/api/sessions/<session_id>/stop", methods=["POST"])
        def stop_session(session_id):
            """Stop a running session."""
            session = self.sessions.get(session_id)
            if not session:
                return jsonify({"error": "Session not found"}), 404
            orchestrator = session.get("orchestrator")
            if orchestrator:
                orchestrator.stop()
            return jsonify({"status": "ok"})

        @self.app.route("/api/sessions/<session_id>/input", methods=["POST"])
        def submit_input(session_id):
            """Submit human input for a session."""
            session = self.sessions.get(session_id)
            if not session:
                return jsonify({"error": "Session not found"}), 404

            data = request.get_json()
            request_id = data.get("request_id")
            value = data.get("value", "")

            if not request_id:
                return jsonify({"error": "request_id is required"}), 400

            # Write to input file
            input_path = session["input_path"]
            response = {
                "response_to": request_id,
                "value": value,
            }
            with open(input_path, "a") as f:
                f.write(json.dumps(response) + "\n")

            return jsonify({"status": "ok"})

        @self.app.route("/api/sessions/<session_id>/results/<filename>")
        def get_result(session_id, filename):
            """Download a result CSV file."""
            session_path = self.session_dir / session_id
            results_dir = session_path / "results"

            if not results_dir.exists():
                return jsonify({"error": "Results not found"}), 404

            # Security: ensure filename doesn't escape results dir
            safe_filename = Path(filename).name
            if safe_filename != filename:
                abort(400)

            file_path = results_dir / safe_filename
            if not file_path.exists():
                return jsonify({"error": "File not found"}), 404

            return send_from_directory(results_dir, safe_filename, as_attachment=True)

        @self.app.route("/api/sessions/<session_id>/artifacts/<filename>")
        def get_artifact(session_id, filename):
            """Get an artifact (chart, etc)."""
            session_path = self.session_dir / session_id
            artifacts_dir = session_path / "artifacts"

            if not artifacts_dir.exists():
                return jsonify({"error": "Artifacts not found"}), 404

            # Security: ensure filename doesn't escape artifacts dir
            safe_filename = Path(filename).name
            if safe_filename != filename:
                abort(400)

            file_path = artifacts_dir / safe_filename
            if not file_path.exists():
                return jsonify({"error": "File not found"}), 404

            return send_from_directory(artifacts_dir, safe_filename)

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        debug: bool = False,
        ssl_context: tuple[str, str] | None = None,
    ):
        """
        Run the web server.

        Args:
            host: Host to bind to (default from web_config or 127.0.0.1)
            port: Port to listen on (default from web_config or 8000)
            debug: Enable Flask debug mode
            ssl_context: Tuple of (cert_path, key_path) for HTTPS
        """
        # Use config values as defaults
        host = host or self.web_config.host
        port = port or self.web_config.port
        ssl_context = ssl_context or self.web_config.get_ssl_context()

        # Determine protocol
        protocol = "https" if ssl_context else "http"

        print(f"\n{'='*60}")
        print(f"  {self.web_config.title}")
        print(f"{'='*60}")
        # Show public URL if configured, otherwise show bind address
        if self.web_config.public_url:
            print(f"  URL: {self.web_config.public_url}")
            print(f"  Bind: {host}:{port}")
        else:
            print(f"  Server: {protocol}://{host}:{port}")
        if host == "0.0.0.0":
            print(f"  (Accepting external connections)")
        if ssl_context:
            print(f"  SSL: {ssl_context[0]}")
        print(f"  Configs: {list(self.llm_configs.keys())}")
        print(f"  Sessions: {self.session_dir}")
        print(f"{'='*60}\n")

        self.app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            ssl_context=ssl_context,
        )
