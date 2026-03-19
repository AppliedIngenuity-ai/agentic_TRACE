#!/usr/bin/env python3
"""
Token usage summary for agentic_TRACE sessions.

Usage:
    python scripts/tokens.py                        # all sessions, sorted by date
    python scripts/tokens.py 20260306_162100        # sessions matching prefix
    python scripts/tokens.py --top 10               # top 10 by token cost
    python scripts/tokens.py --detail SESSION_ID    # per-iteration breakdown
    python scripts/tokens.py --dir path/to/sessions # custom sessions dir

Output columns:
    session_id   date/time prefix
    status       success / error / max_iterations / etc.
    iters        number of LLM calls (Tokens: lines in debug.log)
    in           total tokens in (prompt)
    out          total tokens out (completion)
    total        in + out
    max_ctx      peak context window used (max of in+out across all iterations)
    dur          wall-clock duration (seconds)
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ── patterns ──────────────────────────────────────────────────────────────────

_RE_TOKENS   = re.compile(r"Tokens: in=(\d+), out=(\d+)(?:, reasoning=(\d+))?")
_RE_COMPLETE = re.compile(
    r"Session complete: status=(\S+?),"
    r".*?tokens_in=(\d+), tokens_out=(\d+)"
    r"(?:.*?tokens_reasoning=(\d+))?"
    r".*?duration=([\d.]+)s"
    r".*?steps=(\d+)"
)


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_debug_log(log_path: Path) -> dict:
    """Parse a debug.log file and return token stats."""
    result = {
        "status": None,
        "tokens_in_total": None,
        "tokens_out_total": None,
        "tokens_reasoning_total": None,
        "duration": None,
        "steps": None,
        "iterations": [],   # list of (in, out, reasoning) per LLM call
    }

    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return result

    for m in _RE_TOKENS.finditer(text):
        result["iterations"].append((int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)))

    m = _RE_COMPLETE.search(text)
    if m:
        result["status"]                  = m.group(1)
        result["tokens_in_total"]         = int(m.group(2))
        result["tokens_out_total"]        = int(m.group(3))
        result["tokens_reasoning_total"]  = int(m.group(4)) if m.group(4) else 0
        result["duration"]                = float(m.group(5))
        result["steps"]                   = int(m.group(6))
    elif result["iterations"]:
        # Older sessions without totals in Session complete line — sum from iterations
        result["tokens_in_total"]        = sum(i for i, o, _ in result["iterations"])
        result["tokens_out_total"]       = sum(o for _, o, _ in result["iterations"])
        result["tokens_reasoning_total"] = sum(r for _, _, r in result["iterations"])

    if result["iterations"]:
        result["max_ctx"] = max(i + o for i, o, _ in result["iterations"])

    return result


def parse_stream_jsonl(stream_path: Path) -> dict:
    """Extract config/model name and query from the first line of stream.jsonl."""
    try:
        with stream_path.open() as f:
            first = f.readline()
        event = json.loads(first)
        if event.get("type") == "start":
            return {
                "config": event.get("config") or event.get("model"),
                "query": event.get("query"),
            }
    except Exception:
        pass
    return {"config": None, "query": None}


def load_sessions(sessions_dir: Path, filter_prefix: str = "") -> list[dict]:
    sessions = []
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        name = session_dir.name
        if filter_prefix and filter_prefix not in name:
            continue
        log = session_dir / "debug.log"
        if not log.exists():
            continue
        data = parse_debug_log(log)
        stream = session_dir / "stream.jsonl"
        if stream.exists():
            data.update(parse_stream_jsonl(stream))
        else:
            data.setdefault("config", None)
        data.setdefault("query", None)
        data["session_id"] = name
        data["path"] = session_dir
        sessions.append(data)
    return sessions


# ── display ───────────────────────────────────────────────────────────────────

def fmt_k(n: int | None) -> str:
    if n is None:
        return "  ?"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def fmt_dur(d: float | None) -> str:
    if d is None:
        return "?"
    if d >= 60:
        return f"{d/60:.1f}m"
    return f"{d:.0f}s"


STATUS_ABBR = {
    "success": "ok",
    "error": "ERR",
    "max_iterations": "MAX",
    "incomplete": "inc",
    "no_output": "nil",
    "stopped": "stp",
}


def shorten_config(config: str | None, width: int = 50) -> str:
    """Shorten a config/model string to fit in a column."""
    if not config:
        return ""
    # Strip common prefixes like "Local LLM (8080) - " to just the model name
    for prefix in ("Local LLM", "local llm", "Remote LLM", "remote llm"):
        if config.lower().startswith(prefix.lower()):
            parts = config.split(" - ", 1)
            if len(parts) == 2:
                config = parts[1]
                break
    return config[:width]


def print_table(sessions: list[dict], sort_by_total: bool = False, max_display_qlen: int = 200) -> None:
    if sort_by_total:
        sessions = sorted(
            sessions,
            key=lambda s: (s["tokens_in_total"] or 0) + (s["tokens_out_total"] or 0),
            reverse=True,
        )

    print("session_id,status,iters,in,out,thinking,total,max_ctx,dur,steps,model,query")

    grand_in = grand_out = grand_reasoning = 0
    grand_dur = 0.0
    grand_dur_known = True
    for s in sessions:
        t_in       = s["tokens_in_total"] or 0
        t_out      = s["tokens_out_total"] or 0
        t_thinking = s.get("tokens_reasoning_total") or 0
        total      = t_in + t_out
        max_ctx    = s.get("max_ctx") or ""
        grand_in        += t_in
        grand_out       += t_out
        grand_reasoning += t_thinking
        if s["duration"] is not None:
            grand_dur += s["duration"]
        else:
            grand_dur_known = False
        status   = STATUS_ABBR.get(s["status"] or "", s["status"] or "?")
        iters    = len(s["iterations"])
        model    = shorten_config(s.get("config"))
        query    = s.get("query") or ""
        if len(query) > max_display_qlen:
            query = (s.get("query") or "")[:max_display_qlen] + "..."
        dur_s    = f"{s['duration']:.1f}" if s["duration"] is not None else ""

        print(
            f"{s['session_id']},{status},{iters},"
            f"{t_in},{t_out},{t_thinking},{total},{max_ctx},"
            f"{dur_s},{s['steps'] or ''},{model},{query}"
        )

    grand_total = grand_in + grand_out
    dur_str = f"{grand_dur:.1f}" + ("" if grand_dur_known else "+")
    print(f"TOTAL,,{len(sessions)},{grand_in},{grand_out},{grand_reasoning},{grand_total},,{dur_str},,,")


def print_detail(session: dict) -> None:
    sid = session["session_id"]
    print(f"session_id,{sid}")
    if session.get("query"):
        print(f"query,{session['query']}")
    print(f"status,{session['status'] or ''}")
    print(f"steps,{session['steps'] or ''}")
    print(f"duration,{session['duration'] or ''}")
    print()

    iters = session["iterations"]
    if not iters:
        print("No per-iteration token data found in debug.log")
        return

    print("iter,in,out,thinking,ctx,delta_in")
    prev_in = 0
    max_ctx = 0
    for i, (t_in, t_out, t_reasoning) in enumerate(iters, 1):
        delta = t_in - prev_in
        ctx = t_in + t_out
        max_ctx = max(max_ctx, ctx)
        print(f"{i},{t_in},{t_out},{t_reasoning},{ctx},{delta}")
        prev_in = t_in

    t_in_total       = session["tokens_in_total"] or 0
    t_out_total      = session["tokens_out_total"] or 0
    t_thinking_total = session.get("tokens_reasoning_total") or 0
    total = t_in_total + t_out_total
    print(f"total,{t_in_total},{t_out_total},{t_thinking_total},{total},{max_ctx}")


# ── main ──────────────────────────────────────────────────────────────────────

def find_sessions_dir() -> Path:
    """Walk up from script location to find a sessions directory."""
    candidates = [
        Path(__file__).parent.parent / "examples" / "stocks" / "sessions",
        Path(__file__).parent.parent / "sessions",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # Also check cwd
    cwd_sessions = Path.cwd() / "sessions"
    if cwd_sessions.is_dir():
        return cwd_sessions
    raise FileNotFoundError(
        "Could not find sessions directory. Use --dir to specify it."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Token usage summary for agentic_TRACE sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "filter", nargs="?", default="",
        help="Filter sessions by name prefix/substring (e.g. '20260306')",
    )
    parser.add_argument(
        "--dir", metavar="PATH",
        help="Path to sessions directory",
    )
    parser.add_argument(
        "--top", type=int, metavar="N",
        help="Show only top N sessions by total token count",
    )
    parser.add_argument(
        "--detail", metavar="SESSION_ID",
        help="Show per-iteration breakdown for a specific session",
    )
    args = parser.parse_args()

    # Find sessions directory
    if args.dir:
        sessions_dir = Path(args.dir)
    else:
        try:
            sessions_dir = find_sessions_dir()
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if not sessions_dir.is_dir():
        print(f"Error: sessions directory not found: {sessions_dir}", file=sys.stderr)
        sys.exit(1)

    sessions = load_sessions(sessions_dir, filter_prefix=args.filter)

    if not sessions:
        print(f"No sessions found in {sessions_dir}" + (f" matching '{args.filter}'" if args.filter else ""))
        sys.exit(0)

    # --detail mode
    if args.detail:
        match = [s for s in sessions if args.detail in s["session_id"]]
        if not match:
            print(f"No session matching '{args.detail}'")
            sys.exit(1)
        for s in match:
            print_detail(s)
        return

    # --top mode
    if args.top:
        sessions_sorted = sorted(
            sessions,
            key=lambda s: (s["tokens_in_total"] or 0) + (s["tokens_out_total"] or 0),
            reverse=True,
        )
        total_count = len(sessions_sorted)
        print_table(sessions_sorted[:args.top], sort_by_total=False)
        if total_count > args.top:
            print(f"\n(showing top {args.top} of {total_count} sessions)")
    else:
        print_table(sessions)


if __name__ == "__main__":
    main()
