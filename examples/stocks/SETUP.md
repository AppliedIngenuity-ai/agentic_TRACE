# Stocks Example — Setup & Usage

All commands run from the **project root** unless noted.

---

## 1. Install

```bash
pip install -e .

# For Claude / Anthropic models:
pip install anthropic
```

---

## 2. Configure an LLM

Copy one of the example configs and edit it:

```bash
# Local model (llama.cpp, Ollama, vLLM, LM Studio, etc.)
cp examples/stocks/configs/local_llamacpp.example.yaml examples/stocks/configs/my_model.yaml

# Claude via Anthropic API
cp examples/stocks/configs/anthropic_sonnet.example.yaml examples/stocks/configs/my_model.yaml

# OpenAI
cp examples/stocks/configs/openai_gpt4.example.yaml examples/stocks/configs/my_model.yaml
```

Edit `my_model.yaml` — set `model`, `api_base`, and `api_key_env` for your setup.
Config files named `*.yaml` are gitignored; `*.example.yaml` templates are committed.

---

## 3. Initialize the database

Load seed tables (companies, groups, equations). Run once, or re-run to refresh:

```bash
python -m examples.stocks.load_data
```

Verify it loaded correctly:

```bash
python -m examples.stocks.load_data --summary
```

---

## 4. Download stock prices

```bash
# Download all tickers from companies table (5 years of history by default)
python examples/stocks/scripts/data_loader.py

# Specific tickers only
python examples/stocks/scripts/data_loader.py --tickers AAPL NVDA MSFT

# Custom date range
python examples/stocks/scripts/data_loader.py --days 365

# Force re-download everything
python examples/stocks/scripts/data_loader.py --force

# Check what's in the database without downloading
python examples/stocks/scripts/data_loader.py --summary
```

The database is saved to `examples/stocks/data/market.duckdb` by default.

---

## 5. Run a query (CLI)

```bash
# Basic
python -m examples.stocks.run "Show me AAPL returns for Q1 2024"

# With a specific LLM config
python -m examples.stocks.run "Compare FANG stocks last month" --config examples/stocks/configs/my_model.yaml

# Verbose — shows each tool call and result
python -m examples.stocks.run "Top 5 by volume yesterday" -v

# Extra verbose — full untruncated logs (good for debugging)
python -m examples.stocks.run "Top 5 by volume yesterday" -vv

# Limit iterations (default: 20)
python -m examples.stocks.run "EMA for energy stocks" --max-iter 10

# Save a log file
python -m examples.stocks.run "NVDA RSI last month" --save-log --log-dir ./logs
```

---

## 6. Run the web UI

```bash
# Start with defaults (reads web_config.yaml)
python -m examples.stocks.web_app

# Custom port
python -m examples.stocks.web_app --port 8080

# Force a single LLM config (ignores web_config.yaml model selector)
python -m examples.stocks.web_app --config examples/stocks/configs/my_model.yaml

# Verbose logs in the terminal
python -m examples.stocks.web_app -v
```

Open `http://localhost:8777` (or whatever port) in your browser.

---

## 7. Debug the database

Query the DuckDB database directly:

```bash
# Interactive SQL shell (from examples/stocks/)
python scripts/query_db.py --interactive

# Single query
python scripts/query_db.py "SELECT ticker, COUNT(*) FROM stock_prices GROUP BY ticker ORDER BY 2 DESC LIMIT 10"

# Different output formats
python scripts/query_db.py "SELECT * FROM companies LIMIT 5" --format markdown
python scripts/query_db.py "SELECT * FROM groups LIMIT 5" --format csv

# Custom database path
python scripts/query_db.py --db /path/to/other.duckdb --interactive
```

Run from `examples/stocks/` so the default `--db ./data/market.duckdb` path resolves correctly,
or pass `--db examples/stocks/data/market.duckdb` from the project root.

---

## 8. Session logs

Each run saves a session under `examples/stocks/sessions/<timestamp>/`:

```
sessions/20260304_022952_8270879d/
├── debug.log       # Full trace: every LLM request, tool call, result
├── stream.jsonl    # SSE events (for web replays)
├── input.jsonl     # Input query and config
├── artifacts/      # Generated charts (.png)
└── results/        # Exported CSVs
```

Read `debug.log` to see exactly what the LLM called and what it got back — useful when a run produces wrong results.

---

## Typical workflow

```bash
# First time
pip install -e .
cp examples/stocks/configs/local_llamacpp.example.yaml examples/stocks/configs/my_model.yaml
# edit my_model.yaml
python -m examples.stocks.load_data
python examples/stocks/scripts/data_loader.py

# Run queries
python -m examples.stocks.run "Plot NVDA EMA 20 last 100 trading days" --config examples/stocks/configs/my_model.yaml -v

# Debug a bad run
# open sessions/<id>/debug.log and look for ERROR lines or suspicious tool args
```
