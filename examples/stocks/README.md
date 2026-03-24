# Stock Market Analysis Agent

**An agentic_TRACE example application.**

Ask complex financial questions in plain English. Get verified, auditable answers — tables, charts, and full provenance showing exactly how each number was derived.

<!-- TODO: Replace with actual screenshot -->
<!-- ![Web UI](../../docs/images/stocks-web-ui.png) -->

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Quick Start](#quick-start)
3. [How It Works](#how-it-works)
4. [Example Queries](#example-queries)
5. [Configuration](#configuration)
6. [Directory Structure](#directory-structure)

---

## What It Does

Natural language in, verified analysis out:

```
"Plot the semiconductor stocks, normalized to 1, for the last 90 trading days"
```
→ The agent resolves "last 90 trading days" to actual market dates, filters 18 semiconductor tickers, without needing to know the precise tickers, computes normalized prices, and renders a chart — all with deterministic tools. The LLM never sees or processes the price data.

```
"Which sector had the best total return for Q3 2024? Make a bar graph of the top 5 sectors  bar graph showing the Q3 2024 return. Exclude 'ETF' as a sector."
```
1. `resolve_time` → converts "Q3 2024" to 2024-07-01 through 2024-09-30
2. `create_view` (×2) → fetches all stock closing prices on the first and last trading day of Q3
3. `join` → pairs start/end prices per ticker
4. `create_view` → computes per-stock return: `(end_close - start_close) / start_close × 100`
5. `join` → joins with companies table to add sector labels
6. `aggregate` → averages return by sector (12 groups)
7. `create_view` → excludes ETF, sorts descending, takes top 5, finalizes
8. `chart` → renders bar chart with provenance annotation
9. `post_finalize` → validation confirms output matches query

```
"Show me the qwerty stocks for the past month"
```
1. `explore` → searches for "qwerty" — no matches found
2. `explore(question=...)` → asks the user: "I couldn't find a group called 'qwerty'. Did you mean one of these?"
3. User responds: "computer"
4. `create_view(group="computer")` → group not found — error lists all 40 available groups
5. `explore(question=..., options=[...])` → asks again with suggestions: "Please choose from: BIG_TECH, SOFTWARE, SEMICONDUCTORS, CLOUD, CYBERSECURITY"
6. User responds: "CYBERSECURITY"
7. `create_view` → fetches CYBERSECURITY group (CRWD, OKTA, FTNT, PANW, ZS) for the past month
8. `post_finalize` → validation passes

Every query produces:
- **Data tables** (CSV download) with the exact numbers
- **Charts** (PNG) with provenance annotations embedded in the image
- **Audit trail** — provenance files tracing each column back to its source

---

## Quick Start

### Prerequisites

- Python 3.10+
- An LLM API key (Google Gemini free tier, or Anthropic/OpenAI) or a local model server

### Setup

```bash
# From the repo root
cd agentic_TRACE

# Install
pip install -e .

# Create database + load seed data (companies, groups, equations)
python -m examples.stocks.load_data

# Download stock price history from Yahoo Finance (~2 min)
python examples/stocks/scripts/data_loader.py --db examples/stocks/data/market.duckdb
```

### Configure Your LLM

**Google Gemini (free tier available — fastest path):**

```bash
# Get a free API key at https://aistudio.google.com/apikey
export GEMINI_API_KEY="AIza..."
cp examples/stocks/configs/gemini_3_1_flash_lite.example.yaml examples/stocks/configs/gemini_3_1_flash_lite.yaml
```

**Anthropic Claude:**

```bash
# Get an API key at https://console.anthropic.com (requires API credits, separate from Claude Pro/Max subscription)
export ANTHROPIC_API_KEY="sk-ant-..."
cp examples/stocks/configs/anthropic_sonnet.example.yaml examples/stocks/configs/anthropic_sonnet.yaml
```

**Local model** (llama.cpp, Ollama, vLLM, LM Studio):

```bash
cp examples/stocks/configs/local_llamacpp.example.yaml examples/stocks/configs/my_local.yaml
# Edit my_local.yaml: set api_base to your server URL
```

### Run

**CLI:**
```bash
python -m examples.stocks.run "Show me AAPL returns for Q1 2024" \
    --config examples/stocks/configs/anthropic_sonnet.yaml -v
```

**Web UI:**
```bash
python -m examples.stocks.web_app --port 8000
# Open http://localhost:8000
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: agentic_TRACE` | Run `pip install -e .` from the repo root |
| `FileNotFoundError: market.duckdb` | Run both `load_data` and `data_loader.py` steps |
| Stale price data | Re-run `data_loader.py` to refresh from Yahoo Finance |
| LLM timeout / connection refused | Check `api_base` in your YAML config matches your server |

**Note:** `load_data.py` and `scripts/data_loader.py` are separate steps:
- `load_data.py` — creates DB schema, loads seed data (companies, groups, equations). Re-run to refresh seed data.
- `data_loader.py` — downloads `stock_prices` from Yahoo Finance. Run regularly to keep prices current.

---

## How It Works

### Data Pipeline

```
Seed CSVs (companies.csv, groups.csv, equations.csv)
    │
    ▼
load_data.py ──► market.duckdb (companies, groups, equations tables)
    │
    ▼
data_loader.py ──► market.duckdb (stock_prices table via yfinance)
```

The database tracks 535+ tickers covering the full S&P 500 plus ETFs (SPY, QQQ, IWM, DIA, gold ETFs). 40 named stock groups (FANG, MAG7, SEMICONDUCTORS, BANKS, CYBERSECURITY, HOMEBUILDERS, etc.) and 37 named equations (RSI, MACD, EMA, etc.).

### Tools (8 registered per session)

| Tool | Category | Origin | Description |
|---|---|---|---|
| `explore` | EXPLORE | built-in | Search sources/views by keyword, describe schemas, ask the user |
| `resolve_time` | EXPLORE | new | Convert "ytd", "last 90 trading days", "Q1 2024" to concrete date ranges using actual market days |
| `create_view` | TRANSFORM | extended | Filter, sort, compute columns. `StockCreateViewTool` adds `group=` and `tickers=` shortcuts |
| `aggregate` | TRANSFORM | extended | Group aggregation + per-row functions. `StockAggregateTool` adds stock-specific hints |
| `compute_indicator` | TRANSFORM | new | Technical indicators: EMA, SMA, RSI, MACD, Bollinger, ATR |
| `join` | TRANSFORM | built-in | Join two views or stack (concat) vertically |
| `pivot` | TRANSFORM | built-in | Pivot tables |
| `chart` | OUTPUT | built-in | Line / bar / scatter / area / histogram / box charts as PNG |

### Domain Extensions

**`StockCreateViewTool`** extends `CreateViewTool` with:
- `group="semiconductors"` — resolves to tickers from the `groups` table
- `tickers="AAPL,NVDA,AMD"` — direct ticker filter
- `tickers_from="my_view"` — pull tickers from another view's `ticker` column
- Named equations in `add_columns` — looks up expressions from the `equations` table

**`StockAggregateTool`** extends `AggregateTool` with stock-specific examples in error messages and tool descriptions (partition_by=['ticker'], column='close', etc.) that guide LLMs toward correct tool calls.

**`ComputeIndicatorTool`** — stock-specific technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.) with auto-detection of missing `partition_by` for multi-ticker data.

**`ResolveTimeTool`** handles financial temporal expressions:

| Expression | Resolution |
|---|---|
| `ytd` | Start of year to today |
| `last_month` | Last 30 days |
| `last_20_trading_days` | Actual 20 market days from the DB |
| `Q1_2024` | 2024-01-01 to 2024-03-31 |
| `first_10_trading_days_of_Q1_2024` | First 10 market days of Q1 |

### Lifecycle Hooks

**`pre_process`** — currently a stub; override to inject domain hints before the first LLM call.

**`PostFinalizeValidator`** — runs after every finalization:
1. Rule checks — infinite values, all-null columns, missing charts
2. LLM review — sends a structured summary for validation + user-facing summary generation
3. Self-correction — returns `action="reopen"` if actionable issues are found

---

## Example Queries

### Basic

```
Show me AAPL close price for the last 30 trading days
List the top 10 stocks by daily volume at the end of Q4 2025
FANG stocks this year, sorted by return
```

### Technical Indicators

```
20-day EMA for MAG7, normalized to 1, last 100 trading days
RSI for semiconductor stocks — which are overbought?
MACD for NVDA over the past 6 months
```

### Comparisons and Ranking

```
Which semiconductor stock had the best return last quarter?
Compare FANG vs MAG7 performance year-to-date
Which stocks have a higher 100 day EMA than 20 day EMA as of the last trading day?
```

### Charts

```
Plot the MAG7 stocks, normalized to 1, for the past 90 trading days
Which defense stocks have a net positive ytd gain, sorted by ytd gain and plotted
Line chart: John Deere vs Caterpillar, last 20 trading days
```

### Multi-step / Complex

```
Which SEMICONDUCTORS stocks had the least similar daily return pattern to NVDA over
    the past 90 trading days? Chart the top 3 and NVDA normalized to 1.
Which sector had the best average daily return for Q3 2024? Plot the top 3
```

---

## Configuration

### LLM Config YAML

```yaml
name: "My LLM"
provider: openai             # openai | anthropic | text-tool-calling
model: local-model
api_base: http://localhost:8080/v1
api_key_env: OPENAI_API_KEY
temperature: 0.0
max_tokens: 4096
context_window: 32768

# Optional: restrict tools (reduces prompt for smaller models)
tools:
  - resolve_time
  - explore
  - create_view
  - compute_indicator
  - aggregate
  - chart
```

### Providers

| Provider | Use For |
|---|---|
| `openai` | OpenAI, Gemini, llama.cpp, vLLM, Ollama, LM Studio |
| `anthropic` | Claude via native Anthropic SDK |
| `text-tool-calling` | Models without native tool-call support |

### API Keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."    # Claude
export GEMINI_API_KEY="AIza..."          # Gemini
export OPENAI_API_KEY="sk-..."           # OpenAI
```

### Example Configs

| File | Model |
|---|---|
| `gemini_3_1_flash_lite.example.yaml` | Google Gemini Flash Lite (free tier) |
| `anthropic_sonnet.example.yaml` | Claude Sonnet |
| `anthropic_haiku.example.yaml` | Claude Haiku |
| `openai_gpt4.example.yaml` | GPT-4 |
| `local_llamacpp.example.yaml` | Local via llama.cpp |
| `local_lfm2_tool.example.yaml` | LFM2 (text tool calling) |

### Web Config (`web_config.yaml`)

```yaml
title: "Stock Analysis Agent"
host: "127.0.0.1"
port: 8777
configs:
  - local_8080
  - anthropic_sonnet
default_config: local_8080
poll_interval_ms: 3000
preview_rows: 100
```

### CLI Options

| Option | Description |
|---|---|
| `QUERY` | Query to run (positional) |
| `-q, --query` | Query (alternative) |
| `-v` / `-vv` | Verbose / full logs |
| `--config PATH` | LLM config YAML |
| `--max-iter N` | Max iterations (default: 20) |
| `--db PATH` | DuckDB database |
| `--save-log` | Save execution log |

### Web UI Options

| Option | Description |
|---|---|
| `--host` / `--port` | Bind address and port |
| `--config PATH` | Single LLM config |
| `--config-dir DIR` | Directory of configs |
| `--ssl-cert` / `--ssl-key` | HTTPS |
| `-v` / `-vv` | Verbose |

### Database Schema

**stock_prices** — Daily OHLCV for US equities

| Column | Type |
|---|---|
| ticker | VARCHAR |
| date | DATE |
| open, high, low, close | DOUBLE |
| volume | BIGINT |
| adj_close | DOUBLE |

**companies** — 400+ tickers with sector/industry classification

**groups** — 40 named groups (FANG, MAG7, SEMICONDUCTORS, BANKS, CYBERSECURITY, AIRLINES, ...)

**equations** — 37 named formulas (RSI, MACD, EMA, BOLLINGER_PCT_B, ATR, DAILY_RETURN_PCT, ...)

---

## Directory Structure

```
examples/stocks/
├── run.py                    # CLI entry point
├── web_app.py                # Web UI entry point
├── tools_config.py           # Tool registry (8 tools)
├── data_config.py            # DuckDB source registration
├── prompt_config.py          # Domain system prompt
├── hooks.py                  # pre_process, PostFinalizeValidator
├── tools/
│   ├── stock_create_view.py  # StockCreateViewTool (group/tickers)
│   ├── stock_aggregate.py    # StockAggregateTool (stock hints)
│   ├── compute_indicator.py  # Technical indicators (RSI, EMA, ...)
│   └── resolve_time.py       # Financial date resolution
├── configs/                  # LLM config YAML files
│   ├── *.example.yaml        # Templates (committed)
│   └── *.yaml                # Your configs (gitignored)
├── scripts/
│   ├── data_loader.py        # Download prices from Yahoo Finance
│   └── query_db.py           # Interactive DuckDB query tool
├── data/
│   ├── market.duckdb         # Database
│   └── seed/                 # companies.csv, groups.csv, equations.csv
└── sessions/                 # Per-session output directories
```

> **Want to build your own domain agent?** See [Extending the Framework](../../README.md#extending-the-framework) in the main README.
