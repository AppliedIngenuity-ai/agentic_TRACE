# Setup Guide

Step-by-step instructions for getting agentic_TRACE and the stocks example running from scratch, on Mac or Linux.

---

## Platform-specific: Python & virtual environment

### Mac

macOS ships with an outdated Python. Install a recent version via Homebrew, then create a virtual environment.

**Step 1 — Install Homebrew (if not installed)**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Step 2 — Install Python 3.11+**

```bash
brew install python@3.11
```

Verify:

```bash
python3.11 --version   # should print Python 3.11.x
```

**Step 3 — Create a virtual environment**

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

> Your prompt should now show `(.venv)`. Re-run `source .venv/bin/activate` each time you open a new terminal.

---

### Linux

Most modern Linux distributions include Python 3.10+. Check first:

```bash
python3 --version
```

If it shows 3.10 or newer, proceed. If older, install via your package manager:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install python3.11 python3.11-venv

# Fedora
sudo dnf install python3.11
```

**Create a virtual environment:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

---

## Install the package

With the virtual environment active:

```bash
pip install -e .
```

This installs all core dependencies (openai, duckdb, pandas, Flask, matplotlib, yfinance, etc.).

If you plan to use **Claude / Anthropic** models, also install the Anthropic SDK:

```bash
pip install anthropic
```

---

## Configure an LLM

LLM settings live in YAML files under `examples/stocks/configs/`. These `*.yaml` files are gitignored so your API keys stay private; only `*.example.yaml` templates are committed.

**Copy a template and edit it:**

```bash
# For Google Gemini (free tier available — recommended for getting started)
cp examples/stocks/configs/gemini_3_1_flash_lite.yaml examples/stocks/configs/my_model.yaml

# For Anthropic Claude
cp examples/stocks/configs/anthropic_sonnet.example.yaml examples/stocks/configs/my_model.yaml

# For OpenAI
cp examples/stocks/configs/openai_gpt4.example.yaml examples/stocks/configs/my_model.yaml

# For a local model (llama.cpp, Ollama, LM Studio, vLLM, etc.)
cp examples/stocks/configs/local_llamacpp.example.yaml examples/stocks/configs/my_model.yaml
```

### API Keys

Set your API key as an environment variable. The `api_key_env` field in the YAML tells the system which variable to read.

**Google Gemini** (free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)):

```bash
export GEMINI_API_KEY="AIza..."
```

**Anthropic Claude:**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**OpenAI:**

```bash
export OPENAI_API_KEY="sk-..."
```

To make the key permanent, add the `export` line to your shell profile (`~/.zshrc` on Mac, `~/.bashrc` on Linux), then run `source ~/.zshrc` (or `~/.bashrc`).

**Local models** (llama.cpp, Ollama, etc.) do not need an API key — set `api_key_env: null` in the YAML.

### Example: Gemini config

The `gemini_3_1_flash_lite.yaml` file (which you can copy directly, no edits needed) looks like:

```yaml
name: "Gemini Flash 3-1 Flash Lite"
provider: openai
model: gemini-3.1-flash-lite-preview
api_base: https://generativelanguage.googleapis.com/v1beta/openai/
api_key_env: GEMINI_API_KEY
temperature: 0.0
max_tokens: 4096
context_window: 1000000
```

### Example: Local model config

```yaml
name: "Local LLM"
provider: openai
model: local-model
api_base: http://localhost:8080/v1
api_key_env: null
temperature: 0.0
max_tokens: 4096
context_window: 32768
```

---

## Initialize the database

Run once to create the DuckDB database and seed tables (companies, groups, equations):

```bash
python -m examples.stocks.load_data
```

Verify the tables loaded correctly:

```bash
python -m examples.stocks.load_data --summary
```

---

## Download stock prices

Stock price history is fetched from Yahoo Finance:

```bash
# Download all tickers from the companies table (5 years of history)
python examples/stocks/scripts/data_loader.py --db examples/stocks/data/market.duckdb

# Or just specific tickers
python examples/stocks/scripts/data_loader.py --db examples/stocks/data/market.duckdb --tickers AAPL NVDA MSFT

# Check what's already in the database
python examples/stocks/scripts/data_loader.py --db examples/stocks/data/market.duckdb --summary
```

The database is saved to `examples/stocks/data/market.duckdb` (gitignored).

---

## Run a query (CLI)

```bash
# Basic
python -m examples.stocks.run "Show me AAPL returns for Q1 2024"

# With your LLM config
python -m examples.stocks.run "Compare FANG stocks last month" \
    --config examples/stocks/configs/my_model.yaml

# Verbose — shows each tool call and result
python -m examples.stocks.run "Top 5 by volume yesterday" \
    --config examples/stocks/configs/my_model.yaml -v

# Extra verbose — full untruncated logs (useful for debugging)
python -m examples.stocks.run "NVDA RSI last month" \
    --config examples/stocks/configs/my_model.yaml -vv
```

---

## Run the web UI

The web UI lets you run queries interactively in a browser.

**Step 1 — Copy and edit the web config:**

```bash
cp examples/stocks/web_config.example.yaml examples/stocks/web_config.yaml
```

Edit `web_config.yaml` to list your LLM config(s):

```yaml
title: "Stock Analysis Agent"
host: "127.0.0.1"
port: 8777

configs:
  - my_model          # references examples/stocks/configs/my_model.yaml
default_config: my_model

poll_interval_ms: 3000
preview_rows: 100
```

> The config names in `web_config.yaml` are YAML filenames without the `.yaml` extension, relative to `examples/stocks/configs/`.

**Step 2 — Start the server:**

```bash
python -m examples.stocks.web_app
```

**Step 3 — Open your browser:**

```
http://localhost:8777
```

Other web app options:

```bash
# Custom port
python -m examples.stocks.web_app --port 8080

# Force a single model (skips the model selector)
python -m examples.stocks.web_app --config examples/stocks/configs/my_model.yaml

# Verbose logs in terminal
python -m examples.stocks.web_app -v
```

---

## Complete first-run checklist

```bash
# 1. Activate virtual env
source .venv/bin/activate

# 2. Install
pip install -e .

# 3. Set API key (example: Gemini)
export GEMINI_API_KEY="AIza..."

# 4. Seed the database
python -m examples.stocks.load_data

# 5. Download stock prices
python examples/stocks/scripts/data_loader.py --db examples/stocks/data/market.duckdb

# 6. Copy and edit LLM config
cp examples/stocks/configs/gemini_3_1_flash_lite.yaml examples/stocks/configs/my_model.yaml
# (no edits needed for Gemini if GEMINI_API_KEY is set)

# 7. Run a test query
python -m examples.stocks.run "Show me AAPL returns for Q1 2024" \
    --config examples/stocks/configs/my_model.yaml -v
```

---

## Gitignored files (not included in checkout)

These files must be created locally — they are intentionally excluded from the repository:

| Path | How to create |
|------|---------------|
| `examples/stocks/configs/*.yaml` | Copy from a `*.example.yaml` template |
| `examples/stocks/web_config.yaml` | Copy from `web_config.example.yaml` |
| `examples/stocks/data/market.duckdb` | Created by `load_data` + `data_loader.py` |
| `.env` | Optional: place `export KEY=value` lines here if using python-dotenv |
| `examples/stocks/sessions/` | Auto-created on first run |
| `examples/stocks/logs/` | Auto-created with `--save-log` |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'agentic_TRACE'`**
Run `pip install -e .` from the project root with your virtual environment active.

**`No such file or directory: 'examples/stocks/data/market.duckdb'`**
You need to run `python -m examples.stocks.load_data` first to create the database.

**API key errors (`AuthenticationError`, `401`)**
Check that you exported the correct environment variable (e.g. `GEMINI_API_KEY`) and that it matches the `api_key_env` field in your YAML config.

**`Unknown temporal expression`**
The `resolve_time` tool needs stock price data in the database to count trading days. Make sure `data_loader.py` completed successfully.

**Charts not appearing in web UI**
Check that `examples/stocks/sessions/` is writable. Charts are saved as `.png` files in `sessions/<id>/artifacts/`.
