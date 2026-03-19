#!/usr/bin/env python3
"""
Seed data loader for the stocks example.

Loads/refreshes seed data (companies, groups, equations) from CSV/JSONL files.
Does NOT touch stock_prices table - use examples/stocks/scripts/data_loader.py for that.

Usage:
    python -m examples.stocks.load_data              # Load seed data
    python -m examples.stocks.load_data --summary    # Show database summary

For stock price data:
    python examples/stocks/scripts/data_loader.py --db examples/stocks/data/market.duckdb
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

# Paths relative to this script
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
SEED_DIR = DATA_DIR / "seed"
DB_PATH = DATA_DIR / "market.duckdb"


def create_tables(conn: duckdb.DuckDBPyConnection):
    """Create all required tables if they don't exist."""

    # Stock prices table (structure only - data comes from data_loader.py)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            adj_close DOUBLE,
            PRIMARY KEY (ticker, date)
        )
    """)

    # Companies table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            ticker VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            sector VARCHAR,
            industry VARCHAR
        )
    """)

    # Groups table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS groups (
            group_name VARCHAR NOT NULL,
            ticker VARCHAR NOT NULL,
            description VARCHAR,
            PRIMARY KEY (group_name, ticker)
        )
    """)

    # Equations table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equations (
            name VARCHAR PRIMARY KEY,
            type VARCHAR NOT NULL,
            description VARCHAR NOT NULL,
            keywords VARCHAR,
            rule JSON NOT NULL
        )
    """)

    print("[OK] Tables created/verified")


def load_companies(conn: duckdb.DuckDBPyConnection):
    """Wipe and reload companies from CSV."""
    csv_path = SEED_DIR / "companies.csv"

    if not csv_path.exists():
        print(f"[WARN] Companies CSV not found: {csv_path}")
        return

    # Wipe existing data
    conn.execute("DELETE FROM companies")

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO companies (ticker, name, sector, industry)
            VALUES (?, ?, ?, ?)
        """, [row['ticker'], row['name'], row['sector'], row['industry']])

    count = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    print(f"[OK] Loaded {count} companies")


def load_groups(conn: duckdb.DuckDBPyConnection):
    """Wipe and reload groups from CSV."""
    csv_path = SEED_DIR / "groups.csv"

    if not csv_path.exists():
        print(f"[WARN] Groups CSV not found: {csv_path}")
        return

    # Wipe existing data
    conn.execute("DELETE FROM groups")

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO groups (group_name, ticker, description)
            VALUES (?, ?, ?)
        """, [row['group_name'], row['ticker'], row.get('description', '')])

    count = conn.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
    unique_groups = conn.execute("SELECT COUNT(DISTINCT group_name) FROM groups").fetchone()[0]
    print(f"[OK] Loaded {count} group memberships ({unique_groups} groups)")


def load_equations(conn: duckdb.DuckDBPyConnection):
    """Wipe and reload equations from JSONL."""
    jsonl_path = SEED_DIR / "equations.jsonl"

    if not jsonl_path.exists():
        print(f"[WARN] Equations JSONL not found: {jsonl_path}")
        return

    # Wipe existing data
    conn.execute("DELETE FROM equations")

    count = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                eq = json.loads(line)
                conn.execute("""
                    INSERT INTO equations (name, type, description, keywords, rule)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    eq['name'],
                    eq['type'],
                    eq['description'],
                    eq.get('keywords', ''),
                    json.dumps(eq['rule'])
                ])
                count += 1

    print(f"[OK] Loaded {count} equations")


def show_summary(conn: duckdb.DuckDBPyConnection):
    """Show summary of database contents."""
    print()
    print("=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    # Stock prices
    try:
        result = conn.execute("""
            SELECT
                COUNT(DISTINCT ticker) as tickers,
                COUNT(*) as rows,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM stock_prices
        """).fetchone()
        if result[1] > 0:
            print(f"stock_prices: {result[0]} tickers, {result[1]:,} rows ({result[2]} to {result[3]})")
        else:
            print("stock_prices: (empty)")
    except Exception:
        print("stock_prices: (not loaded)")

    # Companies
    result = conn.execute("SELECT COUNT(*) FROM companies").fetchone()
    print(f"companies: {result[0]} rows")

    # Groups
    result = conn.execute("""
        SELECT COUNT(DISTINCT group_name), COUNT(*) FROM groups
    """).fetchone()
    print(f"groups: {result[0]} groups, {result[1]} memberships")

    # Equations
    try:
        result = conn.execute("""
            SELECT type, COUNT(*) as cnt FROM equations GROUP BY type ORDER BY type
        """).fetchdf()
        if len(result) > 0:
            eq_summary = ", ".join([f"{row['type']}:{row['cnt']}" for _, row in result.iterrows()])
            print(f"equations: {eq_summary}")
        else:
            print("equations: (empty)")
    except Exception:
        print("equations: (not loaded)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Load seed data for stocks example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script loads/refreshes the seed tables (companies, groups, equations)
from CSV/JSONL files. It does NOT modify the stock_prices table.

For stock price data, use the separate data loader:
    python examples/stocks/scripts/data_loader.py --db examples/stocks/data/market.duckdb

Examples:
    python -m examples.stocks.load_data              # Load seed data
    python -m examples.stocks.load_data --summary    # Show database summary
        """
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Just show database summary (don't reload)"
    )
    parser.add_argument(
        "--db",
        type=str,
        help="Custom database path (default: data/market.duckdb)"
    )

    args = parser.parse_args()

    # Determine database path
    db_path = Path(args.db) if args.db else DB_PATH

    print("=" * 60)
    print("STOCKS EXAMPLE - SEED DATA LOADER")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Seed data: {SEED_DIR}")
    print()

    # Just show summary if requested
    if args.summary:
        if not db_path.exists():
            print("[ERROR] Database does not exist. Run without --summary first.")
            sys.exit(1)
        conn = duckdb.connect(str(db_path))
        show_summary(conn)
        conn.close()
        return

    # Ensure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = duckdb.connect(str(db_path))

    try:
        # Create tables
        create_tables(conn)

        # Load seed data (wipes and reloads)
        print()
        print("Loading seed data (wipes existing non-stock tables)...")
        load_companies(conn)
        load_groups(conn)
        load_equations(conn)

        # Show summary
        show_summary(conn)

    finally:
        conn.close()

    print()
    print("[OK] Seed data loaded!")
    print()
    print("Next steps:")
    print("  - To load stock prices: python examples/stocks/scripts/data_loader.py --db examples/stocks/data/market.duckdb")
    print("  - To run queries: python -m examples.stocks.run 'Your query here'")


if __name__ == "__main__":
    main()
