"""
Data loader for financial data
Downloads stock prices and stores in DuckDB

Usage:
    python data_loader.py                      # Update all tickers from companies.csv
    python data_loader.py --tickers AAPL NVDA  # Download specific tickers
    python data_loader.py --days 30            # Last 30 days only
    python data_loader.py --force              # Force re-download all data
    python data_loader.py --summary            # Just show data summary
"""

import duckdb
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import shutil
import time
from typing import List, Optional

# Rate limit delay between Yahoo Finance requests (seconds)
REQUEST_DELAY = 0.51


class DataLoader:
    """Handles downloading and storing financial data"""
    
    def __init__(self, db_path: str = "./data/market.duckdb", read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        if read_only:
            return  # inspection only (e.g. --summary): don't create dirs/tables, just read
        self._ensure_db_exists()
        self._create_tables()
    
    def _ensure_db_exists(self):
        """Create directory if needed"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        conn = duckdb.connect(self.db_path)
        
        # Stock prices table
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
        
        # Company filings table (for future use)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                ticker VARCHAR,
                filing_date DATE,
                filing_type VARCHAR,
                period_end DATE,
                content TEXT,
                PRIMARY KEY (ticker, filing_date, filing_type)
            )
        """)
        
        conn.close()
        print(f"✓ Database initialized at {self.db_path}")
    
    def download_stock_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> dict:
        """
        Download stock price data from Yahoo Finance

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            force_refresh: If True, delete existing data first

        Returns:
            Dictionary with download statistics
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # "Current" = the most recent expected trading day = yesterday (rolled back over weekends).
        # yfinance's end date is EXCLUSIVE, so end_date stays "today" to fetch THROUGH yesterday — but
        # the up-to-date check below compares against yesterday, so a ticker that already has yesterday's
        # data is skipped WITHOUT any Yahoo Finance request.
        latest_expected = _latest_expected_trading_day(datetime.now().date())

        conn = duckdb.connect(self.db_path)
        stats = {"success": [], "failed": [], "skipped": [], "updated": []}

        for ticker in tickers:
            ticker = ticker.upper()

            try:
                actual_start = start_date
                is_update = False

                # Check existing data and determine what we need to fetch
                if not force_refresh:
                    result = conn.execute(f"""
                        SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as cnt
                        FROM stock_prices
                        WHERE ticker = '{ticker}'
                    """).fetchone()

                    existing_count = result[2]

                    if existing_count > 0:
                        max_date = result[1]
                        is_update = True

                        # Already have data through the most recent expected trading day (yesterday)?
                        # Skip the Yahoo Finance query entirely — no network call needed.
                        if max_date >= latest_expected:
                            print(f"⊘ {ticker}: Already up to date (latest: {max_date}, current: {latest_expected})")
                            stats["skipped"].append(ticker)
                            continue

                        # Start from day after latest data
                        actual_start = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")
                        print(f"↻ {ticker}: Updating from {actual_start} to {end_date} (had data through {max_date})...")
                    else:
                        print(f"↓ {ticker}: Downloading from {actual_start} to {end_date}...")
                else:
                    print(f"↓ {ticker}: Downloading from {actual_start} to {end_date} (force refresh)...")

                # Download data (with rate limiting)
                # yfinance uses dashes for class shares (BRK-B), DB stores dots (BRK.B)
                yf_ticker = ticker.replace('.', '-')
                stock = yf.Ticker(yf_ticker)
                df = stock.history(start=actual_start, end=end_date)
                time.sleep(REQUEST_DELAY)

                if df.empty:
                    print(f"✗ {ticker}: No data available")
                    stats["failed"].append(ticker)
                    continue
                
                # Prepare data
                df = df.reset_index()
                df['ticker'] = ticker

                # Normalize column names (yfinance uses different naming)
                df.columns = [c.lower().replace(' ', '_') for c in df.columns]

                # Handle different possible column names from yfinance
                column_mapping = {}
                for col in df.columns:
                    if 'adj' in col.lower() and 'close' in col.lower():
                        column_mapping[col] = 'adj_close'

                if column_mapping:
                    df = df.rename(columns=column_mapping)

                # If adj_close doesn't exist, use close
                if 'adj_close' not in df.columns:
                    df['adj_close'] = df['close']

                # Select columns (only those that exist)
                required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
                df = df[required_cols]

                # Delete existing data if force refresh
                if force_refresh:
                    conn.execute(f"""
                        DELETE FROM stock_prices
                        WHERE ticker = '{ticker}'
                        AND date BETWEEN '{start_date}' AND '{end_date}'
                    """)
                else:
                    # Filter out dates that already exist (yfinance may return overlapping data)
                    # Normalize dates to YYYY-MM-DD format for comparison (yfinance has timezone)
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.date

                    existing_dates = conn.execute(f"""
                        SELECT date FROM stock_prices
                        WHERE ticker = '{ticker}'
                        AND date >= '{df['date'].min()}'
                    """).fetchdf()

                    if len(existing_dates) > 0:
                        existing_set = set(existing_dates['date'].astype(str))
                        df = df[~df['date'].astype(str).isin(existing_set)]

                # Skip insert if no new data after filtering
                if len(df) == 0:
                    print(f"⊘ {ticker}: No new data to insert (yfinance returned existing dates)")
                    stats["skipped"].append(ticker)
                    continue

                # Insert new data
                conn.execute("INSERT INTO stock_prices SELECT * FROM df")

                if is_update:
                    print(f"✓ {ticker}: Updated with {len(df)} new rows")
                    stats["updated"].append(ticker)
                else:
                    print(f"✓ {ticker}: Successfully loaded {len(df)} rows")
                    stats["success"].append(ticker)
                
            except Exception as e:
                print(f"✗ {ticker}: Error - {str(e)}")
                stats["failed"].append(ticker)
        
        conn.close()
        return stats
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary of data in database"""
        conn = duckdb.connect(self.db_path, read_only=self.read_only)
        
        summary = conn.execute("""
            SELECT 
                ticker,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                COUNT(*) as data_points,
                ROUND(AVG(close), 2) as avg_close,
                ROUND(MIN(close), 2) as min_close,
                ROUND(MAX(close), 2) as max_close
            FROM stock_prices
            GROUP BY ticker
            ORDER BY ticker
        """).fetchdf()
        
        conn.close()
        return summary
    
    def clear_data(self, ticker: Optional[str] = None):
        """Clear data from database"""
        conn = duckdb.connect(self.db_path)
        
        if ticker:
            conn.execute(f"DELETE FROM stock_prices WHERE ticker = '{ticker.upper()}'")
            print(f"✓ Cleared data for {ticker.upper()}")
        else:
            conn.execute("DELETE FROM stock_prices")
            print("✓ Cleared all stock price data")
        
        conn.close()


def load_tickers_from_config(db_path: str = "./data/market.duckdb") -> List[str]:
    """Load tickers from companies.csv seed file next to the database."""
    import csv
    from pathlib import Path

    # Resolve companies.csv relative to the database path (seed/ sits beside the .duckdb)
    db_dir = Path(db_path).parent
    companies_path = db_dir / "seed" / "companies.csv"

    if not companies_path.exists():
        raise FileNotFoundError(
            f"companies.csv not found at {companies_path}\n"
            f"Expected seed/ directory next to database at {db_path}"
        )

    tickers = []
    with open(companies_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get('ticker', '')
            if ticker:
                tickers.append(ticker)

    print(f"Loaded {len(tickers)} tickers from {companies_path}")
    return tickers


def _wal(path: str) -> str:
    return path + ".wal"


def _latest_expected_trading_day(today):
    """The most recent day we'd expect market data for: yesterday, rolled back over weekends.
    (Holidays aren't modeled — at worst that's one wasted, empty Yahoo query on a holiday.)"""
    d = today - timedelta(days=1)
    while d.weekday() >= 5:  # Saturday=5, Sunday=6 → step back to Friday
        d -= timedelta(days=1)
    return d


def prepare_hot_swap(db_path: str) -> str:
    """Copy the live DB to a private staging file we can open read-write WITHOUT disturbing the file the
    server holds open. DuckDB allows only one read-write process per file and won't let a writer in while
    the server holds a read-only handle — so we never write the live file; we build a copy and atomically
    swap it in (see commit_hot_swap). The OS-level copy needs no DuckDB lock."""
    staging = db_path + ".staging"
    for p in (staging, _wal(staging)):
        if os.path.exists(p):
            os.remove(p)
    shutil.copy2(db_path, staging)
    if os.path.exists(_wal(db_path)):
        shutil.copy2(_wal(db_path), _wal(staging))
    print(f"⇄ hot-swap: staged a copy at {staging} (the live DB is untouched while it's in use)")
    return staging


def commit_hot_swap(staging: str, db_path: str) -> None:
    """Atomically replace the live DB with the freshly-updated staging copy. os.replace is atomic on
    POSIX, and the server's already-open handle keeps serving the OLD file until it REOPENS the DB — so
    readers never see a half-written file, but new rows appear only after the server reopens/restarts."""
    os.replace(staging, db_path)
    if os.path.exists(_wal(staging)):
        os.replace(_wal(staging), _wal(db_path))
    elif os.path.exists(_wal(db_path)):
        os.remove(_wal(db_path))  # a stale WAL would shadow the swapped-in file


def main():
    """
    Download stock data from Yahoo Finance.

    Usage:
        python data_loader.py                    # Update all tickers from companies.csv
        python data_loader.py --tickers AAPL NVDA  # Download specific tickers
        python data_loader.py --days 30          # Last 30 days only
        python data_loader.py --force            # Force re-download all data
        python data_loader.py --summary          # Just show data summary
    """
    import argparse

    parser = argparse.ArgumentParser(description="Download stock data from Yahoo Finance")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to download (default: all from companies.csv)")
    parser.add_argument("--days", type=int, default=5*365, help="Number of days of history (default: 5 years)")
    parser.add_argument("--force", action="store_true", help="Force re-download existing data")
    parser.add_argument("--summary", action="store_true", help="Just show data summary, don't download")
    parser.add_argument("--db", default="./data/market.duckdb", help="Database path")
    parser.add_argument("--hot-swap", action="store_true",
                        help="update a staged COPY and atomically swap it in, so the loader can run "
                             "while the server holds the DB open (server picks it up on its next reopen)")
    args = parser.parse_args()

    print("Financial Data Loader")
    print("=" * 60)

    # --summary only reads: open the live DB READ-ONLY so it works even while the server holds it open
    # (DuckDB allows many concurrent readers; only a writer is exclusive).
    if args.summary:
        loader = DataLoader(db_path=args.db, read_only=True)
        print("\nDATA SUMMARY")
        summary = loader.get_data_summary()
        if len(summary) > 0:
            print(summary.to_string(index=False))
        else:
            print("No data in database")
        return

    # Get tickers from command line or config
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"Using {len(tickers)} tickers from command line")
    else:
        tickers = load_tickers_from_config(db_path=args.db)

    # Calculate date range
    # Note: yfinance end date is EXCLUSIVE, so we use today to get data through yesterday
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    # Hot-swap: build into a private staged COPY and atomically swap it in, so the loader can run while
    # the server holds the live DB open (DuckDB won't let a writer touch a file that's in use).
    staging = None
    work_db = args.db
    if args.hot_swap and os.path.exists(args.db):
        staging = prepare_hot_swap(args.db)
        work_db = staging

    print(f"\nDownloading data for {len(tickers)} tickers")
    print(f"Date range: {start_date} to {end_date}")
    if args.force:
        print("Mode: FORCE REFRESH")
    print()

    try:
        try:
            loader = DataLoader(db_path=work_db)
        except duckdb.Error as e:
            if "lock" in str(e).lower() and staging is None:
                print("\n✗ The database is locked — the server has it open. Re-run with --hot-swap to "
                      "update a copy and atomically swap it in while the server keeps running.")
                return
            raise

        stats = loader.download_stock_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            force_refresh=args.force
        )

        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print(f"  New:     {len(stats['success'])} tickers")
        print(f"  Updated: {len(stats['updated'])} tickers")
        print(f"  Skipped: {len(stats['skipped'])} tickers (already up to date)")
        print(f"  Failed:  {len(stats['failed'])} tickers")

        if stats['failed']:
            print(f"\n  Failed tickers: {', '.join(stats['failed'])}")

        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        summary = loader.get_data_summary()
        if len(summary) > 0:
            print(summary.to_string(index=False))

        if staging is not None:
            if stats["success"] or stats["updated"]:
                commit_hot_swap(staging, args.db)
                print(f"✓ hot-swap: {args.db} updated. The running server will pick it up on its next reopen/restart.")
            else:
                for p in (staging, _wal(staging)):
                    if os.path.exists(p):
                        os.remove(p)
                print("⊘ hot-swap: nothing changed — live DB left as-is (no swap needed).")
    except BaseException:
        # Never leave a half-built staging file behind (and never touch the live DB on failure).
        if staging is not None:
            for p in (staging, _wal(staging)):
                if os.path.exists(p):
                    os.remove(p)
        raise

    print("\n✓ Data loading complete!")


if __name__ == "__main__":
    main()
