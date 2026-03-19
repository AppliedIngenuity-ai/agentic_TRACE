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
import time
from typing import List, Optional

# Rate limit delay between Yahoo Finance requests (seconds)
REQUEST_DELAY = 0.3


class DataLoader:
    """Handles downloading and storing financial data"""
    
    def __init__(self, db_path: str = "./data/market.duckdb"):
        self.db_path = db_path
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

                        # Check if we need to update (latest data is before end_date)
                        if str(max_date) >= end_date:
                            print(f"⊘ {ticker}: Already up to date (latest: {max_date})")
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
        conn = duckdb.connect(self.db_path)
        
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
    args = parser.parse_args()

    print("Financial Data Loader")
    print("=" * 60)

    loader = DataLoader(db_path=args.db)

    # Just show summary if requested
    if args.summary:
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

    print(f"\nDownloading data for {len(tickers)} tickers")
    print(f"Date range: {start_date} to {end_date}")
    if args.force:
        print("Mode: FORCE REFRESH")
    print()

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

    print("\n✓ Data loading complete!")


if __name__ == "__main__":
    main()
