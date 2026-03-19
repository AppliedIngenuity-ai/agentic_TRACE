#!/usr/bin/env python3
"""
Quick DuckDB query utility
Usage: python query_db.py "SELECT * FROM stock_prices LIMIT 5"
       python query_db.py --interactive
"""

import sys
import duckdb
import argparse
import os

# Enable command history and arrow keys
try:
    import readline
    import atexit

    # Set up history file
    histfile = os.path.join(os.path.expanduser("~"), ".duckdb_query_history")
    try:
        readline.read_history_file(histfile)
        # Default history length
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass

    # Save history on exit
    atexit.register(readline.write_history_file, histfile)

    # Enable tab completion (optional but nice)
    readline.parse_and_bind('tab: complete')

except ImportError:
    # readline not available (Windows)
    pass


def interactive_mode(db_path):
    """Simple interactive SQL shell"""
    conn = duckdb.connect(db_path)

    print(f"Connected to: {db_path}")
    print("Type SQL queries or 'help' for commands. Type 'exit' to quit.\n")

    # Show available tables
    print("Available tables:")
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for table in tables:
        print(f"  - {table[0]}")
    print()

    while True:
        try:
            query = input("duckdb> ").strip()

            if not query:
                continue

            if query.lower() in ('exit', 'quit', '.exit', '.quit'):
                break

            if query.lower() == 'help':
                print("""
Available commands:
  SELECT ...           - Run SQL query
  SHOW TABLES;         - List all tables
  DESCRIBE tablename;  - Show table schema
  exit                 - Exit shell

Example queries:
  SELECT * FROM stock_prices LIMIT 10;
  SELECT ticker, COUNT(*) FROM stock_prices GROUP BY ticker;
  SELECT * FROM stock_prices WHERE ticker='AAPL' ORDER BY date DESC LIMIT 5;
                """)
                continue

            # Execute query
            result = conn.execute(query)

            # Show results
            try:
                df = result.fetchdf()
                if not df.empty:
                    print(df.to_string(index=False))
                    print(f"\n({len(df)} rows)\n")
                else:
                    print("Query executed successfully (no results)\n")
            except Exception:
                # For non-SELECT queries
                print("Query executed successfully\n")

        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except Exception as e:
            print(f"Error: {e}\n")

    conn.close()
    print("Goodbye!")


def run_query(db_path, query, output_format='table'):
    """Run a single query and print results"""
    conn = duckdb.connect(db_path)

    try:
        result = conn.execute(query)
        df = result.fetchdf()

        if output_format == 'csv':
            print(df.to_csv(index=False))
        elif output_format == 'json':
            print(df.to_json(orient='records', indent=2))
        elif output_format == 'markdown':
            print(df.to_markdown(index=False))
        else:
            print(df.to_string(index=False))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Query DuckDB database')
    parser.add_argument('query', nargs='?', help='SQL query to execute')
    parser.add_argument('--db', default='./data/market.duckdb', help='Database path')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--format', choices=['table', 'csv', 'json', 'markdown'],
                       default='table', help='Output format')

    args = parser.parse_args()

    if args.interactive or not args.query:
        interactive_mode(args.db)
    else:
        run_query(args.db, args.query, args.format)


if __name__ == '__main__':
    main()
