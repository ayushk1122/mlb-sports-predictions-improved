# scrape_statcast.py

from pathlib import Path
import pandas as pd
import os
from datetime import datetime, timedelta
from pybaseball import statcast
from pybaseball import cache

cache.disable()

# === Dynamic Project Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def scrape_statcast_today_or_recent(n_days=3, target_date=None):
    if target_date is None:
        print(f"Attempting to scrape Statcast data for today first, then the last {n_days} days...")
        today = datetime.today()
    else:
        print(f"Attempting to scrape Statcast data for target date {target_date}, then the last {n_days} days...")
        today = target_date
    
    successful_scrapes = []

    # Step 1: Try scraping target date (or today if not specified)
    date_str = today.strftime('%Y-%m-%d')
    print(f"Trying target date: {date_str}")
    try:
        df = statcast(start_dt=date_str, end_dt=date_str)
        if not df.empty:
            output_path = RAW_DIR / f"statcast_{date_str}.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} rows for target date to: {output_path}")
            return output_path, date_str
        else:
            print("No data for target date, trying previous days...")
    except Exception as e:
        print(f"Error scraping target date data: {e}")

    # Step 2: Scrape recent past days
    for delta in range(1, n_days + 5):
        check_date = today - timedelta(days=delta)
        date_str = check_date.strftime('%Y-%m-%d')
        try:
            print(f"Trying previous date: {date_str}")
            df = statcast(start_dt=date_str, end_dt=date_str)
            if not df.empty:
                output_path = RAW_DIR / f"statcast_{date_str}.csv"
                df.to_csv(output_path, index=False)
                print(f"Saved {len(df)} rows to: {output_path}")
                return output_path, date_str
        except Exception as e:
            print(f"Error scraping data for {date_str}: {e}")

    print("No Statcast data found for target date or recent days.")
    return None, None

# === NEW: Rolling window statcast scraper ===
def scrape_statcast_rolling(days_back=45, target_date=None, overwrite=False):
    if target_date is None:
        base_date = datetime.today().date()
    else:
        base_date = target_date
    for i in range(1, days_back + 1):
        date = base_date - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        output_path = RAW_DIR / f"statcast_{date_str}.csv"
        if output_path.exists() and not overwrite:
            print(f"Skipping {date_str} (already exists)")
            continue
        try:
            print(f"Scraping Statcast for {date_str}...")
            df = statcast(start_dt=date_str, end_dt=date_str)
            if not df.empty:
                df.to_csv(output_path, index=False)
                print(f"Saved {len(df)} rows to: {output_path}")
            else:
                print(f"No data for {date_str}")
        except Exception as e:
            print(f"Error scraping data for {date_str}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days_back", type=int, default=45, help="How many days back to scrape.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()
    scrape_statcast_rolling(days_back=args.days_back, overwrite=args.overwrite)
