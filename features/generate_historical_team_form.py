# generate_historical_team_form.py

import glob
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Team name mapping
TEAM_NAME_MAP = {
    'New York Yankees': 'NYY', 'Boston Red Sox': 'BOS', 'Tampa Bay Rays': 'TB',
    'Toronto Blue Jays': 'TOR', 'Baltimore Orioles': 'BAL', 'Cleveland Guardians': 'CLE',
    'Detroit Tigers': 'DET', 'Kansas City Royals': 'KC', 'Chicago White Sox': 'CWS',
    'Minnesota Twins': 'MIN', 'Houston Astros': 'HOU', 'Seattle Mariners': 'SEA',
    'Texas Rangers': 'TEX', 'Los Angeles Angels': 'LAA', 'Oakland Athletics': 'OAK',
    'Atlanta Braves': 'ATL', 'Miami Marlins': 'MIA', 'New York Mets': 'NYM',
    'Philadelphia Phillies': 'PHI', 'Washington Nationals': 'WSH', 'Chicago Cubs': 'CHC',
    'Cincinnati Reds': 'CIN', 'Milwaukee Brewers': 'MIL', 'Pittsburgh Pirates': 'PIT',
    'St. Louis Cardinals': 'STL', 'Arizona Diamondbacks': 'ARI', 'Colorado Rockies': 'COL',
    'Los Angeles Dodgers': 'LAD', 'San Diego Padres': 'SD', 'San Francisco Giants': 'SF'
}

# Base and output directories using pathlib
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def scrape_team_form_api_for_date(date_str):
    """Scrape standings from MLB API and save as team_form_<date>.csv."""
    year = datetime.strptime(date_str, "%Y-%m-%d").year
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={year}&standingsTypes=regularSeason&date={date_str}"

    try:
        response = requests.get(url)
        data = response.json()

        teams = []
        for record_type in data.get('records', []):
            for team_rec in record_type.get('teamRecords', []):
                team_name = team_rec['team'].get('name', 'Unknown')
                normalized_name = TEAM_NAME_MAP.get(team_name, team_name)
                teams.append({
                    "team": normalized_name,
                    "wins": team_rec.get('wins'),
                    "losses": team_rec.get('losses'),
                    "run_diff": team_rec.get('runDifferential'),
                    "streak": team_rec.get('streak', {}).get('streakCode', ''),
                    "games_played": team_rec.get('gamesPlayed'),
                    "win_pct": team_rec.get('winningPercentage')
                })

        df = pd.DataFrame(teams)
        output_path = PROCESSED_DIR / f"team_form_{date_str}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved team form for {date_str} to {output_path}")
    except Exception as e:
        logger.error(f"Failed to scrape team form for {date_str}: {e}")

def run_team_form_rolling():
    existing = {f.stem.split("_")[-1] for f in PROCESSED_DIR.glob("team_form_*.csv")}
    results_files = sorted(PROCESSED_DIR.glob("historical_results_*.csv"))

    for result_file in results_files:
        date_str = result_file.stem.split("_")[-1]
        if date_str in existing:
            continue
        scrape_team_form_api_for_date(date_str)

# === NEW: Rolling window team form generator ===
def run_team_form_rolling_window(days_back=45, overwrite=False, target_date=None):
    if target_date is None:
        base_date = datetime.today().date()
    else:
        base_date = target_date
    for i in range(1, days_back + 1):
        date = base_date - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        output_path = PROCESSED_DIR / f"team_form_{date_str}.csv"
        if output_path.exists() and not overwrite:
            logger.info(f"Skipping {date_str} (already exists)")
            continue
        scrape_team_form_api_for_date(date_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days_back", type=int, default=45, help="How many days back to generate team form for.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()
    run_team_form_rolling_window(days_back=args.days_back, overwrite=args.overwrite)

    # cd C:\Users\roman\baseball_forecast_project\features
    # python generate_historical_team_form.py --days_back 45 --overwrite 
