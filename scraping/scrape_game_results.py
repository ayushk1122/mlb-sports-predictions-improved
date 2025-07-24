# scrape_game_results.py

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import argparse

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Normalization to match historical_matchups (non-abbreviated names)
full_name_to_short = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago White Sox": "CWS",
    "Chicago Cubs": "CHC",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Athletics": "OAK",  # Alternative name
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH"
}

def normalize(name):
    return full_name_to_short.get(name.strip(), None)

def scrape_results_for_date(game_date, matchup_csv_path, output_dir):
    try:
        result_path = output_dir / f"historical_results_{game_date}.csv"
        if result_path.exists():
            logger.info(f"Skipping {game_date} — results already exist.")
            return result_path

        matchups = pd.read_csv(matchup_csv_path)
        matchups["game_date"] = game_date
        logger.info(f"Loaded {len(matchups)} matchups from {matchup_csv_path}")

        year, month, day = game_date.year, game_date.month, game_date.day
        url = f"https://www.baseball-reference.com/boxes/index.fcgi?year={year}&month={month}&day={day}"
        logger.info(f"Loading URL: {url}")

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        results = []

        for game in soup.select("div.game_summary"):
            try:
                winner_tag = game.select_one("tr.winner td a")
                loser_tag = game.select_one("tr.loser td a")
                if not winner_tag or not loser_tag:
                    continue

                winner = normalize(winner_tag.text)
                loser = normalize(loser_tag.text)

                if not winner or not loser:
                    logger.warning(f"Could not normalize team names: {winner_tag.text} vs {loser_tag.text}")
                    continue

                for _, row in matchups.iterrows():
                    if {winner, loser} == {row["home_team"], row["away_team"]}:
                        results.append({
                            "game_date": game_date,
                            "home_team": row["home_team"],
                            "away_team": row["away_team"],
                            "winner": winner
                        })
                        break

            except Exception as e:
                logger.warning(f"Error parsing game: {e}")

        if not results:
            logger.warning(f"No results matched the matchups for {game_date}")
            return None

        results_df = pd.DataFrame(results)
        results_df.to_csv(result_path, index=False)
        logger.info(f"Saved {len(results_df)} results to: {result_path}")
        return result_path

    except Exception as e:
        logger.error(f"Error processing {game_date}: {e}")
        return None

def run_rolling_scraper():
    today = datetime.today().date()
    
    # <-- NEW: Use pathlib for portability
    BASE_DIR = Path(__file__).resolve().parents[1]
    raw_dir = BASE_DIR / "data" / "raw" / "historical_matchups"
    output_dir = BASE_DIR / "data" / "processed"

    for delta in range(1, 21):  # Last 20 days (excluding today)
        date = today - timedelta(days=delta)
        matchup_file = raw_dir / f"historical_matchups_{date}.csv"
        if matchup_file.exists():
            scrape_results_for_date(date, matchup_file, output_dir)
        else:
            logger.info(f"No matchup file found for {date}, skipping.")

def scrape_results_for_target_date(target_date):
    """Scrape results for a specific target date."""
    BASE_DIR = Path(__file__).resolve().parents[1]
    raw_dir = BASE_DIR / "data" / "raw"
    historical_matchups_dir = raw_dir / "historical_matchups"
    output_dir = BASE_DIR / "data" / "processed"
    
    # Look for matchup file for the target date in multiple locations
    matchup_file = raw_dir / f"mlb_probable_pitchers_{target_date}.csv"
    historical_matchup_file = historical_matchups_dir / f"historical_matchups_{target_date}.csv"
    
    if matchup_file.exists():
        logger.info(f"Using current matchup file: {matchup_file}")
        return scrape_results_for_date(target_date, matchup_file, output_dir)
    elif historical_matchup_file.exists():
        logger.info(f"Using historical matchup file: {historical_matchup_file}")
        return scrape_results_for_date(target_date, historical_matchup_file, output_dir)
    else:
        logger.error(f"No matchup file found for {target_date}")
        logger.error(f"Checked locations:")
        logger.error(f"  - {matchup_file}")
        logger.error(f"  - {historical_matchup_file}")
        return None

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape MLB game results')
    parser.add_argument('--date', type=str, help='Specific date to scrape (YYYY-MM-DD)')
    parser.add_argument('--rolling', action='store_true', help='Scrape rolling window of last 20 days')
    args = parser.parse_args()
    
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        scrape_results_for_target_date(target_date)
    elif args.rolling:
        run_rolling_scraper()
    else:
        # Default: scrape yesterday
        yesterday = datetime.today().date() - timedelta(days=1)
        scrape_results_for_target_date(yesterday)

# cd C:\Users\roman\baseball_forecast_project\scraping
# python scrape_game_results.py --date 2025-07-23
