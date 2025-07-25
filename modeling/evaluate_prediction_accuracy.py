# evaluate_prediction_accuracy.py

import pandas as pd
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path for imports
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

# Base directory setup
DATA_DIR = BASE_DIR / "data" / "processed"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map abbreviations to full team names for comparison
team_name_map = {
    'BOS': 'Red Sox', 'NYY': 'Yankees', 'TOR': 'Blue Jays', 'BAL': 'Orioles', 'TB': 'Rays',
    'CLE': 'Guardians', 'CWS': 'White Sox', 'KC': 'Royals', 'DET': 'Tigers', 'MIN': 'Twins',
    'HOU': 'Astros', 'SEA': 'Mariners', 'TEX': 'Rangers', 'LAA': 'Angels', 'OAK': 'Athletics',
    'ATL': 'Braves', 'MIA': 'Marlins', 'NYM': 'Mets', 'PHI': 'Phillies', 'WSH': 'Nationals',
    'MIL': 'Brewers', 'STL': 'Cardinals', 'CHC': 'Cubs', 'PIT': 'Pirates', 'CIN': 'Reds',
    'LAD': 'Dodgers', 'SF': 'Giants', 'SD': 'Padres', 'COL': 'Rockies', 'ARI': 'D-backs'
}

def ensure_actual_results_exist(pred_date: str):
    """Ensure actual results exist for the given date, scraping them if needed."""
    actual_file = DATA_DIR / f"historical_results_{pred_date}.csv"
    
    if actual_file.exists():
        logger.info(f"Actual results already exist for {pred_date}")
        return True
    
    # Check if this is a past date (we can only scrape past dates)
    pred_date_obj = datetime.strptime(pred_date, '%Y-%m-%d').date()
    today = datetime.today().date()
    
    if pred_date_obj >= today:
        logger.warning(f"Cannot scrape actual results for {pred_date} (not a past date)")
        return False
    
    # Try to scrape the actual results
    logger.info(f"Actual results not found for {pred_date}, attempting to scrape...")
    try:
        from scraping.scrape_game_results import scrape_results_for_date
        from scraping.scrape_historical_matchups import scrape_historical_matchups
        # Find the matchup file for this date
        BASE_DIR = Path(__file__).resolve().parents[1]
        raw_dir = BASE_DIR / "data" / "raw" / "historical_matchups"
        output_dir = BASE_DIR / "data" / "processed"
        matchup_file = raw_dir / f"historical_matchups_{pred_date}.csv"
        if not matchup_file.exists():
            logger.info(f"Matchup file not found for {pred_date}, generating it...")
            scrape_historical_matchups(pred_date_obj)
        if matchup_file.exists():
            scrape_results_for_date(pred_date_obj, matchup_file, output_dir)
            if actual_file.exists():
                logger.info(f"Successfully scraped actual results for {pred_date}")
                return True
            else:
                logger.error(f"Failed to scrape actual results for {pred_date}")
                return False
        else:
            logger.error(f"No matchup file found for {pred_date}, cannot scrape results.")
            return False
    except Exception as e:
        logger.error(f"Error scraping actual results for {pred_date}: {e}")
        return False

def evaluate_predictions(pred_date: str, auto_scrape=True):
    # Look for prediction files with the correct naming pattern
    # Pattern: readable_win_predictions_for_{target_date}_using_{today_date}.csv
    pred_files = list(PREDICTIONS_DIR.glob(f"readable_win_predictions_for_{pred_date}_using_*.csv"))
    
    if not pred_files:
        print(f"Missing prediction file for {pred_date}")
        print(f"Looked for pattern: readable_win_predictions_for_{pred_date}_using_*.csv")
        print(f"Available prediction files:")
        for f in PREDICTIONS_DIR.glob("readable_win_predictions_for_*.csv"):
            print(f"  - {f.name}")
        return None
    
    # Use the first matching file (there should only be one per date)
    pred_file = pred_files[0]
    print(f"Using prediction file: {pred_file.name}")
    
    actual_file = DATA_DIR / f"historical_results_{pred_date}.csv"

    # Ensure actual results exist
    if auto_scrape and not actual_file.exists():
        if not ensure_actual_results_exist(pred_date):
            print(f"Cannot evaluate {pred_date} - no actual results available")
            return None

    if not actual_file.exists():
        print(f"Missing actual results file for {pred_date}: {actual_file}")
        return None

    pred_df = pd.read_csv(pred_file)
    actual_df = pd.read_csv(actual_file)
    
    # Extract predicted winner from "Pick: TEAM" format
    pred_df['Predicted Winner'] = pred_df['Prediction'].str.replace('Pick: ', '').str.strip()
    
    # Use abbreviated team codes directly (no mapping needed)
    pred_df['home_team'] = pred_df['Home Team']
    pred_df['away_team'] = pred_df['Away Team']

    pred_df['game_date'] = pd.to_datetime(pred_df['Game Date']).dt.date
    actual_df['game_date'] = pd.to_datetime(actual_df['game_date']).dt.date

    # Normalize both DataFrames to abbreviations
    pred_df['home_team'] = pred_df['home_team'].map(team_name_map).fillna(pred_df['home_team'])
    pred_df['away_team'] = pred_df['away_team'].map(team_name_map).fillna(pred_df['away_team'])
    actual_df['home_team'] = actual_df['home_team'].map(team_name_map).fillna(actual_df['home_team'])
    actual_df['away_team'] = actual_df['away_team'].map(team_name_map).fillna(actual_df['away_team'])

    merged = pd.merge(pred_df, actual_df, on=['game_date', 'home_team', 'away_team'], how='inner')
    
    # Normalize both columns to abbreviations before comparison
    merged['Predicted Winner'] = merged['Predicted Winner'].map(team_name_map).fillna(merged['Predicted Winner'])
    merged['winner'] = merged['winner'].map(team_name_map).fillna(merged['winner'])

    merged['Correct'] = merged['Predicted Winner'] == merged['winner']

    correct = merged['Correct'].sum()
    total = len(merged)
    accuracy = round((correct / total) * 100, 2) if total > 0 else 0

    print(f"\nEvaluating Predictions for: {pred_date}")
    print(f"Total Games Evaluated: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {total - correct}")
    print(f"Accuracy: {accuracy}%\n")

    wrong = merged[~merged['Correct']]
    if not wrong.empty:
        print("Incorrect Predictions:")
        print(wrong[['game_date', 'home_team', 'away_team', 'Predicted Winner', 'winner']].to_string(index=False))

    return {
        'date': pred_date,
        'games': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy
    }

def evaluate_range(start_date: str, end_date: str, auto_scrape=True):
    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    summary = []

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        result = evaluate_predictions(date_str, auto_scrape=auto_scrape)
        if result:
            summary.append(result)
        current += timedelta(days=1)

    if summary:
        summary_df = pd.DataFrame(summary)
        print("\nOverall Accuracy Summary:")
        print(summary_df.to_string(index=False))
        log_path = DATA_DIR / "prediction_accuracy_log.csv"
        summary_df.to_csv(log_path, index=False)
        print(f"\nAccuracy log saved to: {log_path}")
    else:
        print("No valid predictions or actuals found for the selected range.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MLB prediction accuracy')
    parser.add_argument('--date', type=str, help='Specific date to evaluate (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str, help='Start date for range evaluation (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for range evaluation (YYYY-MM-DD)')
    parser.add_argument('--no-auto-scrape', action='store_true', help='Disable automatic scraping of actual results')
    args = parser.parse_args()
    
    auto_scrape = not args.no_auto_scrape
    
    if args.date:
        # Evaluate single date
        evaluate_predictions(args.date, auto_scrape=auto_scrape)
    elif args.start_date and args.end_date:
        # Evaluate date range
        evaluate_range(args.start_date, args.end_date, auto_scrape=auto_scrape)
    else:
        # Default: evaluate yesterday
        single_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"No date specified, evaluating yesterday: {single_date}")
        evaluate_predictions(single_date, auto_scrape=auto_scrape)

    # cd C:\Users\roman\baseball_forecast_project\modeling
    # python evaluate_prediction_accuracy.py --date 2025-07-22