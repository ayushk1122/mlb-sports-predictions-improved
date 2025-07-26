# evaluate_prediction_accuracy.py

import pandas as pd
import argparse
import logging
import sys
import numpy as np
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

def calculate_brier_score(predicted_probs, actual_outcomes):
    """
    Calculate the Brier score for probabilistic predictions.
    
    Args:
        predicted_probs: Array of predicted probabilities (0-1)
        actual_outcomes: Array of actual outcomes (1 for home win, 0 for away win)
    
    Returns:
        float: Brier score (lower is better, 0 is perfect)
    """
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("Predicted probabilities and actual outcomes must have the same length")
    
    # Brier score = mean((predicted_prob - actual_outcome)^2)
    brier_score = np.mean((np.array(predicted_probs) - np.array(actual_outcomes)) ** 2)
    return brier_score

def analyze_brier_score_components(predicted_probs, actual_outcomes):
    """
    Analyze Brier score components for deeper insights.
    
    Args:
        predicted_probs: Array of predicted probabilities (0-1)
        actual_outcomes: Array of actual outcomes (1 for home win, 0 for away win)
    
    Returns:
        dict: Dictionary containing Brier score analysis components
    """
    predicted_probs = np.array(predicted_probs)
    actual_outcomes = np.array(actual_outcomes)
    
    # Overall Brier score
    brier_score = np.mean((predicted_probs - actual_outcomes) ** 2)
    
    # Calibration analysis - group predictions into bins
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    calibration_errors = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
        if mask.sum() > 0:
            bin_pred = predicted_probs[mask]
            bin_actual = actual_outcomes[mask]
            avg_pred = np.mean(bin_pred)
            avg_actual = np.mean(bin_actual)
            calibration_error = (avg_pred - avg_actual) ** 2
            calibration_errors.append(calibration_error)
            bin_counts.append(mask.sum())
        else:
            calibration_errors.append(0)
            bin_counts.append(0)
    
    # Weighted average calibration error
    weighted_calibration = np.average(calibration_errors, weights=bin_counts) if sum(bin_counts) > 0 else 0
    
    # Reliability (calibration) component
    reliability = weighted_calibration
    
    # Resolution component (variance of actual outcomes)
    resolution = np.var(actual_outcomes)
    
    # Uncertainty component (variance of predictions)
    uncertainty = np.var(predicted_probs)
    
    return {
        'brier_score': brier_score,
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty,
        'calibration_errors': calibration_errors,
        'bin_counts': bin_counts
    }

def analyze_brier_score_breakdown(predicted_probs, actual_outcomes, confidence_bins=10):
    """
    Detailed breakdown of Brier score performance across different probability ranges.
    
    Args:
        predicted_probs: Array of predicted probabilities
        actual_outcomes: Array of actual outcomes (1 for home win, 0 for away win)
        confidence_bins: Number of probability bins to analyze
    
    Returns:
        DataFrame with breakdown by probability range
    """
    predicted_probs = np.array(predicted_probs)
    actual_outcomes = np.array(actual_outcomes)
    
    # Create probability bins
    bins = np.linspace(0, 1, confidence_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    breakdown_data = []
    
    for i in range(len(bins) - 1):
        mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
        if mask.sum() > 0:
            bin_pred = predicted_probs[mask]
            bin_actual = actual_outcomes[mask]
            
            # Calculate metrics for this bin
            avg_pred = np.mean(bin_pred)
            avg_actual = np.mean(bin_actual)
            brier_score = np.mean((bin_pred - bin_actual) ** 2)
            count = mask.sum()
            
            breakdown_data.append({
                'prob_range': f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                'avg_predicted': round(avg_pred, 3),
                'avg_actual': round(avg_actual, 3),
                'calibration_error': round(abs(avg_pred - avg_actual), 3),
                'brier_score': round(brier_score, 4),
                'count': count,
                'percentage': round(100 * count / len(predicted_probs), 1)
            })
    
    return pd.DataFrame(breakdown_data)

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

    # Calculate Brier score
    # Create actual outcomes: 1 if home team won, 0 if away team won
    merged['home_win'] = (merged['winner'] == merged['home_team']).astype(int)
    
    # Get predicted probabilities for home team winning
    predicted_probs = merged['Win Probability'].values
    actual_outcomes = merged['home_win'].values
    
    brier_score = calculate_brier_score(predicted_probs, actual_outcomes)
    
    # Detailed Brier score analysis
    brier_analysis = analyze_brier_score_components(predicted_probs, actual_outcomes)
    
    # Add Brier score breakdown analysis
    brier_breakdown = analyze_brier_score_breakdown(predicted_probs, actual_outcomes)

    correct = merged['Correct'].sum()
    total = len(merged)
    accuracy = round((correct / total) * 100, 2) if total > 0 else 0

    print(f"\nEvaluating Predictions for: {pred_date}")
    print(f"Total Games Evaluated: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {total - correct}")
    print(f"Accuracy: {accuracy}%")
    print(f"Brier Score: {brier_score:.4f} (lower is better, 0 is perfect)")
    print(f"Brier Score Components:")
    print(f"  - Reliability (Calibration): {brier_analysis['reliability']:.4f}")
    print(f"  - Resolution: {brier_analysis['resolution']:.4f}")
    print(f"  - Uncertainty: {brier_analysis['uncertainty']:.4f}")
    
    # Show Brier score breakdown
    print(f"\nBrier Score Breakdown by Probability Range:")
    print(brier_breakdown.to_string(index=False))
    
    print()

    wrong = merged[~merged['Correct']]
    if not wrong.empty:
        print("Incorrect Predictions:")
        print(wrong[['game_date', 'home_team', 'away_team', 'Predicted Winner', 'winner', 'Win Probability']].to_string(index=False))

    return {
        'date': pred_date,
        'games': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy,
        'brier_score': round(brier_score, 4),
        'brier_reliability': round(brier_analysis['reliability'], 4),
        'brier_resolution': round(brier_analysis['resolution'], 4),
        'brier_uncertainty': round(brier_analysis['uncertainty'], 4),
        'brier_breakdown': brier_breakdown
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
        
        # Calculate aggregate statistics
        total_games = summary_df['games'].sum()
        total_correct = summary_df['correct'].sum()
        overall_accuracy = round((total_correct / total_games) * 100, 2) if total_games > 0 else 0
        
        # Calculate weighted average Brier score (weighted by number of games)
        weighted_brier = round(
            (summary_df['brier_score'] * summary_df['games']).sum() / total_games, 4
        ) if total_games > 0 else 0
        
        # Calculate weighted average Brier components
        weighted_reliability = round(
            (summary_df['brier_reliability'] * summary_df['games']).sum() / total_games, 4
        ) if total_games > 0 else 0
        
        weighted_resolution = round(
            (summary_df['brier_resolution'] * summary_df['games']).sum() / total_games, 4
        ) if total_games > 0 else 0
        
        weighted_uncertainty = round(
            (summary_df['brier_uncertainty'] * summary_df['games']).sum() / total_games, 4
        ) if total_games > 0 else 0
        
        print(f"\nAggregate Statistics:")
        print(f"Total Games: {total_games}")
        print(f"Overall Accuracy: {overall_accuracy}%")
        print(f"Weighted Average Brier Score: {weighted_brier} (lower is better, 0 is perfect)")
        print(f"Brier Score Components:")
        print(f"  - Reliability (Calibration): {weighted_reliability}")
        print(f"  - Resolution: {weighted_resolution}")
        print(f"  - Uncertainty: {weighted_uncertainty}")
        
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
    parser.add_argument('--brier-info', action='store_true', help='Show detailed information about Brier score interpretation')
    args = parser.parse_args()
    
    if args.brier_info:
        print("BRIER SCORE INTERPRETATION:")
        print("==========================")
        print("The Brier score measures the accuracy of probabilistic predictions.")
        print("Formula: Brier Score = mean((predicted_probability - actual_outcome)²)")
        print()
        print("Interpretation:")
        print("- Range: 0 to 1 (0 = perfect predictions, 1 = worst possible predictions)")
        print("- Lower scores indicate better probabilistic accuracy")
        print("- A score of 0.25 represents random guessing (50/50 predictions)")
        print()
        print("Components:")
        print("- Reliability (Calibration): How well-calibrated the probabilities are")
        print("  (Lower is better - indicates predictions match actual frequencies)")
        print("- Resolution: How much the predictions vary (higher can be better)")
        print("- Uncertainty: Natural variability in the outcomes")
        print()
        print("Example: A Brier score of 0.15 indicates good probabilistic accuracy")
        print("         A Brier score of 0.30 indicates poor probabilistic accuracy")
        print()
        sys.exit(0)
    
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