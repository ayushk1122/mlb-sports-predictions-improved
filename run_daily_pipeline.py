# run_daily_pipeline.py

import os
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# === Component imports ===
from scraping.scrape_matchups import run_scrape_matchups
from scraping.scrape_statcast import scrape_statcast_today_or_recent
from features.build_player_event_features import build_player_event_features
from features.build_pitcher_stat_features import build_pitcher_stat_features
from utils.map_batter_ids import enrich_batter_features_by_team
from features.generate_historical_features import generate_all_historical_features  # <-- added
from features.main_features import build_main_features
from features.historical_main_features import build_historical_main_dataset
from modeling.train_model import train_model
from scraping.scrape_historical_matchups import scrape_historical_matchups
from scraping.scrape_game_results import scrape_results_for_date
from features.generate_historical_batter_stats import build_batter_stat_features
from features.generate_historical_pitcher_stats import build_pitcher_stat_features
from features.generate_historical_team_form import scrape_team_form_api_for_date
from scraping.scrape_statcast import scrape_statcast_rolling
from features.generate_historical_batter_stats import run_rolling_batter_generator

# === Logging setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Directory setup ===
BASE_DIR = Path(__file__).resolve().parent

def run_pipeline(target_date=None):
    if target_date is None:
        target_date = datetime.today().date()
    logger.info(f"Starting daily MLB prediction pipeline for {target_date}...")

    # === Calculate rolling window for historical features ===
    window_end = target_date - timedelta(days=1)
    window_dates = [window_end - timedelta(days=i) for i in range(45)]
    window_dates = sorted(window_dates)

    # === Step 0: Generate all historical feature files for the window ===
    BASE_DIR = Path(__file__).resolve().parent
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    HISTORICAL_MATCHUP_DIR = RAW_DIR / "historical_matchups"

    # Generate all statcast files for the window first
    scrape_statcast_rolling(days_back=45, target_date=target_date, overwrite=False)

    # Generate all batter stat features for the window
    run_rolling_batter_generator(n_days=45)

    for date in window_dates:
        date_str = date.strftime('%Y-%m-%d')
        # 1. Historical matchups
        matchup_file = HISTORICAL_MATCHUP_DIR / f"historical_matchups_{date_str}.csv"
        if not matchup_file.exists():
            logger.info(f"Generating matchup file for {date_str}")
            try:
                scrape_historical_matchups(date)
            except Exception as e:
                logger.warning(f"Failed to generate matchup file for {date_str}: {e}")
        # 2. Historical results
        result_file = PROCESSED_DIR / f"historical_results_{date_str}.csv"
        if not result_file.exists():
            logger.info(f"Generating results file for {date_str}")
            try:
                scrape_results_for_date(date, matchup_file, PROCESSED_DIR)
            except Exception as e:
                logger.warning(f"Failed to generate results file for {date_str}: {e}")
        # 3. Pitcher stats
        pitcher_file = PROCESSED_DIR / f"pitcher_stat_features_{date_str}.csv"
        if not pitcher_file.exists() and matchup_file.exists():
            logger.info(f"Generating pitcher stats for {date_str}")
            try:
                build_pitcher_stat_features(matchup_file)
            except Exception as e:
                logger.warning(f"Failed to generate pitcher stats for {date_str}: {e}")
        # 4. Team form
        team_form_file = PROCESSED_DIR / f"team_form_{date_str}.csv"
        if not team_form_file.exists():
            logger.info(f"Generating team form for {date_str}")
            try:
                scrape_team_form_api_for_date(date_str)
            except Exception as e:
                logger.warning(f"Failed to generate team form for {date_str}: {e}")

    # === Step 1: Scrape today's matchups ===
    try:
        logger.info(f"Step 1: Scraping MLB matchups for {target_date}...")
        matchup_csv_path, scraped_game_date = run_scrape_matchups(target_date=target_date)
        scraped_game_date_str = scraped_game_date.strftime('%Y-%m-%d')
        logger.info(f"Matchups scraped and saved to: {matchup_csv_path}")
    except Exception as e:
        logger.error(f"Failed to scrape matchups: {e}")
        return

    # === Step 2: Scrape recent Statcast data ===
    statcast_file, statcast_actual_date = scrape_statcast_today_or_recent(n_days=3, target_date=target_date)
    if not statcast_file:
        logger.error("No Statcast data found. Exiting pipeline.")
        return
    logger.info(f"Using Statcast data from: {statcast_actual_date}")

    # === Step 3: Player-level features ===
    player_feature_file = build_player_event_features(statcast_file)
    if not player_feature_file:
        logger.error("Failed to build player-level features.")
        return

    # === Step 4: Aggregate batter stats by team ===
    try:
        team_feature_file = enrich_batter_features_by_team(player_feature_file, matchup_csv_path, target_date=target_date)
        if not team_feature_file:
            logger.error("Failed to build team batter features.")
            return
        logger.info(f"Team batter features saved to: {team_feature_file}")
    except Exception as e:
        logger.error(f"Team aggregation failed: {e}")
        return

    # === Step 5: Aggregate pitcher stats ===
    try:
        pitcher_feature_file = build_pitcher_stat_features(matchup_csv_path)
        if not pitcher_feature_file:
            logger.error("Failed to build pitcher features.")
            return
        logger.info(f"Pitcher stats saved to: {pitcher_feature_file}")
    except Exception as e:
        logger.error(f"Pitcher aggregation failed: {e}")
        return

    # === Step 6: Build today's main features file ===
    try:
        logger.info(f"Building main features for {target_date} matchups...")
        main_features_path = build_main_features(matchup_csv_path, pitcher_feature_file, team_feature_file)
        logger.info(f"Main features saved to: {main_features_path}")
    except Exception as e:
        logger.error(f"Failed to build main features: {e}")
        return

    # === Step 6A: Generate historical features before building training dataset ===
    try:
        logger.info("Step 6A: Generating all historical feature files...")
        # No-op: already generated above
        logger.info("All historical features generated.")
    except Exception as e:
        logger.error(f"Failed to generate historical features: {e}")
        return

    # === Step 6B: Rebuild historical training dataset ===
    try:
        logger.info("Step 6B: Rebuilding historical training dataset...")
        build_historical_main_dataset(days_back=45, target_date=target_date)
        logger.info("Updated historical_main_features.csv for model training.")
    except Exception as e:
        logger.error(f"Failed to update historical dataset: {e}")
        return

    # === Step 6C: Check if we have enough historical data for training ===
    try:
        historical_path = os.path.join("data", "processed", "historical_main_features.csv")
        if not os.path.exists(historical_path):
            logger.error("Historical dataset not found. Cannot train model without historical data.")
            return
        
        historical_df = pd.read_csv(historical_path)
        if len(historical_df) == 0:
            logger.error("Historical dataset is empty. Cannot train model without historical data.")
            return
        
        if "actual_winner" not in historical_df.columns:
            logger.error("Historical dataset missing 'actual_winner' column. Cannot train model without actual results.")
            return
        
        logger.info(f"Historical dataset ready for training with {len(historical_df)} games")
    except Exception as e:
        logger.error(f"Error checking historical dataset: {e}")
        return

    # === Step 7: Train model and generate predictions ===
    try:
        today_path = main_features_path
        predictions_df = train_model(historical_path, today_path, target_date=target_date)
    except Exception as e:
        logger.error(f"Error during model training or prediction: {e}")
        return

    # === Step 8: Filter predictions for today's matchups ===
    try:
        matchups = pd.read_csv(matchup_csv_path)
        matchups.dropna(subset=["home_team", "away_team"], inplace=True)
        matchups.drop_duplicates(subset=["game_date", "home_team", "away_team"], inplace=True)
        matchups['game_date'] = pd.to_datetime(matchups['game_date'], errors='coerce').dt.date

        matchups_target = matchups[matchups['game_date'] == target_date].copy()

        if matchups_target.empty:
            logger.warning(f"No matchups found for {target_date}")
            return

        translation_dict = {
            'RED SOX': 'BOS', 'YANKEES': 'NYY', 'BLUE JAYS': 'TOR', 'ORIOLES': 'BAL', 'RAYS': 'TB',
            'GUARDIANS': 'CLE', 'WHITE SOX': 'CHW', 'ROYALS': 'KC', 'TIGERS': 'DET', 'TWINS': 'MIN',
            'ASTROS': 'HOU', 'MARINERS': 'SEA', 'RANGERS': 'TEX', 'ANGELS': 'LAA', 'ATHLETICS': 'OAK',
            'BRAVES': 'ATL', 'MARLINS': 'MIA', 'METS': 'NYM', 'PHILLIES': 'PHI', 'NATIONALS': 'WSH',
            'BREWERS': 'MIL', 'CARDINALS': 'STL', 'CUBS': 'CHC', 'PIRATES': 'PIT', 'REDS': 'CIN',
            'DODGERS': 'LAD', 'GIANTS': 'SF', 'PADRES': 'SD', 'ROCKIES': 'COL', 'DIAMONDBACKS': 'ARI',
            'D-BACKS': 'ARI', 'ATLÉTICOS': 'OAK', 'AZULEJOS': 'TOR', 'CARDENALES': 'STL',
            'CERVECEROS': 'MIL', 'GIGANTES': 'SF', 'MARINEROS': 'SEA', 'NACIONALES': 'WSH',
            'PIRATAS': 'PIT', 'REALES': 'KC', 'ROJOS': 'CIN', 'TIGRES': 'DET', 'CACHORROS': 'CHC'
        }

        def normalize(name):
            return translation_dict.get(name.strip().upper(), name.strip().upper())

        matchups_target['home_team'] = matchups_target['home_team'].astype(str).apply(normalize)
        matchups_target['away_team'] = matchups_target['away_team'].astype(str).apply(normalize)
        matchups_target['matchup_key'] = matchups_target['home_team'] + "_" + matchups_target['away_team']

        predictions_df['matchup_key'] = predictions_df['Home Team'] + "_" + predictions_df['Away Team']
        filtered = predictions_df[predictions_df['matchup_key'].isin(matchups_target['matchup_key'])].copy()
        filtered = filtered.merge(matchups_target[['matchup_key', 'game_date']], on='matchup_key', how='left')
        filtered.drop(columns=['matchup_key'], inplace=True)

        filtered_path = BASE_DIR / "data" / "predictions" / f"today_and_tomorrow_predictions.csv"
        filtered.to_csv(filtered_path, index=False)
        logger.info(f"Filtered predictions saved to: {filtered_path}")
    except Exception as e:
        logger.error(f"Failed to filter predictions: {e}")
        return

    # === Step 9: Format final readable output ===
    try:
        df = pd.read_csv(filtered_path)

        readable = df[['game_date', 'Home Team', 'Away Team', 'Win Probability']].copy()
        readable['Win Probability'] = readable['Win Probability'].round(2)
        readable['Prediction'] = readable.apply(
            lambda row: f"Pick: {row['Home Team']}" if row['Win Probability'] >= 0.5 else f"Pick: {row['Away Team']}",
            axis=1
        )

        readable = readable.rename(columns={
            'game_date': 'Game Date',
            'Home Team': 'Home Team',
            'Away Team': 'Away Team',
            'Win Probability': 'Win Probability',
            'Prediction': 'Prediction'
        })

        readable.sort_values(by=['Game Date', 'Home Team'], inplace=True)
        readable.drop_duplicates(subset=['Game Date', 'Home Team', 'Away Team'], inplace=True)

        if isinstance(statcast_actual_date, str):
            statcast_actual_date = datetime.strptime(statcast_actual_date, "%Y-%m-%d").date()

        output_name = f"readable_win_predictions_for_{scraped_game_date_str}_using_{statcast_actual_date.strftime('%Y-%m-%d')}.csv"
        readable_path = BASE_DIR / "data" / "predictions" / output_name
        # readable_path = os.path.join("data", "predictions", output_name)
        readable.to_csv(readable_path, index=False)
        logger.info(f"Clean, deduplicated predictions saved to: {readable_path}")
    except Exception as e:
        logger.error(f"Failed to format readable predictions: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MLB prediction pipeline for a specific date')
    parser.add_argument('--date', type=str, help='Date to run pipeline for (YYYY-MM-DD format). If not provided, uses today.')
    args = parser.parse_args()
    
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        target_date = None
    
    run_pipeline(target_date=target_date)

# cd C:\Users\roman\baseball_forecast_project
# python run_daily_pipeline.py --date 2025-07-23
