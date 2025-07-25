# historical_main_features.py

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === PATH SETUP ===
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
HISTORICAL_MATCHUP_DIR = RAW_DIR / "historical_matchups"

RAW_OUTPUT_PATH = PROCESSED_DIR / "historical_main_features_raw.csv"
CLEAN_OUTPUT_PATH = PROCESSED_DIR / "historical_main_features.csv"

# === TEAM ABBREVIATION MAP ===
TEAM_ABBREV_MAP = {
    "RED SOX": "BOS", "YANKEES": "NYY", "BLUE JAYS": "TOR", "RAYS": "TBR", "ORIOLES": "BAL",
    "WHITE SOX": "CHW", "GUARDIANS": "CLE", "TIGERS": "DET", "ROYALS": "KCR", "TWINS": "MIN",
    "ASTROS": "HOU", "MARINERS": "SEA", "RANGERS": "TEX", "ATHLETICS": "OAK", "ANGELS": "LAA",
    "BRAVES": "ATL", "PHILLIES": "PHI", "METS": "NYM", "MARLINS": "MIA", "NATIONALS": "WSH",
    "CUBS": "CHC", "REDS": "CIN", "PIRATES": "PIT", "BREWERS": "MIL", "CARDINALS": "STL",
    "DODGERS": "LAD", "GIANTS": "SFG", "ROCKIES": "COL", "PADRES": "SDP", "DIAMONDBACKS": "ARI",
    "D-BACKS": "ARI"
}

def normalize_name(name):
    if pd.isna(name):
        return ""
    return (
        name.upper().strip()
            .replace("Á", "A").replace("É", "E").replace("Í", "I")
            .replace("Ó", "O").replace("Ú", "U").replace("Ñ", "N")
            .replace(".", "")
    )

def map_abbrev(team):
    return TEAM_ABBREV_MAP.get(team.upper().strip(), team.upper().strip())

def load_csv_by_prefix(prefix, date_str, directory=PROCESSED_DIR):
    path = directory / f"{prefix}_{date_str}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            if df.empty:
                logger.warning(f"{prefix} for {date_str} is empty (0 rows).")
                return None
            return df
        except Exception as e:
            logger.error(f"Failed to load {prefix} for {date_str}: {e}")
            return None
    else:
        logger.warning(f"Missing {prefix} for {date_str} - file does not exist: {path}")
        return None

def build_historical_main_dataset(days_back=45, target_date=None):
    logger.info(f"=== STARTING HISTORICAL MAIN DATASET BUILD for last {days_back} days ===")
    if target_date is None:
        base_date = datetime.today().date()
    else:
        base_date = target_date
    all_rows = []
    processed_dates = []
    skipped_dates = []
    for i in range(1, days_back + 1):
        date = base_date - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"\n=== Processing game date: {date_str} ===")
        # Build file paths for this date
        result_file = PROCESSED_DIR / f"historical_results_{date_str}.csv"
        matchup_file = HISTORICAL_MATCHUP_DIR / f"historical_matchups_{date_str}.csv"
        pitcher_file = PROCESSED_DIR / f"pitcher_stat_features_{date_str}.csv"
        batter_file = PROCESSED_DIR / f"batter_stat_features_{date_str}.csv"
        team_form_file = PROCESSED_DIR / f"team_form_{date_str}.csv"
        # Try to load all files
        try:
            results_df = pd.read_csv(result_file)
            matchup_df = pd.read_csv(matchup_file)
            pitcher_df = pd.read_csv(pitcher_file)
            batter_df = pd.read_csv(batter_file)
            team_df = pd.read_csv(team_form_file)
        except Exception as e:
            logger.warning(f"Skipping {date_str} due to missing or invalid files: {e}")
            skipped_dates.append(date_str)
            continue
        try:
            # Normalize team and player names
            matchup_df["home_pitcher"] = matchup_df["home_pitcher"].apply(normalize_name)
            matchup_df["away_pitcher"] = matchup_df["away_pitcher"].apply(normalize_name)
            matchup_df["home_team"] = matchup_df["home_team"].str.upper().str.strip()
            matchup_df["away_team"] = matchup_df["away_team"].str.upper().str.strip()
            pitcher_df["full_name"] = pitcher_df["full_name"].apply(normalize_name)
            # Merge pitcher stats
            df = matchup_df.merge(pitcher_df, left_on="home_pitcher", right_on="full_name", how="left")
            df = df.rename(columns={col: f"home_pitcher_{col}" for col in pitcher_df.columns if col != "full_name"})
            df.drop(columns=["full_name"], inplace=True)
            df = df.merge(pitcher_df, left_on="away_pitcher", right_on="full_name", how="left")
            df = df.rename(columns={col: f"away_pitcher_{col}" for col in pitcher_df.columns if col != "full_name"})
            df.drop(columns=["full_name"], inplace=True)
            # Merge batter stats
            batter_df["team"] = batter_df["team"].str.upper().str.strip()
            df = df.merge(batter_df, left_on="home_team", right_on="team", how="left")
            df = df.rename(columns={col: f"home_team_{col}" for col in batter_df.columns if col != "team"})
            df.drop(columns=["team"], inplace=True)
            df = df.merge(batter_df, left_on="away_team", right_on="team", how="left")
            df = df.rename(columns={col: f"away_team_{col}" for col in batter_df.columns if col != "team"})
            df.drop(columns=["team"], inplace=True)
            # Merge team form
            team_df["team"] = team_df["team"].str.upper().str.strip()
            df = df.merge(team_df.add_prefix("home_"), left_on="home_team", right_on="home_team", how="left")
            df = df.merge(team_df.add_prefix("away_"), left_on="away_team", right_on="away_team", how="left")
            # Normalize and merge actual results
            results_df["home_team"] = results_df["home_team"].str.upper().str.strip()
            results_df["away_team"] = results_df["away_team"].str.upper().str.strip()
            results_df["winner"] = results_df["winner"].str.upper().str.strip()
            df = df.merge(results_df, on=["game_date", "home_team", "away_team"], how="inner")
            df["actual_winner"] = df["winner"]
            all_rows.append(df)
            processed_dates.append(date_str)
        except Exception as e:
            logger.error(f"Merge failed for {date_str}: {e}")
            skipped_dates.append(date_str)
            continue
    logger.info(f"\n=== PROCESSING SUMMARY ===")
    logger.info(f"Processed dates: {processed_dates}")
    logger.info(f"Skipped dates: {skipped_dates}")
    logger.info(f"Total rows collected: {sum(len(df) for df in all_rows)}")
    if not all_rows:
        logger.warning("No rows processed. Final dataset was not created.")
        return
    logger.info("Concatenating all rows...")
    final_df = pd.concat(all_rows, ignore_index=True)
    logger.info(f"Final dataset shape: {final_df.shape}")
    final_df.to_csv(RAW_OUTPUT_PATH, index=False)
    logger.info(f"Saved raw dataset to {RAW_OUTPUT_PATH}")
    # CLEANING
    logger.info("Starting data cleaning...")
    initial_rows = len(final_df)
    final_df.drop_duplicates(subset=["game_date", "home_team", "away_team"], inplace=True)
    logger.info(f"After dropping duplicates: {len(final_df)} rows (removed {initial_rows - len(final_df)})")
    final_df.dropna(axis=1, how="all", inplace=True)
    logger.info(f"After dropping all-NaN columns: {len(final_df.columns)} columns")
    for col in final_df.select_dtypes(include=["float64", "int64"]).columns:
        final_df[col] = final_df[col].fillna(final_df[col].mean())
    final_df = final_df[final_df["actual_winner"].notna()]
    logger.info(f"After removing rows with no actual_winner: {len(final_df)} rows")
    final_df["home_team"] = final_df["home_team"].map(map_abbrev)
    final_df["away_team"] = final_df["away_team"].map(map_abbrev)
    final_df["actual_winner"] = final_df["actual_winner"].map(map_abbrev)
    final_df.to_csv(CLEAN_OUTPUT_PATH, index=False)
    logger.info(f"Saved clean final dataset to {CLEAN_OUTPUT_PATH} with {len(final_df)} rows.")
    logger.info(f"Dataset date range: {final_df['game_date'].min()} to {final_df['game_date'].max()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days_back", type=int, default=45, help="How many days back to include in the rolling window.")
    parser.add_argument("--target_date", type=str, default=None, help="End date for the rolling window (YYYY-MM-DD). Defaults to today.")
    args = parser.parse_args()
    target_date = datetime.strptime(args.target_date, "%Y-%m-%d").date() if args.target_date else None
    build_historical_main_dataset(days_back=args.days_back, target_date=target_date)
