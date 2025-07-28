# integrate_enhanced_features.py

import pandas as pd
import logging
from pathlib import Path
from enhanced_feature_engineering import engineer_enhanced_features
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def merge_team_data(df, game_date):
    """
    Merge team form, team batter stats, and team offensive stats data for a specific game date.
    IMPORTANT: Only use data from BEFORE the game date to prevent data leakage.
    """
    logger.info(f"Merging team data for game date: {game_date}")
    
    # Convert game_date to datetime if it's a string
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date)
    
    # Find team form file from BEFORE the game date (not including the game date)
    team_form_files = list(PROCESSED_DIR.glob("team_form_*.csv"))
    if not team_form_files:
        logger.warning("No team form files found")
    else:
        # Find the most recent team form file BEFORE the game date
        closest_form_file = None
        max_date_before = None
        
        for form_file in team_form_files:
            try:
                file_date_str = form_file.stem.replace('team_form_', '')
                file_date = pd.to_datetime(file_date_str)
                
                # Only use data from BEFORE the game date
                if file_date < game_date:
                    if max_date_before is None or file_date > max_date_before:
                        closest_form_file = form_file
                        max_date_before = file_date
            except:
                continue
        
        if closest_form_file:
            logger.info(f"Using team form file from {max_date_before}: {closest_form_file.name}")
            team_form = pd.read_csv(closest_form_file)
            team_form['team'] = team_form['team'].str.upper().str.strip()
            
            # Merge home team form
            df = df.merge(team_form.add_prefix('home_'), left_on='home_team', right_on='home_team', how='left')
            # Merge away team form
            df = df.merge(team_form.add_prefix('away_'), left_on='away_team', right_on='away_team', how='left')
        else:
            logger.warning(f"No team form file found from before date {game_date}")
    
    # Find team batter stats file from BEFORE the game date
    team_batter_files = list(PROCESSED_DIR.glob("team_batter_stats_*.csv"))
    if not team_batter_files:
        logger.warning("No team batter stats files found")
    else:
        # Find the most recent team batter stats file BEFORE the game date
        closest_batter_file = None
        max_date_before = None
        
        for batter_file in team_batter_files:
            try:
                file_date_str = batter_file.stem.replace('team_batter_stats_', '')
                file_date = pd.to_datetime(file_date_str)
                
                # Only use data from BEFORE the game date
                if file_date < game_date:
                    if max_date_before is None or file_date > max_date_before:
                        closest_batter_file = batter_file
                        max_date_before = file_date
            except:
                continue
        
        if closest_batter_file:
            logger.info(f"Using team batter stats file from {max_date_before}: {closest_batter_file.name}")
            team_batter = pd.read_csv(closest_batter_file)
            team_batter['team_name'] = team_batter['team_name'].str.upper().str.strip()
            
            # Merge home team batter stats
            df = df.merge(team_batter.add_prefix('home_team_'), left_on='home_team', right_on='home_team_team_name', how='left')
            df.drop(columns=['home_team_team_name'], inplace=True, errors='ignore')
            
            # Merge away team batter stats
            df = df.merge(team_batter.add_prefix('away_team_'), left_on='away_team', right_on='away_team_team_name', how='left')
            df.drop(columns=['away_team_team_name'], inplace=True, errors='ignore')
        else:
            logger.warning(f"No team batter stats file found from before date {game_date}")
    
    # Find team offensive stats file from BEFORE the game date
    team_offensive_files = list(PROCESSED_DIR.glob("team_offensive_stats_*.csv"))
    if not team_offensive_files:
        logger.warning("No team offensive stats files found")
    else:
        # Find the most recent team offensive stats file BEFORE the game date
        closest_offensive_file = None
        max_date_before = None
        
        for offensive_file in team_offensive_files:
            try:
                file_date_str = offensive_file.stem.replace('team_offensive_stats_', '')
                file_date = pd.to_datetime(file_date_str)
                
                # Only use data from BEFORE the game date
                if file_date < game_date:
                    if max_date_before is None or file_date > max_date_before:
                        closest_offensive_file = offensive_file
                        max_date_before = file_date
            except:
                continue
        
        if closest_offensive_file:
            logger.info(f"Using team offensive stats file from {max_date_before}: {closest_offensive_file.name}")
            team_offensive = pd.read_csv(closest_offensive_file)
            team_offensive['team_name'] = team_offensive['team_name'].str.upper().str.strip()
            
            # Team name mapping from offensive stats to historical data format
            team_name_mapping = {
                'KC': 'KCR',    # Kansas City Royals
                'TB': 'TBR',    # Tampa Bay Rays
                'SD': 'SDP',    # San Diego Padres
                'SF': 'SFG',    # San Francisco Giants
                'CWS': 'CHW',   # Chicago White Sox
                'WSH': 'WSN'    # Washington Nationals
            }
            
            logger.info(f"Team name mapping: {team_name_mapping}")
            
            # Convert offensive stats to dictionary format for easier access
            offensive_stats_dict = {}
            for _, row in team_offensive.iterrows():
                team = row['team_name']
                # Map team name if needed
                mapped_team = team_name_mapping.get(team, team)
                logger.info(f"Mapping team {team} -> {mapped_team}")
                
                stats = {
                    'batting_avg': row.get('batting_avg', 0),
                    'on_base_pct': row.get('on_base_pct', 0),
                    'slugging_pct': row.get('slugging_pct', 0),
                    'ops': row.get('ops', 0),
                    'iso': row.get('iso', 0),
                    'hr_per_game': row.get('hr_per_game', 0),
                    'k_rate': row.get('k_rate', 0),
                    'bb_rate': row.get('bb_rate', 0),
                    'sb_success_rate': row.get('sb_success_rate', 0),
                    'runs_per_game': row.get('runs_per_game', 0),
                    'hits': row.get('hits', 0),
                    'doubles': row.get('doubles', 0),
                    'triples': row.get('triples', 0),
                    'rbi': row.get('rbi', 0),
                    'total_bases': row.get('total_bases', 0),
                    'games': row.get('games', 1)
                }
                offensive_stats_dict[mapped_team] = stats
            
            logger.info(f"Created stats dict for teams: {list(offensive_stats_dict.keys())}")
            logger.info(f"Sample home teams: {df['home_team'].unique()[:5].tolist()}")
            logger.info(f"Sample away teams: {df['away_team'].unique()[:5].tolist()}")
            
            # Add offensive stats to dataframe
            df['home_team_offensive_stats'] = df['home_team'].map(offensive_stats_dict)
            df['away_team_offensive_stats'] = df['away_team'].map(offensive_stats_dict)
            
            # Fill missing values with empty dictionaries
            df['home_team_offensive_stats'] = df['home_team_offensive_stats'].fillna({})
            df['away_team_offensive_stats'] = df['away_team_offensive_stats'].fillna({})
            
            # Check mapping results
            home_missing = df[df['home_team_offensive_stats'] == {}]['home_team'].unique()
            away_missing = df[df['away_team_offensive_stats'] == {}]['away_team'].unique()
            logger.info(f"Home teams missing stats: {home_missing}")
            logger.info(f"Away teams missing stats: {away_missing}")
        else:
            logger.warning(f"No team offensive stats file found from before date {game_date}")
    
    return df

def enhance_main_features(main_features_path):
    """
    Enhance main features with the new engineered features.
    """
    logger.info(f"Enhancing main features from: {main_features_path}")
    
    # Load the main features
    df = pd.read_csv(main_features_path)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} features")
    
    # Apply enhanced feature engineering
    df_enhanced = engineer_enhanced_features(df)
    
    # Save enhanced features
    enhanced_path = main_features_path.parent / f"{main_features_path.stem}_enhanced.csv"
    df_enhanced.to_csv(enhanced_path, index=False)
    logger.info(f"Enhanced features saved to: {enhanced_path}")
    
    return enhanced_path

def enhance_historical_features_with_team_data(historical_path):
    """
    Enhance historical features with team data and new engineered features.
    """
    logger.info(f"Enhancing historical features from: {historical_path}")
    
    # Load the historical features
    df = pd.read_csv(historical_path)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} features")
    
    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Group by game date and merge team data for each date
    enhanced_dfs = []
    
    for game_date in df['game_date'].unique():
        date_df = df[df['game_date'] == game_date].copy()
        date_df = merge_team_data(date_df, game_date)
        enhanced_dfs.append(date_df)
    
    # Combine all enhanced dataframes
    df_enhanced = pd.concat(enhanced_dfs, ignore_index=True)
    logger.info(f"After merging team data: {len(df_enhanced)} rows with {len(df_enhanced.columns)} features")
    
    # Apply enhanced feature engineering
    df_enhanced = engineer_enhanced_features(df_enhanced)
    
    # Reorder columns to match original structure
    # Start with basic identifiers and core features
    core_columns = [
        'game_date', 'away_team', 'home_team', 'away_pitcher', 'home_pitcher',
        'winner', 'actual_winner'
    ]
    
    # Get all other columns (excluding core columns)
    other_columns = [col for col in df_enhanced.columns if col not in core_columns]
    
    # Sort other columns alphabetically for consistency
    other_columns.sort()
    
    # Combine core columns first, then sorted other columns
    final_column_order = core_columns + other_columns
    
    # Reorder the dataframe
    df_enhanced = df_enhanced[final_column_order]
    
    logger.info(f"Reordered columns: {len(df_enhanced.columns)} total columns")
    logger.info(f"First 10 columns: {df_enhanced.columns[:10].tolist()}")
    
    # Save enhanced features
    enhanced_path = historical_path.parent / f"{historical_path.stem}_enhanced.csv"
    
    # Force overwrite by removing existing file first
    if enhanced_path.exists():
        enhanced_path.unlink()
        logger.info(f"Removed existing enhanced file: {enhanced_path}")
    
    df_enhanced.to_csv(enhanced_path, index=False)
    logger.info(f"Enhanced historical features saved to: {enhanced_path}")
    
    # Verify the file was saved correctly
    if enhanced_path.exists():
        verification_df = pd.read_csv(enhanced_path)
        offensive_cols = [col for col in verification_df.columns if 'team_batting_avg_diff' in col or 'team_obp_diff' in col]
        logger.info(f"Verification: Found {len(offensive_cols)} offensive features in saved file")
        logger.info(f"Verification: Total columns in saved file: {len(verification_df.columns)}")
    else:
        logger.error(f"ERROR: Enhanced file was not created at {enhanced_path}")
    
    return enhanced_path

def update_train_model_to_use_enhanced_features():
    """
    Update train_model.py to use enhanced features by default.
    """
    logger.info("Updating train_model.py to use enhanced features...")
    
    train_model_path = BASE_DIR / "modeling" / "train_model.py"
    
    if not train_model_path.exists():
        logger.error(f"train_model.py not found at: {train_model_path}")
        return False
    
    # Read the current train_model.py
    with open(train_model_path, 'r') as f:
        content = f.read()
    
    # Update the default historical path to use enhanced features
    old_path = 'args.historical = PROCESSED_DIR / "historical_main_features.csv"'
    new_path = 'args.historical = PROCESSED_DIR / "historical_main_features_enhanced.csv"'
    
    if old_path in content:
        content = content.replace(old_path, new_path)
        logger.info("Updated historical features path to use enhanced features")
    else:
        logger.warning("Could not find historical features path to update")
    
    # Write the updated content back
    with open(train_model_path, 'w') as f:
        f.write(content)
    
    logger.info("train_model.py updated successfully")
    return True

def create_enhanced_feature_pipeline():
    """
    Create a complete enhanced feature pipeline.
    """
    logger.info("Creating enhanced feature pipeline...")
    
    # Enhance historical features with team data
    historical_path = PROCESSED_DIR / "historical_main_features.csv"
    if historical_path.exists():
        enhance_historical_features_with_team_data(historical_path)
    else:
        logger.warning(f"Historical features not found: {historical_path}")
    
    # Update train_model.py
    update_train_model_to_use_enhanced_features()
    
    logger.info("Enhanced feature pipeline created successfully")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate enhanced features into the pipeline')
    parser.add_argument('--main-features', type=str, help='Path to main features file to enhance')
    parser.add_argument('--historical-features', type=str, help='Path to historical features file to enhance')
    parser.add_argument('--update-pipeline', action='store_true', help='Update train_model.py to use enhanced features')
    parser.add_argument('--full-pipeline', action='store_true', help='Run complete enhanced feature pipeline')
    
    args = parser.parse_args()
    
    if args.full_pipeline:
        create_enhanced_feature_pipeline()
    elif args.main_features:
        enhance_main_features(Path(args.main_features))
    elif args.historical_features:
        enhance_historical_features_with_team_data(Path(args.historical_features))
    elif args.update_pipeline:
        update_train_model_to_use_enhanced_features()
    else:
        # Default: enhance historical features
        historical_path = PROCESSED_DIR / "historical_main_features.csv"
        if historical_path.exists():
            enhance_historical_features_with_team_data(historical_path)
        else:
            logger.error("No historical features found and no specific file provided") 