import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

def test_merge():
    """Test the team offensive stats merge process"""
    
    # Load a small sample of historical data
    df = pd.read_csv("data/processed/historical_main_features.csv")
    df_sample = df.head(5).copy()
    
    logger.info(f"Sample data shape: {df_sample.shape}")
    logger.info(f"Sample game dates: {df_sample['game_date'].tolist()}")
    
    # Test the merge logic for one game date
    game_date = pd.to_datetime('2024-08-01')
    logger.info(f"Testing merge for game date: {game_date}")
    
    # Find team offensive stats files
    team_offensive_files = list(PROCESSED_DIR.glob("team_offensive_stats_*.csv"))
    logger.info(f"Found team offensive files: {[f.name for f in team_offensive_files]}")
    
    if not team_offensive_files:
        logger.warning("No team offensive stats files found")
        return
    
    # Find the most recent team offensive stats file BEFORE the game date
    closest_offensive_file = None
    max_date_before = None
    
    for offensive_file in team_offensive_files:
        try:
            file_date_str = offensive_file.stem.replace('team_offensive_stats_', '')
            file_date = pd.to_datetime(file_date_str)
            logger.info(f"File {offensive_file.name} parsed as date: {file_date}")
            
            # Only use data from BEFORE the game date
            if file_date < game_date:
                if max_date_before is None or file_date > max_date_before:
                    closest_offensive_file = offensive_file
                    max_date_before = file_date
                    logger.info(f"Selected file: {offensive_file.name} (date: {file_date})")
        except Exception as e:
            logger.error(f"Error parsing file {offensive_file.name}: {e}")
            continue
    
    if closest_offensive_file:
        logger.info(f"Using team offensive stats file from {max_date_before}: {closest_offensive_file.name}")
        
        # Load the offensive stats
        team_offensive = pd.read_csv(closest_offensive_file)
        logger.info(f"Loaded offensive stats shape: {team_offensive.shape}")
        logger.info(f"Offensive stats columns: {team_offensive.columns.tolist()}")
        logger.info(f"Team names in offensive stats: {team_offensive['team_name'].tolist()}")
        
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
        
        # Convert offensive stats to dictionary format
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
        
        # Test mapping for sample data
        logger.info(f"Sample home teams: {df_sample['home_team'].tolist()}")
        logger.info(f"Sample away teams: {df_sample['away_team'].tolist()}")
        
        # Add offensive stats to dataframe
        df_sample['home_team_offensive_stats'] = df_sample['home_team'].map(offensive_stats_dict)
        df_sample['away_team_offensive_stats'] = df_sample['away_team'].map(offensive_stats_dict)
        
        # Fill missing values with empty dictionaries
        df_sample['home_team_offensive_stats'] = df_sample['home_team_offensive_stats'].fillna({})
        df_sample['away_team_offensive_stats'] = df_sample['away_team_offensive_stats'].fillna({})
        
        logger.info(f"After merge, columns: {df_sample.columns.tolist()}")
        logger.info(f"Home team offensive stats sample: {df_sample['home_team_offensive_stats'].iloc[0]}")
        logger.info(f"Away team offensive stats sample: {df_sample['away_team_offensive_stats'].iloc[0]}")
        
    else:
        logger.warning(f"No team offensive stats file found from before date {game_date}")

if __name__ == "__main__":
    test_merge() 