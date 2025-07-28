import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the integration functions
import sys
sys.path.append('features')
from integrate_enhanced_features import merge_team_data, engineer_enhanced_features

def test_integration():
    """Test the integration process step by step"""
    
    # Load a small sample of historical data
    df = pd.read_csv('data/processed/historical_main_features.csv')
    df_sample = df.head(10).copy()
    
    logger.info(f"Original data shape: {df_sample.shape}")
    logger.info(f"Original columns: {len(df_sample.columns)}")
    
    # Step 1: Test merge_team_data
    logger.info("\n=== STEP 1: Testing merge_team_data ===")
    game_date = pd.to_datetime('2024-08-01')
    df_merged = merge_team_data(df_sample, game_date)
    
    logger.info(f"After merge shape: {df_merged.shape}")
    logger.info(f"After merge columns: {len(df_merged.columns)}")
    
    # Check if offensive stats columns are present
    if 'home_team_offensive_stats' in df_merged.columns:
        logger.info("✓ home_team_offensive_stats found")
        home_stats = df_merged['home_team_offensive_stats'].iloc[0]
        logger.info(f"Home stats type: {type(home_stats)}")
        if isinstance(home_stats, dict):
            logger.info(f"Home stats keys: {list(home_stats.keys())}")
    else:
        logger.error("✗ home_team_offensive_stats NOT found")
    
    if 'away_team_offensive_stats' in df_merged.columns:
        logger.info("✓ away_team_offensive_stats found")
        away_stats = df_merged['away_team_offensive_stats'].iloc[0]
        logger.info(f"Away stats type: {type(away_stats)}")
        if isinstance(away_stats, dict):
            logger.info(f"Away stats keys: {list(away_stats.keys())}")
    else:
        logger.error("✗ away_team_offensive_stats NOT found")
    
    # Step 2: Test feature engineering
    logger.info("\n=== STEP 2: Testing feature engineering ===")
    df_enhanced = engineer_enhanced_features(df_merged)
    
    logger.info(f"After feature engineering shape: {df_enhanced.shape}")
    logger.info(f"After feature engineering columns: {len(df_enhanced.columns)}")
    
    # Check for offensive features
    offensive_features = [col for col in df_enhanced.columns if 'team_' in col and ('batting_avg' in col or 'obp' in col or 'slg' in col or 'ops' in col)]
    logger.info(f"Offensive features found: {len(offensive_features)}")
    logger.info(f"Offensive features: {offensive_features[:5]}")
    
    # Check if offensive stats columns are still there
    if 'home_team_offensive_stats' in df_enhanced.columns:
        logger.info("✓ home_team_offensive_stats still present after feature engineering")
    else:
        logger.info("✗ home_team_offensive_stats removed by feature engineering")
    
    if 'away_team_offensive_stats' in df_enhanced.columns:
        logger.info("✓ away_team_offensive_stats still present after feature engineering")
    else:
        logger.info("✗ away_team_offensive_stats removed by feature engineering")
    
    # Step 3: Save and reload to test persistence
    logger.info("\n=== STEP 3: Testing save and reload ===")
    test_output_path = 'data/processed/test_enhanced.csv'
    df_enhanced.to_csv(test_output_path, index=False)
    
    # Reload the saved file
    df_reloaded = pd.read_csv(test_output_path)
    logger.info(f"Reloaded data shape: {df_reloaded.shape}")
    logger.info(f"Reloaded columns: {len(df_reloaded.columns)}")
    
    # Check offensive features in reloaded data
    offensive_features_reloaded = [col for col in df_reloaded.columns if 'team_' in col and ('batting_avg' in col or 'obp' in col or 'slg' in col or 'ops' in col)]
    logger.info(f"Offensive features in reloaded data: {len(offensive_features_reloaded)}")
    
    # Clean up test file
    import os
    os.remove(test_output_path)
    
    return df_enhanced

if __name__ == "__main__":
    test_integration() 