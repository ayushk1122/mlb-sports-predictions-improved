import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the feature engineering functions
import sys
sys.path.append('features')
from enhanced_feature_engineering import add_team_offensive_metrics, add_team_strength_indicators

def test_offensive_features():
    """Test if offensive features are being created properly"""
    
    # Load a small sample of the enhanced data
    df = pd.read_csv('data/processed/historical_main_features_enhanced.csv')
    df_sample = df.head(3).copy()
    
    logger.info(f"Sample data shape: {df_sample.shape}")
    logger.info(f"Columns before offensive features: {df_sample.columns.tolist()}")
    
    # Check if offensive stats columns exist
    if 'home_team_offensive_stats' in df_sample.columns and 'away_team_offensive_stats' in df_sample.columns:
        logger.info("✓ Offensive stats columns found")
        
        # Check the content of offensive stats
        home_stats = df_sample['home_team_offensive_stats'].iloc[0]
        away_stats = df_sample['away_team_offensive_stats'].iloc[0]
        
        logger.info(f"Home team stats type: {type(home_stats)}")
        logger.info(f"Away team stats type: {type(away_stats)}")
        
        if isinstance(home_stats, dict):
            logger.info(f"Home team stats keys: {list(home_stats.keys())}")
        else:
            logger.info(f"Home team stats value: {home_stats}")
            
        if isinstance(away_stats, dict):
            logger.info(f"Away team stats keys: {list(away_stats.keys())}")
        else:
            logger.info(f"Away team stats value: {away_stats}")
        
        # Try to add offensive features
        logger.info("Adding team offensive metrics...")
        df_with_offensive = add_team_offensive_metrics(df_sample)
        
        logger.info(f"Columns after offensive metrics: {df_with_offensive.columns.tolist()}")
        
        # Check for offensive features
        offensive_features = [col for col in df_with_offensive.columns if 'team_' in col and ('batting_avg' in col or 'obp' in col or 'slg' in col or 'ops' in col)]
        logger.info(f"Offensive features found: {offensive_features}")
        
        # Try to add strength indicators
        logger.info("Adding team strength indicators...")
        df_with_strength = add_team_strength_indicators(df_with_offensive)
        
        logger.info(f"Columns after strength indicators: {df_with_strength.columns.tolist()}")
        
        # Check for strength features
        strength_features = [col for col in df_with_strength.columns if 'runs_scored' in col or 'offensive_efficiency' in col or 'offensive_composite' in col]
        logger.info(f"Strength features found: {strength_features}")
        
    else:
        logger.error("✗ Offensive stats columns not found!")
        logger.info(f"Available columns: {df_sample.columns.tolist()}")

if __name__ == "__main__":
    test_offensive_features() 