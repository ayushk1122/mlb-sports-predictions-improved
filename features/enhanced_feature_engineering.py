# enhanced_feature_engineering.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def add_pitcher_effectiveness_features(df):
    """
    Add pitcher effectiveness and workload features.
    """
    logger.info("Adding pitcher effectiveness features...")
    
    # Pitcher effectiveness ratios
    df['home_pitcher_effectiveness'] = df['home_pitcher_strikeouts'] / (df['home_pitcher_total_pitches'] + 1)
    df['away_pitcher_effectiveness'] = df['away_pitcher_strikeouts'] / (df['away_pitcher_total_pitches'] + 1)
    df['pitcher_effectiveness_diff'] = df['home_pitcher_effectiveness'] - df['away_pitcher_effectiveness']
    
    # Pitcher workload (pitches per game)
    df['home_pitcher_workload'] = df['home_pitcher_total_pitches'] / (df['home_pitcher_games_played'] + 1)
    df['away_pitcher_workload'] = df['away_pitcher_total_pitches'] / (df['away_pitcher_games_played'] + 1)
    df['pitcher_workload_diff'] = df['home_pitcher_workload'] - df['away_pitcher_workload']
    
    # Pitcher quality composite
    df['home_pitcher_quality'] = (
        df['home_pitcher_avg_velocity'] * 0.3 +
        df['home_pitcher_avg_spin_rate'] * 0.2 +
        df['home_pitcher_effectiveness'] * 0.3 +
        (1 / (df['home_pitcher_workload'] + 1)) * 0.2  # Lower workload is better
    )
    
    df['away_pitcher_quality'] = (
        df['away_pitcher_avg_velocity'] * 0.3 +
        df['away_pitcher_avg_spin_rate'] * 0.2 +
        df['away_pitcher_effectiveness'] * 0.3 +
        (1 / (df['away_pitcher_workload'] + 1)) * 0.2
    )
    
    df['pitcher_quality_diff'] = df['home_pitcher_quality'] - df['away_pitcher_quality']
    
    return df

def add_pitcher_differential_features(df):
    """
    Add differential features between home and away pitchers.
    """
    logger.info("Adding pitcher differential features...")
    
    # Velocity differential
    df['pitcher_velocity_diff'] = df['home_pitcher_avg_velocity'] - df['away_pitcher_avg_velocity']
    
    # Spin rate differential
    df['pitcher_spin_rate_diff'] = df['home_pitcher_avg_spin_rate'] - df['away_pitcher_avg_spin_rate']
    
    # Extension differential
    df['pitcher_extension_diff'] = df['home_pitcher_avg_extension'] - df['away_pitcher_avg_extension']
    
    # Strikeout differential
    df['pitcher_strikeout_diff'] = df['home_pitcher_strikeouts'] - df['away_pitcher_strikeouts']
    
    # Bat speed differential (what pitchers allow)
    df['pitcher_bat_speed_diff'] = df['home_pitcher_avg_bat_speed'] - df['away_pitcher_avg_bat_speed']
    
    # Launch angle differential (what pitchers allow)
    df['pitcher_launch_angle_diff'] = df['home_pitcher_avg_launch_angle'] - df['away_pitcher_avg_launch_angle']
    
    # Exit velocity differential (what pitchers allow)
    df['pitcher_exit_velocity_diff'] = df['home_pitcher_avg_exit_velocity'] - df['away_pitcher_avg_exit_velocity']
    
    return df

def add_pitcher_interaction_features(df):
    """
    Add interaction features between pitcher characteristics.
    """
    logger.info("Adding pitcher interaction features...")
    
    # Velocity × Spin Rate interaction
    df['home_velocity_spin_interaction'] = df['home_pitcher_avg_velocity'] * df['home_pitcher_avg_spin_rate']
    df['away_velocity_spin_interaction'] = df['away_pitcher_avg_velocity'] * df['away_pitcher_avg_spin_rate']
    df['velocity_spin_interaction_diff'] = df['home_velocity_spin_interaction'] - df['away_velocity_spin_interaction']
    
    # Effectiveness × Workload interaction
    df['home_effectiveness_workload'] = df['home_pitcher_effectiveness'] * (1 / (df['home_pitcher_workload'] + 1))
    df['away_effectiveness_workload'] = df['away_pitcher_effectiveness'] * (1 / (df['away_pitcher_workload'] + 1))
    df['effectiveness_workload_diff'] = df['home_effectiveness_workload'] - df['away_effectiveness_workload']
    
    # Strikeout × Velocity interaction
    df['home_strikeout_velocity'] = df['home_pitcher_strikeouts'] * df['home_pitcher_avg_velocity']
    df['away_strikeout_velocity'] = df['away_pitcher_strikeouts'] * df['away_pitcher_avg_velocity']
    df['strikeout_velocity_diff'] = df['home_strikeout_velocity'] - df['away_strikeout_velocity']
    
    return df

def add_clear_pitcher_mismatch_indicators(df):
    """
    Add clear pitcher mismatch indicators for better discrimination.
    """
    logger.info("Adding clear pitcher mismatch indicators...")
    
    # Clear home pitcher advantage
    df['clear_home_pitcher_advantage'] = (
        (df['pitcher_quality_diff'] > 0.1) &  # Better pitcher
        (df['pitcher_effectiveness_diff'] > 0.02) &  # More effective
        (df['pitcher_velocity_diff'] > 2)  # Faster velocity
    ).astype(int)
    
    # Clear away pitcher advantage
    df['clear_away_pitcher_advantage'] = (
        (df['pitcher_quality_diff'] < -0.1) &  # Worse pitcher
        (df['pitcher_effectiveness_diff'] < -0.02) &  # Less effective
        (df['pitcher_velocity_diff'] < -2)  # Slower velocity
    ).astype(int)
    
    # Even pitcher matchup
    df['even_pitcher_matchup'] = (
        (abs(df['pitcher_quality_diff']) < 0.05) &  # Close pitcher quality
        (abs(df['pitcher_effectiveness_diff']) < 0.01) &  # Close effectiveness
        (abs(df['pitcher_velocity_diff']) < 1)  # Close velocity
    ).astype(int)
    
    return df

def add_pitcher_uncertainty_features(df):
    """
    Add features that capture pitcher prediction uncertainty.
    """
    logger.info("Adding pitcher uncertainty features...")
    
    # Close pitcher matchup uncertainty
    df['close_pitcher_uncertainty'] = (
        (abs(df['pitcher_quality_diff']) < 0.05) &
        (abs(df['pitcher_effectiveness_diff']) < 0.01)
    ).astype(int)
    
    # High workload uncertainty (tired pitchers)
    df['high_workload_uncertainty'] = (
        (df['home_pitcher_workload'] > 100) | (df['away_pitcher_workload'] > 100)
    ).astype(int)
    
    # Low sample size uncertainty (few games)
    df['low_sample_uncertainty'] = (
        (df['home_pitcher_games_played'] < 3) | (df['away_pitcher_games_played'] < 3)
    ).astype(int)
    
    return df

def add_pitcher_trend_features(df):
    """
    Add features that capture pitcher trends and patterns.
    """
    logger.info("Adding pitcher trend features...")
    
    # High velocity + high spin rate (premium stuff)
    df['home_premium_stuff'] = (
        (df['home_pitcher_avg_velocity'] > 92) & 
        (df['home_pitcher_avg_spin_rate'] > 2200)
    ).astype(int)
    
    df['away_premium_stuff'] = (
        (df['away_pitcher_avg_velocity'] > 92) & 
        (df['away_pitcher_avg_spin_rate'] > 2200)
    ).astype(int)
    
    df['premium_stuff_diff'] = df['home_premium_stuff'] - df['away_premium_stuff']
    
    # High strikeout rate
    df['home_high_strikeout_rate'] = (df['home_pitcher_effectiveness'] > 0.05).astype(int)
    df['away_high_strikeout_rate'] = (df['away_pitcher_effectiveness'] > 0.05).astype(int)
    df['high_strikeout_rate_diff'] = df['home_high_strikeout_rate'] - df['away_high_strikeout_rate']
    
    # Ground ball pitcher (low launch angle allowed)
    df['home_ground_ball_pitcher'] = (df['home_pitcher_avg_launch_angle'] < 15).astype(int)
    df['away_ground_ball_pitcher'] = (df['away_pitcher_avg_launch_angle'] < 15).astype(int)
    df['ground_ball_pitcher_diff'] = df['home_ground_ball_pitcher'] - df['away_ground_ball_pitcher']
    
    return df

def add_team_form_features(df):
    """
    Add team form features if available in the dataset.
    """
    logger.info("Adding team form features...")
    
    # Check if team form columns exist
    team_form_columns = [
        'home_wins', 'home_losses', 'home_run_diff', 'home_streak', 'home_games_played', 'home_win_pct',
        'away_wins', 'away_losses', 'away_run_diff', 'away_streak', 'away_games_played', 'away_win_pct'
    ]
    
    available_columns = [col for col in team_form_columns if col in df.columns]
    
    if not available_columns:
        logger.warning("No team form columns found in dataset. Skipping team form features.")
        return df
    
    logger.info(f"Found team form columns: {available_columns}")
    
    # Team strength differentials
    if 'home_win_pct' in df.columns and 'away_win_pct' in df.columns:
        df['team_win_pct_diff'] = df['home_win_pct'] - df['away_win_pct']
        df['team_win_pct_advantage'] = np.where(df['team_win_pct_diff'] > 0.1, 1, 
                                               np.where(df['team_win_pct_diff'] < -0.1, -1, 0))
    
    if 'home_run_diff' in df.columns and 'away_run_diff' in df.columns:
        df['team_run_diff_diff'] = df['home_run_diff'] - df['away_run_diff']
        df['team_run_diff_advantage'] = np.where(df['team_run_diff_diff'] > 50, 1,
                                                np.where(df['team_run_diff_diff'] < -50, -1, 0))
    
    # Streak analysis - handle string format like "L2", "W1"
    if 'home_streak' in df.columns and 'away_streak' in df.columns:
        # Convert streak strings to numeric values
        def parse_streak(streak_str):
            if pd.isna(streak_str) or streak_str == '':
                return 0
            try:
                if isinstance(streak_str, str):
                    if streak_str.startswith('W'):
                        return int(streak_str[1:])
                    elif streak_str.startswith('L'):
                        return -int(streak_str[1:])
                    else:
                        return 0
                else:
                    return float(streak_str)
            except:
                return 0
        
        df['home_streak_numeric'] = df['home_streak'].apply(parse_streak)
        df['away_streak_numeric'] = df['away_streak'].apply(parse_streak)
        
        df['home_streak_positive'] = np.where(df['home_streak_numeric'] > 0, 1, 0)
        df['away_streak_positive'] = np.where(df['away_streak_numeric'] > 0, 1, 0)
        df['streak_momentum_diff'] = df['home_streak_numeric'] - df['away_streak_numeric']
    
    # Team form quality indicators
    if 'home_win_pct' in df.columns and 'away_win_pct' in df.columns:
        df['home_strong_team'] = np.where(df['home_win_pct'] > 0.55, 1, 0)
        df['away_strong_team'] = np.where(df['away_win_pct'] > 0.55, 1, 0)
        df['strong_team_matchup'] = df['home_strong_team'] + df['away_strong_team']
    
    return df

def add_team_batter_features(df):
    """
    Add team batter features if available in the dataset.
    """
    logger.info("Adding team batter features...")
    
    # Check if team batter columns exist
    team_batter_columns = [
        'home_team_avg_launch_speed', 'home_team_avg_bat_speed', 'home_team_avg_swing_length',
        'away_team_avg_launch_speed', 'away_team_avg_bat_speed', 'away_team_avg_swing_length'
    ]
    
    available_columns = [col for col in team_batter_columns if col in df.columns]
    
    if not available_columns:
        logger.warning("No team batter columns found in dataset. Skipping team batter features.")
        return df
    
    logger.info(f"Found team batter columns: {available_columns}")
    
    # Team batting differentials
    if 'home_team_avg_launch_speed' in df.columns and 'away_team_avg_launch_speed' in df.columns:
        df['team_launch_speed_diff'] = df['home_team_avg_launch_speed'] - df['away_team_avg_launch_speed']
        df['team_launch_speed_advantage'] = np.where(df['team_launch_speed_diff'] > 2, 1,
                                                    np.where(df['team_launch_speed_diff'] < -2, -1, 0))
    
    if 'home_team_avg_bat_speed' in df.columns and 'away_team_avg_bat_speed' in df.columns:
        df['team_bat_speed_diff'] = df['home_team_avg_bat_speed'] - df['away_team_avg_bat_speed']
        df['team_bat_speed_advantage'] = np.where(df['team_bat_speed_diff'] > 1, 1,
                                                 np.where(df['team_bat_speed_diff'] < -1, -1, 0))
    
    if 'home_team_avg_swing_length' in df.columns and 'away_team_avg_swing_length' in df.columns:
        df['team_swing_length_diff'] = df['home_team_avg_swing_length'] - df['away_team_avg_swing_length']
    
    # Team batting quality indicators
    if 'home_team_avg_launch_speed' in df.columns and 'away_team_avg_launch_speed' in df.columns:
        df['home_power_hitting'] = np.where(df['home_team_avg_launch_speed'] > 85, 1, 0)
        df['away_power_hitting'] = np.where(df['away_team_avg_launch_speed'] > 85, 1, 0)
        df['power_hitting_matchup'] = df['home_power_hitting'] + df['away_power_hitting']
    
    if 'home_team_avg_bat_speed' in df.columns and 'away_team_avg_bat_speed' in df.columns:
        df['home_contact_hitting'] = np.where(df['home_team_avg_bat_speed'] > 70, 1, 0)
        df['away_contact_hitting'] = np.where(df['away_team_avg_bat_speed'] > 70, 1, 0)
        df['contact_hitting_matchup'] = df['home_contact_hitting'] + df['away_contact_hitting']
    
    return df

def add_team_pitcher_interaction_features(df):
    """
    Add interaction features between team form/batting and pitcher characteristics.
    """
    logger.info("Adding team-pitcher interaction features...")
    
    # Team form × Pitcher quality interactions
    if 'team_win_pct_diff' in df.columns and 'pitcher_quality_diff' in df.columns:
        df['team_pitcher_alignment'] = df['team_win_pct_diff'] * df['pitcher_quality_diff']
        df['team_pitcher_misalignment'] = np.where(
            (df['team_win_pct_diff'] > 0) & (df['pitcher_quality_diff'] < 0), 1,
            np.where((df['team_win_pct_diff'] < 0) & (df['pitcher_quality_diff'] > 0), -1, 0)
        )
    
    # Team batting × Pitcher effectiveness interactions
    if 'team_launch_speed_diff' in df.columns and 'pitcher_exit_velocity_diff' in df.columns:
        df['batting_pitcher_matchup'] = df['team_launch_speed_diff'] * df['pitcher_exit_velocity_diff']
    
    # Strong team vs weak pitcher indicators
    if 'home_strong_team' in df.columns and 'away_pitcher_quality' in df.columns:
        df['strong_home_weak_away_pitcher'] = np.where(
            (df['home_strong_team'] == 1) & (df['away_pitcher_quality'] < df['away_pitcher_quality'].median()), 1, 0
        )
    
    if 'away_strong_team' in df.columns and 'home_pitcher_quality' in df.columns:
        df['strong_away_weak_home_pitcher'] = np.where(
            (df['away_strong_team'] == 1) & (df['home_pitcher_quality'] < df['home_pitcher_quality'].median()), 1, 0
        )
    
    return df

def add_comprehensive_team_features(df):
    """
    Add comprehensive team-level indicators and summary features.
    """
    logger.info("Adding comprehensive team features...")
    
    # Overall team strength indicators
    df['home_team_advantages'] = (
        (df['home_win_pct'] > df['away_win_pct']).astype(int) +
        (df['home_run_diff'] > df['away_run_diff']).astype(int) +
        (df['home_streak_numeric'] > df['away_streak_numeric']).astype(int)
    )
    
    df['away_team_advantages'] = (
        (df['away_win_pct'] > df['home_win_pct']).astype(int) +
        (df['away_run_diff'] > df['home_run_diff']).astype(int) +
        (df['away_streak_numeric'] > df['home_streak_numeric']).astype(int)
    )
    
    # Team advantage balance
    df['team_advantage_balance'] = df['home_team_advantages'] - df['away_team_advantages']
    
    # Clear advantage indicators
    df['clear_home_advantage'] = (df['team_advantage_balance'] >= 2).astype(int)
    df['clear_away_advantage'] = (df['team_advantage_balance'] <= -2).astype(int)
    df['even_matchup'] = (df['team_advantage_balance'] == 0).astype(int)
    
    # Confidence indicators
    df['high_confidence_matchup'] = (
        (df['clear_home_advantage'] == 1) | 
        (df['clear_away_advantage'] == 1)
    ).astype(int)
    
    df['low_confidence_matchup'] = (df['even_matchup'] == 1).astype(int)
    
    return df

def add_team_offensive_metrics(df):
    """
    Add comprehensive team offensive metrics based on research recommendations.
    Uses actual team offensive stats data from our scraped files.
    """
    logger.info("Adding team offensive metrics...")
    
    # Check if we have team offensive stats data
    if 'home_team_offensive_stats' not in df.columns or 'away_team_offensive_stats' not in df.columns:
        logger.warning("Team offensive stats not found in dataframe. Skipping offensive metrics.")
        return df
    
    # Helper function to safely extract stats
    def safe_get_stats(stats_obj, key, default=0):
        """Safely extract stats from either dict or NaN values"""
        if pd.isna(stats_obj) or not isinstance(stats_obj, dict):
            return default
        return stats_obj.get(key, default)
    
    # Team batting average differential
    df['team_batting_avg_diff'] = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'batting_avg')) - \
                                 df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'batting_avg'))
    
    # Team on-base percentage differential
    df['team_obp_diff'] = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'on_base_pct')) - \
                         df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'on_base_pct'))
    
    # Team slugging percentage differential
    df['team_slg_diff'] = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'slugging_pct')) - \
                         df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'slugging_pct'))
    
    # Team OPS (on-base + slugging) differential
    df['team_ops_diff'] = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'ops')) - \
                         df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'ops'))
    
    # Team isolated power (ISO) differential
    df['team_iso_diff'] = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'iso')) - \
                         df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'iso'))
    
    # Team home runs per game differential
    df['team_hr_per_game_diff'] = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'hr_per_game')) - \
                                 df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'hr_per_game'))
    
    # Team strikeout rate differential (lower is better for batting)
    df['team_k_rate_diff'] = df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'k_rate')) - \
                            df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'k_rate'))  # Inverted so positive = home advantage
    
    # Team walk rate differential
    df['team_bb_rate_diff'] = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'bb_rate')) - \
                             df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'bb_rate'))
    
    # Team stolen base success rate differential
    df['team_sb_success_rate_diff'] = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'sb_success_rate')) - \
                                     df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'sb_success_rate'))
    
    # Team runs per game differential
    df['team_runs_per_game_diff'] = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'runs_per_game')) - \
                                   df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'runs_per_game'))
    
    # Additional offensive metrics
    # Team hits per game differential
    df['team_hits_per_game_diff'] = (df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'hits')) / 
                                    df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1))) - \
                                   (df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'hits')) / 
                                    df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1)))
    
    # Team doubles per game differential
    df['team_doubles_per_game_diff'] = (df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'doubles')) / 
                                       df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1))) - \
                                      (df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'doubles')) / 
                                       df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1)))
    
    # Team triples per game differential
    df['team_triples_per_game_diff'] = (df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'triples')) / 
                                       df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1))) - \
                                      (df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'triples')) / 
                                       df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1)))
    
    # Team RBI per game differential
    df['team_rbi_per_game_diff'] = (df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'rbi')) / 
                                   df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1))) - \
                                  (df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'rbi')) / 
                                   df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1)))
    
    # Team total bases per game differential
    df['team_total_bases_per_game_diff'] = (df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'total_bases')) / 
                                           df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1))) - \
                                          (df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'total_bases')) / 
                                           df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1)))
    
    return df

def add_team_strength_indicators(df):
    """
    Add team strength indicators based on research recommendations.
    Uses actual team offensive stats data from our scraped files.
    """
    logger.info("Adding team strength indicators...")
    
    # Check if we have team offensive stats data
    if 'home_team_offensive_stats' not in df.columns or 'away_team_offensive_stats' not in df.columns:
        logger.warning("Team offensive stats not found in dataframe. Skipping strength indicators.")
        return df
    
    # Helper function to safely extract stats
    def safe_get_stats(stats_obj, key, default=0):
        """Safely extract stats from either dict or NaN values"""
        if pd.isna(stats_obj) or not isinstance(stats_obj, dict):
            return default
        return stats_obj.get(key, default)
    
    # Run differential (most important according to research)
    # Calculate runs scored and allowed from offensive stats
    home_runs_scored = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'runs'))
    away_runs_scored = df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'runs'))
    
    # For now, we'll use runs scored as a proxy since we don't have runs allowed in offensive stats
    # In a full implementation, we'd need defensive stats as well
    df['home_runs_scored'] = home_runs_scored
    df['away_runs_scored'] = away_runs_scored
    df['runs_scored_diff'] = home_runs_scored - away_runs_scored
    
    # Team offensive efficiency (runs per game)
    home_games = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1))
    away_games = df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'games', 1))
    
    df['home_offensive_efficiency'] = home_runs_scored / home_games
    df['away_offensive_efficiency'] = away_runs_scored / away_games
    df['offensive_efficiency_diff'] = df['home_offensive_efficiency'] - df['away_offensive_efficiency']
    
    # Team offensive composite score (combining multiple offensive indicators)
    home_ops = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'ops'))
    away_ops = df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'ops'))
    home_iso = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'iso'))
    away_iso = df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'iso'))
    home_bb_rate = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'bb_rate'))
    away_bb_rate = df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'bb_rate'))
    
    df['home_offensive_composite'] = (
        home_ops * 0.4 +
        home_iso * 0.3 +
        home_bb_rate * 0.2 +
        df['home_offensive_efficiency'] * 0.1
    )
    
    df['away_offensive_composite'] = (
        away_ops * 0.4 +
        away_iso * 0.3 +
        away_bb_rate * 0.2 +
        df['away_offensive_efficiency'] * 0.1
    )
    
    df['offensive_composite_diff'] = df['home_offensive_composite'] - df['away_offensive_composite']
    
    # Team offensive quality tiers
    df['home_offensive_quality'] = pd.cut(df['home_offensive_composite'], 
                                        bins=[-np.inf, 0.6, 0.7, 0.8, np.inf], 
                                        labels=[0, 1, 2, 3], 
                                        include_lowest=True).astype(int)
    
    df['away_offensive_quality'] = pd.cut(df['away_offensive_composite'], 
                                        bins=[-np.inf, 0.6, 0.7, 0.8, np.inf], 
                                        labels=[0, 1, 2, 3], 
                                        include_lowest=True).astype(int)
    
    df['offensive_quality_diff'] = df['home_offensive_quality'] - df['away_offensive_quality']
    
    # Power vs contact hitting indicators
    home_hr_rate = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'hr_per_game'))
    away_hr_rate = df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'hr_per_game'))
    home_avg = df['home_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'batting_avg'))
    away_avg = df['away_team_offensive_stats'].apply(lambda x: safe_get_stats(x, 'batting_avg'))
    
    df['home_power_contact_ratio'] = home_hr_rate / (home_avg + 0.001)  # Avoid division by zero
    df['away_power_contact_ratio'] = away_hr_rate / (away_avg + 0.001)
    df['power_contact_ratio_diff'] = df['home_power_contact_ratio'] - df['away_power_contact_ratio']
    
    return df

def add_recent_performance_features(df):
    """
    Add recent performance features based on research recommendations.
    """
    logger.info("Adding recent performance features...")
    
    # Check if we have the required recent performance columns
    required_columns = [
        'home_team_recent_wins', 'home_team_recent_losses', 'away_team_recent_wins', 'away_team_recent_losses',
        'home_team_recent_runs_scored', 'home_team_recent_runs_allowed', 'away_team_recent_runs_scored', 'away_team_recent_runs_allowed',
        'home_team_recent_games', 'away_team_recent_games', 'home_streak_numeric', 'away_streak_numeric',
        'home_team_recent_variance', 'away_team_recent_variance'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing recent performance columns: {missing_columns}. Skipping recent performance features.")
        return df
    
    # Recent win percentage (last 10 games)
    df['home_recent_win_pct'] = df['home_team_recent_wins'] / (df['home_team_recent_wins'] + df['home_team_recent_losses'] + 1)
    df['away_recent_win_pct'] = df['away_team_recent_wins'] / (df['away_team_recent_wins'] + df['away_team_recent_losses'] + 1)
    df['recent_win_pct_diff'] = df['home_recent_win_pct'] - df['away_recent_win_pct']
    
    # Recent run differential (last 2 weeks)
    df['home_recent_run_diff'] = df['home_team_recent_runs_scored'] - df['home_team_recent_runs_allowed']
    df['away_recent_run_diff'] = df['away_team_recent_runs_scored'] - df['away_team_recent_runs_allowed']
    df['recent_run_diff_diff'] = df['home_recent_run_diff'] - df['away_recent_run_diff']
    
    # Recent offensive performance
    df['home_recent_runs_per_game'] = df['home_team_recent_runs_scored'] / (df['home_team_recent_games'] + 1)
    df['away_recent_runs_per_game'] = df['away_team_recent_runs_scored'] / (df['away_team_recent_games'] + 1)
    df['recent_runs_per_game_diff'] = df['home_recent_runs_per_game'] - df['away_recent_runs_per_game']
    
    # Recent defensive performance
    df['home_recent_runs_allowed_per_game'] = df['home_team_recent_runs_allowed'] / (df['home_team_recent_games'] + 1)
    df['away_recent_runs_allowed_per_game'] = df['away_team_recent_runs_allowed'] / (df['away_team_recent_games'] + 1)
    df['recent_runs_allowed_per_game_diff'] = df['away_recent_runs_allowed_per_game'] - df['home_recent_runs_allowed_per_game']  # Inverted
    
    # Momentum indicators
    df['home_momentum'] = df['home_streak_numeric'] * df['home_recent_win_pct']
    df['away_momentum'] = df['away_streak_numeric'] * df['away_recent_win_pct']
    df['momentum_diff'] = df['home_momentum'] - df['away_momentum']
    
    # Form consistency (variance in recent performance)
    df['home_form_consistency'] = 1 / (df['home_team_recent_variance'] + 1)  # Lower variance = higher consistency
    df['away_form_consistency'] = 1 / (df['away_team_recent_variance'] + 1)
    df['form_consistency_diff'] = df['home_form_consistency'] - df['away_form_consistency']
    
    return df

def add_pitcher_rest_features(df):
    """
    Add pitcher rest and workload features based on research recommendations.
    """
    logger.info("Adding pitcher rest features...")
    
    # Check if we have the required pitcher rest columns
    required_columns = [
        'home_pitcher_days_since_last_start', 'away_pitcher_days_since_last_start',
        'home_pitcher_quality_starts', 'away_pitcher_quality_starts',
        'home_pitcher_games_played', 'away_pitcher_games_played',
        'home_team_bullpen_era', 'away_team_bullpen_era',
        'home_bullpen_innings_last_3_days', 'away_bullpen_innings_last_3_days'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing pitcher rest columns: {missing_columns}. Skipping pitcher rest features.")
        return df
    
    # Pitcher rest days (critical factor according to research)
    df['home_pitcher_rest_days'] = df['home_pitcher_days_since_last_start']
    df['away_pitcher_rest_days'] = df['away_pitcher_days_since_last_start']
    df['pitcher_rest_days_diff'] = df['home_pitcher_rest_days'] - df['away_pitcher_rest_days']
    
    # Rest advantage indicators
    df['home_pitcher_well_rested'] = (df['home_pitcher_rest_days'] >= 4).astype(int)
    df['away_pitcher_well_rested'] = (df['away_pitcher_rest_days'] >= 4).astype(int)
    df['pitcher_rest_advantage'] = df['home_pitcher_well_rested'] - df['away_pitcher_well_rested']
    
    # Quality start percentage
    df['home_pitcher_quality_start_pct'] = df['home_pitcher_quality_starts'] / (df['home_pitcher_games_played'] + 1)
    df['away_pitcher_quality_start_pct'] = df['away_pitcher_quality_starts'] / (df['away_pitcher_games_played'] + 1)
    df['pitcher_quality_start_pct_diff'] = df['home_pitcher_quality_start_pct'] - df['away_pitcher_quality_start_pct']
    
    # Bullpen strength indicators
    df['home_bullpen_era'] = df['home_team_bullpen_era']
    df['away_bullpen_era'] = df['away_team_bullpen_era']
    df['bullpen_era_diff'] = df['away_bullpen_era'] - df['home_bullpen_era']  # Inverted so positive = home advantage
    
    # Bullpen availability
    df['home_bullpen_rested'] = (df['home_bullpen_innings_last_3_days'] < 10).astype(int)
    df['away_bullpen_rested'] = (df['away_bullpen_innings_last_3_days'] < 10).astype(int)
    df['bullpen_rest_advantage'] = df['home_bullpen_rested'] - df['away_bullpen_rested']
    
    return df

def remove_zero_importance_features(df):
    """
    Remove features with low importance to reduce noise and improve model performance.
    """
    logger.info("Applying feature selection to remove low-importance features...")
    
    # Get the original number of features
    original_features = len(df.columns)
    
    # List of features to always keep (core features)
    core_features = [
        'game_date', 'away_team', 'home_team', 'away_pitcher', 'home_pitcher', 'winner', 'actual_winner'
    ]
    
    # List of features to remove (known to be noisy or redundant)
    features_to_remove = [
        # Remove duplicate/very similar features
        'home_streak_numeric',  # Keep only the derived features
        'away_streak_numeric',  # Keep only the derived features
        
        # Remove features that are too granular
        'home_pitcher_whiffs',  # Usually 0, not very predictive
        'away_pitcher_whiffs',  # Usually 0, not very predictive
        
        # Remove features with too much variance (noise)
        'home_pitcher_games_played',  # Too variable, use workload instead
        'away_pitcher_games_played',  # Too variable, use workload instead
        
        # Remove redundant interaction features if base features are kept
        'home_velocity_spin_interaction',  # Keep only the diff version
        'away_velocity_spin_interaction',  # Keep only the diff version
        'home_effectiveness_workload',     # Keep only the diff version
        'away_effectiveness_workload',     # Keep only the diff version
        'home_strikeout_velocity',         # Keep only the diff version
        'away_strikeout_velocity',         # Keep only the diff version
    ]
    
    # Remove specified features
    for feature in features_to_remove:
        if feature in df.columns:
            df = df.drop(columns=[feature])
            logger.info(f"Removed feature: {feature}")
    
    # Remove features with too many missing values (>50% missing)
    missing_threshold = 0.5
    missing_counts = df.isnull().sum() / len(df)
    high_missing_features = missing_counts[missing_counts > missing_threshold].index.tolist()
    
    for feature in high_missing_features:
        if feature not in core_features:
            df = df.drop(columns=[feature])
            logger.info(f"Removed high-missing feature: {feature} ({missing_counts[feature]:.1%} missing)")
    
    # Remove features with zero variance (constant values)
    zero_var_features = []
    for col in df.columns:
        if col not in core_features and df[col].dtype in ['int64', 'float64']:
            if df[col].var() == 0:
                zero_var_features.append(col)
    
    for feature in zero_var_features:
        df = df.drop(columns=[feature])
        logger.info(f"Removed zero-variance feature: {feature}")
    
    # Remove features with very low variance (near constant)
    low_var_threshold = 0.001
    low_var_features = []
    for col in df.columns:
        if col not in core_features and df[col].dtype in ['int64', 'float64']:
            if df[col].var() < low_var_threshold:
                low_var_features.append(col)
    
    for feature in low_var_features:
        df = df.drop(columns=[feature])
        logger.info(f"Removed low-variance feature: {feature} (variance: {df[feature].var():.6f})")
    
    # Remove highly correlated features (correlation > 0.95)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in core_features]
    
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        
        high_corr_features = []
        for col in upper_triangle.columns:
            high_corr_cols = upper_triangle[col][upper_triangle[col] > 0.95].index.tolist()
            for high_corr_col in high_corr_cols:
                # Keep the feature with more variance (more informative)
                if df[col].var() > df[high_corr_col].var():
                    high_corr_features.append(high_corr_col)
                else:
                    high_corr_features.append(col)
        
        # Remove duplicates and remove features
        high_corr_features = list(set(high_corr_features))
        for feature in high_corr_features:
            if feature in df.columns:
                df = df.drop(columns=[feature])
                logger.info(f"Removed highly correlated feature: {feature}")
    
    # Log feature selection results
    final_features = len(df.columns)
    removed_features = original_features - final_features
    
    logger.info(f"Feature selection completed:")
    logger.info(f"  - Original features: {original_features}")
    logger.info(f"  - Final features: {final_features}")
    logger.info(f"  - Features removed: {removed_features}")
    logger.info(f"  - Reduction: {removed_features/original_features:.1%}")
    
    return df

def apply_advanced_feature_selection(df):
    """
    Apply advanced feature selection techniques to keep only the most predictive features.
    """
    logger.info("Applying advanced feature selection...")
    
    # Get the original number of features
    original_features = len(df.columns)
    
    # List of features to always keep (core features)
    core_features = [
        'game_date', 'away_team', 'home_team', 'away_pitcher', 'home_pitcher', 'winner', 'actual_winner'
    ]
    
    # Define feature categories and their importance thresholds
    feature_categories = {
        'pitcher_quality': 0.015,      # High importance for pitcher features
        'team_form': 0.010,           # Medium importance for team form
        'team_batting': 0.010,        # Medium importance for team batting
        'differential': 0.012,        # Medium-high importance for differentials
        'interaction': 0.008,         # Lower importance for interactions
        'trend': 0.008,              # Lower importance for trends
        'uncertainty': 0.005,        # Lower importance for uncertainty
        'mismatch': 0.005,           # Lower importance for mismatch indicators
        'comprehensive': 0.010       # Medium importance for comprehensive features
    }
    
    # Define feature patterns and their categories
    feature_patterns = {
        'pitcher_quality': [
            'home_pitcher_', 'away_pitcher_', 'pitcher_quality', 'pitcher_effectiveness',
            'pitcher_workload', 'pitcher_velocity', 'pitcher_spin_rate', 'pitcher_extension',
            'pitcher_strikeouts', 'pitcher_bat_speed', 'pitcher_launch_angle', 'pitcher_exit_velocity'
        ],
        'team_form': [
            'home_win_pct', 'away_win_pct', 'home_run_diff', 'away_run_diff',
            'home_streak', 'away_streak', 'home_wins', 'away_wins', 'home_losses', 'away_losses'
        ],
        'team_batting': [
            'home_team_avg_', 'away_team_avg_', 'team_launch_speed', 'team_bat_speed',
            'team_swing_length', 'power_hitting', 'contact_hitting'
        ],
        'differential': [
            'pitcher_velocity_diff', 'pitcher_spin_rate_diff', 'pitcher_extension_diff',
            'pitcher_strikeout_diff', 'pitcher_bat_speed_diff', 'pitcher_launch_angle_diff',
            'pitcher_exit_velocity_diff', 'team_win_pct_diff', 'team_run_diff_diff',
            'team_launch_speed_diff', 'team_bat_speed_diff', 'team_swing_length_diff',
            'team_batting_avg_diff', 'team_obp_diff', 'team_slg_diff', 'team_ops_diff',
            'team_iso_diff', 'team_hr_per_game_diff', 'team_k_rate_diff', 'team_bb_rate_diff',
            'team_sb_success_rate_diff', 'team_runs_per_game_diff', 'team_hits_per_game_diff',
            'team_doubles_per_game_diff', 'team_triples_per_game_diff', 'team_rbi_per_game_diff',
            'team_total_bases_per_game_diff', 'runs_scored_diff', 'offensive_efficiency_diff',
            'offensive_composite_diff', 'offensive_quality_diff', 'power_contact_ratio_diff'
        ],
        'interaction': [
            'velocity_spin_interaction', 'effectiveness_workload', 'strikeout_velocity',
            'team_pitcher_alignment', 'batting_pitcher_matchup'
        ],
        'trend': [
            'premium_stuff', 'high_strikeout_rate', 'ground_ball_pitcher'
        ],
        'uncertainty': [
            'close_pitcher_uncertainty', 'high_workload_uncertainty', 'low_sample_uncertainty'
        ],
        'mismatch': [
            'clear_home_pitcher_advantage', 'clear_away_pitcher_advantage', 'even_pitcher_matchup',
            'clear_home_advantage', 'clear_away_advantage', 'even_matchup'
        ],
        'comprehensive': [
            'home_team_advantages', 'away_team_advantages', 'team_advantage_balance',
            'high_confidence_matchup', 'low_confidence_matchup', 'home_runs_scored', 'away_runs_scored',
            'home_offensive_efficiency', 'away_offensive_efficiency', 'home_offensive_composite',
            'away_offensive_composite', 'home_offensive_quality', 'away_offensive_quality',
            'home_power_contact_ratio', 'away_power_contact_ratio'
        ]
    }
    
    # Apply category-based feature selection
    features_to_keep = set(core_features)
    
    for category, threshold in feature_categories.items():
        if category in feature_patterns:
            category_features = []
            for pattern in feature_patterns[category]:
                category_features.extend([col for col in df.columns if pattern in col])
            
            # Keep all category features for now (we'll apply importance-based selection later)
            features_to_keep.update(category_features)
    
    # Keep only the selected features
    df_selected = df[list(features_to_keep)]
    
    logger.info(f"Advanced feature selection completed:")
    logger.info(f"  - Original features: {original_features}")
    logger.info(f"  - Selected features: {len(df_selected.columns)}")
    logger.info(f"  - Features removed: {original_features - len(df_selected.columns)}")
    
    return df_selected

def engineer_enhanced_features(df):
    """
    Apply comprehensive enhanced feature engineering to the dataset.
    """
    logger.info("Starting enhanced feature engineering...")
    original_features = len(df.columns)
    
    # Apply all feature engineering functions
    df = add_pitcher_effectiveness_features(df)
    df = add_pitcher_differential_features(df)
    df = add_pitcher_interaction_features(df)
    df = add_clear_pitcher_mismatch_indicators(df)
    df = add_pitcher_uncertainty_features(df)
    df = add_pitcher_trend_features(df)
    df = add_team_form_features(df)
    df = add_team_batter_features(df)
    df = add_team_pitcher_interaction_features(df)
    df = add_comprehensive_team_features(df)
    df = add_team_offensive_metrics(df)
    df = add_team_strength_indicators(df)
    df = add_recent_performance_features(df)
    df = add_pitcher_rest_features(df)
    
    # Apply advanced feature selection instead of basic removal
    df = apply_advanced_feature_selection(df)
    
    # Log the feature engineering summary
    final_features = len(df.columns)
    new_features = final_features - original_features
    
    logger.info(f"Enhanced feature engineering completed:")
    logger.info(f"  - Original features: {original_features}")
    logger.info(f"  - Final features: {final_features}")
    logger.info(f"  - New features added: {new_features}")
    
    # Get feature summary
    feature_summary = get_feature_summary(df)
    logger.info("Feature categories:")
    for category, features in feature_summary.items():
        logger.info(f"  - {category}: {len(features)} features")
    
    return df

def get_feature_summary(df):
    """
    Get a summary of all features for analysis.
    """
    feature_categories = {
        'Pitcher Features': [col for col in df.columns if 'pitcher' in col.lower()],
        'Differential Features': [col for col in df.columns if 'diff' in col.lower()],
        'Interaction Features': [col for col in df.columns if 'interaction' in col.lower()],
        'Mismatch Features': [col for col in df.columns if any(x in col.lower() for x in ['advantage', 'even', 'mismatch'])],
        'Uncertainty Features': [col for col in df.columns if 'uncertainty' in col.lower()],
        'Trend Features': [col for col in df.columns if any(x in col.lower() for x in ['premium', 'strikeout', 'ground_ball'])],
        'Other Features': [col for col in df.columns if not any(x in col.lower() for x in ['pitcher', 'diff', 'interaction', 'advantage', 'even', 'mismatch', 'uncertainty', 'premium', 'strikeout', 'ground_ball'])]
    }
    
    summary = {}
    for category, features in feature_categories.items():
        if features:
            summary[category] = {
                'count': len(features),
                'features': features
            }
    
    return summary

if __name__ == "__main__":
    # Test the feature engineering
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parents[1]
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    
    # Load sample data
    sample_file = PROCESSED_DIR / "historical_main_features.csv"
    if sample_file.exists():
        df = pd.read_csv(sample_file)
        logger.info(f"Loaded sample data: {df.shape}")
        
        # Apply feature engineering
        df_enhanced = engineer_enhanced_features(df)
        
        # Get feature summary
        summary = get_feature_summary(df_enhanced)
        
        logger.info("\nFeature Summary:")
        for category, info in summary.items():
            logger.info(f"  {category}: {info['count']} features")
        
        # Save enhanced dataset
        output_path = PROCESSED_DIR / "historical_main_features_enhanced.csv"
        df_enhanced.to_csv(output_path, index=False)
        logger.info(f"Enhanced dataset saved to: {output_path}")
    else:
        logger.error(f"Sample file not found: {sample_file}") 