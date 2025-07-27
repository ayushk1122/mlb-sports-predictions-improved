# train_model.py

from pathlib import Path
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Root directory setup ===
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def calculate_brier_score(predicted_probs, actual_outcomes):
    """Calculate Brier score for probabilistic predictions."""
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("Predicted probabilities and actual outcomes must have the same length")
    
    brier_score = np.mean((np.array(predicted_probs) - np.array(actual_outcomes)) ** 2)
    return brier_score

def analyze_brier_score_breakdown(predicted_probs, actual_outcomes, confidence_bins=10):
    """Detailed breakdown of Brier score performance across different probability ranges."""
    predicted_probs = np.array(predicted_probs)
    actual_outcomes = np.array(actual_outcomes)
    
    bins = np.linspace(0, 1, confidence_bins + 1)
    breakdown_data = []
    
    for i in range(len(bins) - 1):
        mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
        if mask.sum() > 0:
            bin_pred = predicted_probs[mask]
            bin_actual = actual_outcomes[mask]
            
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

def train_and_evaluate_model(historical_path, today_path=None, target_date=None, 
                           train_days=45, test_days=15, rolling_window=True):
    """
    Train model with proper train/test split and realistic evaluation.
    
    Args:
        historical_path: Path to historical features CSV
        today_path: Path to today's features (optional, for prediction mode)
        target_date: Target date for predictions
        train_days: Number of days to use for training
        test_days: Number of days to use for testing
        rolling_window: If True, use rolling window approach; if False, use fixed split
    """
    logger.info(f"Loading historical dataset from {historical_path}")
    historical_df = pd.read_csv(historical_path)
    
    # Ensure data is sorted by date
    historical_df['game_date'] = pd.to_datetime(historical_df['game_date'])
    historical_df = historical_df.sort_values('game_date').reset_index(drop=True)
    
    # Ensure only numeric columns are used for training
    non_feature_cols = [
        "actual_winner", "game_date", "home_team", "away_team",
        "home_pitcher", "away_pitcher", "home_pitcher_full_name", "away_pitcher_full_name"
    ]
    numeric_cols = [col for col in historical_df.columns 
                   if col not in non_feature_cols and pd.api.types.is_numeric_dtype(historical_df[col])]
    
    logger.info(f"Using {len(numeric_cols)} numeric features for training")
    
    if rolling_window:
        # Use rolling window approach (simulates real-world scenario)
        logger.info(f"Using rolling window approach: train on {train_days} days, test on {test_days} days")
        
        # Find the latest date in the dataset
        latest_date = historical_df['game_date'].max()
        
        # Define test period (most recent test_days)
        test_end_date = latest_date
        test_start_date = test_end_date - timedelta(days=test_days-1)
        
        # Define training period (train_days before test period)
        train_end_date = test_start_date - timedelta(days=1)
        train_start_date = train_end_date - timedelta(days=train_days-1)
        
        # Split data
        train_mask = (historical_df['game_date'] >= train_start_date) & (historical_df['game_date'] <= train_end_date)
        test_mask = (historical_df['game_date'] >= test_start_date) & (historical_df['game_date'] <= test_end_date)
        
        train_df = historical_df[train_mask].copy()
        test_df = historical_df[test_mask].copy()
        
        logger.info(f"Training period: {train_start_date.date()} to {train_end_date.date()} ({len(train_df)} games)")
        logger.info(f"Testing period: {test_start_date.date()} to {test_end_date.date()} ({len(test_df)} games)")
        
    else:
        # Use fixed split (for comparison)
        logger.info(f"Using fixed split: train on first {train_days} days, test on last {test_days} days")
        
        total_games = len(historical_df)
        train_size = int(total_games * (train_days / (train_days + test_days)))
        
        train_df = historical_df.iloc[:train_size].copy()
        test_df = historical_df.iloc[train_size:].copy()
        
        logger.info(f"Training set: {len(train_df)} games")
        logger.info(f"Testing set: {len(test_df)} games")
    
    if len(train_df) == 0:
        logger.error("No training data available")
        return None
    
    if len(test_df) == 0:
        logger.error("No testing data available")
        return None
    
    # Prepare training data
    X_train = train_df[numeric_cols]
    y_train = (train_df["actual_winner"] == train_df["home_team"]).astype(int)
    
    # Prepare testing data
    X_test = test_df[numeric_cols]
    y_test = (test_df["actual_winner"] == test_df["home_team"]).astype(int)
    
    logger.info(f"Training on {len(X_train)} games with {X_train.shape[1]} features")
    logger.info(f"Testing on {len(X_test)} games")
    
    # Train base model
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    # Apply isotonic calibration
    logger.info("Applying isotonic calibration to improve probability estimates")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        cv=5,
        method='isotonic'
    )
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate on TEST data (unseen data)
    logger.info("\n=== EVALUATION ON UNSEEN TEST DATA ===")
    
    # Base model evaluation
    y_pred_base = base_model.predict(X_test)
    y_prob_base = base_model.predict_proba(X_test)[:, 1]
    
    acc_base = accuracy_score(y_test, y_pred_base)
    mae_base = mean_absolute_error(y_test, y_prob_base)
    mse_base = mean_squared_error(y_test, y_prob_base)
    brier_base = calculate_brier_score(y_prob_base, y_test)
    
    # Calibrated model evaluation
    y_pred_calibrated = calibrated_model.predict(X_test)
    y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    
    acc_calibrated = accuracy_score(y_test, y_pred_calibrated)
    mae_calibrated = mean_absolute_error(y_test, y_prob_calibrated)
    mse_calibrated = mean_squared_error(y_test, y_prob_calibrated)
    brier_calibrated = calculate_brier_score(y_prob_calibrated, y_test)
    
    # Print results
    logger.info(f"Base Model Performance (Test Set):")
    logger.info(f"  Accuracy: {acc_base:.3f} ({acc_base*100:.1f}%)")
    logger.info(f"  MAE: {mae_base:.3f}")
    logger.info(f"  MSE: {mse_base:.3f}")
    logger.info(f"  Brier Score: {brier_base:.4f}")
    logger.info(f"  Probability Range: {y_prob_base.min():.3f} - {y_prob_base.max():.3f}")
    
    logger.info(f"\nCalibrated Model Performance (Test Set):")
    logger.info(f"  Accuracy: {acc_calibrated:.3f} ({acc_calibrated*100:.1f}%)")
    logger.info(f"  MAE: {mae_calibrated:.3f}")
    logger.info(f"  MSE: {mse_calibrated:.3f}")
    logger.info(f"  Brier Score: {brier_calibrated:.4f}")
    logger.info(f"  Probability Range: {y_prob_calibrated.min():.3f} - {y_prob_calibrated.max():.3f}")
    
    # Brier score breakdown for calibrated model
    logger.info(f"\nBrier Score Breakdown (Calibrated Model):")
    brier_breakdown = analyze_brier_score_breakdown(y_prob_calibrated, y_test)
    logger.info(brier_breakdown.to_string(index=False))
    
    # Calculate improvement
    brier_improvement = ((brier_base - brier_calibrated) / brier_base) * 100
    acc_improvement = ((acc_calibrated - acc_base) / acc_base) * 100 if acc_base > 0 else 0
    
    logger.info(f"\nImprovement from Calibration:")
    logger.info(f"  Brier Score: {brier_improvement:+.1f}% ({brier_base:.4f} → {brier_calibrated:.4f})")
    logger.info(f"  Accuracy: {acc_improvement:+.1f}% ({acc_base:.3f} → {acc_calibrated:.3f})")
    
    # Performance consistency analysis
    test_dates = test_df['game_date'].dt.date.unique()
    daily_performance = []
    
    for date in test_dates:
        date_mask = test_df['game_date'].dt.date == date
        if date_mask.sum() > 0:
            date_probs = y_prob_calibrated[date_mask]
            date_actuals = y_test[date_mask]
            date_acc = accuracy_score(date_actuals, date_probs > 0.5)
            date_brier = calculate_brier_score(date_probs, date_actuals)
            daily_performance.append({
                'date': date,
                'games': date_mask.sum(),
                'accuracy': date_acc,
                'brier_score': date_brier
            })
    
    if daily_performance:
        daily_df = pd.DataFrame(daily_performance)
        logger.info(f"\nDaily Performance Consistency:")
        logger.info(f"  Average Daily Accuracy: {daily_df['accuracy'].mean():.3f}")
        logger.info(f"  Average Daily Brier Score: {daily_df['brier_score'].mean():.4f}")
        logger.info(f"  Accuracy Std Dev: {daily_df['accuracy'].std():.3f}")
        logger.info(f"  Brier Score Std Dev: {daily_df['brier_score'].std():.4f}")
        logger.info(f"  Accuracy Range: {daily_df['accuracy'].min():.3f} - {daily_df['accuracy'].max():.3f}")
    
    # Make predictions for today if provided
    if today_path and os.path.exists(today_path):
        logger.info(f"\n=== MAKING PREDICTIONS FOR TODAY ===")
        logger.info(f"Loading today's features from {today_path}")
        today_df = pd.read_csv(today_path)
        X_today = today_df[numeric_cols]
        
        # Use calibrated model for predictions
        today_prob = calibrated_model.predict_proba(X_today)[:, 1]
        
        result_df = pd.DataFrame({
            "Game Date": today_df["game_date"],
            "Home Team": today_df["home_team"],
            "Away Team": today_df["away_team"],
            "Win Probability": today_prob,
            "Prediction": np.where(today_prob > 0.5,
                                   "Pick: " + today_df["home_team"],
                                   "Pick: " + today_df["away_team"])
        })
        
        # Save predictions
        if target_date is not None:
            output_name = f"readable_win_predictions_for_{target_date.strftime('%Y-%m-%d')}_using_{datetime.today().strftime('%Y-%m-%d')}.csv"
        else:
            output_name = f"readable_win_predictions_for_{today_df['game_date'].iloc[0]}_using_{datetime.today().strftime('%Y-%m-%d')}.csv"
        
        output_path = PROCESSED_DIR / output_name
        result_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        return result_df
    
    return {
        'test_accuracy': acc_calibrated,
        'test_brier_score': brier_calibrated,
        'test_mae': mae_calibrated,
        'test_mse': mse_calibrated,
        'daily_performance': daily_performance if daily_performance else None
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate MLB prediction model')
    parser.add_argument('--historical', type=str, default=None, 
                       help='Path to historical features CSV')
    parser.add_argument('--today', type=str, default=None,
                       help='Path to today\'s features CSV (optional)')
    parser.add_argument('--train-days', type=int, default=45,
                       help='Number of days to use for training')
    parser.add_argument('--test-days', type=int, default=15,
                       help='Number of days to use for testing')
    parser.add_argument('--no-rolling', action='store_true',
                       help='Use fixed split instead of rolling window')
    parser.add_argument('--date', type=str, default=None,
                       help='Target date for predictions (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Default paths
    if args.historical is None:
        args.historical = PROCESSED_DIR / "historical_main_features.csv"
    
    if args.today is None:
        today_str = datetime.today().strftime('%Y-%m-%d')
        args.today = PROCESSED_DIR / f"main_features_{today_str}.csv"
    
    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    
    try:
        result = train_and_evaluate_model(
            historical_path=args.historical,
            today_path=args.today,
            target_date=target_date,
            train_days=args.train_days,
            test_days=args.test_days,
            rolling_window=not args.no_rolling
        )
        
        if result is None:
            logger.error("Model training/evaluation failed")
        else:
            logger.info("Model training and evaluation completed successfully")
            
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        raise
