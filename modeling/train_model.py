# train_model.py

from pathlib import Path
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Root directory setup ===
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def train_model(historical_path, today_path, target_date=None):
    logger.info(f"Loading historical dataset from {historical_path}")
    historical_df = pd.read_csv(historical_path)

    # Ensure only numeric columns are used for training
    non_feature_cols = [
        "actual_winner", "game_date", "home_team", "away_team",
        "home_pitcher", "away_pitcher", "home_pitcher_full_name", "away_pitcher_full_name"
    ]
    numeric_cols = [col for col in historical_df.columns if col not in non_feature_cols and pd.api.types.is_numeric_dtype(historical_df[col])]

    X_train = historical_df[numeric_cols]
    y_train = (historical_df["actual_winner"] == historical_df["home_team"]).astype(int)

    logger.info(f"Training on {len(X_train)} historical games with {X_train.shape[1]} features")

    # Train base model
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)

    # Apply isotonic calibration to improve probability estimates
    logger.info("Applying isotonic calibration to improve probability estimates")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        cv=5,
        method='isotonic'
    )
    calibrated_model.fit(X_train, y_train)

    # Evaluate both models
    y_pred_base = base_model.predict(X_train)
    y_prob_base = base_model.predict_proba(X_train)[:, 1]
    
    y_pred_calibrated = calibrated_model.predict(X_train)
    y_prob_calibrated = calibrated_model.predict_proba(X_train)[:, 1]

    # Compare performance
    acc_base = accuracy_score(y_train, y_pred_base)
    acc_calibrated = accuracy_score(y_train, y_pred_calibrated)
    
    mae_base = mean_absolute_error(y_train, y_prob_base)
    mae_calibrated = mean_absolute_error(y_train, y_prob_calibrated)
    
    mse_base = mean_squared_error(y_train, y_prob_base)
    mse_calibrated = mean_squared_error(y_train, y_prob_calibrated)

    logger.info(f"Base Model - Accuracy: {acc_base:.3f}, MAE: {mae_base:.3f}, MSE: {mse_base:.3f}")
    logger.info(f"Calibrated Model - Accuracy: {acc_calibrated:.3f}, MAE: {mae_calibrated:.3f}, MSE: {mse_calibrated:.3f}")
    
    # Show probability range comparison
    logger.info(f"Base Model Probability Range: {y_prob_base.min():.3f} - {y_prob_base.max():.3f}")
    logger.info(f"Calibrated Model Probability Range: {y_prob_calibrated.min():.3f} - {y_prob_calibrated.max():.3f}")

    # Load today's features
    logger.info(f"Loading today's features from {today_path}")
    today_df = pd.read_csv(today_path)
    X_today = today_df[numeric_cols]

    # Make predictions for today using calibrated model
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

    # Save readable predictions
    if target_date is not None:
        output_name = f"readable_win_predictions_for_{target_date.strftime('%Y-%m-%d')}_using_{datetime.today().strftime('%Y-%m-%d')}.csv"
    else:
        output_name = f"readable_win_predictions_for_{today_df['game_date'].iloc[0]}_using_{datetime.today().strftime('%Y-%m-%d')}.csv"
    output_path = PROCESSED_DIR / output_name
    # output_path = os.path.join("C:/Users/roman/baseball_forecast_project/data/processed", output_name)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

    return result_df


if __name__ == "__main__":
    
    today_str = datetime.today().strftime('%Y-%m-%d')
    historical_path = PROCESSED_DIR / f"historical_main_features.csv"
    today_path = PROCESSED_DIR / f"main_features_{today_str}.csv"

    try:
        train_model(historical_path, today_path)
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
    

    # cd C:\Users\roman\baseball_forecast_project\modeling
    # python train_model.py
