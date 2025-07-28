# train_model.py

from pathlib import Path
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

# Try to import advanced ensemble libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    from sklearn.ensemble import StackingClassifier
    STACKING_AVAILABLE = True
except ImportError:
    STACKING_AVAILABLE = False
    logging.warning("StackingClassifier not available in this scikit-learn version")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Root directory setup ===
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def create_advanced_voting_ensemble():
    """
    Create an advanced voting classifier ensemble with XGBoost and LightGBM.
    """
    logger.info("Creating advanced voting classifier ensemble...")
    
    estimators = []
    
    # RandomForest models with different configurations
    estimators.extend([
        ('rf1', RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )),
        ('rf2', RandomForestClassifier(
            n_estimators=150, 
            max_depth=8, 
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=123
        )),
        ('rf3', RandomForestClassifier(
            n_estimators=100, 
            max_depth=6, 
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=456
        ))
    ])
    
    # Logistic Regression
    estimators.append(('lr', LogisticRegression(
        random_state=42, 
        max_iter=2000,
        C=1.0,
        solver='liblinear',
        tol=1e-4
    )))
    
    # XGBoost if available
    if XGBOOST_AVAILABLE:
        estimators.extend([
            ('xgb1', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )),
            ('xgb2', xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                random_state=789,
                eval_metric='logloss'
            ))
        ])
        logger.info("Added XGBoost models to ensemble")
    
    # LightGBM if available
    if LIGHTGBM_AVAILABLE:
        estimators.extend([
            ('lgb1', lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )),
            ('lgb2', lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                random_state=101,
                verbose=-1
            ))
        ])
        logger.info("Added LightGBM models to ensemble")
    
    # Create voting classifier with soft voting
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    
    logger.info(f"Created advanced ensemble with {len(estimators)} base models")
    return ensemble

def create_stacking_ensemble():
    """
    Create a stacking classifier ensemble for better performance.
    """
    if not STACKING_AVAILABLE:
        logger.warning("StackingClassifier not available, falling back to voting classifier")
        return create_advanced_voting_ensemble()
    
    logger.info("Creating stacking classifier ensemble...")
    
    # Base models
    base_models = [
        ('rf1', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        ('rf2', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=123)),
        ('lr', LogisticRegression(random_state=42, max_iter=2000, solver='liblinear', tol=1e-4))
    ]
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        base_models.append(('xgb', xgb.XGBClassifier(
            n_estimators=100, max_depth=6, random_state=42,
            eval_metric='logloss'
        )))
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        base_models.append(('lgb', lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, random_state=42, verbose=-1
        )))
    
    # Meta-learner (should be simple for small datasets)
    meta_learner = LogisticRegression(random_state=42, max_iter=2000, solver='liblinear', tol=1e-4)
    
    # Create stacking classifier
    ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=3,  # Small CV for small dataset
        n_jobs=-1,
        stack_method='predict_proba'  # Use probabilities for stacking
    )
    
    logger.info(f"Created stacking ensemble with {len(base_models)} base models")
    return ensemble

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
    
    # Handle NaN values in training and test data
    logger.info("Handling missing values in training data...")
    
    # Fill NaN values with median for numeric columns
    X_train_clean = X_train.fillna(X_train.median())
    X_test_clean = X_test.fillna(X_test.median())
    
    # Check for any remaining NaN values
    if X_train_clean.isnull().any().any():
        logger.warning(f"Still have NaN values in training data: {X_train_clean.isnull().sum().sum()}")
        # Drop rows with any remaining NaN values
        X_train_clean = X_train_clean.dropna()
        y_train = y_train[X_train_clean.index]
        logger.info(f"After dropping NaN rows: {len(X_train_clean)} training samples")
    
    if X_test_clean.isnull().any().any():
        logger.warning(f"Still have NaN values in test data: {X_test_clean.isnull().sum().sum()}")
        # Drop rows with any remaining NaN values
        X_test_clean = X_test_clean.dropna()
        y_test = y_test[X_test_clean.index]
        logger.info(f"After dropping NaN rows: {len(X_test_clean)} test samples")
    
    # Train base model
    base_model = RandomForestClassifier(n_estimators=200, random_state=42)
    base_model.fit(X_train_clean, y_train)
    
    # Train multiple ensemble models for comparison
    logger.info("Training multiple ensemble models for comparison...")
    
    ensemble_models = {}
    
    # Advanced voting ensemble
    ensemble_models['advanced_voting'] = create_advanced_voting_ensemble()
    ensemble_models['advanced_voting'].fit(X_train_clean, y_train)
    
    # Stacking ensemble (if available)
    if STACKING_AVAILABLE:
        ensemble_models['stacking'] = create_stacking_ensemble()
        ensemble_models['stacking'].fit(X_train_clean, y_train)
    
    # Apply calibration to base model and all ensemble models
    logger.info("Applying sigmoid calibration to improve probability estimates")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        cv=5,
        method='sigmoid'
    )
    calibrated_model.fit(X_train_clean, y_train)
    
    calibrated_ensembles = {}
    for name, ensemble in ensemble_models.items():
        calibrated_ensembles[name] = CalibratedClassifierCV(
            estimator=ensemble,
            cv=5,
            method='sigmoid'
        )
        calibrated_ensembles[name].fit(X_train_clean, y_train)
    
    # Evaluate on TEST data (unseen data)
    logger.info("\n=== EVALUATION ON UNSEEN TEST DATA ===")
    
    # Base model evaluation
    y_pred_base = base_model.predict(X_test_clean)
    y_prob_base = base_model.predict_proba(X_test_clean)[:, 1]
    
    acc_base = accuracy_score(y_test, y_pred_base)
    mae_base = mean_absolute_error(y_test, y_prob_base)
    mse_base = mean_squared_error(y_test, y_prob_base)
    brier_base = calculate_brier_score(y_prob_base, y_test)
    
    # Evaluate all ensemble models
    ensemble_results = {}
    for name, ensemble in ensemble_models.items():
        y_pred_ensemble = ensemble.predict(X_test_clean)
        y_prob_ensemble = ensemble.predict_proba(X_test_clean)[:, 1]
        
        acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
        mae_ensemble = mean_absolute_error(y_test, y_prob_ensemble)
        mse_ensemble = mean_squared_error(y_test, y_prob_ensemble)
        brier_ensemble = calculate_brier_score(y_prob_ensemble, y_test)
        
        ensemble_results[name] = {
            'accuracy': acc_ensemble,
            'mae': mae_ensemble,
            'mse': mse_ensemble,
            'brier': brier_ensemble,
            'prob_range': (y_prob_ensemble.min(), y_prob_ensemble.max())
        }
    
    # Calibrated model evaluation
    y_pred_calibrated = calibrated_model.predict(X_test_clean)
    y_prob_calibrated = calibrated_model.predict_proba(X_test_clean)[:, 1]
    
    acc_calibrated = accuracy_score(y_test, y_pred_calibrated)
    mae_calibrated = mean_absolute_error(y_test, y_prob_calibrated)
    mse_calibrated = mean_squared_error(y_test, y_prob_calibrated)
    brier_calibrated = calculate_brier_score(y_prob_calibrated, y_test)
    
    # Evaluate calibrated ensemble models
    calibrated_ensemble_results = {}
    for name, calibrated_ensemble in calibrated_ensembles.items():
        y_pred_calibrated_ensemble = calibrated_ensemble.predict(X_test_clean)
        y_prob_calibrated_ensemble = calibrated_ensemble.predict_proba(X_test_clean)[:, 1]
        
        acc_calibrated_ensemble = accuracy_score(y_test, y_pred_calibrated_ensemble)
        mae_calibrated_ensemble = mean_absolute_error(y_test, y_prob_calibrated_ensemble)
        mse_calibrated_ensemble = mean_squared_error(y_test, y_prob_calibrated_ensemble)
        brier_calibrated_ensemble = calculate_brier_score(y_prob_calibrated_ensemble, y_test)
        
        calibrated_ensemble_results[name] = {
            'accuracy': acc_calibrated_ensemble,
            'mae': mae_calibrated_ensemble,
            'mse': mse_calibrated_ensemble,
            'brier': brier_calibrated_ensemble,
            'prob_range': (y_prob_calibrated_ensemble.min(), y_prob_calibrated_ensemble.max())
        }
    
    # Print results
    logger.info(f"Base Model Performance (Test Set):")
    logger.info(f"  Accuracy: {acc_base:.3f} ({acc_base*100:.1f}%)")
    logger.info(f"  MAE: {mae_base:.3f}")
    logger.info(f"  MSE: {mse_base:.3f}")
    logger.info(f"  Brier Score: {brier_base:.4f}")
    logger.info(f"  Probability Range: {y_prob_base.min():.3f} - {y_prob_base.max():.3f}")
    
    # Print ensemble results
    for name, results in ensemble_results.items():
        logger.info(f"\n{name.replace('_', ' ').title()} Ensemble Performance (Test Set):")
        logger.info(f"  Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        logger.info(f"  MAE: {results['mae']:.3f}")
        logger.info(f"  MSE: {results['mse']:.3f}")
        logger.info(f"  Brier Score: {results['brier']:.4f}")
        logger.info(f"  Probability Range: {results['prob_range'][0]:.3f} - {results['prob_range'][1]:.3f}")
    
    logger.info(f"\nCalibrated Model Performance (Test Set):")
    logger.info(f"  Accuracy: {acc_calibrated:.3f} ({acc_calibrated*100:.1f}%)")
    logger.info(f"  MAE: {mae_calibrated:.3f}")
    logger.info(f"  MSE: {mse_calibrated:.3f}")
    logger.info(f"  Brier Score: {brier_calibrated:.4f}")
    logger.info(f"  Probability Range: {y_prob_calibrated.min():.3f} - {y_prob_calibrated.max():.3f}")
    
    # Print calibrated ensemble results
    for name, results in calibrated_ensemble_results.items():
        logger.info(f"\nCalibrated {name.replace('_', ' ').title()} Performance (Test Set):")
        logger.info(f"  Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        logger.info(f"  MAE: {results['mae']:.3f}")
        logger.info(f"  MSE: {results['mse']:.3f}")
        logger.info(f"  Brier Score: {results['brier']:.4f}")
        logger.info(f"  Probability Range: {results['prob_range'][0]:.3f} - {results['prob_range'][1]:.3f}")
    
    # Find best performing model
    all_results = {
        'base': {'accuracy': acc_base, 'brier': brier_base},
        'calibrated_base': {'accuracy': acc_calibrated, 'brier': brier_calibrated}
    }
    
    for name, results in ensemble_results.items():
        all_results[f'ensemble_{name}'] = {'accuracy': results['accuracy'], 'brier': results['brier']}
    
    for name, results in calibrated_ensemble_results.items():
        all_results[f'calibrated_{name}'] = {'accuracy': results['accuracy'], 'brier': results['brier']}
    
    # Find best model by Brier score (lower is better)
    best_model_name = min(all_results.keys(), key=lambda x: all_results[x]['brier'])
    best_accuracy = all_results[best_model_name]['accuracy']
    best_brier = all_results[best_model_name]['brier']
    
    logger.info(f"\n=== BEST PERFORMING MODEL ===")
    logger.info(f"Model: {best_model_name}")
    logger.info(f"Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    logger.info(f"Brier Score: {best_brier:.4f}")
    
    # Brier score breakdown for best model
    if best_model_name.startswith('calibrated_'):
        ensemble_name = best_model_name.replace('calibrated_', '')
        best_probabilities = calibrated_ensembles[ensemble_name].predict_proba(X_test_clean)[:, 1]
    elif best_model_name.startswith('ensemble_'):
        ensemble_name = best_model_name.replace('ensemble_', '')
        best_probabilities = ensemble_models[ensemble_name].predict_proba(X_test_clean)[:, 1]
    elif best_model_name == 'calibrated_base':
        best_probabilities = y_prob_calibrated
    else:
        best_probabilities = y_prob_base
    
    logger.info(f"\nBrier Score Breakdown (Best Model - {best_model_name}):")
    brier_breakdown = analyze_brier_score_breakdown(best_probabilities, y_test)
    logger.info(brier_breakdown.to_string(index=False))
    
    # Calculate improvements
    brier_improvement_base = ((brier_base - brier_calibrated) / brier_base) * 100
    acc_improvement_base = ((acc_calibrated - acc_base) / acc_base) * 100 if acc_base > 0 else 0
    
    brier_improvement_best = ((brier_base - best_brier) / brier_base) * 100
    acc_improvement_best = ((best_accuracy - acc_base) / acc_base) * 100 if acc_base > 0 else 0
    
    logger.info(f"\nImprovement from Calibration (Base Model):")
    logger.info(f"  Brier Score: {brier_improvement_base:+.1f}% ({brier_base:.4f} → {brier_calibrated:.4f})")
    logger.info(f"  Accuracy: {acc_improvement_base:+.1f}% ({acc_base:.3f} → {acc_calibrated:.3f})")
    
    logger.info(f"\nImprovement from Best Model ({best_model_name}):")
    logger.info(f"  Brier Score: {brier_improvement_best:+.1f}% ({brier_base:.4f} → {best_brier:.4f})")
    logger.info(f"  Accuracy: {acc_improvement_best:+.1f}% ({acc_base:.3f} → {best_accuracy:.3f})")
    
    # Performance consistency analysis using best model
    test_dates = test_df['game_date'].dt.date.unique()
    daily_performance = []

    # Feature importance from base RandomForest model (ensemble doesn't have single feature_importances_)
    feature_importance = pd.DataFrame({
        'feature': numeric_cols,
        'importance': base_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\nTop 22 Features (from base RandomForest):")
    logger.info(feature_importance.head(22).to_string(index=False))
    
    # Get the cleaned test dataframe for date analysis
    test_df_clean = test_df.loc[X_test_clean.index].copy()
    
    # Get best model probabilities for daily analysis
    if best_model_name.startswith('calibrated_'):
        ensemble_name = best_model_name.replace('calibrated_', '')
        best_probabilities = calibrated_ensembles[ensemble_name].predict_proba(X_test_clean)[:, 1]
    elif best_model_name.startswith('ensemble_'):
        ensemble_name = best_model_name.replace('ensemble_', '')
        best_probabilities = ensemble_models[ensemble_name].predict_proba(X_test_clean)[:, 1]
    elif best_model_name == 'calibrated_base':
        best_probabilities = y_prob_calibrated
    else:
        best_probabilities = y_prob_base
    
    for date in test_dates:
        date_mask = test_df_clean['game_date'].dt.date == date
        if date_mask.sum() > 0:
            date_probs = best_probabilities[date_mask]
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
        logger.info(f"\nDaily Performance Consistency (Best Model):")
        logger.info(f"  Average Daily Accuracy: {daily_df['accuracy'].mean():.3f}")
        logger.info(f"  Average Daily Brier Score: {daily_df['brier_score'].mean():.4f}")
        logger.info(f"  Accuracy Std Dev: {daily_df['accuracy'].std():.3f}")
        logger.info(f"  Brier Score Std Dev: {daily_df['brier_score'].std():.4f}")
        logger.info(f"  Accuracy Range: {daily_df['accuracy'].min():.3f} - {daily_df['accuracy'].max():.3f}")
    
    if today_path and os.path.exists(today_path):
        logger.info(f"\n=== MAKING PREDICTIONS FOR TODAY ===")
        logger.info(f"Loading today's features from {today_path}")
        today_df = pd.read_csv(today_path)
        X_today = today_df[numeric_cols]

        # Handle NaN values in today's data
        X_today_clean = X_today.fillna(X_today.median())
        if X_today_clean.isnull().any().any():
            logger.warning(f"Still have NaN values in today's data, dropping rows")
            X_today_clean = X_today_clean.dropna()
            today_df = today_df.loc[X_today_clean.index]
        
        # Use best model for predictions
        if best_model_name.startswith('calibrated_'):
            ensemble_name = best_model_name.replace('calibrated_', '')
            today_prob = calibrated_ensembles[ensemble_name].predict_proba(X_today_clean)[:, 1]
        elif best_model_name.startswith('ensemble_'):
            ensemble_name = best_model_name.replace('ensemble_', '')
            today_prob = ensemble_models[ensemble_name].predict_proba(X_today_clean)[:, 1]
        elif best_model_name == 'calibrated_base':
            today_prob = calibrated_model.predict_proba(X_today_clean)[:, 1]
        else:
            today_prob = base_model.predict_proba(X_today_clean)[:, 1]

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
        output_path = PROCESSED_DIR / "predictions" / f"readable_win_predictions_for_{target_date}_using_{datetime.now().strftime('%Y-%m-%d')}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")

        return result_df

    return {
        'test_accuracy': best_accuracy,
        'test_brier_score': best_brier,
        'best_model': best_model_name,
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
    parser.add_argument('--test-days', type=int, default=30,
                       help='Number of days to use for testing')
    parser.add_argument('--no-rolling', action='store_true',
                       help='Use fixed split instead of rolling window')
    parser.add_argument('--date', type=str, default=None,
                       help='Target date for predictions (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Default paths
    if args.historical is None:
        args.historical = PROCESSED_DIR / "historical_main_features_enhanced.csv"
    
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
