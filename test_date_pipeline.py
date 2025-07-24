#!/usr/bin/env python3
"""
Test script for the date-specific MLB prediction pipeline.
This script tests the complete pipeline for a specific date.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def test_date_pipeline(target_date_str):
    """Test the complete pipeline for a specific date."""
    print(f"Testing pipeline for date: {target_date_str}")
    
    # Convert string to date object
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    today = datetime.today().date()
    
    # Determine if we should test evaluation (only for past dates)
    should_test_evaluation = target_date < today
    
    print(f"Today is: {today}")
    print(f"Target date is: {target_date}")
    print(f"Should test evaluation: {should_test_evaluation}")
    
    # Test 1: Run predictions
    print("\n=== Test 1: Running predictions ===")
    cmd1 = [sys.executable, "run_daily_pipeline.py", "--date", target_date_str]
    print(f"Running: {' '.join(cmd1)}")
    
    try:
        result1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=Path.cwd())
        print("STDOUT:", result1.stdout)
        if result1.stderr:
            print("STDERR:", result1.stderr)
        print(f"Return code: {result1.returncode}")
    except Exception as e:
        print(f"Error running predictions: {e}")
        return False
    
    # Test 2: Run evaluation (only for past dates)
    if should_test_evaluation:
        print("\n=== Test 2: Running evaluation with auto-scraping ===")
        cmd2 = [sys.executable, "modeling/evaluate_prediction_accuracy.py", "--date", target_date_str]
        print(f"Running: {' '.join(cmd2)}")
        
        try:
            result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=Path.cwd())
            print("STDOUT:", result2.stdout)
            if result2.stderr:
                print("STDERR:", result2.stderr)
            print(f"Return code: {result2.returncode}")
        except Exception as e:
            print(f"Error running evaluation: {e}")
            return False
    else:
        print(f"\n=== Test 2: Skipping evaluation (target date {target_date_str} is not in the past) ===")
    
    # Test 3: Check generated files
    print("\n=== Test 3: Checking generated files ===")
    base_dir = Path.cwd()
    
    # Check prediction files
    prediction_files = list(base_dir.glob(f"data/predictions/readable_win_predictions_for_{target_date_str}_*.csv"))
    print(f"Prediction files found: {len(prediction_files)}")
    for f in prediction_files:
        print(f"  - {f}")
    
    # Check actual results file (only if we should have evaluated)
    results_file = base_dir / "data" / "processed" / f"historical_results_{target_date_str}.csv"
    if should_test_evaluation:
        print(f"Results file exists: {results_file.exists()}")
        if results_file.exists():
            print(f"  - {results_file}")
    else:
        print(f"Results file check skipped (target date is not in the past)")
    
    # Check matchup files
    current_matchup = base_dir / "data" / "raw" / f"mlb_probable_pitchers_{target_date_str}.csv"
    historical_matchup = base_dir / "data" / "raw" / "historical_matchups" / f"historical_matchups_{target_date_str}.csv"
    print(f"Current matchup file exists: {current_matchup.exists()}")
    print(f"Historical matchup file exists: {historical_matchup.exists()}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_date_pipeline.py YYYY-MM-DD")
        print("Example: python test_date_pipeline.py 2025-07-23")
        sys.exit(1)
    
    target_date = sys.argv[1]
    success = test_date_pipeline(target_date)
    
    if success:
        print("\n✅ Pipeline test completed successfully!")
    else:
        print("\n❌ Pipeline test failed!")
        sys.exit(1) 