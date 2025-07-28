import pandas as pd

# Load the enhanced CSV
df = pd.read_csv('data/processed/historical_main_features_enhanced.csv')

print("Checking for offensive features in the CSV...")
print(f"Total columns: {len(df.columns)}")

# Look for offensive features
offensive_features = [
    'team_batting_avg_diff',
    'team_obp_diff', 
    'team_slg_diff',
    'team_ops_diff',
    'team_hr_per_game_diff',
    'team_runs_per_game_diff',
    'offensive_efficiency_diff',
    'offensive_composite_diff'
]

print("\nSearching for offensive features:")
found_features = []
for feature in offensive_features:
    if feature in df.columns:
        found_features.append(feature)
        print(f"✅ FOUND: {feature}")
    else:
        print(f"❌ NOT FOUND: {feature}")

print(f"\nFound {len(found_features)} out of {len(offensive_features)} offensive features")

# Show first few values if found
if found_features:
    print("\nFirst few values of found features:")
    for feature in found_features[:3]:  # Show first 3
        print(f"{feature}: {df[feature].iloc[0]}")

# Show all columns that contain 'offensive' or 'team_' and 'diff'
print("\nAll columns containing 'offensive' or 'team_' and 'diff':")
offensive_cols = [col for col in df.columns if ('offensive' in col.lower() or ('team_' in col and 'diff' in col))]
for col in offensive_cols:
    print(f"  {col}") 