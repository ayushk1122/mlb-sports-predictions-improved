import pandas as pd

# Load historical data
df = pd.read_csv('data/processed/historical_main_features.csv')
print("Unique home teams:", sorted(df['home_team'].unique()))
print("Unique away teams:", sorted(df['away_team'].unique()))

# Load offensive stats
offensive = pd.read_csv('data/processed/team_offensive_stats_2024.csv')
print("\nOffensive stats teams:", sorted(offensive['team_name'].unique()))

# Find mismatches
historical_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
offensive_teams = set(offensive['team_name'].unique())

print("\nTeams in historical but not in offensive:", sorted(historical_teams - offensive_teams))
print("Teams in offensive but not in historical:", sorted(offensive_teams - historical_teams)) 