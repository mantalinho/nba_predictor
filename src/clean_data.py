"""
clean_data.py
--------------
Cleans the raw NBA data file and prepares it for analysis.
"""

import pandas as pd
import glob

def clean_nba_data(input_file, output_file):
    """Reads, cleans, and saves the NBA data."""
    print("Cleaning NBA data...")

    df = pd.read_csv(input_file)

    # Rename WL column to WIN (if it exists)
    if 'WL' in df.columns:
        df.rename(columns={'WL': 'WIN'}, inplace=True)

    # Convert WIN to numeric (1 for W, 0 for L)
    df['WIN'] = df['WIN'].apply(lambda x: 1 if x == 'W' else 0)

    # Safely handle missing or non-string values in MATCHUP
    def home_away_flag(x):
        if isinstance(x, str):
            return 1 if 'vs.' in x else 0
        return 0  # default if NaN or invalid
    df['IS_HOME'] = df['MATCHUP'].apply(home_away_flag)

    # Convert GAME_DATE to datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

    # Drop rows with missing GAME_DATE just in case
    df = df.dropna(subset=['GAME_DATE'])

    # Save cleaned version
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved to {output_file}")

if __name__ == "__main__":
    # Automatically find the latest raw data file
    latest_raw = sorted(glob.glob("data/nba_games_raw_*.csv"))[-1]
    clean_nba_data(latest_raw, "data/nba_games_clean.csv")
