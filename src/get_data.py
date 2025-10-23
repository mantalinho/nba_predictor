"""
get_data.py
-------------
Fetches NBA game data using nba_api and saves it as a CSV file.
"""

from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
from datetime import datetime

def fetch_nba_data():
    """Fetches regular season NBA games from the official API."""
    print("Fetching NBA data...")

    # Get all regular season games
    gamefinder = leaguegamefinder.LeagueGameFinder(season_type_nullable='Regular Season')
    games = gamefinder.get_data_frames()[0]

    print(f"Fetched {len(games)} games.")
    return games


def save_data(df, filepath):
    """Saves the dataframe to a CSV file."""
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


if __name__ == "__main__":
    data = fetch_nba_data()

    # Keep only relevant columns
    columns_to_keep = ['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'WL', 'PTS']
    data = data[columns_to_keep]

    # Save raw data
    today = datetime.now().strftime("%Y%m%d")
    output_path = f"data/nba_games_raw_{today}.csv"
    save_data(data, output_path)
