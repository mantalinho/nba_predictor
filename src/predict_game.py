"""
predict_game.py
----------------
Simple CLI NBA game predictor using the advanced Random Forest model.
"""

import pandas as pd
import joblib

# 1. Load advanced model
model = joblib.load("data/advanced_model.pkl")

# 2. Load cleaned data to compute rolling averages if needed
df = pd.read_csv("data/nba_games_clean.csv")

# 3. Function to get team rolling averages
def get_team_stats(team_name):
    team_data = df[df["TEAM_NAME"] == team_name].sort_values("GAME_DATE")
    avg_pts_last5 = team_data["PTS"].tail(5).mean()
    opp_pts_last5 = team_data["PTS"].tail(5).mean()  # simplified
    return avg_pts_last5, opp_pts_last5

# 4. CLI Input
team_name = input("Enter your team name: ")
is_home = input("Is the team playing at home? (y/n): ").lower() == 'y'

# 5. Get rolling stats
avg_pts, avg_opp_pts = get_team_stats(team_name)

# 6. Prepare feature vector
X_new = pd.DataFrame([{
    "IS_HOME": int(is_home),
    "AVG_PTS_LAST_5": avg_pts,
    "AVG_OPP_PTS_LAST_5": avg_opp_pts
}])

# 7. Predict
win_prob = model.predict_proba(X_new)[0][1]  # probability of WIN
pred = model.predict(X_new)[0]

print("\n===== Prediction =====")
print(f"Team: {team_name}")
print(f"Home: {'Yes' if is_home else 'No'}")
print(f"Predicted outcome: {'WIN' if pred==1 else 'LOSS'}")
print(f"Win probability: {win_prob:.2%}")
