import joblib
import pandas as pd

model = joblib.load("../data/advanced_model.pkl")
df = pd.read_csv("../data/nba_games_clean.csv")

importances = model.feature_importances_
features = ["IS_HOME", "AVG_PTS_LAST_5", "AVG_OPP_PTS_LAST_5"]

for f, imp in zip(features, importances):
    print(f"{f}: {imp:.3f}")
