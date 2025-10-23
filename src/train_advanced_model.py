"""
train_advanced_model.py
-----------------------
Creates advanced rolling statistics and trains a stronger NBA win predictor model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load the cleaned data
df = pd.read_csv("data/nba_games_clean.csv")

# 2. Sort by date so rolling averages make sense
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["TEAM_NAME", "GAME_DATE"]).reset_index(drop=True)

# 3. Compute rolling averages per team
df["AVG_PTS_LAST_5"] = (
    df.groupby("TEAM_NAME")["PTS"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(drop=True)
)

# For defensive stat (points allowed)
# We need to estimate opponent points per game
df["OPP_PTS"] = df.groupby("TEAM_NAME")["PTS"].shift(-1)  # simple placeholder if opp data missing

# Normally you'd merge on opponent team here — we’ll simplify for now
df["AVG_OPP_PTS_LAST_5"] = (
    df.groupby("TEAM_NAME")["OPP_PTS"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(drop=True)
)

# Fill missing values
df = df.fillna(method="bfill").fillna(method="ffill")

# 4. Select features
features = ["IS_HOME", "AVG_PTS_LAST_5", "AVG_OPP_PTS_LAST_5"]
X = df[features]
y = df["WIN"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train improved model
model = RandomForestClassifier(
    n_estimators=150, max_depth=8, random_state=42
)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("===== Advanced Model Results =====")
print(f"Accuracy: {acc:.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save model
joblib.dump(model, "data/advanced_model.pkl")
print("\n✅ Advanced model saved as data/advanced_model.pkl")
