"""
Train a machine learning model to predict NBA game outcomes (win/loss)
based on cleaned data from data/nba_games_clean.csv
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------- file paths ----------
clean_csv = "data/nba_games_clean.csv"
model_path = "data/advanced_model_30NBA.pkl"

# ---------- load cleaned data ----------
if not os.path.exists(clean_csv):
    raise FileNotFoundError(f"File not found: {clean_csv}. Run clean_data.py first!")

df = pd.read_csv(clean_csv)
print(f"Loaded cleaned data: {len(df)} rows")

# ---------- sanity check ----------
if "WIN" not in df.columns:
    raise KeyError("Column 'WIN' not found. Ensure clean_data.py created it correctly.")
if "TEAM_NAME" not in df.columns or "OPPONENT" not in df.columns:
    raise KeyError("Columns TEAM_NAME or OPPONENT missing from cleaned dataset.")

# ---------- prepare features ----------
# Fill missing values if any
df = df.fillna(method="ffill").fillna(method="bfill")

# encode categorical variables
le_team = LabelEncoder()
le_opp = LabelEncoder()

df["TEAM_ENC"] = le_team.fit_transform(df["TEAM_NAME"])
df["OPPONENT_ENC"] = le_opp.fit_transform(df["OPPONENT"])

# features we’ll use
features = ["TEAM_ENC", "OPPONENT_ENC", "IS_HOME"]
target = "WIN"

X = df[features]
y = df[target]

# ---------- split data ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- train model ----------
model = RandomForestClassifier(
    n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# ---------- evaluate ----------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc:.3f}")
print(classification_report(y_test, y_pred))

# ---------- save model + encoders ----------
os.makedirs("data", exist_ok=True)
joblib.dump({
    "model": model,
    "le_team": le_team,
    "le_opp": le_opp,
    "features": features
}, model_path)

print(f"Model saved to {model_path}")
