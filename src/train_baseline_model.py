"""
train_baseline_model.py
-----------------------
Creates simple features and trains a baseline logistic regression model
to predict NBA game outcomes (win/loss).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1. Load the cleaned data
df = pd.read_csv("data/nba_games_clean.csv")

# 2. Basic feature engineering
# Compute average points per team per season
team_avg = df.groupby("TEAM_NAME")["PTS"].mean().reset_index()
team_avg.rename(columns={"PTS": "AVG_PTS"}, inplace=True)

# Merge back with main DataFrame
df = df.merge(team_avg, on="TEAM_NAME", how="left")

# 3. Handle missing values
# Fill NaN with sensible defaults
df["IS_HOME"].fillna(0, inplace=True)
df["AVG_PTS"].fillna(df["AVG_PTS"].mean(), inplace=True)
df["WIN"].fillna(0, inplace=True)

# Verify data types
df["IS_HOME"] = df["IS_HOME"].astype(int)
df["AVG_PTS"] = df["AVG_PTS"].astype(float)
df["WIN"] = df["WIN"].astype(int)

# 4. Prepare features and labels
X = df[["IS_HOME", "AVG_PTS"]]
y = df["WIN"]

# Double check for NaN (debug safety)
print("NaN counts after cleaning:")
print(X.isna().sum())

# 5. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train baseline logistic regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== Baseline Model Results =====")
print(f"Accuracy: {acc:.3f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save model
joblib.dump(model, "data/baseline_model.pkl")
print("\n✅ Model saved as data/baseline_model.pkl")
