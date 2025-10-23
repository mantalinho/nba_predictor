import pandas as pd

# === File paths ===
raw_csv = r"C:\Users\kvass\OneDrive\Desktop\nba_predictor\data\nba_games_raw_20251023.csv"
output_csv = r"C:\Users\kvass\OneDrive\Desktop\nba_predictor\data\nba_games_clean.csv"

# === Load raw data ===
df = pd.read_csv(raw_csv)
print(f"✅ Loaded raw data: {len(df)} rows")

# === Official 30 NBA teams (team *names*) ===
nba_team_names = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
    "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors",
    "Houston Rockets", "Indiana Pacers", "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies",
    "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
    "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"
]

# === Find column with team names ===
team_col = None
for col in df.columns:
    if "TEAM_NAME" in col.upper():
        team_col = col
        break

if not team_col:
    raise ValueError("❌ No column found containing 'TEAM_NAME'!")

print(f"Detected team column: {team_col}")

# === Filter only NBA teams ===
df = df[df[team_col].isin(nba_team_names)].copy()
print(f"🏀 Rows after filtering NBA teams: {len(df)}")

# === Handle missing values ===
df = df.fillna(method="ffill").fillna(method="bfill")

# === Create WIN column ===
if "WL" in df.columns:
    df["WIN"] = df["WL"].apply(lambda x: 1 if str(x).strip().upper() == "W" else 0)
elif "PTS" in df.columns and "PTS_OPP" in df.columns:
    df["WIN"] = (df["PTS"] > df["PTS_OPP"]).astype(int)
else:
    print("⚠️ No WL or score data found to create WIN column.")

# === Create IS_HOME column ===
if "MATCHUP" in df.columns:
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if isinstance(x, str) and "vs." in x else 0)
else:
    df["IS_HOME"] = 0  # fallback

# === Save cleaned data ===
df.to_csv(output_csv, index=False)
print(f"✅ Cleaned dataset saved to: {output_csv}")
print(f"📊 Final columns: {list(df.columns)}")
