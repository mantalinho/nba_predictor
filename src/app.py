import streamlit as st
import joblib
import pandas as pd

# Load model
MODEL_PATH = "data/advanced_model_30NBA.pkl"

@st.cache_resource
def load_model():
    data = joblib.load(MODEL_PATH)
    return data["model"], data["le_team"], data["le_opp"], data["features"]

model, le_team, le_opp, features = load_model()

st.set_page_config(page_title="NBA Game Predictor", layout="centered")
st.title("🏀 NBA Game Outcome Predictor")

st.write("Select two teams and the home/away status to predict the winner!")

# --- Dropdowns for user input ---
teams = le_team.classes_

col1, col2 = st.columns(2)
with col1:
    team = st.selectbox("Team", teams)
with col2:
    opponent = st.selectbox("Opponent", teams)

is_home = st.radio("Is the team playing at home?", ["Yes", "No"])
is_home_value = 1 if is_home == "Yes" else 0

# --- Predict button ---
if st.button("Predict Winner"):
    if team == opponent:
        st.error("❌ Please choose two different teams.")
    else:
        team_enc = le_team.transform([team])[0]
        opp_enc = le_opp.transform([opponent])[0]

        X_input = pd.DataFrame([[team_enc, opp_enc, is_home_value]], columns=features)
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0]

        winner = team if pred == 1 else opponent
        confidence = prob[pred] * 100

        st.success(f"🏆 Predicted Winner: **{winner}** ({confidence:.1f}% confidence)")
