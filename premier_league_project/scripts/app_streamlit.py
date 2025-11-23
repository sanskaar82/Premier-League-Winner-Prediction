import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Load model
# -----------------------------
model_path = os.path.join("..", "models", "premier_league_winner_model.pkl")
model = joblib.load(model_path)

st.title("🏆 Premier League Winner Predictor")
st.markdown("Predict next season's winner probabilities using machine learning.")

# -----------------------------
# Upload or sample data
# -----------------------------
st.sidebar.header("📂 Upload Season Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")
else:
    st.info("No file uploaded. Using sample 2025 data...")
    df = pd.DataFrame({
        "Team": ["Manchester City", "Liverpool", "Arsenal", "Tottenham", "Manchester United", "Newcastle", "Chelsea", "Aston Villa"],
        "members": [45, 44, 42, 40, 43, 39, 41, 38],
        "foreign_players": [29, 28, 27, 25, 26, 23, 25, 21],
        "mean_age": [27.2, 26.9, 26.5, 25.8, 27.0, 26.3, 26.7, 25.9],
        "MOY": [0.45, 0.38, 0.37, 0.32, 0.34, 0.28, 0.30, 0.27],
        "rank": [1, 2, 3, 4, 5, 6, 7, 8],
        "points": [90, 84, 80, 69, 65, 63, 60, 59],
        "Goal_Diff": [55, 46, 40, 22, 18, 15, 10, 8],
        "Wins": [29, 26, 25, 20, 18, 17, 16, 15],
        "Draws": [5, 6, 7, 9, 11, 12, 12, 14],
        "Losses": [4, 6, 6, 9, 9, 9, 10, 9],
        "Goals_For": [95, 88, 84, 70, 65, 62, 61, 58],
        "Goals_Against": [40, 42, 44, 48, 50, 47, 51, 50],
    })

# -----------------------------
# Predict
# -----------------------------
X = df.drop(columns=["Team"], errors="ignore")

# --- Align features ---
X = X.rename(columns={
    'GF': 'Goals_For',
    'GA': 'Goals_Against',
    'PassAccuracy': 'Pass_Accuracy',
    'Possession': 'possession',
})

# Compute missing engineered features
if 'Goal_Diff' not in X.columns:
    X['Goal_Diff'] = X['Goals_For'] - X['Goals_Against']

# Fill missing model features with defaults
for col in ['foreign_players', 'average_age', 'MOY']:
    if col not in X.columns:
        X[col] = 0  # or realistic defaults, e.g. 10, 26.5, etc.

# Ensure column order matches model training
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

probs = model.predict_proba(X)[:, 1]

df["Win_Probability"] = probs
df["Normalized_Prob"] = df["Win_Probability"] / df["Win_Probability"].sum()

# Sort and display
df_sorted = df.sort_values(by="Normalized_Prob", ascending=False)
st.subheader("🏆 Predicted Winner Probabilities")
st.dataframe(df_sorted[["Team", "Normalized_Prob"]])

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x="Normalized_Prob", y="Team", data=df_sorted, ax=ax, palette="viridis")
ax.set_title("Premier League Winner Probability")
ax.set_xlabel("Probability")
st.pyplot(fig)

# -----------------------------
# Save results
# -----------------------------
output_path = os.path.join("..", "outputs", "predicted_future_results_streamlit.csv")
df_sorted.to_csv(output_path, index=False)
st.success(f"💾 Saved results to {output_path}")
