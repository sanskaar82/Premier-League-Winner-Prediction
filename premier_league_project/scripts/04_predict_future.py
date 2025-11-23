# ============================================
# 04_predict_future.py
# ============================================
import pandas as pd
import joblib

# Load trained model
model_path = "../models/premier_league_winner_model.pkl"
model = joblib.load(model_path)
print(" Model loaded successfully!\n")

# Define feature columns (same as used during training)
features = [
    "members", "foreign_players", "mean_age", "MOY",
    "points", "Goal_Diff", "Wins", "Draws", "Losses",
    "Goals_For", "Goals_Against"
]

# 🧩 Create mock data for 2025 (replace later with real or scraped data)
data_2025 = {
    "Team": [
        "Manchester City", "Arsenal", "Liverpool", "Tottenham", 
        "Manchester United", "Newcastle", "Chelsea", "Aston Villa"
    ],
    "members": [50, 47, 48, 45, 49, 46, 52, 44],
    "foreign_players": [35, 29, 32, 28, 30, 27, 33, 25],
    "mean_age": [26.5, 25.8, 26.9, 26.2, 27.0, 25.7, 26.4, 26.0],
    "MOY": [0.7, 0.5, 0.6, 0.5, 0.4, 0.5, 0.5, 0.4],
    "points": [89, 85, 83, 75, 72, 68, 64, 62],
    "Goal_Diff": [55, 48, 46, 35, 28, 24, 20, 18],
    "Wins": [28, 26, 25, 22, 20, 19, 17, 16],
    "Draws": [5, 7, 8, 9, 12, 11, 13, 14],
    "Losses": [5, 5, 5, 7, 6, 8, 8, 8],
    "Goals_For": [92, 88, 85, 78, 70, 66, 64, 61],
    "Goals_Against": [37, 40, 39, 43, 42, 46, 44, 43],
}

df_2025 = pd.DataFrame(data_2025)

# Make predictions
X_future = df_2025[features]
df_2025["Win_Probability"] = model.predict_proba(X_future)[:, 1]

# Sort by most likely to win
df_2025 = df_2025.sort_values("Win_Probability", ascending=False)

# Display results
print(" Predicted 2025 Premier League Winner Probabilities:\n")
print(df_2025[["Team", "Win_Probability"]].to_string(index=False, justify="center", col_space=15))

# Save to CSV
df_2025.to_csv("../outputs/predicted_2025_results.csv", index=False)
print("\n Saved predictions to ../outputs/predicted_2025_results.csv")
