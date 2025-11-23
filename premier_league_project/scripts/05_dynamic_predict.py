# ============================================
# 05_dynamic_predict.py
# ============================================
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load model ---
model_path = "../models/premier_league_winner_model.pkl"
model = joblib.load(model_path)
print("✅ Model loaded successfully!\n")

# --- Step 2: Define feature columns ---
features = [
    "members", "foreign_players", "mean_age", "MOY",
    "points", "Goal_Diff", "Wins", "Draws", "Losses",
    "Goals_For", "Goals_Against"
]

# --- Step 3: Choose data input mode ---
print("📅 Predict future season winner")
print("1️⃣  Use sample 2025 data (default)")
print("2️⃣  Load your own CSV (e.g., future_stats_2026.csv)")
choice = input("Select (1 or 2): ").strip()

if choice == "2":
    file_path = input("Enter path to your new season CSV file: ").strip()
    df_future = pd.read_csv(file_path)
    print(f"✅ Loaded data from {file_path}")
else:
    # Sample mock data (editable for any season)
    data_future = {
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
    df_future = pd.DataFrame(data_future)
    print("✅ Using built-in sample season data.\n")

# --- Step 4: Make predictions ---
X_future = df_future[features]
df_future["Win_Probability"] = model.predict_proba(X_future)[:, 1]

# Normalize to sum to 1 (for relative odds)
df_future["Normalized_Prob"] = df_future["Win_Probability"] / df_future["Win_Probability"].sum()

# Sort by most likely
df_future = df_future.sort_values("Win_Probability", ascending=False)

# --- Step 5: Display & Save ---
print("\n🏆 Predicted Season Winner Probabilities:\n")
print(df_future[["Team", "Win_Probability", "Normalized_Prob"]]
      .to_string(index=False, justify="center", col_space=15))

# Save results
os.makedirs("../outputs", exist_ok=True)
df_future.to_csv("../outputs/predicted_future_results.csv", index=False)
print("\n💾 Saved predictions to ../outputs/predicted_future_results.csv")

# --- Step 6: Visualization ---
plt.figure(figsize=(10, 6))
sns.barplot(
    x="Normalized_Prob", y="Team", data=df_future,
    palette="Blues_r", edgecolor="black"
)
plt.title("🏆 Premier League — Predicted Title Odds")
plt.xlabel("Normalized Title Probability (sums to 1)")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig("../outputs/plots/future_prediction_chart.png")
plt.show()
print(" Visualization saved to ../outputs/plots/future_prediction_chart.png")
