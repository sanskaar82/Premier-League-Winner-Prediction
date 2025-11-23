# generate_plots.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Paths (relative to scripts/)
DATA_PATH = "../outputs/cleaned_premier_league.csv"
MODEL_PATH = "../models/premier_league_winner_model.pkl"
PLOTS_FOLDER = "../outputs/plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# ========== Figure A: Points distribution histogram ==========
plt.figure(figsize=(8,5))
plt.hist(df['points'], bins=12, edgecolor='black')
plt.title("Distribution of Team Points (2015–2024)")
plt.xlabel("Points")
plt.ylabel("Number of Team-Seasons")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_FOLDER, "points_distribution.png"))
plt.close()

# ========== Figure A(a): Average points per season (you already had this) ==========
season_avg = df.groupby("Season")["points"].mean().reset_index()
plt.figure(figsize=(10,5))
plt.plot(season_avg["Season"], season_avg["points"], marker='o')
plt.xticks(rotation=45)
plt.title("Average Points per Season (2015–2024)")
plt.xlabel("Season")
plt.ylabel("Average Points")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_FOLDER, "avg_points_per_season.png"))
plt.close()

# ========== Figure 2: Correlation heatmap ==========
plt.figure(figsize=(10,9))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_FOLDER, "correlation_heatmap.png"))
plt.close()

# ========== Figure 3: Winners vs Non-winners average comparison ==========
comp = df.groupby("Winner")[["points", "Goal_Diff", "Wins", "Losses"]].mean().T
plt.figure(figsize=(8,5))
comp.plot(kind="bar", rot=0)
plt.title("Average Team Statistics: Winners vs Non-Winners")
plt.ylabel("Average Value")
plt.legend(["Non-Winner (0)","Winner (1)"], loc='best')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_FOLDER, "winner_vs_nonwinner.png"))
plt.close()

# ========== Figure: Points vs Goal_Diff scatter (colored by Winner) ==========
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="Goal_Diff", y="points", hue="Winner", palette="coolwarm", edgecolor="k")
plt.title("Points vs Goal Difference (Champions highlighted)")
plt.xlabel("Goal Difference")
plt.ylabel("Points")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_FOLDER, "points_vs_goal_diff.png"))
plt.close()

# ========== Feature importance plot (from trained Random Forest model) ==========
# Load model, compute feature importances if model provides them
try:
    model = joblib.load(MODEL_PATH)
    # feature names used in training
    features = list(model.feature_names_in_)
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Feature Importance — Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, "feature_importance.png"))
    plt.close()
except Exception as e:
    print("Could not generate feature importance plot:", e)

# ========== Future prediction chart (use your dynamic prediction sample) ==========
# If you have a preds csv saved earlier, load it; otherwise produce sample predictions
preds_csv = "../outputs/predicted_future_results.csv"
if os.path.exists(preds_csv):
    df_pred = pd.read_csv(preds_csv)
else:
    # build a small example DataFrame (same teams you used earlier)
    sample = {
        "Team": ["Manchester City","Liverpool","Arsenal","Tottenham","Man United","Newcastle","Chelsea","Aston Villa"],
        "Win_Probability": [0.59, 0.335, 0.235, 0.010, 0.010, 0.0, 0.0, 0.0]
    }
    df_pred = pd.DataFrame(sample)
# normalize for a clean bar chart
df_pred["Normalized_Prob"] = df_pred["Win_Probability"] / df_pred["Win_Probability"].sum()
df_pred = df_pred.sort_values("Normalized_Prob", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Normalized_Prob", y="Team", data=df_pred, palette="Blues_r")
plt.title("Predicted Title Odds (Normalized)")
plt.xlabel("Normalized Probability")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_FOLDER, "future_prediction_chart.png"))
plt.close()

print("All plots saved to:", PLOTS_FOLDER)
