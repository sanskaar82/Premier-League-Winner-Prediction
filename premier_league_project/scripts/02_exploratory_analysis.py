# 02_exploratory_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up paths
data_path = "../outputs/cleaned_premier_league.csv"
plots_folder = "../outputs/plots"
os.makedirs(plots_folder, exist_ok=True)

# Load data
df = pd.read_csv(data_path)
print("✅ Data loaded successfully!")
print(df.head())

# ---------------------------
# Basic statistics
# ---------------------------
print("\n Summary Statistics:")
print(df.describe())

print("\nWinner distribution:")
print(df['Winner'].value_counts())

# ---------------------------
# Visualizations
# ---------------------------
sns.set(style="whitegrid")

#  Points vs Goal Difference (color by Winner)
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="Goal_Diff", y="points", hue="Winner", palette="coolwarm")
plt.title("Points vs Goal Difference (by Winner)")
plt.savefig(f"{plots_folder}/points_vs_goal_diff.png")
plt.close()

#  Average Points per Season
plt.figure(figsize=(10,6))
season_avg = df.groupby("Season")["points"].mean().reset_index()
sns.lineplot(data=season_avg, x="Season", y="points", marker="o")
plt.title("Average Points per Season")
plt.savefig(f"{plots_folder}/avg_points_per_season.png")
plt.close()

#  Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{plots_folder}/correlation_heatmap.png")
plt.close()

print("\n All plots saved inside ../outputs/plots/")
