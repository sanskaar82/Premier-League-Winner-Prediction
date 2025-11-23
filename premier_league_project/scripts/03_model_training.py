# ============================================
# 03_model_training.py
# ============================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# Load cleaned dataset
data_path = "../outputs/cleaned_premier_league.csv"
df = pd.read_csv(data_path)
print("✅ Data loaded successfully!")
print(df.head())

# Feature selection — choose stats that make sense
features = [
    "members", "foreign_players", "mean_age", "MOY",
    "points", "Goal_Diff", "Wins", "Draws", "Losses",
    "Goals_For", "Goals_Against"
]
X = df[features]
y = df["Winner"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n🔍 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save trained model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/premier_league_winner_model.pkl")
print("\n💾 Model saved to ../models/premier_league_winner_model.pkl")

# Feature importance
import matplotlib.pyplot as plt
import seaborn as sns

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importance — Predicting Premier League Winner")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("../outputs/plots/feature_importance.png")
plt.show()
print("📊 Feature importance plot saved!")
