# 01_clean_data.py
import os
import pandas as pd

# -----------------------------
# 1️⃣ Combine all CSVs
# -----------------------------
data_folder = "../data"
files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

df_list = []
for f in files:
    temp = pd.read_csv(os.path.join(data_folder, f))
    temp["Season"] = f.replace(".csv", "")
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
print(f"✅ Combined shape: {df.shape}\n")

print("✅ Columns:")
print(df.columns)

# -----------------------------
# 2️⃣ Standardize column names
# -----------------------------
rename_map = {
    'DIF': 'Goal_Diff',
    'Gain': 'Wins',
    'Null': 'Draws',
    'defeat': 'Losses',
    'BP': 'Goals_For',
    'BC': 'Goals_Against'
}
df = df.rename(columns=rename_map)

# -----------------------------
# 3️⃣ Basic cleaning
# -----------------------------
print("\nBefore cleaning:", df.shape)

# Fill missing values
df = df.fillna(0)

# Convert numerics
num_cols = ['members', 'foreign_players', 'mean_age', 'rank', 'points',
            'Goal_Diff', 'Wins', 'Draws', 'Losses', 'Goals_For', 'Goals_Against']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# Mark the winner
df['Winner'] = (df['rank'] == 1).astype(int)

# Clean team names
df['Team'] = df['Team'].str.strip()
df = df.drop_duplicates()

print("After cleaning:", df.shape)
print(df.info())

# -----------------------------
# 4️⃣ Save cleaned file
# -----------------------------
output_path = "../outputs/cleaned_premier_league.csv"
df.to_csv(output_path, index=False)
print(f"\n💾 Saved cleaned dataset as {output_path}")
