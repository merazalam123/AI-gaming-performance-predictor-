import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of players
n = 1000

# -----------------------------
# 1. Generate historical data
# -----------------------------

matches_played = np.random.randint(20, 301, n)

win_rate = np.random.uniform(25, 80, n)

avg_kills = np.random.uniform(5, 25, n)

avg_deaths = np.random.uniform(5, 20, n)

avg_assists = np.random.uniform(2, 12, n)

accuracy = np.random.uniform(40, 85, n)

headshot_rate = np.random.uniform(10, 50, n)

avg_damage = np.random.uniform(150, 550, n)

playtime_hours = np.random.uniform(20, 500, n)

recent_win_rate = np.clip(
    win_rate + np.random.normal(0, 10, n),
    10,
    90
)

# -----------------------------------------
# 2. Create a hidden player skill variable
# -----------------------------------------

skill = (
    0.30 * win_rate
    + 0.25 * accuracy
    + 0.20 * (avg_kills / 25 * 100)
    + 0.15 * (avg_damage / 550 * 100)
    + 0.10 * recent_win_rate
)

# Add randomness to represent future match conditions
future_score = skill + np.random.normal(0, 8, n)

# -----------------------------------------
# 3. Convert future score into target
# -----------------------------------------

performance = np.where(
    future_score >= 65,
    "HIGH",
    np.where(
        future_score >= 50,
        "MEDIUM",
        "LOW"
    )
)

# -----------------------------------------
# 4. Create DataFrame
# -----------------------------------------

data = {
    "matches_played": matches_played,
    "win_rate": win_rate,
    "avg_kills": avg_kills,
    "avg_deaths": avg_deaths,
    "avg_assists": avg_assists,
    "accuracy": accuracy,
    "headshot_rate": headshot_rate,
    "avg_damage": avg_damage,
    "playtime_hours": playtime_hours,
    "recent_win_rate": recent_win_rate,
    "performance": performance
}

df = pd.DataFrame(data)

# Round decimal values
df = df.round(2)

# -----------------------------------------
# 5. Save dataset
# -----------------------------------------

df.to_csv("dataset/gaming_data.csv", index=False)

# -----------------------------------------
# 6. Display information
# -----------------------------------------

print("Dataset created successfully! 🎮")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nPerformance distribution:")
print(df["performance"].value_counts())

print("\nDataset saved to:")
print("dataset/gaming_data.csv")