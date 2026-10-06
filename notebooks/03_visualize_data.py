import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/gaming_data.csv")

# -----------------------------
# Graph 1: Performance classes
# -----------------------------

df["performance"].value_counts().plot(kind="bar")

plt.title("Performance Distribution")
plt.xlabel("Performance")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.show()


# -----------------------------
# Graph 2: Win Rate
# -----------------------------

df.boxplot(column="win_rate", by="performance")

plt.title("Win Rate vs Performance")
plt.suptitle("")
plt.xlabel("Performance")
plt.ylabel("Win Rate (%)")
plt.tight_layout()
plt.show()


# -----------------------------
# Graph 3: Accuracy
# -----------------------------

df.boxplot(column="accuracy", by="performance")

plt.title("Accuracy vs Performance")
plt.suptitle("")
plt.xlabel("Performance")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()


# -----------------------------
# Graph 4: Average Kills
# -----------------------------

df.boxplot(column="avg_kills", by="performance")

plt.title("Average Kills vs Performance")
plt.suptitle("")
plt.xlabel("Performance")
plt.ylabel("Average Kills")
plt.tight_layout()
plt.show()