import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/gaming_data.csv")

# Separate features (X) and target (y)
X = df.drop("performance", axis=1)
y = df["performance"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display results
print("Total records:", len(df))

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining performance distribution:")
print(y_train.value_counts())

print("\nTesting performance distribution:")
print(y_test.value_counts())