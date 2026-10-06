import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load dataset
df = pd.read_csv("dataset/gaming_data.csv")


# Separate features and target
X = df.drop("performance", axis=1)
y = df["performance"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Scale the data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create and train Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# Save the model
joblib.dump(model, "model/gaming_model.pkl")


# Save the scaler
joblib.dump(scaler, "model/scaler.pkl")


print("Model saved successfully!")
print("Scaler saved successfully!")
print("\nFiles created:")
print("model/gaming_model.pkl")
print("model/scaler.pkl")