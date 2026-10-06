import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------
# 1. Load dataset
# --------------------------------

df = pd.read_csv("dataset/gaming_data.csv")


# --------------------------------
# 2. Separate X and y
# --------------------------------

X = df.drop("performance", axis=1)
y = df["performance"]


# --------------------------------
# 3. Train-test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# 4. Feature scaling
# --------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# --------------------------------
# 5. Create Logistic Regression
# --------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# --------------------------------
# 6. Train the model
# --------------------------------

model.fit(X_train_scaled, y_train)


# --------------------------------
# 7. Make predictions
# --------------------------------

y_pred = model.predict(X_test_scaled)


# --------------------------------
# 8. Evaluate model
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Results")
print("---------------------------")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))