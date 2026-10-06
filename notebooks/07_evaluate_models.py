import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("dataset/gaming_data.csv")

print("Dataset loaded successfully!")


# --------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("performance", axis=1)
y = df["performance"]


# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 5. CREATE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}


# --------------------------------------------------
# 6. TRAIN AND EVALUATE
# --------------------------------------------------

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Use scaled data for Logistic Regression and KNN
    if name in ["Logistic Regression", "KNN"]:

        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

    else:

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)


    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", round(accuracy, 3))


    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    print("Confusion Matrix:")
    print(cm)


    # Display confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["LOW", "MEDIUM", "HIGH"]
    )

    display.plot()

    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()
    plt.show()