import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# 1. PREPARE DATA
def prepare_data(file_path):
    """Load and preprocess the customer churn dataset."""

    df = pd.read_csv(file_path)

    # Encode Gender
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Drop unnecessary columns
    df = df.drop(["Surname", "Geography"], axis=1)

    # Separate features and target
    X = df.drop(["Exited"], axis=1)
    y = df["Exited"]

    # Drop identifiers
    X = X.drop(columns=["RowNumber", "CustomerId"])

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    return X_train, X_test, y_train, y_test


# 2. TRAIN MODEL
def train_model(X_train, y_train):
    """Train the Random Forest classifier."""

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)

    return rf


# 3. EVALUATE MODEL
def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy and confusion matrix."""

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy score:", accuracy * 100, "%")
    print("\nConfusion Matrix:")
    print(matrix)

    return accuracy, matrix


# 4. SAVE MODEL
def save_model(model, file_path="classifier.joblib"):
    """Save the trained model using Joblib."""

    joblib.dump(model, file_path)
    print("Model saved successfully!")


# 5. LOAD MODEL
def load_model(file_path="classifier.joblib"):
    """Load a previously saved model."""

    model = joblib.load(file_path)
    print("Model loaded successfully!")

    return model
