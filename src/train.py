from sklearn.metrics import accuracy_score
import joblib
import os


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Ensure models directory exists
    os.makedirs("../models", exist_ok=True)

    # Save trained model
    joblib.dump(model, "../models/random_forest_model.pkl")

    return accuracy