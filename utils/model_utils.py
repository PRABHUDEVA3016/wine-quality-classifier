"""
Model Utility Functions
Handles training, evaluation, saving, and loading of the RandomForest model.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("saved_model", "random_forest_model.pkl")
SCALER_PATH = os.path.join("saved_model", "scaler.pkl")

RF_PARAMS = {
    "n_estimators":  200,
    "max_depth":     12,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "random_state":  42,
    "n_jobs":        -1,
}


# ──────────────────────────────────────────────────────────────────────────────
# Train / Evaluate
# ──────────────────────────────────────────────────────────────────────────────
def train_model(X, y, test_size: float = 0.2, random_state: int = 42, **kwargs):
    """
    Split data, train RandomForestClassifier, and return all artefacts.

    Returns
    -------
    model       : trained RandomForestClassifier
    X_train, X_test, y_train, y_test : split arrays
    metrics     : dict with accuracy, confusion_matrix, report, feature_importance
    """
    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Merge custom parameters with defaults
    params = RF_PARAMS.copy()
    params.update(kwargs)
    params["random_state"] = random_state

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":           round(accuracy_score(y_test, y_pred) * 100, 2),
        "roc_auc":            round(roc_auc_score(y_test, y_proba) * 100, 2),
        "confusion_matrix":   confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Bad Wine (0)", "Good Wine (1)"],
            output_dict=True
        ),
        "classification_report_text": classification_report(
            y_test, y_pred, target_names=["Bad Wine (0)", "Good Wine (1)"]
        ),
        "feature_importance": dict(
            zip(X.columns, model.feature_importances_)
        ),
        "train_samples":      len(X_train),
        "test_samples":       len(X_test),
    }

    return model, X_train, X_test, y_train, y_test, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────────────────────────────────
def save_model(model) -> str:
    """Save trained model with joblib. Returns the save path."""
    os.makedirs("saved_model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH


def load_model():
    """Load model from disk. Returns None if not found."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def model_exists() -> bool:
    """Check whether a trained model exists on disk."""
    return os.path.exists(MODEL_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────
def predict_quality(model, input_features: np.ndarray):
    """
    Make a prediction for a single sample.

    Parameters
    ----------
    model         : trained RandomForestClassifier
    input_features: 1-D numpy array with 11 feature values

    Returns
    -------
    prediction    : int   (0 = Bad Wine, 1 = Good Wine)
    confidence    : float (probability of the predicted class, 0–100 %)
    probabilities : dict  {"Bad Wine": p0, "Good Wine": p1}
    """
    sample       = input_features.reshape(1, -1)
    prediction   = int(model.predict(sample)[0])
    proba        = model.predict_proba(sample)[0]
    confidence   = round(float(np.max(proba)) * 100, 2)
    probabilities = {
        "Bad Wine":  round(float(proba[0]) * 100, 2),
        "Good Wine": round(float(proba[1]) * 100, 2),
    }
    return prediction, confidence, probabilities
