"""
model.py
--------
Trains a Random Forest Classifier on the cleaned NEO dataset.
Returns the trained model, metrics, and evaluation data.
No model is persisted to disk.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)

from src.preprocessing import get_X_y, get_feature_names


def train_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train a Random Forest classifier on cleaned NEO data.

    Args:
        df: Cleaned DataFrame from preprocessing

    Returns:
        Tuple of (trained model, metrics dict)
    """
    X, y = get_X_y(df)
    feature_names = get_feature_names(df)

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use class_weight='balanced' to handle class imbalance
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=["Safe", "Hazardous"]),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_names": feature_names,
        "feature_importances": model.feature_importances_,
        "X_test": X_test,
    }

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    metrics["roc_fpr"] = fpr
    metrics["roc_tpr"] = tpr
    metrics["roc_auc"] = auc(fpr, tpr)

    return model, metrics


def predict_single(model: RandomForestClassifier, input_data: Dict[str, float], feature_names: list) -> Dict[str, Any]:
    """
    Run prediction on a single asteroid's input data.

    Args:
        model: Trained RandomForestClassifier
        input_data: Dict of feature_name -> value
        feature_names: Ordered list of feature names used during training

    Returns:
        Dict with 'label', 'probability', and 'confidence'
    """
    values = [input_data.get(f, 0.0) for f in feature_names]
    X_input = np.array(values).reshape(1, -1)

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    return {
        "label": "Potentially Hazardous" if pred == 1 else "Safe",
        "prediction": int(pred),
        "probability_hazardous": float(proba[1]),
        "probability_safe": float(proba[0]),
    }
