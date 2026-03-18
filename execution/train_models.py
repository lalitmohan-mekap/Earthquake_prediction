"""
train_models.py
===============
Train all three model groups for earthquake prediction.

Model 1: Magnitude Prediction (regression)
    - Linear Regression
    - Random Forest Regressor

Model 2: Seismic Risk Classification (Low / Medium / High)
    - Logistic Regression
    - Random Forest Classifier

Model 3: High-Magnitude Probability (binary)
    - Logistic Regression (binary)

Input:  .tmp/train.csv
Output: .tmp/models/*.joblib
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TMP_DIR = os.path.join(BASE_DIR, ".tmp")
TRAIN_FILE = os.path.join(TMP_DIR, "train.csv")
MODEL_DIR = os.path.join(TMP_DIR, "models")

# Feature columns (same across all models)
FEATURE_COLS = [
    "latitude", "longitude", "depth",
    "year", "month", "hour", "day_of_week",
    "days_since_last",
]


def load_train_data():
    """Load training data and prepare feature matrix."""
    if not os.path.exists(TRAIN_FILE):
        print(f"ERROR: {TRAIN_FILE} not found. Run preprocess_data.py first.")
        sys.exit(1)

    df = pd.read_csv(TRAIN_FILE)
    print(f"Loaded {len(df)} training samples")
    # For verification speed, sample 50k events
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42).copy()
        print(f"Subsampled to {len(df)} training samples for verification speed")
    return df


def train_model_1(X: np.ndarray, y_mag: np.ndarray, scaler: StandardScaler):
    """Model 1: Magnitude prediction (regression)."""
    print("\n" + "=" * 60)
    print("MODEL 1: Magnitude Prediction (Regression)")
    print("=" * 60)

    X_scaled = scaler.transform(X)

    # 1a. Linear Regression
    print("\n  Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_scaled, y_mag)
    train_r2 = lr.score(X_scaled, y_mag)
    print(f"  Train R² = {train_r2:.4f}")
    joblib.dump(lr, os.path.join(MODEL_DIR, "model1_linear_regression.joblib"))
    print("  ✓ Saved model1_linear_regression.joblib")

    # 1b. Random Forest Regressor
    print("\n  Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y_mag)  # RF doesn't need scaling
    train_r2_rf = rf.score(X, y_mag)
    print(f"  Train R² = {train_r2_rf:.4f}")
    joblib.dump(rf, os.path.join(MODEL_DIR, "model1_random_forest_regressor.joblib"))
    print("  ✓ Saved model1_random_forest_regressor.joblib")


def train_model_2(X: np.ndarray, y_risk: np.ndarray, scaler: StandardScaler):
    """Model 2: Seismic risk classification (Low / Medium / High)."""
    print("\n" + "=" * 60)
    print("MODEL 2: Seismic Risk Classification")
    print("=" * 60)

    X_scaled = scaler.transform(X)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_risk)
    joblib.dump(le, os.path.join(MODEL_DIR, "model2_label_encoder.joblib"))

    # 2a. Logistic Regression
    print("\n  Training Logistic Regression...")
    log_reg = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    log_reg.fit(X_scaled, y_encoded)
    train_acc = log_reg.score(X_scaled, y_encoded)
    print(f"  Train Accuracy = {train_acc:.4f}")
    joblib.dump(log_reg, os.path.join(MODEL_DIR, "model2_logistic_regression.joblib"))
    print("  ✓ Saved model2_logistic_regression.joblib")

    # 2b. Random Forest Classifier
    print("\n  Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1)
    rf.fit(X, y_encoded)
    train_acc_rf = rf.score(X, y_encoded)
    print(f"  Train Accuracy = {train_acc_rf:.4f}")
    joblib.dump(rf, os.path.join(MODEL_DIR, "model2_random_forest_classifier.joblib"))
    print("  ✓ Saved model2_random_forest_classifier.joblib")


def train_model_3(X: np.ndarray, y_high: np.ndarray, scaler: StandardScaler):
    """Model 3: High-magnitude probability (binary classification)."""
    print("\n" + "=" * 60)
    print("MODEL 3: High-Magnitude Probability (Binary)")
    print("=" * 60)

    X_scaled = scaler.transform(X)

    print("\n  Training Logistic Regression (binary)...")
    log_reg = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    log_reg.fit(X_scaled, y_high)
    train_acc = log_reg.score(X_scaled, y_high)
    print(f"  Train Accuracy = {train_acc:.4f}")
    joblib.dump(log_reg, os.path.join(MODEL_DIR, "model3_logistic_regression_binary.joblib"))
    print("  ✓ Saved model3_logistic_regression_binary.joblib")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_train_data()

    # Prepare features and targets
    X = df[FEATURE_COLS].values
    y_mag = df["mag"].values
    y_risk = df["risk_level"].values
    y_high = df["is_high_magnitude"].values

    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print("✓ Saved scaler.joblib")

    # Train all model groups
    train_model_1(X, y_mag, scaler)
    train_model_2(X, y_risk, scaler)
    train_model_3(X, y_high, scaler)

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print(f"Models saved to: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
