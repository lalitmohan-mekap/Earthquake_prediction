"""
predict.py
==========
Inference script — predict earthquake magnitude, risk level, and high-magnitude
probability from input features using the best trained models.

Input:  Feature values (via CLI arguments or interactive prompt)
Output: Predicted magnitude, risk level, probability
"""

import os
import sys
import numpy as np
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, ".tmp", "models")

FEATURE_NAMES = [
    "latitude", "longitude", "depth",
    "year", "month", "hour", "day_of_week",
    "days_since_last",
    "nst", "gap", "dmin", "rms", "magError", "horizontalError", "depthError",
    "mag_depth_interaction", "gap_rms_interaction", "precision_interaction",
    "mag_lag_1", "depth_change",
    "rolling_mean_mag_20", "rolling_std_mag_20", "rolling_mean_depth_50",
    "lat_bin", "lon_bin", "grid_id"
]


def load_models():
    """Load all models and preprocessing objects."""
    models = {}
    required_files = {
        "scaler": "scaler.joblib",
        "model_ensemble": "model1_ensemble_regressor.joblib",
        "model_risk": "model2_random_forest_classifier.joblib",
        "label_encoder": "model2_label_encoder.joblib",
        "model_binary": "model3_gradient_boosting_binary.joblib",
    }

    for key, fname in required_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run train_models.py first.")
            sys.exit(1)
        models[key] = joblib.load(path)

    return models


def predict(features: np.ndarray, models: dict) -> dict:
    """
    Run all three model groups on the input features.

    Parameters
    ----------
    features : np.ndarray, shape (1, 8)
        [latitude, longitude, depth, year, month, hour, day_of_week, days_since_last]
    models : dict
        Loaded model objects.

    Returns
    -------
    dict with keys: magnitude_lr, magnitude_rf, risk_lr, risk_rf,
                    high_mag_probability
    """
    scaler = models["scaler"]
    X_scaled = scaler.transform(features)

    # Model 1: Magnitude prediction (Ensemble)
    mag_pred = models["model_ensemble"].predict(features)[0]

    # Model 2: Risk classification
    le = models["label_encoder"]
    risk_idx = models["model_risk"].predict(features)[0]
    risk_level = le.inverse_transform([risk_idx])[0]

    # Model 3: High-magnitude probability
    high_prob = models["model_binary"].predict_proba(X_scaled)[0][1]

    return {
        "magnitude": round(float(mag_pred), 2),
        "risk_level": risk_level,
        "high_mag_probability": round(float(high_prob), 4),
    }


def interactive_mode(models: dict):
    """Prompt user for feature values and predict."""
    print("\n" + "=" * 60)
    print("EARTHQUAKE PREDICTION — Interactive Mode")
    print("=" * 60)
    print("Enter feature values (or 'q' to quit):\n")

    while True:
        try:
            values = []
            for name in FEATURE_NAMES:
                val = input(f"  {name}: ")
                if val.lower() == "q":
                    print("Exiting.")
                    return
                values.append(float(val))

            features = np.array(values).reshape(1, -1)
            result = predict(features, models)

            print("\n  --- Predictions ---")
            print(f"  Predicted Magnitude:        {result['magnitude']}")
            print(f"  Seismic Risk Level:         {result['risk_level']}")
            print(f"  High-Magnitude Probability: {result['high_mag_probability']:.2%}")
            print()

        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input or interrupted. Try again or press 'q' to quit.")


def cli_mode(args: list, models: dict):
    """Accept feature values as command-line arguments."""
    if len(args) != len(FEATURE_NAMES):
        print(f"ERROR: Expected {len(FEATURE_NAMES)} features, got {len(args)}.")
        print(f"Usage: python predict.py {' '.join(FEATURE_NAMES)}")
        sys.exit(1)

    values = [float(a) for a in args]
    features = np.array(values).reshape(1, -1)
    result = predict(features, models)

    print("\n--- Earthquake Prediction Results ---")
    print(f"  Predicted Magnitude:        {result['magnitude']}")
    print(f"  Seismic Risk Level:         {result['risk_level']}")
    print(f"  High-Magnitude Probability: {result['high_mag_probability']:.2%}")


def main():
    models = load_models()

    if len(sys.argv) > 1:
        cli_mode(sys.argv[1:], models)
    else:
        interactive_mode(models)


if __name__ == "__main__":
    main()
