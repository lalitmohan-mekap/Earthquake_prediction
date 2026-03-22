"""
evaluate_models.py
==================
Evaluate all trained models on the test set and generate comparison plots.

Input:  .tmp/test.csv, .tmp/models/
Output: .tmp/results.csv, .tmp/plots/eval_*.png
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TMP_DIR = os.path.join(BASE_DIR, ".tmp")
TEST_FILE = os.path.join(TMP_DIR, "test.csv")
MODEL_DIR = os.path.join(TMP_DIR, "models")
PLOT_DIR = os.path.join(TMP_DIR, "plots")
RESULTS_FILE = os.path.join(TMP_DIR, "results.csv")

FEATURE_COLS = [
    "latitude", "longitude", "depth",
    "year", "month", "hour", "day_of_week",
    "days_since_last",
    "nst", "gap", "dmin", "rms", "magError", "horizontalError", "depthError",
    "mag_depth_interaction", "gap_rms_interaction", "precision_interaction",
    "mag_lag_1", "depth_change",
    "rolling_mean_mag_20", "rolling_std_mag_20", "rolling_mean_depth_50",
    "lat_bin", "lon_bin", "grid_id"
]

sns.set_theme(style="whitegrid", font_scale=1.1)


def load_test_data():
    """Load test set."""
    if not os.path.exists(TEST_FILE):
        print(f"ERROR: {TEST_FILE} not found. Run preprocess_data.py first.")
        sys.exit(1)
    df = pd.read_csv(TEST_FILE)
    print(f"Loaded {len(df)} test samples")
    return df


def load_model(name: str):
    """Load a saved model."""
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping.")
        return None
    return joblib.load(path)


def evaluate_model_1(X, X_scaled, y_mag, results):
    """Evaluate magnitude prediction models."""
    print("\n" + "=" * 60)
    print("MODEL 1: Magnitude Prediction (Regression)")
    print("=" * 60)

    models = {
        "Ensemble (RF+GBR)": ("model1_ensemble_regressor.joblib", False),
        "Linear Regression": ("model1_linear_regression.joblib", True),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (name, (fname, use_scaled)) in enumerate(models.items()):
        model = load_model(fname)
        if model is None:
            continue

        X_input = X_scaled if use_scaled else X
        y_pred = model.predict(X_input)

        r2 = r2_score(y_mag, y_pred)
        mae = mean_absolute_error(y_mag, y_pred)
        rmse = np.sqrt(mean_squared_error(y_mag, y_pred))

        print(f"\n  {name}:")
        print(f"    R² Score = {r2:.4f}")
        print(f"    MAE      = {mae:.4f}")
        print(f"    RMSE     = {rmse:.4f}")

        results.append({
            "Model Group": "Model 1 (Regression)",
            "Algorithm": name,
            "R2_Score": round(r2, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Accuracy": "",
            "ROC_AUC": "",
        })

        # Predicted vs Actual scatter
        ax = axes[idx]
        ax.scatter(y_mag, y_pred, alpha=0.3, s=5, color="#2196F3")
        lims = [min(y_mag.min(), y_pred.min()), max(y_mag.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Magnitude")
        ax.set_ylabel("Predicted Magnitude")
        ax.set_title(f"{name}\nR²={r2:.3f}, MAE={mae:.3f}")
        ax.legend()

    fig.suptitle("Model 1: Predicted vs Actual Magnitude", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eval_model1_predictions.png"), dpi=150)
    plt.close(fig)
    print("  ✓ Saved eval_model1_predictions.png")


def evaluate_model_2(X, X_scaled, y_risk, results):
    """Evaluate risk classification models."""
    print("\n" + "=" * 60)
    print("MODEL 2: Seismic Risk Classification")
    print("=" * 60)

    le = load_model("model2_label_encoder.joblib")
    if le is None:
        return
    y_encoded = le.transform(y_risk)

    models = {
        "Logistic Regression": ("model2_logistic_regression.joblib", True),
        "Random Forest Classifier": ("model2_random_forest_classifier.joblib", False),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (name, (fname, use_scaled)) in enumerate(models.items()):
        model = load_model(fname)
        if model is None:
            continue

        X_input = X_scaled if use_scaled else X
        y_pred = model.predict(X_input)

        acc = accuracy_score(y_encoded, y_pred)
        cm = confusion_matrix(y_encoded, y_pred)

        print(f"\n  {name}:")
        print(f"    Accuracy = {acc:.4f}")
        print(f"    Classification Report:")
        print(classification_report(y_encoded, y_pred, target_names=le.classes_))

        results.append({
            "Model Group": "Model 2 (Classification)",
            "Algorithm": name,
            "R2_Score": "",
            "MAE": "",
            "RMSE": "",
            "Accuracy": round(acc, 4),
            "ROC_AUC": "",
        })

        # Confusion matrix heatmap
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nAccuracy={acc:.3f}")

    fig.suptitle("Model 2: Confusion Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eval_model2_confusion.png"), dpi=150)
    plt.close(fig)
    print("  ✓ Saved eval_model2_confusion.png")


def evaluate_model_3(X_scaled, y_high, results):
    """Evaluate high-magnitude probability model."""
    print("\n" + "=" * 60)
    print("MODEL 3: High-Magnitude Probability (Binary)")
    print("=" * 60)

    model = load_model("model3_gradient_boosting_binary.joblib")
    if model is None:
        return

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    acc = accuracy_score(y_high, y_pred)
    fpr, tpr, _ = roc_curve(y_high, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"\n  Logistic Regression (Binary):")
    print(f"    Accuracy = {acc:.4f}")
    print(f"    ROC AUC  = {roc_auc:.4f}")
    print(f"    Classification Report:")
    print(classification_report(y_high, y_pred, target_names=["Not High", "High (≥6.0)"]))

    results.append({
        "Model Group": "Model 3 (Probability)",
        "Algorithm": "Gradient Boosting (Binary)",
        "R2_Score": "",
        "MAE": "",
        "RMSE": "",
        "Accuracy": round(acc, 4),
        "ROC_AUC": round(roc_auc, 4),
    })

    # ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#F44336", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Model 3: ROC Curve — High-Magnitude Prediction")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eval_model3_roc_curve.png"), dpi=150)
    plt.close(fig)
    print("  ✓ Saved eval_model3_roc_curve.png")


def plot_model_comparison(results):
    """Create a summary bar chart comparing all models."""
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # R² comparison (Model 1 only)
    reg = df[df["R2_Score"] != ""].copy()
    if not reg.empty:
        reg["R2_Score"] = reg["R2_Score"].astype(float)
        axes[0].barh(reg["Algorithm"], reg["R2_Score"], color=["#2196F3", "#4CAF50"])
        axes[0].set_xlabel("R² Score")
        axes[0].set_title("Regression: R² Score")
        axes[0].set_xlim(0, 1)

    # Accuracy comparison (Model 2 + 3)
    clf = df[df["Accuracy"] != ""].copy()
    if not clf.empty:
        clf["Accuracy"] = clf["Accuracy"].astype(float)
        colors = ["#FF9800", "#9C27B0", "#F44336"][:len(clf)]
        axes[1].barh(clf["Algorithm"], clf["Accuracy"], color=colors)
        axes[1].set_xlabel("Accuracy")
        axes[1].set_title("Classification: Accuracy")
        axes[1].set_xlim(0, 1)

    # MAE comparison (Model 1 only)
    if not reg.empty:
        reg["MAE"] = reg["MAE"].astype(float)
        axes[2].barh(reg["Algorithm"], reg["MAE"], color=["#2196F3", "#4CAF50"])
        axes[2].set_xlabel("MAE")
        axes[2].set_title("Regression: MAE (lower is better)")

    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eval_model_comparison.png"), dpi=150)
    plt.close(fig)
    print("\n  ✓ Saved eval_model_comparison.png")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = load_test_data()
    X = df[FEATURE_COLS].values
    y_mag = df["mag"].values
    y_risk = df["risk_level"].values
    y_high = df["is_high_magnitude"].values

    # Load scaler
    scaler = load_model("scaler.joblib")
    if scaler is None:
        print("ERROR: scaler.joblib not found. Run train_models.py first.")
        sys.exit(1)
    X_scaled = scaler.transform(X)

    results = []

    evaluate_model_1(X, X_scaled, y_mag, results)
    evaluate_model_2(X, X_scaled, y_risk, results)
    evaluate_model_3(X_scaled, y_high, results)
    plot_model_comparison(results)

    # Save results table
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\n✓ Results saved to {RESULTS_FILE}")
    print("\n" + results_df.to_string(index=False))


if __name__ == "__main__":
    main()
