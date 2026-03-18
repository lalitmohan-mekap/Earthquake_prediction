"""
preprocess_data.py
==================
Clean raw earthquake data and engineer features for ML models.

Input:  .tmp/raw_earthquakes.csv
Output: .tmp/features.csv, .tmp/train.csv, .tmp/test.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TMP_DIR = os.path.join(BASE_DIR, ".tmp")
INPUT_FILE = os.path.join(TMP_DIR, "raw_earthquakes.csv")
FEATURES_FILE = os.path.join(TMP_DIR, "features.csv")
TRAIN_FILE = os.path.join(TMP_DIR, "train.csv")
TEST_FILE = os.path.join(TMP_DIR, "test.csv")

# ---------------------------------------------------------------------------
# Risk thresholds (per synopsis)
# ---------------------------------------------------------------------------
LOW_THRESHOLD = 4.0      # mag < 4.0 → Low
HIGH_THRESHOLD = 5.5     # mag > 5.5 → High, else Medium
HIGH_MAG_THRESHOLD = 6.0 # binary: is_high_magnitude


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw CSV and drop invalid rows."""
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    print(f"  Raw rows: {len(df)}")

    # Keep only earthquake-type events
    if "type" in df.columns:
        df = df[df["type"] == "earthquake"].copy()
        print(f"  After type filter: {len(df)}")

    # Drop rows missing critical columns
    required = ["time", "latitude", "longitude", "depth", "mag"]
    before = len(df)
    df.dropna(subset=required, inplace=True)
    print(f"  Dropped {before - len(df)} rows with missing values → {len(df)} remain")

    # Drop negative depths (data errors)
    df = df[df["depth"] >= 0].copy()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from raw columns."""
    print("Engineering features...")

    # Parse timestamp
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Temporal features
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek  # 0=Monday

    # Recurrence: days since previous event
    df["days_since_last"] = df["time"].diff().dt.total_seconds() / 86400.0
    df["days_since_last"].fillna(0, inplace=True)

    # Depth bins (useful categorical feature for some analysis)
    df["depth_bin"] = pd.cut(
        df["depth"],
        bins=[0, 30, 70, 300, 700],
        labels=["Shallow", "Intermediate", "Deep", "Very_Deep"],
        include_lowest=True,
    )

    # Target columns ----------------------------------------------------------
    # Model 1 target: magnitude (already exists as 'mag')

    # Model 2 target: risk_level (Low / Medium / High)
    conditions = [
        df["mag"] < LOW_THRESHOLD,
        (df["mag"] >= LOW_THRESHOLD) & (df["mag"] <= HIGH_THRESHOLD),
        df["mag"] > HIGH_THRESHOLD,
    ]
    df["risk_level"] = np.select(conditions, ["Low", "Medium", "High"], default="Low")

    # Model 3 target: is_high_magnitude (binary)
    df["is_high_magnitude"] = (df["mag"] >= HIGH_MAG_THRESHOLD).astype(int)

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select final feature columns + targets."""
    feature_cols = [
        "latitude", "longitude", "depth",
        "year", "month", "hour", "day_of_week",
        "days_since_last",
    ]
    target_cols = ["mag", "risk_level", "is_high_magnitude"]

    # Keep only what we need
    out = df[feature_cols + target_cols].copy()
    return out


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run fetch_earthquake_data.py first.")
        sys.exit(1)

    df = load_and_clean(INPUT_FILE)
    df = engineer_features(df)
    features = select_features(df)

    # Drop any remaining NaN (e.g. from depth_bin edge cases)
    features.dropna(inplace=True)
    print(f"\nFinal dataset: {len(features)} events, {len(features.columns)} columns")
    print(f"Columns: {list(features.columns)}")

    # Save full feature set
    features.to_csv(FEATURES_FILE, index=False)
    print(f"Saved features → {FEATURES_FILE}")

    # Train/test split (80/20, stratify on risk_level for balanced classes)
    train, test = train_test_split(
        features, test_size=0.2, random_state=42, stratify=features["risk_level"]
    )
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    print(f"Train: {len(train)} rows → {TRAIN_FILE}")
    print(f"Test:  {len(test)} rows → {TEST_FILE}")

    # Summary stats
    print("\n--- Class Distribution (risk_level) ---")
    print(features["risk_level"].value_counts().to_string())
    print(f"\n--- High-magnitude events (≥{HIGH_MAG_THRESHOLD}) ---")
    print(features["is_high_magnitude"].value_counts().to_string())
    print("\n--- Magnitude stats ---")
    print(features["mag"].describe().to_string())


if __name__ == "__main__":
    main()
