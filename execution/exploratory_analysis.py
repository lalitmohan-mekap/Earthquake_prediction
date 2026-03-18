"""
exploratory_analysis.py
=======================
Generate EDA visualizations from the processed feature set.

Input:  .tmp/features.csv
Output: .tmp/plots/eda_*.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TMP_DIR = os.path.join(BASE_DIR, ".tmp")
INPUT_FILE = os.path.join(TMP_DIR, "features.csv")
PLOT_DIR = os.path.join(TMP_DIR, "plots")

# Style
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
FIGSIZE = (10, 6)


def plot_magnitude_distribution(df: pd.DataFrame):
    """Histogram of earthquake magnitudes."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(df["mag"], bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Earthquake Magnitudes")
    ax.axvline(df["mag"].mean(), color="red", linestyle="--", label=f'Mean = {df["mag"].mean():.2f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eda_magnitude_distribution.png"), dpi=150)
    plt.close(fig)
    print("  ✓ Magnitude distribution")


def plot_depth_vs_magnitude(df: pd.DataFrame):
    """Scatter plot of depth vs magnitude."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    scatter = ax.scatter(
        sample["depth"], sample["mag"],
        c=sample["mag"], cmap="YlOrRd", alpha=0.5, s=10, edgecolors="none"
    )
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Depth vs Magnitude")
    fig.colorbar(scatter, ax=ax, label="Magnitude")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eda_depth_vs_magnitude.png"), dpi=150)
    plt.close(fig)
    print("  ✓ Depth vs magnitude")


def plot_events_per_year(df: pd.DataFrame):
    """Time series of earthquake counts per year."""
    yearly = df.groupby("year").size()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(yearly.index, yearly.values, color="#4CAF50", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Earthquakes")
    ax.set_title("Earthquake Count per Year")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eda_events_per_year.png"), dpi=150)
    plt.close(fig)
    print("  ✓ Events per year")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap of numeric features."""
    numeric_cols = ["latitude", "longitude", "depth", "year", "month",
                    "hour", "day_of_week", "days_since_last", "mag"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.5, ax=ax
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eda_correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  ✓ Correlation heatmap")


def plot_risk_distribution(df: pd.DataFrame):
    """Bar chart of risk level distribution."""
    counts = df["risk_level"].value_counts()
    colors = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[colors.get(x, "#999") for x in counts.index],
                  edgecolor="white")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Count")
    ax.set_title("Seismic Risk Level Distribution")

    # Add count labels on bars
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(val), ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eda_risk_distribution.png"), dpi=150)
    plt.close(fig)
    print("  ✓ Risk distribution")


def plot_geographic_distribution(df: pd.DataFrame):
    """Scatter plot of earthquakes on lat/lon coordinates."""
    fig, ax = plt.subplots(figsize=(14, 7))
    sample = df.sample(n=min(10000, len(df)), random_state=42)
    scatter = ax.scatter(
        sample["longitude"], sample["latitude"],
        c=sample["mag"], cmap="hot_r", alpha=0.4, s=5, edgecolors="none"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Distribution of Earthquakes")
    fig.colorbar(scatter, ax=ax, label="Magnitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eda_geographic_distribution.png"), dpi=150)
    plt.close(fig)
    print("  ✓ Geographic distribution")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run preprocess_data.py first.")
        sys.exit(1)

    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} events from {INPUT_FILE}")
    print("Generating EDA plots...")

    plot_magnitude_distribution(df)
    plot_depth_vs_magnitude(df)
    plot_events_per_year(df)
    plot_correlation_heatmap(df)
    plot_risk_distribution(df)
    plot_geographic_distribution(df)

    print(f"\nAll EDA plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
