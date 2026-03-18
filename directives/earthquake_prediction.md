# Multi-Model Earthquake Prediction — Master SOP

## Overview
This project implements three ML model groups for seismic analysis using USGS earthquake catalog data:
1. **Magnitude Prediction** (regression): Linear Regression + Random Forest Regressor
2. **Seismic Risk Classification** (multiclass): Logistic Regression + Random Forest Classifier
3. **High-Magnitude Probability** (binary): Logistic Regression

## Execution Order
Run scripts in this exact sequence from the project root:

```
python execution/fetch_earthquake_data.py      # Step 1: Download data
python execution/preprocess_data.py            # Step 2: Clean + feature engineer
python execution/exploratory_analysis.py       # Step 3: EDA plots
python execution/train_models.py               # Step 4: Train all models
python execution/evaluate_models.py            # Step 5: Evaluate + compare
python execution/generate_report.py            # Step 6: Final report
```

## Script I/O Map

| Script | Inputs | Outputs |
|--------|--------|---------|
| `fetch_earthquake_data.py` | USGS API URL | `.tmp/raw_earthquakes.csv` |
| `preprocess_data.py` | `.tmp/raw_earthquakes.csv` | `.tmp/features.csv`, `.tmp/train.csv`, `.tmp/test.csv` |
| `exploratory_analysis.py` | `.tmp/features.csv` | `.tmp/plots/eda_*.png` |
| `train_models.py` | `.tmp/train.csv` | `.tmp/models/*.joblib` |
| `evaluate_models.py` | `.tmp/test.csv`, `.tmp/models/` | `.tmp/results.csv`, `.tmp/plots/eval_*.png` |
| `generate_report.py` | `.tmp/results.csv`, `.tmp/plots/` | `deliverables/research_report.md` |

## Edge Cases / Learnings
- USGS API limits responses to 20,000 events per query. The fetch script paginates by year to handle this.
- Some CSV rows may have null depth or magnitude — preprocessing drops these.
- Class imbalance in risk levels: most events are Low magnitude. Use `class_weight='balanced'` in classifiers.
