# Multi-Model Earthquake Prediction

**URP 4301 | Group 41010_3 | 8th Semester 2026**
*ITER, Department of Computer Science & Information Technology*

## Overview

This undergraduate research project implements a multi-model ML framework for earthquake prediction using USGS historical seismic data. The system performs three tasks:

1. **Magnitude Prediction** — Linear Regression + Random Forest Regressor
2. **Seismic Risk Classification** (Low/Medium/High) — Logistic Regression + Random Forest Classifier
3. **High-Magnitude Probability** — Binary Logistic Regression

## Setup

```bash
# Install dependencies
python -m pip install -r requirements.txt
```

## Usage — Run Full Pipeline

```bash
python execution/fetch_earthquake_data.py    # 1. Download USGS data
python execution/preprocess_data.py          # 2. Clean + feature engineer
python execution/exploratory_analysis.py     # 3. Generate EDA plots
python execution/train_models.py             # 4. Train all models
python execution/evaluate_models.py          # 5. Evaluate + compare
python execution/generate_report.py          # 6. Generate research report
```

## Usage — Run Predictions

```bash
# Interactive mode
python execution/predict.py

# CLI mode (lat, lon, depth, year, month, hour, day_of_week, days_since_last)
python execution/predict.py 35.6 139.7 10.0 2025 3 14 2 0.5
```

## Project Structure

```
├── directives/                  # SOPs (Standard Operating Procedures)
│   ├── earthquake_prediction.md # Master SOP
│   └── data_acquisition.md     # Data sourcing SOP
├── execution/                   # Python scripts
│   ├── fetch_earthquake_data.py
│   ├── preprocess_data.py
│   ├── exploratory_analysis.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── predict.py
│   └── generate_report.py
├── deliverables/                # Final outputs
│   └── research_report.md
├── .tmp/                        # Intermediate files (gitignored)
│   ├── raw_earthquakes.csv
│   ├── features.csv, train.csv, test.csv
│   ├── models/
│   └── plots/
├── requirements.txt
└── README.md
```

## Evaluation Metrics

|          Model             |          Metrics           |
|----------------------------|----------------------------|
| Magnitude Prediction       | R² Score, MAE, RMSE        |
| Risk Classification        | Accuracy, Confusion Matrix |
| High-Magnitude Probability | ROC Curve, AUC, Accuracy   |

## References

1. Breiman, "Random Forests," 2001
2. Hastie et al., *The Elements of Statistical Learning*, 2009
3. Zhang et al., "Earthquake magnitude prediction using ML methods," 2019
4. Mignan & Broccardo, "One neuron vs deep learning in aftershock prediction," 2019
5. Alkhalifah & Al-Homoud, "Seismic hazard prediction using ANNs and ML," 2019
