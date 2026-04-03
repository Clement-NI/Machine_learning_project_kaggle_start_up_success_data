# Machine Learning Project — Kaggle Startup Success

**PSL M1 — March 2026**  
Authors: Ognjen, Yuzhen

Final machine learning project on the Kaggle Startup Success dataset (100 000 rows).  
Combines supervised classification and unsupervised clustering to analyse startup outcomes.

# Startup Success Prediction — ML Project

**PSL M1 — March 2026**  
Authors: Ognjen, Yuzhen

## Overview

This project applies supervised and unsupervised machine learning to predict startup outcomes (IPO, Acquisition, Failure) using a 100 000-row Kaggle dataset.

## Structure

```
scientificProject1/
├── task.ipynb               # Main notebook (Part 1: Supervised, Part 2: Unsupervised)
├── data/
│   └── startup_success_dataset.csv
├── report/
│   ├── startup_success_report.docx   # Full written report
│   ├── model_comparison.png
│   ├── roc_3class.png
│   ├── roc_binary.png
│   ├── binary_model_comparison.png
│   ├── cluster_profiles.png
│   └── kmeans_vs_hc.png
└── requirements.txt
```

## Methods

### Part 1 — Supervised Learning
- Preprocessing: `Pipeline` + `ColumnTransformer` (log1p + StandardScaler for numerical, OrdinalEncoder for categorical)
- Models: k-NN, Decision Tree, Logistic Regression, LDA
- Tuning: `GridSearchCV` with 5-fold `StratifiedKFold`
- Binary classification: IPO rows removed → Failure vs Acquisition (macro F1: 0.584 → 0.737)

### Part 2 — Unsupervised Learning
- Dimensionality reduction: PCA (2 components, 39.65% variance)
- Clustering: K-Means (k=3) on full 100k rows
- Validation: Agglomerative / Ward hierarchical clustering on 5 000-row subsample
- Interpretation: cluster profiles by mean feature values

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Open and run `task.ipynb` top to bottom. The notebook is self-contained — all figures are generated inline.
