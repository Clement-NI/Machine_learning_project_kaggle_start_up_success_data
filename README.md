# Machine Learning Project - Startup Success Prediction

Projet final de Machine Learning avec Ognjen et Yuzhen sur le dataset Kaggle Startup Success.

---

## Dataset

**File:** `startup_success_dataset.csv` — 100,000 rows, 11 columns

| Column | Type | Description |
|---|---|---|
| `funding_rounds` | Numerical | Number of funding rounds |
| `founder_experience_years` | Numerical | Years of founder experience |
| `team_size` | Numerical | Size of the team |
| `market_size_billion` | Numerical | Market size in billions |
| `product_traction_users` | Numerical | Number of users |
| `burn_rate_million` | Numerical | Monthly burn rate in millions |
| `revenue_million` | Numerical | Revenue in millions |
| `investor_type` | Categorical | Type of investor (tier2_vc, none, ...) |
| `sector` | Categorical | Industry sector (Health, Fintech, SaaS, Ecommerce, ...) |
| `founder_background` | Categorical | Founder profile (academic, first_time, ex_bigtech, ...) |
| `outcome` | Target | IPO / Acquisition / Failure |

---

## Project Execution Plan

### Step 0 — Setup & Imports
- Import all libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- Load the dataset with `pandas`
- Set global `random_state = 42` for reproducibility

---

### Step 1 — Exploratory Data Analysis (EDA)
- Check shape, data types, missing values
- Compute descriptive statistics (`mean`, `std`, `min`, `max`)
- Visualize distributions of numerical features (histograms)
- Visualize class balance of `outcome` (bar chart)
- Visualize correlations between numerical features (heatmap)
- Analyze categorical features (`investor_type`, `sector`, `founder_background`) with value counts

---

### Step 2 — Preprocessing (using `Pipeline` + `ColumnTransformer`)
- **Numerical features:**
  - Check for skewed distributions → apply log transform if needed
  - Standardize (center + normalize variance) using `StandardScaler`
- **Categorical features:**
  - Encode using `OrdinalEncoder` or one-hot encoding
- **Missing values:**
  - If present: use `SimpleImputer` (mean/median) or `IterativeImputer` (MICE) as in TD10
- **Train/Test split:** 80/20 with `train_test_split(random_state=42)`
- Wrap all steps in a `Pipeline` per model

---

### Step 3 — Supervised Learning: Classification

Goal: Predict `outcome` (IPO / Acquisition / Failure)

#### 3.1 — Model Comparison (as in TD04)
Train and compare the following models using 5-fold cross-validation:
- Logistic Regression
- Linear Discriminant Analysis (LDA)
- k-Nearest Neighbors (k-NN) with hyperparameter tuning on `k`
- Decision Tree Classifier with tuning on `max_depth`

#### 3.2 — Hyperparameter Tuning (as in TD02, TD07)
- Use `GridSearchCV` with 5-fold cross-validation
- Define `param_grid` for each model
- Select best model based on cross-validation accuracy
- Use `n_jobs=-1` for parallel processing

#### 3.3 — Final Evaluation on Test Set (as in TD04)
For the best model(s):
- **Confusion Matrix**
- **Classification Report**: precision, recall, F1-score per class
- **ROC Curve + AUC** (one-vs-rest for multiclass)
- Visualize feature importances (for Decision Tree)

---

### Step 4 — Unsupervised Learning: Clustering (as in TD06)

Goal: Discover natural groupings of startups without using `outcome`

#### 4.1 — Preprocessing for Clustering
- Use only numerical features
- Apply log transforms on skewed features (revenue, traction users)
- Standardize with `StandardScaler`
- Apply **PCA** to reduce dimensions for visualization (2D)

#### 4.2 — K-Means Clustering
- Run k-Means for k = 2 to 10
- Plot **elbow curve** (inertia vs k) to select optimal k
- Visualize clusters in 2D PCA space

#### 4.3 — Hierarchical Clustering (Agglomerative)
- Apply `AgglomerativeClustering`
- Plot **dendrogram** to identify natural cut levels
- Compare cluster assignments with k-Means results

#### 4.4 — Cluster Interpretation
- Compute mean feature values per cluster
- Compare cluster compositions with `outcome` labels (for insight, not training)
- Visualize cluster profiles with bar charts

---

### Step 5 — Validation & Evaluation Summary

| Method | Metric |
|---|---|
| Supervised (classification) | Accuracy, Precision, Recall, F1, ROC-AUC |
| Supervised (cross-validation) | Mean ± Std of CV scores |
| Unsupervised (k-Means) | Inertia, Elbow curve |
| Unsupervised (Hierarchical) | Dendrogram structure |
| Unsupervised (cluster quality) | Visual inspection in PCA 2D space |

---

### Step 6 — Conclusion
- Compare all supervised models
- Identify the best performing model and justify the choice
- Discuss what the unsupervised clusters reveal about startup profiles
- Highlight key features that drive startup success

---

## Project Structure

```
Machine_learning_project_kaggle_start_up_success_data/
├── README.md                        # This file
├── startup_success_dataset.csv      # Raw dataset
└── scientificProject1/
    ├── task.ipynb                   # Main project notebook
    ├── requirements.txt             # Dependencies
    ├── data/                        # Processed data (if any)
    └── code_exemple/                # Reference notebooks (TD corrections)
        ├── TD_01 — Polynomial Regression
        ├── TD_02 — k-NN + GridSearchCV
        ├── TD_03 — Text & Categorical Data
        ├── TD_04 — Multi-model Classification + ROC
        ├── TD_06 — Clustering (k-Means + Hierarchical)
        ├── TD_07 — Decision Trees + Pipelines
        ├── TD_09 — SGD Optimization
        └── TD_10 — Missing Data Imputation
```

---

## Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
```
