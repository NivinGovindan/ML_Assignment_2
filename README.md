# ML Assignment 2 — Breast Cancer WDBC Classification

**Author:** Nivin K G  
**Repo:** [https://github.com/NivinGovindan/ML_Assignment_2](https://github.com/NivinGovindan/ML_Assignment_2)  
**Live App:** https://ml-assignment-2.streamlit.app

---

## a. Problem Statement

Breast cancer is one of the most common cancers among women worldwide, and early
and accurate diagnosis significantly improves survival outcomes. Distinguishing
a **malignant** tumour from a **benign** one is a classic **binary
classification** problem in medical machine learning.

The objective of this assignment is to build, evaluate, and compare **five
different supervised classification algorithms** on a real-world diagnostic
dataset and determine which model performs best when judged by six standard
evaluation metrics — **Accuracy, AUC, Precision, Recall, F1-Score, and
Matthews Correlation Coefficient (MCC)**.

The five models implemented are:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN) Classifier**
4. **Gaussian Naive Bayes**
5. **Support Vector Machine (SVM, RBF kernel)**

A **Streamlit** web application has been deployed on Streamlit Cloud that
loads the trained models and allows interactive exploration of metrics and
sample-by-sample predictions.

---

## b. Dataset Description

| Attribute | Value |
|---|---|
| **Name** | Breast Cancer Wisconsin (Diagnostic) — **WDBC** |
| **Source** | UCI Machine Learning Repository |
| **URL** | [UCI WDBC](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| **File used** | `test_data.csv` (raw `wdbc.data` renamed, headerless) |
| **Instances** | **569** |
| **Features** | **30** numeric features |
| **Target variable** | `diagnosis` — **`M` (malignant) → 0**, **`B` (benign) → 1** |
| **Class distribution** | 357 benign (62.7%) · 212 malignant (37.3%) |
| **Missing values** | None |

### Feature groups

The 30 features are computed from digitized images of fine-needle aspirate
(FNA) of a breast mass. They are organized in **three groups of ten**:

| Group | Example features |
|---|---|
| **Mean** (10 features) | `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave_points_mean`, `symmetry_mean`, `fractal_dimension_mean` |
| **Standard Error** (10 features) | `radius_se`, `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave_points_se`, `symmetry_se`, `fractal_dimension_se` |
| **Worst / Largest** (10 features) | `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave_points_worst`, `symmetry_worst`, `fractal_dimension_worst` |

### Preprocessing applied

- **Dropped** the `id` column (not a feature).
- **Encoded** target: `M → 0`, `B → 1`.
- **`StandardScaler`** applied inside every pipeline, fit on the training split only.
- **Stratified 80/20 train–test split** (`random_state=42`): 455 train / 114 test.

### Why this dataset

- Comfortably exceeds the assignment’s minimum requirements (**≥ 12 features** ✅,
  **≥ 500 instances** ✅).
- All features are numerical → no encoding artefacts distort model comparison.
- Size is large enough for a stable held-out test split.
- Widely used in the literature → results are directly comparable.

---

## c. GitHub Repository Link

**🔗 https://github.com/NivinGovindan/ML_Assignment_2**

Repository structure:
