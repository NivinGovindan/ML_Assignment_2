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

ML_Assignment_2/
├── app.py # Streamlit application (deployment entry point)
├── train_models.py # Training script (generates model/ artifacts)
├── requirements.txt # Pip dependencies
├── requirements-lock.txt # Exact pinned versions
├── README.md
├── test_data.csv # UCI WDBC raw data (headerless)
├── .gitignore
└── model/
├── train_models.py # Source code for training
├── scaler.joblib # StandardScaler artifact
├── feature_names.joblib # 30 feature names
├── test_split.joblib # X_test, y_test
├── metrics.csv # Precomputed metrics
├── logistic_regression.joblib
├── decision_tree.joblib
├── knn.joblib
├── naive_bayes.joblib
└── svm.joblib


---

## d. Models Used — Comparison Table

All five models were trained on the same stratified 80/20 split
(`random_state=42`), with identical preprocessing. Metrics are computed on
the **114-sample held-out test set**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | **0.9623** |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **KNN (k=5)** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes (Gaussian)** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Support Vector Machine** | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | **0.9623** |

### Metric definitions (as required by the assignment)

- **Accuracy** — fraction of correctly classified test samples.
- **AUC** — area under the ROC curve; measures ranking quality across all thresholds.
- **Precision** — `TP / (TP + FP)`; how often a positive prediction is correct.
- **Recall** — `TP / (TP + FN)`; how many true positives are captured.
- **F1** — harmonic mean of Precision and Recall.
- **MCC** — Matthews Correlation Coefficient; a balanced single score (−1 to +1)
  that accounts for all four confusion-matrix cells. Preferred over F1 for
  imbalanced datasets.

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | **Best performer (tied with SVM).** Achieved Accuracy 0.9825, AUC 0.9954, and MCC 0.9623 — near-perfect on the test set with only **1 FP and 1 FN**. The near-linear separability of the WDBC features after standardization allows a simple linear decision boundary to perform extremely well. Fast to train, interpretable coefficients. |
| **Decision Tree** | **Weakest model.** Accuracy 0.9123, AUC 0.9157, MCC 0.8174. A single unpruned tree overfits the training data — it memorizes noise and generalizes poorly. Its lower AUC confirms it ranks the classes less reliably than every other model. Motivating case for **ensemble methods** (Random Forest, Gradient Boosting). |
| **K-Nearest Neighbors (k=5)** | **Solid mid-tier.** Accuracy 0.9561, MCC 0.9054, AUC 0.9788. Distance-based approach struggles slightly because WDBC contains 30 **highly correlated** features — Euclidean distance treats all dimensions as equally informative. Feature selection or PCA would likely close the gap to LR/SVM. |
| **Naive Bayes (Gaussian)** | **Interesting behaviour.** AUC is high (0.9868) but Accuracy is comparatively low (0.9298) and MCC is 0.8492. This is the classic Naive Bayes pattern — **excellent probability ranking** even when the 0/1 thresholded predictions are suboptimal. The conditional-independence assumption is violated (features are highly correlated), but the model is still a strong ranker. Calibration would raise accuracy. |
| **Support Vector Machine (RBF)** | **Best performer (tied with LR).** Accuracy 0.9825, AUC 0.9950, MCC 0.9623 — identical confusion matrix to LR (1 FP, 1 FN). The RBF kernel is barely needed; the data is nearly linearly separable in the 30-D standardized space. The margin-maximizing objective yields a decision boundary essentially identical to LR’s. |

### Key insights across the comparison

1. **Linear models dominate.** LR and SVM tie on every metric — a strong signal
   that WDBC is nearly **linearly separable** once features are standardized.
2. **Single trees underperform** on this dataset; a single decision tree cannot
   exploit the smooth, correlated feature geometry.
3. **AUC and Accuracy diverge for Naive Bayes** — a textbook illustration that
   a model can be a great ranker but a mediocre hard classifier.
4. **MCC is a more honest score** than Accuracy for this mildly imbalanced
   dataset; the ranking by MCC matches the ranking by Accuracy, but the *gaps*
   are more informative.
5. **All 5 models are strong** (Accuracy ≥ 0.91, MCC ≥ 0.81). WDBC is a
   well-separated benchmark — this is expected.

---

## f. Overall Winner for This Dataset

> ### 🥇 **Logistic Regression (tied with Support Vector Machine)**

**Justification:**

Both **Logistic Regression** and **Support Vector Machine** achieve the **highest
scores on every one of the six metrics**:

| Metric | LR / SVM | Next best (KNN) |
|---|---:|---:|
| Accuracy | **0.9825** | 0.9561 |
| AUC | **0.9954 / 0.9950** | 0.9788 |
| Precision | **0.9861** | 0.9589 |
| Recall | **0.9861** | 0.9722 |
| F1 | **0.9861** | 0.9655 |
| MCC | **0.9623** | 0.9054 |

The two models produce **identical confusion matrices** (41 TN, 1 FP, 1 FN, 71 TP)
and differ by only 0.0004 in AUC — a tie for practical purposes.

### Why Logistic Regression is the recommended pick

- **Ties SVM** on Accuracy, Precision, Recall, F1, and MCC.
- Has the **highest AUC** (0.9954 vs 0.9950) — a negligible but technically winning margin.
- **Interpretable** — coefficients map directly to feature importance (useful in a medical context).
- **Fastest** to train and deploy — no kernel matrix, no probability calibration tricks.
- **Deterministic** at a fixed seed; SVM with RBF kernel has slightly more hyperparameters (`C`, `gamma`).

**If strict tie-breaking is required, Logistic Regression is the winner.**
SVM is an equally valid answer and should be listed as joint-winner.

---

## Reproducing these results

```bash
# 1. Clone
git clone https://github.com/NivinGovindan/ML_Assignment_2.git
cd ML_Assignment_2

# 2. Environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) retrain & regenerate model artifacts
python train_models.py

# 4. Launch the Streamlit app
streamlit run app.py