"""
Train all 5 classifiers on the UCI WDBC dataset and persist artifacts
for the Streamlit app.

Run once:  python train_models.py

Outputs (model/):
    scaler.joblib               preprocessing artifact
    feature_names.joblib
    test_split.joblib           {X_test, y_test}
    metrics.csv                 per-model metrics + confusion matrix cells
    <key>.joblib                one file per classifier
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
)

SCRIPT_DIR = Path(__file__).resolve().parent        # <project>/model
ROOT       = SCRIPT_DIR.parent                      # <project>
DATA_PATH  = ROOT / "test_data.csv"
MODEL_DIR  = SCRIPT_DIR                             # artifacts go here

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"test_data.csv not found at {DATA_PATH}. "
        f"Place it in the project root ({ROOT})."
    )
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]


def load_data():
    columns = ["id", "diagnosis"] + FEATURE_NAMES
    df = pd.read_csv(DATA_PATH, header=None, names=columns, na_values="?")
    df["diagnosis"] = df["diagnosis"].map({"M": 0, "B": 1})
    df = df.drop(columns=["id"])
    return df[FEATURE_NAMES].values, df["diagnosis"].values


def build_models():
    return {
        "logistic_regression": ("Logistic Regression", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])),
        "decision_tree": ("Decision Tree", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=42)),
        ])),
        "knn": ("K-Nearest Neighbors", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ])),
        "naive_bayes": ("Gaussian Naive Bayes", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ])),
        "svm": ("Support Vector Machine", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        probability=True, random_state=42)),
        ])),
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # --- save preprocessing + split artifacts ---
    joblib.dump(StandardScaler().fit(X_train), MODEL_DIR / "scaler.joblib")
    joblib.dump(FEATURE_NAMES,                 MODEL_DIR / "feature_names.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test},
                MODEL_DIR / "test_split.joblib")

    rows = []
    for key, (display, pipe) in build_models().items():
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        cm      = confusion_matrix(y_test, y_pred)
        rows.append({
            "model_key": key, "Model": display,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "AUC":       roc_auc_score(y_test, y_proba),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1":        f1_score(y_test, y_pred),
            "MCC":       matthews_corrcoef(y_test, y_pred),
            "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
        })
        joblib.dump(pipe, MODEL_DIR / f"{key}.joblib")
        print(f"[OK] {display:25s} -> model/{key}.joblib")

    df = pd.DataFrame(rows)
    df.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print("\nSaved metrics -> model/metrics.csv")
    print(df.set_index("Model")[["Accuracy", "AUC", "Precision",
                                 "Recall", "F1", "MCC"]].round(4).to_string())


if __name__ == "__main__":
    main()