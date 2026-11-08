"""
Streamlit app — UCI Breast Cancer WDBC, 5-model comparison.

Run locally:   streamlit run app.py
Deploy:        Streamlit Cloud will run `app.py` from the repo root.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT      = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "test_data.csv"

st.set_page_config(page_title="Breast Cancer Classifier",
                   page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# Guard: artifacts must exist (run train_models.py first)
# ---------------------------------------------------------------------------
if not (MODEL_DIR / "metrics.csv").exists():
    st.error("Model artifacts not found. "
             "Run `python train_models.py` and commit the `model/` folder.")
    st.stop()

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    split  = joblib.load(MODEL_DIR / "test_split.joblib")
    models = {k: joblib.load(MODEL_DIR / f"{k}.joblib")
              for k in metrics["model_key"]}
    return metrics, feature_names, scaler, split, models

@st.cache_data
def load_raw_data():
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
    cols = ["id", "diagnosis"] + feature_names
    df = pd.read_csv(DATA_PATH, header=None, names=cols, na_values="?")
    df["diagnosis"] = df["diagnosis"].map({"M": 0, "B": 1})
    return df.drop(columns=["id"])

metrics_df, FEATURE_NAMES, scaler, split, models = load_artifacts()
raw_df = load_raw_data()
X_test, y_test = split["X_test"], split["y_test"]

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🩺 Breast Cancer WDBC")
st.sidebar.caption("5 classifiers · 569 samples · 30 features")
page = st.sidebar.radio("Navigation",
    ["Overview", "Metrics Comparison", "Predict a Sample", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: UCI Breast Cancer Wisconsin (Diagnostic)")

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Breast Cancer Classification — Model Comparison")
    st.markdown(
        "This app compares **five ML classifiers** trained on the "
        "UCI Breast Cancer Wisconsin (Diagnostic) dataset.\n\n"
        "- **569** samples · **30** numeric features · **2** classes "
        "(*malignant* = 0, *benign* = 1)"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples",  "569")
    c2.metric("Features", "30")
    c3.metric("Models",   "5")
    c4.metric("Best MCC", f"{metrics_df['MCC'].max():.4f}")

    st.subheader("Dataset preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Class distribution")
    counts = raw_df["diagnosis"].map({0: "malignant", 1: "benign"}).value_counts()
    fig, ax = plt.subplots(figsize=(5, 3))
    counts.plot(kind="bar", ax=ax, color=["#d9534f", "#5cb85c"])
    ax.set_ylabel("Count"); ax.set_xlabel("")
    st.pyplot(fig); plt.close(fig)

# ---------------------------------------------------------------------------
# Page: Metrics Comparison
# ---------------------------------------------------------------------------
elif page == "Metrics Comparison":
    st.title("Evaluation Metrics")

    display_cols = ["Model", "Accuracy", "AUC", "Precision",
                    "Recall", "F1", "MCC"]
    st.dataframe(
        metrics_df[display_cols].set_index("Model").round(4),
        use_container_width=True,
    )

    metric = st.selectbox("Bar chart — pick a metric",
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics_df.set_index("Model")[metric].sort_values().plot(
        kind="barh", ax=ax, color="#4a90d9")
    ax.set_xlim(0.7, 1.02); ax.set_xlabel(metric)
    ax.set_title(f"{metric} by model (test set)")
    st.pyplot(fig); plt.close(fig)

    st.subheader("Confusion matrices (test set)")
    cols = st.columns(len(metrics_df))
    for col, (_, row) in zip(cols, metrics_df.iterrows()):
        with col:
            st.caption(row["Model"])
            cm = np.array([[row["TN"], row["FP"]],
                           [row["FN"], row["TP"]]])
            fig, ax = plt.subplots(figsize=(2.6, 2.6))
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black",
                            fontsize=12)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Mal", "Ben"]); ax.set_yticklabels(["Mal", "Ben"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig); plt.close(fig)

# ---------------------------------------------------------------------------
# Page: Predict a Sample
# ---------------------------------------------------------------------------
elif page == "Predict a Sample":
    st.title("Predict a Sample")

    model_display = st.selectbox("Model", metrics_df["Model"].tolist())
    model_key = metrics_df.loc[metrics_df["Model"] == model_display,
                               "model_key"].iloc[0]
    model = models[model_key]

    mode = st.radio("Input method",
        ["Random test-set sample", "Upload CSV (one or more rows)"],
        horizontal=True)

    if mode == "Random test-set sample":
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🎲 Shuffle", type="primary"):
                st.session_state["sample_idx"] = int(
                    np.random.randint(0, len(X_test)))
        idx = st.session_state.get("sample_idx", 0)
        x = X_test[idx].reshape(1, -1)
        y_true = int(y_test[idx])
        st.caption(f"Test sample #{idx} · true label: "
                   f"**{'benign' if y_true == 1 else 'malignant'}**")
        st.dataframe(pd.DataFrame(x, columns=FEATURE_NAMES).T
                     .rename(columns={0: "value"}),
                     use_container_width=True)
    else:
        up = st.file_uploader(
            "CSV with 30 numeric feature columns per row (no header, same "
            "order as `model/feature_names.joblib`)", type="csv")
        if up is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        arr = pd.read_csv(up, header=None).values
        if arr.shape[1] != 30:
            st.error(f"Expected 30 columns, got {arr.shape[1]}.")
            st.stop()
        x = arr

    if st.button("🔍 Predict", type="primary"):
        y_pred  = model.predict(x)
        y_proba = model.predict_proba(x)[:, 1]

        st.subheader(f"Prediction — {model_display}")
        for i in range(len(x)):
            label = "benign" if y_pred[i] == 1 else "malignant"
            st.metric(f"Row {i}", label, f"P(benign) = {y_proba[i]:.4f}")

        st.subheader("All five models on the same input")
        rows = []
        for key, m in models.items():
            display = metrics_df.loc[metrics_df["model_key"] == key,
                                     "Model"].iloc[0]
            preds = m.predict(x)
            probs = m.predict_proba(x)[:, 1]
            rows.append({
                "Model": display,
                "Prediction": ["benign" if p == 1 else "malignant" for p in preds],
                "P(benign)": [f"{p:.4f}" for p in probs],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    st.title("About this project")
    st.markdown("""
    **Task:** Binary classification — *malignant* vs *benign* tumours.

    **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) — 569 samples,
    30 numeric features.

    **Models:** Logistic Regression · Decision Tree · K-Nearest Neighbors (k=5)
    · Gaussian Naive Bayes · Support Vector Machine (RBF, probability=True).

    **Preprocessing:** `StandardScaler` inside each pipeline, fit on the
    training split only.

    **Metrics:** Accuracy, AUC, Precision, Recall, F1, MCC on a stratified
    80/20 hold-out set (`random_state=42`).

    **Source:** [UCI ML Repository — WDBC](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
    """)