# -*- coding: utf-8 -*-
"""
app.py - Breast Cancer Diagnosis Predictor
Streamlit web application that loads the trained ML pipeline and returns
a Malignant / Benign prediction from user-supplied tumour measurements.
"""

import json
import pickle
import pathlib

import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
META_PATH  = BASE_DIR / "models" / "model_meta.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🔬",
    layout="wide",
)

# ── Load model & metadata ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Please run `python train_model.py` first."
        )
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return pipeline, meta

pipeline, meta = load_model()

FEATURE_COLS = meta["feature_cols"]
CLASS_NAMES  = meta["class_names"]   # ['B', 'M']
MODEL_NAME   = meta["model_name"]
TEST_ACC     = meta["test_accuracy"]
TEST_AUC     = meta["test_auc"]

# ── Human-readable feature labels ────────────────────────────────────────────
FEATURE_LABELS = {
    "radius_mean":              "Radius (mean)",
    "texture_mean":             "Texture (mean)",
    "perimeter_mean":           "Perimeter (mean)",
    "area_mean":                "Area (mean)",
    "smoothness_mean":          "Smoothness (mean)",
    "compactness_mean":         "Compactness (mean)",
    "concavity_mean":           "Concavity (mean)",
    "concave points_mean":      "Concave Points (mean)",
    "symmetry_mean":            "Symmetry (mean)",
    "fractal_dimension_mean":   "Fractal Dimension (mean)",
    "radius_se":                "Radius (SE)",
    "texture_se":               "Texture (SE)",
    "perimeter_se":             "Perimeter (SE)",
    "area_se":                  "Area (SE)",
    "smoothness_se":            "Smoothness (SE)",
    "compactness_se":           "Compactness (SE)",
    "concavity_se":             "Concavity (SE)",
    "concave points_se":        "Concave Points (SE)",
    "symmetry_se":              "Symmetry (SE)",
    "fractal_dimension_se":     "Fractal Dimension (SE)",
    "radius_worst":             "Radius (worst)",
    "texture_worst":            "Texture (worst)",
    "perimeter_worst":          "Perimeter (worst)",
    "area_worst":               "Area (worst)",
    "smoothness_worst":         "Smoothness (worst)",
    "compactness_worst":        "Compactness (worst)",
    "concavity_worst":          "Concavity (worst)",
    "concave points_worst":     "Concave Points (worst)",
    "symmetry_worst":           "Symmetry (worst)",
    "fractal_dimension_worst":  "Fractal Dimension (worst)",
}

# Typical value ranges derived from the dataset (min, max, default)
FEATURE_RANGES = {
    "radius_mean":             (6.98,  28.11, 14.13),
    "texture_mean":            (9.71,  39.28, 19.29),
    "perimeter_mean":          (43.79, 188.5, 91.97),
    "area_mean":               (143.5, 2501.0, 654.9),
    "smoothness_mean":         (0.053, 0.163, 0.096),
    "compactness_mean":        (0.019, 0.345, 0.104),
    "concavity_mean":          (0.0,   0.427, 0.089),
    "concave points_mean":     (0.0,   0.201, 0.049),
    "symmetry_mean":           (0.106, 0.304, 0.181),
    "fractal_dimension_mean":  (0.050, 0.097, 0.063),
    "radius_se":               (0.112, 2.873, 0.405),
    "texture_se":              (0.360, 4.885, 1.217),
    "perimeter_se":            (0.757, 21.98, 2.866),
    "area_se":                 (6.802, 542.2, 40.34),
    "smoothness_se":           (0.002, 0.031, 0.007),
    "compactness_se":          (0.002, 0.135, 0.025),
    "concavity_se":            (0.0,   0.396, 0.032),
    "concave points_se":       (0.0,   0.053, 0.012),
    "symmetry_se":             (0.008, 0.079, 0.021),
    "fractal_dimension_se":    (0.001, 0.030, 0.004),
    "radius_worst":            (7.93,  36.04, 16.27),
    "texture_worst":           (12.02, 49.54, 25.68),
    "perimeter_worst":         (50.41, 251.2, 107.3),
    "area_worst":              (185.2, 4254.0, 880.6),
    "smoothness_worst":        (0.071, 0.223, 0.132),
    "compactness_worst":       (0.027, 1.058, 0.254),
    "concavity_worst":         (0.0,   1.252, 0.272),
    "concave points_worst":    (0.0,   0.291, 0.115),
    "symmetry_worst":          (0.156, 0.664, 0.290),
    "fractal_dimension_worst": (0.055, 0.208, 0.084),
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Breast Cancer Diagnosis Predictor")
st.markdown(
    "Enter the tumour measurement values below and click **Predict** to receive "
    "a diagnosis prediction (Malignant / Benign)."
)

# ── Model info sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown(f"**Algorithm:** {MODEL_NAME}")
    st.markdown(f"**Test Accuracy:** {TEST_ACC * 100:.2f}%")
    st.markdown(f"**Test ROC-AUC:** {TEST_AUC:.4f}")
    st.markdown("---")
    st.markdown("**Dataset:** Wisconsin Breast Cancer (569 samples, 30 features)")
    st.markdown("**Problem type:** Binary Classification")
    st.markdown("---")
    st.markdown("**Cross-validation AUC scores:**")
    for m, v in meta["cv_results"].items():
        st.markdown(f"- {m}: `{v:.4f}`")

# ── Feature input form ────────────────────────────────────────────────────────
st.subheader("📋 Tumour Feature Inputs")

with st.form("prediction_form"):
    # Group features into three sections: mean, SE, worst
    groups = {
        "Mean Values": [c for c in FEATURE_COLS if c.endswith("_mean")],
        "Standard Error Values": [c for c in FEATURE_COLS if c.endswith("_se")],
        "Worst Values": [c for c in FEATURE_COLS if c.endswith("_worst")],
    }

    input_values = {}

    for group_name, cols in groups.items():
        st.markdown(f"#### {group_name}")
        n_cols = 3
        rows = [cols[i : i + n_cols] for i in range(0, len(cols), n_cols)]
        for row in rows:
            ui_cols = st.columns(len(row))
            for ui_col, feat in zip(ui_cols, row):
                lo, hi, default = FEATURE_RANGES.get(feat, (0.0, 1.0, 0.5))
                step = round((hi - lo) / 1000, 6)
                input_values[feat] = ui_col.number_input(
                    label=FEATURE_LABELS.get(feat, feat),
                    min_value=float(lo),
                    max_value=float(hi * 1.5),   # allow slightly above max
                    value=float(default),
                    step=float(step) if step > 0 else 0.0001,
                    format="%.5f",
                )

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    input_df = pd.DataFrame([[input_values[c] for c in FEATURE_COLS]],
                             columns=FEATURE_COLS)

    pred_class_idx  = pipeline.predict(input_df)[0]
    pred_proba      = pipeline.predict_proba(input_df)[0]
    pred_label      = CLASS_NAMES[pred_class_idx]   # 'B' or 'M'
    confidence      = pred_proba[pred_class_idx] * 100

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if pred_label == "M":
        st.error(
            f"**Malignant (M)**  —  The tumour is predicted to be **malignant**.\n\n"
            f"Confidence: **{confidence:.1f}%**\n\n"
            f"⚠️ This is a tool for educational purposes only. Consult a medical professional."
        )
    else:
        st.success(
            f"**Benign (B)**  —  The tumour is predicted to be **benign**.\n\n"
            f"Confidence: **{confidence:.1f}%**\n\n"
            f"✅ This is a tool for educational purposes only. Consult a medical professional."
        )

    # Probability bar
    st.markdown("#### Class Probabilities")
    prob_df = pd.DataFrame(
        {"Class": ["Benign (B)", "Malignant (M)"], "Probability": pred_proba}
    )
    st.bar_chart(prob_df.set_index("Class"))

    # Feature summary
    with st.expander("📊 Input summary"):
        display_df = pd.DataFrame(
            {"Feature": FEATURE_COLS,
             "Label":   [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
             "Value":   [input_values[c] for c in FEATURE_COLS]}
        ).set_index("Feature")
        st.dataframe(display_df, use_container_width=True)
