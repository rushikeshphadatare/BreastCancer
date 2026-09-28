"""
train_model.py
--------------
Trains multiple classification models on the breast-cancer dataset,
selects the best by cross-validated ROC-AUC, and saves the full
scikit-learn pipeline to models/model.pkl.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import pickle
import pathlib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "breast-cancer.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"
META_PATH  = MODEL_DIR / "model_meta.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load dataset ─────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)

# Drop non-informative ID column if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

# ── Auto-detect target ───────────────────────────────────────────────────────
TARGET_COL = "diagnosis"           # identified from inspection

# Encode target: M → 1 (Malignant), B → 0 (Benign)
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COL])
class_names = le.classes_.tolist()          # ['B', 'M']

feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols].copy()

print(f"Dataset shape : {df.shape}")
print(f"Features      : {len(feature_cols)}")
print(f"Target        : '{TARGET_COL}'  ->  classes {class_names}")
print(f"Class balance : {dict(zip(class_names, np.bincount(y)))}")

# ── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── Model candidates ─────────────────────────────────────────────────────────
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)":           SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n-- Cross-validation (5-fold ROC-AUC) --")
results = {}
for name, clf in candidates.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    results[name] = scores.mean()
    print(f"  {name:<28}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# ── Select best model ────────────────────────────────────────────────────────
best_name = max(results, key=results.get)
print(f"\nBest model: {best_name}  (CV AUC = {results[best_name]:.4f})")

best_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    candidates[best_name]),
])

# ── Final training on full train set ─────────────────────────────────────────
best_pipeline.fit(X_train, y_train)

# ── Evaluation on held-out test set ──────────────────────────────────────────
y_pred       = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n-- Test-set evaluation --")
print(f"  Accuracy : {acc:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── Save pipeline ────────────────────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_pipeline, f)

# Save metadata for app.py
meta = {
    "model_name":   best_name,
    "target_col":   TARGET_COL,
    "feature_cols": feature_cols,
    "class_names":  class_names,   # index 0 → B (Benign), index 1 → M (Malignant)
    "problem_type": "classification",
    "test_accuracy": round(acc, 4),
    "test_auc":      round(auc, 4),
    "cv_results": {k: round(v, 4) for k, v in results.items()},
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nModel saved    -> {MODEL_PATH}")
print(f"Metadata saved -> {META_PATH}")
