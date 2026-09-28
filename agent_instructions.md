# Agent Instructions

## Project Overview
This project is a Machine Learning web application for breast cancer diagnosis prediction.
It uses the Wisconsin Breast Cancer Dataset (569 samples, 30 features) to classify tumours
as Malignant (M) or Benign (B).

## Dataset
- **File:** `data/breast-cancer.csv`
- **Rows:** 569
- **Features:** 30 numeric tumour-cell nucleus measurements
- **Target column:** `diagnosis` (B = Benign, M = Malignant)
- **No missing values**

## Problem Type
Binary Classification (Malignant vs Benign)

## Training Pipeline (`train_model.py`)
1. Load CSV with pandas, drop the `id` column.
2. Encode target with `LabelEncoder` (B→0, M→1).
3. Split 80/20 stratified train/test.
4. Compare 5 classifiers via 5-fold cross-validated ROC-AUC:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - SVM (RBF kernel)
   - K-Nearest Neighbors
5. Retrain best model on full train set.
6. Evaluate on test set (Accuracy + ROC-AUC + Classification Report).
7. Save pipeline → `models/model.pkl`
8. Save metadata → `models/model_meta.json`

## App (`app.py`)
- Streamlit interface with number inputs for all 30 features.
- Features grouped as: Mean Values / Standard Error Values / Worst Values.
- Loads pipeline from `models/model.pkl` and metadata from `models/model_meta.json`.
- Returns predicted class + confidence + class probability bar chart.

## Environment Variables
No API keys are required for this project. See `.env.example` for reference.

## Deployment Notes
- Python 3.9+ recommended.
- All paths are relative to the project root.
- Commit `models/model.pkl` and `models/model_meta.json` before deploying to Streamlit Cloud.
