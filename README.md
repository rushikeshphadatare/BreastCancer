# Breast Cancer Diagnosis Predictor

A Machine Learning web application that predicts whether a breast tumour is **Malignant** or **Benign** based on 30 tumour-cell nucleus measurements from the [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

---

## 🗂️ Project Structure

```
project/
├── app.py               # Streamlit web application
├── train_model.py       # Model training & selection script
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── agent_instructions.md
├── .env.example
├── data/
│   └── breast-cancer.csv   # Source dataset (569 rows × 32 cols)
└── models/
    ├── model.pkl            # Saved best ML pipeline
    └── model_meta.json      # Model metadata (features, classes, scores)
```

---

## ⚙️ Local Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_model.py
```
This will:
- Load `data/breast-cancer.csv`
- Compare 5 classifiers using 5-fold cross-validated ROC-AUC
- Save the best pipeline to `models/model.pkl`
- Save metadata to `models/model_meta.json`

### 5. Launch the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🚀 Streamlit Community Cloud Deployment

1. Push your project to a **public** GitHub repository (include `models/model.pkl` and `models/model_meta.json`).
2. Go to [https://share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your GitHub repo, branch `main`, and set the main file to `app.py`.
4. Click **Deploy** — Streamlit Cloud will install `requirements.txt` automatically.

> **Tip:** Commit the pre-trained `models/model.pkl` so the app works immediately without re-running `train_model.py` on the cloud.

---

## 🧠 ML Pipeline Details

| Step | Detail |
|------|--------|
| Dataset | Wisconsin Breast Cancer (569 samples, 30 features) |
| Target | `diagnosis` — `B` (Benign) / `M` (Malignant) |
| Problem | Binary Classification |
| Preprocessing | `StandardScaler` |
| Models compared | Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN |
| Selection criterion | 5-fold cross-validated ROC-AUC |
| Evaluation | Accuracy, ROC-AUC, Classification Report on 20% held-out test set |

---

## ⚠️ Disclaimer

This application is for **educational and demonstration purposes only**. It is not a medical device and must not be used for actual clinical diagnosis.
