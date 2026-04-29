# Medical AI Suite 🩺

An intelligent health risk assessment platform built with Streamlit and machine learning.

---

## 📌 Project Overview

Medical AI Suite is a full-stack health analytics app that combines machine learning with an interactive interface for risk assessment. Users can input clinical features and receive instant, interpretable risk predictions for conditions such as stroke and heart disease.

Key goals:

- Provide fast, data-driven clinical risk estimations
- Support explainable outputs and confidence scoring
- Demonstrate end-to-end model deployment with Streamlit

---

## 🚀 Features

- Multi-condition risk prediction
- Support for multiple ML classifiers
- Confidence scoring for each prediction
- Interactive Streamlit-based frontend
- Data preprocessing with scaling, encoding, and imbalance handling
- Notebook-based training pipeline for reproducibility

---

## 🧠 Machine Learning Pipeline

The project includes the following ML pipeline steps:

1. Data preprocessing
   - Median imputation for missing BMI values
   - Categorical encoding for fields such as `work_type` and `smoking_status`
   - Feature scaling for clinical variables

2. Imbalance handling
   - SMOTE applied to address class imbalance in medical datasets

3. Model training and selection
   - Trials with classifiers such as Random Forest, XGBoost, SVM
   - Stroke model tuned for recall to prioritize case detection

4. Model serialization
   - Saved models and transformers using `joblib`

---

## 🛠 Tech Stack

- Frontend / Backend: `Streamlit`
- Machine learning: `scikit-learn`, `xgboost`, 'SVM', 'RandomForest', `imbalanced-learn`
- Visualization: `matplotlib`, `seaborn`
- Model persistence: `joblib`

---

## 📁 Repository Structure

- `app/` – Streamlit application code
- `data/` – sample datasets used for model development
- `models/` – saved model artifacts
- `notebooks/` – Model preprocessing & training and exploration notebooks

---

## ▶️ Installation

```bash
git clone https://github.com/rakanHijazeen/Medical-AI-Suite.git
cd Medical-AI-Suite
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
streamlit run app/main.py
```

> If your entrypoint differs, launch the Streamlit app from the actual app file in `app/`.

---

## 🛡 Disclaimer

This project is for educational purposes only and is not intended to replace professional medical advice or diagnosis.

---

## 👤 Author

- **Rakan Hijazeen**
- GitHub: https://github.com/rakanHijazeen
- LinkedIn: https://www.linkedin.com/in/rakan-hijazeen-327647392/
