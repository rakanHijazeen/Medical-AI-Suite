# Medical AI Suite 🩺

An intelligent health risk assessment platform built with Streamlit and machine learning.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Challenges & Technical Hurdles](#-challenges--technical-hurdles)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

## 📌 Project Overview

Medical AI Suite is a full-stack health analytics app that combines machine learning with an interactive interface for risk assessment. Users can input clinical features and receive instant, interpretable risk predictions for conditions such as kidney disease, heart disease, diabetes, and stroke.

Key goals:

- Provide fast, data-driven clinical risk estimations
- Support explainable outputs and confidence scoring
- Demonstrate end-to-end model deployment with Streamlit

---

## 🚀 Features

- Multi-condition risk prediction (kidney disease, heart disease, diabetes, stroke)
- Support for multiple ML classifiers
- Confidence scoring for each prediction
- Interactive Streamlit-based frontend
- Professional PDF report generation with patient details, vital signs tables, reference ranges, and feature impact analysis
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
   - Trials with classifiers such as Random Forest, XGBoost, SVM, Logistic Regression
   - After testing all of them I selected Random Forest for Kidney & Diabetes models, selected Logistic Regression for Heart & Stroke models.

4. Model serialization
   - Saved models and transformers using `joblib`

---

## 🚀 Challenges & Technical Hurdles

### 1. 🚨 Model Logic Inversion & Scaling Mismatch

**Problem:** Initial testing of the Heart Disease model yielded "100% Healthy" predictions even for high-risk inputs (e.g., 80-year-old with high BP). This was traced to a mismatch between the training vector (23 features) and the application input (5 features).

**🔧 Action:** Re-engineered the data pipeline to synchronize the StandardScaler with a new 6-feature numerical set (including a custom interaction term) and aligned the one-hot encoded categorical indices.

**✅ Result:** Restored model sensitivity, achieving a realistic 97% risk probability for critical cases.

### 2. 🚨 Overfitting & "Perfect" Metric Bias (Kidney Model)

**Problem:** The Chronic Kidney Disease (CKD) model achieved 100% accuracy/recall, indicating structural overfitting on highly correlated biomarkers like Hemoglobin and PCV.

**🔧 Action:** Implemented Gaussian Noise Injection to simulate clinical measurement variance and applied Structural Regularization by capping the Random Forest max_depth and increasing min_samples_leaf.

**✅ Result:** Traded artificial precision for clinical reliability, bringing metrics to a robust and generalizable 90-95%.

### 3. 🚨 Stroke Diagnostic Model Optimization

**Problem:** Initial model (XGBoost) produced too many false negatives (high risk) and excessive "false alarms" (clinical noise), undermining user trust.

**🔧 Action:** Swapped XGBoost for Logistic Regression to improve interpretability and tuned the decision threshold to 0.55 to prioritize high-risk detection.

**✅ Result:** Secured 80% Recall (minimizing missed cases) while slashing false alarms by 14% through improved precision.

### 4. 🚨 Explainable AI (XAI) Synchronization

**Problem:** Standard feature importance plots were static and did not reflect the specific reasoning for individual patient predictions. Furthermore, SHAP explainer states were "freezing" in the Streamlit UI.

**🔧 Action:** Developed a unified interpretability layer using SHAP (Shapley Additive Explanations) for non-linear models and coefficient-mapping for linear models. Fixed Matplotlib state issues using global figure clearing to ensure real-time visual updates.

**✅ Result:** A fully transparent diagnostic suite where every prediction is accompanied by a mathematically sound "local" explanation.

## 🛠 Tech Stack

- Frontend / Backend: `Streamlit`
- Machine learning: `scikit-learn`, `xgboost`, `SVM`, `RandomForest`, `imbalanced-learn`
- Visualization: `matplotlib`, `seaborn`
- PDF Generation: `fpdf2`
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

## ▶️ Usage

1. Run the Streamlit app: `streamlit run app/main.py`
2. Select a disease prediction module (Kidney, Heart, Diabetes, or Stroke).
3. Enter patient details and clinical features.
4. View the risk prediction, confidence score, and feature impact explanations.
5. Download a professional PDF report with patient information, vital signs, reference ranges, and diagnostic insights.

> If your entrypoint differs, launch the Streamlit app from the actual app file in `app/`.

---

## 🛡 Disclaimer

This project is for educational and research purposes only and is not intended to replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions. The predictions provided are based on machine learning models trained on historical data and may not account for all individual factors or the latest medical research.

---

## 👤 Author

- **Rakan Hijazeen**
- GitHub: https://github.com/rakanHijazeen
- LinkedIn: https://www.linkedin.com/in/rakan-hijazeen-327647392/
