"""
Configuration module for Medical AI Suite.
Centralized storage for model paths, feature mappings, and reference ranges.
"""

from pathlib import Path

# --- ROOT DIRECTORY ---
ROOT_DIR = Path(__file__).resolve().parents[1]

# --- MODEL PATHS ---
MODEL_PATHS = {
    "kidney": {
        "model": ROOT_DIR / "models" / "kidney_model.pkl",
        "scaler": ROOT_DIR / "models" / "kidney_scaler.pkl",
    },
    "heart": {
        "model": ROOT_DIR / "models" / "heart_model.pkl",
        "scaler": ROOT_DIR / "models" / "heart_scaler.pkl",
    },
    "diabetes": {
        "model": ROOT_DIR / "models" / "diabetes_model.pkl",
        "scaler": ROOT_DIR / "models" / "diabetes_scaler.pkl",
    },
    "stroke": {
        "model": ROOT_DIR / "models" / "stroke_model.pkl",
        "scaler": ROOT_DIR / "models" / "stroke_scaler.pkl",
    },
}

# --- FEATURE NAME MAPPINGS (Human-Readable) ---
FEATURE_NAME_MAP = {
    "kidney": {
        "age": "Age",
        "bp": "Blood Pressure",
        "sg": "Specific Gravity",
        "al": "Albumin",
        "su": "Sugar",
        "rbc": "Red Blood Cells",
        "pc": "Pus Cells",
        "pcc": "Pus Cell Clumps",
        "ba": "Bacteria",
        "bgr": "Blood Glucose Random",
        "bu": "Blood Urea",
        "sc": "Serum Creatinine",
        "sod": "Sodium",
        "pot": "Potassium",
        "hemo": "Hemoglobin",
        "pcv": "Packed Cell Volume",
        "wc": "White Blood Cells",
        "rc": "Red Blood Cell Count",
        "htn": "Hypertension",
        "dm": "Diabetes Mellitus",
        "cad": "Coronary Artery Disease",
        "appet": "Appetite",
        "pe": "Pedal Edema",
        "ane": "Anemia",
    },
    "heart": {
        "age": "Age",
        "trestbps": "Resting Blood Pressure",
        "chol": "Cholesterol",
        "thalach": "Max Heart Rate Achieved",
        "oldpeak": "ST Depression",
        "sex_1": "Sex (Male)",
        "cp_1": "Chest Pain Type 1",
        "cp_2": "Chest Pain Type 2",
        "cp_3": "Chest Pain Type 3",
        "fbs_1": "Fasting Blood Sugar > 120",
        "restecg_1": "Resting ECG 1",
        "restecg_2": "Resting ECG 2",
        "exang_1": "Exercise Induced Angina",
        "slope_1": "Slope of ST 1",
        "slope_2": "Slope of ST 2",
        "ca_1": "Coronary Calcification 1",
        "ca_2": "Coronary Calcification 2",
        "ca_3": "Coronary Calcification 3",
        "ca_4": "Coronary Calcification 4",
        "thal_1": "Thalassemia Type 1",
        "thal_2": "Thalassemia Type 2",
        "thal_3": "Thalassemia Type 3",
        "thal_4": "Thalassemia Type 4",
    },
    "diabetes": {
        "Pregnancies": "Number of Pregnancies",
        "Glucose": "Glucose Level",
        "BP": "Blood Pressure",
        "SkinThickness": "Skin Thickness",
        "Insulin": "Insulin Level",
        "BMI": "Body Mass Index",
        "DPF": "Diabetes Pedigree Function",
        "Age": "Age",
    },
    "stroke": {
        "Gender_Male": "Gender (Male)",
        "Age": "Age",
        "Hypertension": "Hypertension",
        "Heart_Disease": "Heart Disease",
        "Ever_Married_Yes": "Ever Married",
        "Residence_type_Urban": "Urban Residence",
        "Avg_Glucose_Level": "Avg Glucose Level",
        "BMI": "Body Mass Index",
        "Work_Govt": "Work: Govt",
        "Work_Never": "Work: Never",
        "Work_Private": "Work: Private",
        "Work_Self": "Work: Self-employed",
        "Work_children": "Work: Children",
        "Smoke_Unknown": "Smoke: Unknown",
        "Smoke_formerly": "Smoke: Formerly",
        "Smoke_never": "Smoke: Never",
        "Smoke_smokes": "Smoke: Current",
        "Medical_Risk_Factor": "Medical Risk",
    },
}

# --- REFERENCE RANGES FOR CLINICAL CONTEXT ---
REFERENCE_RANGES = {
    "kidney": {
        "age": "18-80 years",
        "bp": "90-120 mmHg",
        "sg": "1.010-1.020",
        "al": "0-2",
        "su": "0",
        "rbc": "Normal",
        "pc": "Normal",
        "pcc": "Not Present",
        "ba": "Not Present",
        "bgr": "70-100 mg/dL",
        "bu": "7-20 mg/dL",
        "sc": "0.7-1.3 mg/dL",
        "sod": "136-145 mEq/L",
        "pot": "3.5-5.0 mEq/L",
        "hemo": "12-16 g/dL",
        "pcv": "36-46%",
        "wc": "4500-11000 /µL",
        "rc": "4.5-5.9 M/µL",
        "htn": "No",
        "dm": "No",
        "cad": "No",
        "appet": "Good",
        "pe": "No",
        "ane": "No",
    },
    "heart": {
        "age": "25-75 years",
        "trestbps": "90-120 mmHg",
        "chol": "<200 mg/dL",
        "thalach": "60-100 bpm",
        "oldpeak": "0.0",
        "sex_1": "N/A",
        "cp_1": "No",
        "cp_2": "No",
        "cp_3": "No",
        "fbs_1": "No",
        "restecg_1": "No",
        "restecg_2": "No",
        "exang_1": "No",
        "slope_1": "No",
        "slope_2": "No",
        "ca_1": "0",
        "ca_2": "0",
        "ca_3": "0",
        "ca_4": "0",
        "thal_1": "No",
        "thal_2": "No",
        "thal_3": "No",
        "thal_4": "No",
    },
    "diabetes": {
        "Pregnancies": "0-5",
        "Glucose": "70-100 mg/dL",
        "BP": "90-120 mmHg",
        "SkinThickness": "20-30 mm",
        "Insulin": "12-166 µIU/mL",
        "BMI": "18.5-24.9",
        "DPF": "0.1-0.8",
        "Age": "21-80 years",
    },
    "stroke": {
        "Gender_Male": "N/A",
        "Age": "18-80 years",
        "Hypertension": "No",
        "Heart_Disease": "No",
        "Ever_Married_Yes": "N/A",
        "Residence_type_Urban": "N/A",
        "Avg_Glucose_Level": "70-100 mg/dL",
        "BMI": "18.5-24.9",
        "Work_Govt": "N/A",
        "Work_Never": "N/A",
        "Work_Private": "N/A",
        "Work_Self": "N/A",
        "Work_children": "N/A",
        "Smoke_Unknown": "N/A",
        "Smoke_formerly": "N/A",
        "Smoke_never": "N/A",
        "Smoke_smokes": "N/A",
        "Medical_Risk_Factor": "Low",
    },
}

# --- FEATURE COLUMN NAMES (For DataFrame operations) ---
FEATURE_COLUMNS = {
    "kidney": [
        "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
        "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
        "htn", "dm", "cad", "appet", "pe", "ane",
    ],
    "heart": [
        "age", "trestbps", "chol", "thalach", "oldpeak",
        "sex_1", "cp_1", "cp_2", "cp_3", "fbs_1",
        "restecg_1", "restecg_2", "exang_1", "slope_1", "slope_2",
        "ca_1", "ca_2", "ca_3", "ca_4",
        "thal_1", "thal_2", "thal_3", "cardio_risk",
    ],
    "diabetes": ["Pregnancies", "Glucose", "BP", "SkinThickness", "Insulin", "BMI", "DPF", "Age"],
    "stroke": [
        "Gender_Male", "Age", "Hypertension", "Heart_Disease",
        "Ever_Married_Yes", "Residence_type_Urban", "Avg_Glucose_Level", "BMI",
        "Work_Govt", "Work_Never", "Work_Private", "Work_Self", "Work_children",
        "Smoke_Unknown", "Smoke_formerly", "Smoke_never", "Smoke_smokes", "Medical_Risk_Factor",
    ],
}

# --- DISEASE INFORMATION ---
DISEASE_CATEGORIES = {
    "kidney": {
        "name": "Kidney Disease",
        "emoji": "🧪",
        "num_features": 24,
    },
    "heart": {
        "name": "Heart Disease",
        "emoji": "❤",
        "num_features": 23,
    },
    "diabetes": {
        "name": "Diabetes",
        "emoji": "🩸",
        "num_features": 8,
    },
    "stroke": {
        "name": "Stroke",
        "emoji": "🧠",
        "num_features": 18,
    },
}

# --- PREDICTION THRESHOLDS ---
RISK_THRESHOLDS = {
    "kidney": 0.5,  # Default classification threshold
    "heart": 0.5,
    "diabetes": 0.5,
    "stroke": 0.55,  # Custom threshold from notebook
}
