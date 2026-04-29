import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.markdown("""
    <style>
    /* Change background color */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Make cards for the input sections */
    div.stNumberInput, div.stSelectbox, div.stSlider {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    /* Style the buttons to look more 'Actionable' */
    .stButton>button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical AI Suite", page_icon="🩺", layout="wide")

# --- CSS FOR STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- SHARED PREDICTION FUNCTION ---
def predict_risk(model_name, features):
    try:
        # 1. Get the directory where main.py is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Go up one level to the root, then into 'models'
        model_path = os.path.join(base_dir, "..", "models", f"{model_name}_model.pkl")
        scaler_path = os.path.join(base_dir, "..", "models", f"{model_name}_scaler.pkl")
        
        # Load the files using the absolute paths
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_array = np.array(features).reshape(1, -1)
        
        # --- HEART SPECIFIC FIX ---
        if model_name == "heart":
            # Your heart scaler only expects the first 5 numerical columns
            numerical_part = input_array[:, :5]
            scaled_numerical = scaler.transform(numerical_part)
            
            # Combine scaled numbers back with the unscaled categorical dummies
            categorical_part = input_array[:, 5:]
            final_input = np.hstack([scaled_numerical, categorical_part])
        else:
            # Kidney, Diabetes, and Stroke scalers were fit on ALL features
            final_input = scaler.transform(input_array)
        
        prediction = model.predict(final_input)[0]
        
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(final_input)[:, 1][0]
        else:
            probability = None
            
        return prediction, probability
    
    except FileNotFoundError:
        st.error(f"File Not Found: Looked in {model_path}. Check if the file exists there!")
        return None, None
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None
    
# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Diagnostic Menu")
selection = st.sidebar.radio("Select Analysis:", ["Overview", "Kidney Disease", "Heart Disease", "Diabetes", "Stroke"])

if selection == "Overview":
    st.title("🏥 Medical AI Diagnostic Suite")
    st.write("Select a category from the sidebar to begin. These models are optimized for clinical accuracy.")

# --- KIDNEY DISEASE (24 Features) ---
elif selection == "Kidney Disease":
    st.title("🧪 Kidney Disease Analysis")
    with st.container(border=True):
        st.subheader("🩸 Vital Signs")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 100, 48)
            bp = st.number_input("Blood Pressure", 50, 180, 80)
            sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
            al = st.slider("Albumin", 0, 5, 1)
        with col2:
            hemo = st.number_input("Hemoglobin", 3.0, 18.0, 15.0)
            pcv = st.number_input("Packed Cell Volume", 10, 55, 44)
            htn = st.selectbox("Hypertension", ["No", "Yes"])
            dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])

    if st.button("Predict Kidney Health"):
        features = [0] * 24
        features[0], features[1], features[2], features[3] = age, bp, sg, al
        features[14], features[15] = hemo, pcv
        features[18] = 1 if htn == "Yes" else 0
        features[19] = 1 if dm == "Yes" else 0
        
        res, prob = predict_risk("kidney", features)
        if res == 1: 
            # High Risk Case
            st.error(f"Result: CKD Detected (Risk Score: {prob:.2%})")
            st.progress(prob) # Bar fills up more as risk increases
        else: 
            # Healthy Case
            st.success(f"Result: Healthy (Confidence: {1-prob:.2%})")
            # show (1-prob) to show how 'full' their health is:
            st.progress(1 - prob)

# --- HEART DISEASE (22 Features - drop_first=True) ---
elif selection == "Heart Disease":
    st.title("❤ Heart Disease Assessment")
    with st.container(border=True):
        st.subheader("🩸 Vital Signs")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 1, 100, 50)
            trestbps = st.number_input("Resting BP", 80, 200, 120)
            chol = st.number_input("Cholesterol", 100, 500, 200)
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
        with col2:
            sex = st.selectbox("Sex", ["Female", "Male"])
            cp = st.selectbox("Chest Pain Type", ["Type 0", "Type 1", "Type 2", "Type 3"])
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])

    if st.button("Predict Heart Health"):
        # Initializing the 22 features required by the model
        features = [0] * 22
        
        # The 5 numerical columns (Indices 0-4) - THESE GET SCALED
        features[0] = age
        features[1] = trestbps
        features[2] = chol
        features[3] = thalach
        features[4] = oldpeak
        
        # The categorical dummies (Indices 5-21) - THESE STAY AS 0 or 1
        features[5] = 1 if sex == "Male" else 0 # sex_1
        
        # CP mapping (cp_1, cp_2, cp_3)
        if "1" in cp: features[6] = 1
        elif "2" in cp: features[7] = 1
        elif "3" in cp: features[8] = 1
        
        features[9] = 1 if fbs == "Yes" else 0 # fbs_1
        features[12] = 1 if exang == "Yes" else 0 # exang_1
        
        res, prob = predict_risk("heart", features)
        if res is not None:
            if res == 1: 
                # High Risk Case
                st.error(f"Result: Heart Disease Detected (Risk Score: {prob:.2%})")
                st.progress(prob) # Bar fills up more as risk increases
            else: 
                # Healthy Case
                st.success(f"Result: Healthy (Confidence: {1-prob:.2%})")
                # show (1-prob) to show how 'full' their health is:
                st.progress(1 - prob)

# --- DIABETES (8 Features) ---
elif selection == "Diabetes":
    st.title("🩸 Diabetes Prediction")
    inputs = [st.number_input(l, 0.0) for l in ["Pregnancies", "Glucose", "BP", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]]
    
    if st.button("Predict Diabetes"):
        res, prob = predict_risk("diabetes", inputs)
        if res == 1: 
            # High Risk Case
            st.error(f"Result: Diabetes Detected (Risk Score: {prob:.2%})")
            st.progress(prob) # Bar fills up more as risk increases
        else: 
            # Healthy Case
            st.success(f"Result: Healthy (Confidence: {1-prob:.2%})")
            # show (1-prob) to show how 'full' their health is:
            st.progress(1 - prob)

# --- STROKE (17 Features) ---
elif selection == "Stroke":
    st.title("🧠 Stroke Risk Analysis")
    with st.container(border=True):
        st.subheader("🩸 Vital Signs")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 1, 100, 50)
            avg_glucose = st.number_input("Avg Glucose", 50.0, 300.0, 100.0)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
            gender = st.selectbox("Gender", ["Female", "Male"])
            residence = st.selectbox("Residence Type", ["Rural", "Urban"])
            married = st.selectbox("Ever Married?", ["No", "Yes"])
        with col2:
            hyper = st.selectbox("Hypertension", ["No", "Yes"])
            heart = st.selectbox("Heart Disease", ["No", "Yes"])
            work = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            smoke = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    if st.button("Predict Stroke Risk"):
        features = [0] * 17
        
        # 1. Basic Features
        features[0] = 1 if gender == "Male" else 0
        features[1] = age
        features[2] = 1 if hyper == "Yes" else 0
        features[3] = 1 if heart == "Yes" else 0
        features[4] = 1 if married == "Yes" else 0
        features[5] = 1 if residence == "Urban" else 0
        features[6] = avg_glucose
        features[7] = bmi
        
        # 2. Work Type Mapping (Indices 8-12)
        work_map = {
            "Govt_job": 8, 
            "Never_worked": 9, 
            "Private": 10, 
            "Self-employed": 11, 
            "children": 12
        }
        features[work_map[work]] = 1
        
        # 3. Smoking Status Mapping (Indices 13-16)
        smoke_map = {
            "Unknown": 13, 
            "formerly smoked": 14, 
            "never smoked": 15, 
            "smokes": 16
        }
        features[smoke_map[smoke]] = 1

        res, prob = predict_risk("stroke", features)
        
        # Applying your custom 0.32 threshold from the notebook
        if prob >= 0.32: 
            st.error(f"Result: High Stroke Risk (Prob(threshold=0.32): {prob:.2%})")
            st.progress(prob) # Bar fills up more as risk increases
        else: 
            st.success(f"Result: Low Risk (Confidence:{1-prob:.2%})")
            st.progress(1 - prob) # Bar fills up more as risk decreases

# --- SIDEBAR FOOTER & RED DISCLAIMER ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ :red[Critical Disclaimer]")
st.sidebar.markdown("""
:red[**This tool is for educational and demonstration purposes only.**] 

The predictions generated by these AI models are based on historical data and **must not** be used as a substitute for professional medical advice, diagnosis, or treatment. 

*If you are experiencing a medical emergency, please contact your local healthcare provider immediately.*
""")