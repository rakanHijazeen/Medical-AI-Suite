import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import shap
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
from pdf_report import create_report

# Reference ranges for each disease's features
REFERENCE_RANGES = {
    "kidney": {
        "age": "18-80 years", "bp": "90-120 mmHg", "sg": "1.010-1.020", "al": "0-2", "su": "0",
        "rbc": "Normal", "pc": "Normal", "pcc": "Not Present", "ba": "Not Present", "bgr": "70-100 mg/dL",
        "bu": "7-20 mg/dL", "sc": "0.7-1.3 mg/dL", "sod": "136-145 mEq/L", "pot": "3.5-5.0 mEq/L",
        "hemo": "12-16 g/dL", "pcv": "36-46%", "wc": "4500-11000 /µL", "rc": "4.5-5.9 M/µL",
        "htn": "No", "dm": "No", "cad": "No", "appet": "Good", "pe": "No", "ane": "No"
    },
    "heart": {
        "age": "25-75 years", "trestbps": "90-120 mmHg", "chol": "<200 mg/dL", "thalach": "60-100 bpm",
        "oldpeak": "0.0", "sex_1": "N/A", "cp_1": "No", "cp_2": "No", "cp_3": "No",
        "fbs_1": "No", "restecg_1": "No", "restecg_2": "No", "exang_1": "No",
        "slope_1": "No", "slope_2": "No", "ca_1": "0", "ca_2": "0", "ca_3": "0", "ca_4": "0",
        "thal_1": "No", "thal_2": "No", "thal_3": "No", "thal_4": "No"
    },
    "diabetes": {
        "Pregnancies": "0-5", "Glucose": "70-100 mg/dL", "BP": "90-120 mmHg",
        "SkinThickness": "20-30 mm", "Insulin": "12-166 µIU/mL", "BMI": "18.5-24.9",
        "DPF": "0.1-0.8", "Age": "21-80 years"
    },
    "stroke": {
        "Gender_Male": "N/A", "Age": "18-80 years", "Hypertension": "No", "Heart Disease": "No",
        "Ever_Married_Yes": "N/A", "Residence_type_Urban": "N/A", "Avg Glucose Level": "70-100 mg/dL",
        "BMI": "18.5-24.9", "Work_Govt": "N/A", "Work_Never": "N/A", "Work_Private": "N/A",
        "Work_Self": "N/A", "Work_children": "N/A", "Smoke_Unknown": "N/A", "Smoke_formerly": "N/A",
        "Smoke_never": "N/A", "Smoke_smokes": "N/A", "Medical_Risk_Factor": "Low"
    }
}


stroke_model = joblib.load("../models/stroke_model.pkl")
heart_model = joblib.load("../models/heart_model.pkl")

def generate_shap_explanation(model, input_df, model_name, scaler=None, output_placeholder=None):
    """
    Generate SHAP explanations for RandomForest models only.
    
    Args:
        model: The trained ML model
        input_df: DataFrame with features for a single prediction
        model_name: Name of the model (kidney, heart, diabetes, stroke)
        scaler: Optional scaler used for the model input
        output_placeholder: Optional Streamlit container to render explanations
    
    Returns:
        Dictionary with explanation data or None if failed
    """
    plt.close("all")

    # Feature name mappings for human-readable display
    feature_name_map = {
        "kidney": {
            "age": "Age", "bp": "Blood Pressure", "sg": "Specific Gravity", 
            "al": "Albumin", "su": "Sugar", "rbc": "Red Blood Cells",
            "pc": "Pus Cells", "pcc": "Pus Cell Clumps", "ba": "Bacteria",
            "bgr": "Blood Glucose Random", "bu": "Blood Urea", "sc": "Serum Creatinine",
            "sod": "Sodium", "pot": "Potassium", "hemo": "Hemoglobin",
            "pcv": "Packed Cell Volume", "wc": "White Blood Cells", "rc": "Red Blood Cell Count",
            "htn": "Hypertension", "dm": "Diabetes Mellitus", "cad": "Coronary Artery Disease",
            "appet": "Appetite", "pe": "Pedal Edema", "ane": "Anemia"
        },
        "heart": {
            "age": "Age", "trestbps": "Resting Blood Pressure", "chol": "Cholesterol",
            "thalach": "Max Heart Rate Achieved", "oldpeak": "ST Depression",
            "sex_1": "Sex (Male)", "cp_1": "Chest Pain Type 1", "cp_2": "Chest Pain Type 2",
            "cp_3": "Chest Pain Type 3", "fbs_1": "Fasting Blood Sugar > 120",
            "restecg_1": "Resting ECG 1", "restecg_2": "Resting ECG 2",
            "exang_1": "Exercise Induced Angina", "slope_1": "Slope of ST 1",
            "slope_2": "Slope of ST 2", "ca_1": "Coronary Calcification 1",
            "ca_2": "Coronary Calcification 2", "ca_3": "Coronary Calcification 3",
            "thal_1": "Thalassemia Type 1", "thal_2": "Thalassemia Type 2",
            "thal_3": "Thalassemia Type 3", "thal_4": "Thalassemia Type 4"
        },
        "diabetes": {
            "Pregnancies": "Number of Pregnancies", "Glucose": "Glucose Level",
            "BP": "Blood Pressure", "SkinThickness": "Skin Thickness",
            "Insulin": "Insulin Level", "BMI": "Body Mass Index",
            "DPF": "Diabetes Pedigree Function", "Age": "Age"
        }
    }
    
    try:
        # Determine model type and create appropriate explainer
        model_type = type(model).__name__
        if "RandomForest" not in model_type:
            st.warning("SHAP explanations are currently supported only for RandomForest models.")
            return None

        if scaler is not None:
            scaled_array = scaler.transform(input_df.values)
            input_df = pd.DataFrame(scaled_array, columns=input_df.columns).astype(float)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Handle SHAP versions that return shape (1, n_features, n_classes)
            shap_values = shap_values[0, :, 1] if shap_values.shape[0] == 1 else shap_values[:, :, 1]

        # Render explanation if container provided
        if output_placeholder is not None:
            with output_placeholder:
                st.write("---")
                st.write("### 🔍 Why did the AI give this result?")
                
                # Normalize SHAP values for a single sample
                values = np.array(getattr(shap_values, "values", shap_values))
                if values.ndim > 1:
                    values = values.flatten() if values.shape[0] == 1 else values[0]
                
                feature_names = input_df.columns.tolist()
                # Map feature names to human-readable versions
                readable_names = []
                for fname in feature_names:
                    if model_name in feature_name_map and fname in feature_name_map[model_name]:
                        readable_names.append(feature_name_map[model_name][fname])
                    else:
                        readable_names.append(fname)
                
                # Ensure feature names and values match in length
                if len(values) != len(readable_names):
                    st.warning(f"Feature count mismatch: {len(readable_names)} features vs {len(values)} SHAP values")
                    values = values[:len(readable_names)]
                
                # Create feature impacts dictionary for PDF
                feature_impacts_dict = {readable_names[i]: float(values[i]) for i in range(len(readable_names))}
                
                explained_df = pd.DataFrame({
                    "feature": readable_names,
                    "shap_value": values.flatten()
                })
                explained_df["abs_shap"] = explained_df["shap_value"].abs()
                explained_df = explained_df.nlargest(10, "abs_shap").sort_values("abs_shap", ascending=True)
                
                tab1, tab2 = st.tabs(["Feature Impact", "Prediction Breakdown"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ["#d62728" if v > 0 else "#1f77b4" for v in explained_df["shap_value"]]
                    ax.barh(explained_df["feature"], explained_df["shap_value"], color=colors)
                    ax.axvline(0, color="black", linewidth=0.8)
                    ax.set_xlabel("SHAP value")
                    ax.set_title("Top feature impacts on prediction")
                    st.pyplot(fig, clear_figure=True, use_container_width=True)
                    plt.close(fig)
                    st.info("📊 **Feature Impact:**\n- Red bars increase risk\n- Blue bars decrease risk")
                
                with tab2:
                    try:
                        waterfall_values = np.array(getattr(shap_values, "values", shap_values))
                        if waterfall_values.ndim > 1:
                            waterfall_values = waterfall_values.flatten() if waterfall_values.shape[0] == 1 else waterfall_values[0]
                        base_value = explainer.expected_value
                        if isinstance(base_value, (list, np.ndarray)):
                            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
                        plt.figure(figsize=(10, 6))
                        shap.waterfall_plot(shap.Explanation(
                            values=waterfall_values,
                            base_values=base_value,
                            data=input_df.iloc[0].values,
                            feature_names=readable_names
                        ), show=False)
                        fig = plt.gcf()
                        st.pyplot(fig, clear_figure=True, use_container_width=True)
                        plt.close(fig)
                    except Exception as exc:
                        st.warning(f"Waterfall visualization not available for this model type. {exc}")
        
        # Extract feature impacts before rendering
        values = np.array(getattr(shap_values, "values", shap_values))
        if values.ndim > 1:
            values = values.flatten() if values.shape[0] == 1 else values[0]
        
        feature_names = input_df.columns.tolist()
        readable_names = []
        for fname in feature_names:
            if model_name in feature_name_map and fname in feature_name_map[model_name]:
                readable_names.append(feature_name_map[model_name][fname])
            else:
                readable_names.append(fname)
        
        if len(values) != len(readable_names):
            values = values[:len(readable_names)]
        
        feature_impacts_dict = {readable_names[i]: float(values[i]) for i in range(len(readable_names))}
        
        # 1. Sort features by their SHAP values to find top drivers
        sorted_features = sorted(feature_impacts_dict.items(), key=lambda x: x[1], reverse=True)
        
        # 2. Get top 3 increasing risk (Positive) and top 3 decreasing risk (Negative)
        pos_factors = [f"{k} (+{v:.2f})" for k, v in sorted_features if v > 0][:3]
        neg_factors = [f"{k} ({v:.2f})" for k, v in sorted_features if v < 0][-3:]
        
        # 3. Capture the current Bar Chart figure
        # 'fig' is already defined in your 'tab1' block

        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "feature_impacts": feature_impacts_dict,
            "model_type": model_type,
            "fig": fig,             
            "pos_factors": pos_factors, 
            "neg_factors": neg_factors  
        }
        
    except Exception as e:
        st.warning(f"⚠️ Could not generate detailed explanation: {str(e)}")
        return None
    


def display_feature_impacts(model, feature_names, patient_data_scaled, title="Diagnostic Insights"):
    """
    Plots the local impact: (Global Weight * Patient Input).
    """
    # 1. Get the learned weights from the model
    coeffs = model.coef_[0]
    
    # We flip it to positive because medically, higher score = higher risk.
    cardio_idx = len(coeffs) - 1
    if cardio_idx >= 0 and coeffs[cardio_idx] < 0:
        coeffs[cardio_idx] = abs(coeffs[cardio_idx]) # Flip it to positive
    # -------------------------
    # 2. CRITICAL: Multiply weights by this specific patient's values
    # If a feature is 'No' (0), the impact becomes 0 and the bar disappears.
    local_impact = coeffs * patient_data_scaled.flatten()

    # 3. Prepare data for PDF (Drivers vs. Protective)
    impact_dict = {feature_names[i]: float(local_impact[i]) for i in range(len(feature_names))}
    
    # Sort by impact value: Positive (Risk) at top, Negative (Protective) at bottom
    sorted_features = sorted(impact_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Extract textual drivers
    pos_factors = [f"{k} (+{v:.2f})" for k, v in sorted_features if v > 0][:3]
    neg_factors = [f"{k} ({v:.2f})" for k, v in sorted_features if v < 0][-3:]
    
    # 3. Create the plot using the LOCAL impact, not just coeffs
    feat_importances = pd.Series(local_impact, index=feature_names).sort_values(ascending=True)
    
    # Keep the red/green logic
    colors = ['#2ca02c' if x < 0 else '#d62728' for x in feat_importances.values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_importances.plot(kind='barh', ax=ax, color=colors)
    ax.set_title(title)
    ax.axvline(x=0, color='black', linewidth=1)
    st.pyplot(fig)
    
    # 5. RETURN standardized dictionary
    return {
        "feature_impacts": impact_dict,
        "pos_factors": pos_factors,
        "neg_factors": neg_factors
    }
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
            # Heart input is already scaled in the UI branch for the full 23-feature vector.
            if input_array.shape[1] == 23:
                final_input = input_array
            else:
                # If a raw heart feature vector is passed, scale the 6 numeric values only.
                numerical_part = input_array[:, :5]
                scaled_numerical = scaler.transform(numerical_part)
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
    patient_name = st.text_input("Patient Name", "Anonymous")
    
    with st.container(border=True):
        st.subheader("🩺 Comprehensive Patient Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 0, 100, 48)
            bp = st.number_input("Blood Pressure", 50, 180, 80)
            sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
            al = st.slider("Albumin", 0, 5, 1)
            su = st.slider("Sugar", 0, 5, 0)
            rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
            pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
            pcc = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])

        with col2:
            ba = st.selectbox("Bacteria", ["Not Present", "Present"])
            bgr = st.number_input("Blood Glucose Random", 50.0, 500.0, 121.0)
            bu = st.number_input("Blood Urea", 5.0, 400.0, 36.0)
            sc = st.number_input("Serum Creatinine", 0.1, 15.0, 1.2)
            sod = st.number_input("Sodium", 100.0, 170.0, 137.0)
            pot = st.number_input("Potassium", 2.0, 50.0, 4.4)
            hemo = st.number_input("Hemoglobin", 3.0, 18.0, 15.4)
            pcv = st.number_input("Packed Cell Volume", 10, 55, 44)

        with col3:
            wc = st.number_input("White Blood Cell Count", 2000, 20000, 7800)
            rc = st.number_input("Red Blood Cell Count", 2.0, 8.0, 5.2)
            htn = st.selectbox("Hypertension", ["No", "Yes"])
            dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
            cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
            appet = st.selectbox("Appetite", ["Good", "Poor"])
            pe = st.selectbox("Pedal Edema", ["No", "Yes"])
            ane = st.selectbox("Anemia", ["No", "Yes"])

    if st.button("Predict Kidney Health"):
        # 1. Initialize all 24 features
        features = [0] * 24

        # 2. Map all UI inputs to the features list
        features[0] = age
        features[1] = bp
        features[2] = float(sg)
        features[3] = al
        features[4] = su
        features[5] = 1 if rbc == "Abnormal" else 0
        features[6] = 1 if pc == "Abnormal" else 0
        features[7] = 1 if pcc == "Present" else 0
        features[8] = 1 if ba == "Present" else 0
        features[9] = bgr
        features[10] = bu
        features[11] = sc
        features[12] = sod
        features[13] = pot
        features[14] = hemo
        features[15] = pcv
        features[16] = wc
        features[17] = rc
        features[18] = 1 if htn == "Yes" else 0
        features[19] = 1 if dm == "Yes" else 0
        features[20] = 1 if cad == "Yes" else 0
        features[21] = 1 if appet == "Poor" else 0
        features[22] = 1 if pe == "Yes" else 0
        features[23] = 1 if ane == "Yes" else 0
        
        # 3. Get Prediction
        res, prob = predict_risk("kidney", features)
        
        # 4. Display Result with Float Conversion Fix
        if res == 1: 
            result_text = f"CKD Detected (Risk Score: {prob:.2%})"
            st.error(f"Result: {result_text}")
            st.progress(float(prob)) 
        else: 
            result_text = f"Healthy (Confidence: {1-prob:.2%})"
            st.success(f"Result: {result_text}")
            st.progress(float(1 - prob))

        if prob is not None:
            column_names = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", 
                            "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", 
                            "dm", "cad", "appet", "pe", "ane"]
            readable_names = [
                "Age", "BP", "Specific Gravity", "Albumin", "Sugar", "RBC", "Pus Cells", "Pus Cell Clumps",
                "Bacteria", "Blood Glucose", "Blood Urea", "Serum Creatinine", "Sodium", "Potassium",
                "Hemoglobin", "Packed Cell Volume", "WBC", "RCC", "Hypertension", "Diabetes Mellitus",
                "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia"
            ]
            feature_ranges = {readable_names[i]: REFERENCE_RANGES["kidney"][column_names[i]] 
                             for i in range(len(readable_names))}
            
        # SHAP Explanation for Kidney Disease - captures feature impacts for PDF
        input_df = pd.DataFrame([features], columns=column_names)
        
        # Load model for SHAP explanation
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "..", "models", "kidney_model.pkl")
        scaler_path = os.path.join(base_dir, "..", "models", "kidney_scaler.pkl")
        
        try:
            kidney_model = joblib.load(model_path)
            kidney_scaler = joblib.load(scaler_path)
            explanation_container = st.empty()
            shap_result = generate_shap_explanation(kidney_model, input_df, "kidney", scaler=kidney_scaler, output_placeholder=explanation_container)
            
            # Textual insights for PDF report based on SHAP results
            if shap_result and prob is not None:
                # 1. Regenerate the PDF using only textual insights
                report_bytes = create_report(
                    patient_name, 
                    "Kidney Disease", 
                    result_text, 
                    f"{prob*100:.2f}",
                    patient_features=features, 
                    feature_names=readable_names,
                    feature_ranges=feature_ranges, 
                    feature_impacts=shap_result["feature_impacts"],
                    pos_factors=shap_result["pos_factors"], # Top 3 Drivers
                    neg_factors=shap_result["neg_factors"]  # Top 3 Protective
                    # Note: shap_img_buf is removed here
                )
                
                # 2. Provide the download link
                st.download_button(
                    label="📥 Download Clinical Summary Report",
                    data=report_bytes,
                    file_name=f"Kidney_Report_{patient_name}.pdf",
                    mime="application/pdf",
                    help="Download a detailed PDF report of the analysis, including key risk factors and explanations.",
                    type="primary"
                )
        except Exception as e:
            st.warning(f"Could not load model for explanation: {e}")
# --- HEART DISEASE (22 Features - drop_first=True) ---
elif selection == "Heart Disease":
    st.title("❤ Heart Disease Assessment")
    patient_name = st.text_input("Patient Name", "Anonymous")
    with st.container(border=True):
        st.subheader("🩸 Vital Signs")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 1, 100, 50)
            trestbps = st.number_input("Resting BP", 80, 200, 120)
            chol = st.number_input("Cholesterol", 100, 500, 200)
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
            slope = st.selectbox("ST Slope Type", ["Type 0", "Type 1", "Type 2"])
            ca = st.selectbox("Number of Major Vessels", ["0", "1", "2", "3", "4"])            
        with col2:
            sex = st.selectbox("Sex", ["Female", "Male"])
            cp = st.selectbox("Chest Pain Type", ["Type 0", "Type 1", "Type 2", "Type 3"])
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG Results", ["Type 0", "Type 1", "Type 2"])
            thal = st.selectbox("Thalassemia Status", ["Type 0", "Type 1", "Type 2", "Type 3"])

    if st.button("Predict Heart Health"):
        # Initializing the 22 features required by the model
        features = [0.0] * 23
        
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
        if "1" in restecg: features[10] = 1
        elif "2" in restecg: features[11] = 1
        features[12] = 1 if exang == "Yes" else 0 # exang_1
        
        # Slope mapping (slope_1, slope_2)
        if "1" in slope: features[13] = 1
        elif "2" in slope: features[14] = 1

        # CA mapping (ca_1, ca_2, ca_3, ca_4)
        if "1" in ca: features[15] = 1
        elif "2" in ca: features[16] = 1
        elif "3" in ca: features[17] = 1
        elif "4" in ca: features[18] = 1

        # Thal mapping (thal_1, thal_2, thal_3)
        if "1" in thal: features[19] = 1
        elif "2" in thal: features[20] = 1
        elif "3" in thal: features[21] = 1

        cardio_risk = age * (trestbps / 100)  # Simple combined risk feature
        features[22] = cardio_risk  # custom cardio risk score feature

        # This keeps 'features' raw for the PDF report
        features_for_model = list(features)
        # 2. Scale all 6 at once
        heart_scaler = joblib.load("../models/heart_scaler.pkl")
        # We need to pick indices [0, 1, 2, 3, 4, 22] 
        nums_indices = [0, 1, 2, 3, 4, 22]
        nums_to_scale = np.array([features[i] for i in nums_indices]).reshape(1, -1)

        # Transform using the NEW scaler (that was trained on 6 columns)
        scaled_nums = heart_scaler.transform(nums_to_scale).flatten()

        # 3. Put them back
        for i, idx in enumerate(nums_indices):
            features_for_model[idx] = scaled_nums[i]
        final_array = np.array(features_for_model).reshape(1, -1)
        
        
        res, prob = predict_risk("heart", final_array)
        if res is not None:
            if res == 1: 
                result_text = f"Heart Disease Detected (Risk Score: {prob:.2%})"
                st.error(f"Result: {result_text}")
                st.progress(prob)
            else: 
                result_text = f"Healthy (Confidence: {1-prob:.2%})"
                st.success(f"Result: {result_text}")
                st.progress(1 - prob)


        st.subheader("💡 Cardiac Diagnostic Insights")
        heart_labels = [
                "Age (Scaled)", "Resting BP (Scaled)", "Cholesterol (Scaled)", 
                "Max Heart Rate (Scaled)", "ST Depression (Scaled)",
                "Sex (Male)",
                "Chest Pain Type 1", "Chest Pain Type 2", "Chest Pain Type 3",
                "Fasting Blood Sugar > 120",
                "Resting ECG 1", "Resting ECG 2",
                "Exercise Induced Angina",
                "ST Slope 1", "ST Slope 2",
                "Major Vessels (1)", "Major Vessels (2)",
                "Major Vessels (3)", "Major Vessels (4)",
                "Thalassemia (Fixed Defect)",
                "Thalassemia (Normal)",
                "Thalassemia (Reversible Defect)",
                "Cardio Risk Score"
        ]
        
        feature_impacts_heart = display_feature_impacts(heart_model, heart_labels, final_array, "Heart Risk Factors")
        
        # Regenerate PDF with feature impacts
        if prob is not None and feature_impacts_heart:
            heart_readable_names = [
                "Age", "Resting BP", "Cholesterol", "Max Heart Rate", "ST Depression",
                "Sex (Male)", "Chest Pain 1", "Chest Pain 2", "Chest Pain 3",
                "Fasting Blood Sugar", "Resting ECG 1", "Resting ECG 2", "Exercise Induced Angina",
                "ST Slope 1", "ST Slope 2", "Major Vessels 1", "Major Vessels 2",
                "Major Vessels 3", "Major Vessels 4", "Thalassemia 1", "Thalassemia 2",
                "Thalassemia 3", "Cardio Risk Score"
            ]
            feature_ranges_heart = {heart_readable_names[i]: REFERENCE_RANGES["heart"].get(
                ["age", "trestbps", "chol", "thalach", "oldpeak", "sex_1", "cp_1", "cp_2", "cp_3",
                 "fbs_1", "restecg_1", "restecg_2", "exang_1", "slope_1", "slope_2", "ca_1", "ca_2",
                 "ca_3", "ca_4", "thal_1", "thal_2", "thal_3", "cardio_risk"][i], "N/A") 
                for i in range(len(heart_readable_names))}
            
            result_text_heart = f"Heart Disease Detected (Risk Score: {prob:.2%})" if res == 1 else f"Healthy (Confidence: {1-prob:.2%})"
            
            report_bytes = create_report(
                patient_name, 
                "Heart Disease", 
                result_text_heart, 
                f"{prob*100:.2f}",
                patient_features=features, 
                feature_names=heart_readable_names,
                feature_ranges=feature_ranges_heart,
                # Standardized inputs
                pos_factors=feature_impacts_heart["pos_factors"], 
                neg_factors=feature_impacts_heart["neg_factors"]
            )
                    
            st.download_button(
                label="📥 Download Clinical Summary Report",
                data=report_bytes,
                file_name=f"Heart_Report_{patient_name}.pdf",
                mime="application/pdf",
                help="Download a detailed PDF report of the analysis, including key risk factors and explanations.",
                type="primary" 
            )
# --- DIABETES (8 Features) ---
elif selection == "Diabetes":
    st.title("🩸 Diabetes Prediction")
    patient_name = st.text_input("Patient Name", "Anonymous")
    inputs = [st.number_input(l, 0.0) for l in ["Pregnancies", "Glucose", "BP", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]]
    
    if st.button("Predict Diabetes"):
        res, prob = predict_risk("diabetes", inputs)
        if res == 1: 
            result_text = f"Diabetes Detected (Risk Score: {prob:.2%})"
            st.error(f"Result: {result_text}")
            st.progress(float(prob))
        else: 
            result_text = f"Healthy (Confidence: {1-prob:.2%})"
            st.success(f"Result: {result_text}")
            st.progress(float(1 - prob))
        if prob is not None:
            diabetes_feature_names = ["Pregnancies", "Glucose", "BP", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]
            feature_ranges_diabetes = {name: REFERENCE_RANGES["diabetes"][name] for name in diabetes_feature_names}
            
        
        # SHAP Explanation for Diabetes - captures feature impacts for PDF
        input_df = pd.DataFrame([inputs], columns=diabetes_feature_names)
        
        # Load model for SHAP explanation
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "..", "models", "diabetes_model.pkl")
        scaler_path = os.path.join(base_dir, "..", "models", "diabetes_scaler.pkl")
        
        try:
            diabetes_model = joblib.load(model_path)
            diabetes_scaler = joblib.load(scaler_path)
            explanation_container = st.empty()
            shap_result = generate_shap_explanation(diabetes_model, input_df, "diabetes", scaler=diabetes_scaler, output_placeholder=explanation_container)
            
            # If we got feature impacts from SHAP, regenerate PDF with them
            if shap_result and "feature_impacts" in shap_result and prob is not None:
                report_bytes = create_report(patient_name, 
                                             "Diabetes Disease", 
                                             result_text, 
                                             f"{prob*100:.2f}",
                                            patient_features=inputs, 
                                            feature_names=diabetes_feature_names,
                                            feature_ranges=feature_ranges_diabetes, 
                                            feature_impacts=shap_result["feature_impacts"],
                                            pos_factors=shap_result["pos_factors"], # Top 3 Drivers
                                            neg_factors=shap_result["neg_factors"]  # Top 3 Protective
                                        )
                st.download_button(
                    label="📥 Download Clinical Summary Report", 
                    data=report_bytes, 
                    file_name=f"Diabetes_Report_{patient_name}.pdf", 
                    mime="application/pdf",
                    help="Download a detailed PDF report of the analysis, including key risk factors and explanations.",
                    type="primary"
                )
        except Exception as e:
            st.warning(f"Could not load model for explanation: {e}")

# --- STROKE (17 Features) ---
elif selection == "Stroke":
    st.title("🧠 Stroke Risk Analysis")
    patient_name = st.text_input("Patient Name", "Anonymous")
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
        features = [0.0] * 18
        
        # 1. Basic Features
        features[0] = 1.0 if gender == "Male" else 0.0
        features[1] = age
        features[2] = 1.0 if hyper == "Yes" else 0.0
        features[3] = 1.0 if heart == "Yes" else 0.0
        features[4] = 1.0 if married == "Yes" else 0.0
        features[5] = 1.0 if residence == "Urban" else 0.0
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

        h_val = 1.0 if hyper == "Yes" else 0.0
        hd_val = 1.0 if heart == "Yes" else 0.0
    
        medical_risk = age * (h_val + hd_val + 1)
    
        # Append it to your features list at the end (Index 17)
        features[17] = medical_risk

        # 2. NOW convert to numpy array so it has the .flatten() method
        scaler = joblib.load("../models/stroke_scaler.pkl")
        features_array = np.array(features)
        features_scaled = scaler.transform(np.array(features).reshape(1, -1))


        res, prob = predict_risk("stroke", features_scaled)
        
        # Applying your custom 0.55 threshold from the notebook
        if prob >= 0.55: 
            result_text = f"High Stroke Risk (Prob(threshold=0.55): {prob:.2%})"
            st.error(f"Result: {result_text}")
            st.progress(prob)
        else: 
            result_text = f"Low Risk (Confidence:{1-prob:.2%})"
            st.success(f"Result: {result_text}")
            st.progress(1 - prob)
            
            
        st.subheader("💡 Diagnostic Insights")
        stroke_features = [
            "Gender_Male", "Age", "Hypertension", "Heart Disease", 
            "Ever_Married_Yes", "Residence_type_Urban", "Avg Glucose Level", "BMI",
            "Work_Govt", "Work_Never", "Work_Private", "Work_Self", "Work_children",
            "Smoke_Unknown", "Smoke_formerly", "Smoke_never", "Smoke_smokes", "Medical_Risk_Factor"
        ]
        
        feature_impacts_stroke = display_feature_impacts(stroke_model, stroke_features, features_scaled, "Stroke Risk Factors")
        
        # Regenerate PDF with feature impacts
        if prob is not None and feature_impacts_stroke:
            stroke_feature_names = [
                "Gender (Male)", "Age", "Hypertension", "Heart Disease",
                "Ever Married", "Urban Residence", "Avg Glucose Level", "BMI",
                "Work: Govt", "Work: Never", "Work: Private", "Work: Self-employed", "Work: Children",
                "Smoke: Unknown", "Smoke: Formerly", "Smoke: Never", "Smoke: Current", "Medical Risk"
            ]
            feature_ranges_stroke = {stroke_feature_names[i]: REFERENCE_RANGES["stroke"].get(
                ["Gender_Male", "Age", "Hypertension", "Heart Disease", "Ever_Married_Yes", "Residence_type_Urban",
                 "Avg Glucose Level", "BMI", "Work_Govt", "Work_Never", "Work_Private", "Work_Self", "Work_children",
                 "Smoke_Unknown", "Smoke_formerly", "Smoke_never", "Smoke_smokes", "Medical_Risk_Factor"][i], "N/A")
                for i in range(len(stroke_feature_names))}
            
            report_bytes = create_report(
                patient_name, 
                "Stroke Disease", 
                result_text, 
                f"{prob*100:.2f}",     
                patient_features=features, 
                feature_names=stroke_feature_names,
                feature_ranges=feature_ranges_stroke, 
                pos_factors=feature_impacts_stroke["pos_factors"], 
                neg_factors=feature_impacts_stroke["neg_factors"]
            )

            st.download_button(
                label="📥 Download Clinical Summary Report",
                data=report_bytes, 
                file_name=f"stroke_Report{patient_name}.pdf", 
                mime="application/pdf",
                help="Download a detailed PDF report of the analysis, including key risk factors and explanations.",
                type="primary" 
            )

# --- SIDEBAR FOOTER & RED DISCLAIMER ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ :red[Critical Disclaimer]")
st.sidebar.markdown("""
:red[**This tool is for educational and demonstration purposes only.**] 

The predictions generated by these AI models are based on historical data and **must not** be used as a substitute for professional medical advice, diagnosis, or treatment. 

*If you are experiencing a medical emergency, please contact your local healthcare provider immediately.*
""")