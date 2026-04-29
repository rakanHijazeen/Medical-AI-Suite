Medical AI Suite 🩺
An Intelligent Health Risk Assessment Platform

📌 Project Overview
Medical AI Suite is a full-stack web application designed to bridge the gap between machine learning and clinical accessibility. By utilizing advanced classification models, the platform allows users to input health metrics and receive instant, data-driven risk assessments for conditions like Stroke and Cardiovascular diseases.

The project emphasizes Explainable AI (XAI) by providing confidence scores and visualizing risk factors, moving beyond simple "Yes/No" predictions.

🚀 Key Features
Predictive Diagnostics: Multi-model support (Random Forest, XGBoost, SVM) for disease risk classification.

Data-Driven Insights: Integrated confidence scoring for every prediction.

Dynamic UI: Responsive, clean interface built for both healthcare providers and patient self-assessment.

Scientific Rigor: Data preprocessing pipeline includes handling class imbalances (SMOTE) and feature scaling for clinical accuracy.

🛠 Tech Stack
Backend & Frontend: streamlit

Machine Learning: Scikit-learn, XGBoost, SVM, RandomForest, matplotlib, seaborn, SMOTE

Deployment & Tools: Joblib (used for saving and loading pre-trained weights and scalers).

📊 Machine Learning Pipeline
To ensure the model is reliable for medical data, the following pipeline was implemented:

Data Preprocessing: Handling missing BMI values via median imputation, Cleaning Strings, binary mapping .

Feature Engineering: One-Hot Encoding for categorical variables like work_type and smoking_status,Label encoding for numerical values, standard scaling.

Handling Imbalance: Applied SMOTE (Synthetic Minority Over-sampling Technique) to address the 95:5 class imbalance typical in medical datasets.

Model Selection: Evaluated several classifiers, selecting SVM(for stroke model) with a custom threshold of 0.32 to maximize Recall= 54% (catching potential cases) without overwhelming users with False Positives.

Installation & Setup
1.Clone the repo
Bash:
git clone https://github.com/rakanHijazeen/Medical-AI-Suite.git
cd Medical-AI-Suite

    2.Install requirements
    Bash
        pip install -r requirements.txt

    3.Launch the app
    Bash
        streamlit run app.py

🛡 Disclaimer
This suite is an educational project and should not be used as a substitute for professional medical diagnosis. The models are trained on public datasets and intended to showcase the application of ML in healthcare.

👤 Author
Rakan Hijazeen
GitHub : https://github.com/rakanHijazeen
LinkedIn : https://www.linkedin.com/in/rakan-hijazeen-327647392/
