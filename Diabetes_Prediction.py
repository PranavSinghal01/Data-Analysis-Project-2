import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner cards and banner styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #0f4c81 0%, #1b76bd 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load, Clean & Train Model (Cached for fast performance)
@st.cache_resource
def load_and_train():
    file_path = os.path.join(os.path.dirname(__file__), "Diabetes_Prediction_Dataset.csv")
    data = pd.read_csv(file_path)
    
    # Drop ID and duplicates
    data = data.drop(columns=['Patient_ID'], errors='ignore').drop_duplicates()
    
    # Clean numeric columns & fix '999' Blood Pressure errors
    num_cols = ['Age', 'BMI', 'Glucose', 'Blood_Pressure', 'Cholesterol', 'Insulin', 'HbA1c']
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['Blood_Pressure'] = data['Blood_Pressure'].replace(999, np.nan)
    
    # Clean target column
    data['Diabetes'] = data['Diabetes'].astype(str).str.strip().map({'Yes': 1, 'No': 0})
    data = data.dropna(subset=['Diabetes'])
    
    # Separate features and target
    X = pd.get_dummies(data.drop(columns=['Diabetes']), drop_first=True)
    y = data['Diabetes']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_imp, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test_imp)) * 100
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return model, imputer, X.columns, accuracy, importances, len(data)

try:
    model, imputer, feature_columns, model_accuracy, feature_importances, total_records = load_and_train()
except FileNotFoundError:
    st.error("Error: 'Diabetes_Prediction_Dataset.csv' not found. Ensure it is uploaded to the same folder on GitHub.")
    st.stop()

# 3. Sidebar Info
with st.sidebar:
    st.markdown(
        """
        <div style="background-color:#e6f2ff; width:72px; height:72px; border-radius:16px;
                    display:flex; align-items:center; justify-content:center;
                    margin-bottom:12px; border: 1px solid #b3d7ff;">
            <svg xmlns="http://w3.org" width="40" height="40" viewBox="0 0 24 24"
                 fill="none" stroke="#1b76bd" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
                <path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/>
                <circle cx="20" cy="10" r="2"/>
            </svg>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.title("Clinical AI Assistant")
    st.markdown("This tool uses a **Random Forest Classifier** trained on patient health records to assess diabetes risk.")
    st.divider()
    st.metric("Model Test Accuracy", f"{model_accuracy:.2f}%")
    st.metric("Training Dataset Size", f"{total_records:,} Patients")
    st.divider()
    st.caption("⚠️ **Disclaimer:** Built for educational and portfolio demonstration purposes only. Not a substitute for professional medical diagnosis.")


# 4. Hero Header
st.markdown("""
    <div class="main-header">
        <h1>🩺 Diabetes Risk Prediction System</h1>
        <p>Complete the patient's clinical, laboratory, and lifestyle profile below to generate an instant risk assessment.</p>
    </div>
""", unsafe_allow_html=True)

# 5. Main Navigation Tabs
tab_predict, tab_insights = st.tabs(["🔍 Patient Risk Assessment", "📊 Model & Feature Insights"])

with tab_predict:
    # Three-column card layout for clean grouping
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown("#### 👤 Demographics & Vitals")
            age = st.slider("Age (years)", min_value=1, max_value=100, value=45)
            gender = st.selectbox("Gender", ["Female", "Male", "Unknown"])
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=28.5, step=0.1)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=250, value=120)

    with col2:
        with st.container(border=True):
            st.markdown("#### 🧪 Lab Results")
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=40, max_value=400, value=140)
            hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=18.0, value=6.5, step=0.1)
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=500, value=120)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=80, max_value=500, value=200)

    with col3:
        with st.container(border=True):
            st.markdown("#### 🏃 Lifestyle & History")
            smoking = st.radio("Smoking Status", ["Never", "Former", "Current"], horizontal=True)
            activity = st.select_slider("Physical Activity Level", options=["Low", "Moderate", "High"], value="Moderate")
            family_history = st.radio("Family History of Diabetes", ["No", "Yes"], horizontal=True)
            st.write("")  # Spacing alignment

    st.write("")
    predict_btn = st.button("Analyze Diabetes Risk", type="primary", use_container_width=True)

    # 6. Prediction Output Section
    if predict_btn:
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'BMI': bmi,
            'Glucose': glucose,
            'Blood_Pressure': blood_pressure,
            'Cholesterol': cholesterol,
            'Insulin': insulin,
            'HbA1c': hba1c,
            'Smoking_Status': smoking,
            'Physical_Activity': activity,
            'Family_History': family_history
        }])
        
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
        input_imputed = imputer.transform(input_encoded)
        
        prediction = model.predict(input_imputed)[0]
        probability = model.predict_proba(input_imputed)[0][1] * 100

        st.divider()
        st.subheader("📋 Diagnostic Assessment Result")

        res_col1, res_col2 = st.columns([1.4, 1], gap="large")

        with res_col1:
            with st.container(border=True):
                if prediction == 1:
                    st.error("### ⚠️ Positive: Elevated Diabetes Risk")
                    st.write("The model indicates a **high likelihood of diabetes** based on the patient's metabolic and clinical profile.")
                else:
                    st.success("### ✅ Negative: Low Diabetes Risk")
                    st.write("The model indicates a **low likelihood of diabetes** based on the current clinical inputs.")

                st.write("**Risk Probability Score:**")
                st.progress(int(round(probability)))
                st.metric(label="Estimated Probability", value=f"{probability:.1f}%")

        with res_col2:
            with st.container(border=True):
                st.markdown("#### Key Patient Biomarkers")
                m1, m2 = st.columns(2)
                m1.metric("Glucose", f"{glucose} mg/dL", delta="High" if glucose >= 126 else "Normal", delta_color="inverse")
                m2.metric("HbA1c", f"{hba1c}%", delta="Elevated" if hba1c >= 6.5 else "Normal", delta_color="inverse")
                m3, m4 = st.columns(2)
                m3.metric("BMI", f"{bmi}", delta="Overweight" if bmi >= 25 else "Healthy", delta_color="inverse")
                m4.metric("Blood Pressure", f"{blood_pressure} mmHg", delta="High" if blood_pressure >= 130 else "Normal", delta_color="inverse")

with tab_insights:
    st.subheader("Top Factors Driving Model Predictions")
    st.write("This chart shows how strongly each clinical and lifestyle feature influences the Random Forest classifier's decisions.")
    
    top_features = feature_importances.head(10).reset_index()
    top_features.columns = ["Feature", "Importance Score"]
    st.bar_chart(top_features.set_index("Feature"), horizontal=True)
