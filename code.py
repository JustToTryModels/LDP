import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os

# Set basic Streamlit config
st.set_page_config(page_title="Liver Disease Prediction", layout="wide")

st.markdown("# 🩺 Liver Disease Prediction App")
st.markdown("Upload patient data to predict the presence of **Liver Disease** using a trained **Random Forest model**.")

# ------------------------------------------------------------------------------
# Load model and features from GitHub
# ------------------------------------------------------------------------------

@st.cache_resource
def load_model_and_features():
    model_url = "https://github.com/JustToTryModels/LDP/raw/main/LDP_RFC_Model/final_random_forest_model.joblib"
    features_url = "https://github.com/JustToTryModels/LDP/raw/main/LDP_RFC_Model/selected_features_list.joblib"

    model_filename = "final_random_forest_model.joblib"
    features_filename = "selected_features_list.joblib"

    # Download model file if not exists
    if not os.path.isfile(model_filename):
        with open(model_filename, "wb") as f:
            f.write(requests.get(model_url).content)

    if not os.path.isfile(features_filename):
        with open(features_filename, "wb") as f:
            f.write(requests.get(features_url).content)

    model = joblib.load(model_filename)
    selected_features = joblib.load(features_filename)

    return model, selected_features

model, selected_features = load_model_and_features()

# Sample Input Info
feature_explanations = {
    "Alkphos_Alkaline_Phosphotase": "Alkaline Phosphatase (U/L)",
    "Sgot_Aspartate_Aminotransferase": "SGOT - Aspartate Aminotransferase (U/L)",
    "Sgpt_Alamine_Aminotransferase": "SGPT - Alamine Aminotransferase (U/L)",
    "Total_Bilirubin": "Total Bilirubin (mg/dL)",
    "Total_Proteins": "Total Proteins (g/dL)",
    "Direct_Bilirubin": "Direct Bilirubin (mg/dL)",
    "ALB_Albumin": "Albumin (g/dL)",
    "A/G_Ratio_Albumin_and_Globulin_Ratio": "Albumin/Globulin Ratio"
}

# ------------------------------------------------------------------------------
# Input Method (Form or CSV Upload)
# ------------------------------------------------------------------------------

input_method = st.radio("Select Input Method", ["Manual Entry", "Upload CSV"])

if input_method == "Manual Entry":
    st.markdown("## ✍️ Provide Patient Details:")

    user_input = {}
    for feature in selected_features:
        explanation = feature_explanations.get(feature, feature)
        val = st.number_input(f"{explanation}", format="%.4f", key=feature)
        user_input[feature] = val

    input_df = pd.DataFrame([user_input])

elif input_method == "Upload CSV":
    st.markdown("## 📤 Upload CSV File:")

    uploaded_file = st.file_uploader("Choose a CSV file with patient lab values", type=["csv"])

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            missing = [f for f in selected_features if f not in df_uploaded.columns]
            if len(missing) > 0:
                st.error(f"CSV is missing required columns: {missing}")
                st.stop()
            else:
                input_df = df_uploaded[selected_features]
                st.success(f"✔️ Loaded {len(input_df)} samples from CSV!")
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
            st.stop()
    else:
        st.warning("⚠️ Please upload a CSV file to continue.")
        input_df = None

# ------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------

if input_df is not None and st.button("🔍 Predict"):
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)

    st.markdown("## 🧪 Prediction Results:")

    results = []
    for i in range(len(input_df)):
        pred = predictions[i]
        prob = probabilities[i][pred]
        label = "Liver Disease" if pred == 1 else "Non-Liver Disease"
        color = "🟥" if pred == 1 else "🟩"
        results.append((i + 1, label, f"{prob*100:.2f}%", color))

    results_df = pd.DataFrame(results, columns=["Patient#", "Prediction", "Confidence", "Status"])
    st.dataframe(results_df, use_container_width=True)

    # Summary counts
    liver_count = np.sum(predictions == 1)
    non_liver_count = np.sum(predictions == 0)

    st.markdown(f"""
    ### 📊 Summary:
    - Total Patients: **{len(predictions)}**
    - 🟥 Predicted with Liver Disease: **{liver_count}**
    - 🟩 Predicted Non-Liver Disease: **{non_liver_count}**
    """)

# ------------------------------------------------------------------------------
# Sample CSV Template
# ------------------------------------------------------------------------------

with st.expander("📁 Download Sample CSV Format"):
    st.markdown("This is the correct format for uploading multiple patient records.")
    sample_df = pd.DataFrame(columns=selected_features)
    st.download_button("📥 Download Template CSV", sample_df.to_csv(index=False), "sample_input_template.csv")

st.markdown("---")
st.markdown("*Model trained using Random Forest and feature selection based on MDI.*")
st.markdown("🔗 [GitHub Repository](https://github.com/JustToTryModels/LDP)")
