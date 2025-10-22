import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io

# ======================================================================================
# Function to Load Model and Features from GitHub
# ======================================================================================
# Use Streamlit's caching to load the model only once, improving performance.
@st.cache_resource
def load_model_from_github():
    """
    Loads the trained RandomForest model and the list of selected features
    directly from a public GitHub repository.
    """
    # URLs to the raw model and features files on GitHub
    model_url = 'https://raw.githubusercontent.com/JustToTryModels/LDP/main/LDP_RFC_Model/final_random_forest_model.joblib'
    features_url = 'https://raw.githubusercontent.com/JustToTryModels/LDP/main/LDP_RFC_Model/selected_features_list.joblib'

    try:
        # Download the model file
        model_response = requests.get(model_url)
        model_response.raise_for_status()  # Raise an exception for bad status codes
        model_file = io.BytesIO(model_response.content)
        model = joblib.load(model_file)

        # Download the features list file
        features_response = requests.get(features_url)
        features_response.raise_for_status()
        features_file = io.BytesIO(features_response.content)
        features = joblib.load(features_file)

        return model, features
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading files from GitHub: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model/features: {e}")
        return None, None

# Load the model and feature list
final_model, best_features = load_model_from_github()

# ======================================================================================
# Streamlit App User Interface
# ======================================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# --- App Title and Description ---
st.title("🩺 Liver Disease Prediction App")
st.markdown("""
This application uses a Random Forest model to predict the likelihood of liver disease based on patient data.
Please enter the patient's test results in the sidebar to get a prediction.

**Model Performance (on test data):**
- **Minority Class Recall (No Disease):** 99.73%
- **Majority Class Recall (Disease):** 99.96%
- **Balanced Accuracy:** 99.84%
""")
st.info("The model and feature list are loaded directly from a public GitHub repository.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Patient Data Input")
st.sidebar.markdown("Enter the values for the following medical tests:")

# Check if model and features loaded successfully before creating input fields
if final_model and best_features:

    # Create a dictionary to hold user inputs
    input_data = {}

    # Define tooltips for better user understanding
    tooltips = {
        'Alkphos_Alkaline_Phosphotase': 'Measures the amount of alkaline phosphatase enzyme in your blood. (IU/L)',
        'Sgot_Aspartate_Aminotransferase': 'Measures the enzyme AST (SGOT) in your blood. (IU/L)',
        'Sgpt_Alamine_Aminotransferase': 'Measures the enzyme ALT (SGPT) in your blood. (IU/L)',
        'Total_Bilirubin': 'Measures the total amount of bilirubin in your blood. (mg/dL)',
        'Total_Proteins': 'Measures the total amount of protein in your blood. (g/dL)',
        'Direct_Bilirubin': 'Measures the amount of direct (conjugated) bilirubin. (mg/dL)',
        'ALB_Albumin': 'Measures the amount of albumin in your blood. (g/dL)',
        'A/G_Ratio_Albumin_and_Globulin_Ratio': 'The ratio of albumin to globulin in the blood.'
    }
    
    # Create input fields for each feature dynamically
    for feature in best_features:
        # Create a more user-friendly label
        label = feature.replace('_', ' ').title()
        
        input_data[feature] = st.sidebar.number_input(
            label=label,
            min_value=0.0,
            max_value=2500.0,  # A high upper bound to accommodate various values
            value=1.0,         # A default value
            step=0.1,
            format="%.1f",
            help=tooltips.get(feature, "Enter the measured value.")
        )
    
    # --- Prediction Button ---
    if st.sidebar.button("Predict Liver Disease Status", type="primary"):
        # 1. Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])
        # Ensure the column order matches the model's training order
        input_df = input_df[best_features]

        # 2. Make Prediction
        prediction = final_model.predict(input_df)
        prediction_proba = final_model.predict_proba(input_df)
        
        # 3. Display Results
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)

        with col1:
            if prediction[0] == 1:
                st.error("Prediction: **LIVER DISEASE DETECTED**", icon="⚠️")
            else:
                st.success("Prediction: **NO LIVER DISEASE DETECTED**", icon="✅")
        
        with col2:
            st.info(f"**Confidence Score**")
            # Display probability for the predicted class
            confidence_score = prediction_proba[0][prediction[0]] * 100
            st.write(f"The model is **{confidence_score:.2f}%** confident in this prediction.")
            st.progress(int(confidence_score))

        # 4. Show input data for confirmation
        with st.expander("Show Input Data"):
            st.dataframe(input_df)
else:
    st.error("Model could not be loaded. The application cannot proceed with predictions. Please check the logs or the source files.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed for demonstration purposes. **Not for medical use.**")
