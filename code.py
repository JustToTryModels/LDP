import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Liver Disease Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL AND FEATURE LOADING ---
# Use st.cache_resource to load the model and features only once
@st.cache_resource
def load_model_and_features():
    """
    Loads the trained model and the list of selected features from GitHub.
    """
    try:
        # URLs to the raw files on GitHub
        model_url = "https://github.com/JustToTryModels/LDP/raw/main/LDP_RFC_Model/final_random_forest_model.joblib"
        features_url = "https://github.com/JustToTryModels/LDP/raw/main/LDP_RFC_Model/selected_features_list.joblib"

        # Download the model file
        model_response = requests.get(model_url)
        model_response.raise_for_status()  # Raise an exception for bad status codes
        model_file = io.BytesIO(model_response.content)
        model = joblib.load(model_file)

        # Download the features file
        features_response = requests.get(features_url)
        features_response.raise_for_status()
        features_file = io.BytesIO(features_response.content)
        features_list = joblib.load(features_file)

        return model, features_list
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Please ensure the GitHub repository and files are public and the URLs are correct.")
        return None, None

# Load the model and features
model, selected_features = load_model_and_features()

# --- HELPER FUNCTION ---
def format_feature_name(name):
    """Cleans up feature names for display."""
    return name.replace('_', ' ').title()

# --- UI LAYOUT ---
if model is not None and selected_features is not None:
    # --- HEADER ---
    st.title("🩺 Liver Disease Prediction Tool")
    st.markdown("""
    This application uses a **Random Forest Classifier** to predict the likelihood of liver disease based on diagnostic measurements.
    
    **Instructions:**
    1.  Adjust the sliders in the sidebar to match the patient's test results.
    2.  The model will automatically update the prediction based on your inputs.
    3.  Review the prediction and confidence score below.
    
    ---
    """)

    # --- SIDEBAR FOR USER INPUT ---
    st.sidebar.header("Patient Data Input")
    st.sidebar.markdown("Use the sliders to enter patient information.")

    # Create a dictionary to hold user inputs
    input_data = {}

    # Dynamically create sliders for each feature the model needs
    # Using educated guesses for min, max, and default values.
    # You can adjust these based on your dataset's statistics (e.g., df.describe()).
    feature_defaults = {
        'Alkphos_Alkaline_Phosphotase': (30, 1000, 120),
        'Sgot_Aspartate_Aminotransferase': (10, 500, 40),
        'Sgpt_Alamine_Aminotransferase': (10, 500, 45),
        'Total_Bilirubin': (0.1, 20.0, 1.0),
        'Total_Proteins': (4.0, 10.0, 7.0),
        'Direct_Bilirubin': (0.1, 10.0, 0.4),
        'ALB_Albumin': (2.0, 6.0, 4.0),
        'A/G_Ratio_Albumin_and_Globulin_Ratio': (0.5, 2.5, 1.1)
    }

    for feature in selected_features:
        min_val, max_val, default_val = feature_defaults.get(feature, (0, 100, 50))
        input_data[feature] = st.sidebar.slider(
            label=format_feature_name(feature),
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.1
        )
    
    st.sidebar.info("The `A/G Ratio` may be automatically calculated in a real lab setting but is included here for completeness.")

    # --- PREDICTION AND DISPLAY ---
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Prediction Result")

        # Convert input data to a DataFrame in the correct order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[selected_features] # Ensure column order matches training

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Display the result
        if prediction == 1:
            st.error("Prediction: Liver Disease LIKELY", icon="⚠️")
            confidence_score = prediction_proba[1]
        else:
            st.success("Prediction: Liver Disease UNLIKELY", icon="✅")
            confidence_score = prediction_proba[0]

        # Display confidence score
        st.metric(label="Confidence Score", value=f"{confidence_score:.2%}")
        st.markdown("""
        *The confidence score represents the model's certainty in its prediction. Higher scores indicate greater confidence.*
        """)
        
        st.warning("""
        **Disclaimer:** This is an AI-powered prediction tool. It is **not a substitute for professional medical advice**, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.
        """)


    with col2:
        st.subheader("Input Values Summary")
        
        # Create a clean DataFrame for display
        display_df = pd.DataFrame({
            "Medical Test": [format_feature_name(f) for f in input_df.columns],
            "Value": [f"{v:.1f}" for v in input_df.iloc[0].values]
        })
        st.table(display_df.set_index("Medical Test"))


    # --- MODEL INFO EXPANDER ---
    with st.expander("ℹ️ About the Model and Features"):
        st.markdown(f"""
        This prediction is generated by a `RandomForestClassifier` model.
        
        **Model Details:**
        - **Imbalance Handling:** `class_weight='balanced'` was used during training.
        - **Feature Selection:** The model was trained on a subset of features selected for optimal performance.
        
        **Features Used for Prediction:**
        """)
        
        # Create a list of features with bullet points
        feature_list_md = ""
        for feature in selected_features:
            feature_list_md += f"- **{format_feature_name(feature)}**\n"
        st.markdown(feature_list_md)

else:
    st.error("Model could not be loaded. The application cannot start.")
    st.markdown("Please check the console or server logs for detailed error messages.")
