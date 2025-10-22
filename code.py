import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .negative {
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND FEATURES
# ============================================================================
@st.cache_resource
def load_model_and_features():
    """Load the trained model and selected features list"""
    try:
        # Load model and features from local files (these will be in your GitHub repo)
        model = joblib.load('final_random_forest_model.joblib')
        features = joblib.load('selected_features_list.joblib')
        return model, features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model and features
model, selected_features = load_model_and_features()

# ============================================================================
# HEADER SECTION
# ============================================================================
st.title("🏥 Liver Disease Prediction System")
st.markdown("---")

# Introduction
with st.expander("ℹ️ About this Application", expanded=False):
    st.markdown("""
    ### About Liver Disease Prediction
    
    This application uses a **Random Forest Classification Model** trained on liver patient data 
    to predict the likelihood of liver disease based on blood test parameters.
    
    #### Model Performance:
    - **Recall Class 0 (Non-Liver Disease)**: 99.73%
    - **Recall Class 1 (Liver Disease)**: 99.96%
    - **Balanced Accuracy**: 99.84%
    
    #### Key Features Used:
    The model uses 8 important blood test parameters to make predictions.
    
    #### How to use:
    1. Enter patient's blood test values in the sidebar
    2. Click 'Predict' to get the prediction
    3. View the detailed results and probability scores
    
    **Note:** This tool is for informational purposes only and should not replace professional medical advice.
    """)

# ============================================================================
# SIDEBAR - INPUT FORM
# ============================================================================
st.sidebar.header("📋 Patient Blood Test Parameters")
st.sidebar.markdown("Enter the patient's test values below:")

# Feature descriptions and normal ranges
feature_info = {
    'Alkphos_Alkaline_Phosphotase': {
        'label': 'Alkaline Phosphotase (ALP)',
        'unit': 'IU/L',
        'normal_range': '44-147',
        'min': 0.0,
        'max': 2000.0,
        'default': 200.0,
        'help': 'Normal range: 44-147 IU/L'
    },
    'Sgot_Aspartate_Aminotransferase': {
        'label': 'SGOT (AST)',
        'unit': 'IU/L',
        'normal_range': '0-40',
        'min': 0.0,
        'max': 2000.0,
        'default': 40.0,
        'help': 'Normal range: 0-40 IU/L'
    },
    'Sgpt_Alamine_Aminotransferase': {
        'label': 'SGPT (ALT)',
        'unit': 'IU/L',
        'normal_range': '0-41',
        'min': 0.0,
        'max': 2000.0,
        'default': 40.0,
        'help': 'Normal range: 0-41 IU/L'
    },
    'Total_Bilirubin': {
        'label': 'Total Bilirubin',
        'unit': 'mg/dL',
        'normal_range': '0.3-1.2',
        'min': 0.0,
        'max': 75.0,
        'default': 1.0,
        'help': 'Normal range: 0.3-1.2 mg/dL'
    },
    'Total_Proteins': {
        'label': 'Total Proteins',
        'unit': 'g/dL',
        'normal_range': '6.0-8.3',
        'min': 0.0,
        'max': 15.0,
        'default': 7.0,
        'help': 'Normal range: 6.0-8.3 g/dL'
    },
    'Direct_Bilirubin': {
        'label': 'Direct Bilirubin',
        'unit': 'mg/dL',
        'normal_range': '0.0-0.3',
        'min': 0.0,
        'max': 50.0,
        'default': 0.2,
        'help': 'Normal range: 0.0-0.3 mg/dL'
    },
    'ALB_Albumin': {
        'label': 'Albumin',
        'unit': 'g/dL',
        'normal_range': '3.5-5.5',
        'min': 0.0,
        'max': 10.0,
        'default': 4.0,
        'help': 'Normal range: 3.5-5.5 g/dL'
    },
    'A/G_Ratio_Albumin_and_Globulin_Ratio': {
        'label': 'A/G Ratio',
        'unit': 'ratio',
        'normal_range': '1.0-2.5',
        'min': 0.0,
        'max': 5.0,
        'default': 1.5,
        'help': 'Normal range: 1.0-2.5'
    }
}

# Collect user inputs
user_inputs = {}

if model is not None and selected_features is not None:
    st.sidebar.markdown("### Enter Values:")
    
    for feature in selected_features:
        info = feature_info[feature]
        
        # Create input field with normal range display
        st.sidebar.markdown(f"**{info['label']}** ({info['unit']})")
        st.sidebar.caption(f"Normal: {info['normal_range']}")
        
        user_inputs[feature] = st.sidebar.number_input(
            label=f"{feature}",
            min_value=info['min'],
            max_value=info['max'],
            value=info['default'],
            step=0.1,
            format="%.2f",
            help=info['help'],
            label_visibility="collapsed"
        )
        st.sidebar.markdown("---")

    # Predict button
    predict_button = st.sidebar.button("🔍 Predict", use_container_width=True)
    
    # Reset button
    if st.sidebar.button("🔄 Reset to Default Values", use_container_width=True):
        st.rerun()

# ============================================================================
# MAIN CONTENT - PREDICTION RESULTS
# ============================================================================

if model is not None and selected_features is not None:
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Entered Test Values")
        
        # Display input values in a nice table
        input_df = pd.DataFrame({
            'Parameter': [feature_info[f]['label'] for f in selected_features],
            'Value': [user_inputs[f] for f in selected_features],
            'Unit': [feature_info[f]['unit'] for f in selected_features],
            'Normal Range': [feature_info[f]['normal_range'] for f in selected_features]
        })
        
        st.dataframe(input_df, use_container_width=True, height=350)
    
    with col2:
        st.header("🎯 Quick Stats")
        
        # Calculate some statistics
        total_params = len(selected_features)
        st.metric("Total Parameters", total_params)
        st.metric("Model Type", "Random Forest")
        st.metric("Model Accuracy", "99.84%")
    
    st.markdown("---")
    
    # Make prediction when button is clicked
    if predict_button:
        with st.spinner('🔄 Analyzing patient data...'):
            # Prepare input data
            input_data = pd.DataFrame([user_inputs])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.header("🎯 Prediction Results")
            
            # Create three columns for results
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-box positive">
                        <h2 style="color: #d32f2f; margin:0;">⚠️ POSITIVE</h2>
                        <p style="margin:5px 0 0 0; font-size:18px;">Liver Disease Detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-box negative">
                        <h2 style="color: #388e3c; margin:0;">✅ NEGATIVE</h2>
                        <p style="margin:5px 0 0 0; font-size:18px;">No Liver Disease Detected</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                st.metric(
                    label="Liver Disease Probability",
                    value=f"{prediction_proba[1]*100:.2f}%",
                    delta=f"{prediction_proba[1]*100 - 50:.2f}% from baseline"
                )
            
            with res_col3:
                st.metric(
                    label="No Liver Disease Probability",
                    value=f"{prediction_proba[0]*100:.2f}%",
                    delta=f"{50 - prediction_proba[0]*100:.2f}% from baseline",
                    delta_color="inverse"
                )
            
            st.markdown("---")
            
            # Probability visualization
            st.subheader("📈 Prediction Probability Distribution")
            
            # Create a gauge chart using plotly
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['No Liver Disease', 'Liver Disease'],
                y=[prediction_proba[0]*100, prediction_proba[1]*100],
                marker=dict(
                    color=['#4CAF50', '#f44336'],
                    line=dict(color='rgb(8,48,107)', width=1.5)
                ),
                text=[f'{prediction_proba[0]*100:.2f}%', f'{prediction_proba[1]*100:.2f}%'],
                textposition='auto',
                textfont=dict(size=16, color='white', family='Arial Black')
            ))
            
            fig.update_layout(
                title='Prediction Confidence',
                yaxis=dict(title='Probability (%)', range=[0, 100]),
                xaxis=dict(title='Classification'),
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on prediction
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if prediction == 1:
                st.warning("""
                ### ⚠️ Action Required
                
                Based on the analysis, the model has detected potential liver disease indicators. 
                
                **Recommended Actions:**
                - 🏥 Consult with a healthcare professional immediately
                - 📋 Get comprehensive liver function tests
                - 🔬 Consider additional diagnostic procedures (ultrasound, CT scan, etc.)
                - 📖 Discuss results with a hepatologist or gastroenterologist
                - ⚕️ Do not ignore these results - early detection is crucial
                
                **Important:** This prediction is based on a machine learning model and should not be 
                considered as a final diagnosis. Always consult with qualified healthcare professionals.
                """)
            else:
                st.success("""
                ### ✅ Results Look Good
                
                Based on the analysis, the model has not detected significant liver disease indicators.
                
                **Recommended Actions:**
                - 🏥 Continue regular health check-ups
                - 🥗 Maintain a healthy diet and lifestyle
                - 💧 Stay hydrated and limit alcohol consumption
                - 🏃 Regular exercise and maintain healthy weight
                - 📅 Schedule periodic liver function tests as recommended by your doctor
                
                **Important:** Even with negative results, maintain regular health monitoring and 
                consult with your healthcare provider about your overall health.
                """)
            
            # Feature importance contribution (if available)
            st.markdown("---")
            st.subheader("🔍 Parameter Analysis")
            
            # Get feature importances from the model
            feature_importance = pd.DataFrame({
                'Parameter': [feature_info[f]['label'] for f in selected_features],
                'Importance': model.feature_importances_,
                'Your Value': [user_inputs[f] for f in selected_features]
            }).sort_values('Importance', ascending=False)
            
            # Create importance chart
            fig_importance = go.Figure()
            
            fig_importance.add_trace(go.Bar(
                x=feature_importance['Importance'],
                y=feature_importance['Parameter'],
                orientation='h',
                marker=dict(
                    color=feature_importance['Importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f'{val:.3f}' for val in feature_importance['Importance']],
                textposition='auto',
            ))
            
            fig_importance.update_layout(
                title='Feature Importance in Prediction',
                xaxis=dict(title='Importance Score'),
                yaxis=dict(title='Parameter'),
                height=400,
                showlegend=False,
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
    
    else:
        # Display message when no prediction has been made yet
        st.info("👈 Please enter the patient's blood test values in the sidebar and click 'Predict' to see results.")
        
        # Show some model information
        st.header("🤖 Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Model Type</h3>
                <p style="font-size: 24px; font-weight: bold; color: #1f77b4;">Random Forest</p>
                <p>100 estimators with balanced class weights</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Training Data</h3>
                <p style="font-size: 24px; font-weight: bold; color: #1f77b4;">Stratified CV</p>
                <p>5-Fold Cross-Validation with stratification</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Feature Selection</h3>
                <p style="font-size: 24px; font-weight: bold; color: #1f77b4;">MDI Method</p>
                <p>Mean Decrease in Impurity</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.error("""
    ⚠️ **Error Loading Model**
    
    The model files could not be loaded. Please ensure:
    1. `final_random_forest_model.joblib` is in the repository
    2. `selected_features_list.joblib` is in the repository
    3. Files are in the root directory of your GitHub repository
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Liver Disease Prediction System</strong></p>
    <p>Powered by Random Forest Machine Learning | Built with Streamlit</p>
    <p style="font-size: 12px;">⚕️ For educational and informational purposes only. Not a substitute for professional medical advice.</p>
    <p style="font-size: 12px;">© 2024 | Model Accuracy: 99.84%</p>
</div>
""", unsafe_allow_html=True)
