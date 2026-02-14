"""
Clinical Decision Support System: 30-Day Readmission Risk Assessment
=====================================================================

A production-grade Streamlit dashboard for healthcare providers to:
1. Input patient data and receive readmission risk scores
2. View SHAP-based explanations for predictions
3. Process batch patient files for population screening

Author: Healthcare Analytics Portfolio
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import joblib
from pathlib import Path
import io
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Hospital Readmission Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTS AND MAPPINGS
# =============================================================================

# ICD-9 Code Groups for user selection
ICD9_CATEGORIES = [
    "Circulatory (Heart disease, Stroke)",
    "Respiratory (Pneumonia, COPD)",
    "Digestive (GI disorders)",
    "Diabetes",
    "Injury/Trauma",
    "Musculoskeletal (Arthritis)",
    "Genitourinary (Kidney disease)",
    "Neoplasms (Cancer)",
    "Other"
]

ICD9_CATEGORY_MAP = {
    "Circulatory (Heart disease, Stroke)": "Circulatory",
    "Respiratory (Pneumonia, COPD)": "Respiratory",
    "Digestive (GI disorders)": "Digestive",
    "Diabetes": "Diabetes",
    "Injury/Trauma": "Injury",
    "Musculoskeletal (Arthritis)": "Musculoskeletal",
    "Genitourinary (Kidney disease)": "Genitourinary",
    "Neoplasms (Cancer)": "Neoplasms",
    "Other": "Other"
}

AGE_BRACKETS = [
    "0-10)", "10-20)", "20-30)", "30-40)", "40-50)",
    "50-60)", "60-70)", "70-80)", "80-90)", "90-100)"
]

RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"]
GENDERS = ["Female", "Male"]
YES_NO = ["Yes", "No"]
MED_CHANGES = ["No", "Ch", "Steady", "Up", "Down"]

# Risk tier thresholds and colors
RISK_TIERS = {
    'low': {'max': 0.15, 'color': '#27ae60', 'label': 'Low Risk'},
    'moderate': {'max': 0.30, 'color': '#f39c12', 'label': 'Moderate Risk'},
    'high': {'max': 0.50, 'color': '#e67e22', 'label': 'High Risk'},
    'very_high': {'max': 1.0, 'color': '#e74c3c', 'label': 'Very High Risk'}
}

# Clinical recommendations by tier
RECOMMENDATIONS = {
    'low': """
    **Standard Discharge Protocol**
    - Provide routine discharge instructions and medication list
    - Schedule standard follow-up appointment within 2 weeks
    - Ensure patient has pharmacy and PCP contact information
    """,
    'moderate': """
    **Enhanced Discharge Education**
    - Review warning signs requiring immediate medical attention
    - Confirm patient understanding of medication regimen
    - Schedule follow-up appointment within 1 week
    - Provide 24/7 nurse hotline contact information
    """,
    'high': """
    **Active Follow-up Required**
    - Schedule follow-up phone call within 48 to 72 hours
    - Consider home health referral for medication management
    - Ensure PCP appointment scheduled within 7 days
    - Arrange transportation assistance if needed
    - Provide clear written discharge instructions
    """,
    'very_high': """
    **Immediate Care Coordination Required**
    - Assign care coordinator before discharge
    - Schedule 24-hour post-discharge phone call
    - Arrange home health visit within 48 hours
    - Complete medication reconciliation with pharmacist
    - Consider social work consult for barriers to care
    - Schedule specialist follow-up within 3 days
    """
}

# =============================================================================
# CACHED RESOURCE LOADERS
# =============================================================================

@st.cache_resource
def load_model():
    """Load the trained XGBoost model (cached)."""
    model_path = Path("outputs/model/xgboost_readmission_model.joblib")
    if not model_path.exists():
        st.error("Model file not found. Please run main.py first to train the model.")
        return None
    return joblib.load(model_path)


@st.cache_resource
def load_shap_explainer(_model):
    """Load SHAP TreeExplainer (cached)."""
    if _model is None:
        return None
    return shap.TreeExplainer(_model)


@st.cache_resource
def load_training_info():
    """Load training info for feature columns."""
    info_path = Path("outputs/model/xgboost_readmission_model.info.joblib")
    if info_path.exists():
        try:
            return joblib.load(info_path)
        except Exception:
            # If loading fails due to compatibility issues, return None
            # We'll extract feature columns from processed data instead
            return None
    return None


@st.cache_data
def load_processed_data():
    """Load processed training data for reference."""
    data_path = Path("data/processed_data.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None


@st.cache_data
def get_feature_columns():
    """Get feature columns from processed data."""
    data_path = Path("data/processed_data.csv")
    if data_path.exists():
        df = pd.read_csv(data_path, nrows=1)
        # Remove target column if present
        cols = [c for c in df.columns if c != 'target']
        return cols
    return []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_risk_tier(risk_score: float) -> str:
    """Determine risk tier based on score."""
    if risk_score < RISK_TIERS['low']['max']:
        return 'low'
    elif risk_score < RISK_TIERS['moderate']['max']:
        return 'moderate'
    elif risk_score < RISK_TIERS['high']['max']:
        return 'high'
    else:
        return 'very_high'


def get_tier_color(risk_score: float) -> str:
    """Get color for risk tier."""
    tier = get_risk_tier(risk_score)
    return RISK_TIERS[tier]['color']


def create_gauge_chart(risk_score: float) -> go.Figure:
    """Create a Plotly gauge chart for risk visualization."""
    
    tier = get_risk_tier(risk_score)
    color = RISK_TIERS[tier]['color']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        number={'suffix': '%', 'font': {'size': 48}},
        title={'text': "30-Day Readmission Risk", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 15], 'color': '#d5f5e3'},
                {'range': [15, 30], 'color': '#fdebd0'},
                {'range': [30, 50], 'color': '#fad7a0'},
                {'range': [50, 100], 'color': '#f5b7b1'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.8,
                'value': risk_score * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        font={'family': "Arial"}
    )
    
    return fig


def create_risk_histogram(risk_scores: pd.Series) -> go.Figure:
    """Create histogram of risk scores for batch processing."""
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=risk_scores * 100,
        nbinsx=20,
        marker_color='#3498db',
        opacity=0.7,
        name='Patients'
    ))
    
    # Add risk tier lines
    fig.add_vline(x=15, line_dash="dash", line_color="#27ae60", 
                  annotation_text="Low/Moderate", annotation_position="top")
    fig.add_vline(x=30, line_dash="dash", line_color="#f39c12",
                  annotation_text="Moderate/High", annotation_position="top")
    fig.add_vline(x=50, line_dash="dash", line_color="#e74c3c",
                  annotation_text="High/Very High", annotation_position="top")
    
    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Risk Score (%)",
        yaxis_title="Number of Patients",
        height=400,
        showlegend=False
    )
    
    return fig


def preprocess_patient_input(patient_data: dict, feature_columns: list) -> pd.DataFrame:
    """
    Preprocess patient input to match model's expected format.
    Mirrors the preprocessing from src/preprocessing.py
    """
    
    # Create base dataframe
    df = pd.DataFrame([patient_data])
    
    # One-hot encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, dtype=int)
    
    # Clean column names (same as preprocessing.py)
    df.columns = [
        col.replace('[', '').replace(']', '').replace('<', 'lt').replace('>', 'gt')
        for col in df.columns
    ]
    
    # Align with model's feature columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure correct column order
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    return df


def create_shap_waterfall(shap_values, feature_names, patient_values, max_display=10):
    """Create a SHAP waterfall visualization using Plotly."""
    
    # Get feature contributions
    contributions = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values,
        'feature_value': patient_values
    })
    
    # Sort by absolute value and get top features
    contributions['abs_shap'] = contributions['shap_value'].abs()
    top_features = contributions.nlargest(max_display, 'abs_shap')
    top_features = top_features.sort_values('shap_value')
    
    # Create waterfall-like bar chart
    colors = ['#e74c3c' if v > 0 else '#27ae60' for v in top_features['shap_value']]
    
    fig = go.Figure(go.Bar(
        x=top_features['shap_value'],
        y=top_features['feature'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in top_features['shap_value']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Contributions to Risk Score",
        xaxis_title="SHAP Value (Impact on Risk)",
        yaxis_title="",
        height=400,
        margin=dict(l=200, r=50, t=50, b=50),
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
    )
    
    # Add annotation
    fig.add_annotation(
        text="Red = Increases Risk | Green = Decreases Risk",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    return fig


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    """Main dashboard function."""
    
    # Load model and resources
    model = load_model()
    explainer = load_shap_explainer(model)
    training_info = load_training_info()
    
    if model is None:
        st.error("Could not load the model. Please ensure the model is trained by running `python main.py`")
        st.stop()
    
    # Get feature columns - try training info first, then fall back to processed data
    feature_columns = []
    if training_info and 'feature_columns' in training_info:
        feature_columns = training_info.get('feature_columns', [])
    
    if not feature_columns:
        feature_columns = get_feature_columns()
    
    if not feature_columns:
        st.error("Could not load feature columns. Please ensure data/processed_data.csv exists.")
        st.stop()
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    
    st.title("🏥 Hospital Readmission Risk Assessment")
    st.markdown("""
    **Clinical Decision Support System** for predicting 30-day readmission risk in diabetic patients.
    
    This tool helps healthcare providers identify high-risk patients at discharge, enabling targeted 
    interventions through Transition of Care programs.
    """)
    
    # ==========================================================================
    # SIDEBAR: PATIENT INPUT FORM
    # ==========================================================================
    
    st.sidebar.header("📋 Patient Information")
    
    with st.sidebar.form("patient_form"):
        st.subheader("Demographics")
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", GENDERS)
        with col2:
            age = st.selectbox("Age Bracket", AGE_BRACKETS, index=6)
        
        race = st.selectbox("Race", RACES)
        
        st.subheader("Admission Details")
        
        time_in_hospital = st.number_input(
            "Time in Hospital (days)", 
            min_value=1, max_value=14, value=4
        )
        
        admission_type = st.selectbox(
            "Admission Type",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            format_func=lambda x: {
                1: "Emergency", 2: "Urgent", 3: "Elective",
                4: "Newborn", 5: "Not Available", 6: "NULL",
                7: "Trauma Center", 8: "Not Mapped"
            }.get(x, str(x))
        )
        
        discharge_disposition = st.selectbox(
            "Discharge Disposition",
            options=[1, 2, 3, 6, 11, 13, 14, 18, 22, 25],
            format_func=lambda x: {
                1: "Home", 2: "Short-term Hospital", 3: "SNF",
                6: "Home Health", 11: "Expired", 13: "Hospice",
                14: "LTC", 18: "Other", 22: "Rehab", 25: "Psych"
            }.get(x, str(x))
        )
        
        admission_source = st.selectbox(
            "Admission Source",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            format_func=lambda x: {
                1: "Physician Referral", 2: "Clinic Referral",
                3: "HMO Referral", 4: "Transfer from Hospital",
                5: "Transfer from SNF", 6: "Transfer from Other",
                7: "Emergency Room", 8: "Court/Law Enforcement",
                9: "Not Available", 10: "Transfer from Critical Care"
            }.get(x, str(x))
        )
        
        st.subheader("Clinical Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            num_lab_procedures = st.number_input(
                "Lab Procedures", min_value=0, max_value=150, value=40
            )
            num_procedures = st.number_input(
                "Other Procedures", min_value=0, max_value=10, value=1
            )
            num_medications = st.number_input(
                "Number of Medications", min_value=0, max_value=80, value=15
            )
        
        with col2:
            number_outpatient = st.number_input(
                "Outpatient Visits (past year)", min_value=0, max_value=50, value=0
            )
            number_emergency = st.number_input(
                "Emergency Visits (past year)", min_value=0, max_value=50, value=0
            )
            number_inpatient = st.number_input(
                "Inpatient Visits (past year)", min_value=0, max_value=20, value=0
            )
        
        number_diagnoses = st.number_input(
            "Number of Diagnoses", min_value=1, max_value=16, value=7
        )
        
        st.subheader("Diagnoses (ICD-9 Categories)")
        
        diag_1 = st.selectbox("Primary Diagnosis", ICD9_CATEGORIES, index=0)
        diag_2 = st.selectbox("Secondary Diagnosis", ICD9_CATEGORIES, index=3)
        diag_3 = st.selectbox("Additional Diagnosis", ICD9_CATEGORIES, index=8)
        
        st.subheader("Diabetes Management")
        
        col1, col2 = st.columns(2)
        with col1:
            diabetes_med = st.selectbox("On Diabetes Medication", YES_NO)
            change = st.selectbox("Medication Changed", ["No", "Ch"])
        
        with col2:
            a1c_result = st.selectbox(
                "A1C Result",
                options=["None", "Norm", "gt7", "gt8"]
            )
            max_glu_serum = st.selectbox(
                "Max Glucose Serum",
                options=["None", "Norm", "gt200", "gt300"]
            )
        
        # Common medications
        st.subheader("Key Medications")
        metformin = st.selectbox("Metformin", MED_CHANGES, index=0)
        insulin = st.selectbox("Insulin", MED_CHANGES, index=0)
        
        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)
    
    # ==========================================================================
    # MAIN PANEL: TABS
    # ==========================================================================
    
    tab1, tab2, tab3 = st.tabs(["📊 Single Patient", "📁 Batch Processing", "ℹ️ Model Info"])
    
    # --------------------------------------------------------------------------
    # TAB 1: SINGLE PATIENT ASSESSMENT
    # --------------------------------------------------------------------------
    
    with tab1:
        if submitted:
            # Prepare patient data
            patient_data = {
                'gender': gender,
                'age': age,
                'race': race,
                'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'number_diagnoses': number_diagnoses,
                'admission_type_id': admission_type,
                'discharge_disposition_id': discharge_disposition,
                'admission_source_id': admission_source,
                'diag_1_group': ICD9_CATEGORY_MAP[diag_1],
                'diag_2_group': ICD9_CATEGORY_MAP[diag_2],
                'diag_3_group': ICD9_CATEGORY_MAP[diag_3],
                'diabetesMed': diabetes_med,
                'change': change,
                'A1Cresult': a1c_result,
                'max_glu_serum': max_glu_serum,
                'metformin': metformin,
                'insulin': insulin
            }
            
            # Preprocess for model
            try:
                X_patient = preprocess_patient_input(patient_data, feature_columns)
                
                # Get prediction
                risk_score = model.predict_proba(X_patient)[:, 1][0]
                tier = get_risk_tier(risk_score)
                
                # Display results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.plotly_chart(create_gauge_chart(risk_score), use_container_width=True)
                
                with col2:
                    # Risk tier badge
                    tier_info = RISK_TIERS[tier]
                    st.markdown(f"""
                    <div style="background-color: {tier_info['color']}; 
                                padding: 20px; border-radius: 10px; 
                                text-align: center; margin-bottom: 20px;">
                        <h2 style="color: white; margin: 0;">{tier_info['label']}</h2>
                        <p style="color: white; font-size: 24px; margin: 10px 0;">
                            {risk_score*100:.1f}% Readmission Probability
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key metrics
                    st.markdown("**Key Risk Indicators:**")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Prior Inpatient", number_inpatient)
                    col_b.metric("Emergency Visits", number_emergency)
                    col_c.metric("Medications", num_medications)
                
                # Clinical Recommendation
                st.markdown("---")
                st.subheader("📋 Clinical Recommendation")
                st.markdown(RECOMMENDATIONS[tier])
                
                # SHAP Explanation
                st.markdown("---")
                st.subheader("🔬 Risk Factor Analysis (SHAP)")
                
                if explainer is not None:
                    shap_values = explainer.shap_values(X_patient)[0]
                    
                    fig_shap = create_shap_waterfall(
                        shap_values, 
                        feature_columns,
                        X_patient.iloc[0].values,
                        max_display=12
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    with st.expander("📖 How to interpret SHAP values"):
                        st.markdown("""
                        **SHAP (SHapley Additive exPlanations)** shows how each feature 
                        contributes to this patient's risk score:
                        
                        - **Red bars** indicate features that INCREASE readmission risk
                        - **Green bars** indicate features that DECREASE readmission risk
                        - **Longer bars** mean stronger influence on the prediction
                        
                        For example, a high number of prior inpatient visits typically 
                        increases risk, while being discharged to home (vs. SNF) may decrease it.
                        """)
                else:
                    st.warning("SHAP explainer not available")
                    
            except Exception as e:
                st.error(f"Error processing patient data: {str(e)}")
                st.info("Please ensure all fields are filled correctly.")
        
        else:
            # Default state
            st.info("👈 Enter patient information in the sidebar and click **Assess Risk** to see results.")
            
            # Show example
            with st.expander("📌 Quick Start Guide"):
                st.markdown("""
                **How to use this tool:**
                
                1. Enter patient demographics in the sidebar
                2. Fill in admission and clinical details
                3. Select diagnosis categories (simplified ICD-9 groups)
                4. Enter diabetes management information
                5. Click "Assess Risk" to see the prediction
                
                **What you will see:**
                - Risk score as a percentage
                - Risk tier (Low, Moderate, High, Very High)
                - Clinical recommendation based on tier
                - SHAP analysis showing which factors drive the risk
                """)
    
    # --------------------------------------------------------------------------
    # TAB 2: BATCH PROCESSING
    # --------------------------------------------------------------------------
    
    with tab2:
        st.subheader("📁 Batch Patient Processing")
        st.markdown("""
        Upload a CSV file with patient data to process multiple patients at once.
        The file should contain the same fields as the single patient form.
        """)
        
        uploaded_file = st.file_uploader(
            "Upload Patient CSV",
            type=['csv'],
            help="Upload a CSV file with patient data"
        )
        
        if uploaded_file is not None:
            try:
                # Load uploaded data
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(batch_df)} patients")
                
                # Show preview
                with st.expander("Preview uploaded data"):
                    st.dataframe(batch_df.head(10))
                
                if st.button("🚀 Run Batch Predictions", use_container_width=True):
                    with st.spinner("Processing patients..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, row in batch_df.iterrows():
                            try:
                                # Preprocess each patient
                                patient_dict = row.to_dict()
                                X = preprocess_patient_input(patient_dict, feature_columns)
                                risk_score = model.predict_proba(X)[:, 1][0]
                                tier = get_risk_tier(risk_score)
                                
                                results.append({
                                    'patient_index': idx,
                                    'risk_score': risk_score,
                                    'risk_percentage': f"{risk_score*100:.1f}%",
                                    'risk_tier': tier,
                                    'action_required': tier in ['high', 'very_high']
                                })
                            except Exception as e:
                                results.append({
                                    'patient_index': idx,
                                    'risk_score': None,
                                    'risk_percentage': 'Error',
                                    'risk_tier': 'Error',
                                    'action_required': False
                                })
                            
                            progress_bar.progress((idx + 1) / len(batch_df))
                        
                        results_df = pd.DataFrame(results)
                        
                        # Summary metrics
                        st.markdown("---")
                        st.subheader("📊 Batch Results Summary")
                        
                        valid_results = results_df[results_df['risk_score'].notna()]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Patients", len(results_df))
                        col2.metric("High/Very High Risk", 
                                   len(valid_results[valid_results['action_required']]))
                        col3.metric("Average Risk", 
                                   f"{valid_results['risk_score'].mean()*100:.1f}%")
                        col4.metric("Max Risk", 
                                   f"{valid_results['risk_score'].max()*100:.1f}%")
                        
                        # Risk distribution
                        st.plotly_chart(
                            create_risk_histogram(valid_results['risk_score']),
                            use_container_width=True
                        )
                        
                        # Tier breakdown
                        st.subheader("Risk Tier Distribution")
                        tier_counts = valid_results['risk_tier'].value_counts()
                        
                        fig_pie = px.pie(
                            values=tier_counts.values,
                            names=tier_counts.index,
                            color=tier_counts.index,
                            color_discrete_map={
                                'low': '#27ae60',
                                'moderate': '#f39c12',
                                'high': '#e67e22',
                                'very_high': '#e74c3c'
                            },
                            title="Patients by Risk Tier"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Results table
                        st.subheader("Detailed Results")
                        
                        # Highlight high risk
                        def highlight_risk(row):
                            if row['risk_tier'] == 'very_high':
                                return ['background-color: #f5b7b1'] * len(row)
                            elif row['risk_tier'] == 'high':
                                return ['background-color: #fad7a0'] * len(row)
                            return [''] * len(row)
                        
                        styled_df = results_df.style.apply(highlight_risk, axis=1)
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv_buffer.getvalue(),
                            file_name="readmission_risk_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        # Template download
        st.markdown("---")
        with st.expander("📋 Download CSV Template"):
            template_data = {
                'gender': ['Female', 'Male'],
                'age': ['60-70)', '70-80)'],
                'race': ['Caucasian', 'AfricanAmerican'],
                'time_in_hospital': [5, 3],
                'num_lab_procedures': [45, 30],
                'num_procedures': [2, 1],
                'num_medications': [15, 10],
                'number_outpatient': [0, 1],
                'number_emergency': [1, 0],
                'number_inpatient': [2, 0],
                'number_diagnoses': [8, 5],
                'admission_type_id': [1, 3],
                'discharge_disposition_id': [1, 1],
                'admission_source_id': [7, 1],
                'diag_1_group': ['Circulatory', 'Diabetes'],
                'diag_2_group': ['Diabetes', 'Other'],
                'diag_3_group': ['Other', 'Other'],
                'diabetesMed': ['Yes', 'Yes'],
                'change': ['Ch', 'No'],
                'A1Cresult': ['gt8', 'None'],
                'max_glu_serum': ['None', 'None'],
                'metformin': ['Steady', 'No'],
                'insulin': ['Up', 'No']
            }
            template_df = pd.DataFrame(template_data)
            
            csv_template = io.StringIO()
            template_df.to_csv(csv_template, index=False)
            
            st.download_button(
                label="📥 Download Template CSV",
                data=csv_template.getvalue(),
                file_name="patient_template.csv",
                mime="text/csv"
            )
    
    # --------------------------------------------------------------------------
    # TAB 3: MODEL INFORMATION
    # --------------------------------------------------------------------------
    
    with tab3:
        st.subheader("ℹ️ Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Details")
            st.markdown("""
            - **Algorithm**: XGBoost Classifier
            - **Dataset**: Diabetes 130-US Hospitals (UCI ML Repository)
            - **Training Samples**: 81,412 patient encounters
            - **Test Samples**: 20,354 patient encounters
            - **Features**: 140 clinical and demographic variables
            """)
        
        with col2:
            st.markdown("### Performance Metrics")
            
            # Display metrics with visual indicators
            metrics = [
                ("Recall (Sensitivity)", 0.558, 0.75, "Ability to catch high-risk patients"),
                ("AUC-ROC", 0.679, 0.70, "Overall discrimination ability"),
                ("Precision", 0.187, 0.50, "Accuracy of high-risk predictions")
            ]
            
            for name, value, target, desc in metrics:
                status = "🟢" if value >= target else "🟡"
                st.metric(
                    label=f"{status} {name}",
                    value=f"{value:.3f}",
                    delta=f"Target: {target}"
                )
                st.caption(desc)
        
        st.markdown("---")
        
        with st.expander("📚 Clinical Context"):
            st.markdown("""
            ### Why Predict Readmissions?
            
            Hospital readmissions within 30 days are a critical quality metric:
            
            - **Cost Impact**: Medicare penalizes hospitals with excess readmission rates
            - **Patient Safety**: Early readmissions often indicate care gaps
            - **Intervention Opportunity**: High-risk patients can receive targeted support
            
            ### Target Metrics Explained
            
            | Metric | Target | Why It Matters |
            |--------|--------|----------------|
            | **Recall** | >0.75 | Catch most high-risk patients (minimize missed cases) |
            | **AUC-ROC** | >0.70 | Model can distinguish between risk levels |
            | **Precision** | >0.50 | Avoid overwhelming care teams with false alarms |
            
            In healthcare, **Recall is prioritized** because missing a high-risk patient 
            (False Negative) is more dangerous than a false alarm (False Positive).
            """)
        
        with st.expander("🔬 Top Predictive Features"):
            st.markdown("""
            Based on SHAP analysis, these features most influence predictions:
            
            1. **number_inpatient**: Prior hospitalizations indicate disease severity
            2. **discharge_disposition_id**: Discharge destination affects follow-up care
            3. **num_lab_procedures**: More tests suggest diagnostic complexity
            4. **number_diagnoses**: Comorbidity burden
            5. **num_medications**: Medication count reflects complexity
            6. **time_in_hospital**: Longer stays indicate complex cases
            7. **number_emergency**: Emergency visits suggest unstable condition
            """)
        
        with st.expander("⚠️ Limitations & Considerations"):
            st.markdown("""
            **Important considerations when using this tool:**
            
            - This model is for **decision support only**, not autonomous decision-making
            - Clinical judgment should always take precedence
            - The model was trained on historical data (1999-2008) and may not reflect current practices
            - Individual patient circumstances may not be fully captured by the features
            - Regular model retraining is recommended as care patterns change
            
            **This tool should be used to:**
            - Flag patients who may benefit from additional follow-up
            - Prioritize care coordinator resources
            - Inform, not replace, clinical decision-making
            """)
    
    # ==========================================================================
    # FOOTER
    # ==========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px;">
        Healthcare Analytics Portfolio Project | 
        Data: UCI ML Repository, Diabetes 130-US Hospitals | 
        For educational and demonstration purposes only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
