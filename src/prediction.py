"""
Deployment-Ready Prediction Module
==================================

Clinical Context:
-----------------
This module provides the interface for clinical deployment:

1. Real-time prediction: Single patient risk assessment
2. Batch prediction: Population screening
3. Risk stratification: Categorize patients into risk tiers

Production Considerations:
- Input validation to prevent errors
- Consistent preprocessing with training pipeline
- Clear risk categories for clinical action
- Explainability for high-risk predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple
from pathlib import Path
import joblib
import warnings

from .preprocessing import group_icd9_codes


# Risk tier thresholds (can be adjusted based on hospital capacity)
RISK_THRESHOLDS = {
    'low': 0.15,      # Below 15%: Standard discharge
    'moderate': 0.30,  # 15-30%: Enhanced education
    'high': 0.50,      # 30-50%: Follow-up call scheduled
    'very_high': 1.0   # Above 50%: Care management referral
}


def predict_readmission(
    patient_data: Dict,
    model=None,
    model_path: str = "outputs/model/xgboost_readmission_model.joblib",
    feature_columns: Optional[List[str]] = None,
    return_explanation: bool = True
) -> Dict:
    """
    Predict 30-day readmission risk for a single patient.
    
    Parameters
    ----------
    patient_data : dict
        Dictionary containing patient features. Keys should match
        the features used during training.
        
        Required fields:
        - age: str, e.g., "[60-70)"
        - gender: str, e.g., "Female"
        - race: str, e.g., "Caucasian"
        - time_in_hospital: int
        - num_lab_procedures: int
        - num_medications: int
        - number_outpatient: int
        - number_emergency: int
        - number_inpatient: int
        - diag_1: str (ICD-9 code)
        - diag_2: str (ICD-9 code)
        - diag_3: str (ICD-9 code)
        - admission_type_id: int
        - discharge_disposition_id: int
        - diabetesMed: str, "Yes" or "No"
        
    model : trained model, optional
        Pre-loaded model. If None, loads from model_path.
    model_path : str
        Path to saved model file.
    feature_columns : list, optional
        List of feature column names in correct order.
    return_explanation : bool, default=True
        Whether to include feature contribution explanation.
    
    Returns
    -------
    dict
        Prediction results including:
        - risk_score: float (0-1 probability)
        - risk_tier: str (low/moderate/high/very_high)
        - prediction: int (0 or 1)
        - recommendation: str (clinical action)
        - top_risk_factors: list (if return_explanation=True)
    
    Example Usage:
    --------------
    >>> patient = {
    ...     'age': '[60-70)',
    ...     'gender': 'Female',
    ...     'time_in_hospital': 5,
    ...     'num_medications': 15,
    ...     'number_inpatient': 2,
    ...     'number_emergency': 1,
    ...     'diag_1': '428',  # Heart failure
    ...     'admission_type_id': 1,
    ...     'diabetesMed': 'Yes'
    ... }
    >>> result = predict_readmission(patient)
    >>> print(f"Risk: {result['risk_score']:.1%} ({result['risk_tier']})")
    Risk: 42.3% (high)
    """
    
    # Load model if not provided
    if model is None:
        model = joblib.load(model_path)
    
    # Preprocess patient data
    patient_df = _preprocess_patient_data(patient_data, feature_columns)
    
    # Get prediction
    risk_score = model.predict_proba(patient_df)[:, 1][0]
    prediction = int(risk_score >= 0.5)
    
    # Determine risk tier
    risk_tier = _get_risk_tier(risk_score)
    
    # Get clinical recommendation
    recommendation = _get_clinical_recommendation(risk_tier)
    
    result = {
        'risk_score': float(risk_score),
        'risk_percentage': f"{risk_score*100:.1f}%",
        'risk_tier': risk_tier,
        'prediction': prediction,
        'prediction_label': 'High Risk' if prediction == 1 else 'Low Risk',
        'recommendation': recommendation
    }
    
    # Add explanation if requested
    if return_explanation:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(patient_df)
            
            # Get top contributing features
            feature_contributions = pd.DataFrame({
                'feature': patient_df.columns,
                'shap_value': shap_values[0],
                'feature_value': patient_df.iloc[0].values
            })
            
            # Separate positive (risk-increasing) and negative (risk-decreasing) factors
            risk_factors = feature_contributions[feature_contributions['shap_value'] > 0]\
                .nlargest(5, 'shap_value')
            protective_factors = feature_contributions[feature_contributions['shap_value'] < 0]\
                .nsmallest(5, 'shap_value')
            
            result['top_risk_factors'] = risk_factors.to_dict('records')
            result['protective_factors'] = protective_factors.to_dict('records')
            
        except Exception as e:
            result['explanation_error'] = str(e)
    
    return result


def _preprocess_patient_data(
    patient_data: Dict,
    feature_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Preprocess single patient data to match training format.
    
    This function must apply the SAME transformations as the training pipeline:
    1. Group ICD-9 codes
    2. One-hot encode categorical variables
    3. Fill missing values
    4. Match feature order to training data
    """
    
    # Convert to DataFrame
    df = pd.DataFrame([patient_data])
    
    # Group ICD-9 diagnosis codes
    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        if diag_col in df.columns:
            df[f'{diag_col}_group'] = df[diag_col].apply(group_icd9_codes)
            df = df.drop(columns=[diag_col])
    
    # Handle missing values
    df = df.fillna('Unknown')
    
    # One-hot encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, dtype=int)
    
    # If feature columns are provided, align to them
    if feature_columns:
        # Add missing columns with 0
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Remove extra columns and reorder
        df = df[feature_columns]
    
    return df


def _get_risk_tier(risk_score: float) -> str:
    """Map risk score to clinical risk tier."""
    
    if risk_score < RISK_THRESHOLDS['low']:
        return 'low'
    elif risk_score < RISK_THRESHOLDS['moderate']:
        return 'moderate'
    elif risk_score < RISK_THRESHOLDS['high']:
        return 'high'
    else:
        return 'very_high'


def _get_clinical_recommendation(risk_tier: str) -> str:
    """
    Get clinical action recommendation based on risk tier.
    
    Clinical Context:
    -----------------
    These recommendations are based on Transition of Care best practices:
    - Low risk: Standard discharge procedures
    - Moderate risk: Enhanced discharge education
    - High risk: Active follow-up (calls, appointments)
    - Very high risk: Care management enrollment
    """
    
    recommendations = {
        'low': (
            "Standard discharge protocol. "
            "Provide routine discharge instructions and medication list. "
            "Schedule standard follow-up appointment within 2 weeks."
        ),
        'moderate': (
            "Enhanced discharge education recommended. "
            "Review warning signs requiring immediate care. "
            "Confirm patient understanding of medication regimen. "
            "Schedule follow-up within 1 week."
        ),
        'high': (
            "Schedule follow-up phone call within 48-72 hours. "
            "Consider home health referral for medication management. "
            "Ensure PCP appointment scheduled within 7 days. "
            "Provide 24/7 nurse hotline information."
        ),
        'very_high': (
            "REFER TO CARE MANAGEMENT PROGRAM. "
            "Schedule post-discharge home visit. "
            "Ensure medication reconciliation completed. "
            "Consider social work consult for barriers to care. "
            "Schedule follow-up with specialist within 3 days."
        )
    }
    
    return recommendations.get(risk_tier, "Please consult clinical team.")


class PatientPredictor:
    """
    Class-based interface for production deployment.
    
    Usage:
    ------
    >>> predictor = PatientPredictor('outputs/model/xgboost_readmission_model.joblib')
    >>> result = predictor.predict(patient_data)
    >>> batch_results = predictor.predict_batch(patient_dataframe)
    """
    
    def __init__(
        self,
        model_path: str = "outputs/model/xgboost_readmission_model.joblib",
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the predictor with a trained model.
        
        Parameters
        ----------
        model_path : str
            Path to the saved model file.
        feature_columns : list, optional
            Feature column names in correct order.
        """
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = joblib.load(self.model_path)
        self.feature_columns = feature_columns
        
        # Try to load training info for feature columns
        info_path = self.model_path.with_suffix('.info.joblib')
        if info_path.exists():
            self.training_info = joblib.load(info_path)
            if 'final_features' in self.training_info.get('preprocessing_info', {}):
                self.feature_columns = self.training_info['preprocessing_info']['final_features']
    
    def predict(self, patient_data: Dict) -> Dict:
        """Predict readmission risk for a single patient."""
        return predict_readmission(
            patient_data,
            model=self.model,
            feature_columns=self.feature_columns
        )
    
    def predict_batch(
        self,
        patients_df: pd.DataFrame,
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Predict readmission risk for multiple patients.
        
        Parameters
        ----------
        patients_df : pd.DataFrame
            DataFrame with patient features.
        return_dataframe : bool, default=True
            Whether to return results as DataFrame.
        
        Returns
        -------
        pd.DataFrame or list
            Prediction results for all patients.
        """
        
        results = []
        
        for idx, patient in patients_df.iterrows():
            patient_dict = patient.to_dict()
            result = self.predict(patient_dict)
            result['patient_idx'] = idx
            results.append(result)
        
        if return_dataframe:
            return pd.DataFrame(results)
        
        return results
    
    def stratify_population(
        self,
        patients_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Stratify patient population by risk tier.
        
        Returns
        -------
        dict
            Dictionary with keys 'low', 'moderate', 'high', 'very_high'
            containing DataFrames of patients in each tier.
        
        Clinical Use:
        -------------
        Population stratification enables targeted interventions:
        - Very High: Immediate care management outreach
        - High: Phone call within 48 hours
        - Moderate: Enhanced discharge education
        - Low: Standard care pathway
        """
        
        predictions = self.predict_batch(patients_df)
        
        stratified = {}
        for tier in ['low', 'moderate', 'high', 'very_high']:
            tier_mask = predictions['risk_tier'] == tier
            stratified[tier] = predictions[tier_mask]
            
            print(f"{tier.upper()}: {tier_mask.sum()} patients ({tier_mask.mean()*100:.1f}%)")
        
        return stratified


# Example patient data for testing
SAMPLE_PATIENT = {
    'age': '[60-70)',
    'gender': 'Female',
    'race': 'Caucasian',
    'time_in_hospital': 7,
    'num_lab_procedures': 50,
    'num_procedures': 3,
    'num_medications': 18,
    'number_outpatient': 0,
    'number_emergency': 1,
    'number_inpatient': 2,
    'number_diagnoses': 9,
    'max_glu_serum': 'None',
    'A1Cresult': '>8',
    'metformin': 'Steady',
    'insulin': 'Up',
    'change': 'Ch',
    'diabetesMed': 'Yes',
    'diag_1': '428',  # Heart failure
    'diag_2': '250',  # Diabetes
    'diag_3': '401',  # Hypertension
    'admission_type_id': 1,
    'discharge_disposition_id': 1,
    'admission_source_id': 7
}


if __name__ == "__main__":
    # Test prediction function
    print("Testing prediction module with sample patient...")
    print("-" * 50)
    
    try:
        result = predict_readmission(SAMPLE_PATIENT)
        
        print(f"\nRisk Score: {result['risk_percentage']}")
        print(f"Risk Tier: {result['risk_tier'].upper()}")
        print(f"Prediction: {result['prediction_label']}")
        print(f"\nRecommendation:\n{result['recommendation']}")
        
        if 'top_risk_factors' in result:
            print("\nTop Risk Factors:")
            for factor in result['top_risk_factors'][:3]:
                print(f"  • {factor['feature']}: {factor['shap_value']:.4f}")
                
    except FileNotFoundError:
        print("Model not found. Please train the model first by running main.py")
