"""
Data Preprocessing Module
=========================

Clinical Context:
-----------------
Proper preprocessing is essential for healthcare ML models because:
1. Missing data is common in EHR systems (incomplete charting, varied documentation)
2. ICD-9 codes are highly granular (700+ codes) - grouping improves generalization
3. Feature encoding affects model interpretability for clinical staff

Key Preprocessing Steps:
1. Handle missing values ('?' placeholders common in medical datasets)
2. Create binary target (readmitted <30 days vs not)
3. Group ICD-9 diagnosis codes into clinical categories
4. One-hot encode categorical variables
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings


# =============================================================================
# ICD-9 Code Groupings
# =============================================================================

# Clinical grouping of ICD-9 diagnosis codes
# Based on standard medical coding categories
ICD9_GROUPS: Dict[str, Tuple[float, float]] = {
    'Circulatory': (390, 459),      # Heart disease, stroke, hypertension
    'Respiratory': (460, 519),       # Pneumonia, COPD, asthma
    'Digestive': (520, 579),         # GI disorders, liver disease
    'Diabetes': (250, 250.99),       # Diabetes-specific codes
    'Injury': (800, 999),            # Trauma, poisoning, complications
    'Musculoskeletal': (710, 739),   # Arthritis, back problems
    'Genitourinary': (580, 629),     # Kidney disease, UTI
    'Neoplasms': (140, 239),         # Cancer
    'Other': (0, 0)                   # Catch-all category
}


def preprocess_data(
    df: pd.DataFrame,
    drop_columns: Optional[List[str]] = None,
    handle_missing: bool = True,
    group_diagnoses: bool = True,
    encode_categorical: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Complete preprocessing pipeline for diabetes readmission prediction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw diabetes dataset.
    drop_columns : list, optional
        Additional columns to drop. Defaults to ID columns and high-missing columns.
    handle_missing : bool, default=True
        Whether to handle missing values.
    group_diagnoses : bool, default=True
        Whether to group ICD-9 codes into clinical categories.
    encode_categorical : bool, default=True
        Whether to one-hot encode categorical variables.
    
    Returns
    -------
    X : pd.DataFrame
        Processed feature matrix.
    y : pd.Series
        Binary target (1 = readmitted <30 days, 0 = otherwise).
    preprocessing_info : dict
        Information about preprocessing steps for reproducibility.
    
    Clinical Note:
    --------------
    We drop 'weight' due to >95% missing values - this is common in EHR data
    as weight is not consistently recorded. Similarly, 'payer_code' and 
    'medical_specialty' have significant missingness.
    """
    
    df = df.copy()
    preprocessing_info = {}
    
    # Step 1: Replace '?' with NaN (common in medical datasets)
    print("Step 1: Handling '?' placeholders...")
    df = replace_missing_placeholders(df)
    
    # Step 2: Create binary target variable
    print("Step 2: Creating binary target variable...")
    y = create_binary_target(df['readmitted'])
    preprocessing_info['target_distribution'] = y.value_counts().to_dict()
    
    # Step 3: Drop unnecessary columns
    print("Step 3: Dropping irrelevant/high-missing columns...")
    default_drops = [
        'encounter_id',     # ID column - no predictive value
        'patient_nbr',      # ID column - no predictive value  
        'weight',           # >95% missing - unreliable
        'payer_code',       # ~40% missing, limited clinical value
        'medical_specialty', # ~50% missing
        'readmitted'        # Target variable (now in y)
    ]
    
    if drop_columns:
        default_drops.extend(drop_columns)
    
    cols_to_drop = [c for c in default_drops if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    preprocessing_info['dropped_columns'] = cols_to_drop
    
    # Step 4: Handle missing values in remaining columns
    if handle_missing:
        print("Step 4: Handling remaining missing values...")
        df = handle_missing_values(df)
    
    # Step 5: Group ICD-9 diagnosis codes
    if group_diagnoses:
        print("Step 5: Grouping ICD-9 diagnosis codes...")
        for diag_col in ['diag_1', 'diag_2', 'diag_3']:
            if diag_col in df.columns:
                df[f'{diag_col}_group'] = df[diag_col].apply(group_icd9_codes)
                df = df.drop(columns=[diag_col])
        preprocessing_info['icd9_grouped'] = True
    
    # Step 6: Encode categorical variables
    if encode_categorical:
        print("Step 6: One-hot encoding categorical variables...")
        df, encoding_info = encode_categorical_variables(df)
        preprocessing_info['encoding_info'] = encoding_info
    
    # Final cleanup
    print("Step 7: Final cleanup...")
    X = df.select_dtypes(include=[np.number])
    
    # Ensure no infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    preprocessing_info['final_features'] = X.columns.tolist()
    preprocessing_info['n_features'] = X.shape[1]
    
    print(f"\nPreprocessing complete!")
    print(f"Final dataset shape: {X.shape}")
    print(f"Target distribution: {preprocessing_info['target_distribution']}")
    
    return X, y, preprocessing_info


def replace_missing_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace '?' placeholders with NaN.
    
    Clinical Context:
    -----------------
    Medical datasets often use '?' or similar placeholders for missing data
    instead of proper null values. This is an artifact of legacy EHR systems.
    """
    
    df = df.replace('?', np.nan)
    
    # Report missing value statistics
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 10]
    
    if len(high_missing) > 0:
        print("\nColumns with >10% missing values:")
        for col, pct in high_missing.items():
            print(f"  {col}: {pct:.1f}%")
    
    return df


def create_binary_target(readmitted: pd.Series) -> pd.Series:
    """
    Convert readmission status to binary target.
    
    Parameters
    ----------
    readmitted : pd.Series
        Original readmission column with values: '<30', '>30', 'NO'
    
    Returns
    -------
    pd.Series
        Binary target: 1 if readmitted within 30 days, 0 otherwise.
    
    Clinical Context:
    -----------------
    30-day readmission is a critical quality metric because:
    1. CMS penalizes hospitals with excess 30-day readmissions
    2. Early readmissions often indicate inadequate discharge planning
    3. Interventions are most effective in the first 30 days post-discharge
    """
    
    binary_target = (readmitted == '<30').astype(int)
    
    # Report class distribution
    positive_rate = binary_target.mean() * 100
    print(f"Positive class (<30 day readmission): {positive_rate:.2f}%")
    print(f"Class imbalance ratio: 1:{int((1-binary_target.mean())/binary_target.mean())}")
    
    return binary_target


def group_icd9_codes(code: str) -> str:
    """
    Group ICD-9 diagnosis codes into clinical categories.
    
    Parameters
    ----------
    code : str
        ICD-9 diagnosis code (may include letters like 'V' or 'E')
    
    Returns
    -------
    str
        Clinical category name.
    
    Clinical Context:
    -----------------
    ICD-9 codes are highly specific (e.g., 250.00 = Type 2 DM without complications,
    250.01 = Type 1 DM without complications). Grouping into broader categories:
    1. Reduces dimensionality (700+ codes → 9 categories)
    2. Improves model generalization
    3. Makes results more interpretable for clinicians
    """
    
    if pd.isna(code):
        return 'Other'
    
    code = str(code).strip()
    
    # Handle special codes (V codes = supplementary, E codes = external causes)
    if code.startswith('V'):
        return 'Other'
    if code.startswith('E'):
        return 'Injury'
    
    # Try to parse numeric code
    try:
        # Remove any non-numeric suffix
        numeric_code = float(''.join(c for c in code if c.isdigit() or c == '.'))
        
        # Check against ICD-9 ranges
        for group_name, (start, end) in ICD9_GROUPS.items():
            if group_name == 'Other':
                continue
            if start <= numeric_code <= end:
                return group_name
                
    except (ValueError, TypeError):
        pass
    
    return 'Other'


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using appropriate strategies.
    
    Strategy:
    - Categorical: Fill with 'Unknown' category
    - Numeric: Fill with median (robust to outliers)
    
    Clinical Context:
    -----------------
    Simple imputation (mode/median) is often preferred in healthcare ML because:
    1. Complex imputation can introduce bias in clinical features
    2. 'Unknown' is a valid clinical state that may be predictive
    3. Transparency is crucial for model interpretability
    """
    
    for col in df.columns:
        if df[col].isna().any():
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                # Categorical or string columns
                df[col] = df[col].fillna('Unknown')
    
    return df


def encode_categorical_variables(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    One-hot encode categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with categorical columns.
    
    Returns
    -------
    df_encoded : pd.DataFrame
        DataFrame with categorical variables one-hot encoded.
    encoding_info : dict
        Information about encoding for reproducibility.
    
    Clinical Context:
    -----------------
    One-hot encoding is preferred over label encoding for:
    1. Nominal variables (no inherent order, e.g., race, diagnosis)
    2. Model interpretability (each category gets its own coefficient)
    3. Avoiding false ordinal relationships
    """
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    encoding_info = {'categorical_columns': categorical_cols}
    
    if not categorical_cols:
        return df, encoding_info
    
    # One-hot encode
    df_encoded = pd.get_dummies(
        df, 
        columns=categorical_cols,
        drop_first=False,  # Keep all categories for interpretability
        dtype=int
    )
    
    # Clean column names for XGBoost compatibility
    # XGBoost doesn't allow [, ], < characters in feature names
    df_encoded.columns = [
        col.replace('[', '').replace(']', '').replace('<', 'lt').replace('>', 'gt')
        for col in df_encoded.columns
    ]
    
    encoding_info['new_columns'] = [
        c for c in df_encoded.columns if c not in df.columns
    ]
    encoding_info['n_new_features'] = len(encoding_info['new_columns'])
    
    print(f"Created {encoding_info['n_new_features']} new features from {len(categorical_cols)} categorical columns")
    
    return df_encoded, encoding_info


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Get predefined feature groups for analysis.
    
    Returns
    -------
    dict
        Dictionary mapping group names to feature patterns.
    
    Clinical Context:
    -----------------
    Features are grouped by clinical domain to facilitate:
    1. Domain-specific analysis
    2. Feature selection by clinical relevance
    3. Interpretability for healthcare professionals
    """
    
    return {
        'demographics': ['race_', 'gender_', 'age_'],
        'admission': ['admission_type_', 'admission_source_', 'discharge_disposition_'],
        'diagnoses': ['diag_1_group_', 'diag_2_group_', 'diag_3_group_'],
        'medications': ['num_medications', 'metformin', 'insulin', 'diabetesMed'],
        'utilization': ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                       'number_outpatient', 'number_emergency', 'number_inpatient'],
        'labs': ['A1Cresult_', 'max_glu_serum_']
    }


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_diabetes_data
    
    df = load_diabetes_data()
    X, y, info = preprocess_data(df)
    
    print("\nPreprocessing Info:")
    for key, value in info.items():
        print(f"{key}: {value}")
