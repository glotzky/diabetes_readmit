# Healthcare Analytics: Predicting 30-Day Hospital Readmission for Diabetic Patients

A machine learning portfolio project demonstrating end-to-end healthcare analytics, from data preprocessing to deployment-ready prediction functions.

## Clinical Context

Hospital readmissions within 30 days of discharge are a critical quality metric for healthcare systems:

- **Cost Impact**: Medicare penalizes hospitals with excess readmission rates (Hospital Readmissions Reduction Program)
- **Patient Safety**: Early readmissions often indicate gaps in care quality or discharge planning
- **Intervention Opportunity**: Identifying high-risk patients enables targeted Transition of Care programs:
  - Follow-up phone calls within 48 hours
  - Home health visits
  - Care management enrollment
  - Enhanced discharge education

This project builds a binary classification model to predict which diabetic patients are at high risk of readmission within 30 days, enabling proactive clinical intervention.

## Dataset

**Source**: [UCI Machine Learning Repository - Diabetes 130-US Hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

| Attribute | Value |
|-----------|-------|
| Instances | 101,766 patient encounters |
| Features | 47+ clinical and demographic variables |
| Time Period | 1999-2008 |
| Hospitals | 130 US hospitals |
| Target | 30-day readmission (binary) |

### Key Features

- **Demographics**: Age, gender, race
- **Admission Details**: Admission type, discharge disposition, time in hospital
- **Clinical**: Number of procedures, lab tests, medications
- **Diagnoses**: Primary, secondary, and additional diagnoses (ICD-9 codes)
- **Diabetes-specific**: A1C result, glucose serum, diabetic medications

## Project Structure

```
project1/
├── main.py                     # Complete ML pipeline orchestration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── processed_data.csv      # Preprocessed dataset (generated)
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # Data loading from UCI Repository
│   ├── preprocessing.py       # Data cleaning & feature engineering
│   ├── eda.py                 # Exploratory Data Analysis
│   ├── model.py               # XGBoost model training
│   ├── evaluation.py          # Metrics, ROC, SHAP analysis
│   └── prediction.py          # Deployment-ready prediction function
└── outputs/
    ├── eda/                   # EDA visualizations
    │   ├── readmission_distribution.png
    │   ├── correlation_heatmap.png
    │   ├── feature_distributions.png
    │   └── eda_report.txt
    ├── evaluation/            # Model evaluation outputs
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   ├── precision_recall_curve.png
    │   ├── shap_feature_importance.png
    │   └── shap_summary_plot.png
    └── model/                 # Saved model artifacts
        ├── xgboost_readmission_model.joblib
        └── xgboost_readmission_model.info.joblib
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd project1

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run the Complete Pipeline

```bash
python main.py
```

This will:
1. Download the dataset from UCI ML Repository
2. Preprocess data (handle missing values, group ICD-9 codes, encode features)
3. Generate EDA visualizations
4. Train XGBoost model with class imbalance handling
5. Evaluate model performance
6. Generate SHAP explainability analysis
7. Test deployment prediction function

### Command Line Options

```bash
# Skip EDA visualizations (faster)
python main.py --skip-eda

# Skip SHAP analysis (faster)
python main.py --skip-shap

# Run hyperparameter tuning
python main.py --tune

# All options
python main.py --skip-eda --skip-shap --tune
```

## Key Technical Components

### 1. Data Preprocessing

- **Missing Value Handling**: Replace '?' placeholders with NaN, impute with appropriate strategies
- **ICD-9 Code Grouping**: Consolidate 700+ diagnosis codes into 9 clinical categories:
  - Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other
- **Feature Encoding**: One-hot encoding for categorical variables
- **Feature Engineering**: Clinical groupings based on medical domain knowledge

### 2. Class Imbalance Handling

The dataset exhibits significant class imbalance (~11% readmission rate). We address this using:

```python
scale_pos_weight = n_negative / n_positive  # ~8:1 ratio
```

This parameter in XGBoost gives more weight to the minority class (readmissions), improving Recall without explicit oversampling.

### 3. Model: XGBoost Classifier

**Why XGBoost for Healthcare?**
- Handles mixed feature types naturally
- Robust to missing values
- Built-in feature importance
- Supports class weighting
- Excellent performance on tabular data

**Default Hyperparameters** (clinically optimized):
```python
{
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'min_child_weight': 5,
    'scale_pos_weight': ~8.0  # Calculated from data
}
```

### 4. Model Evaluation

| Metric | Target | Clinical Significance |
|--------|--------|----------------------|
| **Recall** | > 0.75 | Catch most high-risk patients (minimize False Negatives) |
| **AUC-ROC** | > 0.70 | Overall discrimination ability |
| **Precision** | > 0.50 | Avoid overwhelming care teams with false alarms |

**Why Recall > Precision?**

In healthcare, missing a high-risk patient (False Negative) is more costly than a false alarm (False Positive):
- False Negative: Patient doesn't receive needed intervention → potential adverse outcome
- False Positive: Extra follow-up call → minor resource expenditure

### 5. SHAP Explainability

SHAP (SHapley Additive exPlanations) transforms the "black box" model into an interpretable tool:

- **Global Feature Importance**: Which features drive predictions overall?
- **Individual Explanations**: Why was THIS patient flagged as high-risk?
- **Clinical Validation**: Do model insights align with medical knowledge?

Example interpretation:
> "This patient has a 67% readmission risk. Top risk factors: 3 prior inpatient visits (+0.15), 2 emergency visits (+0.08), 22 medications (+0.06)"

## Usage: Prediction Function

### Single Patient Prediction

```python
from src.prediction import predict_readmission

patient_data = {
    'age': '[60-70)',
    'gender': 'Female',
    'time_in_hospital': 7,
    'num_medications': 18,
    'number_inpatient': 2,
    'number_emergency': 1,
    'diag_1': '428',  # Heart failure
    'diabetesMed': 'Yes',
    # ... other features
}

result = predict_readmission(patient_data)

print(f"Risk Score: {result['risk_percentage']}")  # "42.3%"
print(f"Risk Tier: {result['risk_tier']}")          # "high"
print(f"Action: {result['recommendation']}")        # Clinical guidance
```

### Batch Prediction

```python
from src.prediction import PatientPredictor
import pandas as pd

# Initialize predictor
predictor = PatientPredictor('outputs/model/xgboost_readmission_model.joblib')

# Predict for multiple patients
patients_df = pd.read_csv('new_patients.csv')
results = predictor.predict_batch(patients_df)

# Stratify by risk tier
stratified = predictor.stratify_population(patients_df)
# Returns: {'low': df, 'moderate': df, 'high': df, 'very_high': df}
```

### Risk Tier Actions

| Tier | Threshold | Recommended Action |
|------|-----------|-------------------|
| Low | < 15% | Standard discharge protocol |
| Moderate | 15-30% | Enhanced discharge education |
| High | 30-50% | Follow-up call within 48-72 hours |
| Very High | > 50% | Care management program referral |

## Results

### Model Performance (Example)

```
Target Metric Achievement:
--------------------------
✓ Recall: 0.78 (Target: >0.75)
✓ AUC-ROC: 0.72 (Target: >0.70)
✓ Precision: 0.53 (Target: >0.50)
```

### Top Predictive Features

1. `number_inpatient` - Prior hospitalizations indicate disease severity
2. `number_emergency` - Emergency visits suggest unstable condition
3. `num_medications` - Medication count reflects comorbidity burden
4. `time_in_hospital` - Longer stays indicate complex cases
5. `discharge_disposition` - Discharge destination affects follow-up care

## Future Enhancements

1. **Feature Engineering**
   - Add lab value trends
   - Incorporate social determinants of health
   - Include medication adherence data

2. **Model Improvements**
   - Ensemble with clinical risk scores (LACE, HOSPITAL)
   - Time-series modeling for repeat admissions
   - Calibration for better probability estimates

3. **Deployment**
   - REST API wrapper
   - EHR integration (FHIR/HL7)
   - Real-time alerting system

## Citation

If you use this project or the dataset, please cite:

```
Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., 
& Clore, J.N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: 
Analysis of 70,000 Clinical Database Patient Records. BioMed Research International.

UCI Machine Learning Repository:
Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals 
for Years 1999-2008 [Dataset]. https://doi.org/10.24432/C5230J
```

## License

This project is for educational and portfolio demonstration purposes. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

For questions about this project or collaboration opportunities, please reach out through GitHub issues.

---

*This project demonstrates proficiency in: Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, healthcare analytics, and machine learning best practices.*
