Predicting 30-Day Hospital Readmission (Diabetic Patients)

End-to-end machine learning project for predicting 30-day hospital readmissions using structured EHR data.

This project covers:
- Data ingestion
- Cleaning and feature engineering
- Model training (XGBoost)
- Evaluation and SHAP explainability
- Deployment-ready prediction interface

## Problem
30-day readmissions are a key healthcare quality metric. Identifying high-risk diabetic patients enables early intervention, such as follow-ups, care management, and discharge optimization.

This is framed as a binary classification problem:
Target: Readmitted within 30 days (yes/no)

## Dataset
Source: 
UCI ML Repository – Diabetes 130-US Hospitals (1999–2008)
- 101,766 patient encounters
- 47+ demographic and clinical features
- 130 US hospitals

Target: 30-day readmission

### Feature Categories

Demographics: Age, gender, race
Admission details: 
Type, discharge disposition, length of stay (LOS)
Utilization: Inpatient visits, emergency visits
Diagnoses: ICD-9 codesDiabetes-related: Labs and medications

## Project StructurePlaintextproject1/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── processed_data.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── model.py
│   ├── evaluation.py
│   └── prediction.py
└── outputs/
    ├── eda/
    ├── evaluation/
    └── model/

## Setup

# Clone the repository
git clone <repository-url>
cd project1

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### Run Pipeline

python main.py

Pipeline steps:
- Download dataset
- Clean and preprocess
- Feature engineering (ICD grouping, encoding)
- Train XGBoost classifier
- Evaluate model
- Generate SHAP explanations
- Save model artifacts

Optional flags:
  python main.py --skip-eda
  python main.py --skip-shap
  python main.py --tune

## PreprocessingReplace "?" with NaN
Impute missing valuesGroup ICD-9 codes into 9 clinical categories
One-hot encode categoricals
Domain-driven feature engineering

### ICD Groupings
To reduce complexity, 700+ diagnosis codes are mapped to:
Circulatory, Respiratory, DigestiveDiabetes, Injury, Musculoskeletal
Genitourinary, Neoplasms, Other

### Class Imbalance
The dataset contains ~11% positive class (readmitted). This is handled via the XGBoost parameter: scale_pos_weight = n_negative / n_positive

This improves recall without requiring explicit resampling.

## Model

Algorithm: XGBoost Classifier
Why: 
- Strong tabular performance
- Handles missing values natively
- Supports class weighting
- Excellent feature importance support

### Default 

ParametersJSON{
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 5,
    "scale_pos_weight": ~8
}

## Evaluation

### Example Performance
Recall: 0.78
AUC-ROC: 0.72
Precision: 0.53

Top predictive features:
number_inpatient
number_emergency
num_medications
time_in_hospital
discharge_disposition

## Explainability (SHAP)SHAP is used for:Global feature importancePer-patient explanations
Clinical validation of model logic
Example interpretation:
Prediction: 67% readmission risk
Key drivers: Prior inpatient visits, emergency visits, medication burden

## Inference
## Single Patient

from src.prediction import predict_readmission

patient_data = {
    "age": "[60-70)",
    "gender": "Female",
    "time_in_hospital": 7,
    "num_medications": 18,
    "number_inpatient": 2,
    "number_emergency": 1,
    "diag_1": "428",
    "diabetesMed": "Yes"
}

result = predict_readmission(patient_data)

print(result["risk_percentage"])
print(result["risk_tier"])
print(result["recommendation"])
### Batch PredictionPythonfrom src.prediction import PatientPredictor
import pandas as pd

predictor = PatientPredictor("outputs/model/xgboost_readmission_model.joblib")

df = pd.read_csv("new_patients.csv")
results = predictor.predict_batch(df)

risk_groups = predictor.stratify_population(df)


### Risk 

## Future Work
Add lab trend features
Incorporate social determinants of health (SDOH)
Perform model calibrationEnsemble with established clinical risk scores
REST API deployment
EHR integration

## Citation

Strack et al. (2014): Impact of HbA1c Measurement on Hospital Readmission Rates, BioMed Research International.
UCI Machine Learning Repository: Diabetes 130-US Hospitals Dataset.