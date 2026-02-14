# Predicting 30-Day Hospital Readmission (Diabetic Patients)

End-to-end machine learning project for predicting 30-day hospital readmissions using structured EHR data.

This project covers:
- Data ingestion
- Cleaning + feature engineering
- Model training (XGBoost)
- Evaluation + SHAP explainability
- Deployment-ready prediction interface

---

## Problem

30-day readmissions are a key healthcare quality metric. Identifying high-risk diabetic patients enables early intervention (follow-ups, care management, discharge optimization).

This is framed as a **binary classification problem**:
- Target: readmitted within 30 days (yes/no)

---

## Dataset

**Source:** UCI ML Repository – Diabetes 130-US Hospitals (1999–2008)

- 101,766 patient encounters  
- 47+ demographic + clinical features  
- 130 US hospitals  
- Target: 30-day readmission  

Feature categories:
- Demographics (age, gender, race)
- Admission details (type, discharge disposition, LOS)
- Utilization (inpatient, emergency visits)
- Diagnoses (ICD-9 codes)
- Diabetes-related labs + medications

---

## Project Structure

project1/
├── main.py
├── requirements.txt
├── README.md
├── data/
│ └── processed_data.csv
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── eda.py
│ ├── model.py
│ ├── evaluation.py
│ └── prediction.py
└── outputs/
├── eda/
├── evaluation/
└── model/


---

## Setup
git clone <repository-url>
cd project1

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

Run Pipeline
python main.py

Pipeline steps:
Download dataset
Clean + preprocess
Feature engineering (ICD grouping, encoding)
Train XGBoost classifier
Evaluate model
Generate SHAP explanations
Save model artifacts

Optional flags:
python main.py --skip-eda
python main.py --skip-shap
python main.py --tune

Preprocessing
Replace "?" with NaN
Impute missing values
Group ICD-9 codes into 9 clinical categories
One-hot encode categoricals
Domain-driven feature engineering

ICD groupings reduce 700+ diagnosis codes into:
Circulatory
Respiratory
Digestive
Diabetes
Injury
Musculoskeletal
Genitourinary
Neoplasms
Other
Class Imbalance
~11% positive class (readmitted).

Handled via:
scale_pos_weight = n_negative / n_positive
This improves recall without explicit resampling.

Model
Algorithm: XGBoost Classifier

Why:
Strong tabular performance
Handles missing values
Supports class weighting
Good feature importance support

Default parameters:
{
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 5,
    "scale_pos_weight": ~8
}
Evaluation Targets
Metric	Goal	Rationale
Recall	>0.75	Minimize missed high-risk patients
AUC-ROC	>0.70	Overall discrimination
Precision	>0.50	Avoid alert fatigue
Recall is prioritized over precision due to clinical cost asymmetry.

Example Performance
Recall:     0.78
AUC-ROC:    0.72
Precision:  0.53

Top predictive features:
number_inpatient
number_emergency
num_medications
time_in_hospital
discharge_disposition

Explainability (SHAP)
SHAP is used for:
Global feature importance
Per-patient explanations
Clinical validation of model logic

Example interpretation:
67% readmission risk
Key drivers: prior inpatient visits, emergency visits, medication burden

Inference
Single Patient
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
Batch Prediction
from src.prediction import PatientPredictor
import pandas as pd

predictor = PatientPredictor(
    "outputs/model/xgboost_readmission_model.joblib"
)

df = pd.read_csv("new_patients.csv")
results = predictor.predict_batch(df)

risk_groups = predictor.stratify_population(df)
Risk tiers:

Tier	Threshold
Low	<15%
Moderate	15–30%
High	30–50%
Very High	>50%
Future Work
Add lab trend features

Incorporate social determinants

Model calibration
Ensemble with clinical risk scores
REST API deployment
EHR integration

Citation
Strack et al. (2014)
Impact of HbA1c Measurement on Hospital Readmission Rates
BioMed Research International

UCI Machine Learning Repository
Diabetes 130-US Hospitals Dataset

