# Predicting 30-Day Hospital Readmission (Diabetic Patients)

End-to-end machine learning project for predicting 30-day hospital readmissions using structured EHR data.

This project covers:
- Data ingestion
- Cleaning and feature engineering
- Model training (XGBoost)
- Evaluation and SHAP explainability
- Deployment-ready prediction interface

## Problem

30-day readmissions are a key healthcare quality metric. Identifying high-risk diabetic patients enables early intervention, such as follow-ups, care management, and discharge optimization.

This is framed as a **binary classification problem**:

**Target**: Readmitted within 30 days (yes/no)

## Dataset

**Source**: [UCI ML Repository - Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

- 101,766 patient encounters
- 47+ demographic and clinical features
- 130 US hospitals

**Target**: 30-day readmission

### Feature Categories

| Category | Features |
|----------|----------|
| Demographics | Age, gender, race |
| Admission Details | Type, discharge disposition, length of stay (LOS) |
| Utilization | Inpatient visits, emergency visits |
| Diagnoses | ICD-9 codes |
| Diabetes-related | Labs and medications |

## Project Structure

```
project1/
├── main.py                  # Main pipeline script
├── dashboard.py             # Streamlit CDSS dashboard
├── requirements.txt         # Dependencies
├── README.md
├── data/
│   └── processed_data.csv   # Preprocessed dataset
├── src/
│   ├── data_loader.py       # Data loading from UCI
│   ├── preprocessing.py     # Data cleaning and encoding
│   ├── eda.py               # Exploratory data analysis
│   ├── model.py             # XGBoost training
│   ├── evaluation.py        # Metrics and SHAP
│   └── prediction.py        # Inference functions
└── outputs/
    ├── eda/                 # EDA visualizations
    ├── evaluation/          # Model evaluation plots
    └── model/               # Saved model artifacts
```

## Setup

```bash
# Clone the repository
git clone <repository-url>
cd project1

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Pipeline

```bash
python main.py
```

**Pipeline steps:**
1. Download dataset
2. Clean and preprocess
3. Feature engineering (ICD grouping, encoding)
4. Train XGBoost classifier
5. Evaluate model
6. Generate SHAP explanations
7. Save model artifacts

**Optional flags:**

```bash
python main.py --skip-eda    # Skip EDA visualizations
python main.py --skip-shap   # Skip SHAP analysis
python main.py --tune        # Run hyperparameter tuning
```

### Run Dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`

## Preprocessing

- Replace "?" with NaN
- Impute missing values
- Group ICD-9 codes into 9 clinical categories
- One-hot encode categoricals
- Domain-driven feature engineering

### ICD-9 Groupings

To reduce complexity, 700+ diagnosis codes are mapped to 9 categories:

| Category | Description |
|----------|-------------|
| Circulatory | Heart disease, stroke |
| Respiratory | Pneumonia, COPD |
| Digestive | GI disorders |
| Diabetes | Diabetes-specific codes |
| Injury | Trauma, poisoning |
| Musculoskeletal | Arthritis, back problems |
| Genitourinary | Kidney disease |
| Neoplasms | Cancer |
| Other | All other codes |

### Class Imbalance

The dataset contains ~11% positive class (readmitted). This is handled via the XGBoost parameter:

```
scale_pos_weight = n_negative / n_positive
```

This improves recall without requiring explicit resampling.

## Model

**Algorithm**: XGBoost Classifier

**Why XGBoost?**
- Strong tabular performance
- Handles missing values natively
- Supports class weighting
- Excellent feature importance support

### Default Parameters

```json
{
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 5,
    "scale_pos_weight": 8
}
```

## Evaluation

### Target Metrics

| Metric | Target | Why |
|--------|--------|-----|
| Recall | > 0.75 | Catch most high-risk patients |
| AUC-ROC | > 0.70 | Good discrimination ability |
| Precision | > 0.50 | Avoid excessive false alarms |

### Top Predictive Features

1. `number_inpatient` - Prior hospitalizations
2. `number_emergency` - Emergency visits
3. `num_medications` - Medication burden
4. `time_in_hospital` - Length of stay
5. `discharge_disposition` - Discharge destination

## Explainability (SHAP)

SHAP is used for:
- Global feature importance
- Per-patient explanations
- Clinical validation of model logic

**Example interpretation:**
> Prediction: 67% readmission risk  
> Key drivers: Prior inpatient visits (+0.15), emergency visits (+0.08), medication burden (+0.06)

## Inference

### Single Patient

```python
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

print(result["risk_percentage"])   # "42.3%"
print(result["risk_tier"])         # "high"
print(result["recommendation"])    # Clinical guidance
```

### Batch Prediction

```python
from src.prediction import PatientPredictor
import pandas as pd

predictor = PatientPredictor("outputs/model/xgboost_readmission_model.joblib")

df = pd.read_csv("new_patients.csv")
results = predictor.predict_batch(df)

risk_groups = predictor.stratify_population(df)
```

### Risk Tiers

| Tier | Threshold | Action |
|------|-----------|--------|
| Low | < 15% | Standard discharge |
| Moderate | 15-30% | Enhanced education |
| High | 30-50% | Follow-up call within 48 hours |
| Very High | > 50% | Care management referral |

## Future Work

- Add lab trend features
- Incorporate social determinants of health (SDOH)
- Perform model calibration
- Ensemble with established clinical risk scores
- REST API deployment
- EHR integration (FHIR/HL7)

## Citation

```
Strack, B., DeShazo, J.P., Gennings, C., et al. (2014). 
Impact of HbA1c Measurement on Hospital Readmission Rates: 
Analysis of 70,000 Clinical Database Patient Records. 
BioMed Research International.

UCI Machine Learning Repository: 
Diabetes 130-US Hospitals for Years 1999-2008 [Dataset].
https://doi.org/10.24432/C5230J
```

## License

This project is for educational and portfolio demonstration purposes. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
