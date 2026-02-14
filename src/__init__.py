# Healthcare Analytics: 30-Day Hospital Readmission Prediction
# Modular Machine Learning Pipeline

from .data_loader import load_diabetes_data
from .preprocessing import preprocess_data, create_binary_target, group_icd9_codes
from .eda import run_eda, plot_correlation_heatmap, plot_readmission_distribution
from .model import train_xgboost_model, get_scale_pos_weight
from .evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    explain_with_shap
)
from .prediction import predict_readmission, PatientPredictor

__version__ = "1.0.0"
