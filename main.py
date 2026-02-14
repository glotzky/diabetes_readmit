#!/usr/bin/env python3
"""
Hospital Readmission Prediction Pipeline
=========================================

Healthcare Analytics Portfolio Project
Predicting 30-Day Hospital Readmission for Diabetic Patients

This script orchestrates the complete ML pipeline:
1. Data Loading from UCI ML Repository
2. Preprocessing (missing values, ICD-9 grouping, encoding)
3. Exploratory Data Analysis with visualizations
4. XGBoost model training with class imbalance handling
5. Model evaluation (metrics, ROC, confusion matrix)
6. SHAP explainability analysis
7. Model serialization for deployment

Clinical Goal:
--------------
Enable hospitals to identify high-risk patients at discharge, allowing
implementation of Transition of Care interventions (follow-up calls,
home visits, care management) to reduce preventable readmissions.

Target Metrics:
- Recall > 0.75 (catch most high-risk patients)
- AUC-ROC > 0.70 (good discrimination ability)
- Precision > 0.50 (avoid excessive false alarms)

Usage:
------
    python main.py                    # Run full pipeline
    python main.py --skip-eda         # Skip EDA visualizations
    python main.py --skip-shap        # Skip SHAP analysis (faster)
    python main.py --tune             # Run hyperparameter tuning

Author: Healthcare Analytics Portfolio
Dataset: Diabetes 130-US Hospitals (UCI ML Repository)
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_diabetes_data, save_processed_data
from src.preprocessing import preprocess_data
from src.eda import run_eda, generate_eda_report
from src.model import train_xgboost_model, save_model, tune_hyperparameters
from src.evaluation import evaluate_model, explain_with_shap
from src.prediction import predict_readmission, SAMPLE_PATIENT


def print_header():
    """Print pipeline header."""
    
    header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    HOSPITAL READMISSION PREDICTION PIPELINE                   ║
║                                                                               ║
║              Healthcare Analytics: 30-Day Readmission for Diabetic Patients   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(header)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def print_section(title: str):
    """Print section separator."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main(
    skip_eda: bool = False,
    skip_shap: bool = False,
    tune_params: bool = False,
    save_data: bool = True
):
    """
    Run the complete readmission prediction pipeline.
    
    Parameters
    ----------
    skip_eda : bool
        Skip EDA visualization generation.
    skip_shap : bool
        Skip SHAP explainability analysis.
    tune_params : bool
        Perform hyperparameter tuning.
    save_data : bool
        Save processed data to CSV.
    """
    
    print_header()
    
    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    print_section("STEP 1: DATA LOADING")
    
    print("Loading Diabetes 130-US Hospitals dataset from UCI ML Repository...")
    print("This dataset contains 10 years of clinical care at 130 US hospitals.\n")
    
    df = load_diabetes_data(use_ucimlrepo=True)
    
    print(f"\nLoaded {len(df):,} patient encounters with {df.shape[1]} features")
    
    # =========================================================================
    # STEP 2: PREPROCESSING
    # =========================================================================
    print_section("STEP 2: DATA PREPROCESSING")
    
    print("Preprocessing steps:")
    print("  1. Replace '?' placeholders with NaN")
    print("  2. Create binary target (readmitted <30 days)")
    print("  3. Group 700+ ICD-9 codes into 9 clinical categories")
    print("  4. One-hot encode categorical variables")
    print("  5. Handle remaining missing values\n")
    
    X, y, preprocessing_info = preprocess_data(df)
    
    # Save processed data
    if save_data:
        save_processed_data(
            X.assign(target=y),
            "data/processed_data.csv",
            "processed features and target"
        )
    
    # =========================================================================
    # STEP 3: EXPLORATORY DATA ANALYSIS
    # =========================================================================
    if not skip_eda:
        print_section("STEP 3: EXPLORATORY DATA ANALYSIS")
        
        print("Generating EDA visualizations...")
        print("  - Readmission distribution chart")
        print("  - Feature correlation heatmap")
        print("  - Key feature distributions by outcome")
        print("  - Class imbalance summary\n")
        
        eda_results = run_eda(X, y, output_dir="outputs/eda")
        eda_report = generate_eda_report(X, y, output_path="outputs/eda/eda_report.txt")
    else:
        print_section("STEP 3: EXPLORATORY DATA ANALYSIS (SKIPPED)")
        print("Use --skip-eda=False to generate EDA visualizations")
    
    # =========================================================================
    # STEP 4: HYPERPARAMETER TUNING (OPTIONAL)
    # =========================================================================
    hyperparameters = None
    
    if tune_params:
        print_section("STEP 4: HYPERPARAMETER TUNING")
        
        print("Performing randomized search for optimal hyperparameters...")
        print("This may take 5-10 minutes...\n")
        
        from sklearn.model_selection import train_test_split
        X_train_tune, _, y_train_tune, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        best_params, best_score = tune_hyperparameters(
            X_train_tune, y_train_tune,
            n_iter=20,
            cv_folds=3
        )
        
        hyperparameters = best_params
        print(f"\nBest cross-validation recall: {best_score:.3f}")
    else:
        print_section("STEP 4: HYPERPARAMETER TUNING (SKIPPED)")
        print("Using default clinical-optimized hyperparameters")
        print("Use --tune flag to perform hyperparameter search")
    
    # =========================================================================
    # STEP 5: MODEL TRAINING
    # =========================================================================
    print_section("STEP 5: MODEL TRAINING")
    
    print("Training XGBoost classifier with:")
    print("  - 80/20 train-test split (stratified)")
    print("  - Class imbalance handling via scale_pos_weight")
    print("  - 5-fold cross-validation")
    print("  - Early stopping on test set\n")
    
    model, X_train, y_train, X_test, y_test, training_info = train_xgboost_model(
        X, y,
        hyperparameters=hyperparameters,
        cross_validate=True
    )
    
    # Save the feature columns for prediction
    training_info['feature_columns'] = X.columns.tolist()
    
    # Save model
    model_path = save_model(
        model,
        output_path="outputs/model/xgboost_readmission_model.joblib",
        training_info=training_info
    )
    
    # =========================================================================
    # STEP 6: MODEL EVALUATION
    # =========================================================================
    print_section("STEP 6: MODEL EVALUATION")
    
    print("Evaluating model performance on held-out test set...")
    print("Generating metrics and visualizations:\n")
    
    eval_results = evaluate_model(
        model, X_test, y_test,
        output_dir="outputs/evaluation"
    )
    
    # Print target achievement summary
    print("\n" + "-"*50)
    print("TARGET METRIC ACHIEVEMENT:")
    print("-"*50)
    
    metrics = [
        ('Recall', eval_results['recall'], 0.75, '>'),
        ('AUC-ROC', eval_results['auc_roc'], 0.70, '>'),
        ('Precision', eval_results['precision'], 0.50, '>')
    ]
    
    for name, value, target, direction in metrics:
        achieved = value > target if direction == '>' else value < target
        status = "✓" if achieved else "✗"
        print(f"  {status} {name}: {value:.3f} (Target: {direction}{target})")
    
    # =========================================================================
    # STEP 7: SHAP EXPLAINABILITY
    # =========================================================================
    if not skip_shap:
        print_section("STEP 7: SHAP EXPLAINABILITY ANALYSIS")
        
        print("Computing SHAP values for model interpretability...")
        print("This transforms our 'black box' into an interpretable tool.\n")
        
        shap_results = explain_with_shap(
            model, X_test,
            n_samples=200,  # Sample for faster computation
            output_dir="outputs/evaluation"
        )
    else:
        print_section("STEP 7: SHAP EXPLAINABILITY (SKIPPED)")
        print("Use --skip-shap=False to generate SHAP analysis")
    
    # =========================================================================
    # STEP 8: DEPLOYMENT TEST
    # =========================================================================
    print_section("STEP 8: DEPLOYMENT READINESS TEST")
    
    print("Testing predict_readmission() function with sample patient...")
    print("-"*50)
    
    # Test prediction function
    try:
        result = predict_readmission(
            SAMPLE_PATIENT,
            model=model,
            feature_columns=X.columns.tolist(),
            return_explanation=False  # Skip SHAP for speed
        )
        
        print(f"\nSample Patient Prediction:")
        print(f"  Risk Score: {result['risk_percentage']}")
        print(f"  Risk Tier: {result['risk_tier'].upper()}")
        print(f"  Clinical Action: {result['recommendation'][:80]}...")
        print("\n✓ Prediction function working correctly!")
        
    except Exception as e:
        print(f"\n✗ Prediction test failed: {str(e)}")
    
    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    print("\n" + "="*80)
    print("  PIPELINE COMPLETE")
    print("="*80)
    
    print(f"""
Summary:
--------
• Model trained on {len(X_train):,} samples
• Tested on {len(X_test):,} samples  
• {X.shape[1]} features used

Output Files:
-------------
• Model: outputs/model/xgboost_readmission_model.joblib
• Processed Data: data/processed_data.csv
• EDA Plots: outputs/eda/
• Evaluation Plots: outputs/evaluation/

Key Results:
------------
• Recall: {eval_results['recall']:.3f} (Target: >0.75)
• AUC-ROC: {eval_results['auc_roc']:.3f} (Target: >0.70)
• Precision: {eval_results['precision']:.3f} (Target: >0.50)

Next Steps:
-----------
1. Review SHAP plots to validate clinical relevance
2. Adjust threshold if different Recall/Precision trade-off needed
3. Deploy predict_readmission() function to hospital systems
4. Monitor model performance on new data

Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)
    
    return {
        'model': model,
        'eval_results': eval_results,
        'training_info': training_info,
        'preprocessing_info': preprocessing_info
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hospital Readmission Prediction Pipeline"
    )
    
    parser.add_argument(
        '--skip-eda',
        action='store_true',
        help='Skip EDA visualization generation'
    )
    
    parser.add_argument(
        '--skip-shap',
        action='store_true',
        help='Skip SHAP explainability analysis (faster)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save processed data'
    )
    
    args = parser.parse_args()
    
    results = main(
        skip_eda=args.skip_eda,
        skip_shap=args.skip_shap,
        tune_params=args.tune,
        save_data=not args.no_save
    )
