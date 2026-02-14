"""
Model Evaluation Module
=======================

Clinical Context:
-----------------
In healthcare ML, evaluation metrics must align with clinical goals:

1. Recall (Sensitivity): Most critical metric
   - Measures: What % of actual readmissions did we catch?
   - Target: >75% (missing high-risk patients is dangerous)
   - Cost of error: Patient may not receive needed intervention

2. Precision (PPV): Resource allocation metric
   - Measures: What % of predicted readmissions actually readmit?
   - Target: >50% (avoid overwhelming care teams)
   - Cost of error: Wasted resources on low-risk patients

3. AUC-ROC: Overall discrimination ability
   - Measures: Model's ability to rank patients by risk
   - Target: >0.70 (industry standard for clinical utility)
   - Interpretation: Probability that a random positive ranks higher than negative

4. SHAP Values: Model interpretability
   - Why: Clinicians need to understand predictions
   - Use: Identify risk factors for individual patients
   - Benefit: Builds trust, enables clinical validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score
)

import shap
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    output_dir: str = "outputs/evaluation"
) -> Dict:
    """
    Comprehensive model evaluation with clinical metrics.
    
    Parameters
    ----------
    model : trained classifier
        Model with predict and predict_proba methods.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.
    threshold : float, default=0.5
        Classification threshold for predictions.
    output_dir : str
        Directory to save evaluation outputs.
    
    Returns
    -------
    dict
        Complete evaluation metrics and figures.
    
    Clinical Note:
    --------------
    Default threshold of 0.5 may not be optimal for healthcare.
    Consider lowering threshold to improve Recall at the cost of Precision.
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 1. Basic Metrics
    print("\n1. CLASSIFICATION METRICS")
    print("-"*40)
    
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['f1'] = f1_score(y_test, y_pred)
    
    print(f"   Accuracy:  {results['accuracy']:.3f}")
    print(f"   Recall:    {results['recall']:.3f}  {'✓' if results['recall'] > 0.75 else '✗'} (Target: >0.75)")
    print(f"   Precision: {results['precision']:.3f}  {'✓' if results['precision'] > 0.50 else '✗'} (Target: >0.50)")
    print(f"   F1 Score:  {results['f1']:.3f}")
    
    # 2. Classification Report
    print("\n2. DETAILED CLASSIFICATION REPORT")
    print("-"*40)
    
    report = classification_report(
        y_test, y_pred,
        target_names=['Not Readmitted', 'Readmitted <30d'],
        output_dict=True
    )
    results['classification_report'] = report
    
    print(classification_report(
        y_test, y_pred,
        target_names=['Not Readmitted', 'Readmitted <30d']
    ))
    
    # 3. AUC-ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    results['auc_roc'] = auc(fpr, tpr)
    results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    
    print(f"\n3. AUC-ROC: {results['auc_roc']:.3f}  {'✓' if results['auc_roc'] > 0.70 else '✗'} (Target: >0.70)")
    
    # 4. Generate Plots
    print("\n4. GENERATING EVALUATION PLOTS...")
    
    # Confusion Matrix
    fig_cm = plot_confusion_matrix(
        y_test, y_pred,
        save_path=output_path / "confusion_matrix.png"
    )
    results['confusion_matrix_plot'] = fig_cm
    
    # ROC Curve
    fig_roc = plot_roc_curve(
        y_test, y_pred_proba,
        save_path=output_path / "roc_curve.png"
    )
    results['roc_curve_plot'] = fig_roc
    
    # Precision-Recall Curve
    fig_pr = plot_precision_recall_curve(
        y_test, y_pred_proba,
        save_path=output_path / "precision_recall_curve.png"
    )
    results['pr_curve_plot'] = fig_pr
    
    # 5. Optimal Threshold Analysis
    print("\n5. THRESHOLD ANALYSIS")
    print("-"*40)
    optimal_threshold, threshold_metrics = find_optimal_threshold(y_test, y_pred_proba)
    results['optimal_threshold'] = optimal_threshold
    results['threshold_analysis'] = threshold_metrics
    
    print(f"   Current threshold: {threshold}")
    print(f"   Optimal threshold (max F1): {optimal_threshold:.3f}")
    
    # 6. Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    targets_met = sum([
        results['recall'] > 0.75,
        results['auc_roc'] > 0.70,
        results['precision'] > 0.50
    ])
    
    print(f"\n   Targets Met: {targets_met}/3")
    
    if results['recall'] < 0.75:
        print("   ⚠ Recall below target - consider lowering threshold or rebalancing")
    if results['auc_roc'] < 0.70:
        print("   ⚠ AUC-ROC below target - model may need additional features")
    if results['precision'] < 0.50:
        print("   ⚠ Precision below target - high false positive rate")
    
    if targets_met == 3:
        print("\n   ✓ Model meets all clinical targets!")
    
    print("="*60 + "\n")
    
    return results


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix with clinical annotations.
    
    Clinical Context:
    -----------------
    The confusion matrix reveals:
    - True Positives (TP): High-risk patients correctly identified → intervention opportunity
    - False Negatives (FN): Missed high-risk patients → potential adverse outcome
    - False Positives (FP): Low-risk flagged as high → unnecessary resource use
    - True Negatives (TN): Low-risk correctly identified → appropriate care allocation
    """
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        ax=ax,
        annot_kws={'size': 16},
        square=True
    )
    
    # Labels
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title('Confusion Matrix\nHospital Readmission Prediction', fontsize=16, fontweight='bold')
    ax.set_xticklabels(['Not Readmitted', 'Readmitted <30d'], fontsize=12)
    ax.set_yticklabels(['Not Readmitted', 'Readmitted <30d'], fontsize=12, rotation=0)
    
    # Add clinical annotations
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    annotation_text = (
        f"True Negatives: {tn:,} ({tn/total*100:.1f}%) - Correct low-risk\n"
        f"False Positives: {fp:,} ({fp/total*100:.1f}%) - Unnecessary alerts\n"
        f"False Negatives: {fn:,} ({fn/total*100:.1f}%) - MISSED high-risk ⚠\n"
        f"True Positives: {tp:,} ({tp/total*100:.1f}%) - Caught high-risk ✓"
    )
    
    fig.text(0.5, -0.05, annotation_text, ha='center', fontsize=11, 
             style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curve with AUC score.
    
    Clinical Context:
    -----------------
    The ROC curve shows trade-off between:
    - Sensitivity (True Positive Rate): Catching actual readmissions
    - Specificity (1 - False Positive Rate): Avoiding false alarms
    
    AUC Interpretation:
    - 0.5: No discrimination (random guessing)
    - 0.6-0.7: Poor discrimination
    - 0.7-0.8: Acceptable discrimination
    - 0.8-0.9: Good discrimination
    - >0.9: Excellent discrimination
    """
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='#3498db', lw=3, 
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.5)')
    
    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.3, color='#3498db')
    
    # Mark key thresholds
    for thresh_val in [0.3, 0.5, 0.7]:
        idx = np.argmin(np.abs(thresholds - thresh_val))
        ax.scatter(fpr[idx], tpr[idx], s=100, zorder=5)
        ax.annotate(f'θ={thresh_val}', xy=(fpr[idx], tpr[idx]), 
                   xytext=(fpr[idx]+0.05, tpr[idx]-0.05), fontsize=10)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    ax.set_ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=14)
    ax.set_title('ROC Curve - Readmission Prediction Model', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add interpretation
    if roc_auc >= 0.8:
        quality = "Good"
        color = "green"
    elif roc_auc >= 0.7:
        quality = "Acceptable"
        color = "orange"
    else:
        quality = "Poor"
        color = "red"
    
    ax.text(0.6, 0.2, f'Model Quality: {quality}', fontsize=14, 
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    return fig


def plot_precision_recall_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Clinical Context:
    -----------------
    PR curves are more informative than ROC for imbalanced data.
    
    In healthcare context:
    - High Precision, Low Recall: Few alerts, but most are true → under-detection
    - Low Precision, High Recall: Many alerts, but many false → alert fatigue
    - Goal: Find balance that catches most high-risk without overwhelming care team
    """
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot PR curve
    ax.plot(recall, precision, color='#e74c3c', lw=3,
            label=f'PR Curve (AP = {avg_precision:.3f})')
    
    # Baseline (proportion of positives)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=2,
               label=f'Baseline (No Skill) = {baseline:.3f}')
    
    # Fill area
    ax.fill_between(recall, precision, alpha=0.3, color='#e74c3c')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=14)
    ax.set_ylabel('Precision (PPV)', fontsize=14)
    ax.set_title('Precision-Recall Curve\nOptimizing for High-Risk Patient Detection', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add target zones
    ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, label='Precision Target (0.5)')
    ax.axvline(x=0.75, color='blue', linestyle=':', alpha=0.7, label='Recall Target (0.75)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    return fig


def find_optimal_threshold(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, Dict]:
    """
    Find optimal classification threshold.
    
    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred_proba : np.ndarray
        Predicted probabilities.
    metric : str, default='f1'
        Metric to optimize ('f1', 'recall', 'precision').
    
    Returns
    -------
    optimal_threshold : float
        Threshold that maximizes the chosen metric.
    metrics_by_threshold : dict
        Metrics at various thresholds.
    """
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    metrics = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        metrics.append({
            'threshold': thresh,
            'recall': recall_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred)
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Find optimal based on chosen metric
    optimal_idx = metrics_df[metric].idxmax()
    optimal_threshold = metrics_df.loc[optimal_idx, 'threshold']
    
    return optimal_threshold, metrics_df.to_dict('records')


def explain_with_shap(
    model,
    X_test: pd.DataFrame,
    n_samples: int = 100,
    output_dir: str = "outputs/evaluation",
    feature_names: Optional[List[str]] = None
) -> Dict:
    """
    Generate SHAP explanations for model predictions.
    
    Parameters
    ----------
    model : trained XGBoost model
        Model to explain.
    X_test : pd.DataFrame
        Test data for explanations.
    n_samples : int, default=100
        Number of samples for SHAP analysis.
    output_dir : str
        Directory to save SHAP plots.
    feature_names : list, optional
        Feature names for display.
    
    Returns
    -------
    dict
        SHAP values and importance rankings.
    
    Clinical Context:
    -----------------
    SHAP (SHapley Additive exPlanations) transforms black-box models into
    interpretable tools. For each prediction, SHAP shows:
    
    1. Which features pushed the risk UP (positive SHAP values)
    2. Which features pushed the risk DOWN (negative SHAP values)
    3. The magnitude of each feature's influence
    
    This is crucial for:
    - Clinical validation: Do predictions align with medical knowledge?
    - Patient communication: Explain risk factors to care team
    - Regulatory compliance: Demonstrate model fairness and logic
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("="*60)
    
    # Sample data for faster computation
    if len(X_test) > n_samples:
        X_sample = X_test.sample(n=n_samples, random_state=42)
    else:
        X_sample = X_test
    
    print(f"\n1. Computing SHAP values for {len(X_sample)} samples...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    results['shap_values'] = shap_values
    results['expected_value'] = explainer.expected_value
    
    # 2. Global Feature Importance
    print("\n2. Generating global feature importance plot...")
    
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    shap.summary_plot(
        shap_values, 
        X_sample,
        plot_type="bar",
        show=False,
        max_display=20
    )
    plt.title('SHAP Feature Importance\n(Impact on Readmission Risk)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_path / "shap_feature_importance.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")
    results['feature_importance_plot'] = save_path
    
    # 3. SHAP Summary Plot (Beeswarm)
    print("\n3. Generating SHAP summary plot...")
    
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=20
    )
    plt.title('SHAP Summary Plot\n(Feature Value Impact on Predictions)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_path / "shap_summary_plot.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")
    results['summary_plot'] = save_path
    
    # 4. Top Features Analysis
    print("\n4. TOP 10 MOST INFLUENTIAL FEATURES:")
    print("-"*40)
    
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_shap_value': mean_shap
    }).sort_values('mean_shap_value', ascending=False)
    
    results['feature_importance'] = feature_importance
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['mean_shap_value']:.4f} - {row['feature']}")
    
    # 5. Example Individual Explanation
    print("\n5. Generating individual patient explanation...")
    
    # Find a high-risk prediction
    predictions = model.predict_proba(X_sample)[:, 1]
    high_risk_idx = predictions.argmax()
    
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    shap.force_plot(
        explainer.expected_value,
        shap_values[high_risk_idx],
        X_sample.iloc[high_risk_idx],
        matplotlib=True,
        show=False
    )
    plt.title(f'Individual Patient Explanation (Risk Score: {predictions[high_risk_idx]:.2f})',
              fontsize=14, fontweight='bold')
    
    save_path = output_path / "shap_individual_explanation.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")
    results['individual_explanation'] = save_path
    
    # 6. Clinical Interpretation
    print("\n6. CLINICAL INTERPRETATION:")
    print("-"*40)
    
    top_features = feature_importance.head(5)['feature'].tolist()
    
    print("   Key risk factors identified by the model:")
    
    clinical_interpretations = {
        'number_inpatient': 'Prior hospitalizations indicate disease severity',
        'number_emergency': 'Emergency visits suggest unstable condition',
        'num_medications': 'Medication count reflects comorbidity burden',
        'time_in_hospital': 'Longer stays indicate complex cases',
        'num_lab_procedures': 'More tests suggest diagnostic complexity',
        'num_procedures': 'Procedure count reflects intervention intensity',
        'discharge_disposition': 'Discharge destination affects follow-up care'
    }
    
    for feat in top_features:
        for key, interpretation in clinical_interpretations.items():
            if key in feat.lower():
                print(f"   • {feat}: {interpretation}")
                break
        else:
            print(f"   • {feat}: Important predictor")
    
    print("\n" + "="*60)
    print("SHAP Analysis Complete!")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    # Test evaluation module
    from data_loader import load_diabetes_data
    from preprocessing import preprocess_data
    from model import train_xgboost_model
    
    df = load_diabetes_data()
    X, y, _ = preprocess_data(df)
    
    model, X_train, y_train, X_test, y_test, _ = train_xgboost_model(X, y)
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    # SHAP analysis
    shap_results = explain_with_shap(model, X_test)
