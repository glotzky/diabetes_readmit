"""
XGBoost Modeling Module
=======================

Clinical Context:
-----------------
XGBoost is well-suited for healthcare prediction because:
1. Handles mixed feature types (categorical and continuous)
2. Robust to missing values
3. Can model complex non-linear relationships
4. Supports class weighting for imbalanced data
5. Feature importance is built-in for interpretability

Key Consideration: Class Imbalance
----------------------------------
In healthcare, class imbalance (more non-events than events) is the norm.
We use scale_pos_weight to:
- Give more weight to positive class (readmissions)
- Improve Recall (catch more high-risk patients)
- Reduce False Negatives (missing actual readmissions)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import joblib
from pathlib import Path
import warnings


def get_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for class imbalance handling.
    
    Parameters
    ----------
    y : pd.Series
        Binary target variable.
    
    Returns
    -------
    float
        Ratio of negative to positive samples.
    
    Clinical Context:
    -----------------
    scale_pos_weight = (# not readmitted) / (# readmitted)
    
    This makes the model treat each positive case as if it were
    scale_pos_weight cases, effectively balancing the learning.
    
    Example: If 10% are readmitted, scale_pos_weight = 9.0
    This is equivalent to 9x upsampling the minority class.
    """
    
    n_positive = y.sum()
    n_negative = len(y) - n_positive
    
    scale_weight = n_negative / n_positive
    
    print(f"Class distribution: {n_negative:,} negative, {n_positive:,} positive")
    print(f"Calculated scale_pos_weight: {scale_weight:.2f}")
    
    return scale_weight


def train_xgboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    use_scale_pos_weight: bool = True,
    hyperparameters: Optional[Dict] = None,
    cross_validate: bool = True,
    n_folds: int = 5
) -> Tuple[XGBClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict]:
    """
    Train XGBoost classifier for readmission prediction.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target variable.
    test_size : float, default=0.2
        Proportion of data for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    use_scale_pos_weight : bool, default=True
        Whether to handle class imbalance with scale_pos_weight.
    hyperparameters : dict, optional
        Custom hyperparameters for XGBoost. If None, uses clinical-optimized defaults.
    cross_validate : bool, default=True
        Whether to perform cross-validation.
    n_folds : int, default=5
        Number of folds for cross-validation.
    
    Returns
    -------
    model : XGBClassifier
        Trained XGBoost model.
    X_train, y_train : Training data.
    X_test, y_test : Test data.
    training_info : dict
        Training statistics and cross-validation results.
    
    Clinical Note:
    --------------
    Default hyperparameters are tuned for healthcare data:
    - max_depth=6: Prevents overfitting to noise
    - learning_rate=0.1: Stable convergence
    - n_estimators=200: Sufficient complexity
    - min_child_weight=5: Requires more evidence for splits
    """
    
    training_info = {}
    
    print("="*60)
    print("MODEL TRAINING: XGBoost Classifier")
    print("="*60)
    
    # Step 1: Train-Test Split
    print("\n1. Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class proportions
    )
    
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    print(f"   Training positive rate: {y_train.mean()*100:.2f}%")
    print(f"   Test positive rate: {y_test.mean()*100:.2f}%")
    
    training_info['train_size'] = len(X_train)
    training_info['test_size'] = len(X_test)
    
    # Step 2: Calculate scale_pos_weight
    scale_weight = get_scale_pos_weight(y_train) if use_scale_pos_weight else 1.0
    training_info['scale_pos_weight'] = scale_weight
    
    # Step 3: Set up hyperparameters
    print("\n2. Configuring XGBoost hyperparameters...")
    
    default_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_weight,
        'random_state': random_state,
        'n_jobs': -1,
        'eval_metric': 'auc',  # Use AUC for evaluation
        'tree_method': 'hist'  # Use histogram-based method (faster)
    }
    
    if hyperparameters:
        default_params.update(hyperparameters)
    
    # Print key parameters
    print(f"   n_estimators: {default_params['n_estimators']}")
    print(f"   max_depth: {default_params['max_depth']}")
    print(f"   learning_rate: {default_params['learning_rate']}")
    print(f"   scale_pos_weight: {default_params['scale_pos_weight']:.2f}")
    
    training_info['hyperparameters'] = default_params.copy()
    
    # Step 4: Cross-Validation (optional)
    if cross_validate:
        print(f"\n3. Performing {n_folds}-fold cross-validation...")
        
        cv_model = XGBClassifier(**default_params)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Score with multiple metrics
        cv_recall = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring='recall')
        cv_auc = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring='roc_auc')
        cv_precision = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring='precision')
        
        training_info['cv_recall'] = {'mean': cv_recall.mean(), 'std': cv_recall.std()}
        training_info['cv_auc'] = {'mean': cv_auc.mean(), 'std': cv_auc.std()}
        training_info['cv_precision'] = {'mean': cv_precision.mean(), 'std': cv_precision.std()}
        
        print(f"   CV Recall: {cv_recall.mean():.3f} (+/- {cv_recall.std()*2:.3f})")
        print(f"   CV AUC-ROC: {cv_auc.mean():.3f} (+/- {cv_auc.std()*2:.3f})")
        print(f"   CV Precision: {cv_precision.mean():.3f} (+/- {cv_precision.std()*2:.3f})")
    
    # Step 5: Train Final Model
    print("\n4. Training final model...")
    
    model = XGBClassifier(**default_params)
    
    # Fit with early stopping using eval set
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print("   Training complete!")
    
    # Step 6: Get Feature Importances
    print("\n5. Calculating feature importances...")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    training_info['feature_importance'] = feature_importance
    
    print("\n   Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['importance']:.4f} - {row['feature']}")
    
    print("\n" + "="*60)
    print("Model training complete!")
    print("="*60)
    
    return model, X_train, y_train, X_test, y_test, training_info


def save_model(
    model: XGBClassifier,
    output_path: str = "outputs/model/xgboost_readmission_model.joblib",
    training_info: Optional[Dict] = None
) -> str:
    """
    Save trained model and metadata.
    
    Parameters
    ----------
    model : XGBClassifier
        Trained model to save.
    output_path : str
        Path to save the model.
    training_info : dict, optional
        Training metadata to save alongside model.
    
    Returns
    -------
    str
        Path where model was saved.
    
    Clinical Note:
    --------------
    In production, always version your models and maintain:
    - Model file
    - Training date
    - Feature list (in exact order)
    - Preprocessing parameters
    - Performance metrics on validation data
    """
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")
    
    # Save training info if provided
    if training_info:
        info_path = output_path.with_suffix('.info.joblib')
        joblib.dump(training_info, info_path)
        print(f"Training info saved to: {info_path}")
    
    return str(output_path)


def load_model(model_path: str) -> Tuple[XGBClassifier, Optional[Dict]]:
    """
    Load a saved model and its metadata.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file.
    
    Returns
    -------
    model : XGBClassifier
        Loaded model.
    training_info : dict or None
        Training metadata if available.
    """
    
    model_path = Path(model_path)
    
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Try to load training info
    info_path = model_path.with_suffix('.info.joblib')
    training_info = None
    
    if info_path.exists():
        training_info = joblib.load(info_path)
        print(f"Training info loaded from: {info_path}")
    
    return model, training_info


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    n_iter: int = 20,
    cv_folds: int = 3,
    random_state: int = 42
) -> Tuple[Dict, float]:
    """
    Perform randomized hyperparameter search.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    param_grid : dict, optional
        Parameter distributions to search.
    n_iter : int, default=20
        Number of parameter combinations to try.
    cv_folds : int, default=3
        Number of CV folds.
    random_state : int, default=42
        Random seed.
    
    Returns
    -------
    best_params : dict
        Best hyperparameters found.
    best_score : float
        Best cross-validation score.
    
    Clinical Note:
    --------------
    We optimize for 'recall' because in healthcare:
    - Missing a high-risk patient (False Negative) is more costly
      than unnecessary follow-up (False Positive)
    - Hospitals can implement tiered intervention strategies
    """
    
    from sklearn.model_selection import RandomizedSearchCV
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2, 0.3]
        }
    
    print("Starting hyperparameter search...")
    print(f"Testing {n_iter} combinations with {cv_folds}-fold CV")
    
    scale_weight = get_scale_pos_weight(y_train)
    
    base_model = XGBClassifier(
        scale_pos_weight=scale_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='auc',
        tree_method='hist'
    )
    
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=n_iter,
        scoring='recall',  # Optimize for recall
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    print(f"\nBest Recall Score: {search.best_score_:.3f}")
    print("Best Parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    return search.best_params_, search.best_score_


if __name__ == "__main__":
    # Test model training
    from data_loader import load_diabetes_data
    from preprocessing import preprocess_data
    
    df = load_diabetes_data()
    X, y, _ = preprocess_data(df)
    
    model, X_train, y_train, X_test, y_test, info = train_xgboost_model(X, y)
    
    # Save model
    save_model(model, training_info=info)
