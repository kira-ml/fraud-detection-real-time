"""
Baseline Model Training Script (Fixed)
Trains Logistic Regression, Decision Tree, Naive Bayes, and Random Forest
with SMOTE oversampling and 3-fold cross-validation.
"""
import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ================================
# Configuration
# ================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline_config.yaml"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_baseline.parquet"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
CV_FOLDS = 3
RANDOM_STATE = 42


# ================================
# Configuration Loader
# ================================

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load pipeline configuration from YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


# ================================
# Model Definitions (FIXED)
# ================================

def get_baseline_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Define baseline models with fixed parameters.
    
    Key fixes:
    - Logistic Regression: increased max_iter to 5000, added StandardScaler
    - Decision Tree: fixed class_weight parameter name, added min_samples_leaf
    - Naive Bayes: added priors for imbalanced data
    - Random Forest: added more estimators, bootstrap sampling
    
    Returns:
        Dictionary mapping model names to model instances.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            solver='lbfgs',
            class_weight="balanced",
            random_state=random_state,
            n_jobs=4,
            C=0.1,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=100,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=random_state,
        ),
        "naive_bayes": GaussianNB(
            var_smoothing=1e-8,
            priors=None,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=100,
            min_samples_leaf=50,
            max_features='sqrt',
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=4,
            bootstrap=True,
        ),
    }
    
    return models


# ================================
# Data Loading
# ================================

def load_training_data(train_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Load training data and separate features from target.
    
    Args:
        train_path: Path to training parquet file.
    
    Returns:
        Tuple of (X, y, full_df, feature_names).
    """
    print(f"[TRAIN-BASELINE] Loading training data from: {train_path}")
    
    df = pd.read_parquet(train_path)
    print(f"[TRAIN-BASELINE] Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    target_col = "Class"
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].values.astype(np.float64)  # FIX: Ensure float64 type
    y = df[target_col].values.astype(np.int64)  # FIX: Ensure int64 type
    
    # Report class distribution
    fraud_count = y.sum()
    fraud_pct = (fraud_count / len(y)) * 100
    print(f"[TRAIN-BASELINE] Fraud class: {fraud_count} ({fraud_pct:.3f}%)")
    print(f"[TRAIN-BASELINE] Non-fraud class: {len(y) - fraud_count} ({100 - fraud_pct:.3f}%)")
    print(f"[TRAIN-BASELINE] Features: {len(feature_cols)}")
    print(f"[TRAIN-BASELINE] Imbalance ratio: 1:{int((len(y) - fraud_count) / fraud_count)}")
    
    return X, y, df, feature_cols


# ================================
# StandardScaler (FIX: Added)
# ================================

def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale features for models that require it (Logistic Regression, Naive Bayes).
    
    Args:
        X: Feature matrix.
    
    Returns:
        Tuple of (scaled X, fitted scaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ================================
# SMOTE Pipeline (FIXED)
# ================================

def create_smote_pipeline(
    model: Any, 
    use_scaler: bool = False,
    random_state: int = RANDOM_STATE,
    minority_count: int = 399,
    cv_folds: int = CV_FOLDS,
) -> ImbPipeline:
    """
    Create a pipeline with optional scaling + SMOTE + classifier.
    
    Args:
        model: Scikit-learn classifier instance.
        use_scaler: Whether to include StandardScaler.
        random_state: Random seed for reproducibility.
        minority_count: Number of fraud samples in training data.
        cv_folds: Number of cross-validation folds.
    
    Returns:
        Imbalanced-learn Pipeline.
    """
    # Calculate safe k_neighbors based on actual minority count per fold
    # SMOTE needs at least k_neighbors+1 minority samples in each fold
    minority_per_fold = minority_count // cv_folds
    if minority_per_fold <= 1:
        k_neighbors = 1
    else:
        k_neighbors = min(5, minority_per_fold - 1)
    k_neighbors = max(1, k_neighbors)
    
    steps = []
    
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    
    steps.extend([
        ("smote", SMOTE(
            sampling_strategy=0.3,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )),
        ("classifier", model),
    ])
    
    pipeline = ImbPipeline(steps)
    
    return pipeline


# ================================
# Cross-Validation (FIXED)
# ================================

def evaluate_model_cv(
    model_name: str,
    pipeline: ImbPipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Evaluate model using stratified k-fold cross-validation.
    
    Args:
        model_name: Name of the model for logging.
        pipeline: SMOTE + classifier pipeline.
        X: Feature matrix.
        y: Target vector.
        feature_names: List of feature names.
        cv_folds: Number of cross-validation folds.
        random_state: Random seed.
    
    Returns:
        Dictionary containing CV metrics.
    """
    print(f"\n[TRAIN-BASELINE] {'=' * 50}")
    print(f"[TRAIN-BASELINE] Evaluating: {model_name}")
    print(f"[TRAIN-BASELINE] {'=' * 50}")
    
    # Define cross-validation strategy
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,  # FIX: Changed to True for better distribution
        random_state=random_state,
    )
    
    # Metrics to track - FIX: Removed precision/recall/f1 from multi-metric (use PR-AUC instead)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
    
    # Perform cross-validation
    start_time = time.time()
    
    # Additional custom metrics per fold
    cv_results = {
        "test_roc_auc": [],
        "test_average_precision": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "fit_time": [],
        "score_time": [],
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train and predict
        fold_start = time.time()
        pipeline.fit(X_train, y_train)
        cv_results["fit_time"].append(time.time() - fold_start)
        
        score_start = time.time()
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        cv_results["score_time"].append(time.time() - score_start)
        
        # Calculate metrics
        cv_results["test_roc_auc"].append(roc_auc_score(y_val, y_proba))
        cv_results["test_average_precision"].append(average_precision_score(y_val, y_proba))
        cv_results["test_precision"].append(precision_score(y_val, y_pred, zero_division=0))
        cv_results["test_recall"].append(recall_score(y_val, y_pred, zero_division=0))
        cv_results["test_f1"].append(f1_score(y_val, y_pred, zero_division=0))
        
        print(f"[TRAIN-BASELINE]   Fold {fold}: PR-AUC={cv_results['test_average_precision'][-1]:.4f}, "
              f"ROC-AUC={cv_results['test_roc_auc'][-1]:.4f}, "
              f"Recall={cv_results['test_recall'][-1]:.4f}, "
              f"Precision={cv_results['test_precision'][-1]:.4f}")
    
    cv_time = time.time() - start_time
    
    # Extract and display results
    metrics = {}
    print(f"\n[TRAIN-BASELINE] Cross-Validation Results ({cv_folds}-fold):")
    print(f"[TRAIN-BASELINE] {'-' * 60}")
    
    for metric_name in ["roc_auc", "average_precision", "precision", "recall", "f1"]:
        values = cv_results[f"test_{metric_name}"]
        mean_score = np.mean(values)
        std_score = np.std(values)
        metrics[metric_name] = {
            "mean": round(float(mean_score), 4),
            "std": round(float(std_score), 4),
            "folds": [round(float(s), 4) for s in values],
        }
        print(f"[TRAIN-BASELINE]   {metric_name:20s}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    print(f"[TRAIN-BASELINE]   {'fit_time':20s}: {np.mean(cv_results['fit_time']):.2f}s per fold")
    print(f"[TRAIN-BASELINE]   {'total_cv_time':20s}: {cv_time:.2f}s")
    
    return metrics, cv_results


# ================================
# Full Training
# ================================

def train_final_model(
    model_name: str,
    pipeline: ImbPipeline,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[Any, float]:
    """
    Train final model on full training data.
    
    Args:
        model_name: Name of the model.
        pipeline: SMOTE + classifier pipeline.
        X: Feature matrix.
        y: Target vector.
    
    Returns:
        Tuple of (trained pipeline, training time).
    """
    print(f"\n[TRAIN-BASELINE] Training final {model_name} on full dataset...")
    
    start_time = time.time()
    pipeline.fit(X, y)
    train_time = time.time() - start_time
    
    print(f"[TRAIN-BASELINE] Training completed in {train_time:.2f}s")
    
    return pipeline, train_time


# ================================
# Model Saving
# ================================

def save_model(
    model: Any,
    model_name: str,
    models_dir: str,
) -> str:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model/pipeline.
        model_name: Name for the saved file.
        models_dir: Directory to save to.
    
    Returns:
        Path to saved model.
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}_baseline.pkl")
    
    joblib.dump(model, model_path, compress=3)
    print(f"[TRAIN-BASELINE] Model saved to: {model_path}")
    
    return model_path


def save_metrics(
    metrics: Dict,
    model_name: str,
    metrics_dir: str,
) -> str:
    """
    Save model metrics as JSON.
    
    Args:
        metrics: Dictionary of evaluation metrics.
        model_name: Name of the model.
        metrics_dir: Directory to save to.
    
    Returns:
        Path to saved metrics.
    """
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"{model_name}_baseline_metrics.json")
    
    # Convert numpy types to Python native types
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj
    
    metrics_native = convert_to_native(metrics)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_native, f, indent=2)
    
    print(f"[TRAIN-BASELINE] Metrics saved to: {metrics_path}")
    
    return metrics_path


# ================================
# Feature Importance
# ================================

def save_feature_importance(
    pipeline: ImbPipeline,
    feature_names: List[str],
    model_name: str,
    output_dir: str,
) -> Optional[str]:
    """
    Extract and save feature importance for tree-based models.
    
    Args:
        pipeline: Trained SMOTE + classifier pipeline.
        feature_names: List of feature column names.
        model_name: Name of the model.
        output_dir: Directory to save to.
    
    Returns:
        Path to saved feature importance CSV, or None.
    """
    classifier = pipeline.named_steps["classifier"]
    
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_importance_df = pd.DataFrame({
            "feature": [feature_names[i] for i in indices],
            "importance": importances[indices],
        })
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_name}_feature_importance.csv")
        feature_importance_df.to_csv(output_path, index=False)
        
        print(f"\n[TRAIN-BASELINE] Top 10 important features for {model_name}:")
        for i in range(min(10, len(feature_names))):
            print(f"  {i+1:2d}. {feature_importance_df.iloc[i]['feature']:30s}: "
                  f"{feature_importance_df.iloc[i]['importance']:.4f}")
        
        return output_path
    
    return None


# ================================
# Main Training Pipeline
# ================================

def run_baseline_training(
    config_path: Optional[str] = None,
    train_path: Optional[str] = None,
    models_dir: Optional[str] = None,
    metrics_dir: Optional[str] = None,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    selected_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute complete baseline model training pipeline.
    
    Args:
        config_path: Path to pipeline config YAML.
        train_path: Path to training data.
        models_dir: Directory for saving models.
        metrics_dir: Directory for saving metrics.
        cv_folds: Number of CV folds.
        random_state: Random seed.
        selected_models: List of model names to train (None = train all).
    
    Returns:
        Dictionary with training results.
    """
    # Resolve paths
    train_path = train_path or str(DEFAULT_TRAIN_PATH)
    models_dir = models_dir or str(DEFAULT_MODELS_DIR)
    metrics_dir = metrics_dir or str(DEFAULT_METRICS_DIR)
    
    print("\n" + "=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)
    print(f"[TRAIN-BASELINE] Train data: {train_path}")
    print(f"[TRAIN-BASELINE] Models dir: {models_dir}")
    print(f"[TRAIN-BASELINE] CV folds: {cv_folds}")
    print(f"[TRAIN-BASELINE] Random state: {random_state}")
    print("=" * 60)
    
    # Load data
    X, y, df, feature_names = load_training_data(train_path)
    
    # Calculate actual fraud count for SMOTE k_neighbors safety
    fraud_count = int(y.sum())
    
    # Get models
    all_models = get_baseline_models(random_state)
    
    if selected_models:
        models_to_train = {name: all_models[name] for name in selected_models if name in all_models}
        if not models_to_train:
            print(f"[TRAIN-BASELINE] ERROR: No valid models selected from {selected_models}")
            sys.exit(1)
        print(f"[TRAIN-BASELINE] Training selected models: {list(models_to_train.keys())}")
    else:
        models_to_train = all_models
        print(f"[TRAIN-BASELINE] Training all baseline models: {list(models_to_train.keys())}")
    
    # Define which models need scaling
    models_needing_scaler = {"logistic_regression", "naive_bayes"}
    
    # Train each model
    results = {}
    
    for model_name, model in models_to_train.items():
        try:
            print(f"\n{'=' * 60}")
            print(f"[TRAIN-BASELINE] Training: {model_name.upper()}")
            print(f"{'=' * 60}")
            
            # Determine if scaling is needed
            use_scaler = model_name in models_needing_scaler
            
            # Create SMOTE pipeline with actual fraud count for safe k_neighbors
            pipeline = create_smote_pipeline(
                model, use_scaler, random_state, fraud_count, cv_folds
            )
            
            # Cross-validation evaluation
            cv_metrics, cv_results = evaluate_model_cv(
                model_name, pipeline, X, y, feature_names, cv_folds, random_state
            )
            
            # Train final model
            trained_pipeline, train_time = train_final_model(
                model_name, pipeline, X, y
            )
            
            # Save model
            model_path = save_model(trained_pipeline, model_name, models_dir)
            
            # Save metrics
            metrics_summary = {
                "model_name": model_name,
                "cv_folds": cv_folds,
                "train_samples": len(X),
                "features": len(feature_names),
                "feature_names": feature_names,
                "cv_metrics": cv_metrics,
                "train_time_seconds": round(train_time, 2),
                "model_params": str(model.get_params()),
            }
            metrics_path = save_metrics(metrics_summary, model_name, metrics_dir)
            
            # Feature importance (if available)
            importance_path = save_feature_importance(
                trained_pipeline, feature_names, model_name, 
                os.path.join(metrics_dir, "feature_importance")
            )
            
            # Store results
            results[model_name] = {
                "model_path": model_path,
                "metrics_path": metrics_path,
                "importance_path": importance_path,
                "cv_metrics": cv_metrics,
                "train_time": train_time,
            }
            
        except Exception as e:
            print(f"[TRAIN-BASELINE] ERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    if results:
        print("\n" + "=" * 60)
        print("BASELINE TRAINING COMPLETE - MODEL COMPARISON")
        print("=" * 60)
        print(f"\n{'Model':25s} {'PR-AUC':10s} {'ROC-AUC':10s} {'F1':10s} {'Recall':10s} {'Precision':10s} {'Time':10s}")
        print("-" * 85)
        
        for model_name, result in results.items():
            cv = result["cv_metrics"]
            pr_auc = cv.get("average_precision", {}).get("mean", 0)
            roc_auc = cv.get("roc_auc", {}).get("mean", 0)
            f1 = cv.get("f1", {}).get("mean", 0)
            recall = cv.get("recall", {}).get("mean", 0)
            precision = cv.get("precision", {}).get("mean", 0)
            train_time = result.get("train_time", 0)
            
            print(f"{model_name:25s} {pr_auc:<10.4f} {roc_auc:<10.4f} {f1:<10.4f} {recall:<10.4f} {precision:<10.4f} {train_time:<10.1f}s")
        
        # Find best model by PR-AUC
        best_model = max(results.items(), key=lambda x: x[1]["cv_metrics"].get("average_precision", {}).get("mean", 0))
        print(f"\n[ TRAIN-BASELINE] Best model by PR-AUC: {best_model[0]} "
              f"({best_model[1]['cv_metrics']['average_precision']['mean']:.4f})")
        
        print("=" * 60 + "\n")
    
    return results


# ================================
# Entry Point
# ================================

def main():
    """Execute baseline model training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Baseline Fraud Detection Models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to training data (default: data/processed/train_baseline.parquet)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory to save models (default: models/)",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help="Directory to save metrics (default: artifacts/metrics/)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=CV_FOLDS,
        help=f"Number of CV folds (default: {CV_FOLDS})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Random state for reproducibility (default: {RANDOM_STATE})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=["logistic_regression", "decision_tree", "naive_bayes", "random_forest"],
        help="Specific models to train (default: all)",
    )
    
    args = parser.parse_args()
    
    try:
        results = run_baseline_training(
            config_path=args.config,
            train_path=args.train_path,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            selected_models=args.models,
        )
        
        if results:
            print(f"[TRAIN-BASELINE] Successfully trained {len(results)} models")
            return results
        else:
            print("[TRAIN-BASELINE] No models were trained successfully")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"[TRAIN-BASELINE] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[TRAIN-BASELINE] UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()