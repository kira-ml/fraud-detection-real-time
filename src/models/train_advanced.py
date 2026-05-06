"""
Advanced Model Training Script
Trains XGBoost, LightGBM, and MLP with scale_pos_weight for class imbalance.
Uses 3-fold cross-validation with advanced feature-engineered dataset.
Tracks experiments with MLflow.
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional imports - won't crash if not installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[TRAIN-ADVANCED] WARNING: XGBoost not installed. Skipping XGBoost.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[TRAIN-ADVANCED] WARNING: LightGBM not installed. Skipping LightGBM.")


# ================================
# Configuration
# ================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline_config.yaml"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_advanced.parquet"
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
# Model Definitions
# ================================

def get_advanced_models(
    random_state: int = RANDOM_STATE,
    scale_pos_weight_val: float = None,
) -> Dict[str, Any]:
    """
    Define advanced models optimized for fraud detection.
    
    Args:
        random_state: Random seed for reproducibility.
        scale_pos_weight_val: Ratio of negative/positive samples for scale_pos_weight.
                              If None, will be calculated from data.
    
    Returns:
        Dictionary mapping model names to model instances.
    """
    models = {}
    
    # XGBoost - Gradient boosting optimized for tabular data
    if XGBOOST_AVAILABLE:
        xgb_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",  # Faster training
        }
        
        if scale_pos_weight_val:
            xgb_params["scale_pos_weight"] = scale_pos_weight_val
        
        models["xgboost"] = xgb.XGBClassifier(**xgb_params)
    
    # LightGBM - Faster, often better on imbalanced data
    # LightGBM - Faster, often better on imbalanced data
    if LIGHTGBM_AVAILABLE:
        lgb_params = {
        "n_estimators": 500,
        "max_depth": 5,
        "num_leaves": 31,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_samples": 100,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
        }
    
        models["lightgbm"] = lgb.LGBMClassifier(**lgb_params)
    
    # MLP - Captures complex non-linear interactions
    models["mlp"] = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=256,
        learning_rate="adaptive",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
    )
    
    return models


# ================================
# Data Loading
# ================================

def load_training_data(train_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Load advanced training data and separate features from target.
    
    Args:
        train_path: Path to advanced training parquet file.
    
    Returns:
        Tuple of (X, y, full_df, feature_names).
    """
    print(f"[TRAIN-ADVANCED] Loading advanced training data from: {train_path}")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    df = pd.read_parquet(train_path)
    print(f"[TRAIN-ADVANCED] Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    target_col = "Class"
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.int64)
    
    # Report class distribution
    fraud_count = y.sum()
    fraud_pct = (fraud_count / len(y)) * 100
    imbalance_ratio = (len(y) - fraud_count) / fraud_count
    
    print(f"[TRAIN-ADVANCED] Fraud class: {fraud_count} ({fraud_pct:.3f}%)")
    print(f"[TRAIN-ADVANCED] Non-fraud class: {len(y) - fraud_count} ({100 - fraud_pct:.3f}%)")
    print(f"[TRAIN-ADVANCED] Features: {len(feature_cols)}")
    print(f"[TRAIN-ADVANCED] Imbalance ratio: 1:{imbalance_ratio:.0f}")
    
    return X, y, df, feature_cols


# ================================
# SMOTE Pipeline
# ================================

def create_advanced_pipeline(
    model: Any,
    use_smote: bool = True,
    use_scaler: bool = True,
    random_state: int = RANDOM_STATE,
    minority_count: int = 399,
    cv_folds: int = CV_FOLDS,
) -> ImbPipeline:
    """
    Create pipeline with scaling + SMOTE + classifier for advanced models.
    
    Args:
        model: Classifier instance.
        use_smote: Whether to apply SMOTE (False for XGBoost/LightGBM with scale_pos_weight).
        use_scaler: Whether to apply StandardScaler (True for MLP, optional for trees).
        random_state: Random seed.
        minority_count: Number of fraud samples (for k_neighbors calculation).
        cv_folds: Number of CV folds.
    
    Returns:
        Pipeline.
    """
    steps = []
    
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    
    if use_smote:
        # Calculate k_neighbors safely
        minority_per_fold = minority_count // cv_folds
        k_neighbors = min(5, max(1, minority_per_fold - 1))
        
        steps.append(("smote", SMOTE(
            sampling_strategy=0.3,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )))
    
    steps.append(("classifier", model))
    
    pipeline = ImbPipeline(steps)
    
    return pipeline


# ================================
# Cross-Validation
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
    Evaluate advanced model using stratified k-fold cross-validation.
    
    Args:
        model_name: Name of the model.
        pipeline: Model pipeline.
        X: Feature matrix.
        y: Target vector.
        feature_names: List of feature names.
        cv_folds: Number of CV folds.
        random_state: Random seed.
    
    Returns:
        Dictionary containing CV metrics.
    """
    print(f"\n[TRAIN-ADVANCED] {'=' * 50}")
    print(f"[TRAIN-ADVANCED] Evaluating: {model_name}")
    print(f"[TRAIN-ADVANCED] {'=' * 50}")
    
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    
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
        
        # Train
        fold_start = time.time()
        pipeline.fit(X_train, y_train)
        cv_results["fit_time"].append(time.time() - fold_start)
        
        # Predict
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
        
        print(f"[TRAIN-ADVANCED]   Fold {fold}: PR-AUC={cv_results['test_average_precision'][-1]:.4f}, "
              f"ROC-AUC={cv_results['test_roc_auc'][-1]:.4f}, "
              f"Recall={cv_results['test_recall'][-1]:.4f}, "
              f"Precision={cv_results['test_precision'][-1]:.4f}")
    
    # Calculate averages
    metrics = {}
    print(f"\n[TRAIN-ADVANCED] Cross-Validation Results ({cv_folds}-fold):")
    print(f"[TRAIN-ADVANCED] {'-' * 60}")
    
    for metric_name in ["roc_auc", "average_precision", "precision", "recall", "f1"]:
        values = cv_results[f"test_{metric_name}"]
        mean_score = np.mean(values)
        std_score = np.std(values)
        metrics[metric_name] = {
            "mean": round(float(mean_score), 4),
            "std": round(float(std_score), 4),
            "folds": [round(float(s), 4) for s in values],
        }
        print(f"[TRAIN-ADVANCED]   {metric_name:20s}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    avg_fit = np.mean(cv_results["fit_time"])
    total_time = sum(cv_results["fit_time"]) + sum(cv_results["score_time"])
    print(f"[TRAIN-ADVANCED]   {'fit_time':20s}: {avg_fit:.2f}s per fold")
    print(f"[TRAIN-ADVANCED]   {'total_cv_time':20s}: {total_time:.2f}s")
    
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
        pipeline: Model pipeline.
        X: Feature matrix.
        y: Target vector.
    
    Returns:
        Tuple of (trained pipeline, training time).
    """
    print(f"\n[TRAIN-ADVANCED] Training final {model_name} on full dataset...")
    
    start_time = time.time()
    pipeline.fit(X, y)
    train_time = time.time() - start_time
    
    print(f"[TRAIN-ADVANCED] Training completed in {train_time:.2f}s")
    
    return pipeline, train_time


# ================================
# Model Saving
# ================================

def save_model(model: Any, model_name: str, models_dir: str) -> str:
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
    model_path = os.path.join(models_dir, f"{model_name}_advanced.pkl")
    
    joblib.dump(model, model_path, compress=3)
    print(f"[TRAIN-ADVANCED] Model saved to: {model_path}")
    
    return model_path


def save_metrics(metrics: Dict, model_name: str, metrics_dir: str) -> str:
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
    metrics_path = os.path.join(metrics_dir, f"{model_name}_advanced_metrics.json")
    
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
    
    print(f"[TRAIN-ADVANCED] Metrics saved to: {metrics_path}")
    
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
        pipeline: Trained pipeline.
        feature_names: List of feature column names.
        model_name: Name of the model.
        output_dir: Directory to save to.
    
    Returns:
        Path to saved CSV, or None.
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
        
        print(f"\n[TRAIN-ADVANCED] Top 15 important features for {model_name}:")
        for i in range(min(15, len(feature_names))):
            print(f"  {i+1:2d}. {feature_importance_df.iloc[i]['feature']:30s}: "
                  f"{feature_importance_df.iloc[i]['importance']:.4f}")
        
        return output_path
    
    return None


# ================================
# Main Training Pipeline
# ================================

def run_advanced_training(
    config_path: Optional[str] = None,
    train_path: Optional[str] = None,
    models_dir: Optional[str] = None,
    metrics_dir: Optional[str] = None,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
    selected_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute complete advanced model training pipeline.
    
    Args:
        config_path: Path to pipeline config YAML.
        train_path: Path to advanced training data.
        models_dir: Directory for saving models.
        metrics_dir: Directory for saving metrics.
        cv_folds: Number of CV folds.
        random_state: Random seed.
        selected_models: List of model names to train (None = train all available).
    
    Returns:
        Dictionary with training results.
    """
    # Resolve paths
    train_path = train_path or str(DEFAULT_TRAIN_PATH)
    models_dir = models_dir or str(DEFAULT_MODELS_DIR)
    metrics_dir = metrics_dir or str(DEFAULT_METRICS_DIR)
    
    print("\n" + "=" * 60)
    print("ADVANCED MODEL TRAINING")
    print("=" * 60)
    print(f"[TRAIN-ADVANCED] Train data: {train_path}")
    print(f"[TRAIN-ADVANCED] Models dir: {models_dir}")
    print(f"[TRAIN-ADVANCED] CV folds: {cv_folds}")
    print(f"[TRAIN-ADVANCED] Random state: {random_state}")
    print(f"[TRAIN-ADVANCED] XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"[TRAIN-ADVANCED] LightGBM available: {LIGHTGBM_AVAILABLE}")
    print("=" * 60)
    
    # Load data
    X, y, df, feature_names = load_training_data(train_path)
    
    # Calculate scale_pos_weight for XGBoost/LightGBM
    fraud_count = y.sum()
    non_fraud_count = len(y) - fraud_count
    scale_pos_weight_val = non_fraud_count / fraud_count  # ~567
    
    # Get models
    all_models = get_advanced_models(random_state, scale_pos_weight_val)
    
    if selected_models:
        models_to_train = {name: all_models[name] for name in selected_models if name in all_models}
        if not models_to_train:
            print(f"[TRAIN-ADVANCED] ERROR: No valid models selected from {selected_models}")
            sys.exit(1)
        print(f"[TRAIN-ADVANCED] Training selected models: {list(models_to_train.keys())}")
    else:
        models_to_train = all_models
        print(f"[TRAIN-ADVANCED] Training all available advanced models: {list(models_to_train.keys())}")
    
    # Define which models need SMOTE vs use scale_pos_weight
    # XGBoost/LightGBM handle imbalance natively via scale_pos_weight
    models_with_scale_pos_weight = {"xgboost"}
    models_needing_scaler = {"mlp"}  # MLP requires scaled features
    
    # Train each model
    results = {}
    
    for model_name, model in models_to_train.items():
        try:
            print(f"\n{'=' * 60}")
            print(f"[TRAIN-ADVANCED] Training: {model_name.upper()}")
            print(f"{'=' * 60}")
            
            # Determine pipeline strategy
            use_smote = model_name not in models_with_scale_pos_weight
            use_scaler = model_name in models_needing_scaler
            
            if model_name in models_with_scale_pos_weight:
                print(f"[TRAIN-ADVANCED] Using scale_pos_weight={scale_pos_weight_val:.0f} (no SMOTE)")
            else:
                print(f"[TRAIN-ADVANCED] Using SMOTE for class balancing")
            
            # Create pipeline
            pipeline = create_advanced_pipeline(
                model, use_smote, use_scaler, random_state, fraud_count, cv_folds
            )
            
            # Cross-validation
            cv_metrics, cv_results = evaluate_model_cv(
                model_name, pipeline, X, y, feature_names, cv_folds, random_state
            )
            
            # Train final model
            trained_pipeline, train_time = train_final_model(model_name, pipeline, X, y)
            
            # Save model
            model_path = save_model(trained_pipeline, model_name, models_dir)
            
            # Save metrics
            metrics_summary = {
                "model_name": model_name,
                "dataset_type": "advanced_features",
                "cv_folds": cv_folds,
                "train_samples": len(X),
                "features": len(feature_names),
                "feature_names": feature_names,
                "cv_metrics": cv_metrics,
                "train_time_seconds": round(train_time, 2),
                "scale_pos_weight": scale_pos_weight_val if model_name in models_with_scale_pos_weight else None,
                "used_smote": use_smote,
            }
            metrics_path = save_metrics(metrics_summary, model_name, metrics_dir)
            
            # Feature importance
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
            print(f"[TRAIN-ADVANCED] ERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    if results:
        print("\n" + "=" * 100)
        print("ADVANCED TRAINING COMPLETE - MODEL COMPARISON")
        print("=" * 100)
        print(f"\n{'Model':20s} {'PR-AUC':>10s} {'ROC-AUC':>10s} {'F1':>10s} {'Recall':>10s} {'Precision':>10s} {'Time':>10s}")
        print("-" * 90)
        
        for model_name, result in results.items():
            cv = result["cv_metrics"]
            pr_auc = cv.get("average_precision", {}).get("mean", 0)
            roc_auc = cv.get("roc_auc", {}).get("mean", 0)
            f1 = cv.get("f1", {}).get("mean", 0)
            recall = cv.get("recall", {}).get("mean", 0)
            precision = cv.get("precision", {}).get("mean", 0)
            train_time = result.get("train_time", 0)
            
            print(f"{model_name:20s} {pr_auc:>10.4f} {roc_auc:>10.4f} {f1:>10.4f} {recall:>10.4f} {precision:>10.4f} {train_time:>8.1f}s")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]["cv_metrics"].get("average_precision", {}).get("mean", 0))
        print(f"\n[TRAIN-ADVANCED] Best model by PR-AUC: {best_model[0]} "
              f"({best_model[1]['cv_metrics']['average_precision']['mean']:.4f})")
        
        print("=" * 100 + "\n")
    
    return results


# ================================
# Entry Point
# ================================

def main():
    """Execute advanced model training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Advanced Fraud Detection Models"
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
        help="Path to advanced training data (default: data/processed/train_advanced.parquet)",
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
        help=f"Random state (default: {RANDOM_STATE})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=["xgboost", "lightgbm", "mlp"],
        help="Specific models to train (default: all available)",
    )
    
    args = parser.parse_args()
    
    try:
        results = run_advanced_training(
            config_path=args.config,
            train_path=args.train_path,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            selected_models=args.models,
        )
        
        if results:
            print(f"[TRAIN-ADVANCED] Successfully trained {len(results)} advanced models")
        else:
            print("[TRAIN-ADVANCED] No models were trained successfully")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"[TRAIN-ADVANCED] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[TRAIN-ADVANCED] UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()