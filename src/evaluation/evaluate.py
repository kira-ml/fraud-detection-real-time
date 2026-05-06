"""
Model Evaluation Script
Evaluates trained baseline AND advanced models on their respective test sets.
Supports single model evaluation, multi-model comparison, and cross-type comparison.
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
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ================================
# Configuration
# ================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_TEST_BASELINE_PATH = PROJECT_ROOT / "data" / "processed" / "test_baseline.parquet"
DEFAULT_TEST_ADVANCED_PATH = PROJECT_ROOT / "data" / "processed" / "test_advanced.parquet"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "artifacts" / "evaluation"


# ================================
# Model Loading
# ================================

def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the .pkl model file.
    
    Returns:
        Loaded model/pipeline.
    """
    print(f"[EVALUATE] Loading model: {os.path.basename(model_path)}")
    model = joblib.load(model_path)
    return model


def find_all_models(models_dir: str) -> Tuple[List[str], List[str]]:
    """
    Find all baseline and advanced model files.
    
    Args:
        models_dir: Directory containing model files.
    
    Returns:
        Tuple of (baseline_model_paths, advanced_model_paths).
    """
    baseline_models = []
    advanced_models = []
    
    if not os.path.exists(models_dir):
        return baseline_models, advanced_models
    
    for file in os.listdir(models_dir):
        if file.endswith("_baseline.pkl"):
            baseline_models.append(os.path.join(models_dir, file))
        elif file.endswith("_advanced.pkl"):
            advanced_models.append(os.path.join(models_dir, file))
    
    return sorted(baseline_models), sorted(advanced_models)


# ================================
# Data Loading
# ================================

def load_test_data(test_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Load test data and separate features from target.
    
    Args:
        test_path: Path to test parquet file.
    
    Returns:
        Tuple of (X_test, y_test, feature_names, full_df).
    """
    print(f"[EVALUATE] Loading test data from: {test_path}")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    df = pd.read_parquet(test_path)
    print(f"[EVALUATE] Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    target_col = "Class"
    feature_cols = [col for col in df.columns if col != target_col]
    
    X_test = df[feature_cols].values.astype(np.float64)
    y_test = df[target_col].values.astype(np.int64)
    
    # Report class distribution
    fraud_count = y_test.sum()
    fraud_pct = (fraud_count / len(y_test)) * 100
    print(f"[EVALUATE] Test fraud rate: {fraud_pct:.3f}% ({fraud_count} / {len(y_test)})")
    print(f"[EVALUATE] Features: {len(feature_cols)}")
    
    return X_test, y_test, feature_cols, df


# ================================
# Evaluation Metrics
# ================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for positive class.
    
    Returns:
        Dictionary of metrics.
    """
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Standard metrics
    roc_auc = roc_auc_score(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional business metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "average_precision": round(float(avg_precision), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "specificity": round(float(specificity), 4),
        "false_positive_rate": round(float(false_positive_rate), 4),
        "false_negative_rate": round(float(false_negative_rate), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "test_samples": len(y_true),
        "fraud_caught": int(tp),
        "fraud_missed": int(fn),
        "legitimate_declined": int(fp),
    }
    
    return metrics


def evaluate_model(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_type: str,
) -> Dict[str, Any]:
    """
    Evaluate a single model on the test set.
    
    Args:
        model: Trained model/pipeline.
        model_name: Name of the model for display.
        X_test: Test features.
        y_test: Test labels.
        dataset_type: 'baseline' or 'advanced' for display.
    
    Returns:
        Dictionary of evaluation metrics.
    """
    print(f"\n[EVALUATE] {'=' * 50}")
    print(f"[EVALUATE] Evaluating: {model_name} [{dataset_type.upper()}]")
    print(f"[EVALUATE] {'=' * 50}")
    
    # Generate predictions
    start_time = time.time()
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # Add timing and type info
    metrics["inference_time_seconds"] = round(inference_time, 4)
    metrics["inference_time_ms_per_sample"] = round((inference_time / len(y_test)) * 1000, 2)
    metrics["dataset_type"] = dataset_type
    
    # Display results
    print(f"\n[EVALUATE] Test Set Performance:")
    print(f"[EVALUATE] {'-' * 40}")
    print(f"[EVALUATE]   PR-AUC:              {metrics['pr_auc']:.4f}")
    print(f"[EVALUATE]   ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f"[EVALUATE]   F1 Score:            {metrics['f1_score']:.4f}")
    print(f"[EVALUATE]   Precision:           {metrics['precision']:.4f}")
    print(f"[EVALUATE]   Recall:              {metrics['recall']:.4f}")
    print(f"[EVALUATE]   Specificity:         {metrics['specificity']:.4f}")
    print(f"\n[EVALUATE] Confusion Matrix:")
    print(f"[EVALUATE]   True Negatives:      {metrics['confusion_matrix']['true_negatives']:,}")
    print(f"[EVALUATE]   False Positives:     {metrics['confusion_matrix']['false_positives']:,}  ← Legitimate declined")
    print(f"[EVALUATE]   False Negatives:     {metrics['confusion_matrix']['false_negatives']:,}  ← Fraud missed")
    print(f"[EVALUATE]   True Positives:      {metrics['confusion_matrix']['true_positives']:,}  ← Fraud caught")
    print(f"\n[EVALUATE] Business Impact:")
    print(f"[EVALUATE]   Fraud Caught:        {metrics['fraud_caught']}/{y_test.sum()} ({metrics['recall']*100:.1f}%)")
    print(f"[EVALUATE]   Fraud Missed:        {metrics['fraud_missed']} transactions")
    print(f"[EVALUATE]   Legitimate Declined: {metrics['legitimate_declined']:,} (FPR: {metrics['false_positive_rate']*100:.2f}%)")
    print(f"[EVALUATE]   Inference Time:      {metrics['inference_time_ms_per_sample']:.2f} ms per transaction")
    
    return metrics


# ================================
# Model Comparison
# ================================

def compare_models(
    all_results: Dict[str, Dict],
    title: str = "MODEL COMPARISON (TEST SET)",
) -> None:
    """
    Display comparison table for all evaluated models.
    
    Args:
        all_results: Dictionary mapping model names to their metrics.
        title: Title for the comparison table.
    """
    if len(all_results) < 2:
        return
    
    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")
    
    # Sort by PR-AUC (descending)
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1].get("pr_auc", 0),
        reverse=True
    )
    
    # Header
    header = (f"{'Model':30s} {'Type':10s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'F1':>8s} "
              f"{'Recall':>8s} {'Precision':>8s} {'FPR':>7s} {'Time(ms)':>9s}")
    print(f"\n{header}")
    print("-" * 105)
    
    for model_name, metrics in sorted_models:
        dataset_type = metrics.get("dataset_type", "unknown")
        row = (f"{model_name:30s} "
               f"{dataset_type:10s} "
               f"{metrics.get('pr_auc', 0):>8.4f} "
               f"{metrics.get('roc_auc', 0):>8.4f} "
               f"{metrics.get('f1_score', 0):>8.4f} "
               f"{metrics.get('recall', 0):>8.4f} "
               f"{metrics.get('precision', 0):>8.4f} "
               f"{metrics.get('false_positive_rate', 0)*100:>6.2f}% "
               f"{metrics.get('inference_time_ms_per_sample', 0):>8.2f}")
        print(row)
    
    # Best model
    best_model = sorted_models[0]
    print(f"\n[EVALUATE] 🏆 Best Model: {best_model[0]} "
          f"(PR-AUC: {best_model[1]['pr_auc']:.4f}, "
          f"FPR: {best_model[1]['false_positive_rate']*100:.2f}%)")
    
    # Business recommendation
    print(f"\n[EVALUATE] 💼 Business Recommendation:")
    best_precision_model = max(sorted_models, key=lambda x: x[1].get("precision", 0))
    best_recall_model = max(sorted_models, key=lambda x: x[1].get("recall", 0))
    
    print(f"[EVALUATE]   • Best balance (PR-AUC):     {best_model[0]}")
    print(f"[EVALUATE]   • Fewest false alarms:       {best_precision_model[0]} "
          f"(Precision: {best_precision_model[1]['precision']:.4f})")
    print(f"[EVALUATE]   • Most fraud caught:         {best_recall_model[0]} "
          f"(Recall: {best_recall_model[1]['recall']:.4f})")
    
    print(f"{'=' * 100}\n")


# ================================
# Save Results
# ================================

def save_evaluation_results(
    results: Dict[str, Any],
    model_name: str,
    output_dir: str,
) -> str:
    """
    Save evaluation results to JSON.
    
    Args:
        results: Evaluation metrics dictionary.
        model_name: Name of the model.
        output_dir: Directory to save results.
    
    Returns:
        Path to saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    output_path = os.path.join(output_dir, f"{model_name}_test_metrics.json")
    
    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"[EVALUATE] Results saved to: {output_path}")
    
    return output_path


# ================================
# Main Evaluation Pipeline
# ================================

def run_evaluation(
    model_path: Optional[str] = None,
    models_dir: Optional[str] = None,
    test_baseline_path: Optional[str] = None,
    test_advanced_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    evaluate_all: bool = True,
    model_type: Optional[str] = None,  # 'baseline', 'advanced', or None for both
) -> Dict[str, Any]:
    """
    Execute the evaluation pipeline for baseline and/or advanced models.
    
    Args:
        model_path: Path to a single model file (optional).
        models_dir: Directory containing multiple models.
        test_baseline_path: Path to baseline test data.
        test_advanced_path: Path to advanced test data.
        output_dir: Directory to save results.
        evaluate_all: If True, evaluate all models found.
        model_type: 'baseline', 'advanced', or None for both.
    
    Returns:
        Dictionary with evaluation results for all models.
    """
    # Resolve paths
    models_dir = models_dir or str(DEFAULT_MODELS_DIR)
    test_baseline_path = test_baseline_path or str(DEFAULT_TEST_BASELINE_PATH)
    test_advanced_path = test_advanced_path or str(DEFAULT_TEST_ADVANCED_PATH)
    output_dir = output_dir or str(DEFAULT_METRICS_DIR)
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION (TEST SET)")
    print("=" * 60)
    
    all_results = {}
    
    # Determine which models to evaluate
    if model_path:
        # Single model
        model_files = [model_path] if os.path.exists(model_path) else []
        if not model_files:
            raise FileNotFoundError(f"Model not found: {model_path}")
    else:
        # Find all models
        baseline_files, advanced_files = find_all_models(models_dir)
        
        model_files = []
        if model_type is None or model_type == "baseline":
            model_files.extend(baseline_files)
        if model_type is None or model_type == "advanced":
            model_files.extend(advanced_files)
        
        if not evaluate_all and not model_files:
            print("[EVALUATE] No models found to evaluate.")
            return {}
    
    print(f"[EVALUATE] Found {len(model_files)} model(s) to evaluate:")
    for mf in model_files:
        print(f"[EVALUATE]   - {os.path.basename(mf)}")
    
    # Cache test data to avoid reloading
    test_data_cache = {}
    
    # Evaluate each model
    for model_file in model_files:
        try:
            # Extract model name and type
            model_name = os.path.basename(model_file).replace(".pkl", "")
            
            # Determine dataset type and load appropriate test data
            if "_baseline" in model_name:
                dataset_type = "baseline"
                if "baseline" not in test_data_cache:
                    test_data_cache["baseline"] = load_test_data(test_baseline_path)
                X_test, y_test, feature_names, _ = test_data_cache["baseline"]
            elif "_advanced" in model_name:
                dataset_type = "advanced"
                if "advanced" not in test_data_cache:
                    test_data_cache["advanced"] = load_test_data(test_advanced_path)
                X_test, y_test, feature_names, _ = test_data_cache["advanced"]
            else:
                # Try to infer from filename
                print(f"[EVALUATE] WARNING: Cannot determine dataset type for {model_name}, trying both...")
                try:
                    X_test, y_test, feature_names, _ = load_test_data(test_baseline_path)
                    dataset_type = "baseline"
                except:
                    X_test, y_test, feature_names, _ = load_test_data(test_advanced_path)
                    dataset_type = "advanced"
            
            # Load model
            model = load_model(model_file)
            
            # Evaluate
            metrics = evaluate_model(model, model_name, X_test, y_test, dataset_type)
            
            # Save results
            save_evaluation_results(metrics, model_name, output_dir)
            
            all_results[model_name] = metrics
            
        except Exception as e:
            print(f"[EVALUATE] ERROR evaluating {model_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare all models if multiple evaluated
    if len(all_results) > 1:
        # Separate comparisons
        baseline_results = {k: v for k, v in all_results.items() if v.get("dataset_type") == "baseline"}
        advanced_results = {k: v for k, v in all_results.items() if v.get("dataset_type") == "advanced"}
        
        if len(baseline_results) > 1:
            compare_models(baseline_results, "BASELINE MODEL COMPARISON (TEST SET)")
        
        if len(advanced_results) > 1:
            compare_models(advanced_results, "ADVANCED MODEL COMPARISON (TEST SET)")
        
        # Cross-type comparison (baseline vs advanced)
        if baseline_results and advanced_results:
            compare_models(all_results, "FULL MODEL COMPARISON — BASELINE vs ADVANCED (TEST SET)")
    
    return all_results


# ================================
# Entry Point
# ================================

def main():
    """Execute model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Trained Models on Test Set (Baseline & Advanced)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a single model .pkl file (e.g., models/xgboost_advanced.pkl)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing model files (default: models/)",
    )
    parser.add_argument(
        "--test-baseline",
        type=str,
        default=None,
        help="Path to baseline test data (default: data/processed/test_baseline.parquet)",
    )
    parser.add_argument(
        "--test-advanced",
        type=str,
        default=None,
        help="Path to advanced test data (default: data/processed/test_advanced.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: artifacts/evaluation/)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["baseline", "advanced"],
        help="Evaluate only baseline or advanced models (default: both)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Evaluate all models in models-dir (default: True)",
    )
    
    args = parser.parse_args()
    
    try:
        results = run_evaluation(
            model_path=args.model,
            models_dir=args.models_dir,
            test_baseline_path=args.test_baseline,
            test_advanced_path=args.test_advanced,
            output_dir=args.output_dir,
            evaluate_all=args.all,
            model_type=args.type,
        )
        
        if results:
            print(f"[EVALUATE] ✅ Evaluation complete. {len(results)} model(s) evaluated.")
        else:
            print("[EVALUATE] ⚠️ No models were evaluated.")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"[EVALUATE] ❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[EVALUATE] ❌ UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()