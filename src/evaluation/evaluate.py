"""
Model Evaluation Script
Evaluates trained baseline AND advanced models on their respective test sets.
Supports single model evaluation, multi-model comparison, cross-type comparison,
and threshold sweep analysis for operational decision-making.
"""
import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from src.database.connection import get_db_session
from sqlalchemy import text

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
    
    # Store probability arrays for threshold sweep
    metrics["y_proba"] = y_proba
    metrics["y_true"] = y_test
    
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
# Threshold Sweep Analysis (NEW)
# ================================

def run_threshold_sweep(
    all_results: Dict[str, Dict],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Sweep decision thresholds for LightGBM and Logistic Regression to find
    optimal operating points for different business scenarios.
    
    Compares the precision-recall tradeoff at configurable thresholds and
    calculates the business cost for each operating point.
    
    Args:
        all_results: Dictionary of model evaluation results (must contain
                     'y_proba' and 'y_true' arrays for each model).
        output_dir: Directory to save threshold analysis JSON.
    
    Returns:
        Dictionary with threshold sweep results for each model analyzed.
    """
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP ANALYSIS")
    print("=" * 70)
    
    # Focus on the two most operationally interesting models
    target_models = ["lightgbm_advanced", "logistic_regression_baseline"]
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Assumptions for business cost calculation
    avg_fraud_loss = 500.0       # Average loss per missed fraud ($)
    avg_churn_cost = 300.0       # Customer lifetime value loss per false decline ($)
    
    sweep_results = {}
    
    # Pre-define multi-line headers outside f-strings
    hdr_fraud_caught = "Fraud\nCaught"
    hdr_false_alarms = "False\nAlarms"
    hdr_business_cost = "Business\nCost"
    
    for model_name in target_models:
        if model_name not in all_results:
            print(f"[THRESHOLD] Model '{model_name}' not found in results. Skipping.")
            continue
        
        model_data = all_results[model_name]
        y_proba = model_data.get("y_proba")
        y_true = model_data.get("y_true")
        
        if y_proba is None or y_true is None:
            print(f"[THRESHOLD] No probability data for {model_name}. Skipping.")
            continue
        
        display_name = model_name.replace("_baseline", "").replace("_advanced", "")
        total_fraud = int(y_true.sum())
        total_legit = int(len(y_true) - total_fraud)
        
        print(f"\n[THRESHOLD] Model: {display_name}")
        print(f"[THRESHOLD] Total transactions: {len(y_true):,}")
        print(f"[THRESHOLD] Fraud cases: {total_fraud}")
        print(f"[THRESHOLD] Legitimate transactions: {total_legit:,}")
        print(f"\n[THRESHOLD] {'Threshold':>10s} {'Recall':>8s} {'Precision':>10s} "
              f"{'FPR':>8s} {hdr_fraud_caught:>8s} {hdr_false_alarms:>10s} {'F1':>8s} "
              f"{hdr_business_cost:>12s}")
        print(f"[THRESHOLD] {'-' * 75}")
        
        model_thresholds = []
        
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
            
            # Business cost: missed fraud cost + churn cost from false declines
            business_cost = (fn * avg_fraud_loss) + (fp * avg_churn_cost)
            
            result = {
                "threshold": float(t),
                "recall": round(float(recall_val), 4),
                "precision": round(float(precision_val), 4),
                "fpr": round(float(fpr), 4),
                "f1_score": round(float(f1), 4),
                "fraud_caught": int(tp),
                "fraud_missed": int(fn),
                "false_alarms": int(fp),
                "true_negatives": int(tn),
                "business_cost": round(float(business_cost), 2),
            }
            model_thresholds.append(result)
            
            # Highlight recommended thresholds
            marker = ""
            if t == 0.3:
                marker = " <- Fraud wave response"
            elif t == 0.5:
                marker = " <- Balanced (default)"
            elif t == 0.7:
                marker = " <- Customer experience priority"
            
            print(f"[THRESHOLD] {t:>10.1f} {recall_val:>8.4f} {precision_val:>10.4f} "
                  f"{fpr:>8.4f} {tp:>5}/{total_fraud:>3} {fp:>10,} {f1:>8.4f} "
                  f"${business_cost:>10,.0f}{marker}")
        
        # Find best threshold by F1 score
        best_f1 = max(model_thresholds, key=lambda x: x["f1_score"])
        print(f"\n[THRESHOLD] Best F1 threshold: {best_f1['threshold']:.1f} "
              f"(F1={best_f1['f1_score']:.4f}, Recall={best_f1['recall']:.4f}, "
              f"Precision={best_f1['precision']:.4f})")
        
        # Find best threshold by business cost
        best_cost = min(model_thresholds, key=lambda x: x["business_cost"])
        print(f"[THRESHOLD] Lowest business cost: {best_cost['threshold']:.1f} "
              f"(${best_cost['business_cost']:,.0f}, Fraud caught: {best_cost['fraud_caught']}/{total_fraud}, "
              f"False alarms: {best_cost['false_alarms']:,})")
        
        sweep_results[model_name] = {
            "display_name": display_name,
            "total_fraud": total_fraud,
            "total_legitimate": total_legit,
            "thresholds": model_thresholds,
            "best_f1_threshold": best_f1["threshold"],
            "best_cost_threshold": best_cost["threshold"],
            "business_assumptions": {
                "avg_fraud_loss": avg_fraud_loss,
                "avg_churn_cost": avg_churn_cost,
            },
        }
    
    # Cross-model comparison at key thresholds
    if len(sweep_results) >= 2:
        print("\n" + "=" * 70)
        print("CROSS-MODEL COMPARISON AT KEY THRESHOLDS")
        print("=" * 70)
        
        hdr_fc2 = "Fraud\nCaught"
        hdr_fa2 = "False\nAlarms"
        
        for t in [0.3, 0.5, 0.7]:
            print(f"\n[THRESHOLD] --- Threshold = {t:.1f} ---")
            print(f"[THRESHOLD] {'Model':30s} {'Recall':>8s} {'Precision':>10s} "
                  f"{'FPR':>8s} {hdr_fc2:>8s} {hdr_fa2:>10s} {'Cost':>10s}")
            print(f"[THRESHOLD] {'-' * 80}")
            
            for model_name, data in sweep_results.items():
                thresh_data = next((th for th in data["thresholds"] if abs(th["threshold"] - t) < 0.01), None)
                if thresh_data:
                    print(f"[THRESHOLD] {data['display_name']:30s} "
                          f"{thresh_data['recall']:>8.4f} {thresh_data['precision']:>10.4f} "
                          f"{thresh_data['fpr']:>8.4f} "
                          f"{thresh_data['fraud_caught']:>5}/{data['total_fraud']:<3} "
                          f"{thresh_data['false_alarms']:>10,} "
                          f"${thresh_data['business_cost']:>9,.0f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    sweep_path = os.path.join(output_dir, "threshold_sweep_analysis.json")
    
    # Strip numpy arrays before saving
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
    
    with open(sweep_path, "w") as f:
        json.dump(convert(sweep_results), f, indent=2)
    
    print(f"\n[THRESHOLD] Threshold analysis saved to: {sweep_path}")
    
    return sweep_results

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
    
    # Strip probability arrays before saving to JSON (too large)
    results_to_save = {k: v for k, v in results.items() if k not in ("y_proba", "y_true")}
    
    output_path = os.path.join(output_dir, f"{model_name}_test_metrics.json")
    
    with open(output_path, "w") as f:
        json.dump(convert(results_to_save), f, indent=2)
    
    print(f"[EVALUATE] Results saved to: {output_path}")
    
    return output_path


def save_metrics_to_database(
    model_name: str,
    metrics: Dict[str, Any],
    dataset_type: str,
) -> bool:
    """Save evaluation metrics to monitoring_metrics table.
    
    Args:
        model_name: Name of the evaluated model.
        metrics: Dictionary of evaluation metrics.
        dataset_type: 'baseline' or 'advanced'.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        with get_db_session() as session:
            metric_rows = [
                ("pr_auc", metrics.get("pr_auc"), None),
                ("roc_auc", metrics.get("roc_auc"), None),
                ("f1_score", metrics.get("f1_score"), None),
                ("recall", metrics.get("recall"), None),
                ("precision", metrics.get("precision"), None),
                ("fpr", metrics.get("false_positive_rate"), None),
                ("fnr", metrics.get("false_negative_rate"), None),
            ]
            
            for metric_type, metric_value, baseline_value in metric_rows:
                if metric_value is not None:
                    session.execute(
                        text("""
                            INSERT INTO monitoring_metrics (
                                model_name, metric_type, metric_value,
                                baseline_value, deviation_percent,
                                is_warning, details
                            ) VALUES (
                                :name, :mtype, :mval,
                                :base, 0, false,
                                :details
                            )
                        """),
                        {
                            "name": f"{model_name}_{dataset_type}",
                            "mtype": metric_type,
                            "mval": float(metric_value),
                            "base": baseline_value,
                            "details": json.dumps({
                                "evaluation_type": "test_set",
                                "test_samples": metrics.get("test_samples"),
                                "fraud_count": metrics.get("fraud_caught", 0) + metrics.get("fraud_missed", 0),
                            })
                        }
                    )
        
        print(f"[EVALUATE] Metrics saved to monitoring_metrics for {model_name}")
        return True
    except Exception as e:
        print(f"[EVALUATE] WARNING: Failed to save metrics to DB: {e}")
        return False



# ================================
# Missed Fraud Analysis (NEW)
# ================================

def analyze_missed_fraud(
    all_results: Dict[str, Dict],
    output_dir: str,
) -> Optional[Dict]:
    """
    Profile the fraud cases that LightGBM missed.
    Compares feature distributions between caught and missed fraud.
    
    Args:
        all_results: Dictionary of model evaluation results (must contain
                     'y_proba' and 'y_true' arrays for lightgbm_advanced).
        output_dir: Directory to save analysis JSON.
    
    Returns:
        Dictionary with missed fraud analysis, or None if data unavailable.
    """
    model_name = "lightgbm_advanced"
    
    if model_name not in all_results:
        print("[MISSED-FRAUD] LightGBM results not found. Skipping.")
        return None
    
    model_data = all_results[model_name]
    y_proba = model_data.get("y_proba")
    y_true = model_data.get("y_true")
    
    if y_proba is None or y_true is None:
        print("[MISSED-FRAUD] No probability data available. Skipping.")
        return None
    
    # Load test DataFrame to get feature values
    test_path = PROJECT_ROOT / "data" / "processed" / "test_advanced.parquet"
    if not test_path.exists():
        print("[MISSED-FRAUD] Test data not found. Skipping.")
        return None
    
    df_test = pd.read_parquet(test_path)
    feature_cols = [col for col in df_test.columns if col != "Class"]
    
    # Get predictions at default threshold
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Split into missed fraud and caught fraud
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    
    missed = df_test[fn_mask]
    caught = df_test[tp_mask]
    
    n_missed = len(missed)
    n_caught = len(caught)
    total_fraud = n_missed + n_caught
    
    print("\n" + "=" * 70)
    print("MISSED FRAUD ANALYSIS")
    print("=" * 70)
    print(f"[MISSED-FRAUD] Total fraud in test: {total_fraud}")
    print(f"[MISSED-FRAUD] Caught: {n_caught} ({n_caught/total_fraud*100:.1f}%)")
    print(f"[MISSED-FRAUD] Missed: {n_missed} ({n_missed/total_fraud*100:.1f}%)")
    
    # Features to compare
    key_features = [
        "Amount_raw", "fraud_direction_score", "fraud_feature_magnitude",
        "V14", "V17", "V4", "V14_V10", "V16_V17",
        "time_since_last_txn", "hour", "txn_count_10min", "V8"
    ]
    
    # Keep only features that exist
    available_features = [f for f in key_features if f in df_test.columns]
    
    # Add fraud probabilities to missed cases
    missed_probas = y_proba[fn_mask]
    caught_probas = y_proba[tp_mask]
    
    # Build comparison
    comparison = {}
    for feat in available_features:
        comparison[feat] = {
            "missed_mean": round(float(missed[feat].mean()), 4),
            "caught_mean": round(float(caught[feat].mean()), 4),
            "missed_median": round(float(missed[feat].median()), 4),
            "caught_median": round(float(caught[feat].median()), 4),
            "missed_std": round(float(missed[feat].std()), 4),
            "caught_std": round(float(caught[feat].std()), 4),
        }
    
    # Print comparison table
    print(f"\n[MISSED-FRAUD] Feature Comparison: Missed vs Caught Fraud")
    print(f"[MISSED-FRAUD] {'Feature':30s} {'Missed Mean':>12s} {'Caught Mean':>12s} {'Difference':>12s}")
    print(f"[MISSED-FRAUD] {'-' * 70}")
    
    for feat in available_features:
        m_mean = comparison[feat]["missed_mean"]
        c_mean = comparison[feat]["caught_mean"]
        diff = m_mean - c_mean
        direction = "↑" if diff > 0 else "↓"
        print(f"[MISSED-FRAUD] {feat:30s} {m_mean:>12.4f} {c_mean:>12.4f} {diff:>11.4f} {direction}")
    
    # Model confidence on missed cases
    print(f"\n[MISSED-FRAUD] Model Confidence on Missed Fraud:")
    print(f"[MISSED-FRAUD]   Mean probability: {missed_probas.mean():.4f}")
    print(f"[MISSED-FRAUD]   Median probability: {np.median(missed_probas):.4f}")
    print(f"[MISSED-FRAUD]   Min: {missed_probas.min():.4f}, Max: {missed_probas.max():.4f}")
    
    print(f"\n[MISSED-FRAUD] Model Confidence on Caught Fraud:")
    print(f"[MISSED-FRAUD]   Mean probability: {caught_probas.mean():.4f}")
    print(f"[MISSED-FRAUD]   Median probability: {np.median(caught_probas):.4f}")
    
    # Build output
    analysis = {
        "total_fraud": total_fraud,
        "caught_count": n_caught,
        "missed_count": n_missed,
        "missed_probabilities": {
            "mean": round(float(missed_probas.mean()), 4),
            "median": round(float(np.median(missed_probas)), 4),
            "min": round(float(missed_probas.min()), 4),
            "max": round(float(missed_probas.max()), 4),
        },
        "caught_probabilities": {
            "mean": round(float(caught_probas.mean()), 4),
            "median": round(float(np.median(caught_probas)), 4),
        },
        "feature_comparison": comparison,
        "interpretation": {
            "likely_low_amount": comparison.get("Amount_raw", {}).get("missed_mean", 0) < comparison.get("Amount_raw", {}).get("caught_mean", 0),
            "likely_low_direction_score": comparison.get("fraud_direction_score", {}).get("missed_mean", 0) < comparison.get("fraud_direction_score", {}).get("caught_mean", 0),
        }
    }
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    analysis_path = os.path.join(output_dir, "missed_fraud_analysis.json")
    
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [convert(v) for v in obj]
        return obj
    
    with open(analysis_path, "w") as f:
        json.dump(convert(analysis), f, indent=2)
    
    print(f"\n[MISSED-FRAUD] Analysis saved to: {analysis_path}")
    
    return analysis



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
    run_threshold_analysis: bool = True,  # NEW: run threshold sweep
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
        run_threshold_analysis: If True, run threshold sweep after evaluation.
    
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
            # Save to database
            save_metrics_to_database(model_name, metrics, dataset_type)
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
    
    # ================================
    # Threshold Sweep
    # ================================
    if run_threshold_analysis and len(all_results) >= 2:
        sweep_results = run_threshold_sweep(all_results, output_dir)
        all_results["_threshold_sweep"] = sweep_results

    # ================================
    # Missed Fraud Analysis
    # ================================
    if len(all_results) >= 1:
        analyze_missed_fraud(all_results, output_dir)

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
    parser.add_argument(
        "--no-threshold-sweep",
        action="store_true",
        help="Skip threshold sweep analysis",
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
            run_threshold_analysis=not args.no_threshold_sweep,
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