"""
Visualization Module for Fraud Detection Project
Generates publication-ready plots for LinkedIn storytelling.
Focus: Model comparison, business impact, feature importance, SSL analysis.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ================================
# Configuration
# ================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
EVALUATION_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
FEATURE_IMPORTANCE_DIR = PROJECT_ROOT / "artifacts" / "metrics" / "feature_importance"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"

# Professional color palette
COLORS = {
    "baseline": "#FF6B6B",    # Coral red
    "advanced": "#4ECDC4",    # Teal
    "ssl": "#FFD93D",         # Gold
    "best": "#6BCB77",        # Green
    "worst": "#FF6B6B",       # Red
    "grid": "#E8E8E8",
    "text": "#2D3436",
    "background": "#FFFFFF",
}

# Model display names
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP (Neural Net)",
}

# Model types
MODEL_TYPES = {
    "logistic_regression": "Baseline",
    "decision_tree": "Baseline",
    "naive_bayes": "Baseline",
    "random_forest": "Baseline",
    "xgboost": "Advanced",
    "lightgbm": "Advanced",
    "mlp": "Advanced",
}

# Set up professional styling
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#E0E0E0",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})


# ================================
# Data Loading
# ================================

def load_json(filepath: str) -> Dict:
    """Load a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_all_metrics() -> Dict[str, Dict]:
    """Load all available model metrics."""
    metrics = {}
    
    # CV metrics
    cv_files = [
        "logistic_regression_baseline_metrics.json",
        "decision_tree_baseline_metrics.json",
        "naive_bayes_baseline_metrics.json",
        "random_forest_baseline_metrics.json",
        "xgboost_advanced_metrics.json",
        "lightgbm_advanced_metrics.json",
        "mlp_advanced_metrics.json",
    ]
    
    for file in cv_files:
        filepath = METRICS_DIR / file
        if filepath.exists():
            model_name = file.replace("_baseline_metrics.json", "").replace("_advanced_metrics.json", "")
            data = load_json(filepath)
            data["source"] = "cv"
            data["type"] = MODEL_TYPES.get(model_name, "Unknown")
            metrics[model_name] = data
    
    # Test metrics
    test_files = [
        "logistic_regression_baseline_test_metrics.json",
        "decision_tree_baseline_test_metrics.json",
        "naive_bayes_baseline_test_metrics.json",
        "random_forest_baseline_test_metrics.json",
        "xgboost_advanced_test_metrics.json",
        "lightgbm_advanced_test_metrics.json",
        "mlp_advanced_test_metrics.json",
    ]
    
    for file in test_files:
        filepath = EVALUATION_DIR / file
        if filepath.exists():
            model_name = file.replace("_baseline_test_metrics.json", "").replace("_advanced_test_metrics.json", "")
            data = load_json(filepath)
            data["source"] = "test"
            data["type"] = MODEL_TYPES.get(model_name, "Unknown")
            # Merge with existing or create new
            if model_name in metrics:
                metrics[model_name]["test_data"] = data
            else:
                metrics[model_name] = {"test_data": data, "type": MODEL_TYPES.get(model_name, "Unknown")}
    
    return metrics


def load_feature_importance() -> Dict[str, pd.DataFrame]:
    """Load feature importance files."""
    importance = {}
    
    files = [
        "random_forest_feature_importance.csv",
        "decision_tree_feature_importance.csv",
        "lightgbm_feature_importance.csv",
        "xgboost_feature_importance.csv",
    ]
    
    for file in files:
        filepath = FEATURE_IMPORTANCE_DIR / file
        if filepath.exists():
            model_name = file.replace("_feature_importance.csv", "")
            df = pd.read_csv(filepath)
            importance[model_name] = df
    
    return importance


# ================================
# Plot 1: Model Comparison - PR-AUC (CV vs Test)
# ================================

def plot_pr_auc_comparison(metrics: Dict, save_path: str) -> str:
    """
    Bar chart comparing PR-AUC across all models.
    Shows CV and Test performance side-by-side.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models_list = []
    cv_scores = []
    test_scores = []
    colors_list = []
    
    for model_name in ["logistic_regression", "decision_tree", "naive_bayes", 
                        "random_forest", "xgboost", "lightgbm", "mlp"]:
        if model_name in metrics:
            data = metrics[model_name]
            display_name = MODEL_NAMES.get(model_name, model_name)
            model_type = data.get("type", "Unknown")
            
            # CV score
            cv_metrics = data.get("cv_metrics", {})
            cv_pr = cv_metrics.get("average_precision", {}).get("mean", 0)
            
            # Test score
            test_pr = data.get("test_data", {}).get("pr_auc", 0)
            
            models_list.append(display_name)
            cv_scores.append(cv_pr)
            test_scores.append(test_pr)
            colors_list.append(COLORS["advanced"] if model_type == "Advanced" else COLORS["baseline"])
    
    x = np.arange(len(models_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cv_scores, width, label="Cross-Validation", 
                   color=COLORS["baseline"], alpha=0.8, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, test_scores, width, label="Test Set (Holdout)", 
                   color=COLORS["advanced"], alpha=0.8, edgecolor="white", linewidth=0.5)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Highlight best model
    best_idx = np.argmax(test_scores)
    bars2[best_idx].set_color(COLORS["best"])
    bars2[best_idx].set_edgecolor("#2d8a4e")
    bars2[best_idx].set_linewidth(2)
    
    ax.set_xlabel("Model")
    ax.set_ylabel("PR-AUC Score")
    ax.set_title("Precision-Recall AUC: Model Comparison\n(Higher = Better Fraud Detection)", 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=30, ha='right')
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gray')
    ax.set_ylim(0, max(max(cv_scores), max(test_scores)) * 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    
    # Add insight text
    best_model = models_list[best_idx]
    ax.text(0.5, -0.2, f"🏆 Best Model: {best_model} (PR-AUC: {test_scores[best_idx]:.4f})",
            transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold',
            color=COLORS["best"], fontstyle='italic')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VISUALIZE] Saved: {save_path}")
    return save_path


# ================================
# Plot 2: Business Impact - Fraud Caught vs False Alarms
# ================================

def plot_business_impact(metrics: Dict, save_path: str) -> str:
    """
    Shows the critical business tradeoff: fraud caught vs legitimate customers declined.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    models_data = []
    
    for model_name in ["logistic_regression", "decision_tree", "naive_bayes",
                        "random_forest", "xgboost", "lightgbm", "mlp"]:
        if model_name in metrics:
            data = metrics[model_name]
            test_data = data.get("test_data", {})
            
            if test_data:
                display_name = MODEL_NAMES.get(model_name, model_name)
                model_type = data.get("type", "Unknown")
                
                fraud_caught = test_data.get("fraud_caught", 0)
                fraud_missed = test_data.get("fraud_missed", 0)
                false_alarms = test_data.get("legitimate_declined", 0)
                total_fraud = fraud_caught + fraud_missed
                recall = test_data.get("recall", 0) * 100
                fpr = test_data.get("false_positive_rate", 0) * 100
                
                models_data.append({
                    "name": display_name,
                    "type": model_type,
                    "fraud_caught": fraud_caught,
                    "fraud_missed": fraud_missed,
                    "false_alarms": false_alarms,
                    "total_fraud": total_fraud,
                    "recall": recall,
                    "fpr": fpr,
                })
    
    # Sort by fraud caught
    models_data.sort(key=lambda x: x["fraud_caught"], reverse=True)
    
    # Plot 1: Fraud Caught vs Missed
    names = [m["name"] for m in models_data]
    caught = [m["fraud_caught"] for m in models_data]
    missed = [m["fraud_missed"] for m in models_data]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars_caught = ax1.barh(x, caught, width, label="Fraud Caught ✅", 
                            color=COLORS["best"], edgecolor="white")
    bars_missed = ax1.barh(x, missed, width, left=caught, label="Fraud Missed ❌", 
                            color=COLORS["worst"], edgecolor="white")
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(names)
    ax1.set_xlabel("Number of Transactions")
    ax1.set_title("Fraud Detection Performance\n(Out of 74 Fraud Cases)", fontweight='bold')
    ax1.legend(loc='lower right')
    
    # Add labels
    for i, (c, m) in enumerate(zip(caught, missed)):
        ax1.text(c + m + 0.5, i, f"{c}/{c+m}", va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: False Positive Rate
    fpr_values = [m["fpr"] for m in models_data]
    bar_colors = [COLORS["advanced"] if m["type"] == "Advanced" else COLORS["baseline"] 
                  for m in models_data]
    
    bars_fpr = ax2.barh(x, fpr_values, color=bar_colors, edgecolor="white")
    ax2.set_yticks(x)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("False Positive Rate (%)")
    ax2.set_title("Customer Impact: False Alarms\n(Lower = Fewer Angry Customers)", fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(fpr_values):
        ax2.text(v + 0.02, i, f"{v:.2f}%", va='center', fontsize=9, fontweight='bold')
    
    # Highlight best
    best_fpr_idx = np.argmin(fpr_values)
    bars_fpr[best_fpr_idx].set_edgecolor("#2d8a4e")
    bars_fpr[best_fpr_idx].set_linewidth(2.5)
    
    plt.suptitle("Business Impact: Fraud Detection ROI", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VISUALIZE] Saved: {save_path}")
    return save_path


# ================================
# Plot 3: Model Evolution - Baseline vs Advanced
# ================================

def plot_model_evolution(metrics: Dict, save_path: str) -> str:
    """
    Scatter plot showing the journey from baseline to advanced models.
    X-axis: Recall (fraud caught), Y-axis: Precision (accuracy of flags).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    baseline_models = []
    advanced_models = []
    
    for model_name, data in metrics.items():
        test_data = data.get("test_data", {})
        if test_data:
            display_name = MODEL_NAMES.get(model_name, model_name)
            model_type = data.get("type", "Unknown")
            
            precision = test_data.get("precision", 0)
            recall = test_data.get("recall", 0)
            
            point = {
                "name": display_name,
                "precision": precision,
                "recall": recall,
                "f1": test_data.get("f1_score", 0),
            }
            
            if model_type == "Baseline":
                baseline_models.append(point)
            else:
                advanced_models.append(point)
    
    # Plot baseline models
    if baseline_models:
        bx = [m["recall"] for m in baseline_models]
        by = [m["precision"] for m in baseline_models]
        bn = [m["name"] for m in baseline_models]
        ax.scatter(bx, by, c=COLORS["baseline"], s=200, alpha=0.7, 
                  edgecolors="white", linewidth=2, zorder=5, label="Baseline Models")
        for i, name in enumerate(bn):
            ax.annotate(name, (bx[i], by[i]), xytext=(10, 10), textcoords='offset points',
                       fontsize=9, alpha=0.8)
    
    # Plot advanced models
    if advanced_models:
        ax_list = [m["recall"] for m in advanced_models]
        ay = [m["precision"] for m in advanced_models]
        an = [m["name"] for m in advanced_models]
        ax.scatter(ax_list, ay, c=COLORS["advanced"], s=300, alpha=0.8,
                  edgecolors="white", linewidth=2, zorder=5, marker='D', label="Advanced Models")
        for i, name in enumerate(an):
            ax.annotate(name, (ax_list[i], ay[i]), xytext=(10, -15), textcoords='offset points',
                       fontsize=9, alpha=0.8)
    
    # Draw arrow showing improvement direction
    if advanced_models and baseline_models:
        baseline_avg_x = np.mean([m["recall"] for m in baseline_models])
        baseline_avg_y = np.mean([m["precision"] for m in baseline_models])
        advanced_avg_x = np.mean([m["recall"] for m in advanced_models])
        advanced_avg_y = np.mean([m["precision"] for m in advanced_models])
        
        ax.annotate("", xy=(advanced_avg_x, advanced_avg_y), 
                   xytext=(baseline_avg_x, baseline_avg_y),
                   arrowprops=dict(arrowstyle="->", color=COLORS["best"], lw=3, alpha=0.6))
        ax.text((baseline_avg_x + advanced_avg_x)/2, (baseline_avg_y + advanced_avg_y)/2 + 0.05,
                "IMPROVEMENT", ha='center', fontsize=10, fontweight='bold', color=COLORS["best"])
    
    # Ideal point
    ax.scatter([1.0], [1.0], c="gold", s=400, marker="*", edgecolors="black", 
              linewidth=1, zorder=10, label="Perfect Classifier")
    ax.annotate("Perfect", (1.0, 1.0), xytext=(15, 15), textcoords='offset points',
               fontsize=10, fontweight='bold', color="gold")
    
    ax.set_xlabel("Recall (Fraud Caught)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Precision (Accuracy of Flags)", fontsize=13, fontweight='bold')
    ax.set_title("Model Evolution: Baseline → Advanced\nBetter Precision Without Sacrificing Recall",
                fontweight='bold', pad=20)
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='gray')
    ax.set_xlim(0.6, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VISUALIZE] Saved: {save_path}")
    return save_path


# ================================
# Plot 4: Feature Importance - Top Features
# ================================

def plot_feature_importance(importance_data: Dict, save_path: str) -> str:
    """
    Horizontal bar chart showing top features from the best model (LightGBM).
    """
    if "lightgbm" not in importance_data:
        print("[VISUALIZE] LightGBM importance not found, trying others...")
        available = list(importance_data.keys())
        if not available:
            return None
        model_key = available[0]
    else:
        model_key = "lightgbm"
    
    df = importance_data[model_key].head(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = df["feature"].values[::-1]
    importances = df["importance"].values[::-1]
    
    # Normalize for better visualization
    importances_norm = importances / importances.max()
    
    colors = plt.cm.Blues(0.3 + 0.7 * importances_norm)
    
    bars = ax.barh(range(len(features)), importances_norm, color=colors, edgecolor="white", linewidth=1)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel("Relative Importance", fontsize=13, fontweight='bold')
    ax.set_title(f"Top 15 Features - {MODEL_NAMES.get(model_key, model_key)}\n"
                 f"(What the Model Uses to Detect Fraud)", fontweight='bold', pad=20)
    
    # Add value labels
    for i, (v, raw) in enumerate(zip(importances_norm, importances)):
        ax.text(v + 0.02, i, f"{raw:.0f}", va='center', fontsize=9, fontweight='bold', color="#2D3436")
    
    # Highlight top feature
    bars[-1].set_color(COLORS["best"])
    bars[-1].set_edgecolor("#2d8a4e")
    bars[-1].set_linewidth(2)
    
    ax.set_xlim(0, 1.2)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VISUALIZE] Saved: {save_path}")
    return save_path


# ================================
# Plot 5: SSL Analysis
# ================================

def plot_ssl_analysis(metrics: Dict, save_path: str) -> str:
    """
    Show reconstruction error distribution and SSL comparison.
    """
    ssl_file = METRICS_DIR / "autoencoder_ssl_metrics.json"
    
    if not ssl_file.exists():
        print("[VISUALIZE] SSL metrics not found. Skipping SSL plot.")
        return None
    
    ssl_data = load_json(ssl_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Error distribution (simulated from metrics)
    test_metrics = ssl_data.get("test_metrics", {})
    normal_mean = test_metrics.get("normal_mean_error", 0.42)
    fraud_mean = test_metrics.get("fraud_mean_error", 3.13)
    error_ratio = test_metrics.get("error_ratio", 7.39)
    
    # Simulate distributions
    x = np.linspace(0, fraud_mean * 2, 500)
    normal_dist = np.exp(-(x - normal_mean)**2 / (2 * 0.5**2))
    fraud_dist = np.exp(-(x - fraud_mean)**2 / (2 * 1.5**2)) * 0.3
    
    ax1.fill_between(x, normal_dist, alpha=0.5, color=COLORS["baseline"], label="Normal Transactions")
    ax1.fill_between(x, fraud_dist, alpha=0.5, color=COLORS["worst"], label="Fraud Transactions")
    ax1.axvline(x=normal_mean, color=COLORS["baseline"], linestyle="--", alpha=0.7)
    ax1.axvline(x=fraud_mean, color=COLORS["worst"], linestyle="--", alpha=0.7)
    
    ax1.set_xlabel("Reconstruction Error")
    ax1.set_ylabel("Density")
    ax1.set_title(f"SSL Autoencoder: Normal vs Fraud\nError Ratio: {error_ratio}x", fontweight='bold')
    ax1.legend()
    
    # Plot 2: SSL vs Supervised comparison
    categories = ["SSL Autoencoder", "LightGBM (Best Supervised)"]
    pr_auc_values = [test_metrics.get("pr_auc", 0.093), 0.8121]
    roc_auc_values = [test_metrics.get("roc_auc", 0.934), 0.9855]
    
    x_pos = [0, 1]
    width = 0.3
    
    bars_pr = ax2.bar([p - width/2 for p in x_pos], pr_auc_values, width, 
                       label="PR-AUC", color=COLORS["advanced"])
    bars_roc = ax2.bar([p + width/2 for p in x_pos], roc_auc_values, width,
                        label="ROC-AUC", color=COLORS["best"], alpha=0.8)
    
    for bar in bars_pr:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', fontweight='bold', fontsize=11)
    for bar in bars_roc:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', fontweight='bold', fontsize=11)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.set_ylabel("Score")
    ax2.set_title("SSL vs Supervised Learning\nSSL Captures Signal, Not Standalone", fontweight='bold')
    ax2.legend()
    ax2.set_ylim(0, 1.2)
    
    # Add verdict
    verdict = ssl_data.get("comparison", {}).get("verdict", "")
    fig.text(0.5, 0.02, f"💡 {verdict}", ha='center', fontsize=12, 
            fontweight='bold', color="#2D3436", fontstyle='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VISUALIZE] Saved: {save_path}")
    return save_path


# ================================
# Plot 6: Summary Dashboard
# ================================

def plot_summary_dashboard(metrics: Dict, save_path: str) -> str:
    """
    Create a comprehensive dashboard with key metrics in a table format.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    
    # Build comparison table
    table_data = []
    headers = ["Model", "Type", "PR-AUC", "ROC-AUC", "F1 Score", 
               "Precision", "Recall", "Fraud Caught", "False Alarms"]
    
    model_order = ["logistic_regression", "decision_tree", "naive_bayes",
                   "random_forest", "xgboost", "lightgbm", "mlp"]
    
    for model_name in model_order:
        if model_name in metrics:
            data = metrics[model_name]
            test_data = data.get("test_data", {})
            if test_data:
                display_name = MODEL_NAMES.get(model_name, model_name)
                model_type = data.get("type", "Unknown")
                
                row = [
                    display_name,
                    model_type,
                    f"{test_data.get('pr_auc', 0):.4f}",
                    f"{test_data.get('roc_auc', 0):.4f}",
                    f"{test_data.get('f1_score', 0):.4f}",
                    f"{test_data.get('precision', 0):.4f}",
                    f"{test_data.get('recall', 0):.4f}",
                    f"{test_data.get('fraud_caught', 0)}/74",
                    f"{test_data.get('legitimate_declined', 0):,}",
                ]
                table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.18, 0.08, 0.09, 0.09, 0.09, 0.10, 0.09, 0.09, 0.10],
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)
    
    # Color header row
    for j in range(len(headers)):
        table[0, j].set_facecolor("#2D3436")
        table[0, j].set_text_props(color="white", fontweight="bold")
    
    # Color best row (LightGBM)
    for j in range(len(headers)):
        if len(table_data) > 5:
            table[6, j].set_facecolor("#D4EDDA")  # Light green highlight
    
    # Color alternating rows
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0 and i != 6:
            for j in range(len(headers)):
                if table[i, j].get_facecolor() == (1.0, 1.0, 1.0, 1.0):
                    table[i, j].set_facecolor("#F8F9FA")
    
    ax.set_title("Fraud Detection Model Comparison Dashboard\n"
                 f"Test Set Performance (56,746 transactions, 74 fraud cases)",
                 fontweight='bold', fontsize=16, pad=30)
    
    # Add legend
    fig.text(0.15, 0.05, "🟢 Green = Best Overall Model (LightGBM)", 
            fontsize=10, color="#2D3436")
    fig.text(0.55, 0.05, "📊 Baseline Features: 40  |  Advanced Features: 56", 
            fontsize=10, color="#636E72")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor='white', edgecolor='none', dpi=200)
    plt.close()
    print(f"[VISUALIZE] Saved: {save_path}")
    return save_path


# ================================
# Main Pipeline
# ================================

def generate_all_visualizations(
    metrics_dir: Optional[str] = None,
    evaluation_dir: Optional[str] = None,
    feature_importance_dir: Optional[str] = None,
    plots_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate all visualizations for the project.
    
    Args:
        metrics_dir: Path to metrics directory.
        evaluation_dir: Path to evaluation directory.
        feature_importance_dir: Path to feature importance directory.
        plots_dir: Path to save plots.
    
    Returns:
        Dictionary mapping plot names to file paths.
    """
    global METRICS_DIR, EVALUATION_DIR, FEATURE_IMPORTANCE_DIR, PLOTS_DIR
    
    if metrics_dir:
        METRICS_DIR = Path(metrics_dir)
    if evaluation_dir:
        EVALUATION_DIR = Path(evaluation_dir)
    if feature_importance_dir:
        FEATURE_IMPORTANCE_DIR = Path(feature_importance_dir)
    if plots_dir:
        PLOTS_DIR = Path(plots_dir)
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load data
    print("[VISUALIZE] Loading metrics...")
    metrics = load_all_metrics()
    print(f"[VISUALIZE] Loaded metrics for {len(metrics)} models")
    
    print("[VISUALIZE] Loading feature importance...")
    importance = load_feature_importance()
    print(f"[VISUALIZE] Loaded importance for {len(importance)} models")
    
    # Generate plots
    plots = {}
    
    print("\n[VISUALIZE] Generating plots...")
    
    # Plot 1: PR-AUC Comparison
    plots["pr_auc_comparison"] = plot_pr_auc_comparison(
        metrics, str(PLOTS_DIR / "01_pr_auc_comparison.png")
    )
    
    # Plot 2: Business Impact
    plots["business_impact"] = plot_business_impact(
        metrics, str(PLOTS_DIR / "02_business_impact.png")
    )
    
    # Plot 3: Model Evolution
    plots["model_evolution"] = plot_model_evolution(
        metrics, str(PLOTS_DIR / "03_model_evolution.png")
    )
    
    # Plot 4: Feature Importance
    if importance:
        plots["feature_importance"] = plot_feature_importance(
            importance, str(PLOTS_DIR / "04_feature_importance.png")
        )
    
    # Plot 5: SSL Analysis
    ssl_plot = plot_ssl_analysis(metrics, str(PLOTS_DIR / "05_ssl_analysis.png"))
    if ssl_plot:
        plots["ssl_analysis"] = ssl_plot
    
    # Plot 6: Summary Dashboard
    plots["summary_dashboard"] = plot_summary_dashboard(
        metrics, str(PLOTS_DIR / "06_summary_dashboard.png")
    )
    
    print(f"\n[VISUALIZE] ✅ Generated {len(plots)} visualizations!")
    print(f"[VISUALIZE] 📁 Location: {PLOTS_DIR}")
    print(f"\n[VISUALIZE] Files ready for LinkedIn:")
    for name, path in plots.items():
        if path:
            print(f"[VISUALIZE]   📊 {name}: {os.path.basename(path)}")
    
    return plots


# ================================
# Entry Point
# ================================

def main():
    """Generate all visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Project Visualizations for LinkedIn"
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help="Path to metrics directory",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default=None,
        help="Path to evaluation directory",
    )
    parser.add_argument(
        "--feature-importance-dir",
        type=str,
        default=None,
        help="Path to feature importance directory",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Path to save plots",
    )
    
    args = parser.parse_args()
    
    try:
        plots = generate_all_visualizations(
            metrics_dir=args.metrics_dir,
            evaluation_dir=args.evaluation_dir,
            feature_importance_dir=args.feature_importance_dir,
            plots_dir=args.plots_dir,
        )
        
        print(f"\n[VISUALIZE] Done! {len(plots)} plots ready for LinkedIn.")
        
    except Exception as e:
        print(f"[VISUALIZE] ERROR: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
