"""
Visualization Module for Fraud Detection Project
Generates publication-ready plots for LinkedIn storytelling.
Focus: Model comparison, business impact, feature importance, threshold analysis.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

# ================================
# Configuration
# ================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
EVALUATION_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
FEATURE_IMPORTANCE_DIR = PROJECT_ROOT / "artifacts" / "metrics" / "feature_importance"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"

# Professional color palette — colorblind-friendly
COLORS = {
    "baseline": "#D55E00",      # Vermillion
    "advanced": "#0072B2",      # Deep blue
    "lightgbm": "#009E73",      # Bluish green
    "best": "#009E73",          
    "worst": "#D55E00",         
    "grid": "#E0E0E0",
    "text": "#333333",
}

# Clean display names
MODEL_NAMES = {
    "logistic_regression": "Logistic\nRegression",
    "decision_tree": "Decision\nTree",
    "naive_bayes": "Naive\nBayes",
    "random_forest": "Random\nForest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
}

# Styling — clean, minimal, publication-ready
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": "#CCCCCC",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ================================
# Data Loading
# ================================

def load_json(filepath: str) -> Dict:
    with open(filepath, "r") as f:
        return json.load(f)


def load_all_metrics() -> Dict[str, Dict]:
    """Load CV and test metrics for all models."""
    metrics = {}

    # Load CV metrics
    for file in METRICS_DIR.glob("*_metrics.json"):
        if "test_metrics" in file.name:
            continue
        model_name = file.stem.replace("_baseline_metrics", "").replace("_advanced_metrics", "")
        data = load_json(file)
        data["source"] = "cv"
        data["type"] = "Advanced" if "advanced" in file.stem else "Baseline"
        metrics[model_name] = data

    # Load test metrics and merge
    for file in EVALUATION_DIR.glob("*_test_metrics.json"):
        model_name = file.stem.replace("_baseline_test_metrics", "").replace("_advanced_test_metrics", "")
        data = load_json(file)
        if model_name in metrics:
            metrics[model_name]["test_data"] = data
        else:
            metrics[model_name] = {
                "test_data": data,
                "type": "Advanced" if "advanced" in file.stem else "Baseline",
                "source": "test",
            }

    return metrics


def load_feature_importance() -> Dict[str, pd.DataFrame]:
    importance = {}
    for file in FEATURE_IMPORTANCE_DIR.glob("*_feature_importance.csv"):
        model_name = file.stem.replace("_feature_importance", "")
        importance[model_name] = pd.read_csv(file)
    return importance


# ================================
# Helper: model ordering
# ================================

def get_model_order(metrics: Dict) -> List[str]:
    """Return models ordered by test PR-AUC descending."""
    scored = []
    for name, data in metrics.items():
        test_pr = data.get("test_data", {}).get("pr_auc", 0)
        scored.append((name, test_pr))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored]


# ================================
# Plot 1: Precision-Recall Curves
# ================================

def plot_pr_curves(metrics: Dict, save_path: str) -> str:
    """
    Overlay precision-recall curves for all models on the same axes.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    model_order = get_model_order(metrics)
    plot_models = [m for m in model_order if "test_data" in metrics.get(m, {})]

    for i, model_name in enumerate(plot_models):
        data = metrics[model_name]
        test_data = data.get("test_data", {})
        model_type = data.get("type", "Baseline")

        precision = test_data.get("precision", 0)
        recall = test_data.get("recall", 0)
        pr_auc = test_data.get("pr_auc", 0)
        display_name = MODEL_NAMES.get(model_name, model_name).replace("\n", " ")
        display_name = display_name.replace("🏆 ", "")

        is_best = (model_name == plot_models[0])
        color = COLORS["advanced"] if model_type == "Advanced" else COLORS["baseline"]
        alpha = 1.0 if is_best else 0.6
        size = 200 if is_best else 100
        zorder = 10 if is_best else 5
        marker = "D" if model_type == "Advanced" else "o"

        ax.scatter(recall, precision, c=color, s=size, alpha=alpha,
                   edgecolors="white" if is_best else "none",
                   linewidth=2 if is_best else 0,
                   zorder=zorder, marker=marker,
                   label=f"{display_name} (PR-AUC={pr_auc:.3f})")

    # No-fraud baseline
    fraud_rate = 0.00172
    ax.axhline(y=fraud_rate, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(0.98, fraud_rate + 0.02, f"Random classifier ({fraud_rate:.3f})",
            ha='right', fontsize=8, color="gray", transform=ax.get_yaxis_transform())

    # Annotation box — "Highest PR-AUC" instead of "Best"
    best_model = plot_models[0] if plot_models else None
    if best_model:
        best_data = metrics[best_model]["test_data"]
        total_fraud = best_data.get('fraud_caught', 0) + best_data.get('fraud_missed', 0)
        ax.text(0.05, 0.15,
                f"Highest PR-AUC in test: {MODEL_NAMES.get(best_model, best_model).replace(chr(10), ' ')}\n"
                f"Detected {best_data.get('fraud_caught', '?')}/{total_fraud} fraud cases\n"
                f"{best_data.get('legitimate_declined', '?')} false alarms\n"
                f"FPR: {best_data.get('false_positive_rate', 0)*100:.2f}%",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F8F8", edgecolor="#CCCCCC", alpha=0.9))

    ax.set_xlabel("Recall (Fraud Caught)", fontweight='bold')
    ax.set_ylabel("Precision (Flag Accuracy)", fontweight='bold')
    ax.set_title("Precision-Recall: Every Model on One Plot\nHigher & further right = better fraud detection",
                 fontweight='bold', pad=15)
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#DDDDDD', fontsize=8)
    ax.set_xlim(0.7, 0.92)
    ax.set_ylim(0, 0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VISUALIZE] Saved: {save_path}")
    return save_path


# ================================
# Plot 2: Business Impact — Fraud Caught vs False Alarms
# ================================

def plot_business_impact(metrics: Dict, save_path: str) -> str:
    """
    Dual bar chart: fraud caught (left) and false positive rate (right).
    Added vertical line at 97 fraud cases.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    model_order = get_model_order(metrics)
    model_order.reverse()  # Worst to best for horizontal bars

    names = []
    caught = []
    missed = []
    fpr_vals = []
    bar_colors = []

    for model_name in model_order:
        data = metrics.get(model_name, {})
        test_data = data.get("test_data", {})
        if not test_data:
            continue
        display_name = MODEL_NAMES.get(model_name, model_name).replace("\n", " ").replace("🏆 ", "")
        model_type = data.get("type", "Baseline")

        names.append(display_name)
        c = test_data.get("fraud_caught", 0)
        m = test_data.get("fraud_missed", 0)
        caught.append(c)
        missed.append(m)
        fpr_vals.append(test_data.get("false_positive_rate", 0) * 100)
        bar_colors.append(COLORS["advanced"] if model_type == "Advanced" else COLORS["baseline"])

    y = np.arange(len(names))

    # Left: Fraud caught vs missed
    ax1.barh(y, caught, height=0.6, color=COLORS["best"], edgecolor="white", label="Fraud Caught")
    ax1.barh(y, missed, height=0.6, left=caught, color=COLORS["worst"], edgecolor="white", alpha=0.5, label="Fraud Missed")
    for i, (c, m_val) in enumerate(zip(caught, missed)):
        total = c + m_val
        ax1.text(total + 0.3, i, f"{c}/{total}", va='center', fontsize=8, fontweight='bold')
    
    # Vertical line at total fraud (97)
    total_fraud = caught[0] + missed[0] if caught and missed else 97
    ax1.axvline(x=total_fraud, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(total_fraud + 2, 0, f'Total fraud: {total_fraud}', rotation=90, va='bottom', fontsize=8, color='gray')
    
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("Transactions")
    ax1.set_title("Fraud Caught vs Missed", fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_xlim(0, max([c + m for c, m in zip(caught, missed)]) * 1.2)

    # Right: False positive rate
    best_fpr_idx = np.argmin(fpr_vals)
    for i, (v, c) in enumerate(zip(fpr_vals, bar_colors)):
        edge_color = "#2d8a4e" if i == best_fpr_idx else "white"
        edge_width = 2.5 if i == best_fpr_idx else 0.5
        ax2.barh(i, v, height=0.6, color=c, edgecolor=edge_color, linewidth=edge_width)
    for i, v in enumerate(fpr_vals):
        ax2.text(v + 0.02, i, f"{v:.2f}%", va='center', fontsize=8, fontweight='bold')
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("False Positive Rate (%)")
    ax2.set_title("Customer Impact: False Alarms", fontweight='bold')

    fig.suptitle("Business Impact: What Each Model Means for Operations",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VISUALIZE] Saved: {save_path}")
    return save_path


# ================================
# Plot 3: Feature Importance
# ================================

def plot_feature_importance(importance_data: Dict, save_path: str) -> str:
    """
    Horizontal bar chart: top 10 LightGBM features with annotation.
    """
    if "lightgbm" not in importance_data:
        available = list(importance_data.keys())
        if not available:
            return None
        model_key = available[0]
    else:
        model_key = "lightgbm"

    df = importance_data[model_key].head(10)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    features = df["feature"].values[::-1]
    importances = df["importance"].values[::-1]
    importances_norm = importances / importances.max()

    # Categorize features by color
    engineered_keywords = ["fraud_direction", "fraud_feature", "hour_sin", "time_since",
                           "txn_count", "amount_", "V17_V", "V14_V", "V12_V", "V16_V",
                           "V3_V", "std_amount", "velocity_spike"]
    colors_list = []
    for feat in features:
        is_engineered = any(kw in feat for kw in engineered_keywords)
        colors_list.append("#009E73" if is_engineered else "#999999")

    ax.barh(range(len(features)), importances_norm, color=colors_list, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Relative Importance", fontweight='bold')
    ax.set_title("What LightGBM Uses to Detect Fraud\n🟢 = Engineered features | ⚫ = Raw PCA components",
                 fontweight='bold', pad=15)
    ax.set_xlim(0, 1.15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#009E73", label="Engineered features (temporal, velocity, domain)"),
        Patch(facecolor="#999999", label="Raw PCA components"),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True,
              facecolor='white', edgecolor='#DDDDDD')

    # Annotation
    ax.text(0.02, -0.15, 
            "Note: Engineered features rank higher than raw PCA components in this model. "
            "Results may differ with other architectures.",
            transform=ax.transAxes, fontsize=8, color="#666666", ha='left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VISUALIZE] Saved: {save_path}")
    return save_path


# ================================
# Plot 4: SHAP Summary
# ================================

def plot_shap_summary(save_path: str) -> Optional[str]:
    """
    Generate SHAP beeswarm plot for the LightGBM model.
    Shows which features push predictions toward fraud vs legitimate.
    Added footnote about 500 sampled transactions.
    """
    try:
        import shap
    except ImportError:
        print("[VISUALIZE] SHAP not installed. Install with: pip install shap")
        return None
    
    model_path = PROJECT_ROOT / "models" / "lightgbm_advanced.pkl"
    test_path = PROJECT_ROOT / "data" / "processed" / "test_advanced.parquet"
    
    if not model_path.exists():
        print("[VISUALIZE] LightGBM model not found. Skipping SHAP.")
        return None
    if not test_path.exists():
        print("[VISUALIZE] Test data not found. Skipping SHAP.")
        return None
    
    print("[VISUALIZE] Loading model and data for SHAP...")
    model = joblib.load(model_path)
    df_test = pd.read_parquet(test_path)
    
    feature_cols = [col for col in df_test.columns if col != "Class"]
    X_test = df_test[feature_cols].values
    
    # Sample 500 rows for speed
    n_samples = min(500, len(X_test))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[sample_idx]
    
    print(f"[VISUALIZE] Computing SHAP on {n_samples} samples...")
    
    classifier = model.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_sample)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_cols,
        max_display=15, show=False
    )
    fig.suptitle("SHAP: What Drives LightGBM Fraud Predictions\n"
                 "Red = higher feature value | Right = pushes toward fraud",
                 fontsize=12, fontweight='bold', y=1.02)
    
    # Footnote
    ax.text(0.98, 0.02, 
            "Note: SHAP values based on 500 sampled test transactions. "
            "Positive values push toward fraud prediction.",
            transform=ax.transAxes, ha='right', fontsize=8, color="#666666")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor='white', edgecolor='none', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[VISUALIZE] Saved: {save_path}")
    return save_path


# ================================
# Plot 5: Threshold Tradeoff
# ================================

def plot_threshold_tradeoff(metrics: Dict, save_path: str) -> str:
    """
    Show business cost vs threshold tradeoff with criticism-proof framing.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    lightgbm_data = metrics.get("lightgbm", {}).get("test_data", {})
    lr_data = metrics.get("logistic_regression", {}).get("test_data", {})

    if lightgbm_data and lr_data:
        models_info = [
            {
                "name": "LightGBM\n(highest precision)",
                "recall": lightgbm_data.get("recall", 0.763),
                "precision": lightgbm_data.get("precision", 0.796),
                "fpr": lightgbm_data.get("false_positive_rate", 0.0003),
                "color": COLORS["lightgbm"],
                "marker": "D",
                "size": 250,
            },
            {
                "name": "Logistic Regression\n(highest recall)",
                "recall": lr_data.get("recall", 0.887),
                "precision": lr_data.get("precision", 0.073),
                "fpr": lr_data.get("false_positive_rate", 0.0149),
                "color": COLORS["baseline"],
                "marker": "o",
                "size": 200,
            },
        ]

        for m in models_info:
            ax.scatter(m["recall"], m["precision"], c=m["color"], s=m["size"],
                       marker=m["marker"], edgecolors="white", linewidth=2,
                       zorder=10, label=m["name"])
            ax.annotate(m["name"].replace("\n", " "),
                        (m["recall"], m["precision"]),
                        xytext=(15, -15), textcoords='offset points',
                        fontsize=9, fontweight='bold', color=m["color"])

        # Arrow showing the tradeoff direction
        ax.annotate("", xy=(lr_data.get("recall", 0.887), lr_data.get("precision", 0.073)),
                     xytext=(lightgbm_data.get("recall", 0.763), lightgbm_data.get("precision", 0.796)),
                     arrowprops=dict(arrowstyle="<->", color="#666666", lw=2, alpha=0.5,
                                     connectionstyle="arc3,rad=-0.2"))
        ax.text(0.81, 0.35, "Precision-Recall\ntradeoff", ha='center', fontsize=9,
                color="#666666", fontstyle='italic')

        # FPR annotations — factual, not boastful
        ax.text(0.05, 0.95,
                f"LightGBM: {lightgbm_data.get('legitimate_declined', 19):,} false alarms\n"
                f"Logistic Regression: {lr_data.get('legitimate_declined', 1096):,} false alarms\n"
                f"{lr_data.get('legitimate_declined', 1096) // max(lightgbm_data.get('legitimate_declined', 1), 1)}× more customer friction",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8E1", edgecolor="#FFD54F", alpha=0.9))

    ax.set_xlabel("Recall (Fraud Caught)", fontweight='bold')
    ax.set_ylabel("Precision (Flag Accuracy)", fontweight='bold')
    ax.set_title("Threshold Tuning: Choosing an Operating Point\nHigher recall catches more fraud, but lower precision means more false alarms",
                 fontweight='bold', pad=15)
    ax.set_xlim(0.7, 0.95)
    ax.set_ylim(0, 0.9)
    ax.legend(loc='lower left', fontsize=8, frameon=True, facecolor='white', edgecolor='#DDDDDD')

    # Footnote
    ax.text(0.98, 0.02, 
            "Note: Operating point should be chosen based on business priorities — "
            "no single threshold is universally 'correct'.",
            transform=ax.transAxes, ha='right', fontsize=8, color="#666666")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"[VISUALIZE] Saved: {save_path}")
    return save_path


# ================================
# Plot 6: Summary Table
# ================================

def plot_summary_dashboard(metrics: Dict, save_path: str) -> str:
    """
    Clean comparison table with business cost column.
    Added Business Cost column and footnote.
    """
    # Get actual numbers from data
    lightgbm_data = metrics.get("lightgbm", {}).get("test_data", {})
    total_fraud = lightgbm_data.get("fraud_caught", 0) + lightgbm_data.get("fraud_missed", 0)
    total_transactions = lightgbm_data.get("test_samples", 0)

    # Business cost assumptions (configurable)
    avg_fraud_loss = 500.0
    avg_churn_cost = 300.0

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.axis('off')

    headers = ["Model", "Type", "PR-AUC", "Precision", "Recall", 
               "Fraud Caught", "False Alarms", "FPR", "Business Cost"]
    table_data = []

    model_order = get_model_order(metrics)

    for model_name in model_order:
        data = metrics.get(model_name, {})
        test_data = data.get("test_data", {})
        if not test_data:
            continue
        display_name = MODEL_NAMES.get(model_name, model_name).replace("\n", " ").replace("🏆 ", "")
        model_type = data.get("type", "Unknown")

        is_best = (model_name == model_order[0])

        # Business cost calculation
        fraud_missed = test_data.get("fraud_missed", 0)
        false_alarms = test_data.get("legitimate_declined", 0)
        business_cost = (fraud_missed * avg_fraud_loss) + (false_alarms * avg_churn_cost)

        row = [
            f"★ {display_name}" if is_best else f"  {display_name}",
            model_type,
            f"{test_data.get('pr_auc', 0):.4f}",
            f"{test_data.get('precision', 0):.4f}",
            f"{test_data.get('recall', 0):.4f}",
            f"{test_data.get('fraud_caught', 0)}/{total_fraud}",
            f"{test_data.get('legitimate_declined', 0):,}",
            f"{test_data.get('false_positive_rate', 0)*100:.2f}%",
            f"${business_cost:,.0f}",
        ]
        table_data.append(row)

    if not table_data:
        print("[VISUALIZE] No data for summary dashboard")
        return None

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#333333")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=10)

    # Highlight best model row
    best_row_idx = None
    for i, row in enumerate(table_data):
        if row[0].startswith("★"):
            best_row_idx = i + 1  # +1 because row 0 is header
            break

    if best_row_idx:
        for j in range(len(headers)):
            table[best_row_idx, j].set_facecolor("#D5F5E3")
            table[best_row_idx, j].set_text_props(fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i != best_row_idx and i % 2 == 0:
            for j in range(len(headers)):
                if table[i, j].get_facecolor() == (1.0, 1.0, 1.0, 1.0):
                    table[i, j].set_facecolor("#F5F5F5")

    ax.set_title(f"Fraud Detection Model Comparison\n{total_transactions:,} transactions, {total_fraud} fraud cases — held-out test set",
                 fontweight='bold', fontsize=13, pad=20)
    
    # Footnote with business cost assumptions
    ax.text(0.5, -0.05, 
            f"Note: Business cost estimated as ${avg_fraud_loss:,.0f} per missed fraud + ${avg_churn_cost:,.0f} per false alarm. "
            f"Actual costs vary by institution.",
            ha='center', fontsize=8, color="#666666", transform=ax.transAxes)

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
    """Generate all visualizations."""
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

    print("[VISUALIZE] Loading metrics...")
    metrics = load_all_metrics()
    print(f"[VISUALIZE] Loaded metrics for {len(metrics)} models")

    print("[VISUALIZE] Loading feature importance...")
    importance = load_feature_importance()
    print(f"[VISUALIZE] Loaded importance for {len(importance)} models")

    plots = {}

    # Plot 1: PR Curves
    plots["pr_curves"] = plot_pr_curves(metrics, str(PLOTS_DIR / "01_pr_curves.png"))

    # Plot 2: Business Impact
    plots["business_impact"] = plot_business_impact(metrics, str(PLOTS_DIR / "02_business_impact.png"))

    # Plot 3: Feature Importance
    if importance:
        plots["feature_importance"] = plot_feature_importance(importance, str(PLOTS_DIR / "03_feature_importance.png"))

    # Plot 4: Threshold Tradeoff
    plots["threshold_tradeoff"] = plot_threshold_tradeoff(metrics, str(PLOTS_DIR / "04_threshold_tradeoff.png"))

    # Plot 5: SHAP Summary
    shap_plot = plot_shap_summary(str(PLOTS_DIR / "05_shap_summary.png"))
    if shap_plot:
        plots["shap_summary"] = shap_plot

    # Plot 6: Summary Dashboard
    plots["summary_dashboard"] = plot_summary_dashboard(metrics, str(PLOTS_DIR / "06_summary_dashboard.png"))

    print(f"\n[VISUALIZE] ✅ Generated {len(plots)} visualizations!")
    print(f"[VISUALIZE] 📁 Location: {PLOTS_DIR}")
    return plots


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Project Visualizations")
    parser.add_argument("--metrics-dir", type=str, default=None)
    parser.add_argument("--evaluation-dir", type=str, default=None)
    parser.add_argument("--feature-importance-dir", type=str, default=None)
    parser.add_argument("--plots-dir", type=str, default=None)
    args = parser.parse_args()

    try:
        plots = generate_all_visualizations(
            metrics_dir=args.metrics_dir,
            evaluation_dir=args.evaluation_dir,
            feature_importance_dir=args.feature_importance_dir,
            plots_dir=args.plots_dir,
        )
        print(f"\n[VISUALIZE] Done! {len(plots)} plots ready.")
    except Exception as e:
        print(f"[VISUALIZE] ERROR: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()