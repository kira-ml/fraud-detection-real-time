"""
Exploratory Data Analysis Module
Baseline implementation for generating descriptive statistics and visualizations
to understand the credit card fraud dataset characteristics.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


# ================================
# Configuration Loader
# ================================

def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to config file. If None, auto-resolves relative to project root.

    Returns:
        Dictionary containing pipeline configuration.
    """
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "config" / "pipeline_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# ================================
# Plotting Configuration
# ================================

def set_plotting_style() -> None:
    """Set consistent plotting style for all EDA figures."""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("Set2")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


# ================================
# Data Loading
# ================================

def load_cleaned_data(config: Dict) -> pd.DataFrame:
    """
    Load the cleaned dataset from the preprocessing step.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        DataFrame containing cleaned transaction data.
    """
    data_path = config["paths"]["cleaned_data"]

    if not os.path.isabs(data_path):
        project_root = Path(__file__).resolve().parent.parent
        data_path = str(project_root / data_path)

    print(f"[EDA] Loading data from: {data_path}")
    df = pd.read_parquet(data_path)

    print(f"[EDA] Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    return df


# ================================
# Statistical Analysis
# ================================

def compute_class_distribution(df: pd.DataFrame) -> Dict:
    """
    Compute and display class distribution statistics.

    Args:
        df: Input DataFrame with 'Class' column.

    Returns:
        Dictionary with class distribution metrics.
    """
    class_counts = df["Class"].value_counts()
    total = len(df)

    distribution = {
        "total_transactions": int(total),
        "legitimate": int(class_counts.get(0, 0)),
        "fraud": int(class_counts.get(1, 0)),
        "fraud_ratio_pct": round(class_counts.get(1, 0) / total * 100, 4),
    }

    print("\n" + "=" * 50)
    print("CLASS DISTRIBUTION")
    print("=" * 50)
    print(f"  Total Transactions: {distribution['total_transactions']:,}")
    print(f"  Legitimate:         {distribution['legitimate']:,}")
    print(f"  Fraudulent:         {distribution['fraud']:,}")
    print(f"  Fraud Ratio:        {distribution['fraud_ratio_pct']}%")
    print("=" * 50)

    return distribution


def compute_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for key numerical features.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with summary statistics.
    """
    key_features = ["Amount", "Time"]
    stats = {}

    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    for col in key_features:
        if col in df.columns:
            col_stats = {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "median": round(float(df[col].median()), 4),
            }
            stats[col] = col_stats

            print(f"\n  {col}:")
            print(f"    Mean:   {col_stats['mean']:,.2f}")
            print(f"    Std:    {col_stats['std']:,.2f}")
            print(f"    Min:    {col_stats['min']:,.2f}")
            print(f"    Max:    {col_stats['max']:,.2f}")
            print(f"    Median: {col_stats['median']:,.2f}")

    print("=" * 50)

    return stats


def compute_feature_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Compute correlations of all features with the target Class.

    Args:
        df: Input DataFrame with 'Class' column.

    Returns:
        Series of correlations with Class, sorted by absolute value descending.
    """
    target = "Class"

    # Select numeric features only (exclude Time if raw, keep scaled)
    feature_cols = [col for col in df.columns if col != target]

    correlations = df[feature_cols].corrwith(df[target]).sort_values(
        key=abs, ascending=False
    )

    print("\n" + "=" * 50)
    print("TOP FEATURES CORRELATED WITH FRAUD")
    print("=" * 50)

    for feature, corr in correlations.head(15).items():
        direction = "(positive)" if corr > 0 else "(negative)"
        print(f"  {feature:8s}: {corr:+.4f} {direction}")

    print("=" * 50)

    return correlations


def compute_pca_variance(df: pd.DataFrame) -> Dict:
    """
    Analyze variance explained by PCA components V1-V28.

    Args:
        df: Input DataFrame with V1-V28 columns.

    Returns:
        Dictionary with PCA variance analysis.
    """
    pca_cols = [f"V{i}" for i in range(1, 29)]
    pca_cols_present = [col for col in pca_cols if col in df.columns]

    if not pca_cols_present:
        print("[EDA] No PCA features found. Skipping PCA variance analysis.")
        return {}

    variances = df[pca_cols_present].var().sort_values(ascending=False)
    total_variance = variances.sum()
    explained = variances / total_variance
    cumulative = explained.cumsum()

    print("\n" + "=" * 50)
    print("PCA VARIANCE ANALYSIS (Top 5 Components)")
    print("=" * 50)

    for i, (col, var_ratio) in enumerate(explained.head(5).items(), 1):
        cum = cumulative[col]
        print(f"  {col}: {var_ratio:.4f} ({var_ratio*100:.2f}%) — Cumulative: {cum*100:.2f}%")

    print(f"\n  Total variance explained by 2 components: {cumulative.iloc[1]:.4f}")
    print(f"  Total variance explained by 5 components: {cumulative.iloc[4]:.4f}")
    print("=" * 50)

    return {
        "explained_variance_ratio": explained.to_dict(),
        "cumulative_variance": cumulative.to_dict(),
    }


# ================================
# Visualization Functions
# ================================

def plot_class_distribution(
    df: pd.DataFrame,
    distribution: Dict,
    output_dir: str,
) -> str:
    """
    Plot class distribution as a bar chart.

    Args:
        df: Input DataFrame.
        distribution: Class distribution dictionary.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    labels = ["Legitimate", "Fraudulent"]
    counts = [distribution["legitimate"], distribution["fraud"]]
    colors = ["#2ecc71", "#e74c3c"]

    axes[0].bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Class Distribution (Bar Chart)")
    axes[0].set_ylabel("Number of Transactions")
    for i, count in enumerate(counts):
        axes[0].text(i, count + max(counts) * 0.02, f"{count:,}",
                     ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.3f%%",
        explode=(0, 0.1),
        shadow=True,
        startangle=90,
    )
    axes[1].set_title("Class Distribution (Pie Chart)")

    plt.suptitle(
        f"Fraud Detection Dataset — Class Imbalance\n"
        f"Fraud Ratio: {distribution['fraud_ratio_pct']}%",
        fontweight="bold",
    )

    filepath = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(filepath)
    plt.close()

    print(f"[EDA] Class distribution plot saved to: {filepath}")

    return filepath


def plot_amount_histogram(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot histogram of transaction amounts split by class.

    Args:
        df: Input DataFrame.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    legit = df[df["Class"] == 0]["Amount"]
    fraud = df[df["Class"] == 1]["Amount"]

    # Full range histogram
    axes[0, 0].hist(legit, bins=100, alpha=0.7, label="Legitimate", color="#2ecc71",
                    edgecolor="black", linewidth=0.3)
    axes[0, 0].hist(fraud, bins=50, alpha=0.9, label="Fraud", color="#e74c3c",
                    edgecolor="black", linewidth=0.3)
    axes[0, 0].set_title("Amount Distribution (Full Range)")
    axes[0, 0].set_xlabel("Amount ($)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()
    axes[0, 0].set_yscale("log")

    # Zoomed range (0-500)
    axes[0, 1].hist(legit[legit <= 500], bins=100, alpha=0.7, label="Legitimate",
                    color="#2ecc71", edgecolor="black", linewidth=0.3)
    axes[0, 1].hist(fraud[fraud <= 500], bins=50, alpha=0.9, label="Fraud",
                    color="#e74c3c", edgecolor="black", linewidth=0.3)
    axes[0, 1].set_title("Amount Distribution (0–$500)")
    axes[0, 1].set_xlabel("Amount ($)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()

    # Log-transformed histogram
    axes[1, 0].hist(np.log1p(legit), bins=100, alpha=0.7, label="Legitimate",
                    color="#2ecc71", edgecolor="black", linewidth=0.3)
    axes[1, 0].hist(np.log1p(fraud), bins=50, alpha=0.9, label="Fraud",
                    color="#e74c3c", edgecolor="black", linewidth=0.3)
    axes[1, 0].set_title("Log-Transformed Amount Distribution")
    axes[1, 0].set_xlabel("log(Amount + 1)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()

    # Box plot
    box_data = [legit, fraud]
    bp = axes[1, 1].boxplot(box_data, tick_labels=["Legitimate", "Fraud"],
                            patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    axes[1, 1].set_title("Amount Box Plot (Outliers Hidden)")
    axes[1, 1].set_ylabel("Amount ($)")

    plt.suptitle("Transaction Amount Analysis by Class", fontweight="bold")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "amount_histogram.png")
    plt.savefig(filepath)
    plt.close()

    print(f"[EDA] Amount histogram saved to: {filepath}")

    return filepath


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot correlation heatmap of PCA features with Class.

    Args:
        df: Input DataFrame.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    pca_cols = [f"V{i}" for i in range(1, 29)]
    pca_cols_present = [col for col in pca_cols if col in df.columns]

    target_cols = pca_cols_present + ["Class"]
    corr_matrix = df[target_cols].corr()

    # Extract correlation with Class
    class_corr = corr_matrix["Class"].drop("Class").sort_values(key=abs, ascending=False)
    top_features = class_corr.head(15).index.tolist()

    # Correlation heatmap of top features
    plot_cols = top_features + ["Class"]
    plot_corr = df[plot_cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Heatmap of top features
    mask = np.triu(np.ones_like(plot_corr, dtype=bool), k=1)
    sns.heatmap(
        plot_corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        square=True,
        linewidths=0.5,
        ax=axes[0],
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Correlation Heatmap — Top 15 Features with Class")

    # Bar chart of correlations with Class
    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in class_corr.head(15).values]
    axes[1].barh(
        range(len(top_features)),
        class_corr.head(15).values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_yticks(range(len(top_features)))
    axes[1].set_yticklabels(top_features)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Correlation with Fraud (Class=1)")
    axes[1].set_title("Top 15 Features Correlated with Fraud")
    axes[1].axvline(x=0, color="black", linestyle="-", linewidth=0.8)

    for i, (feature, corr) in enumerate(class_corr.head(15).items()):
        axes[1].text(
            corr + (0.02 if corr >= 0 else -0.02),
            i,
            f"{corr:+.3f}",
            va="center",
            fontsize=9,
        )

    plt.suptitle("Feature Correlation Analysis", fontweight="bold")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(filepath)
    plt.close()

    print(f"[EDA] Correlation heatmap saved to: {filepath}")

    return filepath


def plot_fraud_by_hour(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot fraud frequency by hour of day.

    Args:
        df: Input DataFrame with 'Time' column (scaled or raw).

    Returns:
        Path to saved figure.
    """
    # Derive hour from the dataset
    # If Time is scaled, we derive from the raw version. Use cleaned data which
    # may still have Time information. We'll compute hour based on the
    # assumption that max raw Time is ~172792 seconds (~48 hours).
    if "hour" in df.columns:
        hour_series = df["hour"]
        title_suffix = "(from engineered features)"
    else:
        # Derive hour from Time if raw values are accessible
        # Since data is scaled, we check if an unscaled Time exists
        print("[EDA] Hour column not found. Computing from Time...")
        hour_series = (df["Time"] % 86400 // 3600).astype(int) if "Time" in df.columns else None

    if hour_series is None:
        print("[EDA] Cannot compute hour. Skipping fraud-by-hour plot.")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fraud rate by hour
    hourly_stats = df.groupby(hour_series).agg(
        total=("Class", "count"),
        fraud=("Class", "sum"),
    )
    hourly_stats["fraud_rate"] = hourly_stats["fraud"] / hourly_stats["total"] * 100

    # Transaction volume by hour
    axes[0].bar(hourly_stats.index, hourly_stats["total"], color="#3498db",
                edgecolor="black", linewidth=0.5)
    axes[0].set_title("Transaction Volume by Hour")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Number of Transactions")

    # Fraud rate by hour
    axes[1].bar(hourly_stats.index, hourly_stats["fraud_rate"], color="#e74c3c",
                edgecolor="black", linewidth=0.5)
    axes[1].set_title("Fraud Rate by Hour")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].axhline(y=df["Class"].mean() * 100, color="black", linestyle="--",
                    label=f"Average ({df['Class'].mean()*100:.3f}%)")
    axes[1].legend()

    plt.suptitle("Temporal Fraud Patterns", fontweight="bold")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "fraud_by_hour.png")
    plt.savefig(filepath)
    plt.close()

    print(f"[EDA] Fraud by hour plot saved to: {filepath}")

    return filepath


def save_correlation_ranking(
    correlations: pd.Series,
    output_dir: str,
) -> str:
    """
    Save feature correlation ranking to CSV.

    Args:
        correlations: Series of correlations with Class.
        output_dir: Directory to save the CSV.

    Returns:
        Path to saved CSV.
    """
    ranking_df = correlations.reset_index()
    ranking_df.columns = ["Feature", "Correlation_with_Fraud"]
    ranking_df["Abs_Correlation"] = ranking_df["Correlation_with_Fraud"].abs()
    ranking_df = ranking_df.sort_values("Abs_Correlation", ascending=False)

    filepath = os.path.join(output_dir, "feature_correlation_ranking.csv")
    ranking_df.to_csv(filepath, index=False)

    print(f"[EDA] Correlation ranking saved to: {filepath}")

    return filepath


# ================================
# Main EDA Pipeline
# ================================

def run_eda(config_path: Optional[str] = None) -> Dict:
    """
    Run the full exploratory data analysis pipeline.

    Args:
        config_path: Optional path to pipeline config YAML.

    Returns:
        Dictionary containing all EDA results.
    """
    config = load_config(config_path)
    output_dir = config["paths"]["eda_dir"]

    if not os.path.isabs(output_dir):
        project_root = Path(__file__).resolve().parent.parent
        output_dir = str(project_root / output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Set plotting style
    set_plotting_style()

    print("[EDA] Starting Exploratory Data Analysis...")
    print(f"[EDA] Output directory: {output_dir}")

    # Load data
    df = load_cleaned_data(config)

    # 1. Class distribution
    print("[EDA] Analyzing class distribution...")
    distribution = compute_class_distribution(df)
    plot_class_distribution(df, distribution, output_dir)

    # 2. Summary statistics
    print("[EDA] Computing summary statistics...")
    stats = compute_summary_statistics(df)

    # 3. Feature correlations
    print("[EDA] Computing feature correlations...")
    correlations = compute_feature_correlations(df)
    plot_correlation_heatmap(df, output_dir)
    save_correlation_ranking(correlations, output_dir)

    # 4. Amount analysis
    print("[EDA] Analyzing transaction amounts...")
    plot_amount_histogram(df, output_dir)

    # 5. Temporal patterns
    print("[EDA] Analyzing temporal patterns...")
    plot_fraud_by_hour(df, output_dir)

    # 6. PCA variance
    print("[EDA] Analyzing PCA variance...")
    pca_analysis = compute_pca_variance(df)

    print(f"\n[EDA] Analysis complete. {len(os.listdir(output_dir))} files saved to {output_dir}")

    return {
        "distribution": distribution,
        "statistics": stats,
        "correlations": correlations.to_dict(),
        "pca_analysis": pca_analysis,
    }


# ================================
# Entry Point
# ================================

def main():
    """
    Execute EDA as a standalone script.
    Loads cleaned data and generates all analysis outputs.
    """
    try:
        results = run_eda()

        print("\n" + "=" * 60)
        print("EDA COMPLETE — KEY FINDINGS")
        print("=" * 60)
        print(f"  Dataset:        {results['distribution']['total_transactions']:,} transactions")
        print(f"  Fraud Rate:     {results['distribution']['fraud_ratio_pct']}%")
        print(f"  Top Features:   Check reports/eda/feature_correlation_ranking.csv")
        print(f"  Visualizations: reports/eda/")
        print("=" * 60)

        return results

    except FileNotFoundError as e:
        print(f"[EDA] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"[EDA] UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()