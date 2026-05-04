"""
Feature Engineering Module
Baseline and Advanced implementations for creating temporal, velocity, and interaction features.
Both pipelines are independent — run one or both, compare their outputs.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


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

    print(f"[FEATURES] Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"[FEATURES] Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    return df


# ================================
# Common Utilities
# ================================

def validate_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has a usable Time column. If scaled, derive raw time
    from distribution characteristics or use as-is for ordering.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with 'Time_raw' column for window calculations.
    """
    if "Time_raw" not in df.columns:
        # Time is already scaled. For ordering and window calculations,
        # we use the scaled Time since it preserves the ordering.
        print("[FEATURES] Using scaled Time for temporal ordering (monotonic relationship preserved).")
        df["Time_raw"] = df["Time"]
    return df


def save_feature_config(feature_names: List[str], output_path: str) -> None:
    """
    Save the list of feature names used for training.

    Args:
        feature_names: List of feature column names.
        output_path: Path to save the JSON config.
    """
    import json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"features": feature_names, "count": len(feature_names)}, f, indent=2)
    print(f"[FEATURES] Feature config saved to: {output_path}")


# ================================
# Baseline Feature Engineering
# ================================

def create_time_features_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic time-based features from the Time column.

    Args:
        df: Input DataFrame with 'Time_raw' column.

    Returns:
        DataFrame with added hour, day, and cyclical time features.
    """
    print("[FEATURES-BASELINE] Creating time features...")

    # Extract hour and day from raw time (seconds)
    df["hour"] = (df["Time_raw"].abs() // 3600 % 24).astype(int)
    df["day"] = (df["Time_raw"].abs() // 86400).astype(int)

    # Cyclical encoding for hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    print(f"[FEATURES-BASELINE]   hour range: {df['hour'].min()}–{df['hour'].max()}")
    print(f"[FEATURES-BASELINE]   day range:  {df['day'].min()}–{df['day'].max()}")

    return df


def create_velocity_features_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling-window velocity features using global aggregation.
    No user/card ID available, so windows are computed over all transactions
    ordered by time.

    Args:
        df: Input DataFrame sorted by Time_raw.

    Returns:
        DataFrame with velocity features added.
    """
    print("[FEATURES-BASELINE] Creating velocity features...")

    # Ensure sorted by time
    df = df.sort_values("Time_raw").reset_index(drop=True)

    windows = {
        "1h": 3600,
        "24h": 86400,
    }

    for window_name, window_seconds in windows.items():
        # Use rolling window on sorted data
        # Convert to a time-indexed series for rolling operations
        print(f"[FEATURES-BASELINE]   Computing {window_name} window features...")

        # Count of transactions in window
        df[f"txn_count_{window_name}"] = (
            df["Time_raw"]
            .rolling(window=int(window_seconds / df["Time_raw"].diff().median()), min_periods=1)
            .count()
            if df["Time_raw"].diff().median() > 0
            else np.arange(1, len(df) + 1)
        )

        # For proper time-based rolling, use expanding windows on sorted data
        time_values = df["Time_raw"].values
        amount_values = df["Amount"].values

        counts = np.zeros(len(df))
        avg_amounts = np.zeros(len(df))
        std_amounts = np.zeros(len(df))

        for i in range(len(df)):
            # Find transactions within the window
            cutoff = time_values[i] - window_seconds
            mask = (time_values[: i + 1] >= cutoff) & (time_values[: i + 1] <= time_values[i])
            window_amounts = amount_values[: i + 1][mask]
            counts[i] = len(window_amounts)
            avg_amounts[i] = window_amounts.mean() if len(window_amounts) > 0 else 0
            std_amounts[i] = window_amounts.std() if len(window_amounts) > 1 else 0

        df[f"txn_count_{window_name}"] = counts.astype(int)
        df[f"avg_amount_{window_name}"] = avg_amounts
        df[f"std_amount_{window_name}"] = std_amounts

        print(f"[FEATURES-BASELINE]     txn_count_{window_name}: "
              f"mean={counts.mean():.1f}, max={counts.max():.0f}")
        print(f"[FEATURES-BASELINE]     avg_amount_{window_name}: "
              f"mean={avg_amounts.mean():.4f}, std={avg_amounts.std():.4f}")

    return df


def run_feature_engineering_baseline(
    df: pd.DataFrame,
    config: Dict,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Execute the baseline feature engineering pipeline.

    Args:
        df: Cleaned DataFrame.
        config: Pipeline configuration dictionary.

    Returns:
        Tuple of (feature DataFrame, list of feature column names).
    """
    print("\n" + "=" * 60)
    print("BASELINE FEATURE ENGINEERING")
    print("=" * 60)

    df = df.copy()

    # Ensure Time column for ordering
    df = validate_time_column(df)

    # 1. Time-based features
    df = create_time_features_baseline(df)

    # 2. Velocity features
    df = create_velocity_features_baseline(df)

    # 3. Drop Time column (not a feature for training)
    if "Time" in df.columns:
        df.drop(columns=["Time"], inplace=True)
    if "Time_raw" in df.columns:
        df.drop(columns=["Time_raw"], inplace=True)

    # 4. Define feature columns (exclude target)
    target_col = "Class"
    feature_cols = [col for col in df.columns if col != target_col]

    print(f"\n[FEATURES-BASELINE] Total features: {len(feature_cols)}")
    print(f"[FEATURES-BASELINE] Feature columns: {feature_cols}")
    print(f"[FEATURES-BASELINE] Target column: {target_col}")
    print("=" * 60 + "\n")

    return df, feature_cols


# ================================
# Advanced Feature Engineering
# ================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between top correlated PCA components.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with interaction features.
    """
    print("[FEATURES-ADVANCED] Creating interaction features...")

    # Top correlated pairs from EDA
    interactions = {
        "V17_V14": ("V17", "V14"),
        "V12_V10": ("V12", "V10"),
        "V4_V11": ("V4", "V11"),
        "V3_V7": ("V3", "V7"),
    }

    for name, (col_a, col_b) in interactions.items():
        if col_a in df.columns and col_b in df.columns:
            df[name] = df[col_a] * df[col_b]
            print(f"[FEATURES-ADVANCED]   {name} = {col_a} × {col_b}")

    return df


def create_amount_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features representing the deviation of current transaction
    from recent historical averages.

    Args:
        df: Input DataFrame with velocity features.

    Returns:
        DataFrame with ratio features.
    """
    print("[FEATURES-ADVANCED] Creating amount ratio features...")

    if "avg_amount_1h" in df.columns:
        df["amount_ratio_1h"] = df["Amount"] / (df["avg_amount_1h"] + 1e-6)
        print("[FEATURES-ADVANCED]   amount_ratio_1h = Amount / avg_amount_1h")

    if "avg_amount_24h" in df.columns:
        df["amount_ratio_24h"] = df["Amount"] / (df["avg_amount_24h"] + 1e-6)
        print("[FEATURES-ADVANCED]   amount_ratio_24h = Amount / avg_amount_24h")

    if "std_amount_1h" in df.columns:
        df["amount_zscore_1h"] = (df["Amount"] - df["avg_amount_1h"]) / (df["std_amount_1h"] + 1e-6)
        print("[FEATURES-ADVANCED]   amount_zscore_1h = (Amount - avg) / std")

    return df


def create_extended_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create velocity features with finer and coarser granularity windows.
    Includes short-term burst detection.

    Args:
        df: Input DataFrame sorted by time.

    Returns:
        DataFrame with extended velocity features.
    """
    print("[FEATURES-ADVANCED] Creating extended velocity features...")

    df = df.sort_values("Time_raw").reset_index(drop=True)

    time_values = df["Time_raw"].values
    amount_values = df["Amount"].values

    # 10-minute window (600 seconds) — catches rapid bursts
    window_10min = 600
    counts_10min = np.zeros(len(df))
    avg_10min = np.zeros(len(df))

    for i in range(len(df)):
        cutoff = time_values[i] - window_10min
        mask = (time_values[: i + 1] >= cutoff) & (time_values[: i + 1] <= time_values[i])
        window_amounts = amount_values[: i + 1][mask]
        counts_10min[i] = len(window_amounts)
        avg_10min[i] = window_amounts.mean() if len(window_amounts) > 0 else 0

    df["txn_count_10min"] = counts_10min.astype(int)
    df["avg_amount_10min"] = avg_10min

    print(f"[FEATURES-ADVANCED]   txn_count_10min: mean={counts_10min.mean():.1f}")
    print(f"[FEATURES-ADVANCED]   avg_amount_10min: mean={avg_10min.mean():.4f}")

    return df


def create_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features measuring time since last transaction (recency).
    Fraudsters often perform rapid successive transactions.

    Args:
        df: Input DataFrame sorted by time.

    Returns:
        DataFrame with recency features.
    """
    print("[FEATURES-ADVANCED] Creating recency features...")

    df = df.sort_values("Time_raw").reset_index(drop=True)

    # Time since previous transaction
    df["seconds_since_last_txn"] = df["Time_raw"].diff().fillna(0)
    df["seconds_since_last_txn"] = df["seconds_since_last_txn"].clip(lower=0)

    print(f"[FEATURES-ADVANCED]   seconds_since_last_txn: "
          f"mean={df['seconds_since_last_txn'].mean():.1f}s, "
          f"median={df['seconds_since_last_txn'].median():.1f}s")

    return df


def create_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unsupervised anomaly scores using Isolation Forest on PCA features.
    These serve as input features for the supervised model.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with anomaly score feature.
    """
    print("[FEATURES-ADVANCED] Creating anomaly detection features...")

    pca_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]

    if not pca_cols:
        print("[FEATURES-ADVANCED]   No PCA columns found. Skipping.")
        return df

    X_pca = df[pca_cols].values

    # Isolation Forest for anomaly detection
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,  # Expected ~1% anomalies
        random_state=42,
        n_jobs=-1,
    )

    # Fit and get anomaly scores (-1 for outliers, 1 for inliers, shift to 0-1)
    raw_scores = iso_forest.fit_predict(X_pca)
    df["anomaly_score"] = (-raw_scores + 1) / 2  # 0 = normal, 1 = anomalous

    # Also get decision function scores for finer granularity
    df["anomaly_decision"] = -iso_forest.score_samples(X_pca)

    anomaly_rate = (df["anomaly_score"] > 0.5).mean() * 100
    print(f"[FEATURES-ADVANCED]   anomaly_score: {anomaly_rate:.1f}% flagged as anomalous")

    return df


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create statistical moment features and feature selection.
    Adds amount decile for robust outlier handling.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with statistical features.
    """
    print("[FEATURES-ADVANCED] Creating statistical features...")

    # Amount percentile rank
    df["amount_percentile"] = df["Amount"].rank(pct=True)
    print(f"[FEATURES-ADVANCED]   amount_percentile: 0–1 normalized rank")

    # Amount decile (1-10)
    df["amount_decile"] = pd.qcut(df["Amount"].rank(method="first"), 10, labels=False) + 1
    print(f"[FEATURES-ADVANCED]   amount_decile: 10 bins")

    # Is zero amount flag
    df["is_zero_amount"] = (df["Amount"] == df["Amount"].min()).astype(int)
    print(f"[FEATURES-ADVANCED]   is_zero_amount: {(df['is_zero_amount'] == 1).sum()} transactions")

    # Is night transaction (hour 0–5)
    if "hour" in df.columns:
        df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)
        print(f"[FEATURES-ADVANCED]   is_night: {(df['is_night'] == 1).sum()} transactions")

    return df


def select_top_features(df: pd.DataFrame, n_top: int = 20) -> List[str]:
    """
    Select top N features by absolute correlation with Class.
    Excludes engineered features that derive from the target or are identifiers.

    Args:
        df: Input DataFrame.
        n_top: Number of top features to retain.

    Returns:
        List of selected feature names.
    """
    if "Class" not in df.columns:
        return [col for col in df.columns if col != "Class"]

    exclude = ["Class", "Time", "Time_raw"]
    feature_cols = [col for col in df.columns if col not in exclude and df[col].dtype in ["float64", "int64", "int32"]]

    correlations = df[feature_cols].corrwith(df["Class"]).abs().sort_values(ascending=False)
    selected = correlations.head(n_top).index.tolist()

    print(f"[FEATURES-ADVANCED] Top {n_top} features selected by correlation:")
    for i, feat in enumerate(selected[:10], 1):
        print(f"[FEATURES-ADVANCED]   {i:2d}. {feat} ({correlations[feat]:.4f})")

    return selected


def run_feature_engineering_advanced(
    df: pd.DataFrame,
    config: Dict,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Execute the advanced feature engineering pipeline.

    Args:
        df: Cleaned DataFrame.
        config: Pipeline configuration dictionary.

    Returns:
        Tuple of (feature DataFrame, list of feature column names).
    """
    print("\n" + "=" * 60)
    print("ADVANCED FEATURE ENGINEERING")
    print("=" * 60)

    df = df.copy()

    # Ensure Time column for ordering
    df = validate_time_column(df)

    # 1. Run baseline features first (foundation)
    df, _ = run_feature_engineering_baseline(df, config)

    # Re-add Time_raw for advanced temporal features
    df["Time_raw"] = df.get("Time_raw", df.get("Time", np.arange(len(df))))

    # 2. Interaction features
    df = create_interaction_features(df)

    # 3. Amount ratio features (deviation from normal)
    df = create_amount_ratio_features(df)

    # 4. Extended velocity (finer granularity)
    df = create_extended_velocity_features(df)

    # 5. Recency features
    df = create_recency_features(df)

    # 6. Anomaly detection features
    df = create_anomaly_features(df)

    # 7. Statistical features
    df = create_statistical_features(df)

    # 8. Clean up temporary columns
    if "Time_raw" in df.columns:
        df.drop(columns=["Time_raw"], inplace=True)
    if "Time" in df.columns:
        df.drop(columns=["Time"], inplace=True)

    # 9. Define feature columns (exclude target)
    target_col = "Class"
    feature_cols = [col for col in df.columns if col != target_col]

    # 10. Optional: feature selection for dimensionality reduction
    if config["feature_engineering"]["feature_selection"]["enabled"]:
        n_top = config["feature_engineering"]["feature_selection"]["top_n_correlated"]
        selected_features = select_top_features(df, n_top)
        feature_cols = selected_features

    print(f"\n[FEATURES-ADVANCED] Total features: {len(feature_cols)}")
    print(f"[FEATURES-ADVANCED] Feature columns: {feature_cols}")
    print(f"[FEATURES-ADVANCED] Target column: {target_col}")
    print("=" * 60 + "\n")

    return df, feature_cols


# ================================
# Main Pipeline
# ================================

def run_feature_engineering(
    config_path: Optional[str] = None,
    mode: str = "both",
) -> Dict:
    """
    Run feature engineering pipeline in baseline, advanced, or both modes.

    Args:
        config_path: Optional path to pipeline config YAML.
        mode: 'baseline', 'advanced', or 'both'.

    Returns:
        Dictionary with paths to saved feature files and feature name lists.
    """
    config = load_config(config_path)
    df = load_cleaned_data(config)

    results = {}

    project_root = Path(__file__).resolve().parent.parent

    if mode in ("baseline", "both"):
        print("\n" + "=" * 60)
        print("RUNNING BASELINE FEATURE ENGINEERING")
        print("=" * 60)

        df_baseline, baseline_features = run_feature_engineering_baseline(df.copy(), config)

        # Save
        baseline_path = config["paths"]["features_data"]
        if not os.path.isabs(baseline_path):
            baseline_path = str(project_root / baseline_path)
        baseline_path = baseline_path.replace(".parquet", "_baseline.parquet")

        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        df_baseline.to_parquet(baseline_path, index=False)
        print(f"[FEATURES-BASELINE] Saved to: {baseline_path}")

        # Save feature config
        baseline_config_path = config["paths"]["feature_config"].replace(".json", "_baseline.json")
        if not os.path.isabs(baseline_config_path):
            baseline_config_path = str(project_root / baseline_config_path)
        save_feature_config(baseline_features, baseline_config_path)

        results["baseline"] = {
            "path": baseline_path,
            "features": baseline_features,
            "config_path": baseline_config_path,
        }

    if mode in ("advanced", "both"):
        print("\n" + "=" * 60)
        print("RUNNING ADVANCED FEATURE ENGINEERING")
        print("=" * 60)

        df_advanced, advanced_features = run_feature_engineering_advanced(df.copy(), config)

        # Save
        advanced_path = config["paths"]["features_data"]
        if not os.path.isabs(advanced_path):
            advanced_path = str(project_root / advanced_path)
        advanced_path = advanced_path.replace(".parquet", "_advanced.parquet")

        os.makedirs(os.path.dirname(advanced_path), exist_ok=True)
        df_advanced.to_parquet(advanced_path, index=False)
        print(f"[FEATURES-ADVANCED] Saved to: {advanced_path}")

        # Save feature config
        advanced_config_path = config["paths"]["feature_config"].replace(".json", "_advanced.json")
        if not os.path.isabs(advanced_config_path):
            advanced_config_path = str(project_root / advanced_config_path)
        save_feature_config(advanced_features, advanced_config_path)

        results["advanced"] = {
            "path": advanced_path,
            "features": advanced_features,
            "config_path": advanced_config_path,
        }

    return results


# ================================
# Entry Point
# ================================

def main():
    """
    Execute feature engineering as a standalone script.
    Runs both baseline and advanced by default.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["baseline", "advanced", "both"],
        help="Which feature set to generate (default: both)",
    )
    args = parser.parse_args()

    try:
        results = run_feature_engineering(mode=args.mode)

        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 60)

        for mode_name, result in results.items():
            print(f"\n  {mode_name.upper()}:")
            print(f"    Features: {len(result['features'])}")
            print(f"    Data:     {result['path']}")
            print(f"    Config:   {result['config_path']}")

        print("=" * 60)

        return results

    except FileNotFoundError as e:
        print(f"[FEATURES] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"[FEATURES] UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()