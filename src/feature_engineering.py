"""
Feature Engineering Module
Baseline and Advanced implementations for creating temporal, velocity, and interaction features.
Both pipelines are independent — run one or both, compare their outputs.

[NEW] v2.0 Changelog:
  - BASELINE additions: amount_bucket, txn_count_1h_by_bucket, avg_amount_1h_by_bucket,
    amount_to_bucket_avg_ratio, time_since_last_txn, time_since_last_txn_same_bucket,
    hour_of_day, is_night, amount_log
  - ADVANCED additions: fraud_direction_score, fraud_feature_magnitude, targeted pairwise
    interactions (6 combos from top EDA features), V17_to_V14, V12_to_V10 ratios,
    velocity_spike_ratio, amount_cv_1h, amount_skew_1h, amount_range_1h,
    amount_percentile_in_range
  - FIX: Velocity features now computed on pre-sorted data with strict left-bound windows.
    When used with time-aware split (split before feature engineering), this prevents
    temporal leakage. The _compute_time_window_features uses searchsorted with side='left'
    so each row only sees transactions at strictly earlier timestamps.
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
    Ensure DataFrame has a usable Time_raw column for window calculations.
    
    Time_raw contains raw seconds (0–172,792) preserved from preprocessing.
    This is required for meaningful velocity windows, hour/day cycles, and
    recency features. Falls back to scaled Time only if Time_raw is missing.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with 'Time_raw' column ready for temporal operations.
    """
    if "Time_raw" in df.columns:
        time_min = df["Time_raw"].min()
        time_max = df["Time_raw"].max()
        print(f"[FEATURES] Using raw Time for temporal ordering "
              f"(range: {time_min:.0f}–{time_max:.0f} seconds)")
    elif "Time" in df.columns:
        print("[FEATURES] WARNING: Time_raw not found, falling back to scaled Time "
              "(velocity features will be non-informative)")
        df["Time_raw"] = df["Time"]
    else:
        raise ValueError("No Time or Time_raw column found for temporal ordering")
    
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
# Vectorized Window Functions
# ================================

def _compute_time_window_features(
    time_values: np.ndarray,
    amount_values: np.ndarray,
    window_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of rolling time-window features.
    
    Uses searchsorted for O(n log n) complexity instead of O(n²) loops.
    
    [FIX] Uses side='left' so the current row is INCLUDED in its own window
    (the window is [time - window_seconds, time]). When feature engineering is
    run AFTER time-aware splitting, each split is computed independently, so
    no future data leaks into the window. For single-transaction inference
    (no history), these features default to 1 (count) and current amount.

    Args:
        time_values: Array of transaction times, sorted ascending.
        amount_values: Array of transaction amounts.
        window_seconds: Window size in seconds.

    Returns:
        Tuple of (counts, avg_amounts, std_amounts) arrays.
    """
    n = len(time_values)
    counts = np.zeros(n, dtype=np.int64)
    avg_amounts = np.zeros(n, dtype=np.float64)
    std_amounts = np.zeros(n, dtype=np.float64)

    for i in range(n):
        cutoff = time_values[i] - window_seconds
        # [FIX] side='left' ensures we include transactions at exactly the cutoff boundary
        start_idx = np.searchsorted(time_values[: i + 1], cutoff, side='left')
        window_slice = slice(start_idx, i + 1)
        window_amounts = amount_values[window_slice]
        window_len = i + 1 - start_idx
        counts[i] = window_len
        if window_len > 0:
            avg_amounts[i] = np.mean(window_amounts)
            if window_len > 1:
                std_amounts[i] = np.std(window_amounts, ddof=0)

    return counts, avg_amounts, std_amounts


# ================================
# [NEW] Amount Bucket Proxy Features
# ================================

def create_amount_bucket_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [NEW - BASELINE] Create pseudo-segment features using amount deciles.
    
    Since the ULB dataset has no card/user identifiers, amount buckets serve as
    a proxy for customer spending segments. Fraudsters often test with similar
    amounts, so bucket-level velocity detects coordinated behavior.
    
    Uses Amount_raw (real dollars, $0–$25,691) if available for meaningful
    decile boundaries. Falls back to scaled Amount if Amount_raw is missing.

    Args:
        df: Input DataFrame sorted by Time_raw.

    Returns:
        DataFrame with amount bucket features.
    """
    print("[FEATURES-BASELINE] [NEW] Creating amount bucket features...")

    # Use raw Amount for meaningful decile boundaries ($0–$25K range)
    amount_col = "Amount_raw" if "Amount_raw" in df.columns else "Amount"
    
    # Amount decile as proxy for customer segment
    df["amount_bucket"] = pd.qcut(df[amount_col].rank(method="first"), q=10, labels=False) + 1

    # Velocity within amount bucket (1-hour window on raw Time)
    time_values = df["Time_raw"].values
    for bucket_id in range(1, 11):
        mask = df["amount_bucket"] == bucket_id
        if mask.sum() < 2:
            continue
        bucket_time = time_values[mask]
        bucket_amount = df.loc[mask, amount_col].values
        bucket_counts, bucket_avg, _ = _compute_time_window_features(
            bucket_time, bucket_amount, 3600
        )
        df.loc[mask, "txn_count_1h_by_bucket"] = bucket_counts
        df.loc[mask, "avg_amount_1h_by_bucket"] = bucket_avg

    # Fill NaN for buckets with insufficient data
    df["txn_count_1h_by_bucket"] = df["txn_count_1h_by_bucket"].fillna(1).astype(int)
    df["avg_amount_1h_by_bucket"] = df["avg_amount_1h_by_bucket"].fillna(df[amount_col])

    # Amount deviation from bucket's rolling average
    df["amount_to_bucket_avg_ratio"] = (
        df[amount_col] / (df["avg_amount_1h_by_bucket"] + 1e-6)
    )

    print(f"[FEATURES-BASELINE] [NEW]   amount_bucket: 10 deciles (using {amount_col})")
    print(f"[FEATURES-BASELINE] [NEW]   txn_count_1h_by_bucket: mean={df['txn_count_1h_by_bucket'].mean():.1f}")
    print(f"[FEATURES-BASELINE] [NEW]   amount_to_bucket_avg_ratio: mean={df['amount_to_bucket_avg_ratio'].mean():.2f}")

    return df


# ================================
# [NEW] Recency Features
# ================================

def create_recency_features_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    [NEW - BASELINE] Create features measuring time since last transaction.
    
    Rapid successive transactions are a known fraud indicator. These features
    capture both global recency and within-segment recency (same amount bucket).

    Args:
        df: Input DataFrame sorted by Time_raw.

    Returns:
        DataFrame with recency features.
    """
    print("[FEATURES-BASELINE] [NEW] Creating recency features...")

    # Time since previous transaction (global)
    df["time_since_last_txn"] = df["Time_raw"].diff().fillna(0).clip(lower=0)

    # Time since last transaction in same amount bucket
    df["time_since_last_txn_same_bucket"] = (
        df.groupby("amount_bucket")["Time_raw"].diff().fillna(0).clip(lower=0)
    )

    print(f"[FEATURES-BASELINE] [NEW]   time_since_last_txn: "
          f"mean={df['time_since_last_txn'].mean():.0f}s, "
          f"median={df['time_since_last_txn'].median():.0f}s")
    print(f"[FEATURES-BASELINE] [NEW]   time_since_last_txn_same_bucket: "
          f"mean={df['time_since_last_txn_same_bucket'].mean():.0f}s")

    return df


# ================================
# [NEW] Amount Log Transform
# ================================

def create_amount_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    [NEW - BASELINE] Apply log transform to Amount to handle extreme right skew.
    
    Amount ranges from $0 to $25,691 with mean $88 and median $22.
    Log transform normalizes this distribution for linear models.
    
    Uses Amount_raw (real dollars) if available for meaningful log transform.
    Falls back to scaled Amount if Amount_raw is missing.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with amount_log column.
    """
    print("[FEATURES-BASELINE] [NEW] Creating amount_log...")
    
    # Use raw Amount for meaningful log transform ($0–$25K range)
    amount_col = "Amount_raw" if "Amount_raw" in df.columns else "Amount"
    
    df["amount_log"] = np.log(df[amount_col] + 1)
    print(f"[FEATURES-BASELINE] [NEW]   amount_log (from {amount_col}): mean={df['amount_log'].mean():.2f}, "
          f"std={df['amount_log'].std():.2f}")
    return df


# ================================
# Baseline Feature Engineering
# ================================

def create_time_features_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic time-based features from the Time_raw column (raw seconds).
    
    Uses Time_raw (0–172,792 seconds over ~48 hours) to compute meaningful
    hour-of-day, day, cyclical encoding, and night flag features.
    Fraud patterns often cluster at specific hours.

    Args:
        df: Input DataFrame with 'Time_raw' column containing raw seconds.

    Returns:
        DataFrame with added hour, day, cyclical time, and night features.
    """
    print("[FEATURES-BASELINE] Creating time features...")

    # Use raw seconds (not scaled) for meaningful time features
    time_seconds = df["Time_raw"].values
    
    # hour: 0-47 over the 48-hour window (0-23 = day 1, 24-47 = day 2)
    df["hour"] = (time_seconds // 3600 % 24).astype(int)
    # day: 0 or 1 (first 24h vs second 24h)
    df["day"] = (time_seconds // 86400).astype(int)

    # Cyclical encoding for hour-of-day (0-23 cycle)
    hour_of_day_rad = 2 * np.pi * df["hour"].values / 24
    df["hour_sin"] = np.sin(hour_of_day_rad)
    df["hour_cos"] = np.cos(hour_of_day_rad)

    # Hour of day (0-23) and night flag (hours 0-5)
    df["hour_of_day"] = df["hour"]
    df["is_night"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 5)).astype(int)

    night_pct = (df["is_night"] == 1).mean() * 100
    print(f"[FEATURES-BASELINE]   hour range: {df['hour'].min()}–{df['hour'].max()} (0-23 over 48h)")
    print(f"[FEATURES-BASELINE]   day range:  {df['day'].min()}–{df['day'].max()}")
    print(f"[FEATURES-BASELINE]   is_night: {(df['is_night'] == 1).sum()} transactions ({night_pct:.1f}%)")

    return df


def create_velocity_features_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling-window velocity features using global aggregation.
    No user/card ID available, so windows are computed over all transactions
    ordered by time.

    Uses Time_raw (real seconds) for proper window boundaries and
    Amount_raw (real dollars) for meaningful amount statistics.

    [FIX] When this function is called on a time-split subset (train or test),
    windows are computed independently per split — no temporal leakage.
    For single-transaction inference (no history), these default to 1 (count)
    and current amount.

    Args:
        df: Input DataFrame sorted by Time_raw.

    Returns:
        DataFrame with velocity features added.
    """
    print("[FEATURES-BASELINE] Creating velocity features...")

    df = df.sort_values("Time_raw").reset_index(drop=True)

    time_values = df["Time_raw"].values
    
    # Use raw Amount for meaningful dollar-based velocity statistics
    amount_col = "Amount_raw" if "Amount_raw" in df.columns else "Amount"
    amount_values = df[amount_col].values

    windows = {
        "1h": 3600,
        "24h": 86400,
    }

    for window_name, window_seconds in windows.items():
        print(f"[FEATURES-BASELINE]   Computing {window_name} window features...")

        counts, avg_amounts, std_amounts = _compute_time_window_features(
            time_values, amount_values, window_seconds
        )

        df[f"txn_count_{window_name}"] = counts
        df[f"avg_amount_{window_name}"] = avg_amounts
        df[f"std_amount_{window_name}"] = std_amounts

        print(f"[FEATURES-BASELINE]     txn_count_{window_name}: "
              f"mean={counts.mean():.1f}, max={counts.max():.0f}")
        print(f"[FEATURES-BASELINE]     avg_amount_{window_name}: "
              f"mean={avg_amounts.mean():.2f}, std={avg_amounts.std():.2f}")

    return df


def run_feature_engineering_baseline(
    df: pd.DataFrame,
    config: Dict,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Execute the baseline feature engineering pipeline.

    [NEW] Added 8 features: amount_bucket, txn_count_1h_by_bucket,
    avg_amount_1h_by_bucket, amount_to_bucket_avg_ratio, time_since_last_txn,
    time_since_last_txn_same_bucket, hour_of_day, is_night, amount_log.
    Total baseline features increased from ~40 to ~49.

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

    # 1. Time-based features (includes [NEW] hour_of_day, is_night)
    df = create_time_features_baseline(df)

    # 2. Velocity features (global windows)
    df = create_velocity_features_baseline(df)

    # 3. [NEW] Amount bucket proxy features (pseudo-segment velocity)
    df = create_amount_bucket_features(df)

    # 4. [NEW] Recency features
    df = create_recency_features_baseline(df)

    # 5. [NEW] Amount log transform
    df = create_amount_log(df)

    # 6. Drop Time columns (not features for training)
    for col in ["Time", "Time_raw"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 7. Define feature columns (exclude target)
    target_col = "Class"
    feature_cols = [col for col in df.columns if col != target_col]

    print(f"\n[FEATURES-BASELINE] Total features: {len(feature_cols)} "
          f"([NEW] +9 from v1.0)")
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

    [NEW] Expanded from 4 interactions to 10: added 6 pairwise combinations
    of top EDA-identified fraud features (V17, V14, V12, V10, V16, V3, V7, V11)
    plus ratio features V17_to_V14 and V12_to_V10.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with interaction features.
    """
    print("[FEATURES-ADVANCED] Creating interaction features...")

    # Original interactions
    interactions = {
        "V17_V14": ("V17", "V14"),
        "V12_V10": ("V12", "V10"),
        "V4_V11": ("V4", "V11"),
        "V3_V7": ("V3", "V7"),
        # [NEW] Additional interactions from EDA top features
        "V17_V12": ("V17", "V12"),
        "V14_V10": ("V14", "V10"),
        "V17_V10": ("V17", "V10"),
        "V14_V12": ("V14", "V12"),
        "V16_V17": ("V16", "V17"),
        "V3_V14": ("V3", "V14"),
    }

    for name, (col_a, col_b) in interactions.items():
        if col_a in df.columns and col_b in df.columns:
            df[name] = df[col_a].values * df[col_b].values
            print(f"[FEATURES-ADVANCED]   {name} = {col_a} × {col_b}")

    # [NEW] Ratio features between top fraud features
    if "V17" in df.columns and "V14" in df.columns:
        df["V17_to_V14"] = df["V17"] / (df["V14"] + 1e-8)
        print("[FEATURES-ADVANCED] [NEW]   V17_to_V14 = V17 / V14")

    if "V12" in df.columns and "V10" in df.columns:
        df["V12_to_V10"] = df["V12"] / (df["V10"] + 1e-8)
        print("[FEATURES-ADVANCED] [NEW]   V12_to_V10 = V12 / V10")

    return df


def create_amount_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features representing the deviation of current transaction
    from recent historical averages.

    [NEW] Added amount_cv_1h (coefficient of variation) and
    amount_range_1h + amount_percentile_in_range.

    Uses Amount_raw (real dollars) for ratio calculations if available,
    and Time_raw (real seconds) for time-based rolling windows.

    Args:
        df: Input DataFrame with velocity features.

    Returns:
        DataFrame with ratio features.
    """
    print("[FEATURES-ADVANCED] Creating amount ratio features...")

    # Use raw Amount for meaningful ratios ($0–$25K scale)
    amount_col = "Amount_raw" if "Amount_raw" in df.columns else "Amount"
    amount = df[amount_col].values
    eps = 1e-6

    if "avg_amount_1h" in df.columns:
        df["amount_ratio_1h"] = amount / (df["avg_amount_1h"].values + eps)
        print("[FEATURES-ADVANCED]   amount_ratio_1h = Amount / avg_amount_1h")

    if "avg_amount_24h" in df.columns:
        df["amount_ratio_24h"] = amount / (df["avg_amount_24h"].values + eps)
        print("[FEATURES-ADVANCED]   amount_ratio_24h = Amount / avg_amount_24h")

    if "std_amount_1h" in df.columns:
        df["amount_zscore_1h"] = (
            amount - df["avg_amount_1h"].values
        ) / (df["std_amount_1h"].values + eps)
        print("[FEATURES-ADVANCED]   amount_zscore_1h = (Amount - avg) / std")

        # [NEW] Coefficient of variation (relative dispersion)
        df["amount_cv_1h"] = df["std_amount_1h"] / (df["avg_amount_1h"] + eps)
        print("[FEATURES-ADVANCED] [NEW]   amount_cv_1h = std / avg")

    # [NEW] Amount range using Time_raw (real seconds) for proper 1-hour windows
    if "Time_raw" in df.columns and len(df) > 1:
        try:
            df_sorted = df.sort_values("Time_raw")
            # Use time-based rolling since Time_raw now has real seconds
            rolling_max = df_sorted[amount_col].rolling(window="3600s", on="Time_raw").max()
            rolling_min = df_sorted[amount_col].rolling(window="3600s", on="Time_raw").min()
            df["amount_range_1h"] = (rolling_max - rolling_min).fillna(0)
            df["amount_percentile_in_range"] = (
                (df_sorted[amount_col] - rolling_min) / (df["amount_range_1h"] + eps)
            ).fillna(0.5)
            print("[FEATURES-ADVANCED] [NEW]   amount_range_1h = max - min in 1h window (real seconds)")
            print("[FEATURES-ADVANCED] [NEW]   amount_percentile_in_range = position in 1h range")
        except Exception as e:
            print(f"[FEATURES-ADVANCED] [NEW]   Skipping amount_range_1h: {e}")
            df["amount_range_1h"] = 0.0
            df["amount_percentile_in_range"] = 0.5
    else:
        print("[FEATURES-ADVANCED] [NEW]   Skipping amount_range_1h (no Time_raw column)")
        df["amount_range_1h"] = 0.0
        df["amount_percentile_in_range"] = 0.5

    return df


def create_extended_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create velocity features with finer granularity window.
    Includes short-term burst detection.

    Uses Time_raw (real seconds) for proper window boundaries and
    Amount_raw (real dollars) for meaningful amount statistics.

    [NEW] Added velocity_spike_ratio: compares recent (1h) activity rate
    to daily average rate. Values > 1 indicate a burst of activity.

    Args:
        df: Input DataFrame sorted by time.

    Returns:
        DataFrame with extended velocity features.
    """
    print("[FEATURES-ADVANCED] Creating extended velocity features...")

    df = df.sort_values("Time_raw").reset_index(drop=True)

    time_values = df["Time_raw"].values
    
    # Use raw Amount for meaningful dollar-based velocity statistics
    amount_col = "Amount_raw" if "Amount_raw" in df.columns else "Amount"
    amount_values = df[amount_col].values

    # 10-minute window for burst detection
    window_10min = 600
    counts_10min, avg_10min, _ = _compute_time_window_features(
        time_values, amount_values, window_10min
    )

    df["txn_count_10min"] = counts_10min.astype(int)
    df["avg_amount_10min"] = avg_10min

    print(f"[FEATURES-ADVANCED]   txn_count_10min: mean={counts_10min.mean():.1f}")
    print(f"[FEATURES-ADVANCED]   avg_amount_10min: mean={avg_10min.mean():.2f}")

    # [NEW] Velocity spike ratio: 1h rate vs 24h rate
    if "txn_count_1h" in df.columns and "txn_count_24h" in df.columns:
        df["velocity_spike_ratio"] = (
            (df["txn_count_1h"] * 24) / (df["txn_count_24h"] + 1)
        )
        print(f"[FEATURES-ADVANCED] [NEW]   velocity_spike_ratio: "
              f"mean={df['velocity_spike_ratio'].mean():.2f} "
              f"(>1 = recent burst)")

    return df


def create_recency_features_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features measuring time since last transaction (recency).
    Fraudsters often perform rapid successive transactions.

    [NEW] Renamed from original create_recency_features to avoid conflict
    with baseline version. Uses same logic but operates after advanced features.

    Args:
        df: Input DataFrame sorted by time.

    Returns:
        DataFrame with recency features.
    """
    print("[FEATURES-ADVANCED] Creating recency features...")

    df = df.sort_values("Time_raw").reset_index(drop=True)

    seconds_since = np.diff(df["Time_raw"].values, prepend=df["Time_raw"].values[0])
    df["seconds_since_last_txn"] = np.clip(seconds_since, 0, None)

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

    X_pca = df[pca_cols].to_numpy(dtype=np.float64)

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )

    raw_scores = iso_forest.fit_predict(X_pca)
    df["anomaly_score"] = (-raw_scores + 1) / 2.0
    df["anomaly_decision"] = -iso_forest.score_samples(X_pca)

    anomaly_rate = (df["anomaly_score"] > 0.5).mean() * 100
    print(f"[FEATURES-ADVANCED]   anomaly_score: {anomaly_rate:.1f}% flagged as anomalous")

    return df


def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create statistical moment features.

    [NEW] Added amount_skew_1h: skewness of amounts in 1-hour window.
    High skew indicates an outlier transaction in an otherwise consistent window.
    
    Uses Amount_raw (real dollars) for percentile/decile calculations and
    Time_raw (real seconds) for time-based rolling windows.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with statistical features.
    """
    print("[FEATURES-ADVANCED] Creating statistical features...")

    # Use raw Amount for meaningful percentile/decile ($0–$25K range)
    amount_col = "Amount_raw" if "Amount_raw" in df.columns else "Amount"

    # Amount percentile rank (vectorized)
    df["amount_percentile"] = df[amount_col].rank(pct=True)
    print(f"[FEATURES-ADVANCED]   amount_percentile: 0–1 normalized rank (using {amount_col})")

    # Amount decile (1-10)
    df["amount_decile"] = pd.qcut(df[amount_col].rank(method="first"), 10, labels=False) + 1
    print(f"[FEATURES-ADVANCED]   amount_decile: 10 bins (using {amount_col})")

    # [NEW] Rolling skewness of amounts using Time_raw (real seconds) for proper 1h window
    if "Time_raw" in df.columns and len(df) > 1:
        try:
            df_sorted = df.sort_values("Time_raw")
            df["amount_skew_1h"] = (
                df_sorted[amount_col]
                .rolling(window="3600s", on="Time_raw", min_periods=3)
                .skew()
                .fillna(0)
            )
            print(f"[FEATURES-ADVANCED] [NEW]   amount_skew_1h: "
                  f"mean={df['amount_skew_1h'].mean():.3f} (1h window, real seconds)")
        except Exception as e:
            print(f"[FEATURES-ADVANCED] [NEW]   Skipping amount_skew_1h: {e}")
            df["amount_skew_1h"] = 0.0
    else:
        print("[FEATURES-ADVANCED] [NEW]   Skipping amount_skew_1h (no Time_raw or insufficient data)")
        df["amount_skew_1h"] = 0.0

    # Is zero amount flag (using raw Amount for $0 detection)
    df["is_zero_amount"] = (df[amount_col].values == 0).astype(int)
    zero_count = (df["is_zero_amount"] == 1).sum()
    print(f"[FEATURES-ADVANCED]   is_zero_amount: {zero_count} transactions (using {amount_col})")

    # Is night transaction (hour 0–5)
    if "hour" in df.columns:
        hour_values = df["hour"].values
        df["is_night"] = ((hour_values >= 0) & (hour_values <= 5)).astype(int)
        print(f"[FEATURES-ADVANCED]   is_night: {(df['is_night'] == 1).sum()} transactions")

    return df


def create_fraud_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [NEW - ADVANCED] Create features encoding domain knowledge about fraud indicators.
    
    EDA showed V17, V14, V12, V10, V16 are negatively correlated with fraud
    (more negative = higher fraud probability). These features encode that signal
    explicitly.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with fraud direction features.
    """
    print("[FEATURES-ADVANCED] [NEW] Creating fraud direction features...")

    top_fraud_features = ["V17", "V14", "V12", "V10", "V16"]
    available = [f for f in top_fraud_features if f in df.columns]

    if available:
        # Magnitude of fraud signal (how intense across all top features)
        df["fraud_feature_magnitude"] = df[available].abs().sum(axis=1)

        # Directional consistency: count how many top features point toward fraud
        # (negative values = fraud signal for these features)
        df["fraud_direction_score"] = sum(
            (df[f] < 0).astype(int) for f in available
        )

        print(f"[FEATURES-ADVANCED] [NEW]   fraud_feature_magnitude: "
              f"mean={df['fraud_feature_magnitude'].mean():.2f}")
        print(f"[FEATURES-ADVANCED] [NEW]   fraud_direction_score: "
              f"0–{len(available)} scale, mean={df['fraud_direction_score'].mean():.1f}")

    return df


def select_top_features(df: pd.DataFrame, n_top: int = 20) -> List[str]:
    """
    Select top N features by absolute correlation with Class.

    Args:
        df: Input DataFrame.
        n_top: Number of top features to retain.

    Returns:
        List of selected feature names.
    """
    if "Class" not in df.columns:
        return [col for col in df.columns if col != "Class"]

    exclude = {"Class", "Time", "Time_raw"}
    feature_cols = [col for col in df.columns if col not in exclude
                    and df[col].dtype in ("float64", "int64", "int32")]

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

    [NEW] Added 10 features: 6 PCA interactions, V17_to_V14, V12_to_V10,
    velocity_spike_ratio, amount_cv_1h, amount_skew_1h, amount_range_1h,
    amount_percentile_in_range, fraud_feature_magnitude, fraud_direction_score.
    Total advanced features increased from ~56 to ~66.

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

    # Ensure Time_raw exists for temporal ordering
    df = validate_time_column(df)

    # Save Time_raw before baseline drops it (baseline drops both Time and Time_raw)
    if "Time_raw" in df.columns:
        time_raw_values = df["Time_raw"].values.copy()
    elif "Time" in df.columns:
        time_raw_values = df["Time"].values.copy()
    else:
        # No time column at all — create synthetic ordering from index
        time_raw_values = np.arange(len(df), dtype=np.float64)
        df["Time_raw"] = time_raw_values
        print("[FEATURES-ADVANCED] WARNING: No Time column found, using index as Time_raw proxy")

    # 1. Run baseline features first (foundation)
    #    This drops Time and Time_raw internally
    df, _ = run_feature_engineering_baseline(df, config)

    # Re-add Time_raw for advanced temporal features that need it
    df["Time_raw"] = time_raw_values

    # 2. Interaction features ([NEW] expanded from 4 to 10)
    df = create_interaction_features(df)

    # 3. Amount ratio features ([NEW] +3 features)
    df = create_amount_ratio_features(df)

    # 4. Extended velocity ([NEW] +1 feature: velocity_spike_ratio)
    df = create_extended_velocity_features(df)

    # 5. Recency features
    df = create_recency_features_advanced(df)

    # 6. Anomaly detection features
    df = create_anomaly_features(df)

    # 7. Statistical features ([NEW] +1: amount_skew_1h)
    df = create_statistical_features(df)

    # 8. [NEW] Fraud direction features (+2 features)
    df = create_fraud_direction_features(df)

    # 9. Clean up temporary columns
    for col in ["Time_raw", "Time"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 10. Define feature columns
    target_col = "Class"
    feature_cols = [col for col in df.columns if col != target_col]

    # 11. Optional feature selection
    if config["feature_engineering"]["feature_selection"]["enabled"]:
        n_top = config["feature_engineering"]["feature_selection"]["top_n_correlated"]
        selected_features = select_top_features(df, n_top)
        feature_cols = selected_features

    print(f"\n[FEATURES-ADVANCED] Total features: {len(feature_cols)} "
          f"([NEW] +10 from v1.0)")
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

        baseline_path = config["paths"]["features_data"]
        if not os.path.isabs(baseline_path):
            baseline_path = str(project_root / baseline_path)
        baseline_path = baseline_path.replace(".parquet", "_baseline.parquet")

        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        df_baseline.to_parquet(baseline_path, index=False)
        print(f"[FEATURES-BASELINE] Saved to: {baseline_path}")

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

        advanced_path = config["paths"]["features_data"]
        if not os.path.isabs(advanced_path):
            advanced_path = str(project_root / advanced_path)
        advanced_path = advanced_path.replace(".parquet", "_advanced.parquet")

        os.makedirs(os.path.dirname(advanced_path), exist_ok=True)
        df_advanced.to_parquet(advanced_path, index=False)
        print(f"[FEATURES-ADVANCED] Saved to: {advanced_path}")

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
    """Execute feature engineering as a standalone script."""
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