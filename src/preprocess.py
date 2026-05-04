"""
Data Preprocessing Module
Baseline implementation for cleaning and scaling credit card transaction data.
Handles duplicate removal, feature scaling, and prepares data for downstream feature engineering.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import RobustScaler, StandardScaler


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
# Data Cleaning
# ================================

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from the DataFrame.
    
    Args:
        df: Input DataFrame potentially containing duplicates.
    
    Returns:
        Tuple of (deduplicated DataFrame, number of rows removed).
    """
    initial_count = len(df)
    df_deduped = df.drop_duplicates().reset_index(drop=True)
    removed_count = initial_count - len(df_deduped)
    
    if removed_count > 0:
        print(f"[PREPROCESS] Removed {removed_count:,} duplicate rows "
              f"({removed_count / initial_count * 100:.4f}% of data)")
    else:
        print("[PREPROCESS] No duplicate rows found.")
    
    return df_deduped, removed_count


def validate_no_nulls(df: pd.DataFrame) -> None:
    """
    Verify no null values exist (dataset is pre-cleaned per validation report).
    
    Args:
        df: Input DataFrame to check.
    
    Raises:
        ValueError: If null values are found.
    """
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    
    if not null_cols.empty:
        raise ValueError(
            f"Unexpected null values found: {null_cols.to_dict()}"
        )
    
    print("[PREPROCESS] Null check passed: 0 null values confirmed.")


def check_negative_amounts(df: pd.DataFrame) -> int:
    """
    Check for negative transaction amounts (should be zero per validation).
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Count of negative amounts.
    """
    negative_count = int((df["Amount"] < 0).sum())
    
    if negative_count > 0:
        print(f"[PREPROCESS] WARNING: Found {negative_count} negative amounts. "
              f"Setting to 0.")
        df.loc[df["Amount"] < 0, "Amount"] = 0.0
    
    return negative_count


# ================================
# Feature Scaling
# ================================

def create_scaler(scaler_type: str) -> object:
    """
    Create a scaler instance based on configuration.
    
    Args:
        scaler_type: Type of scaler ('standard' or 'robust').
    
    Returns:
        Scaler instance.
    """
    if scaler_type == "robust":
        return RobustScaler()
    else:
        return StandardScaler()


def scale_features(
    df: pd.DataFrame,
    feature_columns: list,
    scaler_type: str,
) -> Tuple[pd.DataFrame, object]:
    """
    Scale specified features using the configured scaler type.
    
    Args:
        df: Input DataFrame.
        feature_columns: List of column names to scale.
        scaler_type: 'standard' for StandardScaler, 'robust' for RobustScaler.
    
    Returns:
        Tuple of (DataFrame with scaled features, fitted scaler object).
    """
    scaler = create_scaler(scaler_type)
    
    print(f"[PREPROCESS] Scaling features {feature_columns} using {scaler_type} scaler...")
    
    # Extract features to scale
    X = df[feature_columns].values
    
    # Fit and transform
    X_scaled = scaler.fit_transform(X)
    
    # Replace in DataFrame
    df_scaled = df.copy()
    df_scaled[feature_columns] = X_scaled
    
    # Log scaling statistics
    for i, col in enumerate(feature_columns):
        mean_val = X_scaled[:, i].mean()
        std_val = X_scaled[:, i].std()
        print(f"[PREPROCESS]   {col}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    return df_scaled, scaler


def log_transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to Amount to reduce right-skew.
    Creates Amount_log = log(Amount + 1).
    
    Args:
        df: Input DataFrame.
    
    Returns:
        DataFrame with new Amount_log column.
    """
    df["Amount_log"] = np.log1p(df["Amount"])
    
    skew_before = df["Amount"].skew()
    skew_after = df["Amount_log"].skew()
    
    print(f"[PREPROCESS] Amount log-transform: skew {skew_before:.2f} -> {skew_after:.2f}")
    
    return df


# ================================
# Main Preprocessing Pipeline
# ================================

def preprocess_data(
    df: pd.DataFrame,
    config_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, object]:
    """
    Run the full preprocessing pipeline on the validated DataFrame.
    
    Args:
        df: Validated raw DataFrame from Data Validation.
        config_path: Optional path to pipeline config YAML.
    
    Returns:
        Tuple of (cleaned DataFrame, fitted scaler object).
    """
    config = load_config(config_path)
    
    scaler_type = config["preprocessing"]["scaler_type"]
    scale_features_list = config["preprocessing"]["scale_features"]
    remove_duplicates_flag = config["preprocessing"]["remove_duplicates"]
    apply_log_transform = config["preprocessing"]["log_transform_amount"]
    
    print("[PREPROCESS] Starting data preprocessing...")
    print(f"[PREPROCESS] Input shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # 1. Validate no nulls
    print("[PREPROCESS] Validating no null values...")
    validate_no_nulls(df)
    
    # 2. Remove duplicates
    if remove_duplicates_flag:
        print("[PREPROCESS] Removing duplicate rows...")
        df, removed = remove_duplicates(df)
    else:
        print("[PREPROCESS] Skipping duplicate removal (config setting).")
        removed = 0
    
    # 3. Check for negative amounts
    print("[PREPROCESS] Checking for negative amounts...")
    negative_count = check_negative_amounts(df)
    
    # 4. Scale Amount and Time
    print("[PREPROCESS] Scaling numerical features...")
    df, scaler = scale_features(df, scale_features_list, scaler_type)
    
    # 5. Log transform Amount
    if apply_log_transform:
        print("[PREPROCESS] Applying log transform to Amount...")
        df = log_transform_amount(df)
    
    print(f"[PREPROCESS] Output shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"[PREPROCESS] Removed {removed:,} duplicates, handled {negative_count} negative amounts")
    
    return df, scaler


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the cleaned DataFrame to Parquet format.
    
    Args:
        df: Cleaned DataFrame.
        output_path: Path to save the Parquet file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[PREPROCESS] Cleaned data saved to: {output_path}")


def save_scaler(scaler: object, output_path: str) -> None:
    """
    Save the fitted scaler to disk for inference use.
    
    Args:
        scaler: Fitted scaler object.
        output_path: Path to save the scaler.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(scaler, output_path)
    print(f"[PREPROCESS] Scaler saved to: {output_path}")


def print_preprocessing_summary(df: pd.DataFrame, scaler_type: str) -> None:
    """
    Print a summary of the preprocessed data to console.
    
    Args:
        df: Preprocessed DataFrame.
        scaler_type: Type of scaler used.
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Rows:             {df.shape[0]:,}")
    print(f"Columns:          {df.shape[1]}")
    print(f"Scaler:           {scaler_type}")
    print(f"Memory:           {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("-" * 60)
    print("Amount statistics (scaled):")
    if "Amount" in df.columns:
        print(f"  Mean:           {df['Amount'].mean():.4f}")
        print(f"  Std:            {df['Amount'].std():.4f}")
        print(f"  Min:            {df['Amount'].min():.4f}")
        print(f"  Max:            {df['Amount'].max():.4f}")
    if "Amount_log" in df.columns:
        print(f"  Amount_log mean: {df['Amount_log'].mean():.4f}")
        print(f"  Amount_log std:  {df['Amount_log'].std():.4f}")
    print("-" * 60)
    print("Time statistics (scaled):")
    if "Time" in df.columns:
        print(f"  Mean:           {df['Time'].mean():.4f}")
        print(f"  Std:            {df['Time'].std():.4f}")
    print("=" * 60 + "\n")


# ================================
# Entry Point
# ================================

def main():
    """
    Execute data preprocessing as a standalone script.
    Loads validated data, applies cleaning and scaling, saves outputs.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate import validate_data, load_config as val_load_config
    
    try:
        # Load configuration for output paths
        config = load_config()
        
        # Load validated DataFrame
        print("[PREPROCESS] Loading validated data...")
        
        # Import and run ingestion + validation
        from ingest import ingest_data
        df_raw = ingest_data()
        df_validated, _ = validate_data(df_raw)
        
        # Run preprocessing
        df_cleaned, scaler = preprocess_data(df_validated)
        
        # Save outputs
        cleaned_path = config["paths"]["cleaned_data"]
        scaler_path = config["paths"]["scaler"]
        
        if not os.path.isabs(cleaned_path):
            project_root = Path(__file__).resolve().parent.parent
            cleaned_path = str(project_root / cleaned_path)
        
        if not os.path.isabs(scaler_path):
            project_root = Path(__file__).resolve().parent.parent
            scaler_path = str(project_root / scaler_path)
        
        save_cleaned_data(df_cleaned, cleaned_path)
        save_scaler(scaler, scaler_path)
        
        # Print summary
        scaler_type = config["preprocessing"]["scaler_type"]
        print_preprocessing_summary(df_cleaned, scaler_type)
        
        print(f"[PREPROCESS] Pipeline complete. {len(df_cleaned):,} rows ready for feature engineering.")
        
        return df_cleaned, scaler
        
    except FileNotFoundError as e:
        print(f"[PREPROCESS] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        
    except ValueError as e:
        print(f"[PREPROCESS] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"[PREPROCESS] UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()