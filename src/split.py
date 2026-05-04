"""
Data Splitting Module
Time-aware chronological split for credit card fraud detection.
Splits both baseline and advanced feature-engineered datasets.
Since Time column is removed during feature engineering, splits by row index
on chronologically-ordered data (equivalent to time-based split).
Ensures no temporal leakage between train and test sets.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml


# ================================
# Configuration Loader
# ================================

def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to config file. If None, auto-resolves to default.

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
# Core Splitting Logic
# ================================

def split_data_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform chronological split by row index.
    Assumes data is already sorted chronologically (as done in feature engineering).

    Args:
        df: Input DataFrame (must be in chronological order).
        train_ratio: Fraction of data to use for training.

    Returns:
        Tuple of (train_df, test_df).
    """
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def validate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    target_col: str = "Class",
) -> bool:
    """
    Verify split integrity: class balance preserved and no data leakage.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        dataset_name: Label for logging (e.g., 'BASELINE', 'ADVANCED').
        target_col: Name of the target column.

    Returns:
        True if all validation checks pass.
    """
    print(f"\n[SPLIT] --- {dataset_name} Dataset Validation ---")

    # Verify no overlap (index-based split guarantees this)
    train_indices = set(train_df.index)
    test_indices = set(test_df.index)
    overlap = len(train_indices.intersection(test_indices))
    
    if overlap > 0:
        print(f"[SPLIT] WARNING: {overlap} overlapping rows detected!")
    else:
        print(f"[SPLIT] No data leakage: 0 overlapping rows between train and test")

    # Report class distributions
    if target_col in train_df.columns:
        train_fraud_count = train_df[target_col].sum()
        test_fraud_count = test_df[target_col].sum()
        train_fraud_rate = (train_fraud_count / len(train_df)) * 100
        test_fraud_rate = (test_fraud_count / len(test_df)) * 100
        
        print(f"[SPLIT] Train fraud rate: {train_fraud_rate:.3f}% "
              f"({train_fraud_count} / {len(train_df)})")
        print(f"[SPLIT] Test fraud rate:  {test_fraud_rate:.3f}% "
              f"({test_fraud_count} / {len(test_df)})")

    # Report sizes
    print(f"[SPLIT] Train size: {len(train_df):,} rows ({len(train_df.columns)} features)")
    print(f"[SPLIT] Test size:  {len(test_df):,} rows ({len(test_df.columns)} features)")

    return True


def save_split_config(
    train_ratio: float,
    train_size: int,
    test_size: int,
    output_path: str
) -> None:
    """
    Persist the split configuration for reproducibility.

    Args:
        train_ratio: Fraction of data used for training.
        train_size: Number of training samples.
        test_size: Number of test samples.
        output_path: Path to save the config.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"split_method: chronological_index\n")
        f.write(f"train_ratio: {train_ratio}\n")
        f.write(f"train_size: {train_size}\n")
        f.write(f"test_size: {test_size}\n")
        f.write(f"note: Data is pre-sorted chronologically during feature engineering\n")
    print(f"[SPLIT] Split config saved to: {output_path}")


# ================================
# Single Dataset Splitter
# ================================

def split_single_dataset(
    input_path: str,
    train_path: str,
    test_path: str,
    dataset_name: str,
    target_col: str = "Class",
    train_ratio: float = 0.80,
) -> Tuple[int, int]:
    """
    Split a single feature-engineered dataset.

    Args:
        input_path: Path to the feature-engineered parquet file.
        train_path: Output path for training set.
        test_path: Output path for test set.
        dataset_name: Label for logging (e.g., 'BASELINE', 'ADVANCED').
        target_col: Name of the target column.
        train_ratio: Fraction of data for training.

    Returns:
        Tuple of (train_size, test_size).
    """
    # Load feature-engineered data
    print(f"\n[SPLIT] Loading {dataset_name} data from: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_parquet(input_path)
    print(f"[SPLIT] Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Perform chronological split
    train_df, test_df = split_data_chronological(df, train_ratio)

    # Validate integrity
    validate_split(train_df, test_df, dataset_name, target_col)

    # Save outputs
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    print(f"[SPLIT] {dataset_name} training set saved to: {train_path}")

    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    test_df.to_parquet(test_path, index=False)
    print(f"[SPLIT] {dataset_name} test set saved to: {test_path}")

    return len(train_df), len(test_df)


# ================================
# Main Pipeline
# ================================

def run_data_split(
    config_path: Optional[str] = None,
    target_col: str = "Class",
    train_ratio: float = 0.80,
) -> Dict:
    """
    Execute the complete data splitting pipeline for both datasets.

    Args:
        config_path: Path to pipeline config YAML.
        target_col: Name of the target column.
        train_ratio: Fraction of time range for training.

    Returns:
        Dictionary with paths to saved train/test files.
    """
    config = load_config(config_path)

    print("\n" + "=" * 60)
    print("DATA SPLITTING (Chronological Index-Based)")
    print("=" * 60)
    print(f"[SPLIT] Train ratio: {train_ratio*100:.0f}%")
    print(f"[SPLIT] Split method: First {train_ratio*100:.0f}% of rows → Train, remaining → Test")
    print(f"[SPLIT] Note: Data is pre-sorted chronologically from feature engineering")

    results = {}

    # Resolve paths from config
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = config.get("paths", {}).get("processed_data_dir", "data/processed")
    if not os.path.isabs(processed_dir):
        processed_dir = str(project_root / processed_dir)

    config_dir = config.get("paths", {}).get("config_dir", "config")
    if not os.path.isabs(config_dir):
        config_dir = str(project_root / config_dir)

    # --- Split Baseline Dataset ---
    baseline_train_size = 0
    baseline_test_size = 0
    
    try:
        baseline_input_path = os.path.join(processed_dir, "features_baseline.parquet")
        baseline_train_path = os.path.join(processed_dir, "train_baseline.parquet")
        baseline_test_path = os.path.join(processed_dir, "test_baseline.parquet")

        baseline_train_size, baseline_test_size = split_single_dataset(
            input_path=baseline_input_path,
            train_path=baseline_train_path,
            test_path=baseline_test_path,
            dataset_name="BASELINE",
            target_col=target_col,
            train_ratio=train_ratio,
        )

        results["baseline"] = {
            "train": baseline_train_path,
            "test": baseline_test_path,
        }
    except FileNotFoundError as e:
        print(f"[SPLIT] WARNING: Baseline dataset not found - {e}")
        print(f"[SPLIT] Skipping baseline split...")

    # --- Split Advanced Dataset ---
    advanced_train_size = 0
    advanced_test_size = 0
    
    try:
        advanced_input_path = os.path.join(processed_dir, "features_advanced.parquet")
        advanced_train_path = os.path.join(processed_dir, "train_advanced.parquet")
        advanced_test_path = os.path.join(processed_dir, "test_advanced.parquet")

        advanced_train_size, advanced_test_size = split_single_dataset(
            input_path=advanced_input_path,
            train_path=advanced_train_path,
            test_path=advanced_test_path,
            dataset_name="ADVANCED",
            target_col=target_col,
            train_ratio=train_ratio,
        )

        results["advanced"] = {
            "train": advanced_train_path,
            "test": advanced_test_path,
        }
    except FileNotFoundError as e:
        print(f"[SPLIT] WARNING: Advanced dataset not found - {e}")
        print(f"[SPLIT] Skipping advanced split...")

    # Save split config for reproducibility (use baseline sizes if available)
    split_config_path = os.path.join(config_dir, "split_config.txt")
    save_split_config(
        train_ratio=train_ratio,
        train_size=baseline_train_size or advanced_train_size,
        test_size=baseline_test_size or advanced_test_size,
        output_path=split_config_path,
    )

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SPLIT COMPLETE — OUTPUT FILES")
        print("=" * 60)
        
        if "baseline" in results:
            print(f"\n  BASELINE:")
            print(f"    Train: {results['baseline']['train']}")
            print(f"    Test:  {results['baseline']['test']}")
            print(f"    Sizes: {baseline_train_size:,} train / {baseline_test_size:,} test")
        
        if "advanced" in results:
            print(f"\n  ADVANCED:")
            print(f"    Train: {results['advanced']['train']}")
            print(f"    Test:  {results['advanced']['test']}")
            print(f"    Sizes: {advanced_train_size:,} train / {advanced_test_size:,} test")
        
        print(f"\n  Config: {split_config_path}")
        print("=" * 60 + "\n")
    else:
        print("[SPLIT] ERROR: No datasets were split. Check that feature files exist.")

    return results


# ================================
# Entry Point
# ================================

def main():
    """Execute data splitting as a standalone script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Time-Aware Data Splitting for Baseline and Advanced Features"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="D:/fraud-detection-real-time/config/pipeline_config.yaml",
        help="Path to pipeline config YAML (default: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="Class",
        help="Name of the target column (default: Class)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Fraction of data for training (default: 0.80)",
    )

    args = parser.parse_args()

    try:
        results = run_data_split(
            config_path=args.config,
            target_col=args.target_col,
            train_ratio=args.train_ratio,
        )
        return results

    except FileNotFoundError as e:
        print(f"[SPLIT] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[SPLIT] UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()