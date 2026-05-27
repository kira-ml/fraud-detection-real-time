"""
Data Splitting Module
Time-aware chronological split for credit card fraud detection.
Splits cleaned/preprocessed data BEFORE feature engineering to prevent temporal leakage.
Accepts DataFrames directly for pipeline composability.
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
    time_col: str = "Time",
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Perform time-aware chronological split.
    
    Sorts by Time column ascending, then takes the first train_ratio% of
    transactions for training and the remainder for testing. This ensures
    no future transactions leak into training data.

    Args:
        df: Input DataFrame (must contain a Time column for ordering).
        train_ratio: Fraction of time range to use for training (default: 0.80).
        time_col: Name of the time column (default: 'Time').

    Returns:
        Tuple of (train_df, test_df, split_timestamp).
        split_timestamp is the Time value at the split boundary (for logging).
    """
    if time_col not in df.columns:
        raise ValueError(
            f"Time column '{time_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Sort chronologically
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    
    # Find the split point by time value
    max_time = df_sorted[time_col].max()
    min_time = df_sorted[time_col].min()
    split_timestamp = min_time + train_ratio * (max_time - min_time)
    
    # Split: all transactions before split_timestamp → train
    train_mask = df_sorted[time_col] <= split_timestamp
    train_df = df_sorted[train_mask].copy()
    test_df = df_sorted[~train_mask].copy()
    
    # Verify split is non-empty
    if len(train_df) == 0:
        raise ValueError("Training set is empty. Check train_ratio or time column.")
    if len(test_df) == 0:
        raise ValueError("Test set is empty. Reduce train_ratio.")
    
    print(f"[SPLIT] Time range: {min_time:.0f} → {max_time:.0f}")
    print(f"[SPLIT] Split timestamp: {split_timestamp:.0f}")
    print(f"[SPLIT] Train: {len(train_df):,} rows (Time {df_sorted[time_col].iloc[0]:.0f} → {split_timestamp:.0f})")
    print(f"[SPLIT] Test:  {len(test_df):,} rows (Time {split_timestamp:.0f} → {df_sorted[time_col].iloc[-1]:.0f})")
    print(f"[SPLIT] Temporal leakage check: max train time ({train_df[time_col].max():.0f}) < "
          f"min test time ({test_df[time_col].min():.0f}) → "
          f"{'PASS' if train_df[time_col].max() < test_df[time_col].min() else 'FAIL'}")
    
    return train_df, test_df, split_timestamp


def validate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Class",
) -> Dict:
    """
    Verify split integrity and report class distributions.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        target_col: Name of the target column.

    Returns:
        Dictionary with validation results.
    """
    print(f"\n[SPLIT] --- Split Validation ---")
    
    total = len(train_df) + len(test_df)
    train_pct = len(train_df) / total * 100
    test_pct = len(test_df) / total * 100
    
    print(f"[SPLIT] Total rows:   {total:,}")
    print(f"[SPLIT] Train:        {len(train_df):,} ({train_pct:.1f}%)")
    print(f"[SPLIT] Test:         {len(test_df):,} ({test_pct:.1f}%)")
    
    # Class distribution
    if target_col in train_df.columns and target_col in test_df.columns:
        train_fraud = int(train_df[target_col].sum())
        test_fraud = int(test_df[target_col].sum())
        total_fraud = train_fraud + test_fraud
        
        train_fraud_rate = (train_fraud / len(train_df)) * 100
        test_fraud_rate = (test_fraud / len(test_df)) * 100
        overall_fraud_rate = (total_fraud / total) * 100
        
        print(f"[SPLIT] Train fraud:  {train_fraud} ({train_fraud_rate:.4f}%)")
        print(f"[SPLIT] Test fraud:   {test_fraud} ({test_fraud_rate:.4f}%)")
        print(f"[SPLIT] Total fraud:  {total_fraud} ({overall_fraud_rate:.4f}%)")
        
        # Check if fraud is preserved in both splits
        fraud_preserved = (train_fraud > 0) and (test_fraud > 0)
        if not fraud_preserved:
            print(f"[SPLIT] WARNING: Fraud class not present in both splits!")
    
    # Check that train ends before test begins (temporal integrity)
    # Time column may have been scaled — skip if not present
    if "Time" in train_df.columns and "Time" in test_df.columns:
        max_train_time = train_df["Time"].max()
        min_test_time = test_df["Time"].min()
        temporal_integrity = max_train_time < min_test_time
        print(f"[SPLIT] Temporal integrity: {'PASS' if temporal_integrity else 'FAIL'}")
    
    # No overlap check (using index values if original indices differ)
    train_idx = set(train_df.index)
    test_idx = set(test_df.index)
    overlap = len(train_idx.intersection(test_idx))
    print(f"[SPLIT] Index overlap: {overlap} rows ({'PASS' if overlap == 0 else 'WARNING'})")
    
    return {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_ratio": round(train_pct / 100, 4),
        "train_fraud_count": train_fraud if target_col in train_df.columns else None,
        "test_fraud_count": test_fraud if target_col in test_df.columns else None,
    }


def save_split_config(
    train_ratio: float,
    split_timestamp: float,
    train_size: int,
    test_size: int,
    output_path: str,
) -> None:
    """
    Persist split configuration for reproducibility.

    Args:
        train_ratio: Fraction of time range used for training.
        split_timestamp: Time value at the split boundary.
        train_size: Number of training samples.
        test_size: Number of test samples.
        output_path: Path to save the config file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"split_method: chronological_time_based\n")
        f.write(f"train_ratio: {train_ratio}\n")
        f.write(f"split_timestamp: {split_timestamp:.6f}\n")
        f.write(f"train_size: {train_size}\n")
        f.write(f"test_size: {test_size}\n")
        f.write(f"note: Split occurs BEFORE feature engineering to prevent temporal leakage\n")
        f.write(f"note: Train and test sets are engineered independently\n")
    
    print(f"[SPLIT] Split config saved to: {output_path}")


# ================================
# Main Pipeline Function
# ================================
def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    target_col: str = "Class",
    time_col: str = "Time_raw",
    save_paths: Optional[Dict[str, str]] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Split cleaned data into train/test sets BEFORE feature engineering.
    
    This is the main entry point for the pipeline orchestrator.
    Splits chronologically by Time_raw column (raw seconds), then optionally saves to disk.
    Falls back to 'Time' if Time_raw is not available.

    Args:
        df: Cleaned/preprocessed DataFrame from preprocessing step.
        train_ratio: Fraction of time range for training (default: 0.80).
        target_col: Name of the target column (default: 'Class').
        time_col: Name of the time column (default: 'Time_raw' for raw seconds).
                  Falls back to 'Time' if Time_raw doesn't exist.
        save_paths: Optional dict with 'train' and 'test' keys for output paths.
        config_path: Optional path to pipeline config for saving split config.

    Returns:
        Dictionary containing:
            - 'train_df': Training DataFrame
            - 'test_df': Test DataFrame
            - 'split_timestamp': Time value at split boundary
            - 'train_path': Path to saved train file (if save_paths provided)
            - 'test_path': Path to saved test file (if save_paths provided)
            - 'validation': Validation results dict
    """
    # Use Time_raw (raw seconds) for correct chronological split
    # Fall back to Time if Time_raw doesn't exist (backward compatibility)
    if time_col not in df.columns:
        if "Time" in df.columns:
            print(f"[SPLIT] WARNING: '{time_col}' not found, falling back to 'Time' (scaled)")
            time_col = "Time"
        else:
            raise ValueError(f"Neither '{time_col}' nor 'Time' column found in DataFrame")
    
    print("\n" + "=" * 60)
    print("DATA SPLITTING (Time-Aware Chronological)")
    print("=" * 60)
    print(f"[SPLIT] Input shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"[SPLIT] Train ratio: {train_ratio*100:.0f}%")
    print(f"[SPLIT] Time column: {time_col}")
    
    # Perform the split
    train_df, test_df, split_timestamp = split_data_chronological(
        df, train_ratio=train_ratio, time_col=time_col
    )
    
    # Validate
    validation_results = validate_split(train_df, test_df, target_col=target_col)
    
    # Save train and test sets if paths provided
    result = {
        "train_df": train_df,
        "test_df": test_df,
        "split_timestamp": split_timestamp,
        "validation": validation_results,
    }
    
    if save_paths:
        if "train" in save_paths:
            os.makedirs(os.path.dirname(save_paths["train"]), exist_ok=True)
            train_df.to_parquet(save_paths["train"], index=False)
            print(f"[SPLIT] Train set saved to: {save_paths['train']}")
            result["train_path"] = save_paths["train"]
        
        if "test" in save_paths:
            os.makedirs(os.path.dirname(save_paths["test"]), exist_ok=True)
            test_df.to_parquet(save_paths["test"], index=False)
            print(f"[SPLIT] Test set saved to: {save_paths['test']}")
            result["test_path"] = save_paths["test"]
    
    # Save split config for reproducibility
    if config_path:
        config = load_config(config_path)
        config_dir = config.get("paths", {}).get("config_dir", "config")
        project_root = Path(__file__).resolve().parent.parent
        if not os.path.isabs(config_dir):
            config_dir = str(project_root / config_dir)
        
        split_config_path = os.path.join(config_dir, "split_config.txt")
        save_split_config(
            train_ratio=train_ratio,
            split_timestamp=split_timestamp,
            train_size=len(train_df),
            test_size=len(test_df),
            output_path=split_config_path,
        )
    
    print("=" * 60 + "\n")
    
    return result


# ================================
# Entry Point (Standalone Mode)
# ================================

def main():
    """
    Execute data splitting as a standalone script.
    Loads cleaned data from disk, splits, and saves train/test sets.
    
    Note: Standalone mode loads from cleaned.parquet. In pipeline mode,
    use split_data(df) directly to pass DataFrame from preprocessing step.
    """
    import argparse
    
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    
    parser = argparse.ArgumentParser(
        description="Time-Aware Data Splitting (Split BEFORE Feature Engineering)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Fraction of time range for training (default: 0.80)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save train/test sets to disk (default: True)",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        
        # Load cleaned data
        cleaned_path = config["paths"]["cleaned_data"]
        project_root = Path(__file__).resolve().parent.parent
        if not os.path.isabs(cleaned_path):
            cleaned_path = str(project_root / cleaned_path)
        
        print(f"[SPLIT] Loading cleaned data from: {cleaned_path}")
        df_cleaned = pd.read_parquet(cleaned_path)
        
        # Set up save paths
        save_paths = None
        if args.save:
            processed_dir = config.get("paths", {}).get("processed_data_dir", "data/processed")
            if not os.path.isabs(processed_dir):
                processed_dir = str(project_root / processed_dir)
            
            save_paths = {
                "train": os.path.join(processed_dir, "train_raw.parquet"),
                "test": os.path.join(processed_dir, "test_raw.parquet"),
            }
        
        # Split
        result = split_data(
            df=df_cleaned,
            train_ratio=args.train_ratio,
            target_col="Class",
            time_col="Time",
            save_paths=save_paths,
            config_path=args.config,
        )
        
        print(f"[SPLIT] Complete. Train: {result['validation']['train_size']:,} rows, "
              f"Test: {result['validation']['test_size']:,} rows")
        
        return result

    except FileNotFoundError as e:
        print(f"[SPLIT] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[SPLIT] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[SPLIT] UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()