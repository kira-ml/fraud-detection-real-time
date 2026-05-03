"""
Data Ingestion Module
Baseline implementation for loading raw credit card transaction data.
Reads CSV, performs sanity checks, and returns DataFrame for downstream pipeline stages.
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
# Data Ingestion
# ================================

def read_raw_data(file_path: str) -> pd.DataFrame:
    """
    Read raw CSV file into a Pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file.
    
    Returns:
        DataFrame containing the raw transaction data.
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if df.empty:
        raise ValueError(f"Data file is empty: {file_path}")
    
    return df


def validate_schema(
    df: pd.DataFrame, 
    expected_rows: int, 
    expected_columns: int
) -> Tuple[bool, list]:
    """
    Validate the DataFrame against expected dimensions and schema.
    
    Args:
        df: Input DataFrame to validate.
        expected_rows: Expected number of rows.
        expected_columns: Expected number of columns.
    
    Returns:
        Tuple of (is_valid, list_of_warnings).
    """
    warnings = []
    is_valid = True
    
    actual_rows, actual_cols = df.shape
    
    if actual_rows != expected_rows:
        warnings.append(
            f"Row count mismatch: expected {expected_rows:,}, got {actual_rows:,}"
        )
    
    if actual_cols != expected_columns:
        warnings.append(
            f"Column count mismatch: expected {expected_columns}, got {actual_cols}"
        )
        is_valid = False
    
    # Verify all columns are numeric
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        warnings.append(f"Non-numeric columns detected: {non_numeric_cols}")
        is_valid = False
    
    # Check for null values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        for col, count in null_cols.items():
            warnings.append(f"Column '{col}' has {count} null values")
    
    return is_valid, warnings


def log_data_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics of the ingested data to console.
    
    Args:
        df: Input DataFrame to summarize.
    """
    rows, cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    print("=" * 60)
    print("DATA INGESTION SUMMARY")
    print("=" * 60)
    print(f"Rows:        {rows:,}")
    print(f"Columns:     {cols}")
    print(f"Memory:      {memory_mb:.2f} MB")
    print(f"Data types:")
    for dtype in df.dtypes.value_counts().index:
        count = df.dtypes.value_counts()[dtype]
        print(f"  {dtype}: {count} columns")
    print("-" * 60)
    print("Column list:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col} ({df[col].dtype})")
    print("-" * 60)
    print("First 5 rows:")
    print(df.head().to_string())
    print("=" * 60)


def ingest_data(config_path: Optional[str] = None) -> pd.DataFrame:
    """
    Main ingestion function: load config, read data, validate, and return DataFrame.
    
    Args:
        config_path: Optional path to pipeline config YAML.
    
    Returns:
        Validated pandas DataFrame containing raw transaction data.
    """
    # Load configuration
    config = load_config(config_path)
    
    raw_path = config["paths"]["raw_data"]
    expected_rows = config["ingestion"]["expected_rows"]
    expected_columns = config["ingestion"]["expected_columns"]
    
    # Resolve path relative to project root if not absolute
    if not os.path.isabs(raw_path):
        project_root = Path(__file__).resolve().parent.parent
        raw_path = str(project_root / raw_path)
    
    print(f"[INGEST] Source: {raw_path}")
    
    # Read data
    df_raw = read_raw_data(raw_path)
    print(f"[INGEST] Loaded {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")
    
    # Validate
    is_valid, warnings = validate_schema(df_raw, expected_rows, expected_columns)
    
    if warnings:
        print("[INGEST] Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not is_valid:
        raise ValueError(
            "Data validation failed. See warnings above for details."
        )
    
    print("[INGEST] Schema validation passed.")
    
    # Summary
    log_data_summary(df_raw)
    
    return df_raw


# ================================
# Entry Point
# ================================

def main():
    """
    Execute data ingestion as a standalone script.
    Can be run directly: python src/ingest.py
    """
    try:
        df_raw = ingest_data()
        print(f"\n[INGEST] Successfully ingested {len(df_raw):,} transactions.")
        return df_raw
        
    except FileNotFoundError as e:
        print(f"[INGEST] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        
    except ValueError as e:
        print(f"[INGEST] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"[INGEST] UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()