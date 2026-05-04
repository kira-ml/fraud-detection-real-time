"""
Data Validation Module
Baseline implementation for validating raw credit card transaction data.
Uses pandera to enforce schema constraints and data quality rules.
Generates a validation report and passes clean data to downstream stages.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandera.pandas as pa
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
# Schema Definition
# ================================

def build_schema(config: Dict) -> pa.DataFrameSchema:
    """
    Build a pandera schema from pipeline configuration.
    
    Args:
        config: Pipeline configuration dictionary.
    
    Returns:
        pandera DataFrameSchema object.
    """
    amount_min = config["validation"]["amount_min"]
    amount_max = config["validation"]["amount_max"]
    class_values = config["validation"]["class_values"]
    
    schema = pa.DataFrameSchema(
        columns={
            "Time": pa.Column(
                float,
                nullable=False,
                description="Seconds elapsed since first transaction",
            ),
            **{
                f"V{i}": pa.Column(
                    float,
                    nullable=False,
                    description=f"PCA-transformed feature V{i}",
                )
                for i in range(1, 29)
            },
            "Amount": pa.Column(
                float,
                checks=[
                    pa.Check.ge(amount_min),
                    pa.Check.le(amount_max),
                ],
                nullable=False,
                description="Transaction amount in local currency",
            ),
            "Class": pa.Column(
                int,
                checks=[
                    pa.Check.isin(class_values),
                ],
                nullable=False,
                description="Target: 0 = legitimate, 1 = fraud",
            ),
        },
        strict=True,
        coerce=False,
    )
    
    return schema


# ================================
# Validation Helpers
# ================================

def check_expected_columns(df: pd.DataFrame, expected_columns: List[str]) -> List[str]:
    """
    Verify all expected columns exist and no extra columns are present.
    
    Args:
        df: Input DataFrame.
        expected_columns: List of expected column names.
    
    Returns:
        List of warnings (empty if all checks pass).
    """
    warnings = []
    
    missing_cols = set(expected_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_columns)
    
    if missing_cols:
        warnings.append(f"Missing columns: {sorted(missing_cols)}")
    
    if extra_cols:
        warnings.append(f"Extra columns found: {sorted(extra_cols)}")
    
    return warnings


def check_data_types(df: pd.DataFrame, expected_dtypes: Dict[str, str]) -> List[str]:
    """
    Verify column data types match expectations.
    
    Args:
        df: Input DataFrame.
        expected_dtypes: Dict mapping column name to expected dtype string.
    
    Returns:
        List of warnings (empty if all checks pass).
    """
    warnings = []
    
    for col, expected_dtype in expected_dtypes.items():
        if col in df.columns:
            actual_dtype = df[col].dtype
            if expected_dtype == "float64" and actual_dtype != np.float64:
                warnings.append(
                    f"Column '{col}' has dtype {actual_dtype}, expected float64"
                )
            elif expected_dtype == "int64" and actual_dtype != np.int64:
                warnings.append(
                    f"Column '{col}' has dtype {actual_dtype}, expected int64"
                )
    
    return warnings


def check_null_values(df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    """
    Check for null values across all columns.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Tuple of (null_counts dict, warning messages).
    """
    null_counts = df.isnull().sum().to_dict()
    nulls_present = {col: count for col, count in null_counts.items() if count > 0}
    
    warnings = []
    if nulls_present:
        for col, count in nulls_present.items():
            warnings.append(f"Column '{col}' has {count} null values")
    
    return null_counts, warnings


def check_duplicates(df: pd.DataFrame) -> Tuple[int, List[str]]:
    """
    Check for duplicate rows.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Tuple of (duplicate_count, warning messages).
    """
    duplicate_count = int(df.duplicated().sum())
    
    warnings = []
    if duplicate_count > 0:
        warnings.append(
            f"Found {duplicate_count} duplicate rows "
            f"({duplicate_count / len(df) * 100:.4f}% of data)"
        )
    
    return duplicate_count, warnings


def check_class_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute class distribution for reporting.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Dict with class counts and percentages.
    """
    class_counts = df["Class"].value_counts().to_dict()
    total = len(df)
    
    distribution = {
        "total_transactions": int(total),
        "legitimate": int(class_counts.get(0, 0)),
        "fraud": int(class_counts.get(1, 0)),
        "fraud_ratio_pct": round(class_counts.get(1, 0) / total * 100, 4),
    }
    
    return distribution


def check_amount_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute amount statistics for anomaly detection.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Dict with amount statistics.
    """
    amount = df["Amount"]
    
    stats = {
        "min": float(amount.min()),
        "max": float(amount.max()),
        "mean": round(float(amount.mean()), 4),
        "median": float(amount.median()),
        "negative_count": int((amount < 0).sum()),
        "zero_count": int((amount == 0).sum()),
    }
    
    return stats


# ================================
# Validation Report
# ================================

def generate_report(
    schema_valid: bool,
    column_warnings: List[str],
    dtype_warnings: List[str],
    null_counts: Dict[str, int],
    null_warnings: List[str],
    duplicate_count: int,
    duplicate_warnings: List[str],
    class_dist: Dict[str, Any],
    amount_stats: Dict[str, float],
    config: Dict,
) -> Dict:
    """
    Generate a structured validation report.
    Duplicates are treated as warnings, not critical failures.
    
    Returns:
        Dictionary containing the complete validation report.
    """
    has_null = any(count > 0 for count in null_counts.values())
    has_duplicates = duplicate_count > 0
    has_critical_warnings = bool(column_warnings or dtype_warnings or null_warnings)
    
    # Overall pass: schema valid, no nulls, no critical warnings
    # Duplicates alone do not cause a FAIL
    overall_pass = schema_valid and not has_null and not has_critical_warnings
    
    expected_col_count = config["ingestion"]["expected_columns"]
    expected_col_names = config["ingestion"]["column_names"]
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "checks": {
            "schema_validation": {
                "status": "PASS" if schema_valid else "FAIL",
                "method": "pandera",
            },
            "column_check": {
                "status": "PASS" if not column_warnings else "FAIL",
                "expected_columns": expected_col_count,
                "actual_columns": len(expected_col_names),
                "warnings": column_warnings,
            },
            "data_types": {
                "status": "PASS" if not dtype_warnings else "FAIL",
                "warnings": dtype_warnings,
            },
            "null_values": {
                "status": "PASS" if not has_null else "FAIL",
                "total_nulls": int(sum(null_counts.values())),
                "nulls_by_column": {k: int(v) for k, v in null_counts.items()},
                "warnings": null_warnings,
            },
            "duplicates": {
                "status": "WARN" if has_duplicates else "PASS",
                "duplicate_count": duplicate_count,
                "duplicate_pct": round(duplicate_count / class_dist["total_transactions"] * 100, 4) if class_dist["total_transactions"] > 0 else 0,
                "warnings": duplicate_warnings,
            },
            "amount_validation": {
                "status": "PASS" if amount_stats["negative_count"] == 0 else "FAIL",
                "negative_count": amount_stats["negative_count"],
                "zero_count": amount_stats["zero_count"],
                "min": amount_stats["min"],
                "max": amount_stats["max"],
                "mean": amount_stats["mean"],
                "median": amount_stats["median"],
            },
            "class_distribution": class_dist,
        },
    }
    
    return report


def save_report(report: Dict, report_path: str) -> None:
    """
    Save validation report to JSON file.
    
    Args:
        report: Validation report dictionary.
        report_path: Path to save the report.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"[VALIDATE] Report saved to: {report_path}")


def print_report_summary(report: Dict) -> None:
    """
    Print a human-readable summary of the validation report to console.
    
    Args:
        report: Validation report dictionary.
    """
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"Timestamp:       {report['timestamp']}")
    print(f"Overall Status:  {report['overall_status']}")
    print("-" * 60)
    
    for check_name, check_data in report["checks"].items():
        if check_name == "class_distribution":
            continue
        status = check_data.get("status", "N/A")
        print(f"  {check_name}: {status}")
    
    print("-" * 60)
    
    # Class distribution
    cd = report["checks"]["class_distribution"]
    print(f"Transactions:    {cd['total_transactions']:,}")
    print(f"Legitimate:      {cd['legitimate']:,}")
    print(f"Fraud:           {cd['fraud']:,}")
    print(f"Fraud Ratio:     {cd['fraud_ratio_pct']}%")
    
    # Duplicates
    dup = report["checks"]["duplicates"]
    if dup["duplicate_count"] > 0:
        print(f"Duplicates:      {dup['duplicate_count']:,} ({dup['duplicate_pct']}%)")
    else:
        print("Duplicates:      0")
    
    # Nulls
    nulls = report["checks"]["null_values"]
    if nulls["total_nulls"] > 0:
        print(f"Null Values:     {nulls['total_nulls']}")
    else:
        print("Null Values:     0")
    
    print("=" * 60 + "\n")


# ================================
# Main Validation Pipeline
# ================================

def validate_data(
    df: pd.DataFrame,
    config_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run all validation checks on the input DataFrame.
    
    Args:
        df: Raw DataFrame from Data Ingestion.
        config_path: Optional path to pipeline config YAML.
    
    Returns:
        Tuple of (validated DataFrame, validation report dict).
    
    Raises:
        ValueError: If critical validation checks fail and config says to stop.
    """
    config = load_config(config_path)
    stop_on_failure = config["validation"]["stop_on_failure"]
    expected_columns = config["ingestion"]["column_names"]
    
    print("[VALIDATE] Starting data validation...")
    
    # Build expected dtypes dict
    expected_dtypes = {col: "float64" for col in expected_columns if col != "Class"}
    expected_dtypes["Class"] = "int64"
    
    # 1. Column presence check
    print("[VALIDATE] Checking column presence...")
    column_warnings = check_expected_columns(df, expected_columns)
    
    # 2. Data type check
    print("[VALIDATE] Checking data types...")
    dtype_warnings = check_data_types(df, expected_dtypes)
    
    # 3. Schema validation via pandera
    print("[VALIDATE] Running pandera schema validation...")
    try:
        schema = build_schema(config)
        schema.validate(df, lazy=True)
        schema_valid = True
    except pa.errors.SchemaErrors as e:
        schema_valid = False
        print(f"[VALIDATE] Schema errors: {e}")
    
    # 4. Null value check
    print("[VALIDATE] Checking null values...")
    null_counts, null_warnings = check_null_values(df)
    
    # 5. Duplicate check
    print("[VALIDATE] Checking duplicate rows...")
    duplicate_count, duplicate_warnings = check_duplicates(df)
    
    # 6. Class distribution
    print("[VALIDATE] Computing class distribution...")
    class_dist = check_class_distribution(df)
    
    # 7. Amount statistics
    print("[VALIDATE] Computing amount statistics...")
    amount_stats = check_amount_statistics(df)
    
    # Generate report
    report = generate_report(
        schema_valid=schema_valid,
        column_warnings=column_warnings,
        dtype_warnings=dtype_warnings,
        null_counts=null_counts,
        null_warnings=null_warnings,
        duplicate_count=duplicate_count,
        duplicate_warnings=duplicate_warnings,
        class_dist=class_dist,
        amount_stats=amount_stats,
        config=config,
    )
    
    # Save report
    report_path = config["paths"]["validation_report"]
    if not os.path.isabs(report_path):
        project_root = Path(__file__).resolve().parent.parent
        report_path = str(project_root / report_path)
    
    save_report(report, report_path)
    print_report_summary(report)
    
    # Halt on critical failure if configured
    if report["overall_status"] == "FAIL" and stop_on_failure:
        raise ValueError(
            "Data validation failed with critical errors. "
            f"See report for details: {report_path}"
        )
    
    if report["overall_status"] == "FAIL":
        print("[VALIDATE] WARNING: Validation failed but stop_on_failure is False. Continuing...")
    else:
        print("[VALIDATE] All validation checks passed.")
    
    return df, report


# ================================
# Entry Point
# ================================

def main():
    """
    Execute data validation as a standalone script.
    Loads raw data from ingestion step and runs validation.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest import ingest_data
    
    try:
        print("[VALIDATE] Loading raw data from ingestion...")
        df_raw = ingest_data()
        
        df_validated, report = validate_data(df_raw)
        
        print(f"\n[VALIDATE] Validation complete. {len(df_validated):,} rows passed.")
        
        return df_validated
        
    except FileNotFoundError as e:
        print(f"[VALIDATE] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        
    except ValueError as e:
        print(f"[VALIDATE] CRITICAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"[VALIDATE] UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()