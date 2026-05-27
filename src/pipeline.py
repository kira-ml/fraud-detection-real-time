"""
Master Pipeline Orchestrator
=============================
Runs the full fraud detection pipeline in correct order to prevent temporal leakage:

  ingest → validate → preprocess → SPLIT → feature_engineering(train) + feature_engineering(test)
                                         → train → evaluate

The SPLIT happens BEFORE feature engineering. Train and test sets are engineered
independently, ensuring velocity windows never peek into future data.

Usage:
  python src/pipeline.py --mode full          # Run everything
  python src/pipeline.py --mode baseline      # Baseline features + models only
  python src/pipeline.py --mode advanced      # Advanced features + models only
  python src/pipeline.py --step train         # Run from training onward (needs prior outputs)
  python src/pipeline.py --dry-run            # Run on 5,000 sample rows for quick testing
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import ingest_data
from src.validate import validate_data
from src.preprocess import preprocess_data, save_cleaned_data, save_scaler
from src.split import split_data
from src.feature_engineering import (
    run_feature_engineering_baseline,
    run_feature_engineering_advanced,
    save_feature_config,
)
from src.models.train_baseline import run_baseline_training
from src.models.train_advanced import run_advanced_training
from src.evaluation.evaluate import run_evaluation


# ================================
# Path Resolution
# ================================

def resolve_path(relative_path: str) -> str:
    """Convert a config-relative path to an absolute path."""
    if Path(relative_path).is_absolute():
        return relative_path
    return str(PROJECT_ROOT / relative_path)


# ================================
# Pipeline Steps
# ================================

def step_ingest(config_path: str, dry_run: bool = False) -> pd.DataFrame:
    """Step 1: Ingest raw data."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA INGESTION")
    print("=" * 70)
    df = ingest_data(config_path)
    if dry_run:
        df = df.sample(n=min(5000, len(df)), random_state=42).sort_values("Time").reset_index(drop=True)
        print(f"[PIPELINE] DRY RUN: Sampled {len(df):,} rows for quick testing")
    return df


def step_validate(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    """Step 2: Validate data."""
    print("\n" + "=" * 70)
    print("STEP 2: DATA VALIDATION")
    print("=" * 70)
    df_valid, report = validate_data(df, config_path)
    return df_valid


def step_preprocess(df: pd.DataFrame, config_path: str) -> tuple:
    """Step 3: Preprocess data. Returns (df_clean, scaler)."""
    print("\n" + "=" * 70)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 70)
    df_clean, scaler = preprocess_data(df, config_path)
    return df_clean, scaler


def step_split(df: pd.DataFrame, config: Dict) -> Dict:
    """Step 4: Split data BEFORE feature engineering."""
    print("\n" + "=" * 70)
    print("STEP 4: TIME-AWARE DATA SPLITTING (BEFORE FEATURE ENGINEERING)")
    print("=" * 70)
    
    train_ratio = config["splitting"]["train_ratio"]
    
    # Resolve save paths
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    
    save_paths = {
        "train": str(Path(processed_dir) / "train_raw.parquet"),
        "test": str(Path(processed_dir) / "test_raw.parquet"),
    }
    
    # Use Time_raw (raw seconds) for correct chronological split
    # Falls back to Time if Time_raw doesn't exist
    time_col = "Time_raw" if "Time_raw" in df.columns else "Time"
    
    result = split_data(
        df=df,
        train_ratio=train_ratio,
        target_col="Class",
        time_col=time_col,
        save_paths=save_paths,
    )
    
    # Also save the processed_dir for later steps
    result["processed_dir"] = processed_dir
    
    return result


def step_feature_engineering_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict,
) -> Dict:
    """Step 5a: Engineer baseline features on train and test INDEPENDENTLY."""
    print("\n" + "=" * 70)
    print("STEP 5a: BASELINE FEATURE ENGINEERING (Train & Test Independently)")
    print("=" * 70)
    
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    artifacts_dir = resolve_path(config["paths"]["artifacts_dir"])
    
    # Engineer train
    print("\n[PIPELINE] Engineering BASELINE features for TRAIN set...")
    df_train_fe, train_features = run_feature_engineering_baseline(train_df.copy(), config)
    
    # Engineer test
    print("\n[PIPELINE] Engineering BASELINE features for TEST set...")
    df_test_fe, test_features = run_feature_engineering_baseline(test_df.copy(), config)
    
    # Verify feature columns match
    assert train_features == test_features, \
        f"Feature mismatch! Train: {len(train_features)}, Test: {len(test_features)}"
    
    # Save
    train_path = str(Path(processed_dir) / "train_baseline.parquet")
    test_path = str(Path(processed_dir) / "test_baseline.parquet")
    
    df_train_fe.to_parquet(train_path, index=False)
    df_test_fe.to_parquet(test_path, index=False)
    
    # Save feature config
    feature_config_path = str(Path(artifacts_dir) / "feature_config_baseline.json")
    save_feature_config(train_features, feature_config_path)
    
    print(f"[PIPELINE] Baseline train features saved: {train_path} ({len(df_train_fe):,} rows, {len(train_features)} features)")
    print(f"[PIPELINE] Baseline test features saved:  {test_path} ({len(df_test_fe):,} rows, {len(test_features)} features)")
    
    return {
        "train_path": train_path,
        "test_path": test_path,
        "features": train_features,
        "feature_count": len(train_features),
    }


def step_feature_engineering_advanced(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict,
) -> Dict:
    """Step 5b: Engineer advanced features on train and test INDEPENDENTLY."""
    print("\n" + "=" * 70)
    print("STEP 5b: ADVANCED FEATURE ENGINEERING (Train & Test Independently)")
    print("=" * 70)
    
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    artifacts_dir = resolve_path(config["paths"]["artifacts_dir"])
    
    # Engineer train
    print("\n[PIPELINE] Engineering ADVANCED features for TRAIN set...")
    df_train_fe, train_features = run_feature_engineering_advanced(train_df.copy(), config)
    
    # Engineer test
    print("\n[PIPELINE] Engineering ADVANCED features for TEST set...")
    df_test_fe, test_features = run_feature_engineering_advanced(test_df.copy(), config)
    
    # Verify feature columns match
    assert train_features == test_features, \
        f"Feature mismatch! Train: {len(train_features)}, Test: {len(test_features)}"
    
    # Save
    train_path = str(Path(processed_dir) / "train_advanced.parquet")
    test_path = str(Path(processed_dir) / "test_advanced.parquet")
    
    df_train_fe.to_parquet(train_path, index=False)
    df_test_fe.to_parquet(test_path, index=False)
    
    # Save feature config
    feature_config_path = str(Path(artifacts_dir) / "feature_config_advanced.json")
    save_feature_config(train_features, feature_config_path)
    
    print(f"[PIPELINE] Advanced train features saved: {train_path} ({len(df_train_fe):,} rows, {len(train_features)} features)")
    print(f"[PIPELINE] Advanced test features saved:  {test_path} ({len(df_test_fe):,} rows, {len(test_features)} features)")
    
    return {
        "train_path": train_path,
        "test_path": test_path,
        "features": train_features,
        "feature_count": len(train_features),
    }


def step_train_baseline(config: Dict, feature_result: Dict) -> Dict:
    """Step 6a: Train baseline models."""
    print("\n" + "=" * 70)
    print("STEP 6a: BASELINE MODEL TRAINING")
    print("=" * 70)
    
    models_dir = resolve_path("models")
    metrics_dir = resolve_path("artifacts/metrics")
    
    results = run_baseline_training(
        train_path=feature_result["train_path"],
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        cv_folds=config["training"]["cross_validation"]["n_folds"],
        random_state=config["random_state"],
    )
    return results


def step_train_advanced(config: Dict, feature_result: Dict) -> Dict:
    """Step 6b: Train advanced models."""
    print("\n" + "=" * 70)
    print("STEP 6b: ADVANCED MODEL TRAINING")
    print("=" * 70)
    
    models_dir = resolve_path("models")
    metrics_dir = resolve_path("artifacts/metrics")
    
    results = run_advanced_training(
        train_path=feature_result["train_path"],
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        cv_folds=config["training"]["cross_validation"]["n_folds"],
        random_state=config["random_state"],
    )
    return results


def step_evaluate(
    feature_result: Dict,
    model_type: str,
) -> Dict:
    """Step 7: Evaluate models on test set."""
    print("\n" + "=" * 70)
    print(f"STEP 7: MODEL EVALUATION ({model_type.upper()})")
    print("=" * 70)
    
    models_dir = resolve_path("models")
    output_dir = resolve_path("artifacts/evaluation")
    
    if model_type == "baseline":
        results = run_evaluation(
            models_dir=models_dir,
            test_baseline_path=feature_result["test_path"],
            output_dir=output_dir,
            model_type="baseline",
        )
    else:
        results = run_evaluation(
            models_dir=models_dir,
            test_advanced_path=feature_result["test_path"],
            output_dir=output_dir,
            model_type="advanced",
        )
    
    return results


# ================================
# Full Pipeline
# ================================

def run_full_pipeline(
    config_path: str = "config/pipeline_config.yaml",
    mode: str = "full",
    dry_run: bool = False,
    skip_ingest: bool = False,
) -> Dict:
    """
    Execute the complete fraud detection pipeline.

    Args:
        config_path: Path to pipeline config YAML.
        mode: 'baseline', 'advanced', or 'full'.
        dry_run: If True, sample 5,000 rows for quick testing.
        skip_ingest: If True, start from validation (data already loaded).

    Returns:
        Dictionary with all pipeline results.
    """
    import yaml
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    start_time = time.time()
    pipeline_results = {}
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  FRAUD DETECTION PIPELINE — FULL ORCHESTRATION".center(68) + "█")
    print("█" + f"  Mode: {mode.upper()}, Dry Run: {dry_run}".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # ============================================================
    # Steps 1-4: Data Preparation (shared across modes)
    # ============================================================
    
    if not skip_ingest:
        df_raw = step_ingest(config_path, dry_run)
        pipeline_results["ingestion"] = {"rows": len(df_raw)}
    else:
        # Load already-ingested data
        cleaned_path = resolve_path(config["paths"]["cleaned_data"])
        print(f"[PIPELINE] Skipping ingest, loading from: {cleaned_path}")
        df_raw = pd.read_parquet(cleaned_path)
    
    df_valid = step_validate(df_raw, config_path)
    pipeline_results["validation"] = {"rows": len(df_valid)}
    
    df_clean, scaler = step_preprocess(df_valid, config_path)
    
    # Save cleaned data and scaler
    cleaned_path = resolve_path(config["paths"]["cleaned_data"])
    scaler_path = resolve_path(config["paths"]["scaler"])
    save_cleaned_data(df_clean, cleaned_path)
    save_scaler(scaler, scaler_path)
    pipeline_results["preprocessing"] = {"rows": len(df_clean)}
    
    # STEP 4: SPLIT BEFORE FEATURE ENGINEERING (CRITICAL)
    split_result = step_split(df_clean, config)
    pipeline_results["split"] = {
        "train_rows": len(split_result["train_df"]),
        "test_rows": len(split_result["test_df"]),
        "split_timestamp": split_result["split_timestamp"],
    }
    
    train_df = split_result["train_df"]
    test_df = split_result["test_df"]
    
    print(f"\n[PIPELINE] ✅ Temporal leakage prevention: "
          f"Split ({len(train_df):,} train / {len(test_df):,} test) "
          f"BEFORE feature engineering")
    
    # ============================================================
    # Steps 5-7: Model-Specific (baseline, advanced, or both)
    # ============================================================
    
    if mode in ("baseline", "full"):
        # Feature engineering
        baseline_fe_result = step_feature_engineering_baseline(train_df, test_df, config)
        pipeline_results["baseline_features"] = {
            "feature_count": baseline_fe_result["feature_count"],
            "train_path": baseline_fe_result["train_path"],
            "test_path": baseline_fe_result["test_path"],
        }
        
        # Training
        baseline_train_result = step_train_baseline(config, baseline_fe_result)
        pipeline_results["baseline_training"] = {
            "models_trained": list(baseline_train_result.keys()),
        }
        
        # Evaluation
        baseline_eval_result = step_evaluate(baseline_fe_result, "baseline")
        pipeline_results["baseline_evaluation"] = {
            "models_evaluated": list(baseline_eval_result.keys()) if baseline_eval_result else [],
        }
    
    if mode in ("advanced", "full"):
        # Feature engineering
        advanced_fe_result = step_feature_engineering_advanced(train_df, test_df, config)
        pipeline_results["advanced_features"] = {
            "feature_count": advanced_fe_result["feature_count"],
            "train_path": advanced_fe_result["train_path"],
            "test_path": advanced_fe_result["test_path"],
        }
        
        # Training
        advanced_train_result = step_train_advanced(config, advanced_fe_result)
        pipeline_results["advanced_training"] = {
            "models_trained": list(advanced_train_result.keys()),
        }
        
        # Evaluation
        advanced_eval_result = step_evaluate(advanced_fe_result, "advanced")
        pipeline_results["advanced_evaluation"] = {
            "models_evaluated": list(advanced_eval_result.keys()) if advanced_eval_result else [],
        }
    
    # ============================================================
    # Summary
    # ============================================================
    
    total_time = time.time() - start_time
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  PIPELINE COMPLETE".center(68) + "█")
    print("█" + f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Print artifact locations
    print(f"\n[PIPELINE] 📁 Key Outputs:")
    print(f"  Cleaned data:     {cleaned_path}")
    print(f"  Scaler:           {scaler_path}")
    if mode in ("baseline", "full"):
        print(f"  Baseline train:   {pipeline_results.get('baseline_features', {}).get('train_path', 'N/A')}")
        print(f"  Baseline test:    {pipeline_results.get('baseline_features', {}).get('test_path', 'N/A')}")
    if mode in ("advanced", "full"):
        print(f"  Advanced train:   {pipeline_results.get('advanced_features', {}).get('train_path', 'N/A')}")
        print(f"  Advanced test:    {pipeline_results.get('advanced_features', {}).get('test_path', 'N/A')}")
    print(f"  Models:           {resolve_path('models/')}")
    print(f"  Evaluation:       {resolve_path('artifacts/evaluation/')}")
    
    return pipeline_results


# ================================
# Entry Point
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Fraud Detection Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py --mode full           # Run complete pipeline
  python src/pipeline.py --mode baseline       # Baseline only
  python src/pipeline.py --mode advanced       # Advanced only
  python src/pipeline.py --dry-run             # Quick test on 5,000 rows
  python src/pipeline.py --skip-ingest         # Resume from validation
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["baseline", "advanced", "full"],
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--config", type=str, default="config/pipeline_config.yaml",
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run on 5,000 sample rows for quick testing",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip ingestion (data already loaded)",
    )

    args = parser.parse_args()

    try:
        results = run_full_pipeline(
            config_path=args.config,
            mode=args.mode,
            dry_run=args.dry_run,
            skip_ingest=args.skip_ingest,
        )
        print(f"\n[PIPELINE] ✅ Pipeline finished successfully.")
        return results

    except FileNotFoundError as e:
        print(f"\n[PIPELINE] ❌ FILE NOT FOUND: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n[PIPELINE] ❌ VALIDATION ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[PIPELINE] ❌ UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()