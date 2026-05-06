"""
Monitoring & Maintenance Module — Offline Model Performance Drift Detection

Periodically loads new labeled transactions, evaluates the deployed model,
and compares against baseline metrics to detect performance degradation.

Architecture:
    BaselineMetricsLoader → loads reference metrics from evaluation report
    MonitoringDataLoader → loads new labeled transaction batch
    DriftDetector → computes current metrics, compares against baseline, flags warnings
    MonitoringReport → saves timestamped report with drift flags

Production upgrade path:
    - Replace CSV loader with data warehouse connection (ClickHouse/BigQuery)
    - Emit drift alerts to Slack/PagerDuty via webhook
    - Trigger automated retraining CI/CD pipeline on drift detection
    - Store reports in cloud object storage (S3/GCS) instead of local filesystem
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringConfig:
    """Immutable configuration for the monitoring module.

    Attributes:
        new_data_path: Path to CSV with new labeled transactions.
        model_path: Path to the deployed model artifact (joblib/pickle).
        baseline_report_path: Path to the baseline evaluation JSON report.
        output_dir: Directory to save timestamped monitoring reports.
        fpr_increase_threshold: Relative increase in FPR that triggers a warning.
            E.g., 0.20 means a 20% relative increase.
        pr_auc_drop_threshold: Relative drop in PR-AUC that triggers a warning.
            E.g., 0.10 means a 10% relative drop.
        target_column: Name of the ground truth column in the new data.
        probability_threshold: Decision threshold for binary classification.
            If None, uses the threshold from the baseline report.
    """

    new_data_path: Path
    model_path: Path
    baseline_report_path: Path
    output_dir: Path = field(default_factory=lambda: Path("reports"))
    fpr_increase_threshold: float = 0.20
    pr_auc_drop_threshold: float = 0.10
    target_column: str = "Class"
    probability_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.fpr_increase_threshold <= 1.0:
            raise ValueError(
                f"fpr_increase_threshold must be in [0, 1], "
                f"got {self.fpr_increase_threshold}"
            )
        if not 0.0 <= self.pr_auc_drop_threshold <= 1.0:
            raise ValueError(
                f"pr_auc_drop_threshold must be in [0, 1], "
                f"got {self.pr_auc_drop_threshold}"
            )
        if not self.new_data_path.exists():
            raise FileNotFoundError(
                f"New data file not found: {self.new_data_path.resolve()}"
            )
        if not self.baseline_report_path.exists():
            raise FileNotFoundError(
                f"Baseline report not found: {self.baseline_report_path.resolve()}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path.resolve()}"
            )


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineMetrics:
    """Reference metrics from the evaluation baseline.

    Attributes:
        pr_auc: Precision-Recall AUC from baseline evaluation.
        roc_auc: ROC-AUC from baseline evaluation.
        false_positive_rate: FPR at the deployed threshold.
        recall: Recall at the deployed threshold.
        precision: Precision at the deployed threshold.
        threshold: Decision threshold used.
        test_samples: Number of samples in baseline evaluation.
        fraud_ratio: Proportion of fraud in baseline test set.
    """

    pr_auc: float
    roc_auc: float
    false_positive_rate: float
    recall: float
    precision: float
    threshold: float
    test_samples: int
    fraud_ratio: float


@dataclass(frozen=True)
class CurrentMetrics:
    """Metrics computed on the new monitoring batch.

    Attributes:
        pr_auc: Current Precision-Recall AUC.
        roc_auc: Current ROC-AUC.
        false_positive_rate: Current FPR.
        recall: Current recall.
        precision: Current precision.
        n_samples: Number of samples in the monitoring batch.
        fraud_count: Number of fraud cases in the monitoring batch.
        tp: True positives.
        fp: False positives.
        tn: True negatives.
        fn: False negatives.
    """

    pr_auc: float
    roc_auc: float
    false_positive_rate: float
    recall: float
    precision: float
    n_samples: int
    fraud_count: int
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class DriftFlags:
    """Drift detection results comparing current vs baseline metrics.

    Attributes:
        fpr_warning: True if FPR increased beyond the threshold.
        pr_auc_warning: True if PR-AUC dropped beyond the threshold.
        fpr_relative_change: Relative change in FPR (current/baseline - 1).
        pr_auc_relative_change: Relative change in PR-AUC (1 - current/baseline).
        any_warning: True if any drift flag is raised.
    """

    fpr_warning: bool
    pr_auc_warning: bool
    fpr_relative_change: float
    pr_auc_relative_change: float

    @property
    def any_warning(self) -> bool:
        """True if any drift warning flag is raised."""
        return self.fpr_warning or self.pr_auc_warning


# ---------------------------------------------------------------------------
# Baseline Metrics Loader
# ---------------------------------------------------------------------------


def load_baseline_metrics(report_path: Path) -> BaselineMetrics:
    """Load reference evaluation metrics from a JSON report.

    Expected JSON structure matches the output of src/evaluation/evaluate.py:
        {
            "pr_auc": 0.8121,
            "roc_auc": 0.9855,
            "false_positive_rate": 0.0002,
            "recall": 0.7838,
            "precision": 0.8529,
            "confusion_matrix": { ... },
            "test_samples": 56746,
            "fraud_rate": 0.0013
        }

    Also supports the metrics JSON format from train_baseline.py:
        {
            "cv_metrics": {
                "average_precision": {"mean": 0.81},
                "roc_auc": {"mean": 0.98},
                ...
            }
        }

    Args:
        report_path: Path to the evaluation JSON file.

    Returns:
        Populated BaselineMetrics dataclass.

    Raises:
        FileNotFoundError: If the report does not exist.
        KeyError: If required fields are missing from the JSON.
        json.JSONDecodeError: If the JSON is malformed.
    """
    logger.info("Loading baseline metrics from: %s", report_path.resolve())

    with open(report_path, "r") as f:
        data = json.load(f)

    # Handle both evaluation report and training metrics formats
    if "cv_metrics" in data:
        # Training metrics format — extract mean CV scores
        cv = data["cv_metrics"]
        pr_auc = cv.get("average_precision", {}).get("mean", 0.0)
        roc_auc = cv.get("roc_auc", {}).get("mean", 0.0)
        recall = cv.get("recall", {}).get("mean", 0.0)
        precision = cv.get("precision", {}).get("mean", 0.0)
        fpr = 1.0 - cv.get("specificity", cv.get("tnr", {})).get("mean", 1.0)
        threshold = data.get("threshold", 0.5)
        test_samples = data.get("train_samples", 0)
        fraud_ratio = data.get("fraud_rate", 0.0)
    else:
        # Evaluation report format — extract test set metrics
        pr_auc = data.get("pr_auc", data.get("average_precision", 0.0))
        roc_auc = data.get("roc_auc", 0.0)
        recall = data.get("recall", 0.0)
        precision = data.get("precision", 0.0)
        fpr = data.get("false_positive_rate", data.get("fpr", 0.0))
        threshold = data.get("threshold", 0.5)
        test_samples = data.get("test_samples", 0)
        fraud_ratio = data.get(
            "fraud_rate",
            data.get("fraud_caught", 0) / max(test_samples, 1),
        )

    baseline = BaselineMetrics(
        pr_auc=float(pr_auc),
        roc_auc=float(roc_auc),
        false_positive_rate=float(fpr),
        recall=float(recall),
        precision=float(precision),
        threshold=float(threshold),
        test_samples=int(test_samples),
        fraud_ratio=float(fraud_ratio),
    )

    logger.info(
        "Baseline loaded: PR-AUC=%.4f, FPR=%.4f%%, Recall=%.4f, Samples=%d",
        baseline.pr_auc,
        baseline.false_positive_rate * 100,
        baseline.recall,
        baseline.test_samples,
    )

    return baseline


# ---------------------------------------------------------------------------
# Monitoring Data Loader
# ---------------------------------------------------------------------------


def load_monitoring_data(
    data_path: Path,
    target_column: str = "Class",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load new labeled transaction data for monitoring.

    Expects a CSV file with features and ground truth labels.
    The feature columns must match the training feature order.

    Args:
        data_path: Path to the CSV file with labeled transactions.
        target_column: Name of the ground truth column.

    Returns:
        Tuple of (X, y, full_df):
            X: Feature matrix as float64 numpy array.
            y: Label vector as int64 numpy array.
            full_df: Complete DataFrame for inspection.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the target column is missing or empty.
    """
    logger.info("Loading monitoring data from: %s", data_path.resolve())

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError(f"Monitoring data file is empty: {data_path}")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    feature_cols = [col for col in df.columns if col != target_column]

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_column].values.astype(np.int64)

    fraud_count = int(y.sum())
    logger.info(
        "Loaded %d samples, %d features, %d fraud cases (%.2f%%)",
        len(df),
        len(feature_cols),
        fraud_count,
        (fraud_count / len(df)) * 100 if len(df) > 0 else 0,
    )

    if fraud_count == 0:
        logger.warning(
            "No fraud cases in monitoring batch — "
            "PR-AUC and recall metrics will be undefined."
        )

    return X, y, df


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_model(model_path: Path) -> Any:
    """Load the deployed model artifact.

    Supports joblib and pickle formats. Tries joblib first (used by
    train_baseline.py), falls back to pickle.

    Args:
        model_path: Path to the serialized model file.

    Returns:
        Deserialized model object with predict_proba() interface.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If deserialization fails for both formats.
    """
    logger.info("Loading model from: %s", model_path.resolve())

    errors: List[str] = []

    # Try joblib first (training pipeline format)
    try:
        import joblib
        model = joblib.load(model_path)
        logger.info("Model loaded successfully via joblib")
        return model
    except Exception as exc:
        errors.append(f"joblib: {exc}")

    # Fallback to pickle
    try:
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully via pickle")
        return model
    except Exception as exc:
        errors.append(f"pickle: {exc}")

    raise ValueError(
        f"Failed to load model from {model_path}. Errors: {'; '.join(errors)}"
    )


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------


def compute_current_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> CurrentMetrics:
    """Compute evaluation metrics on the monitoring batch.

    Args:
        y_true: Ground truth labels (0 = legitimate, 1 = fraud).
        y_proba: Predicted fraud probabilities from the model.
        threshold: Decision threshold for binary classification.

    Returns:
        CurrentMetrics dataclass with all computed values.

    Raises:
        ValueError: If y_true and y_proba have mismatched lengths,
            or if no fraud cases exist (cannot compute PR-AUC).
    """
    if len(y_true) != len(y_proba):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_proba={len(y_proba)}"
        )

    n_samples = len(y_true)
    fraud_count = int(y_true.sum())

    # Binary predictions at threshold
    y_pred = (y_proba >= threshold).astype(np.int64)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrics
    pr_auc = 0.0
    roc_auc = 0.0

    if fraud_count > 0 and len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, y_proba))
        roc_auc = float(roc_auc_score(y_true, y_proba))
    else:
        logger.warning(
            "Cannot compute PR-AUC/ROC-AUC: batch has %d fraud cases "
            "(%d unique classes)",
            fraud_count,
            len(np.unique(y_true)),
        )

    recall = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(
        tp / (tp + fp) if (tp + fp) > 0 else 0.0
    )
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return CurrentMetrics(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        false_positive_rate=fpr,
        recall=recall,
        precision=precision,
        n_samples=n_samples,
        fraud_count=fraud_count,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


# ---------------------------------------------------------------------------
# Drift Detection
# ---------------------------------------------------------------------------


def detect_drift(
    baseline: BaselineMetrics,
    current: CurrentMetrics,
    fpr_threshold: float = 0.20,
    pr_auc_threshold: float = 0.10,
) -> DriftFlags:
    """Compare current metrics against baseline and flag degradation.

    Drift is detected when:
        - FPR increases by more than fpr_threshold (relative).
          Formula: (current_fpr / baseline_fpr) - 1 > fpr_threshold
        - PR-AUC decreases by more than pr_auc_threshold (relative).
          Formula: 1 - (current_pr_auc / baseline_pr_auc) > pr_auc_threshold

    Edge cases:
        - If baseline FPR is 0, any non-zero current FPR triggers a warning.
        - If baseline PR-AUC is 0, skip PR-AUC comparison.

    Args:
        baseline: Reference metrics from evaluation.
        current: Metrics computed on the monitoring batch.
        fpr_threshold: Relative FPR increase threshold (e.g., 0.20 = 20%).
        pr_auc_threshold: Relative PR-AUC drop threshold (e.g., 0.10 = 10%).

    Returns:
        DriftFlags indicating which warnings are raised.
    """
    # FPR drift: relative increase
    if baseline.false_positive_rate > 0:
        fpr_change = (
            current.false_positive_rate / baseline.false_positive_rate
        ) - 1.0
    else:
        # Baseline FPR is 0 — any positive FPR is a warning
        fpr_change = 1.0 if current.false_positive_rate > 0 else 0.0

    fpr_warning = fpr_change > fpr_threshold

    # PR-AUC drift: relative decrease
    if baseline.pr_auc > 0 and current.pr_auc > 0:
        pr_auc_change = 1.0 - (current.pr_auc / baseline.pr_auc)
    else:
        pr_auc_change = 0.0

    pr_auc_warning = pr_auc_change > pr_auc_threshold

    flags = DriftFlags(
        fpr_warning=fpr_warning,
        pr_auc_warning=pr_auc_warning,
        fpr_relative_change=round(float(fpr_change), 6),
        pr_auc_relative_change=round(float(pr_auc_change), 6),
    )

    if flags.any_warning:
        logger.warning(
            "DRIFT DETECTED: FPR_warning=%s (change=%.2f%%), "
            "PR-AUC_warning=%s (change=%.2f%%)",
            flags.fpr_warning,
            flags.fpr_relative_change * 100,
            flags.pr_auc_warning,
            flags.pr_auc_relative_change * 100,
        )
    else:
        logger.info(
            "No drift detected: FPR_change=%.2f%%, PR-AUC_change=%.2f%%",
            flags.fpr_relative_change * 100,
            flags.pr_auc_relative_change * 100,
        )

    return flags


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def generate_monitoring_report(
    baseline: BaselineMetrics,
    current: CurrentMetrics,
    flags: DriftFlags,
    model_name: str,
    output_dir: Path,
    monitoring_batch_name: str = "",
) -> Path:
    """Save a timestamped monitoring report as JSON.

    The report includes baseline metrics, current metrics, drift flags,
    and metadata for audit trail purposes.

    Args:
        baseline: Reference evaluation metrics.
        current: Metrics from the monitoring batch.
        flags: Drift detection results.
        model_name: Name of the deployed model.
        output_dir: Directory to save the report.
        monitoring_batch_name: Optional identifier for the batch.

    Returns:
        Path to the saved report file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"monitoring_report_{timestamp}.json"
    report_path = output_dir / filename

    report = {
        "report_timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "monitoring_batch": monitoring_batch_name or f"batch_{timestamp}",
        "baseline": {
            "pr_auc": baseline.pr_auc,
            "roc_auc": baseline.roc_auc,
            "false_positive_rate": baseline.false_positive_rate,
            "recall": baseline.recall,
            "precision": baseline.precision,
            "threshold": baseline.threshold,
            "test_samples": baseline.test_samples,
            "fraud_ratio": baseline.fraud_ratio,
        },
        "current": {
            "pr_auc": current.pr_auc,
            "roc_auc": current.roc_auc,
            "false_positive_rate": current.false_positive_rate,
            "recall": current.recall,
            "precision": current.precision,
            "n_samples": current.n_samples,
            "fraud_count": current.fraud_count,
            "confusion_matrix": {
                "true_negatives": current.tn,
                "false_positives": current.fp,
                "false_negatives": current.fn,
                "true_positives": current.tp,
            },
        },
        "drift_flags": {
            "fpr_warning": flags.fpr_warning,
            "pr_auc_warning": flags.pr_auc_warning,
            "any_warning": flags.any_warning,
            "fpr_relative_change": flags.fpr_relative_change,
            "pr_auc_relative_change": flags.pr_auc_relative_change,
            "fpr_increase_threshold": 0.20,  # From config
            "pr_auc_drop_threshold": 0.10,   # From config
        },
        "action_required": flags.any_warning,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Monitoring report saved to: %s", report_path.resolve())

    return report_path


# ---------------------------------------------------------------------------
# Main Monitoring Pipeline
# ---------------------------------------------------------------------------


def run_monitoring(
    config: MonitoringConfig,
) -> Dict[str, Any]:
    """Execute the complete monitoring pipeline.

    Loads baseline metrics, new labeled data, and the deployed model,
    computes current performance metrics, detects drift against baseline,
    and generates a timestamped report.

    Args:
        config: Immutable monitoring configuration.

    Returns:
        Dictionary containing:
            - report_path: Path to the saved JSON report.
            - baseline: BaselineMetrics object.
            - current: CurrentMetrics object.
            - flags: DriftFlags object.
            - action_required: Boolean indicating if intervention is needed.

    Raises:
        FileNotFoundError: If any required input file is missing.
        ValueError: If data validation fails or metrics cannot be computed.
    """
    logger.info("=" * 60)
    logger.info("MONITORING PIPELINE START")
    logger.info("=" * 60)

    # Step 1: Load baseline metrics
    baseline = load_baseline_metrics(config.baseline_report_path)

    # Step 2: Load new monitoring data
    X_new, y_new, df_new = load_monitoring_data(
        config.new_data_path,
        target_column=config.target_column,
    )

    # Step 3: Load deployed model
    model = load_model(config.model_path)

    # Step 4: Run inference
    logger.info("Running inference on %d samples...", len(X_new))
    y_proba = model.predict_proba(X_new)[:, 1]

    # Step 5: Determine threshold
    threshold = (
        config.probability_threshold
        if config.probability_threshold is not None
        else baseline.threshold
    )
    logger.info("Using decision threshold: %.4f", threshold)

    # Step 6: Compute current metrics
    current = compute_current_metrics(y_new, y_proba, threshold)

    logger.info(
        "Current metrics: PR-AUC=%.4f, FPR=%.4f%%, Recall=%.4f, "
        "Precision=%.4f, TP=%d, FP=%d, FN=%d, TN=%d",
        current.pr_auc,
        current.false_positive_rate * 100,
        current.recall,
        current.precision,
        current.tp,
        current.fp,
        current.fn,
        current.tn,
    )

    # Step 7: Detect drift
    flags = detect_drift(
        baseline=baseline,
        current=current,
        fpr_threshold=config.fpr_increase_threshold,
        pr_auc_threshold=config.pr_auc_drop_threshold,
    )

    # Step 8: Generate report
    report_path = generate_monitoring_report(
        baseline=baseline,
        current=current,
        flags=flags,
        model_name=config.model_path.stem,
        output_dir=config.output_dir,
    )

    logger.info("=" * 60)
    logger.info(
        "MONITORING COMPLETE — Action Required: %s",
        flags.any_warning,
    )
    logger.info("=" * 60)

    return {
        "report_path": report_path,
        "baseline": baseline,
        "current": current,
        "flags": flags,
        "action_required": flags.any_warning,
    }