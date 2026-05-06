"""
Model Deployment Module — Production-Grade Flask REST API

Serves real-time fraud predictions with automatic best-model selection.
Loads the highest-performing model (by PR-AUC) from the models directory,
applies preprocessing identical to the training pipeline, and exposes
a /predict POST endpoint with structured request logging.

Architecture:
    ModelRegistry → selects best model from evaluation reports
    TransactionPreprocessor → stateless feature engineering
    FraudModelServer → Flask API orchestration + logging

Production scaling path:
    - Replace Flask with FastAPI (async, OpenAPI, dependency injection)
    - Replace filesystem registry with MLflow Model Registry
    - Add online feature store for per-card velocity features
    - Add Prometheus metrics endpoint for SRE monitoring
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureAssemblyConfig:
    """Defines the exact feature vector order for inference.

    MUST match the training feature order exactly. Any discrepancy
    between this config and the training pipeline causes silent
    prediction corruption (model receives features in wrong positions).

    Attributes:
        pca_features_ordered: Top-k PCA features in training order.
        velocity_windows_seconds: Rolling window sizes used for velocity features.
            Each window produces 3 features: count, avg_amount, std_amount.
        include_amount_scaled: Whether to include StandardScaler-transformed Amount_log.
        include_amount_log: Whether to include raw log-transformed Amount.
        include_time_cyclical: Whether to include hour_sin and hour_cos.
    """

    pca_features_ordered: Tuple[str, ...] = (
        "V17", "V14", "V12", "V10", "V16",
        "V3", "V7", "V11", "V4", "V18",
    )
    velocity_windows_seconds: Tuple[int, ...] = (3600, 86400)
    include_amount_scaled: bool = True
    include_amount_log: bool = True
    include_time_cyclical: bool = True

    @property
    def velocity_feature_count(self) -> int:
        """Number of velocity features: windows × (count, avg, std)."""
        return len(self.velocity_windows_seconds) * 3

    @property
    def total_feature_count(self) -> int:
        """Total number of features in the assembled vector."""
        count = len(self.pca_features_ordered) + self.velocity_feature_count
        if self.include_amount_scaled:
            count += 1
        if self.include_amount_log:
            count += 1
        if self.include_time_cyclical:
            count += 2
        return count


@dataclass(frozen=True)
class ModelCandidate:
    """A candidate model with its evaluation metrics.

    Attributes:
        model_path: Absolute path to the serialized model file.
        model_name: Human-readable model identifier.
        pr_auc: Precision-Recall AUC from evaluation (primary metric).
        roc_auc: ROC-AUC from evaluation (secondary).
        recall: Recall at the evaluated threshold.
        precision: Precision at the evaluated threshold.
        fpr: False positive rate at the evaluated threshold.
        inference_ms: Per-sample inference time in milliseconds.
    """

    model_path: Path
    model_name: str
    pr_auc: float
    roc_auc: float
    recall: float
    precision: float
    fpr: float
    inference_ms: float


@dataclass(frozen=True)
class ServingConfig:
    """Immutable configuration for the fraud detection serving module.

    Attributes:
        models_dir: Directory containing model .pkl files and evaluation JSONs.
        scaler_path: Path to fitted StandardScaler for Amount (shared across models).
        threshold: Decision threshold (probability >= threshold → fraud).
        primary_metric: Evaluation metric to rank models by (default: pr_auc).
        max_input_amount: Upper bound for Amount validation (reject outliers).
        log_level: Python logging level for the module.
        enable_request_logging: Whether to write structured request logs.
    """

    models_dir: Path
    scaler_path: Path
    threshold: float = 0.5
    primary_metric: str = "pr_auc"
    max_input_amount: float = 30000.0
    log_level: str = "INFO"
    enable_request_logging: bool = True
    feature_config: FeatureAssemblyConfig = field(
        default_factory=FeatureAssemblyConfig
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.max_input_amount <= 0:
            raise ValueError(
                f"max_input_amount must be positive, got {self.max_input_amount}"
            )
        if not self.models_dir.exists():
            raise FileNotFoundError(
                f"Models directory not found: {self.models_dir.resolve()}"
            )
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found: {self.scaler_path.resolve()}"
            )


# ---------------------------------------------------------------------------
# Model Registry — Automatic Best-Model Selection
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Scans the models directory and selects the best model for serving.

    Reads evaluation JSON files to rank candidates by the primary metric.
    Supports sklearn, LightGBM, XGBoost, and PyTorch models via format detection.

    Selection logic:
        1. Scan models_dir for .pkl and .pth files.
        2. For each model, find its paired evaluation JSON.
        3. Parse evaluation metrics from JSON.
        4. Rank by primary_metric (descending).
        5. Return the top candidate.
        6. If the top candidate fails to load, try the next best.

    Production upgrade path:
        Replace filesystem scanning with MLflow Model Registry API:
            mlflow_client.get_latest_versions(name="fraud_model", stages=["Production"])
    """

    # File extensions for supported model formats
    MODEL_EXTENSIONS: Tuple[str, ...] = (".pkl", ".pth")

    # Known evaluation JSON filename patterns
    EVAL_SUFFIXES: Tuple[str, ...] = (
        "_evaluation.json",
        "_metrics.json",
        "_eval.json",
    )

    def __init__(
        self,
        models_dir: Path,
        primary_metric: str = "pr_auc",
    ) -> None:
        """Initialize the registry and select the best model.

        Args:
            models_dir: Directory containing model and evaluation files.
            primary_metric: Metric to rank models by (must exist in eval JSON).

        Raises:
            FileNotFoundError: If no valid model/evaluation pairs are found.
            ValueError: If no model can be successfully loaded.
        """
        self._models_dir = models_dir
        self._primary_metric = primary_metric
        self._logger = logging.getLogger(__name__)

        # Discover and rank candidates
        candidates = self._discover_candidates()
        if not candidates:
            raise FileNotFoundError(
                f"No model/evaluation pairs found in {models_dir.resolve()}. "
                f"Expected .pkl/.pth files with paired _evaluation.json files."
            )

        self._logger.info(
            "Discovered %d model candidate(s) in %s",
            len(candidates),
            models_dir.resolve(),
        )

        # Rank by primary metric (descending)
        candidates.sort(key=lambda c: getattr(c, primary_metric), reverse=True)

        # Log ranking
        for rank, candidate in enumerate(candidates, start=1):
            metric_value = getattr(candidate, primary_metric)
            self._logger.info(
                "  #%d: %s — %s=%.4f (recall=%.4f, precision=%.4f, FPR=%.4f)",
                rank,
                candidate.model_name,
                primary_metric,
                metric_value,
                candidate.recall,
                candidate.precision,
                candidate.fpr,
            )

        # Try loading models in rank order (graceful fallback)
        self.selected_candidate: Optional[ModelCandidate] = None
        self._model: Any = None

        for candidate in candidates:
            try:
                self._model = self._load_model(candidate.model_path)
                self.selected_candidate = candidate
                self._logger.info(
                    "✅ Selected model: %s (PR-AUC=%.4f, %s=%s)",
                    candidate.model_name,
                    candidate.pr_auc,
                    primary_metric,
                    getattr(candidate, primary_metric),
                )
                break
            except Exception as exc:
                self._logger.warning(
                    "Failed to load %s: %s. Trying next candidate...",
                    candidate.model_name,
                    exc,
                )
                continue

        if self._model is None or self.selected_candidate is None:
            raise ValueError(
                "All model candidates failed to load. "
                "Check logs for per-model error details."
            )

    @property
    def model(self) -> Any:
        """The loaded model object (sklearn-compatible API)."""
        return self._model

    @property
    def model_name(self) -> str:
        """Human-readable name of the selected model."""
        if self.selected_candidate is None:
            return "unknown"
        return self.selected_candidate.model_name

    def _discover_candidates(self) -> List[ModelCandidate]:
        """Scan models_dir and build candidate list from evaluation files.

        Returns:
            List of ModelCandidate objects with parsed evaluation metrics.

        For each .pkl or .pth file, searches for a paired evaluation JSON
        using common naming conventions: <model>_evaluation.json,
        <model>_metrics.json, or <model>_eval.json.
        """
        candidates: List[ModelCandidate] = []
        model_files = list(self._models_dir.glob("*.pkl")) + list(
            self._models_dir.glob("*.pth")
        )

        for model_path in model_files:
            # Skip scaler files
            if "scaler" in model_path.name.lower():
                continue

            eval_path = self._find_eval_file(model_path)
            if eval_path is None:
                self._logger.warning(
                    "No evaluation JSON found for %s — skipping",
                    model_path.name,
                )
                continue

            try:
                candidate = self._parse_candidate(model_path, eval_path)
                candidates.append(candidate)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                self._logger.warning(
                    "Failed to parse evaluation for %s: %s",
                    model_path.name,
                    exc,
                )
                continue

        return candidates

    def _find_eval_file(self, model_path: Path) -> Optional[Path]:
        """Find the evaluation JSON paired with a model file.

        Searches for files with the same stem plus known eval suffixes.

        Args:
            model_path: Path to the .pkl or .pth model file.

        Returns:
            Path to evaluation JSON, or None if not found.
        """
        stem = model_path.stem  # e.g., "lightgbm_advanced"

        # Try: lightgbm_advanced_evaluation.json
        for suffix in self.EVAL_SUFFIXES:
            candidate = model_path.parent / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

        # Try: lightgbm_advanced.json (if stem doesn't end with suffix already)
        # Also try common naming patterns without the model name prefix
        for suffix in self.EVAL_SUFFIXES:
            # Handle case where JSON is just named like the model but without suffix pattern
            for pattern in [f"{stem}.json"]:
                candidate = model_path.parent / pattern
                if candidate.exists() and candidate != model_path:
                    return candidate

        return None

    def _parse_candidate(
        self, model_path: Path, eval_path: Path
    ) -> ModelCandidate:
        """Parse a ModelCandidate from model and evaluation files.

        Args:
            model_path: Path to the serialized model.
            eval_path: Path to the evaluation JSON.

        Returns:
            Populated ModelCandidate.

        Raises:
            KeyError: If required metrics are missing from evaluation JSON.
            json.JSONDecodeError: If evaluation JSON is malformed.
        """
        with open(eval_path, "r") as f:
            metrics = json.load(f)

        # Extract metrics with fallbacks for different naming conventions
        pr_auc = metrics.get("pr_auc", metrics.get("average_precision", 0.0))
        roc_auc = metrics.get("roc_auc", 0.0)
        recall = metrics.get("recall", 0.0)
        precision = metrics.get("precision", 0.0)
        fpr = metrics.get(
            "false_positive_rate",
            metrics.get("fpr", 0.0),
        )
        inference_ms = metrics.get(
            "inference_time_ms_per_sample",
            metrics.get("inference_time_seconds", 0.0) * 1000.0,
        )

        return ModelCandidate(
            model_path=model_path,
            model_name=model_path.stem,
            pr_auc=float(pr_auc),
            roc_auc=float(roc_auc),
            recall=float(recall),
            precision=float(precision),
            fpr=float(fpr),
            inference_ms=float(inference_ms),
        )

    def _load_model(self, path: Path) -> Any:
        """Load a model with automatic format detection.

        Supports:
            - Pickle: sklearn models, XGBoost sklearn wrapper, LightGBM sklearn wrapper
            - PyTorch: .pth files (torch.load)

        Args:
            path: Path to the model file.

        Returns:
            Deserialized model object with predict_proba() interface.

        Raises:
            ValueError: If format is unsupported or file is corrupted.
        """
        import pickle

        suffix = path.suffix.lower()

        if suffix == ".pkl":
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as exc:
                raise ValueError(
                    f"Failed to unpickle model from {path}: {exc}"
                ) from exc

        elif suffix == ".pth":
            # PRODUCTION NOTE: Add torch load with weights_only=True for security
            # Currently torch.load(..., weights_only=False) has security risks
            try:
                import torch
                return torch.load(path, map_location="cpu", weights_only=True)
            except ImportError:
                raise ValueError(
                    f"Cannot load PyTorch model from {path}: torch not installed"
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to load PyTorch model from {path}: {exc}"
                ) from exc
        else:
            raise ValueError(
                f"Unsupported model format: {suffix}. Expected .pkl or .pth"
            )


# ---------------------------------------------------------------------------
# Artifact Loading (Scaler)
# ---------------------------------------------------------------------------


def load_scaler(path: Path) -> Any:
    """Load the fitted StandardScaler from disk.

    Args:
        path: Filesystem path to the scaler pickle file.

    Returns:
        Fitted sklearn StandardScaler instance.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If deserialization fails.
    """
    import pickle

    if not path.exists():
        raise FileNotFoundError(f"Scaler not found at {path.resolve()}")

    try:
        with open(path, "rb") as f:
            scaler = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as exc:
        raise ValueError(
            f"Failed to deserialize scaler from {path}: {exc}"
        ) from exc

    logger = logging.getLogger(__name__)
    logger.info("Loaded scaler from %s", path.resolve())
    return scaler


# ---------------------------------------------------------------------------
# Transaction Preprocessor
# ---------------------------------------------------------------------------


class TransactionPreprocessor:
    """Applies feature engineering identical to the training pipeline.

    Stateless except for the fitted scaler. Designed to be reused
    across multiple worker threads without mutation.

    Production upgrade path:
        Inject OnlineFeatureStoreClient for per-card velocity features.
        Current implementation returns placeholder zeros because
        single-request inference cannot compute true rolling windows.
    """

    REQUIRED_FIELDS: Tuple[str, ...] = ("Time", "Amount")

    def __init__(
        self,
        scaler: Any,
        feature_config: FeatureAssemblyConfig,
    ) -> None:
        """Initialize preprocessor with fitted scaler and feature layout.

        Args:
            scaler: Fitted sklearn StandardScaler for Amount_log.
            feature_config: Defines feature vector assembly order.
        """
        self._scaler = scaler
        self._feature_config = feature_config
        self._logger = logging.getLogger(f"{__name__}.TransactionPreprocessor")

    def transform(self, payload: Dict[str, Any]) -> np.ndarray:
        """Convert raw JSON payload into model-ready feature vector.

        Processing order (must match training pipeline):
            1. Validate and extract Amount, Time, V1–V28
            2. Amount_log = log(Amount + 1)
            3. Amount_scaled = scaler.transform(Amount_log)
            4. Cyclical time: hour_sin, hour_cos
            5. Velocity features: placeholder zeros
            6. Top-k PCA features
            7. Concatenate in FeatureAssemblyConfig order

        Args:
            payload: Raw JSON dict with keys Time, Amount, V1–V28.

        Returns:
            2D numpy array of shape (1, n_features) for model.predict_proba().

        Raises:
            ValueError: If required fields are missing or values are invalid.
        """
        self._validate_payload(payload)

        features: List[float] = []

        # --- Step 1: Amount features ---
        amount_raw = float(payload["Amount"])
        amount_log = float(np.log1p(amount_raw))

        if self._feature_config.include_amount_scaled:
            # StandardScaler expects 2D input
            amount_scaled = float(
                self._scaler.transform([[amount_log]])[0, 0]
            )
            features.append(amount_scaled)

        if self._feature_config.include_amount_log:
            features.append(amount_log)

        # --- Step 2: Cyclical time features ---
        if self._feature_config.include_time_cyclical:
            time_raw = float(payload["Time"])
            hour_of_day = (time_raw / 3600.0) % 24
            hour_sin = float(np.sin(2 * np.pi * hour_of_day / 24.0))
            hour_cos = float(np.cos(2 * np.pi * hour_of_day / 24.0))
            features.extend([hour_sin, hour_cos])

        # --- Step 3: Velocity features (placeholder) ---
        # PRODUCTION: Replace with online feature store lookup
        velocity_count = self._feature_config.velocity_feature_count
        features.extend([0.0] * velocity_count)

        # --- Step 4: Top-k PCA features in training order ---
        for pca_col in self._feature_config.pca_features_ordered:
            if pca_col not in payload:
                raise ValueError(
                    f"Missing PCA feature: '{pca_col}'. "
                    f"All {len(self._feature_config.pca_features_ordered)} "
                    f"PCA features from feature_config are required."
                )
            features.append(float(payload[pca_col]))

        feature_array = np.array(features, dtype=np.float64).reshape(1, -1)

        expected = self._feature_config.total_feature_count
        if feature_array.shape[1] != expected:
            raise RuntimeError(
                f"Feature count mismatch: assembled {feature_array.shape[1]} "
                f"features, expected {expected}. Check FeatureAssemblyConfig."
            )

        return feature_array

    def _validate_payload(self, payload: Dict[str, Any]) -> None:
        """Check payload structure and value ranges.

        Args:
            payload: Raw request JSON body.

        Raises:
            ValueError: On missing fields or out-of-range values.
        """
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in payload:
                raise ValueError(f"Missing required field: '{field}'")

        # Validate Amount
        amount = payload["Amount"]
        if not isinstance(amount, (int, float)):
            raise ValueError(
                f"'Amount' must be numeric, got {type(amount).__name__}"
            )
        if amount < 0:
            raise ValueError(f"'Amount' must be >= 0, got {amount}")
        # Note: max_input_amount is validated in the server layer for clearer
        # error separation (business logic vs feature engineering).

        # Validate Time
        time_val = payload["Time"]
        if not isinstance(time_val, (int, float)):
            raise ValueError(
                f"'Time' must be numeric, got {type(time_val).__name__}"
            )
        if time_val < 0:
            raise ValueError(f"'Time' must be >= 0, got {time_val}")


# ---------------------------------------------------------------------------
# Structured Request Logger
# ---------------------------------------------------------------------------


class RequestLogger:
    """Writes structured JSON logs per prediction request.

    Log format designed for ingestion into ELK / Datadog / CloudWatch:
        {"timestamp":"...","transaction_hash":"...","fraud_probability":0.92,...}

    Production upgrade path:
        Emit logs to Kafka topic or directly to time-series DB (ClickHouse).
        Add OpenTelemetry spans for distributed tracing across services.
    """

    def __init__(self, log_path: Path, enabled: bool = True) -> None:
        """Initialize the request logger.

        Args:
            log_path: Filesystem path for the log file.
            enabled: If False, all log calls are no-ops (for testing).
        """
        self._log_path = log_path
        self._enabled = enabled
        self._logger = logging.getLogger(f"{__name__}.RequestLogger")

        if enabled:
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Dedicated file handler to avoid mixing with app logs
            self._file_handler = logging.FileHandler(str(log_path))
            self._file_handler.setFormatter(
                logging.Formatter(
                    '{"timestamp":"%(asctime)s",%(message)s}',
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
            self._logger.addHandler(self._file_handler)
            self._logger.setLevel(logging.INFO)
            self._logger.propagate = False

    def log_prediction(
        self,
        transaction_hash: str,
        fraud_probability: float,
        is_fraud: bool,
        latency_ms: float,
        input_amount: float,
        threshold: float,
        model_name: str,
    ) -> None:
        """Record a single prediction with structured context.

        Args:
            transaction_hash: SHA-256 hash of input payload.
            fraud_probability: Model output probability (0–1).
            is_fraud: Binary decision after thresholding.
            latency_ms: End-to-end prediction time in milliseconds.
            input_amount: Raw transaction amount from request.
            threshold: Decision threshold used.
            model_name: Name of the model that served this prediction.
        """
        if not self._enabled:
            return

        log_entry = (
            f'"transaction_hash":"{transaction_hash}",'
            f'"fraud_probability":{fraud_probability:.6f},'
            f'"is_fraud":{str(is_fraud).lower()},'
            f'"latency_ms":{latency_ms:.2f},'
            f'"input_amount":{input_amount:.2f},'
            f'"threshold":{threshold:.4f},'
            f'"model_name":"{model_name}"'
        )
        self._logger.info(log_entry)


# ---------------------------------------------------------------------------
# Model Server
# ---------------------------------------------------------------------------


class FraudModelServer:
    """Flask-based REST API for real-time fraud prediction.

    Automatically selects the best model from the models directory
    based on evaluation metrics. Exposes /predict and /health endpoints.

    Usage:
        config = ServingConfig(
            models_dir=Path("D:/fraud-detection-real-time/models"),
            scaler_path=Path("D:/fraud-detection-real-time/models/"
                              "autoencoder_ssl_scaler.pkl"),
        )
        server = FraudModelServer(config)
        server.run()

    Production upgrade path:
        1. Flask → FastAPI (async handlers, dependency injection, OpenAPI)
        2. Gunicorn + Uvicorn workers behind Nginx reverse proxy
        3. Kubernetes liveness/readiness probes via /health
        4. Prometheus /metrics endpoint for latency histogram, error rate
        5. Model hot-reload via filesystem watch or MLflow webhook
    """

    def __init__(self, config: ServingConfig) -> None:
        """Initialize the server with best-model selection.

        Args:
            config: Immutable serving configuration.

        Raises:
            FileNotFoundError: If models_dir or scaler_path don't exist.
            ValueError: If no viable model can be loaded.
        """
        self._config = config
        self._setup_logging()
        self._logger = logging.getLogger(__name__)

        self._logger.info("=" * 60)
        self._logger.info("Initializing Fraud Detection Model Server")
        self._logger.info("=" * 60)

        # -- Step 1: Select best model --
        self._logger.info(
            "Scanning models directory: %s", config.models_dir.resolve()
        )
        self._registry = ModelRegistry(
            models_dir=config.models_dir,
            primary_metric=config.primary_metric,
        )

        # -- Step 2: Load scaler --
        self._scaler = load_scaler(config.scaler_path)

        # -- Step 3: Wire preprocessing --
        self._preprocessor = TransactionPreprocessor(
            scaler=self._scaler,
            feature_config=config.feature_config,
        )

        # -- Step 4: Request logger --
        self._request_logger = RequestLogger(
            log_path=Path("logs/service.log"),
            enabled=config.enable_request_logging,
        )

        # -- Step 5: Create Flask app --
        self._app = Flask("fraud_detection_api")
        self._register_routes()

        # -- Summary --
        self._logger.info("─" * 60)
        self._logger.info("Server initialized successfully.")
        self._logger.info("  Model:       %s", self._registry.model_name)
        self._logger.info(
            "  PR-AUC:      %.4f",
            self._registry.selected_candidate.pr_auc
            if self._registry.selected_candidate
            else 0.0,
        )
        self._logger.info("  Threshold:   %.4f", config.threshold)
        self._logger.info(
            "  Features:    %d", config.feature_config.total_feature_count
        )
        self._logger.info("=" * 60)

    def _setup_logging(self) -> None:
        """Configure application-level logging."""
        logging.basicConfig(
            level=getattr(
                logging, self._config.log_level.upper(), logging.INFO
            ),
            format=(
                "%(asctime)s [%(levelname)s] "
                "%(name)s: %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    def _register_routes(self) -> None:
        """Register Flask route handlers."""

        @self._app.route("/health", methods=["GET"])
        def health() -> Tuple[Dict[str, Any], int]:
            """Health check endpoint (Kubernetes-compatible).

            Returns:
                Service health status and current model metadata.
            """
            status = {
                "status": "healthy",
                "model": self._registry.model_name,
                "threshold": self._config.threshold,
                "primary_metric": self._config.primary_metric,
                "pr_auc": (
                    self._registry.selected_candidate.pr_auc
                    if self._registry.selected_candidate
                    else None
                ),
            }
            return status, 200

        @self._app.route("/predict", methods=["POST"])
        def predict() -> Tuple[Any, int]:
            """Main fraud prediction endpoint.

            Expects JSON body: {"Time": ..., "Amount": ..., "V1": ..., ... "V28": ...}
            Returns fraud_probability (0–1) and is_fraud (boolean).

            Returns:
                JSON response with prediction result and HTTP status code.
            """
            start_time = time.perf_counter()

            # --- Parse JSON ---
            try:
                payload = request.get_json(force=True)
            except Exception as exc:
                return (
                    jsonify(
                        {
                            "error": "Invalid JSON body",
                            "detail": str(exc),
                        }
                    ),
                    400,
                )

            if payload is None:
                return jsonify({"error": "Request body must be JSON"}), 400

            # --- Validate Amount (business rule) ---
            if payload.get("Amount", 0) > self._config.max_input_amount:
                return (
                    jsonify(
                        {
                            "error": "Amount exceeds maximum",
                            "detail": (
                                f"Amount {payload['Amount']} exceeds "
                                f"maximum {self._config.max_input_amount}"
                            ),
                        }
                    ),
                    422,
                )

            # --- Transaction hash (for idempotency & tracing) ---
            payload_str = json.dumps(payload, sort_keys=True)
            txn_hash = hashlib.sha256(payload_str.encode()).hexdigest()

            # --- Preprocessing ---
            try:
                features = self._preprocessor.transform(payload)
            except ValueError as exc:
                self._logger.warning(
                    "Preprocessing failed [txn=%s]: %s",
                    txn_hash[:16],
                    exc,
                )
                return (
                    jsonify(
                        {
                            "error": "Invalid input",
                            "detail": str(exc),
                            "transaction_hash": txn_hash,
                        }
                    ),
                    422,
                )

            # --- Inference ---
            try:
                proba = self._registry.model.predict_proba(features)
                fraud_probability = float(proba[0, 1])
            except Exception as exc:
                self._logger.exception(
                    "Inference failed [txn=%s]", txn_hash[:16]
                )
                return (
                    jsonify(
                        {
                            "error": "Inference error",
                            "detail": "Internal server error during prediction",
                            "transaction_hash": txn_hash,
                        }
                    ),
                    500,
                )

            is_fraud = fraud_probability >= self._config.threshold
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # --- Structured logging ---
            self._request_logger.log_prediction(
                transaction_hash=txn_hash,
                fraud_probability=fraud_probability,
                is_fraud=is_fraud,
                latency_ms=latency_ms,
                input_amount=float(payload.get("Amount", 0)),
                threshold=self._config.threshold,
                model_name=self._registry.model_name,
            )

            self._logger.debug(
                "Prediction [txn=%s]: prob=%.4f fraud=%s lat=%.2fms model=%s",
                txn_hash[:16],
                fraud_probability,
                is_fraud,
                latency_ms,
                self._registry.model_name,
            )

            # --- Response ---
            return (
                jsonify(
                    {
                        "fraud_probability": round(fraud_probability, 6),
                        "is_fraud": is_fraud,
                        "transaction_hash": txn_hash,
                        "model": self._registry.model_name,
                    }
                ),
                200,
            )

    def run(
        self,
        host: str = "localhost",
        port: int = 5000,
        debug: bool = False,
    ) -> None:
        """Start the Flask development server.

        WARNING: Flask's built-in server is single-threaded. For production:
            gunicorn -w 4 -b 0.0.0.0:5000 "serve:app"

        Args:
            host: Bind address.
            port: Bind port.
            debug: Flask debug mode (MUST be False in production).
        """
        self._logger.info(
            "Starting development server on %s:%d (debug=%s)...",
            host,
            port,
            debug,
        )
        self._app.run(host=host, port=port, debug=debug)


# ---------------------------------------------------------------------------
# WSGI Application Entry Point
# ---------------------------------------------------------------------------
# Gunicorn: gunicorn src.serve:app
# The app is built lazily on first import using environment variables.

_APP_INSTANCE: Optional[Flask] = None


def get_app() -> Flask:
    """Build or retrieve the Flask application instance.

    Uses environment variables for configuration:
        FRAUD_MODELS_DIR    — path to models directory
        FRAUD_SCALER_PATH   — path to scaler .pkl
        FRAUD_THRESHOLD     — decision threshold (default: 0.5)
        FRAUD_METRIC        — primary ranking metric (default: pr_auc)

    Returns:
        Configured Flask application.
    """
    global _APP_INSTANCE
    if _APP_INSTANCE is not None:
        return _APP_INSTANCE

    config = ServingConfig(
        models_dir=Path(
            os.environ.get(
                "FRAUD_MODELS_DIR",
                "D:/fraud-detection-real-time/models",
            )
        ),
        scaler_path=Path(
            os.environ.get(
                "FRAUD_SCALER_PATH",
                "D:/fraud-detection-real-time/models/"
                "autoencoder_ssl_scaler.pkl",
            )
        ),
        threshold=float(os.environ.get("FRAUD_THRESHOLD", "0.5")),
        primary_metric=os.environ.get("FRAUD_METRIC", "pr_auc"),
    )
    server = FraudModelServer(config)
    _APP_INSTANCE = server._app
    return _APP_INSTANCE


# Module-level app for WSGI servers
app = get_app()