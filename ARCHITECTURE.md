# ARCHITECTURE.md

## Project: Real-Time Credit Card Fraud Detection — Full Architecture

### 1. System Overview

This document defines the complete architecture for an end-to-end machine learning pipeline that detects fraudulent credit card transactions at the point of authorization. It covers both a **Baseline** implementation for local development on an Intel Core i5 laptop using Python, Pandas, and Scikit-learn, and an **Advanced** production implementation designed for distributed computing, real-time streaming, and automated operations. Both share identical component boundaries and data contracts, ensuring seamless progression from prototype to production.

---

### 2. Design Principles

| Principle | Baseline | Advanced |
|---|---|---|
| Compute | Single-node, in-memory | Distributed, cloud-native |
| Storage | Local filesystem (CSV, Parquet, Pickle) | Cloud object storage, Feature Store, Model Registry |
| Orchestration | Manual script execution | Airflow / Kubeflow DAGs, CI/CD pipelines |
| Latency | Not applicable (offline) | Sub-50ms real-time serving |
| Observability | Console logs, local reports | Dashboards, alerts, automated drift detection |
| Scaling Path | Identical stage interfaces and data contracts allow drop-in replacement of each component |

---

### 3. Pipeline Architecture (Textual Diagram)

```
                          ┌──────────────────────────────────────────────────────────────┐
                          │                     ORCHESTRATION LAYER                       │
                          │         (Manual scripts → Airflow/Kubeflow DAGs)              │
                          └──────────────────────────────────────────────────────────────┘

   DATA SOURCES                    PIPELINE STAGES                           SERVING & MONITORING
   ─────────────                   ────────────────                          ────────────────────

 ┌──────────────┐              ┌──────────────────────┐
 │  Raw CSV     │──────────────▶  1. DATA INGESTION   │
 │  Kafka Topic │              └──────────┬───────────┘
 │  S3 Bucket   │                         │
 └──────────────┘                         ▼
                                ┌──────────────────────┐
                                │  2. DATA VALIDATION  │──────────────▶ validation_report.json
                                └──────────┬───────────┘
                                          │
                                          ▼
                                ┌──────────────────────┐
                                │ 3. DATA PREPROCESSING│──────────────▶ scaler.pkl, cleaned.parquet
                                └──────────┬───────────┘
                                          │
                                          ▼
                                ┌──────────────────────┐
                                │     4. EDA           │──────────────▶ reports/eda/
                                └──────────┬───────────┘
                                          │
                                          ▼
                                ┌──────────────────────┐
                                │ 5. FEATURE ENGINEER. │──────────────▶ features.parquet, feature_config.json
                                └──────────┬───────────┘
                                          │
                                          ▼
                                ┌──────────────────────┐
                                │  6. DATA SPLITTING   │──────────────▶ train.parquet, test.parquet
                                └──────────┬───────────┘
                                          │
                                          ▼
                                ┌──────────────────────┐
                                │  7. MODEL TRAINING   │──────────────▶ model.pkl, model_metadata.json
                                └──────────┬───────────┘
                                          │
                                          ▼
                                ┌──────────────────────┐
                                │ 8. MODEL EVALUATION  │──────────────▶ evaluation_report.json
                                └──────────┬───────────┘
                                          │
                                          ▼
                                ┌──────────────────────┐
                                │  9. MODEL DEPLOYMENT │◀─────── REST/gRPC requests
                                └──────────┬───────────┘
                                          │                       ┌──────────────────────┐
                                          └───────────────────────▶ 10. MONITORING &      │
                                                                   │     MAINTENANCE      │
                                                                   └──────────────────────┘
```

---

### 4. Component Specifications

---

## 4.1 Data Ingestion

### Baseline Approach

**Description:** A minimal file-reader module that loads the raw dataset from a local CSV file into a Pandas DataFrame. This module acts as a single point of entry for offline development and experimentation on a laptop with limited memory.

**Responsibilities:**
- Read a CSV file path provided via a configuration variable (`config/pipeline_config.yaml`).
- Perform a quick sanity check on number of rows and columns loaded, reporting memory usage.
- Return a DataFrame object for downstream steps.

**Input:** A file path string (`data/raw/creditcard.csv`) pointing to a local CSV. The CSV contains columns like `Time`, `Amount`, `Class`, PCA features `V1`–`V28`.

**Output:** A pandas DataFrame object held in memory, containing all raw rows and columns as they appear in the file (`df_raw`). Schema: float64 for numeric features, int64 for `Class`. Console log confirms row count and memory footprint.

### Advanced Approach

**Description:** A scalable ingestion service that supports both batch historical loads and real-time streaming. It abstracts the data source, allowing seamless switching between a data lake and a message queue while ensuring exactly-once semantics.

**Responsibilities:**
- Batch ingestion from cloud object storage (S3/GCS) using Apache Spark or Dask for parallel reads.
- Real-time ingestion from Kafka topics (`transactions.raw`) with Avro schema validation, using a schema registry.
- Idempotent record handling using transaction IDs to avoid duplicates.
- Publish raw events to a staging area (data lake zone / Kafka dead-letter queue on failure).
- Automatically register data schema and version in a metadata store (e.g., Hive Metastore or AWS Glue Catalog).

**Input:** Source identifiers: an S3 prefix pattern (e.g., `s3://fraud-lake/raw/yyyy/mm/dd/`) or a Kafka bootstrap server and topic. Schema definition from a central schema registry (e.g., Apicurio, Confluent).

**Output:** A distributed DataFrame (Spark/Dask) for batch, or a structured streaming DataFrame for real-time. Events are also persisted to a raw zone in Parquet format with partition key `dt=YYYY-MM-DD` for query efficiency.

---

## 4.2 Data Validation

### Baseline Approach

**Description:** A lightweight validation script that checks basic expectations about the dataset before preprocessing. It uses the `pandera` library to define schema constraints in code.

**Responsibilities:**
- Verify that all required columns exist (`Time`, `V1`–`V28`, `Amount`, `Class`).
- Check that `Amount` is non-negative.
- Check that `Class` is either 0 or 1.
- Report number of null values per column.
- Raise `ValidationError` if critical constraints fail, halting the pipeline.

**Input:** The raw DataFrame from Data Ingestion.

**Output:** A validation report (`reports/validation_report.json`) with pass/fail status per expectation. The unchanged DataFrame if validation passes; otherwise raises an exception with details.

### Advanced Approach

**Description:** A full data contract enforcement layer using Great Expectations (GE) integrated into the CI/CD pipeline. Validation runs automatically every time new data lands, with results stored for auditing and anomaly detection.

**Responsibilities:**
- Define a suite of expectations: column existence, types, value ranges (e.g., `Amount` between 0 and 25,000), null proportions < 5%, distribution shifts vs. reference window.
- Run validation as a pre-commit hook in the feature pipeline and as a step in an orchestrated DAG (Airflow/Prefect).
- Publish validation results to a data quality dashboard (e.g., GE Data Docs on S3) and send alerts (Slack, PagerDuty) on failure.
- Perform time-window comparisons (e.g., Kolmogorov-Smirnov test on `Amount` distribution vs. last week) to catch concept drift early.
- Store validation artifacts (expectation suites, checkpoint results) in a versioned object store.

**Input:** A DataFrame from batch/streaming Data Ingestion, plus a GE expectation suite JSON and checkpoint configuration.

**Output:** A validation result document (JSON) with detailed per-expectation outcomes, success/failure flags, and distribution diagnostic plots uploaded to cloud storage. Downstream steps only proceed if critical validations pass.

---

## 4.3 Data Cleaning / Preprocessing

### Baseline Approach

**Description:** A straightforward script that handles missing values, scales numerical features, and prepares the data for modeling with minimal memory footprint. All operations are performed eagerly on a pandas DataFrame.

**Responsibilities:**
- Impute missing values: replace NaNs in PCA features with 0 (common for anonymized data) and in `Amount` with median if needed.
- Apply `StandardScaler` to `Amount` and `Time`; PCA features (`V1`–`V28`) are already scaled but may be re-normalized if needed.
- Remove duplicate rows based on transaction identifiers if present.
- Drop any rows with impossible values (negative `Amount` if found despite validation).
- Encode `Class` as integer labels.

**Input:** Validated DataFrame.

**Output:** `data/processed/cleaned.parquet` — a cleaned pandas DataFrame with exactly the same columns, now free of nulls and with numeric features standardized. `artifacts/scaler.pkl` — fitted scaler object saved for later use during inference.

### Advanced Approach

**Description:** A modular preprocessing pipeline that runs as a series of stateless transformations in a distributed compute framework and supports point-in-time correctness. It is designed for both batch training and real-time serving.

**Responsibilities:**
- Use Apache Spark ML Pipelines or a Scikit-learn pipeline packaged inside a serverless function (e.g., AWS Lambda with a feature transformation container) for portability.
- Apply consistent imputation strategies stored in the feature store metadata: PCA nulls → 0, categorical nulls → a special token.
- Scale features using statistics computed only on the training set (precomputed and stored in a configuration store).
- Handle categorical features (if dataset version includes MCC, terminal type) using one-hot or target encoding with smoothing, avoiding data leakage by fitting on a holdout split.
- Log every transformation step and its parameters in MLflow for lineage tracking.
- Output transformation artifacts (scaler, imputer, encoder) as serialized objects in a model registry (MLflow Model).

**Input:** Cleaned DataFrame (batch view from Spark/Dask), plus configuration referencing a pre-existing train/validation split time boundary to prevent future information leakage.

**Output:** A transformed dataset saved in columnar format (Parquet) partitioned by date, and a serialized preprocessing pipeline saved as an MLflow artifact with versioning. During serving, the same pipeline is applied in the model server.

---

## 4.4 Exploratory Data Analysis (EDA)

### Baseline Approach

**Description:** An interactive Jupyter notebook that generates basic descriptive statistics and visualizations to understand the data. The output is a manually reviewed set of plots and tables, not part of the automated pipeline.

**Responsibilities:**
- Print class distribution: count and percentage of fraud (`Class=1`).
- Compute summary statistics for `Amount` and `Time`.
- Plot histograms of transaction amounts split by fraud vs. non-fraud.
- Plot a correlation heatmap of PCA features (`V1`–`V28`) to spot highly correlated dimensions.
- Check for time-based patterns: fraud rate by hour of day (derived from `Time` feature).
- Display a table of missing values per column.

**Input:** `data/processed/cleaned.parquet` from Data Preprocessing.

**Output:** A set of PNG images and HTML tables saved in `reports/eda/`, plus key insights documented in the notebook's markdown cells.

### Advanced Approach

**Description:** Automated, scheduled EDA that produces a comprehensive data profile and drift report with minimal human intervention. The results are versioned and shared as an interactive dashboard.

**Responsibilities:**
- Run `pandas_profiling` or `Sweetviz` on a scheduled cadence (daily) over the recent batch of transactions.
- Compare current data distribution with a reference profile (e.g., from the training window) to highlight feature drift.
- Automatically detect target distribution changes (fraud rate shift) and flag if the imbalance ratio deviates by more than 20% from baseline.
- Generate interactive HTML reports and upload to a shared cloud storage or a BI tool (e.g., a Databricks notebook dashboard).
- Trigger alerts if unexpected patterns (e.g., sudden spike in zero amounts, new dominant merchant categories) appear.

**Input:** Batch DataFrame from the cleaned data store, plus a reference profile JSON (generated during initial model training).

**Output:** A detailed EDA report (HTML/JSON) saved to a cloud bucket with a timestamped path. Key metrics (fraud rate, missing percentages, drift p-values) are logged to MLflow as artifacts for traceability.

---

## 4.5 Feature Engineering

### Baseline Approach

**Description:** A Pandas-based feature transformer that creates simple aggregated velocity features using rolling windows on transaction history. All calculations are done in-memory on the historical dataset for training.

**Responsibilities:**
- Sort data by `Time` and compute global temporal features (public datasets lack user/card IDs).
- Create time-based features: `hour` (0–23), `day` (integer), `hour_sin`, `hour_cos` (cyclical encoding).
- Create rolling-window velocity features using Pandas `.rolling()`:
  - `txn_count_1h`: count of transactions in the last 3600 seconds.
  - `txn_count_24h`: count of transactions in the last 86400 seconds.
  - `avg_amount_1h`: average transaction amount in the last 3600 seconds.
  - `avg_amount_24h`: average transaction amount in the last 86400 seconds.
- Add `Amount_log` = `log(Amount + 1)` for better scaling.
- Produce a final feature set that combines original PCA features with newly engineered features.

**Input:** `data/processed/cleaned.parquet` with a `Time` column (seconds from first transaction). Feature configuration specifying window sizes (3600, 86400).

**Output:** `data/processed/features.parquet` — a pandas DataFrame with all original and new columns. `artifacts/feature_config.json` — list of feature names used for training.

### Advanced Approach

**Description:** A Feature Store-based pipeline that decouples feature computation from model training and ensures point-in-time correctness for historical and real-time features. Velocity features are precomputed at scale using stream processing or batch windows and stored for online serving.

**Responsibilities:**
- Maintain an offline feature store (e.g., Feast, Tecton) where all features are registered and versioned.
- Compute batch velocity features using Spark SQL or Flink: aggregate sessions per card ID over sliding windows of 1h, 6h, 24h, 7d — counts, mean amounts, max amounts, standard deviations.
- Store those features in a feature group with event timestamps, ensuring training uses only features available at the time of each transaction (no look-ahead bias).
- For real-time inference, serve the latest precomputed features from an online store (Redis/DynamoDB) and compute on-the-fly features (e.g., current transaction amount deviation from 1h moving average) in the model server.
- Include advanced transformations: PCA feature cross-products (interactions), cluster distances from an anomaly detector (e.g., distance to nearest Isolation Forest centroid as a feature), and rolling standard deviation of spending.
- Automate feature registration and documentation through a CI/CD step that runs feature validations (min/max values, lack of nulls) before promoting to production.

**Input:** Streaming transaction events (Kafka) with card/user identifiers, and batch historical tables in the data lake. Feature registry YAML defining feature views and entities.

**Output:** Offline feature dataset (in Parquet) for training, with a historical feature retrieval API. Online feature vectors returned by the feature server at serving time (<10 ms). All feature engineering code versioned and stored as a Feast Feature Repository.

---

## 4.6 Data Splitting

### Baseline Approach

**Description:** A simple time-aware split that divides the dataset into training and test sets based on a temporal cutoff. The final portion of time becomes the holdout set, mimicking production deployment conditions.

**Responsibilities:**
- Sort transactions by `Time` (seconds elapsed since first transaction).
- Determine a split timestamp such that the first 80% of the time range is training, the remaining 20% is testing.
- Ensure no data leakage: all transactions in training occur before the earliest test transaction.
- Save the resulting DataFrames and record the split timestamp.

**Input:** `data/processed/features.parquet` with the `Time` column still present for splitting logic.

**Output:** `data/processed/train.parquet` and `data/processed/test.parquet`, each with the same feature columns. `config/split_timestamp.txt` recording the boundary value.

### Advanced Approach

**Description:** A robust, automated splitting strategy that uses time-series cross-validation for model selection and a holdout window that respects real-world deployment delay and fraud label reporting lag.

**Responsibilities:**
- Perform time-series split with multiple backtesting folds (e.g., expanding window, 3 folds). For each fold, train on data up to time T, validate on data in [T, T+delta].
- Automatically select the split intervals using a scheduling configuration (e.g., train on 12 weeks, validate on next 2 weeks).
- Implement a "delay-aware" split: exclude transactions that would not yet be labeled due to fraud reporting lag (simulating real-world reporting delay of 1–3 days). The validation set does not include the most recent unlabeled data.
- Use stratified sampling within the validation window to maintain fraud prevalence, if necessary.
- Save split metadata (timestamps, fold assignments) in a versioned artifact to be reused in evaluation.
- Integrate with a feature store to retrieve point-in-time correct feature views for each fold.

**Input:** Feature store's historical feature set with timestamps, plus a configuration YAML specifying fold durations, reporting delay, and test window.

**Output:** A set of training and validation DataFrames per fold stored in Parquet, along with a holdout test set for final evaluation. A timestamp boundary map is logged to MLflow for reproducibility.

---

## 4.7 Model Training

### Baseline Approach

**Description:** A single Python script that trains three baseline classifiers using Scikit-learn on the local training set. It handles class imbalance with SMOTE oversampling and uses simple 3-fold cross-validation for model selection.

**Responsibilities:**
- Load `train.parquet`; separate `X_train` (drop `Class`, `Time`) and `y_train`.
- Apply SMOTE (from imbalanced-learn) to the training folds to balance classes.
- Perform 3-fold cross-validation for each candidate: Logistic Regression, Decision Tree (max_depth=5), Random Forest (n_estimators=50).
- Compare mean PR-AUC across folds and select the best algorithm.
- Retrain the best model on the full training set.
- Serialize the best model.

**Input:** `data/processed/train.parquet` from Data Splitting.

**Output:** `artifacts/model.pkl` — serialized Scikit-learn model (pipeline including scaler if embedded). `artifacts/model_metadata.json` — selected algorithm, hyperparameters, feature list, and CV PR-AUC.

### Advanced Approach

**Description:** A fully orchestrated training pipeline with experiment tracking, distributed hyperparameter tuning, and model selection using a model registry. Supports multiple model types including gradient boosting and neural networks.

**Responsibilities:**
- Use an orchestrator (Airflow / Kubeflow Pipelines) that triggers training on new labeled data or on a schedule.
- Load features from the offline feature store using point-in-time joins.
- Handle severe class imbalance via a combination of SMOTE, cost-sensitive learning (`scale_pos_weight` in XGBoost), and downsampling with probability calibration.
- Execute hyperparameter optimization with Optuna or Hyperopt across a cluster (e.g., distributed on Kubernetes), exploring XGBoost, LightGBM, and a small MLP (in TensorFlow/PyTorch) with tree-structured Parzen Estimators.
- Train an Isolation Forest for anomaly detection on the non-fraud class to generate an anomaly score feature later.
- Use MLflow to track all experiments: parameters, metrics, feature importance, and the resulting model artifact.
- Register the best candidate in the MLflow Model Registry, promote to "Staging" after passing evaluation thresholds.
- Store the model in a format conducive to low-latency deployment: XGBoost native format, ONNX, or compiled with `treelite`.

**Input:** Feature view from feature store (offline), training configuration YAML (hyperparameter search space, algorithms to include, objective metric PR-AUC).

**Output:** A registered model version in MLflow with production-ready artifacts (ONNX or `model.bst`), a complete set of training logs, and a model card documenting performance and fairness metrics.

---

## 4.8 Model Evaluation

### Baseline Approach

**Description:** A script that loads the test set and the trained model to compute essential performance metrics, with a focus on PR-AUC due to extreme class imbalance. Results are printed and saved to a JSON file.

**Responsibilities:**
- Load `test.parquet` and `model.pkl`.
- Generate predictions and probability scores on the test set.
- Compute PR-AUC, ROC-AUC, and precision, recall, F1 at default threshold 0.5.
- Determine a threshold that achieves a target recall of 80% (or a fixed false positive rate of 1%) and report precision at that threshold.
- Generate and save a confusion matrix at the selected threshold.

**Input:** `data/processed/test.parquet` and `artifacts/model.pkl`.

**Output:** `reports/evaluation_report.json` — file with all computed metrics (PR-AUC, ROC-AUC, precision@80recall, FPR@80recall, optimal_threshold). `reports/confusion_matrix.csv`.

### Advanced Approach

**Description:** A multi-faceted evaluation harness that simulates production conditions, computes business impact metrics, and validates model performance across data slices before promotion to production.

**Responsibilities:**
- Perform time-series aware backtesting: evaluate each fold from cross-validation and aggregate metrics.
- Compute business metrics: estimated fraud loss saved = fraud detected × average transaction amount × (1 − chargeback rate), and false positive rate translating to customer insult rate.
- Slice evaluation: performance by transaction amount buckets, hour of day, merchant category (if available) to ensure no biased degradation for specific segments.
- A/B test shadow deployment: log production traffic predictions from the challenger model and compare against the current champion model without affecting decisions; compute evaluation on these shadow logs.
- Generate a model validation report in HTML format with drift detection insights (population stability index, feature drift).
- Implement a gating function: if test PR-AUC drops more than 5% relative to the previous champion, or false positive rate exceeds 2%, block auto-promotion.
- Store evaluation artifacts in MLflow and update the model registry stage (Staging → Production or Archived).

**Input:** Model artifact from training, test set (holdout), and optionally streaming production logs for shadow evaluation. Configuration file specifying thresholds and slice definitions.

**Output:** A comprehensive evaluation report (JSON/HTML) with slice-level metrics and business KPIs, a decision flag (`promote: true/false`), and an updated model registry stage.

---

## 4.9 Model Deployment

### Baseline Approach

**Description:** A local REST API built with Flask that loads the serialized model and serves fraud predictions on single transaction requests. It runs on the laptop to mimic a real-time authorization endpoint for testing.

**Responsibilities:**
- Load `model.pkl` and `scaler.pkl` into memory on startup.
- Expose a `/predict` POST endpoint that accepts a JSON payload with feature values.
- Apply the same preprocessing and feature engineering steps identical to training (scale `Amount`, compute cyclical time, etc.).
- Return a JSON response with `fraud_probability` and `is_fraud` using the tuned threshold from evaluation.
- Log each request and prediction to `logs/service.log` for later analysis.

**Input:** HTTP POST request to `http://localhost:5000/predict` with a JSON body: `{"Time": 150000, "V1": -1.5, ..., "V28": 0.3, "Amount": 120.0}`.

**Output:** JSON response: `{"fraud_probability": 0.92, "is_fraud": true}`. Service runs as a single-process Flask development server.

### Advanced Approach

**Description:** A containerized, cloud-native deployment serving predictions at scale with ultra-low latency, integrated directly into the payment authorization gateway.

**Responsibilities:**
- Package the model artifact (ONNX format or compiled XGBoost) along with a lightweight serving runtime (FastAPI, KServe, or a custom C++ wrapper) in a Docker image.
- Use a model registry (MLflow) to pull the production model version; the serving container auto-updates on version promotion via a CI/CD pipeline (GitOps with ArgoCD).
- Deploy on Kubernetes with horizontal pod autoscaling (HPA) to handle transaction volume spikes. Latency SLA guaranteed by gRPC or REST endpoints behind a load balancer.
- Integrate with an online feature store: the model server calls the feature server (Redis/DynamoDB) to fetch precomputed velocity features using the card ID, combining them with the request payload before inference. Total feature retrieval + inference < 50 ms.
- Expose a `/v1/model/predict` endpoint that conforms to a versioned API contract, returning the fraud score, binary decision, and SHAP values for explainability (precomputed or using FastTreeSHAP).
- Implement canary deployments to gradually roll out new model versions, routing a small percentage of traffic to the canary and monitoring error rates and latency before full promotion.
- Circuit breaking and automatic fallback to a rule-based system if the model service fails to respond within 100 ms.

**Input:** Real-time transaction event from the payment gateway, containing raw transaction fields (`amount`, `mcc`, `terminal_type`, `timestamp`) and a card/user identifier.

**Output:** A structured response with `fraud_score` (0–1), `decision` (`APPROVE`/`DECLINE`/`REVIEW`), and `explanation` dict (top contributing feature names and Shapley values) for regulatory adverse action notices. Latency and response logged to a monitoring sink (Elasticsearch/Kafka).

---

## 4.10 Monitoring & Maintenance

### Baseline Approach

**Description:** A simple offline monitoring script that runs periodically on the laptop. It compares a recent batch of labeled transactions with the model's predictions to detect performance degradation.

**Responsibilities:**
- Load a CSV file of recent transactions with true fraud labels (simulating production feedback from `data/monitoring/new_transactions_labeled.csv`).
- Run the model on those transactions and compute the same evaluation metrics as in Model Evaluation (PR-AUC, false positive rate).
- Compare current metrics with baseline values from `reports/evaluation_report.json`.
- Print a warning to console if the false positive rate increases by >20% relative or PR-AUC drops by >10%.
- Save the report to disk as a timestamped log.

**Input:** `data/monitoring/new_transactions_labeled.csv` (features + ground truth labels), `artifacts/model.pkl`, `reports/evaluation_report.json` (baseline metrics).

**Output:** A timestamped `reports/monitoring_report_YYYYMMDD.json` with current metrics, baseline metrics, and drift flags (`fpr_drift_warning: true/false`).

### Advanced Approach

**Description:** A fully automated, continuous monitoring system that tracks data drift, concept drift, and model performance in production, triggering retraining pipelines autonomously when degradation is detected.

**Responsibilities:**
- Log all production predictions and outcomes (once fraud labels arrive) in a centralized monitoring database (e.g., Elasticsearch, ClickHouse). Include prediction timestamp, model version, feature values, score, and ground truth after the reporting delay period.
- Use Evidently AI or Alibi Detect to run scheduled jobs that compute:
  - **Data drift:** Population Stability Index (PSI) and Kolmogorov-Smirnov statistics for each feature between the training distribution and a recent window of production data.
  - **Prediction drift:** Distribution of predicted scores over time.
  - **Performance monitoring:** Once ground truth is available, compute rolling daily PR-AUC and false positive rate.
- Define alert rules: if PSI > 0.2 for any key feature, or daily PR-AUC drops below the threshold, send an alert to the fraud ops Slack channel and open an incident ticket.
- Implement an automated retraining trigger: if the model's false positive rate exceeds the business SLA for 3 consecutive days, a CI/CD pipeline kicks off a new training run using fresh labeled data and the latest feature definitions.
- Maintain a dashboard (Grafana) to visualize real-time and historical metrics: transaction volume, average fraud score, approval/decline rates, latency percentiles (p50, p95, p99).
- Use model registry capabilities to track which model version is in production and correlate with metrics; rollback automatically if the new model significantly degrades a key metric.

**Input:** Streaming inference logs (Kafka/Kinesis), delayed ground truth data feed (from chargeback/dispute settlement system), and training reference statistics stored in MLflow.

**Output:** Real-time Grafana dashboards, alert notifications (Slack/PagerDuty), automated retraining pipeline triggers, and a model health report artifact (`healthy`/`warning`/`degraded`) stored alongside the model version in the registry.

---

### 5. Technology Stack Comparison

| Component | Baseline | Advanced |
|---|---|---|
| Language | Python 3.9+ | Python 3.10+ / Java (Flink) |
| Data Manipulation | Pandas 1.5+ | Apache Spark 3.x, Dask |
| Validation | Pandera 0.16+ | Great Expectations 0.18+ |
| Visualization | Matplotlib, Seaborn | Sweetviz, Evidently AI, Grafana |
| ML Framework | Scikit-learn 1.3+ | XGBoost, LightGBM, PyTorch/TensorFlow |
| Imbalance Handling | SMOTE (imbalanced-learn) | SMOTE + scale_pos_weight + downsampling |
| Experiment Tracking | Manual (JSON logs) | MLflow 2.x |
| Hyperparameter Tuning | Manual 3-fold CV | Optuna / Hyperopt on Kubernetes |
| Model Serialization | Joblib (`.pkl`) | ONNX, Treelite, native XGBoost |
| Model Registry | None (local files) | MLflow Model Registry |
| Feature Store | None (in-script computation) | Feast / Tecton |
| Orchestration | Manual scripts | Airflow 2.x / Kubeflow Pipelines |
| Serving | Flask dev server | FastAPI on Kubernetes (KServe) |
| Online Store | N/A | Redis / DynamoDB |
| Message Queue | N/A | Apache Kafka / AWS Kinesis |
| Monitoring DB | N/A | Elasticsearch / ClickHouse |
| CI/CD | None | GitHub Actions + ArgoCD |
| Infrastructure | Local laptop | AWS/GCP/Azure (EKS/GKE/AKS) |

---

### 6. Directory Structure

```
fraud-detection/
├── config/
│   ├── pipeline_config.yaml            # Baseline: paths and parameters
│   └── training_config.yaml            # Advanced: hyperparameter search space
├── data/
│   ├── raw/
│   │   └── creditcard.csv              # Original dataset (gitignored)
│   ├── processed/
│   │   ├── cleaned.parquet
│   │   ├── features.parquet
│   │   ├── train.parquet
│   │   └── test.parquet
│   └── monitoring/
│       └── new_transactions_labeled.csv
├── artifacts/
│   ├── scaler.pkl
│   ├── feature_config.json
│   ├── model.pkl
│   ├── model_metadata.json
│   └── model.onnx                      # Advanced deployment artifact
├── reports/
│   ├── validation_report.json
│   ├── eda/
│   │   ├── class_distribution.png
│   │   ├── correlation_heatmap.png
│   │   └── fraud_by_hour.png
│   ├── evaluation_report.json
│   ├── confusion_matrix.csv
│   └── monitoring_report_YYYYMMDD.json
├── logs/
│   └── service.log
├── src/
│   ├── ingest.py
│   ├── validate.py
│   ├── preprocess.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── split.py
│   ├── train.py
│   ├── evaluate.py
│   ├── serve.py
│   └── monitor.py
├── notebooks/
│   └── eda_interactive.ipynb
├── docker/
│   ├── Dockerfile.train                 # Advanced: training container
│   └── Dockerfile.serve                 # Advanced: serving container
├── k8s/
│   ├── deployment.yaml                  # Advanced: K8s deployment manifest
│   └── hpa.yaml                         # Advanced: horizontal pod autoscaler
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

---

### 7. Data Contract (Schema)

The following schema is the contract between Feature Engineering and all downstream stages (Splitting, Training, Evaluation, Deployment, Monitoring).

| Feature | Type | Description |
|---|---|---|
| `Time` | float64 | Seconds elapsed since first transaction (dropped before training) |
| `V1` – `V28` | float64 | PCA-transformed anonymized features (already scaled) |
| `Amount` | float64 | Transaction amount in local currency |
| `Amount_log` | float64 | Natural log of `Amount + 1` |
| `hour` | int64 | Hour of transaction extracted from `Time` (0–23) |
| `day` | int64 | Day index relative to first transaction |
| `hour_sin` | float64 | Cyclical sine encoding of hour: `sin(2π × hour / 24)` |
| `hour_cos` | float64 | Cyclical cosine encoding of hour: `cos(2π × hour / 24)` |
| `txn_count_1h` | int64 | Global transaction count in prior 3,600 seconds |
| `txn_count_24h` | int64 | Global transaction count in prior 86,400 seconds |
| `avg_amount_1h` | float64 | Mean transaction amount in prior 3,600 seconds |
| `avg_amount_24h` | float64 | Mean transaction amount in prior 86,400 seconds |
| `Class` | int64 | Target label: 1 = fraudulent, 0 = legitimate |

---

### 8. Execution Flow

**Baseline — Sequential Manual Execution:**

```powershell
python src/ingest.py
python src/validate.py
python src/preprocess.py
python src/eda.py
python src/feature_engineering.py
python src/split.py
python src/train.py
python src/evaluate.py
```

Start the local prediction service:
```powershell
python src/serve.py
```

Test the endpoint:
```powershell
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"Time": 150000, "V1": -1.5, "V28": 0.3, "Amount": 120.0}'
```

Run periodic offline monitoring:
```powershell
python src/monitor.py
```

**Advanced — Automated DAG Triggers:**
- `data_ingestion_dag`: Runs hourly, ingests from Kafka/S3.
- `validation_trigger`: Fires on new data landing in raw zone.
- `training_dag`: Triggered by schedule (weekly) or monitoring alert.
- `evaluation_dag`: Runs after training, gates promotion.
- `deployment_pipeline`: CI/CD on model registry stage change.

---

### 9. Production Scaling Path

Each Baseline component maps directly to its Advanced counterpart with identical interface contracts.

| Baseline Component | Production Replacement |
|---|---|
| CSV file read via Pandas | Kafka consumer + S3 batch reads via Spark |
| Pandera assertions in script | Great Expectations suite in Airflow DAG |
| Pandas `.fillna()` and `StandardScaler` | Spark ML Pipeline / serverless preprocessing container |
| Jupyter notebook EDA | Automated Sweetviz/Evidently scheduled reports |
| Pandas `.rolling()` velocity features | Flink/Spark Streaming windowed aggregations in Feast |
| Scikit-learn Random Forest CV | XGBoost/LightGBM with Optuna on GPU-enabled Kubernetes |
| Joblib serialization to `.pkl` | ONNX format + MLflow Model Registry |
| Flask localhost dev server | FastAPI Docker container on K8s with online feature store fetch |
| Offline monitor script | Evidently AI + Grafana dashboards + automated retraining triggers |