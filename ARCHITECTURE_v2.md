# ARCHITECTURE.md

## Project: Real-Time Credit Card Fraud Detection — Full Architecture

### Dataset Profile (Discovered via EDA)

| Property | Value |
|---|---|
| Total Transactions | 284,807 |
| Legitimate Transactions | 284,315 |
| Fraudulent Transactions | 492 |
| Fraud Ratio | 0.1727% |
| Time Range | 0 to 172,792 seconds (~48 hours) |
| Amount Range | $0.00 to $25,691.16 |
| Amount Median | $22.00 |
| Amount Mean | $88.35 |
| PCA Features | V1–V28 (anonymized, mean ≈ 0, varying std) |
| Top Fraud-Correlated Features | V17 (-0.33), V14 (-0.30), V12 (-0.26), V10 (-0.22), V16 (-0.20) |
| Null Values | None detected |
| Duplicates | To be verified during validation |

---

### 1. System Overview

This document defines the complete architecture for an end-to-end machine learning pipeline that detects fraudulent credit card transactions at the point of authorization. It covers both a **Baseline** implementation for local development on an Intel Core i5 laptop and an **Advanced** production implementation. The dataset used is the ULB Credit Card Fraud Detection dataset with 284,807 transactions, 31 columns (Time, V1–V28, Amount, Class), and an extreme class imbalance of 0.17% fraud.

---

### 2. Pipeline Architecture (Textual Diagram)

```
 ┌──────────────┐
 │  Raw CSV     │  creditcard.csv (284,807 rows × 31 columns)
 └──────┬───────┘
        │
        ▼
 ┌──────────────────────┐
 │  1. DATA INGESTION   │  Pandas read_csv → df_raw
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  2. DATA VALIDATION  │  Schema, nulls, duplicates, Amount ≥ 0, Class ∈ {0,1}
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ 3. DATA PREPROCESSING│  StandardScaler on Amount & Time, log transform
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │     4. EDA           │  Class distribution, correlations, time patterns
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ 5. FEATURE ENGINEER. │  Velocity features (1h, 24h windows), cyclical time
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  6. DATA SPLITTING   │  Time-aware 80/20 split (first 80% time → train)
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  7. MODEL TRAINING   │  SMOTE + Logistic Regression, Decision Tree, Random Forest
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ 8. MODEL EVALUATION  │  PR-AUC primary metric, threshold at 80% recall
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  9. MODEL DEPLOYMENT │  Flask API → localhost:5000/predict
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────────────┐
 │ 10. MONITORING & MAINTENANCE │  Offline drift checks, metric comparison
 └──────────────────────────────┘
```

---

### 3. Component Specifications

---

## 4.1 Data Ingestion

### Baseline Approach

**Description:** A minimal file-reader module that loads the raw ULB Credit Card Fraud Detection CSV file into a Pandas DataFrame. The dataset contains 284,807 transactions with 31 columns: `Time` (seconds elapsed), `V1`–`V28` (PCA-transformed features), `Amount` (transaction amount in local currency), and `Class` (0 = legitimate, 1 = fraud).

**Responsibilities:**
- Read the CSV file path from `config/pipeline_config.yaml`.
- Perform a quick sanity check: confirm 284,807 rows and 31 columns loaded.
- Report memory usage and column schema to console.
- Return a DataFrame object for downstream steps.

**Input:** A file path string (`data/raw/creditcard.csv`). The CSV contains no header issues; all 31 columns are numeric (float64 for V1–V28, Amount, Time; int64 for Class).

**Output:** `df_raw` — a Pandas DataFrame (284,807 rows × 31 columns). Schema: `Time` (float64), `V1`–`V28` (float64), `Amount` (float64), `Class` (int64).

### Advanced Approach

**Description:** A scalable ingestion service that supports both batch historical loads and real-time streaming from payment gateways. Abstracts the data source, allowing seamless switching between a data lake and a message queue while ensuring exactly-once semantics with transaction IDs.

**Responsibilities:**
- Batch ingestion from cloud object storage (S3/GCS) using Apache Spark or Dask for parallel reads of partitioned Parquet files.
- Real-time ingestion from Kafka topics (`transactions.raw`) with Avro schema validation, using a schema registry.
- Idempotent record handling using transaction IDs to avoid duplicates during reprocessing.
- Publish raw events to a staging area (data lake raw zone) and route failures to a dead-letter queue.
- Automatically register data schema and version in a metadata store (Hive Metastore or AWS Glue Catalog).

**Input:** Source identifiers: an S3 prefix pattern (`s3://fraud-lake/raw/dt=YYYY-MM-DD/`) or a Kafka bootstrap server and topic. Schema definition from a central schema registry (Apicurio, Confluent).

**Output:** A distributed DataFrame (Spark/Dask) for batch processing, or a structured streaming DataFrame for real-time. Events persisted to a raw zone in Parquet format partitioned by `dt=YYYY-MM-DD`.

---

## 4.2 Data Validation

### Baseline Approach

**Description:** A lightweight validation script using `pandera` that checks basic expectations against the raw DataFrame. Ensures the dataset matches the known ULB schema before any processing occurs.

**Responsibilities:**
- Verify all 31 columns exist: `Time`, `V1`–`V28`, `Amount`, `Class`.
- Check data types: all columns are numeric (float64 or int64).
- Validate `Amount` >= 0 (no negative transaction amounts allowed).
- Validate `Class` is strictly 0 or 1.
- Check for null values in all columns (expected: 0 nulls).
- Check for duplicate rows and report count.
- Halt pipeline with `ValidationError` if critical constraints fail.

**Input:** The raw DataFrame from Data Ingestion (`df_raw`).

**Output:** `reports/validation_report.json` — pass/fail status per expectation including duplicate count and null summary. The unchanged DataFrame if validation passes.

### Advanced Approach

**Description:** A full data contract enforcement layer using Great Expectations (GE) integrated into CI/CD pipelines. Validation runs automatically on every data landing event, with results stored for auditing and drift detection.

**Responsibilities:**
- Define a comprehensive expectation suite: column existence, types, value ranges (Amount 0–25,000 based on observed max of 25,691), null proportions = 0%, distribution of `Class` (fraud ratio ~0.15–0.20%).
- Run validation as a pre-commit hook in the feature pipeline and as a step in an orchestrated DAG (Airflow/Prefect).
- Publish validation results to a data quality dashboard (GE Data Docs on S3) and send alerts (Slack/PagerDuty) on failure.
- Perform time-window comparisons: Kolmogorov-Smirnov test on `Amount` distribution and PCA feature distributions vs. reference window to catch concept drift.
- Store expectation suites and checkpoint results in a versioned object store.

**Input:** DataFrame from batch/streaming Data Ingestion, plus a GE expectation suite JSON and checkpoint configuration.

**Output:** A validation result document (JSON) with per-expectation outcomes, success/failure flags, and distribution diagnostic plots. Downstream steps only proceed if critical validations pass.

---

## 4.3 Data Cleaning / Preprocessing

### Baseline Approach

**Description:** A straightforward script that scales numerical features and prepares data for modeling. Since the ULB dataset is pre-cleaned (no nulls, PCA already applied), the focus is on scaling `Time` and `Amount` to match the PCA features' scale.

**Responsibilities:**
- No null imputation needed (dataset confirmed clean during validation).
- Remove duplicate rows if any were detected during validation.
- Apply `StandardScaler` to `Amount` and `Time` features (fit on training data only, but for baseline we scale the full dataset for exploration).
- PCA features (`V1`–`V28`) are already centered and scaled (mean ≈ 0, std ≈ 1); no additional scaling required.
- Apply `RobustScaler` as an alternative to `Amount` given its extreme skew (mean 88, median 22, max 25,691) to reduce outlier influence.
- Save the fitted scaler for inference.

**Input:** Validated DataFrame.

**Output:** `data/processed/cleaned.parquet` — DataFrame with scaled `Amount` and `Time`, original PCA features. `artifacts/scaler.pkl` — fitted scaler object.

### Advanced Approach

**Description:** A modular preprocessing pipeline that runs as a series of stateless transformations in a distributed framework, supporting point-in-time correctness for both batch training and real-time serving.

**Responsibilities:**
- Use Apache Spark ML Pipelines or a containerized Scikit-learn pipeline for portability across training and serving.
- Apply `Amount` scaling using statistics computed only on the training set (precomputed and stored in a configuration store).
- `Amount` log transformation: `Amount_log = log(Amount + 1)` to handle the extreme right skew (median $22 vs max $25,691).
- Handle any new categorical features (MCC, terminal type) using one-hot or target encoding with smoothing, fit on a holdout to prevent leakage.
- Log every transformation step with parameters in MLflow for lineage tracking.
- Output transformation artifacts (scaler, encoder) as versioned objects in MLflow Model Registry.

**Input:** Validated DataFrame (batch from Spark/Dask), plus configuration referencing train/validation split boundary.

**Output:** Transformed dataset in Parquet partitioned by date, and a serialized preprocessing pipeline in MLflow with versioning. The identical pipeline is applied in the model server during inference.

---

## 4.4 Exploratory Data Analysis (EDA)

### Baseline Approach

**Description:** A script that generates descriptive statistics and visualizations to understand the ULB dataset's characteristics. Based on actual EDA findings: 284,807 transactions, 492 fraud cases (0.17%), top fraud-correlated features identified as V17, V14, V12, V10, V16.

**Responsibilities:**
- Print class distribution: 284,315 legitimate (99.83%), 492 fraud (0.17%).
- Compute summary statistics: `Amount` (mean $88.35, median $22.00, max $25,691.16), `Time` (range 0–172,792 seconds ≈ 48 hours).
- Plot histogram of `Amount` by class (fraud vs. non-fraud) — note extreme skew and differing distributions.
- Plot correlation heatmap focused on V1–V28 with `Class`, highlighting top features: V17 (-0.33), V14 (-0.30), V12 (-0.26), V10 (-0.22).
- Plot fraud frequency by hour of day derived from `Time` (48-hour window allows 2 daily cycles).
- Display top 15 features ranked by absolute correlation with `Class`.
- Run PCA variance analysis to confirm dimensionality reduction potential.

**Input:** `data/processed/cleaned.parquet` from Data Preprocessing.

**Output:** PNG figures saved to `reports/eda/` (class_distribution.png, amount_histogram.png, correlation_heatmap.png, fraud_by_hour.png, feature_correlation_ranking.csv). Key findings logged to console.

### Advanced Approach

**Description:** Automated, scheduled EDA producing comprehensive data profiles and drift reports. Compares current data distributions against the training reference to detect shifts.

**Responsibilities:**
- Run `Sweetviz` or `pandas_profiling` on a daily schedule over the recent batch of transactions.
- Compare current distributions with reference profile from training window (281K legit, 492 fraud, fraud ratio 0.17%).
- Flag if fraud ratio deviates by >20% from baseline (outside 0.14–0.21% range).
- Monitor top fraud-correlated features (V17, V14, V12, V10) for distribution drift using PSI.
- Generate interactive HTML reports uploaded to cloud storage or Databricks dashboard.
- Trigger alerts if unexpected patterns emerge (e.g., new clusters in PCA space, Amount distribution shift).

**Input:** Batch DataFrame from cleaned data store, plus reference profile JSON from initial training run.

**Output:** Timestamped EDA report (HTML/JSON) in cloud storage. Key metrics (fraud rate, missing percentages, drift p-values per feature) logged to MLflow.

---

## 4.5 Feature Engineering

### Baseline Approach

**Description:** A Pandas-based feature transformer that creates temporal and velocity features from the `Time` column. Since the ULB dataset lacks user/card identifiers, all velocity features are computed globally over rolling time windows.

**Responsibilities:**
- Convert `Time` (seconds) into `hour` (0–47 over 48-hour window) and `day` (0 or 1).
- Create cyclical time features: `hour_sin = sin(2π × hour / 24)`, `hour_cos = cos(2π × hour / 24)`.
- Sort by `Time`, then compute global rolling-window features using Pandas `.rolling()`:
  - `txn_count_1h`: count of transactions in the prior 3,600 seconds.
  - `txn_count_24h`: count of transactions in the prior 86,400 seconds.
  - `avg_amount_1h`: mean transaction amount in prior 3,600 seconds.
  - `avg_amount_24h`: mean transaction amount in prior 86,400 seconds.
  - `std_amount_1h`: standard deviation of amounts in prior 3,600 seconds.
- Apply log transform: `Amount_log = log(Amount + 1)`.
- Keep the top 10 fraud-correlated PCA features highlighted in EDA (V17, V14, V12, V10, V16, V3, V7, V11, V4, V18) and drop low-correlation features if dimensionality reduction is desired.
- Drop original `Time` column before training.

**Input:** `data/processed/cleaned.parquet` with `Time` column still present.

**Output:** `data/processed/features.parquet` — DataFrame with original features plus engineered features. `artifacts/feature_config.json` — list of final feature column names used for training.

### Advanced Approach

**Description:** A Feature Store-based pipeline (Feast/Tecton) that decouples feature computation from model training. Velocity features are precomputed at scale using stream processing and stored for low-latency online serving.

**Responsibilities:**
- Maintain an offline feature store where all features are registered and versioned.
- Compute velocity features using Spark SQL or Flink: per-card-ID aggregations over sliding windows of 1h, 6h, 24h, 7d — counts, mean/std/max amounts, transaction frequency, time since last transaction.
- Store features with event timestamps for point-in-time correct training (no look-ahead bias).
- For real-time inference, serve precomputed features from an online store (Redis/DynamoDB) and compute stream features (current amount deviation from 1h average) in the model server.
- Advanced features: interaction terms between top PCA features (V17×V14, V12×V10), Isolation Forest anomaly scores as a feature, recency-frequency metrics.
- Automate feature validation (min/max values, non-null checks) in CI/CD before promotion.

**Input:** Streaming transaction events (Kafka) with card/user identifiers, plus batch historical tables. Feature registry YAML defining feature views and entities.

**Output:** Offline feature dataset (Parquet) for training via historical retrieval API. Online feature vectors from feature server at serving time (<10 ms). Feature engineering code versioned as a Feast Feature Repository.

---

## 4.6 Data Splitting

### Baseline Approach

**Description:** A time-aware chronological split that divides the 48-hour dataset into training and test sets. The first 80% of the time range (0–138,233 seconds) becomes training; the remaining 20% (138,234–172,792 seconds) becomes the holdout test set.

**Responsibilities:**
- Sort transactions by `Time` ascending.
- Calculate split timestamp: 80% of max `Time` value = 0.80 × 172,792 = 138,233 seconds.
- Training set: all transactions where `Time` ≤ 138,233.
- Test set: all transactions where `Time` > 138,233.
- Verify no temporal leakage: max training `Time` < min test `Time`.
- Report class distribution in each split to confirm fraud ratio is preserved (~0.17% in both).

**Input:** `data/processed/features.parquet` with `Time` column for splitting logic.

**Output:** `data/processed/train.parquet` and `data/processed/test.parquet`. `config/split_timestamp.txt` recording the boundary value (138,233 seconds).

### Advanced Approach

**Description:** A robust, automated time-series cross-validation strategy with delay-aware splits that account for fraud reporting lag in production.

**Responsibilities:**
- Perform expanding-window cross-validation with 3 folds over the time series.
- Each fold: train on [0, T], validate on [T, T+delta] where delta represents a deployment cycle (e.g., 6 hours given the 48-hour dataset).
- Implement delay-aware split: exclude the most recent 3,600 seconds (1 hour) from validation to simulate fraud label reporting delay.
- Use stratified sampling within validation windows to maintain the 0.17% fraud prevalence.
- Save split metadata (timestamps, fold assignments) as versioned artifacts for reproducible evaluation.
- Integrate with feature store for point-in-time correct feature retrieval per fold.

**Input:** Feature store historical feature set with timestamps, plus configuration YAML specifying fold durations and reporting delay.

**Output:** Per-fold training and validation DataFrames in Parquet, plus a holdout test set. Timestamp boundary map logged to MLflow.

---

## 4.7 Model Training

### Baseline Approach

**Description:** A single Python script training three baseline classifiers with Scikit-learn. Handles the 0.17% fraud ratio using SMOTE oversampling during cross-validation. Uses 3-fold cross-validation on the 80% training split for model selection.

**Responsibilities:**
- Load `train.parquet`; separate `X_train` (all features except `Class`, `Time`) and `y_train`.
- Apply SMOTE (imbalanced-learn) to balance classes in each training fold (synthesizing fraud examples to match legitimate count).
- Train three candidates with 3-fold cross-validation:
  - Logistic Regression (baseline linear, high interpretability).
  - Decision Tree (max_depth=5, captures simple non-linear rules).
  - Random Forest (n_estimators=50, max_depth=10, robust ensemble).
- Compare mean PR-AUC across folds (primary metric due to 0.17% imbalance).
- Retrain best model on full training set.
- Serialize with joblib.

**Input:** `data/processed/train.parquet` (~227,845 rows, 80% of 284,807).

**Output:** `artifacts/model.pkl` — Scikit-learn model. `artifacts/model_metadata.json` — selected algorithm, hyperparameters, feature list, CV PR-AUC score.

### Advanced Approach

**Description:** A fully orchestrated training pipeline with experiment tracking, distributed hyperparameter tuning, and model registry integration. Trains multiple model types optimized for low-latency inference.

**Responsibilities:**
- Use Airflow/Kubeflow Pipelines triggering training on new labeled data or schedule.
- Load features from offline feature store using point-in-time joins.
- Handle extreme class imbalance (0.17%) via combined strategy: SMOTE + cost-sensitive learning (`scale_pos_weight` ~578 in XGBoost = 284,315/492) + probability calibration.
- Distributed hyperparameter optimization with Optuna across Kubernetes, exploring XGBoost, LightGBM, and a small MLP.
- Train Isolation Forest on legitimate transactions to produce anomaly scores as supplementary features.
- Track all experiments in MLflow: parameters, PR-AUC, feature importance, model artifacts.
- Register best model in MLflow Model Registry → "Staging" after passing evaluation gates.
- Export to low-latency format: ONNX or Treelite-compiled XGBoost for microsecond inference.

**Input:** Feature view from offline feature store, training configuration YAML (hyperparameter search space, objective = PR-AUC).

**Output:** Registered model version in MLflow with ONNX artifact, complete training logs, feature importance plot, model card with performance and fairness metrics.

---

## 4.8 Model Evaluation

### Baseline Approach

**Description:** A script evaluating the trained model on the held-out 20% test set (~56,962 transactions). Prioritizes PR-AUC due to the 0.17% class imbalance and determines an optimal decision threshold.

**Responsibilities:**
- Load `test.parquet` and `model.pkl`.
- Generate fraud probability scores for all test transactions.
- Compute PR-AUC (primary metric), ROC-AUC (secondary reference).
- Compute precision, recall, F1 at default threshold 0.5.
- Find threshold achieving 80% recall (captures 80% of fraud) and report precision and false positive rate at that threshold.
- Find threshold achieving 1% false positive rate and report recall.
- Generate confusion matrix at selected deployment threshold.
- Save all metrics and the recommended threshold.

**Input:** `data/processed/test.parquet` (~56,962 rows), `artifacts/model.pkl`.

**Output:** `reports/evaluation_report.json` (PR-AUC, ROC-AUC, precision@80recall, FPR@80recall, threshold@80recall, threshold@1fpr). `reports/confusion_matrix.csv`.

### Advanced Approach

**Description:** A multi-faceted evaluation harness simulating production conditions with business impact metrics and slice-level performance validation.

**Responsibilities:**
- Time-series backtesting: aggregate metrics across all cross-validation folds.
- Compute business metrics: fraud loss saved (fraud detected × $88.35 avg amount), customer insult rate (false positive %), estimated operational cost of review queue.
- Slice evaluation: performance by Amount buckets ($0–22, $22–100, $100–500, $500+), hour of day, day 0 vs day 1.
- Shadow deployment: log challenger model predictions on live traffic, compare to champion offline; promote only if PR-AUC improves ≥2%.
- Generate HTML validation report with drift insights (PSI per feature, prediction distribution shift).
- Gating function: block promotion if PR-AUC drops >5% vs champion or false positive rate exceeds 2%.
- Store evaluation artifacts in MLflow, update registry stage (Staging → Production or Archived).

**Input:** Model artifact, holdout test set, optional shadow production logs.

**Output:** Comprehensive evaluation report (JSON/HTML) with slice metrics, business KPIs, `promote: true/false` flag, updated model registry stage.

---

## 4.9 Model Deployment

### Baseline Approach

**Description:** A Flask REST API serving fraud predictions locally on `localhost:5000`. Loads the serialized model and scaler, applies identical preprocessing to incoming requests, and returns fraud probabilities.

**Responsibilities:**
- Load `model.pkl` and `scaler.pkl` into memory on startup.
- Expose POST `/predict` endpoint accepting JSON transactions.
- Apply preprocessing: scale `Amount` using loaded scaler, compute `Amount_log`, `hour_sin/cos`, velocity features from request context.
- Return `{"fraud_probability": 0.92, "is_fraud": true/false}` using the threshold selected during evaluation.
- Log every request to `logs/service.log` with timestamp, input hash, and prediction.

**Input:** POST `http://localhost:5000/predict` — JSON body: `{"Time": 150000, "V1": -1.5, ..., "V28": 0.3, "Amount": 120.0}`.

**Output:** JSON response: `{"fraud_probability": 0.92, "is_fraud": true}`. Logs appended to `logs/service.log`.

### Advanced Approach

**Description:** Containerized, cloud-native deployment on Kubernetes serving predictions with sub-50ms latency, integrated into the payment authorization gateway.

**Responsibilities:**
- Package ONNX model + FastAPI serving runtime in Docker image.
- Pull production model version automatically from MLflow Model Registry via CI/CD (ArgoCD).
- Deploy on Kubernetes with horizontal pod autoscaling (target CPU 70%, min 2 pods, max 50).
- Integrate online feature store: fetch precomputed velocity features from Redis using card ID (<10 ms), combine with request payload, run inference (<40 ms).
- Expose `/v1/model/predict` with versioned API contract; return fraud score, binary decision, and SHAP explanations for regulatory compliance.
- Canary deployments: 5% → 25% → 100% traffic, monitor error rate and latency at each stage.
- Circuit breaker: if model service p99 latency exceeds 100ms, fallback to rule-based system.

**Input:** Real-time transaction from payment gateway with card ID, Amount, MCC, terminal type, timestamp.

**Output:** `fraud_score` (0–1), `decision` (APPROVE/DECLINE/REVIEW), `explanation` (top 5 features with Shapley values). Latency logged to monitoring sink.

---

## 4.10 Monitoring & Maintenance

### Baseline Approach

**Description:** An offline monitoring script that periodically loads a new batch of labeled transactions and compares model performance against the evaluation baseline.

**Responsibilities:**
- Load `data/monitoring/new_transactions_labeled.csv` (simulated production feedback with ground truth).
- Run model inference and compute PR-AUC, false positive rate, recall.
- Compare against baseline metrics from `reports/evaluation_report.json`.
- Flag warning if false positive rate increases >20% relative or PR-AUC drops >10%.
- Save timestamped monitoring report.

**Input:** `data/monitoring/new_transactions_labeled.csv`, `artifacts/model.pkl`, `reports/evaluation_report.json`.

**Output:** `reports/monitoring_report_YYYYMMDD.json` with current metrics, baseline metrics, drift flags.

### Advanced Approach

**Description:** Fully automated continuous monitoring tracking data drift, concept drift, and model performance. Triggers automatic retraining when degradation is detected.

**Responsibilities:**
- Log all predictions and delayed ground truth (chargeback data) to ClickHouse/Elasticsearch.
- Scheduled Evidently AI jobs computing: data drift (PSI/KS per feature vs training reference), prediction drift (score distribution shift), performance (rolling daily PR-AUC, FPR).
- Alert rules: PSI > 0.2 on V17/V14/V12, daily PR-AUC below threshold → Slack + incident ticket.
- Automated retraining: if FPR exceeds SLA for 3 consecutive days → CI/CD triggers full pipeline rerun.
- Grafana dashboards: transaction volume, average fraud score, approval/decline/review rates, latency p50/p95/p99.
- Model version tracking in registry with automatic rollback if new version degrades key metrics.

**Input:** Streaming inference logs, delayed ground truth feed, training reference statistics.

**Output:** Real-time dashboards, alert notifications, automated retraining triggers, model health report (healthy/warning/degraded) in registry.

---

### 5. Data Contract (Schema)

| Feature | Type | Description |
|---|---|---|
| `Time` | float64 | Seconds elapsed since first transaction (0–172,792). Dropped before training. |
| `V1`–`V28` | float64 | PCA-transformed anonymized features (mean ≈ 0, std ≈ 1) |
| `Amount` | float64 | Transaction amount ($0.00–$25,691.16, median $22.00) |
| `Amount_log` | float64 | Natural log of `Amount + 1` (reduces right-skew) |
| `hour` | int64 | Hour extracted from `Time` (0–47 over ~48 hours) |
| `hour_sin` | float64 | Cyclical sine encoding: `sin(2π × hour / 24)` |
| `hour_cos` | float64 | Cyclical cosine encoding: `cos(2π × hour / 24)` |
| `txn_count_1h` | int64 | Global transaction count in prior 3,600 seconds |
| `txn_count_24h` | int64 | Global transaction count in prior 86,400 seconds |
| `avg_amount_1h` | float64 | Mean transaction amount in prior 3,600 seconds |
| `avg_amount_24h` | float64 | Mean transaction amount in prior 86,400 seconds |
| `std_amount_1h` | float64 | Standard deviation of amounts in prior 3,600 seconds |
| `Class` | int64 | Target: 1 = fraud (492 cases, 0.17%), 0 = legitimate |

---

### 6. Directory Structure

```
fraud-detection-real-time/
├── config/
│   ├── pipeline_config.yaml
│   └── split_timestamp.txt
├── data/
│   ├── raw/
│   │   └── creditcard.csv              # 284,807 rows × 31 columns
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
│   └── model_metadata.json
├── reports/
│   ├── validation_report.json
│   ├── eda/
│   │   ├── class_distribution.png
│   │   ├── amount_histogram.png
│   │   ├── correlation_heatmap.png
│   │   ├── fraud_by_hour.png
│   │   └── feature_correlation_ranking.csv
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
├── requirements.txt
├── .gitignore
├── ARCHITECTURE.md
└── README.md
```

---

### 7. Execution Flow (Baseline)

```powershell
python src/ingest.py          # Load CSV, confirm 284,807 rows
python src/validate.py        # Schema check, nulls, duplicates
python src/preprocess.py      # Scale Amount & Time, save scaler
python src/eda.py             # Generate plots and statistics
python src/feature_engineering.py  # Velocity features, cyclical time
python src/split.py           # 80/20 time-aware split
python src/train.py           # SMOTE + Logistic Regression, Decision Tree, Random Forest
python src/evaluate.py        # PR-AUC, threshold tuning
python src/serve.py           # Start Flask API on localhost:5000
```