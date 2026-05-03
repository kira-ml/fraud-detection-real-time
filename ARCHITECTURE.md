# ARCHITECTURE.md

## Project: Real-Time Credit Card Fraud Detection – Baseline Architecture

### 1. System Overview
This architecture implements an end-to-end machine learning pipeline for credit card fraud detection. It is designed to run entirely on a single Intel Core i5 laptop with minimal dependencies, using the ULB Credit Card Fraud Detection dataset. The pipeline is modular, with each stage producing versioned artifacts that flow into the next. The primary objective is to establish a repeatable, testable baseline that can be refactored for a distributed production environment.

### 2. Design Principles
- **Simplicity First:** Use pure Python, Pandas, and Scikit-learn exclusively. No distributed computing or containers required.
- **Modularity:** Each pipeline stage is an independent script that reads from disk and writes to disk. Stages can be developed and debugged in isolation.
- **Resource Awareness:** All operations fit comfortably in 8–16 GB of RAM. Rolling window features are computed using Pandas groupby operations, not streaming frameworks.
- **Observability:** Every stage logs its inputs, outputs, and key metrics to the console and a structured JSON file for traceability.
- **Production Alignment:** Data contracts and stage interfaces mirror what a production system expects, even if the underlying technology differs.

---

### 3. System Architecture Diagram (Textual)

```
[data/raw/creditcard.csv]
          |
          v
+---------------------+
|  1. Data Ingestion  |  Read CSV, return DataFrame
+---------------------+
          |
          | df_raw (in-memory passed)
          v
+---------------------+
|  2. Data Validation |  Schema & content checks
+---------------------+
          |
          | reports/validation_report.json
          v
+--------------------------+
| 3. Data Preprocessing    |  Impute, scale, save scaler
+--------------------------+
          |
          | data/processed/cleaned.parquet
          | artifacts/scaler.pkl
          v
+--------------------------+
| 4. Exploratory Data      |  Descriptive stats, plots
|    Analysis (EDA)        |
+--------------------------+
          |
          | reports/eda/ (figures, tables)
          v
+----------------------------+
| 5. Feature Engineering     |  Velocity features, cyclical time
+----------------------------+
          |
          | data/processed/features.parquet
          | artifacts/feature_config.json
          v
+----------------------+
| 6. Data Splitting    |  Time-aware train/test split
+----------------------+
          |
          | data/processed/train.parquet
          | data/processed/test.parquet
          v
+----------------------+
| 7. Model Training    |  SMOTE + Scikit-learn classifiers
+----------------------+
          |
          | artifacts/model.pkl
          | artifacts/model_metadata.json
          v
+-----------------------+
| 8. Model Evaluation   |  PR-AUC, threshold tuning
+-----------------------+
          |
          | reports/evaluation_report.json
          v
+-----------------------+
| 9. Model Deployment   |  Local Flask REST API
+-----------------------+
          |
          | (running on localhost:5000)
          v
+-----------------------------+
| 10. Monitoring &           |
|     Maintenance             |  Offline drift comparison
+-----------------------------+
          |
          | reports/monitoring_report.json
```

---

### 4. Component Specifications

#### 4.1 Data Ingestion

**Description:** Reads the raw CSV dataset from local disk. This is the sole entry point; all source connection details are encapsulated here.

**Responsibilities:**
- Load CSV into a Pandas DataFrame.
- Report row count, column count, and memory usage.
- Handle file-not-found errors gracefully.

**Input:** A configuration file `config/pipeline_config.yaml` specifying `data_path: "data/raw/creditcard.csv"`.

**Output:** An in-memory Pandas DataFrame (`df_raw`). Console log: `"Loaded 284807 rows, 31 columns. Memory: 67.3 MB."`

#### 4.2 Data Validation

**Description:** A script that asserts basic schema and value constraints against the raw DataFrame. It uses `pandera` to define a lightweight contract.

**Responsibilities:**
- Verify columns: `Time`, `V1`–`V28`, `Amount`, `Class` exist.
- Check `Amount` >= 0.
- Check `Class` is in {0, 1}.
- Report null count per column.
- Raise `ValueError` on critical failures.

**Input:** The raw DataFrame from 4.1.

**Output:** A `reports/validation_report.json` with a list of expectations and their pass/fail status. Passes the DataFrame forward if no critical failures.

#### 4.3 Data Preprocessing

**Description:** Cleans and standardizes numerical features. Handles missing values and duplicates.

**Responsibilities:**
- Impute nulls: `Amount` with median if needed, PCA features with 0.
- Drop duplicate rows.
- Apply `StandardScaler` to `Amount` and `Time` columns (fit only, not applied yet to all features to preserve PCA structure).
- Save the fitted scaler.

**Input:** Validated DataFrame.

**Output:** `data/processed/cleaned.parquet` (cleaned DataFrame) and `artifacts/scaler.pkl`.

#### 4.4 Exploratory Data Analysis (EDA)

**Description:** Generates summary statistics and visualizations to understand data distributions and class imbalance. Runs as a separate notebook or script for interactive analysis.

**Responsibilities:**
- Print class distribution: count and percentage of fraud (`Class=1`).
- Plot histogram of `Amount` by class.
- Display correlation heatmap of `V1`–`V28`.
- Plot fraud frequency by hour (derived from `Time`).
- Save all figures to `reports/eda/`.

**Input:** `data/processed/cleaned.parquet`.

**Output:** PNG figures and a summary HTML/CSV table in `reports/eda/`.

#### 4.5 Feature Engineering

**Description:** Creates temporally aware features based on the `Time` column, which represents seconds elapsed since the first transaction.

**Responsibilities:**
- Convert `Time` (seconds) to `hour` (integer 0–23) and `day` (integer).
- Create cyclical time features: `hour_sin = sin(2*pi*hour/24)`, `hour_cos`.
- Sort by `Time`, then compute global rolling-window features using Pandas `rolling`:
  - `txn_count_1h`: count of transactions in the last 3600 seconds.
  - `txn_amount_avg_1h`: average amount in the last 3600 seconds.
- Log-transform `Amount` to `Amount_log`.
- Drop original `Time` column before training.

**Input:** `data/processed/cleaned.parquet`.

**Output:** `data/processed/features.parquet` and `artifacts/feature_config.json` (list of final feature column names).

#### 4.6 Data Splitting

**Description:** Splits the feature set chronologically to prevent temporal data leakage, simulating how the model would be deployed.

**Responsibilities:**
- Sort data by the original `Time` column before splitting.
- Split at the timestamp that corresponds to the first 80% of the total time range.
- Training set: transactions with `Time` <= split_timestamp.
- Test set: later transactions.
- Save both sets.

**Input:** `data/processed/features.parquet` (still containing `Time` for split logic).

**Output:** `data/processed/train.parquet`, `data/processed/test.parquet`, and `config/split_timestamp.txt`.

#### 4.7 Model Training

**Description:** Trains multiple baseline classifiers using Scikit-learn, addressing class imbalance with SMOTE oversampling on the training set.

**Responsibilities:**
- Load training data; separate `X_train` (drop `Class`, `Time`) and `y_train`.
- Apply SMOTE to balance the minority class in the training folds.
- Perform 3-fold cross-validation for each candidate: Logistic Regression, Decision Tree (max_depth=5), Random Forest (n_estimators=50).
- Compare mean PR-AUC across folds.
- Retrain the best model on the full training set.
- Save the model and metadata.

**Input:** `data/processed/train.parquet`.

**Output:** `artifacts/model.pkl` and `artifacts/model_metadata.json` (model type, hyperparameters, feature list, CV PR-AUC).

#### 4.8 Model Evaluation

**Description:** Calculates final performance metrics on the held-out test set, with emphasis on the Precision-Recall curve due to extreme class imbalance.

**Responsibilities:**
- Load test data and trained model.
- Predict probabilities on the test set.
- Compute PR-AUC and ROC-AUC.
- Find the decision threshold that achieves 80% recall and report the corresponding precision and false positive rate.
- Generate and save a confusion matrix at that threshold.

**Input:** `data/processed/test.parquet`, `artifacts/model.pkl`.

**Output:** `reports/evaluation_report.json` (PR-AUC, ROC-AUC, precision@80recall, FPR@80recall, optimal_threshold).

#### 4.9 Model Deployment

**Description:** A lightweight Flask web service that loads the trained model and serves predictions on single JSON requests, simulating a real-time authorization endpoint.

**Responsibilities:**
- On startup, load `model.pkl` and `scaler.pkl`.
- Expose POST `/predict` endpoint.
- Accept JSON with raw feature values (matching training schema).
- Apply preprocessing and feature engineering steps identical to training (scale `Amount`, compute cyclical time, etc.).
- Return `{"fraud_probability": 0.93, "is_fraud": true/false}` using the tuned threshold.
- Log request and response to `logs/service.log`.

**Input:** HTTP POST to `localhost:5000/predict` with JSON body.

**Output:** JSON prediction response. Log entry appended.

#### 4.10 Monitoring & Maintenance

**Description:** An offline script that runs periodically. It loads a newly acquired batch of labeled transactions (simulating production feedback) and compares model performance against the baseline.

**Responsibilities:**
- Load a newer CSV file (e.g., `data/monitoring/new_transactions_labeled.csv`).
- Run prediction and compute the same metrics as in 4.8.
- Compare current PR-AUC and false positive rate with the baseline from `reports/evaluation_report.json`.
- Log a warning to console if false positive rate increases by >20% relative, or PR-AUC drops by >10%.

**Input:** `data/monitoring/new_transactions_labeled.csv`, `artifacts/model.pkl`, `reports/evaluation_report.json`.

**Output:** A timestamped `reports/monitoring_report_YYYYMMDD.json` with current metrics and drift flags.

---

### 5. Technology Stack

| Component | Baseline Technology |
|---|---|
| Language | Python 3.9+ |
| Data Manipulation | Pandas 1.5+ |
| Numerical Operations | NumPy 1.24+ |
| Visualization | Matplotlib 3.7+, Seaborn 0.12+ |
| Validation | Pandera 0.16+ |
| ML Framework | Scikit-learn 1.3+, Imbalanced-learn 0.11+ |
| Model Serialization | Joblib 1.3+ |
| API Serving | Flask 2.3+ |
| Configuration | YAML (PyYAML) |
| Logging | Python `logging` module |

---

### 6. Project Directory Structure

```
fraud-detection-baseline/
├── config/
│   └── pipeline_config.yaml        # Paths and parameters
├── data/
│   ├── raw/
│   │   └── creditcard.csv          # Original dataset (gitignored)
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
│   │   └── ...
│   ├── evaluation_report.json
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
├── ARCHITECTURE.md
└── README.md
```

---

### 7. Execution Flow

To run the full pipeline sequentially from the terminal:

```bash
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
```bash
python src/serve.py
```

Test the endpoint:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 150000, "V1": -1.5, ..., "Amount": 120.0}'
```

Run periodic monitoring:
```bash
python src/monitor.py
```

---

### 8. Data Contract (Schema)

The following describes the schema of the `features.parquet` file consumed by model training and serving.

| Feature | Type | Description |
|---|---|---|
| `V1` – `V28` | float64 | PCA-transformed features (already scaled) |
| `Amount` | float64 | Transaction amount (USD) |
| `Amount_log` | float64 | Natural log of `Amount + 1` |
| `hour` | int64 | Hour of transaction (0–23) |
| `day` | int64 | Day index relative to first transaction |
| `hour_sin` | float64 | Cyclical sine of hour |
| `hour_cos` | float64 | Cyclical cosine of hour |
| `txn_count_1h` | int64 | Count of transactions in prior 3600 seconds |
| `txn_amount_avg_1h` | float64 | Mean transaction amount in prior 3600 seconds |
| `Class` | int64 | Target: 1 for fraud, 0 for legitimate |

---

### 9. Path to Production Scaling

Each baseline component maps directly to a production equivalent listed in the detailed architecture.

| Baseline Component | Production Replacement |
|---|---|
| CSV file read | Kafka consumer + S3 batch reads (Spark) |
| Pandera assertions | Great Expectations suite in Airflow DAG |
| Pandas preprocessing | Spark ML Pipeline / Feature Store transformations |
| Jupyter EDA notebook | Automated Sweetviz/Evidently reports in CI |
| Pandas rolling windows | Flink/Spark Streaming windows in Feature Store |
| Scikit-learn Random Forest | XGBoost/LightGBM with Optuna on GPU cluster |
| Joblib serialization | ONNX + MLflow Model Registry |
| Flask local API | FastAPI in Docker on Kubernetes with online feature fetch |
| Offline monitoring script | Evidently AI automated comparisons, Grafana dashboards, auto-retrain triggers |