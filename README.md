# Real-Time Credit Card Fraud Detection

## Project Overview
This project detects fraudulent credit card transactions in real-time at the point of authorization—before funds are cleared. The goal is to maximize fraud detection while minimizing false declines that frustrate legitimate customers.

**Industry:** Payments & Retail Banking  
**Workflow Stage:** Real-time authorization gateway (front-office)  
**Prediction Type:** Binary classification with probability scoring

---

## Business Context

### Problem Statement
Global card fraud losses exceed **$30 billion annually**. Beyond direct losses, false positives (declining legitimate transactions) damage customer trust and lead to cardholder churn. A 1% improvement in precision can save a mid-sized issuer tens of millions of dollars through reduced fraud losses and retained customer lifetime value.

### Current Limitations
Legacy rule-based systems (e.g., "block all foreign transactions over $500") suffer from:
- Rigid thresholds that fraudsters easily bypass
- Rapidly decaying effectiveness as tactics evolve
- Overwhelmingly high false positive rates
- Inability to detect complex, coordinated fraud patterns (botnets, synthetic identities)

### Key Stakeholders
- Fraud risk operations teams
- Cardholders (end customers)
- Issuing banks

---

## ML Problem Framing

| Aspect | Detail |
|--------|--------|
| **Inputs** | Tabular & sequential data: transaction amount, merchant category code (MCC), time of day, terminal type, rolling historical velocity features |
| **Output** | Fraud probability score (0–1) + binary decision threshold |
| **Prediction Frequency** | Per transaction, strictly real-time |
| **Latency Constraint** | 50–100 milliseconds |
| **Interpretability Requirement** | Yes — for adverse action notices on blocked cards |

### Why ML Works
ML captures complex, non-linear interactions between variables (e.g., unusual purchase time + specific merchant type + geographic distance from prior transaction) that are impossible to encode as static rules.

---

## Data

### Source Datasets
- **Credit Card Fraud Detection** (ULB / Kaggle) — primary dataset used

### Specifications
- 284,807 transactions (284,315 legitimate + 492 fraud)
- Fraud ratio: **0.17%** (extreme class imbalance)
- 28 PCA-transformed anonymized features (V1–V28) + Time + Amount
- Time range: ~48 hours (0–172,792 seconds)
- Amount range: $0.00–$25,691.16 (median $22.00, mean $88.35)
- No null values, no duplicates

### Feature Engineering
Two feature sets created for experimentation:

| Feature Set | Features | Description |
|-------------|----------|-------------|
| **Baseline** | 40 | V1–V28 + Amount/Time transforms + velocity + cyclical time |
| **Advanced** | 56 | Baseline + interaction terms (V17×V14, V12×V10, etc.) + anomaly scores + amount percentiles + recency |

### Known Limitations
- Raw categorical features (merchant names, location details) removed due to PII
- No user/card identifiers — velocity features computed globally, not per-card
- Single-transaction inference uses placeholder zeros for velocity features

---

## Project Structure

```
.
├── data/
│   ├── raw/                  # Original dataset (creditcard.csv)
│   ├── processed/            # Cleaned & feature-engineered parquet files
│   │   ├── cleaned.parquet
│   │   ├── features_baseline.parquet    # 40 features
│   │   ├── features_advanced.parquet    # 56 features
│   │   ├── train_baseline.parquet
│   │   ├── train_advanced.parquet
│   │   ├── test_baseline.parquet
│   │   └── test_advanced.parquet
│   ├── monitoring/           # Labeled transactions for drift detection
│   └── data_logging/         # EDA reports & visualizations
├── notebooks/                # Exploratory analysis & prototyping
├── src/
│   ├── models/               # Model training scripts
│   │   ├── train_baseline.py     # LR, DT, RF, NB with SMOTE + 3-fold CV
│   │   ├── train_advanced.py     # XGBoost, LightGBM, MLP
│   │   └── train_ssl.py          # Autoencoder SSL
│   ├── evaluation/           # Metrics calculation & visualization
│   │   └── evaluate.py
│   ├── serve.py              # Flask REST API with auto model selection
│   └── monitor.py            # Offline drift detection & model health checks
├── models/                   # Trained model artifacts (.pkl) + evaluation JSONs
├── artifacts/
│   ├── metrics/              # Training CV metrics
│   └── evaluation/           # Test set evaluation reports
├── reports/                  # Monitoring reports (timestamped JSON)
├── config/                   # Pipeline configuration YAML
├── logs/                     # API request logs
├── requirements.txt
└── README.md
```

---

## Experiment Results

### Full Model Comparison Matrix

**Quadrant 1: Baseline Models × Baseline Features (40 features)**

| Model | CV PR-AUC | Test PR-AUC | Test Recall | Test Precision | Test F1 | Test FPR |
|-------|-----------|-------------|-------------|----------------|---------|----------|
| **Random Forest** | 0.8222 | 0.8004 | 0.7973 | 0.4041 | 0.5364 | 0.15% |
| Logistic Regression | 0.7200 | 0.8037 | 0.8649 | 0.1098 | 0.1945 | 0.92% |
| Decision Tree | 0.6219 | 0.3584 | 0.8243 | 0.0987 | 0.1762 | 0.98% |
| Naive Bayes | 0.0931 | 0.3585 | 0.7973 | 0.0511 | 0.0961 | 1.93% |

**Quadrant 2: Baseline Models × Advanced Features (56 features)**

| Model | CV PR-AUC | Test PR-AUC | Test Recall | Test Precision | Test F1 | Test FPR |
|-------|-----------|-------------|-------------|----------------|---------|----------|
| Random Forest | 0.8140 | 0.8043 | 0.7973 | 0.5221 | 0.6310 | 0.10% |
| Logistic Regression | 0.7308 | 0.7411 | 0.8784 | 0.0781 | 0.1434 | 1.35% |
| Decision Tree | 0.5821 | 0.4686 | 0.8243 | 0.1564 | 0.2629 | 0.58% |
| Naive Bayes | 0.0881 | 0.4102 | 0.7973 | 0.0491 | 0.0925 | 2.02% |

**Advanced Models × Advanced Features (56 features)**

| Model | Test PR-AUC | Test Recall | Test Precision | Test F1 | Test FPR |
|-------|-------------|-------------|----------------|---------|----------|
| **🏆 LightGBM** | **0.8121** | 0.7838 | **0.8529** | **0.8169** | **0.02%** |
| XGBoost | 0.7882 | 0.7568 | 0.7671 | 0.7619 | 0.03% |
| MLP | 0.7621 | 0.6757 | 0.9259 | 0.7813 | 0.01% |

### Key Findings

| Insight | Detail |
|---------|--------|
| **🏆 Best model: LightGBM Advanced** | PR-AUC 0.8121, only 10 false positives out of 56,746 test transactions (0.02% FPR) |
| **Random Forest competitive** | RF Baseline had best CV score (0.8222) but LightGBM generalized better to test set |
| **Advanced features improved RF precision** | RF precision jumped from 0.4041 → 0.5221 with advanced features (+29%) |
| **Naive Bayes unusable** | PR-AUC < 0.10 regardless of feature set — fundamentally wrong for this data |
| **LR high recall, terrible precision** | 86-88% recall but ~5-11% precision — too many false positives for production |

### Winner: LightGBM Advanced

| Metric | Value | Business Impact |
|--------|-------|-----------------|
| Fraud Caught | 58/74 (78.4%) | Catches ~4 of 5 fraud attempts |
| Legitimate Declined | 10/56,672 (0.02%) | Near-zero customer insult rate |
| Inference Time | <1ms per transaction | Well within 50ms SLA |

---

## Model Deployment

### Flask REST API (`src/serve.py`)

Production-ready inference server with automatic best-model selection:

- **Model Registry**: Scans `models/` directory, reads evaluation JSONs, auto-selects best model by PR-AUC
- **Graceful fallback**: If top model fails to load, tries next-best candidate
- **Structured logging**: JSON-format request logs to `logs/service.log`
- **Health endpoint**: `GET /health` returns model info and status

### Starting the Server

```bash
# Development
python -c "from src.serve import get_app; app = get_app(); app.run(host='localhost', port=5000)"

# Production
gunicorn -w 4 -b 0.0.0.0:5000 src.serve:app
```

### API Usage

```bash
# Health check
curl http://localhost:5000/health

# Fraud prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 150000, "Amount": 120.0, "V1": -1.5, ... "V28": -0.4}'

# Response
{
  "fraud_probability": 0.1213,
  "is_fraud": false,
  "transaction_hash": "2145c...",
  "model": "lightgbm_advanced"
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAUD_MODELS_DIR` | `models/` | Path to model artifacts |
| `FRAUD_SCALER_PATH` | `models/autoencoder_ssl_scaler.pkl` | Path to fitted scaler |
| `FRAUD_THRESHOLD` | `0.5` | Decision threshold for fraud classification |
| `FRAUD_METRIC` | `pr_auc` | Metric to rank models by |

---

## Monitoring & Maintenance (`src/monitor.py`)

Offline monitoring pipeline that detects model performance degradation over time. Designed to run on a schedule against new labeled transaction batches (e.g., chargeback data).

### Drift Detection Rules

| Metric | Threshold | Action |
|--------|-----------|--------|
| **False Positive Rate** | >20% relative increase vs baseline | 🟡 Warning flagged |
| **PR-AUC** | >10% relative drop vs baseline | 🟡 Warning flagged |
| **Both healthy** | Within thresholds | ✅ No action required |

### Usage

```bash
python -c "
from pathlib import Path
from src.monitor import MonitoringConfig, run_monitoring

config = MonitoringConfig(
    new_data_path=Path('data/monitoring/new_transactions_labeled.csv'),
    model_path=Path('models/lightgbm_advanced.pkl'),
    baseline_report_path=Path('models/lightgbm_advanced_evaluation.json'),
    output_dir=Path('reports'),
)

result = run_monitoring(config)
print(f'Report: {result[\"report_path\"]}')
print(f'Action required: {result[\"action_required\"]}')
"
```

### Monitoring Report Format

```json
{
  "report_timestamp": "2026-05-07T00:08:36",
  "model_name": "lightgbm_advanced",
  "baseline": {
    "pr_auc": 0.8121,
    "false_positive_rate": 0.0002,
    "recall": 0.7838,
    "precision": 0.8529
  },
  "current": {
    "pr_auc": 0.8124,
    "false_positive_rate": 0.00018,
    "recall": 0.7838,
    "precision": 0.8529,
    "confusion_matrix": {
      "true_negatives": 56662,
      "false_positives": 10,
      "false_negatives": 16,
      "true_positives": 58
    }
  },
  "drift_flags": {
    "fpr_warning": false,
    "pr_auc_warning": false,
    "any_warning": false,
    "fpr_relative_change": -0.1177,
    "pr_auc_relative_change": -0.0003
  },
  "action_required": false
}
```

### Production Integration Path

| Current | Production Upgrade |
|---------|-------------------|
| CSV file input | Data warehouse (ClickHouse/BigQuery) |
| Local reports | Cloud object storage (S3/GCS) |
| Console logging | Slack/PagerDuty alerts + Grafana dashboard |
| Manual execution | Scheduled Airflow/Prefect DAG |
| Static thresholds | Adaptive thresholds based on historical variance |

---

## Model Approaches

### Baseline Models

| Model | Key Strengths | Purpose |
|-------|---------------|---------|
| **Logistic Regression** | Fast inference, highly interpretable coefficients | Strict linear baseline, feature significance analysis |
| **Decision Tree** | Captures non-linear rules natively, fully transparent | Simple rule extraction baseline |
| **Naive Bayes** | Excellent for categorical probability combinations, extremely fast | Probabilistic baseline for sparse categorical features |
| **Random Forest** | Robust to overfitting, built-in feature importance | Strong non-linear ensemble baseline |

### Advanced Models

| Model | Key Strengths | Implementation Notes |
|-------|---------------|---------------------|
| **LightGBM** | Industry standard, best test performance (PR-AUC 0.8121) | Deployed as production model |
| **XGBoost** | Strong gradient boosting baseline | PR-AUC 0.7882 on test set |
| **Multilayer Perceptron (MLP)** | Highest precision (0.9259), lowest FPR (0.01%) | Trade-off: lower recall (0.6757) |

### Self-Supervised Learning (Advanced)

| Model | Key Strengths | Implementation Notes |
|-------|---------------|---------------------|
| **Autoencoder SSL** | Unsupervised anomaly detection; scaler used for preprocessing | Trained; anomaly features integrated into advanced feature set |

*(TabNet and SimCLR architectures designed but pending implementation)*

---

## Evaluation Metrics

### Technical Metrics (Primary)
- **Precision-Recall AUC (PR-AUC)** — prioritized over ROC-AUC due to extreme class imbalance
- **Recall @ Fixed False Positive Rate** (e.g., recall at 1% FPR)
- **F1 Score** — harmonic mean of precision and recall

### Business Metrics
| Metric | Description | Current Value (LightGBM) |
|--------|-------------|---------------------------|
| Fraud Caught | Number of fraud transactions detected | 58/74 (78.4%) |
| False Positive Rate | Legitimate transactions incorrectly declined | 0.02% (10/56,672) |
| Inference Latency | End-to-end prediction time | <1ms per transaction |
| Model | Currently deployed | LightGBM Advanced |

---

## Setup & Installation

```bash
# Clone repository
git clone <repo-url>
cd fraud-detection-real-time

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Train all baseline models with SMOTE + 3-fold CV
python src/models/train_baseline.py --train-path data/processed/train_advanced.parquet --models logistic_regression decision_tree random_forest naive_bayes

# Train advanced models
python src/models/train_advanced.py --train-path data/processed/train_advanced.parquet --model lightgbm
python src/models/train_advanced.py --train-path data/processed/train_advanced.parquet --model xgboost

# Evaluate on test set
python src/evaluation/evaluate.py --model models/lightgbm_advanced.pkl --test-advanced data/processed/test_advanced.parquet --type advanced

# Start inference API
python -c "from src.serve import get_app; app = get_app(); app.run(host='localhost', port=5000)"

# Run model monitoring
python -c "from pathlib import Path; from src.monitor import MonitoringConfig, run_monitoring; config = MonitoringConfig(new_data_path=Path('data/monitoring/new_transactions_labeled.csv'), model_path=Path('models/lightgbm_advanced.pkl'), baseline_report_path=Path('models/lightgbm_advanced_evaluation.json'), output_dir=Path('reports')); result = run_monitoring(config); print(f'Action required: {result[\"action_required\"]}')"
```

---

## Key Dependencies

- Python 3.8+
- scikit-learn
- XGBoost / LightGBM
- imbalanced-learn (SMOTE)
- pandas, numpy, joblib
- Flask (serving API)
- PyTorch (SSL models)
- MLflow (experiment tracking — optional)

---

## Constraints & Requirements

| Requirement | Specification | Status |
|-------------|---------------|--------|
| Inference Latency | ≤ 100ms end-to-end | ✅ <1ms achieved |
| Model Interpretability | Required for regulatory notices | ⚠️ SHAP pending |
| Deployment Format | ONNX or pickle | ✅ joblib/pickle |
| Class Imbalance Handling | SMOTE + cost-sensitive learning | ✅ Implemented |
| Auto Model Selection | Best model by PR-AUC | ✅ ModelRegistry |
| Drift Monitoring | FPR + PR-AUC vs baseline | ✅ Offline monitor |

---

## Contributing

1. Create a feature branch from `main`
2. Run tests before committing: `pytest tests/`
3. Log all experiments using MLflow
4. Request code review before merging

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Last updated: 07/05/2026 — Baseline pipeline complete, 11 models trained, LightGBM deployed via Flask API, offline monitoring validated*