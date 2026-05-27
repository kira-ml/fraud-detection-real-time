# Real-Time Credit Card Fraud Detection

## Project Overview

A production-minded machine learning system for detecting fraudulent credit card transactions at the point of authorization—before funds clear. The system balances two competing objectives: catching fraud and avoiding false declines that frustrate legitimate customers and drive churn.

**Industry:** Payments & Retail Banking  
**Workflow Stage:** Real-time authorization gateway  
**Prediction Type:** Binary classification with calibrated probability scoring  
**Pipeline:** End-to-end — ingest → validate → preprocess → split → feature engineer → train → evaluate → deploy → monitor

![Model Comparison Dashboard](artifacts/plots/01_pr_curves.png)

---

## Business Context

### Problem Statement

Global card fraud losses exceed **$30 billion annually**. Beyond direct losses, false positives carry hidden costs: an estimated 20% of customers whose legitimate transactions are declined never return to that card. A 1% improvement in precision can save a mid-sized issuer tens of millions annually through reduced fraud losses and retained customer lifetime value.

### Limitations of Legacy Systems

Rule-based systems ("block all foreign transactions over $500") suffer from rigid thresholds that fraudsters adapt to quickly, false positive rates that overwhelm operations teams, and an inability to detect coordinated patterns like botnets or synthetic identity rings.

### Stakeholders

- **Fraud operations teams** — need actionable scores, not just binary decisions
- **Cardholders** — expect seamless transactions with minimal friction
- **Issuing banks** — balance fraud loss against customer experience cost

---

## ML Problem Framing

| Aspect | Detail |
|--------|--------|
| **Inputs** | 28 PCA-transformed features (V1–V28), transaction amount, timestamp |
| **Output** | Fraud probability score (0–1) with binary decision at configurable threshold |
| **Prediction Frequency** | Per transaction, sub-millisecond |
| **Latency Budget** | ≤ 100ms end-to-end |
| **Primary Metric** | Precision-Recall AUC (chosen over accuracy due to 0.17% fraud rate) |
| **Interpretability** | Feature importance rankings + SHAP explanations |

---

## Data

### Source

**Credit Card Fraud Detection** (ULB Brussels / Kaggle) — 284,807 transactions over ~48 hours.

### Characteristics

| Property | Value |
|----------|-------|
| Transactions | 284,807 (284,315 legitimate + 492 fraud) |
| Fraud Ratio | 0.17% (extreme class imbalance) |
| Features | 28 PCA components (V1–V28) + Time + Amount |
| Time Range | 0–172,792 seconds (~48 hours) |
| Amount Range | $0.00–$25,691.16 (median $22.00, mean $88.35) |
| Missing Values | None |
| Duplicates | 1,081 rows (0.38%) — removed during preprocessing |

### Limitations

- No card-level or merchant-level identifiers (velocity features are global, not per-entity)
- PCA-transformed features are anonymized — original transaction attributes are unavailable
- 48-hour window limits the model's exposure to weekly or seasonal patterns

---

## Feature Engineering

Feature engineering is the backbone of this project. Two independent feature sets are generated from the same raw data, with the split occurring **before** feature engineering to prevent temporal leakage.

### Baseline Feature Set (50 features)

Built from raw Time and Amount columns preserved through preprocessing.

| Category | Features | Rationale |
|----------|----------|-----------|
| **Temporal** | `hour` (0–23), `day` (0–1), `hour_sin`, `hour_cos`, `hour_of_day`, `is_night` | Fraud clusters at specific hours; cyclical encoding preserves distance between 23:00 and 00:00 |
| **Velocity (1h, 24h)** | `txn_count_1h`, `avg_amount_1h`, `std_amount_1h`, `txn_count_24h`, `avg_amount_24h`, `std_amount_24h` | Rapid succession or amount deviation from recent history signals fraud |
| **Amount Bucket Proxy** | `amount_bucket` (10 deciles), `txn_count_1h_by_bucket`, `avg_amount_1h_by_bucket`, `amount_to_bucket_avg_ratio` | Since no card ID exists, amount deciles serve as a proxy for customer spending segments |
| **Recency** | `time_since_last_txn`, `time_since_last_txn_same_bucket` | Fraudsters often execute rapid successive transactions |
| **Transform** | `amount_log` | Log transform handles the extreme right skew ($0–$25K, mean $88) |

### Advanced Feature Set (80 features)

Extends baseline with 30 additional engineered features.

| Category | Features | Rationale |
|----------|----------|-----------|
| **PCA Interactions** | `V17_V14`, `V12_V10`, `V4_V11`, `V3_V7`, `V17_V12`, `V14_V10`, `V17_V10`, `V14_V12`, `V16_V17`, `V3_V14` | Top EDA-correlated features (V17, V14, V12, V10, V16) combined pairwise |
| **PCA Ratios** | `V17_to_V14`, `V12_to_V10` | Relative behavior between top fraud signals |
| **Amount Deviation** | `amount_ratio_1h`, `amount_ratio_24h`, `amount_zscore_1h`, `amount_cv_1h` | Normalized deviation from short-term and daily averages |
| **Burst Detection** | `txn_count_10min`, `velocity_spike_ratio` | Short-term activity spikes; spike ratio > 1 indicates recent burst |
| **Domain Knowledge** | `fraud_direction_score` (0–5), `fraud_feature_magnitude` | Encodes EDA finding that V17/V14/V12/V10/V16 are negatively correlated with fraud |
| **Anomaly** | `anomaly_score`, `anomaly_decision` (Isolation Forest) | Unsupervised signal as supplementary input to supervised model |

### Design Decisions

**Raw value preservation:** `preprocess.py` scales Time and Amount with StandardScaler for model training but preserves `Time_raw` (0–172,792 seconds) and `Amount_raw` ($0–$25K) columns. All temporal and amount-based features are computed from these raw values, ensuring velocity windows use real seconds and amount statistics use real dollars.

---

## Pipeline Architecture

### Execution Order (Leakage Prevention)

```
ingest → validate → preprocess → SPLIT → feature_engineer(train) + feature_engineer(test) → train → evaluate
```

The split occurs **before** feature engineering. Train and test sets are engineered independently, ensuring velocity windows never observe future transactions. This is validated by a temporal integrity check: `max(train.Time_raw) < min(test.Time_raw)`.

### Orchestration

A single command runs the complete pipeline:

```bash
python src/pipeline.py --mode full        # All models
python src/pipeline.py --mode baseline    # Baseline only
python src/pipeline.py --mode advanced    # Advanced only
python src/pipeline.py --dry-run          # Quick test on 5,000 rows
```

The orchestrator (`src/pipeline.py`) wires 9 modules in sequence, passing DataFrames between stages rather than relying on intermediate file I/O. Each module exposes a clean function signature (`f(DataFrame) → DataFrame`) for composability and testability.

### Module Map

| Step | Module | Key Function | Input | Output |
|------|--------|-------------|-------|--------|
| 1 | `ingest.py` | `ingest_data()` | CSV path | DataFrame |
| 2 | `validate.py` | `validate_data()` | DataFrame | DataFrame + report |
| 3 | `preprocess.py` | `preprocess_data()` | DataFrame | DataFrame + scaler |
| 4 | `split.py` | `split_data()` | DataFrame | train_df, test_df |
| 5 | `feature_engineering.py` | `run_feature_engineering_baseline()` / `_advanced()` | DataFrame (per split) | DataFrame + feature list |
| 6 | `train_baseline.py` / `train_advanced.py` | `run_baseline_training()` / `run_advanced_training()` | train path | model artifacts |
| 7 | `evaluate.py` | `run_evaluation()` | model + test path | metrics JSON |
| — | `serve.py` | Flask API | JSON request | fraud probability |
| — | `monitor.py` | `run_monitoring()` | labeled batch | drift report |

---

## Experiment Results

Seven models were trained and evaluated on an honest, leakage-free time split (80% train / 20% test, split by raw timestamp before feature engineering). All metrics reported on the held-out test set.

**Baseline models** are trained on the 50-feature baseline set. **Advanced models** receive the baseline features plus 30 additional engineered features (80 total). The two tiers never cross — this is a clean comparison of whether the extra feature engineering adds value.

### Baseline Models (50 features)

| Model | CV PR-AUC | Test PR-AUC | Test Recall | Test Precision | Test FPR |
|-------|-----------|-------------|-------------|----------------|----------|
| **Logistic Regression** | 0.7253 | **0.7971** | **0.8866** | 0.0728 | 1.49% |
| Random Forest | **0.8192** | 0.7946 | 0.8041 | **0.4105** | **0.15%** |
| Decision Tree | 0.6280 | 0.5121 | 0.7938 | 0.0358 | 2.83% |
| Naive Bayes | 0.0855 | 0.3595 | 0.7938 | 0.0450 | 2.23% |

### Advanced Models (80 features)

| Model | CV PR-AUC | Test PR-AUC | Test Recall | Test Precision | Test FPR |
|-------|-----------|-------------|-------------|----------------|----------|
| **🏆 LightGBM** | **0.8579** | **0.7925** | 0.7629 | **0.7957** | **0.03%** |
| XGBoost | 0.8425 | 0.7833 | 0.7629 | 0.6727 | 0.05% |
| MLP | 0.8131 | 0.7770 | 0.7629 | 0.7327 | 0.04% |

### Key Findings

**Model selection — LightGBM wins on operational cost, not just metrics.** LightGBM catches 74 of 97 fraud cases (76.3%) while incorrectly flagging only 19 of 73,337 legitimate transactions — a false positive rate of 0.03%. For context, a typical rule-based system at 70% recall would generate approximately 3,500 false positives on the same volume, 185× more than LightGBM.

**The real story is the business cost.** Logistic Regression catches more fraud (86/97, 88.7%) but generates 1,096 false alarms — a 1.49% false positive rate. In a system processing 100,000 transactions daily, that's 1,490 annoyed customers calling their bank. LightGBM's 0.03% FPR produces only 30 false flags per day. Assuming $500 average fraud loss and $300 customer churn cost per false decline:

| Model | Fraud Caught | False Alarms | Business Cost |
|-------|-------------|-------------|---------------|
| Logistic Regression (default threshold) | 86/97 | 1,096 | $334,300 |
| LightGBM (default threshold) | 74/97 | 19 | **$17,200** |
| LightGBM (threshold 0.7) | 73/97 | 13 | **$15,900** |

**LightGBM reduces operational cost by ~95% compared to the high-recall baseline**, even though it catches fewer fraud cases. Trading 12-13 missed frauds for over 1,000 fewer false alarms is the correct business decision at scale.

**Feature engineering impact — engineered features dominate raw PCA.** `fraud_direction_score` (encoding domain knowledge that V17/V14/V12/V10/V16 are negatively correlated with fraud) is LightGBM's #1 feature. `fraud_feature_magnitude` is XGBoost's top feature at 28.9% importance. The top 5 LightGBM features are all either engineered interaction terms or domain-driven scores — none are raw PCA components. Temporal features (`hour_sin`, `time_since_last_txn`) and velocity features (`txn_count_10min`, `std_amount_24h`) appear in the top 15, confirming they carry signal when computed from raw seconds rather than scaled values.

![Feature Importance](artifacts/plots/03_feature_importance.png)

**SHAP analysis reveals feature direction, not just importance.** High values of `V14_V10` strongly push predictions toward fraud, likely capturing a specific attack vector. `fraud_direction_score` shows a clean linear relationship — acting as a maliciousness score. High `V8` pushes predictions away from fraud, suggesting it captures normal high-value transaction patterns and acts as a whitelisting signal.

![SHAP Summary](artifacts/plots/05_shap_summary.png)

**Baseline vs. advanced trade-off:** Random Forest on 50 baseline features achieves a test PR-AUC of 0.7946 — only 0.002 above LightGBM on 80 features. However, RF generates 112 false alarms (0.15% FPR) compared to LightGBM's 19 (0.03% FPR). The advanced feature set doesn't dramatically improve recall — it dramatically improves precision. For deployment scenarios where model simplicity matters, RF offers a strong alternative; where operational cost dominates, LightGBM is the clear winner.

**Naive Bayes and Decision Tree** are included as lower bounds. Naive Bayes fails (PR-AUC < 0.10) regardless of feature set — expected given the strong feature dependencies in this domain.

### Winner: LightGBM Advanced

| Metric | Value | Context |
|--------|-------|---------|
| Fraud Caught | 74/97 (76.3%) | Detects ~3 of 4 fraud attempts |
| False Positives | 19/73,337 (0.03%) | 1 in 3,860 legitimate transactions flagged |
| Precision | 79.6% | Nearly 4 of 5 flagged transactions are genuine fraud |
| Business Cost (default) | $17,200 | vs $334,300 for high-recall baseline |
| Inference Time | <1ms per transaction | Well within 50ms SLA |

---

## Error Analysis: Why 23 Fraud Cases Go Undetected

All three advanced models (LightGBM, XGBoost, MLP) miss exactly the same 23 fraud cases — a hard recall ceiling at 76.3%. Investigating these cases reveals they share a distinct profile:

| Feature | Missed Fraud (mean) | Caught Fraud (mean) | Interpretation |
|---------|--------------------|--------------------|----------------|
| `fraud_direction_score` | 2.57 | 4.84 | Missed fraud triggers only 2-3 of 5 fraud signals |
| `fraud_feature_magnitude` | 6.08 | 24.20 | 4× weaker overall fraud signal intensity |
| `V14_V10` | 0.78 | 33.62 | The #1 SHAP feature is 43× lower |
| `V14` | -1.57 | -7.76 | Missed fraud lacks the strong negative V14 signal |
| `V17` | +0.80 | -3.93 | Missed fraud has *positive* V17 (caught fraud is strongly negative) |

**Model confidence confirms the pattern:** the 23 missed cases have a mean predicted fraud probability of **4.7%** (median 1.1%), while caught fraud has a mean of **98.8%** (median 99.9%). The model isn't uncertain about these cases — it's confident they're legitimate.

**Conclusion:** This is a data ceiling, not a model problem. The 23 missed fraud cases don't express the PCA patterns the model learned to associate with fraud. Without card-level behavioral history, merchant category codes, or device fingerprints, a subset of fraud will always resemble normal spending in an anonymized feature space. Closing this gap would require features unavailable in this dataset.

---

## Threshold Tuning: Choosing an Operating Point

The decision threshold is configurable — different business scenarios demand different tradeoffs. Measured results from a full threshold sweep (0.1–0.9):

### LightGBM

| Threshold | Recall | Precision | FPR | Fraud Caught | False Alarms | Business Cost | Use Case |
|-----------|--------|-----------|-----|-------------|-------------|---------------|----------|
| 0.3 | 77.3% | 70.1% | 0.04% | 75/97 | 32 | $20,600 | Fraud wave response |
| 0.5 | 76.3% | 79.6% | 0.03% | 74/97 | 19 | $17,200 | Balanced operations (default) |
| 0.7 | 75.3% | 84.9% | 0.02% | 73/97 | 13 | $15,900 | Customer experience priority |

### Logistic Regression (high-recall alternative)

| Threshold | Recall | Precision | FPR | Fraud Caught | False Alarms | Business Cost |
|-----------|--------|-----------|-----|-------------|-------------|---------------|
| 0.3 | 90.7% | 3.9% | 2.93% | 88/97 | 2,150 | $649,500 |
| 0.5 | 88.7% | 7.3% | 1.49% | 86/97 | 1,096 | $334,300 |
| 0.7 | 84.5% | 13.9% | 0.69% | 82/97 | 508 | $159,900 |

Even at the most conservative threshold (0.9), Logistic Regression generates 174 false alarms compared to LightGBM's 4. The advanced feature set doesn't just improve recall — it fundamentally changes the precision-recall tradeoff in a way that makes the model operationally viable at scale.

![Threshold Tradeoff](artifacts/plots/04_threshold_tradeoff.png)

---

## Model Deployment

### Flask REST API (`src/serve.py`)

- **Auto model selection:** Scans `models/` directory at startup, selects best model by PR-AUC from evaluation metadata
- **Graceful fallback:** If the top-ranked model fails to load, the registry tries the next candidate
- **PostgreSQL audit trail:** Every prediction logged to `transactions` and `predictions` tables for downstream monitoring and feedback collection
- **Structured logging:** JSON-format request logs to `logs/service.log`

```bash
# Start server
python -c "from src.serve import get_app; app = get_app(); app.run(host='localhost', port=5000)"

# Predict
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 150000, "Amount": 120.0, "V1": -1.5, ... "V28": -0.4}'

# Response
{
  "fraud_probability": 0.082,
  "is_fraud": false,
  "transaction_hash": "2145c...",
  "model": "lightgbm_advanced"
}
```

---

## Monitoring & Maintenance

### Drift Detection (`src/monitor.py`)

An offline monitoring pipeline compares current model performance against evaluation baselines. Designed to run on a schedule against newly labeled transaction batches.

| Metric | Threshold | Action |
|--------|-----------|--------|
| False Positive Rate | >20% relative increase vs baseline | Warning flagged |
| PR-AUC | >10% relative drop vs baseline | Warning flagged |

Supports both CSV-based monitoring and database-sourced monitoring via `vw_transaction_results` view in PostgreSQL.

---

## Database Infrastructure

A PostgreSQL 18 database (`fraud_detection`) provides an audit trail and monitoring backbone:

| Table | Purpose |
|-------|---------|
| `transactions` | Raw transaction storage (284,808 rows) |
| `predictions` | Model prediction logging from `/predict` endpoint |
| `ground_truth` | Chargeback/feedback labels for performance tracking |
| `model_registry` | Deployed model version tracking |
| `monitoring_metrics` | Evaluation and drift metrics history |

---

## What I Would Do With Real Data

This dataset uses PCA-anonymized features with no card or merchant identifiers — a common limitation of public fraud detection benchmarks. With access to real payment network data, I would add:

- **Card-level velocity features:** Per-card transaction frequency, amount deviation from cardholder baseline, geographic velocity between transactions
- **Merchant risk signals:** Category codes (MCC), merchant fraud history, time since merchant's first transaction in the network
- **Network graph features:** Shared attributes between transactions (device fingerprints, IP prefixes, BIN ranges) to detect coordinated fraud rings and synthetic identity patterns
- **Streaming architecture:** Replace batch pipeline with Kafka/Flink for true sub-100ms feature computation on sliding windows, enabling real-time velocity features at authorization time
- **Feedback loop:** Integrate chargeback data as ground truth labels to automatically retrain on confirmed fraud patterns

---

## Constraints & Current Status

| Requirement | Specification | Status |
|-------------|---------------|--------|
| Inference Latency | ≤ 100ms | ✅ <1ms |
| Temporal Leakage Prevention | Split before feature engineering | ✅ Validated |
| Class Imbalance Handling | SMOTE + cost-sensitive learning | ✅ |
| Auto Model Selection | Best model by PR-AUC at startup | ✅ |
| Drift Monitoring | FPR + PR-AUC vs baseline | ✅ |
| Database Audit Trail | PostgreSQL prediction + transaction logging | ✅ |
| Model Interpretability | SHAP explanations | ✅ |
| Error Analysis | Missed fraud profiling | ✅ |
| Threshold Tuning | Measured sweep with business cost | ✅ |

---

## Setup

```bash
git clone <repo-url>
cd fraud-detection-real-time
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run full pipeline
python src/pipeline.py --mode full

# Quick test
python src/pipeline.py --mode baseline --dry-run

# Generate visualizations
python src/evaluation/visualize.py

# Start API
python -c "from src.serve import get_app; app = get_app(); app.run(host='localhost', port=5000)"
```

---

## Dependencies

Python 3.8+ · scikit-learn · XGBoost · LightGBM · imbalanced-learn · pandas · NumPy · Flask · PyTorch · SHAP · PostgreSQL (SQLAlchemy) · joblib · PyYAML

---

## License

MIT — see [LICENSE](LICENSE).

---

*Last updated: 2026-05-28 — Full pipeline validated with leakage-free temporal split. LightGBM deployed at 0.7925 test PR-AUC, 79.6% precision, 0.03% FPR. Threshold sweep, SHAP analysis, and missed fraud profiling complete. Business cost analysis confirms LightGBM reduces operational loss by ~95% compared to high-recall baseline.*
