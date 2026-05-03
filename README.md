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
- **IEEE-CIS Fraud Detection** (Kaggle)
- **Credit Card Fraud Detection** (ULB / Kaggle)

### Specifications
- ~280,000 to 500,000+ transactions
- Highly imbalanced: typically <0.5% fraud
- Anonymized numerical features (PCA-transformed)

### Known Limitations
- Raw categorical features (merchant names, location details) removed due to PII
- Limited user history depth in public datasets

---

## Project Structure

```
.
├── data/
│   ├── raw/              # Original dataset files
│   ├── processed/        # Cleaned & feature-engineered datasets
│   └── external/         # Third-party data sources
├── notebooks/            # Exploratory analysis & prototyping
├── src/
│   ├── data/             # Data ingestion, cleaning, splitting
│   ├── features/         # Feature engineering (velocity features, aggregations)
│   ├── models/           # Model training, hyperparameter tuning
│   ├── evaluation/       # Metrics calculation & visualization
│   └── inference/        # Real-time prediction pipeline
├── tests/                # Unit & integration tests
├── configs/              # Model & pipeline configuration files
├── models/               # Saved/trained model artifacts
├── docs/                 # Documentation
├── requirements.txt      # Python dependencies
└── README.md
```

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
| **XGBoost / LightGBM** | Industry standard for tabular data | Tree depth limited to ensure microsecond inference |
| **Isolation Forests** | Unsupervised anomaly detection | Complements supervised models; detects novel/zero-day fraud patterns |
| **Multilayer Perceptron (MLP)** | Captures complex feature interactions | Optimizable for GPU-accelerated real-time inference |

---

## Evaluation Metrics

### Technical Metrics (Primary)
- **Precision-Recall AUC (PR-AUC)** — prioritized over ROC-AUC due to extreme class imbalance
- **Recall @ Fixed False Positive Rate** (e.g., recall at 1% FPR)

### Business Metrics
| Metric | Description |
|--------|-------------|
| Fraud Value Saved ($) | Total monetary value of caught fraud |
| False Positive Rate | Percentage of legitimate transactions incorrectly declined ("customer insult rate") |
| Inference Latency (ms) | End-to-end prediction time, must stay within SLA |

---

## Setup & Installation

```bash
# Clone repository
git clone <repo-url>
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run data preprocessing
python src/data/preprocess.py

# Train baseline model
python src/models/train_baseline.py --model logistic_regression

# Train advanced model
python src/models/train_advanced.py --model xgboost

# Evaluate
python src/evaluation/evaluate.py --model-path models/xgboost.pkl
```

---

## Key Dependencies

- Python 3.8+
- scikit-learn
- XGBoost / LightGBM
- imbalanced-learn (SMOTE)
- pandas, numpy
- ONNX Runtime (inference optimization)
- MLflow (experiment tracking)

---

## Constraints & Requirements

| Requirement | Specification |
|-------------|---------------|
| Inference Latency | ≤ 100ms end-to-end |
| Model Interpretability | Required for regulatory/adverse action notices |
| Deployment Format | ONNX or high-performance C++ backend with Python wrapper |
| Class Imbalance Handling | SMOTE, cost-sensitive learning, appropriate evaluation metrics |

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

*Last updated: 03/05/2026*
