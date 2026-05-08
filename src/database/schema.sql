-- ============================================================================
-- Fraud Detection Database Schema (PostgreSQL 18)
-- Production-ready schema for real-time credit card fraud detection
-- ============================================================================

-- Enable UUID generation for all tables
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- 1. TRANSACTIONS TABLE
-- Stores raw incoming transaction requests from the API
-- ============================================================================
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp_received TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Core features from ULB dataset
    time_seconds FLOAT NOT NULL,
    amount FLOAT NOT NULL CHECK (amount >= 0),
    
    -- PCA-transformed features V1-V28
    v1 FLOAT, v2 FLOAT, v3 FLOAT, v4 FLOAT, v5 FLOAT,
    v6 FLOAT, v7 FLOAT, v8 FLOAT, v9 FLOAT, v10 FLOAT,
    v11 FLOAT, v12 FLOAT, v13 FLOAT, v14 FLOAT, v15 FLOAT,
    v16 FLOAT, v17 FLOAT, v18 FLOAT, v19 FLOAT, v20 FLOAT,
    v21 FLOAT, v22 FLOAT, v23 FLOAT, v24 FLOAT, v25 FLOAT,
    v26 FLOAT, v27 FLOAT, v28 FLOAT,
    
    -- Future: Entity identifiers for graph-based models
    card_id VARCHAR(50),
    merchant_id VARCHAR(50),
    merchant_category VARCHAR(10),
    device_id VARCHAR(50),
    ip_address VARCHAR(45)
);

-- ============================================================================
-- 2. PREDICTIONS TABLE
-- Stores model output for each transaction
-- ============================================================================
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id UUID NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    
    -- Model metadata
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) DEFAULT '1.0',
    
    -- Prediction output
    fraud_probability FLOAT NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    is_fraud BOOLEAN NOT NULL,
    decision_threshold FLOAT DEFAULT 0.5,
    
    -- Performance tracking
    inference_time_ms FLOAT DEFAULT 0.0,
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 3. GROUND TRUTH TABLE
-- Stores actual outcomes from chargebacks, manual review, customer reports
-- ============================================================================
CREATE TABLE IF NOT EXISTS ground_truth (
    feedback_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id UUID NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    
    -- Actual label: 0 = legitimate, 1 = fraud
    true_label INTEGER NOT NULL CHECK (true_label IN (0, 1)),
    
    -- Source of ground truth
    feedback_source VARCHAR(50) DEFAULT 'unknown',
    -- Examples: 'chargeback', 'manual_review', 'customer_report', 'delayed_settlement'
    
    -- Metadata
    reported_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    
    -- Ensure one feedback per transaction
    UNIQUE(transaction_id)
);

-- ============================================================================
-- 4. MODEL REGISTRY TABLE
-- Tracks all deployed models and their metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_registry (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_filepath VARCHAR(500),
    
    -- Performance metrics stored as JSON
    metrics JSONB,
    -- Example: {"pr_auc": 0.8121, "recall": 0.7838, "precision": 0.8529, "fpr": 0.0002}
    
    -- Model status
    is_active BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    decommissioned_at TIMESTAMP WITH TIME ZONE,
    
    -- Unique constraint per model name + version
    UNIQUE(model_name, model_version)
);

-- ============================================================================
-- 5. MONITORING METRICS TABLE
-- Stores drift detection and performance monitoring results
-- ============================================================================
CREATE TABLE IF NOT EXISTS monitoring_metrics (
    check_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    check_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Metric details
    metric_type VARCHAR(50) NOT NULL,
    -- Examples: 'pr_auc', 'fpr', 'recall', 'precision', 'psi', 'data_drift'
    
    metric_value FLOAT,
    baseline_value FLOAT,
    deviation_percent FLOAT,
    
    -- Alert status
    is_warning BOOLEAN DEFAULT FALSE,
    is_critical BOOLEAN DEFAULT FALSE,
    
    -- Additional details as JSON
    details JSONB,
    -- Example: {"window": "daily", "sample_size": 1000, "threshold_breach": "fpr_relative_change"}
    
    -- Action taken
    action_taken VARCHAR(100),
    -- Examples: 'none', 'alert_sent', 'model_rollback', 'retraining_triggered'
    
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Transactions
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp 
    ON transactions(timestamp_received DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_amount 
    ON transactions(amount);
CREATE INDEX IF NOT EXISTS idx_transactions_card 
    ON transactions(card_id) WHERE card_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_transactions_merchant 
    ON transactions(merchant_id) WHERE merchant_id IS NOT NULL;

-- Predictions
CREATE INDEX IF NOT EXISTS idx_predictions_transaction 
    ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model_time 
    ON predictions(model_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_fraud_prob 
    ON predictions(fraud_probability);

-- Ground Truth
CREATE INDEX IF NOT EXISTS idx_ground_truth_transaction 
    ON ground_truth(transaction_id);
CREATE INDEX IF NOT EXISTS idx_ground_truth_label 
    ON ground_truth(true_label);
CREATE INDEX IF NOT EXISTS idx_ground_truth_source 
    ON ground_truth(feedback_source);

-- Monitoring
CREATE INDEX IF NOT EXISTS idx_monitoring_model_time 
    ON monitoring_metrics(model_name, check_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_monitoring_type 
    ON monitoring_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_monitoring_warning 
    ON monitoring_metrics(is_warning) WHERE is_warning = TRUE;

-- ============================================================================
-- ANALYTICS VIEWS
-- ============================================================================

-- View 1: Complete transaction results with predictions and ground truth
CREATE OR REPLACE VIEW vw_transaction_results AS
SELECT 
    t.transaction_id,
    t.timestamp_received,
    t.time_seconds,
    t.amount,
    t.card_id,
    t.merchant_id,
    p.fraud_probability,
    p.is_fraud AS predicted_fraud,
    p.model_name,
    p.model_version,
    p.inference_time_ms,
    gt.true_label AS actual_fraud,
    gt.feedback_source,
    gt.reported_at AS feedback_reported_at,
    CASE 
        WHEN p.is_fraud = TRUE AND gt.true_label = 1 THEN 'true_positive'
        WHEN p.is_fraud = TRUE AND gt.true_label = 0 THEN 'false_positive'
        WHEN p.is_fraud = FALSE AND gt.true_label = 1 THEN 'false_negative'
        WHEN p.is_fraud = FALSE AND gt.true_label = 0 THEN 'true_negative'
        ELSE 'unlabeled'
    END AS result_category,
    CASE 
        WHEN p.is_fraud = TRUE AND gt.true_label = 1 THEN '✅ Caught'
        WHEN p.is_fraud = TRUE AND gt.true_label = 0 THEN '❌ False Alarm'
        WHEN p.is_fraud = FALSE AND gt.true_label = 1 THEN '⚠️ Missed Fraud'
        WHEN p.is_fraud = FALSE AND gt.true_label = 0 THEN '✅ OK'
        ELSE '⏳ Pending'
    END AS result_status
FROM transactions t
LEFT JOIN predictions p ON t.transaction_id = p.transaction_id
LEFT JOIN ground_truth gt ON t.transaction_id = gt.transaction_id;

-- View 2: Daily model performance metrics
CREATE OR REPLACE VIEW vw_daily_performance AS
SELECT 
    DATE(t.timestamp_received) AS transaction_date,
    p.model_name,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN p.is_fraud = TRUE THEN 1 ELSE 0 END) AS flagged_as_fraud,
    SUM(CASE WHEN gt.true_label = 1 THEN 1 ELSE 0 END) AS actual_fraud_count,
    SUM(CASE WHEN p.is_fraud = TRUE AND gt.true_label = 1 THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN p.is_fraud = TRUE AND gt.true_label = 0 THEN 1 ELSE 0 END) AS false_positives,
    SUM(CASE WHEN p.is_fraud = FALSE AND gt.true_label = 1 THEN 1 ELSE 0 END) AS false_negatives,
    SUM(CASE WHEN p.is_fraud = FALSE AND gt.true_label = 0 THEN 1 ELSE 0 END) AS true_negatives,
    
    -- Calculated metrics
    CASE 
        WHEN SUM(CASE WHEN gt.true_label = 1 THEN 1 ELSE 0 END) > 0 
        THEN SUM(CASE WHEN p.is_fraud = TRUE AND gt.true_label = 1 THEN 1 ELSE 0 END)::FLOAT / 
             SUM(CASE WHEN gt.true_label = 1 THEN 1 ELSE 0 END)
        ELSE 0 
    END AS recall,
    CASE 
        WHEN SUM(CASE WHEN p.is_fraud = TRUE THEN 1 ELSE 0 END) > 0 
        THEN SUM(CASE WHEN p.is_fraud = TRUE AND gt.true_label = 1 THEN 1 ELSE 0 END)::FLOAT / 
             SUM(CASE WHEN p.is_fraud = TRUE THEN 1 ELSE 0 END)
        ELSE 0 
    END AS precision,
    CASE 
        WHEN SUM(CASE WHEN gt.true_label = 0 THEN 1 ELSE 0 END) > 0 
        THEN SUM(CASE WHEN p.is_fraud = TRUE AND gt.true_label = 0 THEN 1 ELSE 0 END)::FLOAT / 
             SUM(CASE WHEN gt.true_label = 0 THEN 1 ELSE 0 END)
        ELSE 0 
    END AS false_positive_rate,
    AVG(p.fraud_probability) AS avg_fraud_score,
    AVG(p.inference_time_ms) AS avg_latency_ms,
    AVG(t.amount) AS avg_transaction_amount
FROM transactions t
JOIN predictions p ON t.transaction_id = p.transaction_id
LEFT JOIN ground_truth gt ON t.transaction_id = gt.transaction_id
GROUP BY DATE(t.timestamp_received), p.model_name;

-- View 3: Model leaderboard (latest deployed version)
CREATE OR REPLACE VIEW vw_model_leaderboard AS
SELECT 
    model_name,
    model_version,
    metrics->>'pr_auc' AS pr_auc,
    metrics->>'recall' AS recall,
    metrics->>'precision' AS precision,
    metrics->>'fpr' AS false_positive_rate,
    metrics->>'f1' AS f1_score,
    is_active,
    deployed_at
FROM model_registry
WHERE is_active = TRUE
ORDER BY (metrics->>'pr_auc')::FLOAT DESC;

-- View 4: Recent alerts and warnings
CREATE OR REPLACE VIEW vw_recent_alerts AS
SELECT 
    model_name,
    check_timestamp,
    metric_type,
    metric_value,
    baseline_value,
    deviation_percent,
    is_warning,
    is_critical,
    action_taken
FROM monitoring_metrics
WHERE (is_warning = TRUE OR is_critical = TRUE)
  AND check_timestamp >= CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY check_timestamp DESC;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function: Get model performance for a date range
CREATE OR REPLACE FUNCTION get_model_performance(
    p_model_name VARCHAR,
    p_start_date DATE DEFAULT CURRENT_DATE - INTERVAL '7 days',
    p_end_date DATE DEFAULT CURRENT_DATE
)
RETURNS TABLE (
    metric_date DATE,
    total_txns BIGINT,
    fraud_caught BIGINT,
    false_alarms BIGINT,
    missed_fraud BIGINT,
    recall_rate FLOAT,
    precision_rate FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        transaction_date,
        total_transactions,
        true_positives,
        false_positives,
        false_negatives,
        recall,
        precision
    FROM vw_daily_performance
    WHERE model_name = p_model_name
      AND transaction_date BETWEEN p_start_date AND p_end_date
    ORDER BY transaction_date DESC;
END;
$$ LANGUAGE plpgsql;

-- Function: Record feedback and auto-update monitoring
CREATE OR REPLACE FUNCTION record_feedback(
    p_transaction_id UUID,
    p_true_label INTEGER,
    p_source VARCHAR DEFAULT 'manual_review',
    p_notes TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO ground_truth (transaction_id, true_label, feedback_source, notes)
    VALUES (p_transaction_id, p_true_label, p_source, p_notes)
    ON CONFLICT (transaction_id) 
    DO UPDATE SET 
        true_label = EXCLUDED.true_label,
        feedback_source = EXCLUDED.feedback_source,
        notes = EXCLUDED.notes,
        reported_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INITIAL DATA: Insert current model
-- ============================================================================
INSERT INTO model_registry (model_name, model_version, metrics, is_active)
VALUES (
    'lightgbm_advanced',
    '1.0',
    '{"pr_auc": 0.8121, "recall": 0.7838, "precision": 0.8529, "fpr": 0.0002, "f1": 0.8169}'::JSONB,
    TRUE
)
ON CONFLICT (model_name, model_version) DO NOTHING;

-- ============================================================================
-- DONE
-- ============================================================================