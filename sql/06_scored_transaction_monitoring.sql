-- FraudOps SQL Analytics
-- File: 06_scored_transaction_monitoring.sql
--
-- Business question:
-- What is happening in the scored transaction queue created by the API,
-- dashboard demo seeding, or batch scoring workflow?
--
-- Why this matters:
-- After a model is trained, risk teams need to monitor scored transactions,
-- decision output, high-risk share, and recent activity.
--
-- Tables used:
-- - scored_transactions: fraud probability, risk tier, decision, and scored_at.
-- - transactions: optional amount, merchant, location, and transaction timestamp.
--
-- Output meaning:
-- These queries summarize scored transaction counts, risk-tier distribution,
-- decision distribution, top high-risk rows, recent scored rows, and high-risk
-- share of all scored transactions.

-- Result: scored transaction monitoring summary.
SELECT
    COUNT(*) AS scored_transaction_count,
    SUM(CASE WHEN risk_tier = 'High' THEN 1 ELSE 0 END) AS high_risk_count,
    SUM(CASE WHEN decision = 'Manual Review' THEN 1 ELSE 0 END) AS manual_review_count,
    COALESCE(
        ROUND(100.0 * SUM(CASE WHEN risk_tier = 'High' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2),
        0
    ) AS high_risk_share_percent,
    ROUND(AVG(fraud_probability), 4) AS average_fraud_probability
FROM scored_transactions;

-- Result: scored transaction count and average probability by risk tier.
SELECT
    risk_tier,
    COUNT(*) AS transaction_count,
    ROUND(AVG(fraud_probability), 4) AS average_fraud_probability
FROM scored_transactions
GROUP BY risk_tier
ORDER BY
    CASE risk_tier
        WHEN 'High' THEN 1
        WHEN 'Medium' THEN 2
        WHEN 'Low' THEN 3
        ELSE 4
    END;

-- Result: scored transaction count by decision.
SELECT
    decision,
    COUNT(*) AS transaction_count,
    ROUND(AVG(fraud_probability), 4) AS average_fraud_probability
FROM scored_transactions
GROUP BY decision
ORDER BY transaction_count DESC, average_fraud_probability DESC;

-- Result: top high-risk scored transactions.
SELECT
    s.transaction_id,
    ROUND(s.fraud_probability, 4) AS fraud_probability,
    s.risk_tier,
    s.decision,
    s.scored_at,
    t.amount,
    t.merchant_category,
    t.location,
    t.transaction_time
FROM scored_transactions AS s
LEFT JOIN transactions AS t
    ON s.transaction_id = t.transaction_id
WHERE s.risk_tier = 'High'
   OR s.decision = 'Manual Review'
ORDER BY s.fraud_probability DESC, s.scored_at DESC
LIMIT 10;

-- Result: most recent scored transactions.
SELECT
    s.transaction_id,
    ROUND(s.fraud_probability, 4) AS fraud_probability,
    s.risk_tier,
    s.decision,
    s.scored_at,
    t.amount,
    t.merchant_category,
    t.location
FROM scored_transactions AS s
LEFT JOIN transactions AS t
    ON s.transaction_id = t.transaction_id
ORDER BY s.scored_at DESC, s.fraud_probability DESC
LIMIT 10;
