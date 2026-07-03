-- FraudOps SQL Analytics
-- File: 05_threshold_review_volume.sql
--
-- Business question:
-- How does changing the fraud probability threshold affect review volume and
-- operational workload?
--
-- Why this matters:
-- Fraud models do not make decisions in a vacuum. A low threshold may catch
-- more suspicious activity but can overwhelm manual reviewers. A high threshold
-- reduces workload but may miss risky transactions.
--
-- Tables used:
-- - scored_transactions: model probability, risk tier, decision, and score time.
-- - transactions: optional amount context joined by transaction_id.
--
-- Output meaning:
-- The first query estimates how many scored transactions would be reviewed at
-- each probability threshold. Because scored API transactions may not include
-- true labels, this focuses on operational burden rather than precision/recall.
-- The second query summarizes average fraud probability by assigned risk tier.

-- Result: estimated review workload by probability threshold.
WITH thresholds(threshold_value) AS (
    SELECT 0.10 UNION ALL
    SELECT 0.30 UNION ALL
    SELECT 0.50 UNION ALL
    SELECT 0.70 UNION ALL
    SELECT 0.90
),
scored AS (
    SELECT
        s.transaction_id,
        s.fraud_probability,
        s.risk_tier,
        s.decision,
        COALESCE(t.amount, 0) AS amount
    FROM scored_transactions AS s
    LEFT JOIN transactions AS t
        ON s.transaction_id = t.transaction_id
)
SELECT
    printf('>= %.2f', thresholds.threshold_value) AS probability_threshold,
    COUNT(scored.transaction_id) AS scored_transaction_count,
    COALESCE(
        SUM(CASE WHEN scored.fraud_probability >= thresholds.threshold_value THEN 1 ELSE 0 END),
        0
    ) AS estimated_review_volume,
    COALESCE(
        ROUND(
            100.0 * SUM(CASE WHEN scored.fraud_probability >= thresholds.threshold_value THEN 1 ELSE 0 END)
            / NULLIF(COUNT(scored.transaction_id), 0),
            2
        ),
        0
    ) AS estimated_review_rate_percent,
    COALESCE(SUM(CASE WHEN scored.risk_tier = 'High' THEN 1 ELSE 0 END), 0) AS high_risk_transaction_count,
    COALESCE(SUM(CASE WHEN scored.decision = 'Manual Review' THEN 1 ELSE 0 END), 0) AS manual_review_count,
    ROUND(
        COALESCE(
            SUM(CASE WHEN scored.fraud_probability >= thresholds.threshold_value THEN scored.amount ELSE 0 END),
            0
        ),
        2
    ) AS flagged_transaction_amount
FROM thresholds
LEFT JOIN scored
    ON 1 = 1
GROUP BY thresholds.threshold_value
ORDER BY thresholds.threshold_value;

-- Result: average model probability and workload by risk tier.
SELECT
    risk_tier,
    COUNT(*) AS scored_transaction_count,
    ROUND(AVG(fraud_probability), 4) AS average_fraud_probability,
    SUM(CASE WHEN decision = 'Manual Review' THEN 1 ELSE 0 END) AS manual_review_count
FROM scored_transactions
GROUP BY risk_tier
ORDER BY
    CASE risk_tier
        WHEN 'High' THEN 1
        WHEN 'Medium' THEN 2
        WHEN 'Low' THEN 3
        ELSE 4
    END;
