-- FraudOps SQL Analytics
-- File: 07_business_impact_summary.sql
--
-- Business question:
-- What is the operational and financial impact proxy of scored high-risk
-- transactions?
--
-- Why this matters:
-- Risk teams translate model scores into business terms: review workload,
-- flagged value, high-risk value, and potential prevented loss. These numbers
-- help connect analytics work to operational decision-making.
--
-- Tables used:
-- - scored_transactions: model probability, risk tier, decision, and scored_at.
-- - transactions: amount and merchant context joined by transaction_id.
--
-- Honest assumptions:
-- If true labels are not available for scored API transactions, estimated
-- prevented loss is only a proxy based on high-risk transaction amounts. It is
-- not confirmed savings and should not be claimed as production-grade fraud
-- performance.
--
-- Output meaning:
-- The first query gives a one-row business-impact proxy. The second query shows
-- risk-tier distribution. The third query shows where flagged value is
-- concentrated by merchant category.

-- Result: business impact proxy for scored transactions.
WITH scored AS (
    SELECT
        s.transaction_id,
        s.fraud_probability,
        s.risk_tier,
        s.decision,
        s.scored_at,
        COALESCE(t.amount, 0) AS amount,
        COALESCE(t.merchant_category, 'unknown') AS merchant_category
    FROM scored_transactions AS s
    LEFT JOIN transactions AS t
        ON s.transaction_id = t.transaction_id
)
SELECT
    COUNT(*) AS scored_transactions,
    COALESCE(
        SUM(CASE WHEN risk_tier = 'High' OR decision = 'Manual Review' THEN 1 ELSE 0 END),
        0
    ) AS flagged_transactions,
    ROUND(
        COALESCE(
            SUM(CASE WHEN risk_tier = 'High' OR decision = 'Manual Review' THEN amount ELSE 0 END),
            0
        ),
        2
    ) AS flagged_transaction_volume,
    ROUND(
        COALESCE(SUM(CASE WHEN risk_tier = 'High' THEN amount ELSE 0 END), 0),
        2
    ) AS high_risk_transaction_amount,
    ROUND(
        COALESCE(SUM(CASE WHEN risk_tier = 'High' THEN amount ELSE 0 END), 0),
        2
    ) AS estimated_prevented_loss_proxy,
    COALESCE(SUM(CASE WHEN decision = 'Manual Review' THEN 1 ELSE 0 END), 0) AS manual_review_workload,
    COALESCE(
        ROUND(100.0 * SUM(CASE WHEN decision = 'Manual Review' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2),
        0
    ) AS manual_review_rate_percent
FROM scored;

-- Result: risk-tier distribution for scored transactions.
SELECT
    risk_tier,
    COUNT(*) AS transaction_count,
    ROUND(AVG(fraud_probability), 4) AS average_fraud_probability,
    ROUND(SUM(COALESCE(t.amount, 0)), 2) AS transaction_amount
FROM scored_transactions AS s
LEFT JOIN transactions AS t
    ON s.transaction_id = t.transaction_id
GROUP BY risk_tier
ORDER BY
    CASE risk_tier
        WHEN 'High' THEN 1
        WHEN 'Medium' THEN 2
        WHEN 'Low' THEN 3
        ELSE 4
    END;

-- Result: flagged transaction value by merchant category.
SELECT
    COALESCE(t.merchant_category, 'unknown') AS merchant_category,
    COUNT(*) AS flagged_transactions,
    ROUND(SUM(COALESCE(t.amount, 0)), 2) AS flagged_transaction_volume,
    ROUND(AVG(s.fraud_probability), 4) AS average_fraud_probability
FROM scored_transactions AS s
LEFT JOIN transactions AS t
    ON s.transaction_id = t.transaction_id
WHERE s.risk_tier = 'High'
   OR s.decision = 'Manual Review'
GROUP BY COALESCE(t.merchant_category, 'unknown')
ORDER BY flagged_transaction_volume DESC, flagged_transactions DESC;
