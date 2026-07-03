-- FraudOps SQL Analytics
-- File: 04_merchant_risk_analysis.sql
--
-- Business question:
-- Which merchant categories have concentrated fraud risk?
--
-- Why this matters:
-- Merchant-category analysis helps fraud and risk teams decide where to focus
-- monitoring, review rules, merchant outreach, or additional model features.
--
-- Tables used:
-- - transactions: cleaned labeled transactions with merchant_category, amount,
--   and is_fraud.
--
-- Output meaning:
-- Each row is a merchant category with volume, fraud count, fraud rate, total
-- amount, fraud amount, and average transaction amount. The HAVING clause keeps
-- categories with at least two transactions so one-off categories do not
-- dominate the sample analysis.

-- Result: merchant categories ranked by fraud risk.
SELECT
    merchant_category,
    COUNT(*) AS transaction_count,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
        2
    ) AS fraud_rate_percent,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount,
    ROUND(AVG(amount), 2) AS average_amount
FROM transactions
GROUP BY merchant_category
HAVING COUNT(*) >= 2
ORDER BY fraud_rate_percent DESC, fraud_count DESC, transaction_count DESC;
