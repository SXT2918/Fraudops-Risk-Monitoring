-- FraudOps SQL Analytics
-- File: 02_hourly_fraud_patterns.sql
--
-- Business question:
-- Do certain hours of the day show elevated fraud risk or unusual transaction
-- behavior?
--
-- Why this matters:
-- Late-night or off-hour activity can be a signal for account takeover,
-- card testing, or automated abuse. Hourly analysis helps risk teams decide
-- whether review rules should be time-aware.
--
-- Tables used:
-- - transactions: cleaned labeled transactions with transaction_time, amount,
--   and is_fraud.
--
-- Output meaning:
-- Each row represents an hour of day from the transaction timestamp, with
-- transaction count, fraud count, fraud rate, total amount, and average amount.

-- Result: transaction and fraud metrics by hour of day.
SELECT
    CAST(strftime('%H', transaction_time) AS INTEGER) AS transaction_hour,
    COUNT(*) AS transaction_count,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
        2
    ) AS fraud_rate_percent,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(amount), 2) AS average_amount
FROM transactions
GROUP BY CAST(strftime('%H', transaction_time) AS INTEGER)
ORDER BY transaction_hour;
