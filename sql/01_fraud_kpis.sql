-- FraudOps SQL Analytics
-- File: 01_fraud_kpis.sql
--
-- Business question:
-- What is the overall fraud portfolio health of the transaction dataset?
--
-- Why this matters:
-- Fraud teams need a fast baseline view of transaction volume, fraud count,
-- fraud rate, and dollar exposure before investigating patterns by time,
-- amount, merchant category, or model threshold.
--
-- Tables used:
-- - transactions: cleaned labeled transactions from the ingestion pipeline.
--
-- Output meaning:
-- This single-row KPI summary shows transaction count, fraud count, fraud
-- rate, total transaction volume, average amount, fraud volume, average fraud
-- amount, and non-fraud volume.

-- Result: fraud portfolio health summary.
SELECT
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS total_fraud_cases,
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
        2
    ) AS fraud_rate_percent,
    ROUND(SUM(amount), 2) AS total_transaction_volume,
    ROUND(AVG(amount), 2) AS average_transaction_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_transaction_volume,
    ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END), 2) AS average_fraud_amount,
    ROUND(SUM(CASE WHEN is_fraud = 0 THEN amount ELSE 0 END), 2) AS non_fraud_transaction_volume
FROM transactions;
