-- FraudOps SQL Analytics
-- File: 03_amount_bucket_analysis.sql
--
-- Business question:
-- Which transaction amount ranges carry the most fraud risk?
--
-- Why this matters:
-- Very small charges can indicate card testing, while larger charges can
-- represent higher loss exposure. Bucketing amounts makes risk patterns easier
-- to read than inspecting raw transaction values.
--
-- Tables used:
-- - transactions: cleaned labeled transactions with amount and is_fraud.
--
-- Output meaning:
-- Each row is an amount bucket with transaction count, fraud count, fraud rate,
-- total amount, and average amount.

-- Result: fraud risk by transaction amount bucket.
WITH bucketed_transactions AS (
    SELECT
        transaction_id,
        amount,
        is_fraud,
        CASE
            WHEN amount < 1 THEN 'Under $1'
            WHEN amount >= 1 AND amount < 10 THEN '$1-$10'
            WHEN amount >= 10 AND amount < 50 THEN '$10-$50'
            WHEN amount >= 50 AND amount < 100 THEN '$50-$100'
            WHEN amount >= 100 AND amount < 500 THEN '$100-$500'
            ELSE '$500+'
        END AS amount_bucket,
        CASE
            WHEN amount < 1 THEN 1
            WHEN amount >= 1 AND amount < 10 THEN 2
            WHEN amount >= 10 AND amount < 50 THEN 3
            WHEN amount >= 50 AND amount < 100 THEN 4
            WHEN amount >= 100 AND amount < 500 THEN 5
            ELSE 6
        END AS bucket_sort
    FROM transactions
)
SELECT
    amount_bucket,
    COUNT(*) AS transaction_count,
    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
        2
    ) AS fraud_rate_percent,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(amount), 2) AS average_amount
FROM bucketed_transactions
GROUP BY amount_bucket, bucket_sort
ORDER BY bucket_sort;
