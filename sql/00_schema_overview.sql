-- FraudOps SQL Analytics
-- File: 00_schema_overview.sql
--
-- Business question:
-- What tables support the transaction monitoring workflow, and what columns
-- are available for analysis?
--
-- Why this matters:
-- Recruiters and reviewers can quickly see that FraudOps has a real data
-- storage layer behind the Python, API, model, and dashboard pieces.
--
-- Tables used:
-- - sqlite_master: SQLite metadata table used to inspect available tables.
-- - transactions: cleaned labeled transaction records from ingestion.
-- - scored_transactions: model-scored transaction decisions from the API or
--   seed scoring script.
--
-- Expected transactions structure:
--   transaction_id TEXT PRIMARY KEY
--   user_id TEXT NOT NULL
--   amount REAL NOT NULL
--   merchant_category TEXT NOT NULL
--   transaction_time TEXT NOT NULL
--   location TEXT NOT NULL
--   is_fraud INTEGER NOT NULL
--
-- Expected scored_transactions structure:
--   transaction_id TEXT PRIMARY KEY
--   fraud_probability REAL NOT NULL
--   risk_tier TEXT NOT NULL
--   decision TEXT NOT NULL
--   scored_at TEXT NOT NULL
--   FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id)
--
-- Future PostgreSQL notes:
-- - transaction_id could remain TEXT or become UUID if upstream systems use UUIDs.
-- - transaction_time and scored_at should become TIMESTAMPTZ.
-- - risk_tier and decision could be constrained with CHECK constraints or enums.
-- - indexes on transaction_time, merchant_category, risk_tier, decision, and
--   scored_at would help larger analytical workloads.

-- Result: available user tables and their CREATE statements.
SELECT
    name AS table_name,
    type AS object_type,
    sql AS create_statement
FROM sqlite_master
WHERE type = 'table'
  AND name NOT LIKE 'sqlite_%'
ORDER BY name;

-- Result: transaction table schema statement.
SELECT
    sql AS transactions_create_statement
FROM sqlite_master
WHERE type = 'table'
  AND name = 'transactions';

-- Result: scored transaction table schema statement.
SELECT
    sql AS scored_transactions_create_statement
FROM sqlite_master
WHERE type = 'table'
  AND name = 'scored_transactions';

-- Result: row counts by table.
SELECT
    'transactions' AS table_name,
    COUNT(*) AS row_count
FROM transactions
UNION ALL
SELECT
    'scored_transactions' AS table_name,
    COUNT(*) AS row_count
FROM scored_transactions;
