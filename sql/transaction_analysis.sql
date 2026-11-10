-- 1. Total transactions and simulated fraud count
SELECT
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_percentage
FROM raw_transactions;


-- 2. Transactions by type
SELECT
    transaction_type,
    COUNT(*) AS transaction_count,
    ROUND(SUM(amount), 2) AS total_amount
FROM raw_transactions
GROUP BY transaction_type
ORDER BY transaction_count DESC;


-- 3. Simulated fraud by transaction type
SELECT
    transaction_type,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_percentage
FROM raw_transactions
GROUP BY transaction_type
ORDER BY fraud_transactions DESC;