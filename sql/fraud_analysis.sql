
-- 1. Average transaction amount by fraud status
SELECT
    is_fraud,
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS average_amount,
    ROUND(MAX(amount), 2) AS maximum_amount
FROM raw_transactions
GROUP BY is_fraud
ORDER BY is_fraud;


-- 2. Fraud distribution by transaction amount
SELECT
    CASE
        WHEN amount < 10000 THEN 'Under 10K'
        WHEN amount < 50000 THEN '10K - 50K'
        WHEN amount < 100000 THEN '50K - 100K'
        ELSE '100K+'
    END AS amount_range,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(is_fraud) / COUNT(*), 4
    ) AS fraud_percentage
FROM raw_transactions
GROUP BY 1
ORDER BY MIN(amount);


-- 3. Simulated fraud activity by time step
SELECT
    step,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions
FROM raw_transactions
GROUP BY step
ORDER BY fraud_transactions DESC
LIMIT 10;
