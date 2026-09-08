-- ============================================================
-- Customer Churn Production ML Pipeline
-- Data Quality Checks
-- ============================================================

USE customer_churn;

-- 1. Check total number of customers
SELECT
    COUNT(*) AS total_customers
FROM customers;


-- 2. Check for duplicate customer IDs
SELECT
    customerID,
    COUNT(*) AS duplicate_count
FROM customers
GROUP BY customerID
HAVING COUNT(*) > 1;


-- 3. Check missing values in important numeric columns
SELECT
    SUM(customerID IS NULL) AS missing_customer_id,
    SUM(tenure IS NULL) AS missing_tenure,
    SUM(MonthlyCharges IS NULL) AS missing_monthly_charges,
    SUM(TotalCharges IS NULL) AS missing_total_charges,
    SUM(Churn IS NULL) AS missing_churn
FROM customers;


-- 4. Check invalid negative values
SELECT
    COUNT(*) AS invalid_negative_values
FROM customers
WHERE tenure < 0
   OR MonthlyCharges < 0
   OR TotalCharges < 0;


-- 5. Check valid churn categories
SELECT
    Churn,
    COUNT(*) AS customer_count
FROM customers
GROUP BY Churn
ORDER BY customer_count DESC;


-- 6. Check contract categories
SELECT
    Contract,
    COUNT(*) AS customer_count
FROM customers
GROUP BY Contract
ORDER BY customer_count DESC;


-- 7. Check internet service categories
SELECT
    InternetService,
    COUNT(*) AS customer_count
FROM customers
GROUP BY InternetService
ORDER BY customer_count DESC;