-- ============================================================
-- Customer Churn Production ML Pipeline
-- Customer Analytics
-- ============================================================

USE customer_churn;


-- 1. Customer distribution by gender
SELECT
    gender,
    COUNT(*) AS customer_count
FROM customers
GROUP BY gender
ORDER BY customer_count DESC;


-- 2. Customer distribution by senior citizen status
SELECT
    SeniorCitizen,
    COUNT(*) AS customer_count
FROM customers
GROUP BY SeniorCitizen
ORDER BY SeniorCitizen;


-- 3. Average monthly charges by contract type
SELECT
    Contract,
    COUNT(*) AS customer_count,
    ROUND(AVG(MonthlyCharges), 2) AS average_monthly_charges
FROM customers
GROUP BY Contract
ORDER BY average_monthly_charges DESC;


-- 4. Average tenure by contract type
SELECT
    Contract,
    ROUND(AVG(tenure), 2) AS average_tenure_months
FROM customers
GROUP BY Contract
ORDER BY average_tenure_months DESC;


-- 5. Customer distribution by payment method
SELECT
    PaymentMethod,
    COUNT(*) AS customer_count
FROM customers
GROUP BY PaymentMethod
ORDER BY customer_count DESC;


-- 6. Average monthly charges by internet service
SELECT
    InternetService,
    COUNT(*) AS customer_count,
    ROUND(AVG(MonthlyCharges), 2) AS average_monthly_charges
FROM customers
GROUP BY InternetService
ORDER BY average_monthly_charges DESC;


-- 7. Customers with high monthly charges
SELECT
    customerID,
    Contract,
    InternetService,
    tenure,
    MonthlyCharges,
    TotalCharges,
    Churn
FROM customers
WHERE MonthlyCharges >= 80
ORDER BY MonthlyCharges DESC;


-- 8. Customers with low tenure
SELECT
    customerID,
    Contract,
    tenure,
    MonthlyCharges,
    TotalCharges,
    Churn
FROM customers
WHERE tenure <= 6
ORDER BY tenure, MonthlyCharges DESC;