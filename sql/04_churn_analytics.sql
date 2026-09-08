-- ============================================================
-- Customer Churn Production ML Pipeline
-- Churn Analytics
-- ============================================================

USE customer_churn;


-- 1. Overall churn distribution
SELECT
    Churn,
    COUNT(*) AS customer_count,
    ROUND(
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers),
        2
    ) AS percentage
FROM customers
GROUP BY Churn
ORDER BY customer_count DESC;


-- 2. Churn rate by contract type
SELECT
    Contract,
    COUNT(*) AS total_customers,
    SUM(Churn = 'Yes') AS churned_customers,
    ROUND(
        SUM(Churn = 'Yes') * 100.0 / COUNT(*),
        2
    ) AS churn_rate_percentage
FROM customers
GROUP BY Contract
ORDER BY churn_rate_percentage DESC;


-- 3. Churn rate by internet service
SELECT
    InternetService,
    COUNT(*) AS total_customers,
    SUM(Churn = 'Yes') AS churned_customers,
    ROUND(
        SUM(Churn = 'Yes') * 100.0 / COUNT(*),
        2
    ) AS churn_rate_percentage
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_percentage DESC;


-- 4. Churn rate by payment method
SELECT
    PaymentMethod,
    COUNT(*) AS total_customers,
    SUM(Churn = 'Yes') AS churned_customers,
    ROUND(
        SUM(Churn = 'Yes') * 100.0 / COUNT(*),
        2
    ) AS churn_rate_percentage
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_percentage DESC;


-- 5. Churn rate by tenure group
SELECT
    CASE
        WHEN tenure <= 6 THEN '0-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_group,
    COUNT(*) AS total_customers,
    SUM(Churn = 'Yes') AS churned_customers,
    ROUND(
        SUM(Churn = 'Yes') * 100.0 / COUNT(*),
        2
    ) AS churn_rate_percentage
FROM customers
GROUP BY tenure_group
ORDER BY
    MIN(tenure);


-- 6. Average monthly charges by churn status
SELECT
    Churn,
    COUNT(*) AS customer_count,
    ROUND(AVG(MonthlyCharges), 2) AS average_monthly_charges,
    ROUND(AVG(TotalCharges), 2) AS average_total_charges
FROM customers
GROUP BY Churn
ORDER BY Churn;


-- 7. High-value churned customers
SELECT
    customerID,
    tenure,
    Contract,
    InternetService,
    MonthlyCharges,
    TotalCharges,
    Churn
FROM customers
WHERE Churn = 'Yes'
  AND MonthlyCharges >= 80
ORDER BY MonthlyCharges DESC;


-- 8. Month-to-month customers at higher churn risk
SELECT
    customerID,
    tenure,
    InternetService,
    MonthlyCharges,
    TotalCharges,
    Churn
FROM customers
WHERE Contract = 'Month-to-month'
  AND Churn = 'Yes'
ORDER BY MonthlyCharges DESC;