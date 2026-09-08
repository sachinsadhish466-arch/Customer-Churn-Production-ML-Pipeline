-- ============================================================
-- Customer Churn Production ML Pipeline
-- SQL Schema
-- ============================================================

CREATE DATABASE IF NOT EXISTS customer_churn;

USE customer_churn;

-- Customer-level churn analytics table
CREATE TABLE IF NOT EXISTS customers (
    customerID VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(20),
    SeniorCitizen TINYINT,
    Partner VARCHAR(10),
    Dependents VARCHAR(10),
    tenure INT,
    PhoneService VARCHAR(10),
    MultipleLines VARCHAR(30),
    InternetService VARCHAR(30),
    OnlineSecurity VARCHAR(30),
    OnlineBackup VARCHAR(30),
    DeviceProtection VARCHAR(30),
    TechSupport VARCHAR(30),
    StreamingTV VARCHAR(30),
    StreamingMovies VARCHAR(30),
    Contract VARCHAR(30),
    PaperlessBilling VARCHAR(10),
    PaymentMethod VARCHAR(50),
    MonthlyCharges DECIMAL(10,2),
    TotalCharges DECIMAL(12,2),
    Churn VARCHAR(10)
);