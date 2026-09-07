\# Customer Churn Prediction – Production ML Pipeline



An end-to-end machine learning pipeline for predicting customer churn, designed with production-oriented practices including data validation, feature engineering, model training, experiment tracking, API serving, containerization, automated testing, and configuration management.



\---



\## 1. Project Overview



Customer churn is a major business problem for subscription-based organizations.



The objective of this project is to build a reusable machine learning pipeline that predicts whether a customer is likely to churn based on customer demographics, services, contract information, billing information, tenure, and spending behavior.



Unlike a traditional machine learning notebook, this project separates data processing, feature engineering, model prediction, API serving, testing, and configuration into reusable components.



\---



\## 2. Business Problem



Customer acquisition is often more expensive than retaining existing customers.



A churn prediction system can help businesses:



\- Identify customers with a high probability of churn

\- Prioritize retention campaigns

\- Understand important churn-related characteristics

\- Support proactive customer engagement

\- Reduce unnecessary retention costs



The model produces:



\- Churn prediction

\- Churn probability

\- Classification threshold used for the prediction



\---



\## 3. Solution



The project implements the following workflow:



```text

Customer Data

&#x20;     ↓

Data Loading

&#x20;     ↓

Data Validation

&#x20;     ↓

Data Cleaning

&#x20;     ↓

Feature Engineering

&#x20;     ↓

Train / Validation / Test

&#x20;     ↓

Machine Learning Models

&#x20;     ↓

Model Evaluation

&#x20;     ↓

Threshold Optimization

&#x20;     ↓

MLflow Experiment Tracking

&#x20;     ↓

Saved Model Artifact

&#x20;     ↓

FastAPI Prediction API

&#x20;     ↓

Streamlit Interface

&#x20;     ↓

Docker Containers



\---



## 4. Architecture

![Customer Churn Production ML Pipeline Architecture](docs/architecture.png)

The architecture represents the complete machine learning workflow from data ingestion and validation through model training, experiment tracking, API serving, Streamlit visualization, and Docker-based deployment.
