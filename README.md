# Customer Churn Prediction – Production ML Pipeline

An end-to-end machine learning pipeline for predicting customer churn, designed with production-oriented practices including data validation, feature engineering, model training, experiment tracking, API serving, containerization, automated testing, and configuration management.

---

## 1. Project Overview

Customer churn is a major business problem for subscription-based organizations.

The objective of this project is to build a reusable machine learning pipeline that predicts whether a customer is likely to churn based on:

- Customer demographics
- Services
- Contract information
- Billing information
- Tenure
- Spending behavior

Unlike a traditional machine learning notebook, this project separates data processing, feature engineering, model prediction, API serving, testing, and configuration into reusable components.

---

## 2. Business Problem

Customer acquisition is often more expensive than retaining existing customers.

A churn prediction system can help businesses:

- Identify customers with a high probability of churn
- Prioritize retention campaigns
- Understand important churn-related characteristics
- Support proactive customer engagement
- Reduce unnecessary retention costs

### Model Output

The prediction service produces:

- Churn prediction
- Churn probability
- Classification threshold used for the prediction

---

## 3. Solution

The project implements the following end-to-end workflow:

```text
Customer Data
      ↓
Data Loading
      ↓
Data Validation
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Train / Validation / Test
      ↓
Machine Learning Models
      ↓
Model Evaluation
      ↓
Threshold Optimization
      ↓
MLflow Experiment Tracking
      ↓
Saved Model Artifact
      ↓
FastAPI Prediction API
      ↓
Streamlit Interface
      ↓
Docker Containers
```

### Architecture

<p align="center">
  <img src="docs/architecture.png" alt="Customer Churn Production ML Pipeline Architecture" width="100%">
</p>

The architecture represents the complete machine learning workflow from data ingestion and validation through model training, experiment tracking, API serving, Streamlit visualization, and Docker-based deployment.

---

## 4. Technology Stack

### Programming & Data

- Python
- Pandas
- NumPy
- SciPy
- SQL
- SQLAlchemy
- PyMySQL

### Machine Learning

- Scikit-learn
- Logistic Regression
- Random Forest
- XGBoost
- Feature Engineering
- Cross-Validation
- Hyperparameter Optimization
- Classification Threshold Optimization
- Class Imbalance Handling

### Experiment Tracking

- MLflow

### API & Application

- FastAPI
- Pydantic
- Streamlit

### Testing & CI

- Pytest
- FastAPI TestClient
- GitHub Actions

### Deployment & Containerization

- Docker
- Docker Compose

### Configuration

- YAML
- Environment Variables

---

## 5. Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer-level information including:

- Demographics
- Partner and dependent information
- Phone services
- Internet services
- Security and support services
- Streaming services
- Contract information
- Payment method
- Monthly charges
- Total charges
- Customer tenure
- Churn status

The raw dataset is intentionally excluded from Git version control through `.gitignore`.

---

## 6. Project Structure

```text
Customer-Churn-Production-ML-Pipeline/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_model_evaluation.ipynb
│
├── sql/
│   ├── 01_schema.sql
│   ├── 02_data_quality.sql
│   ├── 03_customer_analytics.sql
│   └── 04_churn_analytics.sql
│
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── data_validator.py
│   │   └── data_cleaner.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   └── model_predictor.py
│   │
│   └── utils/
│       └── config_loader.py
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── app/
│   └── app.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data_validator.py
│   ├── test_feature_engineering.py
│   ├── test_model_predictor.py
│   └── test_doubles.py
│
├── models/
│   ├── final_xgboost_pipeline.pkl
│   ├── churn_threshold.pkl
│   └── model_metadata.json
│
├── docs/
│   └── architecture.png
│
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── config.yaml
├── .env.example
├── .gitignore
└── README.md
```

> **Note:** Raw datasets and trained model binaries are excluded from Git version control. The `data/` and `models/` directories above represent the local project structure.

---

## 7. Data Pipeline

The data processing pipeline consists of the following stages.

### 7.1 Data Loading

The raw Telco Customer Churn dataset is loaded using Pandas.

### 7.2 Data Validation

The validation layer checks:

- Required columns
- Empty datasets
- Negative tenure values
- Negative monthly charges
- Negative total charges
- Required target availability when applicable

### 7.3 Data Cleaning

The cleaning process handles blank `TotalCharges` values.

Customers with zero tenure and missing `TotalCharges` are assigned:

```text
TotalCharges = 0
```

This follows the business meaning that a customer with zero tenure has not accumulated a total charge.

### 7.4 Feature Engineering

The pipeline creates:

```text
AverageMonthlySpend = TotalCharges / tenure
```

For customers with zero tenure:

```text
AverageMonthlySpend = MonthlyCharges
```

The target variable is encoded as:

```text
No  → 0
Yes → 1
```

The `customerID` column is removed before model prediction.

---

## 8. Exploratory & Statistical Analysis

The analysis evaluates relationships between customer characteristics and churn.

Statistical analysis includes:

- Monthly Charges vs Churn
- Tenure vs Churn
- Contract Type vs Churn
- Internet Service vs Churn

Statistical tests include:

- Welch's t-test
- Chi-square test
- Cramér's V

These analyses help identify customer characteristics associated with churn before model development.

---

## 9. Machine Learning Pipeline

The machine learning workflow includes:

1. Data preparation
2. Feature engineering
3. Train / validation / test splitting
4. Numerical feature preprocessing
5. Categorical feature preprocessing
6. Model training
7. Cross-validation
8. Hyperparameter optimization
9. Model evaluation
10. Threshold optimization
11. Final model selection
12. Model artifact creation

### Preprocessing

Numerical features use:

- Median imputation
- Standard scaling

Categorical features use:

- Most-frequent imputation
- One-hot encoding
- Unknown-category handling

---

## 10. Models

The project evaluates multiple classification algorithms.

### Logistic Regression

Used as a baseline classification model.

### Random Forest

Used to capture nonlinear relationships and feature interactions.

### XGBoost

Used as the final optimized model after hyperparameter tuning and evaluation.

Class imbalance is addressed during model development using appropriate imbalance-handling strategies.

---

## 11. Model Evaluation

The project evaluates classification performance using metrics appropriate for churn prediction.

The evaluation process considers:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Because churn detection is a business-sensitive classification problem, threshold optimization is also performed instead of relying blindly on the default `0.5` threshold.

The selected classification threshold is stored separately and used during inference.

---

## 12. Threshold Optimization

The model generates churn probabilities.

Instead of always using:

```text
probability >= 0.5
```

the project evaluates classification thresholds and selects a threshold based on the chosen evaluation objective.

The selected threshold is stored separately:

```text
models/churn_threshold.pkl
```

The API returns the threshold used for each prediction.

---

## 13. MLflow Experiment Tracking

MLflow is used for experiment tracking and artifact management.

Tracked information includes:

- Model parameters
- Evaluation metrics
- Classification threshold
- Model metadata
- Trained model artifact

This provides experiment reproducibility and a structured way to compare model experiments.

---

## 14. Model Artifact Management

Trained model binaries are intentionally excluded from Git version control.

The following local artifacts are generated during model development:

```text
models/
├── final_xgboost_pipeline.pkl
├── churn_threshold.pkl
└── model_metadata.json
```

The trained XGBoost pipeline contains the preprocessing and model prediction workflow required during inference.

The model artifact is loaded by the prediction service.

---

## 15. FastAPI Prediction API

The trained model is exposed through a FastAPI service.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "service": "customer-churn-prediction-api"
}
```

### Prediction

```http
POST /predict
```

The API accepts customer information and returns:

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8049,
  "prediction_label": "Likely to Churn",
  "classification_threshold": 0.5
}
```

### API Validation

Incoming requests are validated using Pydantic schemas.

Validation includes constraints such as:

- Non-negative tenure
- Non-negative monthly charges
- Non-negative total charges
- Valid `SeniorCitizen` values
- Valid prediction response ranges

---

## 16. API Model Loading

The API uses lazy model loading.

The model is not loaded when the FastAPI module is imported.

Instead, the production model is initialized when the prediction pipeline is first required.

This provides several benefits:

- Faster application startup
- Better testability
- Health checks do not require model initialization
- Tests can inject a lightweight test model
- Production requests still use the real trained model

---

## 17. Streamlit Application

A Streamlit interface provides an interactive frontend for the prediction API.

The workflow is:

```text
User Input
    ↓
Streamlit
    ↓
FastAPI
    ↓
Prediction Pipeline
    ↓
XGBoost Model
    ↓
Prediction + Probability
    ↓
Streamlit Result
```

The interface displays:

- Churn prediction
- Churn probability
- Prediction label
- Classification threshold

---

## 18. Docker Architecture

The project uses Docker Compose to run the API and Streamlit services.

```text
                    Docker Compose
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       FastAPI Container       Streamlit Container
              │                       │
              │◄──── HTTP Request ────┘
              │
              ▼
        XGBoost Pipeline
```

### FastAPI

```text
http://localhost:8000
```

### FastAPI Swagger Documentation

```text
http://localhost:8000/docs
```

### Streamlit

```text
http://localhost:8501
```

The Docker configuration uses separate containers for the API and Streamlit frontend.

---

## 19. Testing

The project includes automated tests covering:

- Data schema validation
- Data quality validation
- Feature engineering
- Average monthly spending calculation
- Model prediction
- Prediction probability
- FastAPI health endpoint
- FastAPI prediction endpoint

### Test Result

The current local test suite contains:

```text
10 tests
```

Current result:

```text
10 passed
```

The tests use dependency injection and a fake model for prediction tests, allowing CI to validate application behavior without requiring the locally generated trained model artifact.

---

## 20. Continuous Integration

GitHub Actions automatically runs the test suite when changes are pushed to the `main` branch or when pull requests target `main`.

The CI workflow is:

```text
Git Push
    ↓
GitHub Actions
    ↓
Checkout Repository
    ↓
Setup Python 3.10
    ↓
Install Dependencies
    ↓
Run Pytest
    ↓
Pass / Fail
```

This provides automated validation of code changes.

---

## 21. Configuration Management

Application configuration is maintained in:

```text
config.yaml
```

The configuration includes:

- Project metadata
- Data paths
- Model path
- Threshold path
- Metadata path
- API configuration
- Streamlit configuration

Environment-specific values can also be supplied through environment variables.

Example:

```text
.env.example
```

---

## 22. SQL Analytics

The project includes SQL scripts for data quality and customer analytics.

```text
sql/
├── 01_schema.sql
├── 02_data_quality.sql
├── 03_customer_analytics.sql
└── 04_churn_analytics.sql
```

These scripts demonstrate SQL-based data validation and analytical workflows alongside the Python machine learning pipeline.

---

## 23. Current Implementation Status

### Implemented

- End-to-end customer churn prediction pipeline
- Data loading
- Data validation
- Data cleaning
- Feature engineering
- Statistical analysis
- Logistic Regression
- Random Forest
- XGBoost
- Cross-validation
- Hyperparameter optimization
- Threshold optimization
- MLflow experiment tracking
- Model artifact management
- FastAPI prediction API
- Pydantic request validation
- Streamlit interface
- Docker
- Docker Compose
- Pytest test suite
- Dependency injection for testing
- Lazy model loading
- GitHub Actions CI
- YAML configuration management

### Future Production Extensions

The architecture can be extended toward cloud deployment using services such as:

- AWS S3 for data and model artifacts
- AWS ECR for Docker images
- AWS ECS or EKS for container deployment
- CloudWatch for monitoring and logging
- Managed relational databases

These AWS services are **future architecture extensions and are not currently deployed as part of this repository**.

---

## 24. Production Engineering Practices

This project demonstrates several production-oriented machine learning engineering practices:

- Modular Python architecture
- Separation of concerns
- Reusable data processing components
- Data validation before inference
- Feature engineering pipeline
- Configuration-driven paths
- Dependency injection
- Lazy model loading
- API request validation
- Model artifact separation
- Automated testing
- Continuous integration
- Containerized services
- Experiment tracking
- SQL-based analytics

---

## 25. Future Improvements

Potential future improvements include:

- Model monitoring
- Data drift detection
- Automated model retraining
- MLflow Model Registry
- Cloud deployment
- API authentication
- API rate limiting
- Production database integration
- Centralized logging
- Model performance monitoring
- Automated CI/CD deployment
- Infrastructure-as-Code

---

## 26. Author

**Sachin S**

Data Science | Machine Learning | Python | SQL

---

## 27. Project Links

### GitHub Repository

[Customer Churn Production ML Pipeline](https://github.com/sachinsadhish466-arch/Customer-Churn-Production-ML-Pipeline)

### Technologies

```text
Python
SQL
Pandas
NumPy
Scikit-learn
XGBoost
MLflow
FastAPI
Streamlit
Docker
GitHub Actions
```

---

## 28. License

This project is intended for educational, portfolio, and demonstration purposes.