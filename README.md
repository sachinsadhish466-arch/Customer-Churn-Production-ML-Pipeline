# Customer Churn Prediction â€“ Production ML Pipeline



<p align="center">



### End-to-End Machine Learning Pipeline with FastAPI, Docker, AWS ECS Fargate, Monitoring & Drift Detection



</p>



<p align="center">



<strong>From Customer Data â†’ Machine Learning â†’ API â†’ Docker â†’ AWS â†’ Monitoring</strong>



</p>



\---



## 1. Project Overview



Customer churn prediction is a binary classification problem where the objective is to identify customers who are likely to discontinue a service.



This project goes beyond a traditional machine learning notebook by implementing a reusable, production-oriented machine learning system covering:



\* Data loading

\* Data validation

\* Data cleaning

\* Exploratory Data Analysis

\* Statistical analysis

\* Feature engineering

\* Machine learning model development

\* Cross-validation

\* Hyperparameter tuning

\* Classification threshold evaluation

\* MLflow experiment tracking

\* Model artifact management

\* FastAPI inference API

\* Request ID tracing

\* Structured application logging

\* Prediction monitoring

\* Feature drift monitoring

\* Automated testing

\* Docker

\* Docker Compose

\* GitHub Actions

\* AWS ECR

\* AWS ECS Fargate

\* AWS IAM

\* AWS CloudWatch Logs



> \*\*Project Positioning:\*\* This is a production-oriented portfolio implementation. It demonstrates practical ML engineering and cloud deployment patterns but should not be represented as a fully enterprise production-grade system.



\---



## 2. Business Problem



Customer acquisition is often more expensive than retaining existing customers.



A churn prediction system can help businesses:



\* Identify customers at high risk of churn

\* Prioritize retention campaigns

\* Support proactive customer engagement

\* Identify churn-related customer characteristics

\* Improve customer retention strategies

\* Reduce unnecessary retention costs

\* Support data-driven customer success decisions



### Model Output



The prediction service produces:



\* Churn prediction

\* Churn probability

\* Prediction label

\* Classification threshold used for the prediction



\---



## 3. Solution



The project implements a complete machine learning lifecycle:



```text

Customer Data

&#x20;     â†“

Data Loading

&#x20;     â†“

Data Validation

&#x20;     â†“

Data Cleaning

&#x20;     â†“

Exploratory Data Analysis

&#x20;     â†“

Statistical Analysis

&#x20;     â†“

Feature Engineering

&#x20;     â†“

Train / Validation / Test

&#x20;     â†“

Model Development

&#x20;     â†“

Cross Validation

&#x20;     â†“

Hyperparameter Tuning

&#x20;     â†“

Model Evaluation

&#x20;     â†“

Classification Threshold Evaluation

&#x20;     â†“

Final XGBoost Model

&#x20;     â†“

MLflow Experiment Tracking

&#x20;     â†“

Model Artifact

&#x20;     â†“

FastAPI

&#x20;     â†“

Automated Testing

&#x20;     â†“

Docker

&#x20;     â†“

Amazon ECR

&#x20;     â†“

Amazon ECS Fargate

&#x20;     â†“

CloudWatch Logs

&#x20;     â†“

Prediction Monitoring

&#x20;     â†“

Feature Drift Detection

```



\---



## 4. Complete Architecture



<p align="center">

&#x20; <img src="docs/architecture.png" alt="Customer Churn Production ML Pipeline Complete Architecture" width="100%">

</p>



### End-to-End Architecture



```text

&#x20;                        CUSTOMER DATA

&#x20;                             â”‚

&#x20;                             â–¼

&#x20;                 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                 â”‚   Raw Telco Dataset    â”‚

&#x20;                 â”‚      CSV / SQL          â”‚

&#x20;                 â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                              â”‚

&#x20;                              â–¼

&#x20;                 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                 â”‚    Data Loading         â”‚

&#x20;                 â”‚       Pandas            â”‚

&#x20;                 â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                              â”‚

&#x20;                              â–¼

&#x20;                 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                 â”‚    Data Validation      â”‚

&#x20;                 â”‚ Schema + Quality Checks â”‚

&#x20;                 â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                              â”‚

&#x20;                              â–¼

&#x20;                 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                 â”‚     Data Cleaning       â”‚

&#x20;                 â”‚ Missing Value Handling  â”‚

&#x20;                 â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                              â”‚

&#x20;                              â–¼

&#x20;                 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                 â”‚  Feature Engineering    â”‚

&#x20;                 â”‚ AverageMonthlySpend     â”‚

&#x20;                 â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                              â”‚

&#x20;                              â–¼

&#x20;                 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                 â”‚ Train / Validation /    â”‚

&#x20;                 â”‚       Test Split        â”‚

&#x20;                 â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                              â”‚

&#x20;                              â–¼

&#x20;             â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;             â”‚      MODEL DEVELOPMENT           â”‚

&#x20;             â”‚                                  â”‚

&#x20;             â”‚  Logistic Regression             â”‚

&#x20;             â”‚  Random Forest                   â”‚

&#x20;             â”‚  XGBoost                         â”‚

&#x20;             â”‚                                  â”‚

&#x20;             â”‚  Cross Validation                â”‚

&#x20;             â”‚  Hyperparameter Tuning           â”‚

&#x20;             â”‚  Model Evaluation                â”‚

&#x20;             â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                             â”‚

&#x20;                             â–¼

&#x20;                   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                   â”‚  Final XGBoost    â”‚

&#x20;                   â”‚      Model        â”‚

&#x20;                   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                             â”‚

&#x20;                             â–¼

&#x20;                   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                   â”‚      MLflow       â”‚

&#x20;                   â”‚ Experiment        â”‚

&#x20;                   â”‚ Tracking          â”‚

&#x20;                   â”‚ Model Artifacts   â”‚

&#x20;                   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                             â”‚

&#x20;                             â–¼

&#x20;                   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                   â”‚ Saved Model       â”‚

&#x20;                   â”‚ + Threshold       â”‚

&#x20;                   â”‚ + Metadata        â”‚

&#x20;                   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                             â”‚

&#x20;                             â–¼

&#x20;                   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                   â”‚      Docker       â”‚

&#x20;                   â”‚ FastAPI + Model   â”‚

&#x20;                   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                             â”‚

&#x20;                             â–¼

&#x20;                   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                   â”‚    Amazon ECR     â”‚

&#x20;                   â”‚ Container Registryâ”‚

&#x20;                   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                             â”‚

&#x20;                             â–¼

&#x20;                   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;                   â”‚  Amazon ECS       â”‚

&#x20;                   â”‚     Fargate       â”‚

&#x20;                   â”‚                   â”‚

&#x20;                   â”‚ FastAPI Container â”‚

&#x20;                   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                             â”‚

&#x20;               â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;               â”‚             â”‚             â”‚

&#x20;               â–¼             â–¼             â–¼

&#x20;         CloudWatch      Monitoring     Drift

&#x20;            Logs         Endpoint      Detection

&#x20;               â”‚             â”‚             â”‚

&#x20;               â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;                             â”‚

&#x20;                             â–¼

&#x20;                   Customer Prediction

```



\---



## 5. Technology Stack



### 5.1 Programming & Data



\* Python 3.10

\* Pandas

\* NumPy

\* SciPy

\* SQL

\* SQLAlchemy

\* PyMySQL



### 5.2 Machine Learning



\* Scikit-learn

\* Logistic Regression

\* Random Forest

\* XGBoost

\* Feature Engineering

\* Cross-Validation

\* RandomizedSearchCV

\* Hyperparameter Optimization

\* Class Imbalance Handling

\* Classification Threshold Evaluation



### 5.3 Experiment Tracking



\* MLflow



### 5.4 API & Application



\* FastAPI

\* Pydantic

\* Uvicorn

\* Streamlit



### 5.5 Testing



\* Pytest

\* FastAPI TestClient

\* Dependency Injection

\* Test Doubles / Fake Models



### 5.6 Containerization



\* Docker

\* Docker Compose



### 5.7 Cloud



\* Amazon ECR

\* Amazon ECS

\* AWS Fargate

\* AWS IAM

\* Amazon CloudWatch Logs



### 5.8 Version Control & CI



\* Git

\* GitHub

\* GitHub Actions



### 5.9 Configuration



\* YAML

\* Environment Variables



\---



## 6. Dataset



The project uses the IBM Telco Customer Churn dataset.



The dataset contains customer-level information including:



\* Customer demographics

\* Partner and dependent information

\* Phone services

\* Internet services

\* Security services

\* Support services

\* Streaming services

\* Contract information

\* Payment method

\* Monthly charges

\* Total charges

\* Customer tenure

\* Churn status



### Important Features



```text

gender

SeniorCitizen

Partner

Dependents

tenure

PhoneService

MultipleLines

InternetService

OnlineSecurity

OnlineBackup

DeviceProtection

TechSupport

StreamingTV

StreamingMovies

Contract

PaperlessBilling

PaymentMethod

MonthlyCharges

TotalCharges

Churn

```



The raw dataset is excluded from Git version control through `.gitignore`.



\---



## 7. Data Validation & Cleaning



### 7.1 Data Validation



The reusable validation layer checks:



\* Required columns

\* Empty datasets

\* Missing target column when required

\* Negative tenure values

\* Negative monthly charges

\* Negative total charges



Example validation flow:



```text

Input Data

&#x20;   â†“

Schema Validation

&#x20;   â†“

Data Quality Validation

&#x20;   â†“

Validated Dataset

```



### 7.2 Data Cleaning



The original dataset contains blank values in `TotalCharges`.



Investigation showed that the affected customers have:



```text

tenure = 0

```



Therefore, the project handles those records using:



```text

TotalCharges = 0

```



This is based on the business interpretation that customers with zero tenure have not accumulated total charges.



The implementation does not use arbitrary mean or median replacement for these specific records.



\---



## 8. Exploratory Data Analysis



The project performs exploratory analysis to understand customer behavior and churn patterns.



Areas analyzed include:



\* Monthly charges

\* Total charges

\* Tenure

\* Contract type

\* Internet service

\* Customer demographics

\* Customer services

\* Churn distribution



\---



## 9. Statistical Analysis



Statistical analysis was performed before model development.



The project evaluates:



### 9.1 Monthly Charges vs Churn



Welch's independent t-test.



### 9.2 Tenure vs Churn



Welch's independent t-test.



### 9.3 Contract vs Churn



Chi-square test and CramÃ©r's V.



### 9.4 Internet Service vs Churn



Chi-square test and CramÃ©r's V.



These analyses help identify relationships between customer characteristics and churn before building predictive models.



\---



## 10. Feature Engineering



The project creates a derived feature:



### AverageMonthlySpend



```text

AverageMonthlySpend =

&#x20;   TotalCharges / tenure

```



For customers with zero tenure:



```text

AverageMonthlySpend = MonthlyCharges

```



The target is encoded as:



```text

No  â†’ 0

Yes â†’ 1

```



The `customerID` column is removed before model prediction because it is an identifier rather than a meaningful predictive feature.



\---



## 11. Machine Learning Pipeline



The machine learning workflow includes:



```text

Data Preparation

&#x20;     â†“

Feature Engineering

&#x20;     â†“

Train / Validation / Test Split

&#x20;     â†“

Preprocessing

&#x20;     â†“

Model Training

&#x20;     â†“

Stratified Cross Validation

&#x20;     â†“

Hyperparameter Optimization

&#x20;     â†“

Model Evaluation

&#x20;     â†“

Threshold Evaluation

&#x20;     â†“

Final Model Selection

```



\---



## 12. Preprocessing Pipeline



### 12.1 Numerical Features



The numerical preprocessing pipeline uses:



```text

Median Imputation

&#x20;      â†“

StandardScaler

```



### 12.2 Categorical Features



The categorical preprocessing pipeline uses:



```text

Most-Frequent Imputation

&#x20;      â†“

OneHotEncoder

&#x20;      â†“

handle\_unknown="ignore"

```



The preprocessing and estimator are combined into a Scikit-learn pipeline to ensure consistent transformations during training and inference.



\---



## 13. Class Imbalance Handling



Customer churn datasets commonly contain fewer churned customers than non-churned customers.



The project accounts for class imbalance using model-specific strategies.



Examples include:



```text

class\_weight

```



and for XGBoost:



```text

scale\_pos\_weight

```



This helps the models pay appropriate attention to the churn class.



\---



## 14. Models Evaluated



### 14.1 Logistic Regression



Used as a baseline classification model.



Advantages:



\* Simple

\* Interpretable

\* Fast

\* Strong baseline for binary classification



### 14.2 Random Forest



Used to capture:



\* Nonlinear relationships

\* Feature interactions

\* Complex decision boundaries



### 14.3 XGBoost



XGBoost was selected as the final model after model comparison and hyperparameter optimization.



The final trained artifact is:



```text

models/final\_xgboost\_pipeline.pkl

```



\---



## 15. Cross Validation & Hyperparameter Optimization



The project uses stratified cross-validation to maintain class proportions across folds.



The XGBoost model is optimized using randomized hyperparameter search.



The process evaluates combinations of model parameters using cross-validation and selects a strong configuration based on ROC-AUC performance.



\---



## 16. Final Model Performance



The final XGBoost model achieved:



| Metric                   |     Result |

| ------------------------ | ---------: |

| Cross-Validation ROC-AUC | \*\*0.8492\*\* |

| Test ROC-AUC             | \*\*0.8478\*\* |

| Test Accuracy            | \*\*75.94%\*\* |

| Test Precision           | \*\*53.08%\*\* |

| Test Recall              | \*\*80.75%\*\* |

| Test F1 Score            | \*\*64.05%\*\* |

| Classification Threshold |   \*\*0.50\*\* |



### Interpretation



The model achieves a test ROC-AUC of approximately:



```text

0.848

```



and a test recall of:



```text

80.75%

```



The relatively high recall is useful in a churn-retention scenario where missing potential churners can be costly.



\---



## 17. Classification Threshold



The model generates a probability of churn.



The project evaluates classification thresholds rather than blindly relying on the default threshold.



The persisted final threshold is:



```text

0.50

```



The threshold is saved separately:



```text

models/churn\_threshold.pkl

```



During inference:



```text

Probability >= Threshold

&#x20;       â†“

&#x20;  Churn = 1

```



otherwise:



```text

Probability < Threshold

&#x20;       â†“

&#x20;  Churn = 0

```



The API also returns the threshold used for the prediction.



\---



## 18. MLflow Experiment Tracking



MLflow is used for local experiment tracking and artifact management.



Tracked information includes:



\* Model parameters

\* Evaluation metrics

\* Classification threshold

\* Model metadata

\* Tested model artifact



The MLflow workflow provides a structured way to track experiments and model-related artifacts.



### Important Scope



This project implements:



```text

Local MLflow Experiment Tracking

\+

Artifact Management

```



It does \*\*not\*\* claim to implement:



```text

Centralized MLflow Model Registry

```



or enterprise model governance.



\---



## 19. Model Artifacts



The following model artifacts are generated locally:



```text

models/

â”œâ”€â”€ final\_xgboost\_pipeline.pkl

â”œâ”€â”€ churn\_threshold.pkl

â””â”€â”€ model\_metadata.json

```



### Model Metadata



The metadata records:



\* Model type

\* Classification threshold

\* Cross-validation ROC-AUC

\* Test accuracy

\* Test precision

\* Test recall

\* Test F1

\* Test ROC-AUC



Model binary files are intentionally excluded from Git version control.



\---



## 20. FastAPI Prediction API



The trained XGBoost pipeline is exposed through FastAPI.



The API provides:



```text

GET  /health

GET  /ready

POST /predict

GET  /monitoring

GET  /drift

```



\---



## 21. API Endpoints



### 21.1 Health Endpoint



```http

GET /health

```



Example response:



```json

{

&#x20; "status": "healthy",

&#x20; "service": "customer-churn-prediction-api"

}

```



This verifies that the application is responding.



### 21.2 Readiness Endpoint



```http

GET /ready

```



The readiness endpoint verifies:



```text

Model Loaded

Threshold Loaded

Prediction Pipeline Available

```



Example:



```json

{

&#x20; "status": "ready",

&#x20; "service": "customer-churn-prediction-api",

&#x20; "model\_loaded": true,

&#x20; "threshold\_loaded": true

}

```



This endpoint is also used by the Docker/ECS health-check configuration.



### 21.3 Prediction Endpoint



```http

POST /predict

```



The API accepts customer information and returns:



```json

{

&#x20; "churn\_prediction": 1,

&#x20; "churn\_probability": 0.7654,

&#x20; "prediction\_label": "Likely to Churn",

&#x20; "classification\_threshold": 0.5

}

```



\---



## 22. API Request Validation



Incoming requests are validated using Pydantic.



Validation includes:



\* Non-negative tenure

\* Non-negative monthly charges

\* Non-negative total charges

\* `SeniorCitizen` constrained to valid values

\* Probability range validation

\* Prediction range validation

\* Threshold range validation



Invalid requests are rejected before entering the prediction pipeline.



\---



## 23. Request ID Tracing



Each API request receives a unique request ID.



The request ID is:



\* Generated at request entry

\* Stored in the request state

\* Included in application logs

\* Returned through the `X-Request-ID` response header

\* Included in relevant error responses



Example:



```text

Request started

&#x20;     â†“

request\_id generated

&#x20;     â†“

Prediction

&#x20;     â†“

Request completed

&#x20;     â†“

X-Request-ID returned

```



This provides a basic mechanism for tracing individual requests through the application.



\---



## 24. Structured Application Logging



The application uses structured logging for important service events.



Logged events include:



```text

Request started

Request completed

Request failed

Health check requested

Readiness check requested

Prediction request received

Prediction completed

Monitoring metrics requested

Drift monitoring requested

```



Prediction logs include information such as:



```text

request\_id

prediction

probability

classification\_threshold

prediction\_label

latency\_ms

```



\---



## 25. Lazy Model Loading



The prediction model uses lazy loading.



The model is not loaded immediately when the FastAPI module is imported.



Instead:



```text

FastAPI starts

&#x20;     â†“

Application available

&#x20;     â†“

Prediction pipeline requested

&#x20;     â†“

Model initialized

```



Benefits include:



\* Better testability

\* Cleaner application startup

\* Reduced unnecessary initialization

\* Easier dependency injection

\* Ability to inject lightweight test models



\---



## 26. Prediction Monitoring



The API logs prediction events in JSON Lines format.



Each prediction record contains:



```text

timestamp

request\_id

prediction

probability

classification\_threshold

prediction\_label

latency\_ms

```



The monitoring endpoint:



```http

GET /monitoring

```



calculates:



\* Total predictions

\* Churn predictions

\* Stay predictions

\* Churn prediction rate

\* Average churn probability

\* High-risk predictions

\* High-risk prediction rate

\* Average prediction latency



Example:



```json

{

&#x20; "total\_predictions": 1,

&#x20; "churn\_predictions": 1,

&#x20; "stay\_predictions": 0,

&#x20; "churn\_prediction\_rate": 1.0,

&#x20; "average\_churn\_probability": 0.7654,

&#x20; "high\_risk\_predictions": 1,

&#x20; "high\_risk\_prediction\_rate": 1.0,

&#x20; "average\_latency\_ms": 22.6

}

```



\---



## 27. Feature Drift Monitoring



The project includes a lightweight feature drift monitoring implementation.



Reference statistics are maintained for:



```text

tenure

MonthlyCharges

TotalCharges

AverageMonthlySpend

```



The monitoring process compares reference statistics against a current production-like dataset.



\---



## 28. Drift Calculation



The current implementation measures relative change in feature means:



```text

relative\_change =

&#x20;   |current\_mean - reference\_mean|

&#x20;   / |reference\_mean|

```



A feature is marked as drifted when:



```text

relative\_change >= 0.20

```



The configured threshold is:



```text

20%

```



\---



## 29. Simulated Production Batch



A simulated production batch is included to demonstrate drift detection.



The batch intentionally shifts charge-related customer behavior.



The verified monitoring result detected drift in:



```text

MonthlyCharges

TotalCharges

AverageMonthlySpend

```



while:



```text

tenure

```



remained below the configured drift threshold.



Example response:



```json

{

&#x20; "overall\_drift\_detected": true,

&#x20; "total\_features\_checked": 4,

&#x20; "drifted\_feature\_count": 3,

&#x20; "drifted\_features": \[

&#x20;   "MonthlyCharges",

&#x20;   "TotalCharges",

&#x20;   "AverageMonthlySpend"

&#x20; ]

}

```



> \*\*Important:\*\* This is a lightweight portfolio implementation of drift monitoring. It is not intended to replace a full statistical monitoring platform.



\---



## 30. Streamlit Interface



A Streamlit application provides an interactive frontend for the prediction API.



The workflow is:



```text

User Input

&#x20;   â†“

Streamlit

&#x20;   â†“

FastAPI

&#x20;   â†“

Prediction Pipeline

&#x20;   â†“

XGBoost

&#x20;   â†“

Prediction + Probability

&#x20;   â†“

Streamlit Result

```



The interface displays:



\* Churn prediction

\* Churn probability

\* Prediction label

\* Classification threshold



The Streamlit interface is designed for local/containerized demonstration.



The AWS deployment described below focuses on the FastAPI inference service.



\---



## 31. Docker Architecture



The project contains separate Docker configurations for the API and Streamlit interface.



```text

&#x20;                Docker Compose

&#x20;                      â”‚

&#x20;           â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;           â”‚                     â”‚

&#x20;           â–¼                     â–¼

&#x20;    FastAPI Container      Streamlit Container

&#x20;       Port 8000              Port 8501

&#x20;           â”‚                     â”‚

&#x20;           â”‚â—„â”€â”€â”€â”€ HTTP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

&#x20;           â”‚

&#x20;           â–¼

&#x20;      XGBoost Pipeline

```



\---



## 32. Docker Containers



### 32.1 FastAPI Docker Container



The API is containerized using:



```text

Dockerfile

```



The container:



\* Installs Python dependencies

\* Copies API code

\* Copies source code

\* Copies model artifacts

\* Copies configuration

\* Starts Uvicorn

\* Exposes port `8000`



### 32.2 Streamlit Docker Container



The Streamlit frontend uses:



```text

Dockerfile.streamlit

```



The container exposes:



```text

8501

```



### 32.3 Docker Compose



The complete local application can be started using:



```cmd

docker compose up --build

```



API:



```text

http://localhost:8000

```



Swagger:



```text

http://localhost:8000/docs

```



Streamlit:



```text

http://localhost:8501

```



Stop the application:



```cmd

docker compose down

```



\---



## 33. AWS Deployment



The FastAPI application has been \*\*actually deployed and verified on AWS\*\*.



The deployment architecture uses:



```text

Docker

&#x20;  â†“

Amazon ECR

&#x20;  â†“

Amazon ECS

&#x20;  â†“

AWS Fargate

&#x20;  â†“

CloudWatch Logs

```



AWS region:



```text

ap-south-1

```



\---



## 34. AWS Services Used



| AWS Service     | Purpose                        |

| --------------- | ------------------------------ |

| Amazon ECR      | Container image registry       |

| Amazon ECS      | Container orchestration        |

| AWS Fargate     | Serverless container execution |

| IAM             | ECS task execution permissions |

| CloudWatch Logs | Application/container logging  |



\---



## 35. Amazon ECR



The Docker image is stored in an Amazon ECR repository:



```text

customer-churn-production-ml-pipeline

```



The image is pushed to ECR and pulled by the ECS task.



Architecture:



```text

Local Docker Image

&#x20;      â†“

docker push

&#x20;      â†“

Amazon ECR

&#x20;      â†“

ECS pulls image

```



ECR image scanning is enabled for the repository.



\---



## 36. Amazon ECS Fargate



The FastAPI application runs as an ECS Fargate task.



Configuration:



```text

Launch Type:

FARGATE



CPU:

512



Memory:

1024 MB



Container Port:

8000

```



The ECS service was successfully verified with:



```text

Desired Count:       1

Running Count:       1

Pending Count:       0

Deployment Status:   COMPLETED

Service Status:      ACTIVE

```



\---



## 37. IAM



The ECS task uses an ECS task execution role.



The execution role allows the ECS task to perform required container execution operations such as:



```text

Pull image from ECR

Write container logs to CloudWatch

```



The project does not store AWS credentials in the repository.



\---



## 38. CloudWatch Logs



Application/container logs are sent to:



```text

/ecs/customer-churn-api

```



CloudWatch was verified to contain:



\* Uvicorn startup logs

\* Request logs

\* Health-check logs

\* Prediction logs

\* Monitoring logs

\* Drift monitoring logs



Example application event:



```text

Prediction completed

request\_id=<request-id>

prediction=1

probability=0.7654

latency\_ms=...

label=Likely to Churn

```



\---



## 39. AWS Network Architecture



The current demonstration deployment exposes the ECS service through a public IP.



```text

Internet

&#x20;   â”‚

&#x20;   â–¼

Public IP

&#x20;   â”‚

&#x20;   â–¼

ECS Fargate Task

&#x20;   â”‚

&#x20;   â–¼

FastAPI

&#x20;   â”‚

&#x20;   â–¼

XGBoost Model

```



### Important limitation



The Fargate public IP is \*\*ephemeral\*\*.



If the ECS task is replaced or restarted, the public IP can change.



The current architecture does not yet include:



```text

Application Load Balancer

HTTPS / TLS

Custom Domain

Route 53

```



This is intentionally documented as a learning/demo deployment rather than a fully hardened public production architecture.



\---



## 40. AWS API Verification



The deployed FastAPI service was successfully verified.



### 40.1 Health Verification



Endpoint:



```text

GET /health

```



Result:



```text

healthy

```



### 40.2 Readiness Verification



Endpoint:



```text

GET /ready

```



Result:



```text

ready

model\_loaded = true

threshold\_loaded = true

```



### 40.3 Prediction Verification



Endpoint:



```text

POST /predict

```



Verified response:



```json

{

&#x20; "churn\_prediction": 1,

&#x20; "churn\_probability": 0.7654,

&#x20; "prediction\_label": "Likely to Churn",

&#x20; "classification\_threshold": 0.5

}

```



### 40.4 Monitoring Verification



Endpoint:



```text

GET /monitoring

```



Successfully returned prediction metrics.



### 40.5 Drift Verification



Endpoint:



```text

GET /drift

```



Successfully detected intentionally shifted production-like features.



\---



## 41. Production-Oriented Monitoring Architecture



```text

&#x20;               API Request

&#x20;                    â”‚

&#x20;                    â–¼

&#x20;              FastAPI API

&#x20;                    â”‚

&#x20;         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;         â”‚                     â”‚

&#x20;         â–¼                     â–¼

&#x20;    Prediction            Request Log

&#x20;         â”‚

&#x20;         â–¼

&#x20;Prediction JSONL

&#x20;         â”‚

&#x20;         â–¼

&#x20;   /monitoring

&#x20;         â”‚

&#x20;         â–¼

Operational Metrics

```



\---



## 42. Drift Monitoring Architecture



```text

Reference Dataset

&#x20;      â”‚

&#x20;      â–¼

Reference Statistics

&#x20;      â”‚

&#x20;      â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

&#x20;      â”‚               â”‚

&#x20;      â–¼               â–¼

Reference Mean    Current Mean

&#x20;                      â–²

&#x20;                      â”‚

&#x20;            Production-like Data

&#x20;                      â”‚

&#x20;                      â–¼

&#x20;               Drift Calculation

&#x20;                      â”‚

&#x20;                      â–¼

&#x20;                 Drift Report

&#x20;                      â”‚

&#x20;                      â–¼

&#x20;                   /drift

```



\---



## 43. Automated Testing



The project includes automated tests covering:



\* API health endpoint

\* API readiness endpoint

\* API prediction endpoint

\* API monitoring endpoint

\* API drift endpoint

\* Request ID behavior

\* Data schema validation

\* Data quality validation

\* Feature engineering

\* Average monthly spending calculation

\* Model prediction

\* Prediction probability

\* Prediction monitoring

\* Reference statistics

\* Drift calculations

\* Drift report generation



\---



## 44. Final Test Result



The final verified test suite contains:



```text

26 tests

```



Result:



```text

26 passed

```



The tests use dependency injection and test doubles/fake models where appropriate, allowing application behavior to be tested without depending entirely on the production model artifact.



\---



## 45. CI / GitHub Actions



The repository includes GitHub Actions workflows.



The CI workflow validates the project using Python 3.10 and Pytest.



The AWS-oriented workflow can:



```text

Checkout Repository

&#x20;       â†“

Setup Python 3.10

&#x20;       â†“

Install Dependencies

&#x20;       â†“

Run Tests

&#x20;       â†“

Build Docker Image

&#x20;       â†“

Verify Docker Image

```



The AWS deployment workflow is currently a \*\*manual validation/build workflow\*\*.



It does not automatically deploy every GitHub commit to AWS.



\---



## 46. Configuration Management



Application configuration is maintained in:



```text

config.yaml

```



Configuration includes areas such as:



\* Project metadata

\* Data paths

\* Model paths

\* Threshold paths

\* Metadata paths

\* API configuration

\* Monitoring configuration

\* Drift threshold

\* Reference statistics path

\* Current monitoring data path



Environment-specific values can be provided through environment variables where applicable.



Sensitive credentials are not stored in the repository.



\---



## 47. SQL Analytics



The repository includes SQL scripts for analytical and data-quality workflows:



```text

sql/

â”œâ”€â”€ 01\_schema.sql

â”œâ”€â”€ 02\_data\_quality.sql

â”œâ”€â”€ 03\_customer\_analytics.sql

â””â”€â”€ 04\_churn\_analytics.sql

```



These demonstrate SQL-based:



\* Schema creation

\* Data-quality checks

\* Customer analytics

\* Churn analytics



The current trained/inference pipeline uses the processed Telco dataset and does not claim that the AWS prediction service is directly connected to a production MySQL database.



\---



## 48. Project Structure



```text

Customer-Churn-Production-ML-Pipeline/

â”‚

â”œâ”€â”€ .github/

â”‚   â””â”€â”€ workflows/

â”‚       â”œâ”€â”€ aws-deploy.yml

â”‚       â””â”€â”€ ci.yml

â”‚

â”œâ”€â”€ api/

â”‚   â”œâ”€â”€ main.py

â”‚   â””â”€â”€ schemas.py

â”‚

â”œâ”€â”€ app/

â”‚   â””â”€â”€ app.py

â”‚

â”œâ”€â”€ aws/

â”‚   â”œâ”€â”€ README.md

â”‚   â”œâ”€â”€ deployment-config.yaml

â”‚   â”œâ”€â”€ ecs-task-definition.json

â”‚   â””â”€â”€ ecs-trust-policy.json

â”‚

â”œâ”€â”€ data/

â”‚   â”œâ”€â”€ raw/

â”‚   â”œâ”€â”€ processed/

â”‚   â””â”€â”€ monitoring/

â”‚       â””â”€â”€ simulated\_production\_batch.csv

â”‚

â”œâ”€â”€ docs/

â”‚   â”œâ”€â”€ architecture.png

â”‚   â””â”€â”€ aws\_architecture.md

â”‚

â”œâ”€â”€ notebooks/

â”‚   â”œâ”€â”€ 01\_data\_understanding.ipynb

â”‚   â”œâ”€â”€ 02\_data\_cleaning\_eda.ipynb

â”‚   â”œâ”€â”€ 03\_statistical\_analysis.ipynb

â”‚   â”œâ”€â”€ 04\_feature\_engineering.ipynb

â”‚   â”œâ”€â”€ 05\_Machine\_Learning.ipynb

â”‚   â””â”€â”€ 06\_mlflow\_experiment\_tracking.ipynb

â”‚

â”œâ”€â”€ sql/

â”‚   â”œâ”€â”€ 01\_schema.sql

â”‚   â”œâ”€â”€ 02\_data\_quality.sql

â”‚   â”œâ”€â”€ 03\_customer\_analytics.sql

â”‚   â””â”€â”€ 04\_churn\_analytics.sql

â”‚

â”œâ”€â”€ src/

â”‚   â”œâ”€â”€ data/

â”‚   â”‚   â”œâ”€â”€ data\_loader.py

â”‚   â”‚   â”œâ”€â”€ data\_validator.py

â”‚   â”‚   â””â”€â”€ data\_cleaner.py

â”‚   â”‚

â”‚   â”œâ”€â”€ features/

â”‚   â”‚   â””â”€â”€ feature\_engineering.py

â”‚   â”‚

â”‚   â”œâ”€â”€ models/

â”‚   â”‚   â””â”€â”€ model\_predictor.py

â”‚   â”‚

â”‚   â”œâ”€â”€ monitoring/

â”‚   â”‚   â”œâ”€â”€ create\_simulated\_batch.py

â”‚   â”‚   â”œâ”€â”€ drift\_monitor.py

â”‚   â”‚   â”œâ”€â”€ prediction\_logger.py

â”‚   â”‚   â”œâ”€â”€ prediction\_monitor.py

â”‚   â”‚   â””â”€â”€ reference\_statistics.py

â”‚   â”‚

â”‚   â”œâ”€â”€ utils/

â”‚   â”‚   â”œâ”€â”€ config\_loader.py

â”‚   â”‚   â”œâ”€â”€ logger.py

â”‚   â”‚   â””â”€â”€ request\_id.py

â”‚   â”‚

â”‚   â””â”€â”€ pipeline.py

â”‚

â”œâ”€â”€ tests/

â”‚   â”œâ”€â”€ test\_api.py

â”‚   â”œâ”€â”€ test\_data\_validator.py

â”‚   â”œâ”€â”€ test\_drift\_monitor.py

â”‚   â”œâ”€â”€ test\_drift\_report.py

â”‚   â”œâ”€â”€ test\_doubles.py

â”‚   â”œâ”€â”€ test\_feature\_engineering.py

â”‚   â”œâ”€â”€ test\_model\_predictor.py

â”‚   â”œâ”€â”€ test\_prediction\_monitor.py

â”‚   â””â”€â”€ test\_reference\_statistics.py

â”‚

â”œâ”€â”€ models/

â”‚   â”œâ”€â”€ final\_xgboost\_pipeline.pkl

â”‚   â”œâ”€â”€ churn\_threshold.pkl

â”‚   â””â”€â”€ model\_metadata.json

â”‚

â”œâ”€â”€ Dockerfile

â”œâ”€â”€ Dockerfile.streamlit

â”œâ”€â”€ docker-compose.yml

â”œâ”€â”€ requirements.txt

â”œâ”€â”€ pytest.ini

â”œâ”€â”€ config.yaml

â”œâ”€â”€ .env.example

â”œâ”€â”€ .dockerignore

â”œâ”€â”€ .gitignore

â””â”€â”€ README.md

```



> \*\*Note:\*\* Raw datasets and model binary artifacts are excluded from Git version control. The `data/` and `models/` directories shown above represent the local project structure.



\---



## 49. Production Engineering Practices



This project demonstrates practical machine learning engineering practices including:



\* Modular Python architecture

\* Separation of concerns

\* Reusable data processing components

\* Data validation

\* Data quality checks

\* Feature engineering

\* Scikit-learn preprocessing pipelines

\* Cross-validation

\* Hyperparameter tuning

\* Class imbalance handling

\* Model artifact management

\* Configuration-driven paths

\* Dependency injection

\* Lazy model loading

\* Pydantic request validation

\* Request ID tracing

\* Structured application logging

\* Prediction monitoring

\* Feature drift monitoring

\* Automated testing

\* CI validation

\* Docker containerization

\* Docker Compose

\* MLflow experiment tracking

\* AWS ECR

\* AWS ECS Fargate

\* CloudWatch logging



\---



## 50. Production Limitations



Although this project follows production-oriented practices, it is important to distinguish the implementation from a fully enterprise production system.



### 50.1 No Application Load Balancer



The current AWS demonstration uses direct public access to the ECS task.



A production architecture should normally use:



```text

Internet

&#x20;  â†“

Application Load Balancer

&#x20;  â†“

ECS Fargate

```



### 50.2 No HTTPS/TLS Termination



The current demonstration does not include a complete HTTPS/TLS layer.



A production deployment should provide secure HTTPS access.



### 50.3 Ephemeral ECS Public IP



The current Fargate task uses a public IP.



That IP may change when the task is replaced or restarted.



A stable production endpoint would normally use an Application Load Balancer and domain.



### 50.4 Prediction Logs Are Not Persistent



Prediction events are currently written to JSON Lines files inside the application container.



Therefore, these logs are not guaranteed to survive container replacement.



A production system could use:



```text

CloudWatch

S3

RDS

DynamoDB

OpenSearch

```



or another centralized persistence/observability solution.



### 50.5 Lightweight Drift Detection



The current implementation uses relative mean change.



A production system could use:



\* PSI

\* KS test

\* Jensen-Shannon divergence

\* Population distribution comparisons

\* Feature-level statistical tests

\* Dedicated ML monitoring platforms



### 50.6 No Automated Model Retraining



The current system detects drift but does not automatically retrain and redeploy the model.



A future workflow could be:



```text

Drift Detected

&#x20;     â†“

Trigger Retraining

&#x20;     â†“

Evaluate Model

&#x20;     â†“

Register Model

&#x20;     â†“

Approval

&#x20;     â†“

Deploy

```



### 50.7 No Centralized MLflow Model Registry



MLflow is currently used for local experiment tracking and artifact management.



A centralized model registry is not currently implemented.



### 50.8 No Automated AWS Deployment on Every Commit



GitHub Actions currently provides validation/build workflows.



The project does not automatically deploy every GitHub commit directly to ECS.



\---



## 51. Future Improvements



Potential improvements include:



### Infrastructure



\* Application Load Balancer

\* HTTPS/TLS

\* Route 53

\* Custom domain

\* Private ECS networking

\* Restricted security groups

\* Infrastructure as Code

\* Terraform

\* AWS CDK



### Data & Storage



\* Amazon RDS

\* Amazon S3

\* Persistent prediction storage

\* Data versioning

\* Feature store



### ML Operations



\* MLflow Model Registry

\* Automated model retraining

\* Model versioning

\* Model approval workflow

\* Model promotion

\* Champion/challenger models

\* Automated rollback



### Monitoring



\* CloudWatch metrics

\* CloudWatch alarms

\* Centralized prediction logging

\* Advanced data drift detection

\* Model performance monitoring

\* Alerting



### Deployment



\* Automated CI/CD

\* ECS rolling deployments

\* Blue/green deployment

\* ECS autoscaling

\* Development/staging/production environments



### Security



\* AWS Secrets Manager

\* API authentication

\* API authorization

\* Rate limiting

\* WAF

\* Private networking



\---



## 52. Current Implementation Status



### 52.1 Data & Analytics



\* \[x] Data loading

\* \[x] Data validation

\* \[x] Data cleaning

\* \[x] Exploratory Data Analysis

\* \[x] Statistical analysis

\* \[x] Feature engineering

\* \[x] SQL analytics scripts



### 52.2 Machine Learning



\* \[x] Logistic Regression

\* \[x] Random Forest

\* \[x] XGBoost

\* \[x] Cross-validation

\* \[x] Hyperparameter optimization

\* \[x] Class imbalance handling

\* \[x] Model evaluation

\* \[x] Threshold evaluation

\* \[x] Final XGBoost model



### 52.3 MLOps



\* \[x] MLflow experiment tracking

\* \[x] Model artifact management

\* \[x] Configuration management

\* \[x] Reusable inference pipeline



### 52.4 API



\* \[x] FastAPI

\* \[x] Pydantic validation

\* \[x] Health endpoint

\* \[x] Readiness endpoint

\* \[x] Prediction endpoint

\* \[x] Monitoring endpoint

\* \[x] Drift endpoint

\* \[x] Lazy model loading

\* \[x] Request ID tracing

\* \[x] Structured logging



### 52.5 Monitoring



\* \[x] Prediction logging

\* \[x] Prediction metrics

\* \[x] Latency monitoring

\* \[x] High-risk prediction monitoring

\* \[x] Reference statistics

\* \[x] Feature drift detection

\* \[x] Simulated production batch



### 52.6 Testing



\* \[x] Unit tests

\* \[x] API tests

\* \[x] Model predictor tests

\* \[x] Feature engineering tests

\* \[x] Monitoring tests

\* \[x] Drift tests

\* \[x] Reference statistics tests

\* \[x] 26 tests passing



### 52.7 Containerization



\* \[x] Docker

\* \[x] Docker Compose

\* \[x] API container

\* \[x] Streamlit container

\* \[x] Container health check



### 52.8 Cloud Deployment



\* \[x] AWS ECR

\* \[x] AWS ECS

\* \[x] AWS Fargate

\* \[x] IAM ECS task execution role

\* \[x] CloudWatch Logs

\* \[x] Public API verification

\* \[x] AWS readiness verification

\* \[x] AWS prediction verification

\* \[x] AWS monitoring verification

\* \[x] AWS drift verification



### 52.9 Version Control & CI



\* \[x] Git

\* \[x] GitHub

\* \[x] GitHub Actions

\* \[x] Automated test workflow

\* \[x] Docker build workflow

\* \[x] Repository secret scan

\* \[x] Clean Git working tree



\---



## 53. Key Portfolio Takeaways



This project demonstrates hands-on exposure to the complete machine learning lifecycle:



```text

Data

&#x20;â†“

Analytics

&#x20;â†“

Machine Learning

&#x20;â†“

Experiment Tracking

&#x20;â†“

Model Artifact

&#x20;â†“

API

&#x20;â†“

Testing

&#x20;â†“

Docker

&#x20;â†“

AWS

&#x20;â†“

Monitoring

&#x20;â†“

Drift Detection

```



The strongest engineering capabilities demonstrated include:



\* Python

\* SQL

\* Pandas

\* NumPy

\* Scikit-learn

\* XGBoost

\* MLflow

\* FastAPI

\* Pydantic

\* Pytest

\* Docker

\* Docker Compose

\* GitHub Actions

\* Amazon ECR

\* Amazon ECS

\* AWS Fargate

\* IAM

\* CloudWatch

\* API monitoring

\* Feature drift monitoring



\---



## 54. Why This Project Is Different from a Traditional ML Project



A traditional churn project might look like:



```text

CSV

&#x20;â†“

Jupyter Notebook

&#x20;â†“

Model

&#x20;â†“

Accuracy

```



This project extends the workflow into:



```text

Customer Data

&#x20;     â†“

Data Validation

&#x20;     â†“

Data Cleaning

&#x20;     â†“

Feature Engineering

&#x20;     â†“

Statistical Analysis

&#x20;     â†“

ML Pipeline

&#x20;     â†“

Model Training

&#x20;     â†“

Cross Validation

&#x20;     â†“

Hyperparameter Tuning

&#x20;     â†“

Model Evaluation

&#x20;     â†“

MLflow

&#x20;     â†“

Model Artifact

&#x20;     â†“

FastAPI

&#x20;     â†“

Automated Tests

&#x20;     â†“

Docker

&#x20;     â†“

Amazon ECR

&#x20;     â†“

Amazon ECS Fargate

&#x20;     â†“

CloudWatch

&#x20;     â†“

Prediction Monitoring

&#x20;     â†“

Feature Drift Detection

```



The project therefore focuses not only on building a model, but also on demonstrating how a machine learning model can be packaged, tested, served, deployed, and monitored.



\---



## 55. Running Locally



### 55.1 Clone Repository



```bash

git clone https://github.com/sachinsadhish466-arch/Customer-Churn-Production-ML-Pipeline.git

cd Customer-Churn-Production-ML-Pipeline

```



### 55.2 Create Python 3.10 Virtual Environment



Windows:



```cmd

py -3.10 -m venv .venv

```



Activate:



```cmd

.venv\\Scripts\\activate

```



Verify:



```cmd

python --version

```



Expected:



```text

Python 3.10.x

```



### 55.3 Install Dependencies



```cmd

pip install -r requirements.txt

```



### 55.4 Run Tests



```cmd

pytest -v

```



Expected final result:



```text

26 passed

```



\---



## 56. Running FastAPI Locally



Start the API:



```cmd

uvicorn api.main:app --reload

```



API:



```text

http://localhost:8000

```



Swagger:



```text

http://localhost:8000/docs

```



Health:



```text

http://localhost:8000/health

```



Readiness:



```text

http://localhost:8000/ready

```



\---



## 57. Running Streamlit Locally



Start Streamlit:



```cmd

streamlit run app/app.py

```



Application:



```text

http://localhost:8501

```



\---



## 58. Running with Docker Compose



Build and start:



```cmd

docker compose up --build

```



API:



```text

http://localhost:8000

```



Swagger:



```text

http://localhost:8000/docs

```



Streamlit:



```text

http://localhost:8501

```



Stop:



```cmd

docker compose down

```



\---



## 59. Example Prediction Workflow



```text

Client

&#x20;  â”‚

&#x20;  â”‚ POST /predict

&#x20;  â–¼

FastAPI

&#x20;  â”‚

&#x20;  â–¼

Pydantic Validation

&#x20;  â”‚

&#x20;  â–¼

Data Validation

&#x20;  â”‚

&#x20;  â–¼

Data Cleaning

&#x20;  â”‚

&#x20;  â–¼

Feature Engineering

&#x20;  â”‚

&#x20;  â–¼

XGBoost Pipeline

&#x20;  â”‚

&#x20;  â–¼

Churn Probability

&#x20;  â”‚

&#x20;  â–¼

Classification Threshold

&#x20;  â”‚

&#x20;  â–¼

Prediction Response

&#x20;  â”‚

&#x20;  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º Request Logs

&#x20;  â”‚

&#x20;  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º Prediction Monitoring

```



\---



## 60. Example Prediction



### Input



```text

Customer Information

&#x20;       â†“

FastAPI /predict

&#x20;       â†“

Feature Engineering

&#x20;       â†“

XGBoost

```



### Output



```json

{

&#x20; "churn\_prediction": 1,

&#x20; "churn\_probability": 0.7654,

&#x20; "prediction\_label": "Likely to Churn",

&#x20; "classification\_threshold": 0.5

}

```



Interpretation:



```text

Prediction:

Likely to Churn



Probability:

76.54%



Threshold:

50%

```



\---



## 61. Security & Credential Handling



No AWS access keys or secret credentials are stored in the repository.



The project uses AWS IAM roles for ECS task execution.



Sensitive configuration should be supplied through secure environment/configuration mechanisms rather than committed to Git.



The repository also uses `.gitignore` rules for:



```text

.env

.env.\*

\*.pem

\*.key

secrets/

```



\---



## 62. Project Verification Summary



The completed system was verified at multiple levels.



### Local Application



```text

FastAPI

âœ“ Health

âœ“ Readiness

âœ“ Prediction

âœ“ Monitoring

âœ“ Drift

```



### Automated Testing



```text

26 passed

```



### Docker



```text

âœ“ API container

âœ“ Streamlit container

âœ“ Docker Compose

âœ“ Health check

```



### AWS



```text

âœ“ ECR image

âœ“ ECS Fargate task

âœ“ IAM execution role

âœ“ CloudWatch Logs

âœ“ Health endpoint

âœ“ Readiness endpoint

âœ“ Prediction endpoint

âœ“ Monitoring endpoint

âœ“ Drift endpoint

```



### GitHub



```text

âœ“ Repository

âœ“ Git history

âœ“ Secret scan

âœ“ GitHub Actions

âœ“ Clean working tree

```



\---



## 63. Repository



### GitHub Repository



\[https://github.com/sachinsadhish466-arch/Customer-Churn-Production-ML-Pipeline](https://github.com/sachinsadhish466-arch/Customer-Churn-Production-ML-Pipeline)



\---



## 64. Author



\*\*Sachin S\*\*



Data Science | Machine Learning | Python | SQL | MLOps | Cloud Deployment



\---



## 65. Project Status



```text

PROJECT STATUS: COMPLETED

```



Implemented and verified:



```text

âœ“ Machine Learning Pipeline

âœ“ XGBoost Model

âœ“ Cross Validation

âœ“ Hyperparameter Optimization

âœ“ MLflow Experiment Tracking

âœ“ Model Artifact Management

âœ“ FastAPI API

âœ“ Pydantic Validation

âœ“ Request ID Tracing

âœ“ Structured Logging

âœ“ Prediction Monitoring

âœ“ Feature Drift Detection

âœ“ Automated Testing

âœ“ Docker

âœ“ Docker Compose

âœ“ Streamlit Interface

âœ“ GitHub Actions

âœ“ Amazon ECR

âœ“ Amazon ECS Fargate

âœ“ IAM

âœ“ CloudWatch Logs

âœ“ AWS API Verification

âœ“ Complete Architecture Documentation

âœ“ GitHub Repository Cleanup

```



\---



## 66. License



This project is intended for educational, portfolio, and demonstration purposes.




