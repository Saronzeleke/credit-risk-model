# Technical Documentation

## Credit Risk Prediction System

**Author**: Saron Zeleke

## 1. System Architecture

┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Data Layer    │──▶│   Model Layer   │──▶│   Deployment    │
│  - Raw Data     │   │  - XGBoost      │   │  - FastAPI      │
│  - Processed    │   │  - RF, GB       │   │  - Streamlit    │
│  - Features     │   │  - Logistic     │   │  - MLflow       │
└─────────────────┘   └─────────────────┘   └─────────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Data Flow     │   │   Training      │   │   Monitoring    │
│  - RFM creation │   │  - GridSearchCV │   │  - SHAP         │
│  - Feature eng  │   │  - CV=5         │   │  - Metrics      │
│  - Scaling      │   │  - MLflow logs  │   │  - Reports      │
└─────────────────┘   └─────────────────┘   └─────────────────┘



## 2. Data Processing Pipeline

### 2.1 RFM Target Creation

# Creates proxy target using:

- Recency: Days since last transaction
- Frequency: Number of transactions
- Monetary: Total transaction amount

# Clustering approach

- K-Means with n_clusters=3
- High-risk = highest recency + lowest frequency/monetary
- Silhouette score: 0.3-0.7 (good separation)

**2.2 Feature Engineering**

| Feature            | Type        | Description                     |
|-------------------|------------|---------------------------------|
| TransactionHour    | Temporal   | Hour of transaction (0-23)      |
| TransactionDay     | Temporal   | Day of month                     |
| TransactionMonth   | Temporal   | Month of year                    |
| TransactionDayOfWeek | Temporal | Day of week (0-6)                |
| Amount             | Numeric    | Transaction amount               |
| Value              | Numeric    | Transaction value                |
| CountryCode        | Categorical| Origin country                   |
| CurrencyCode       | Categorical| Transaction currency             |
| ProviderId         | Categorical| Service provider                 |
| ProductCategory    | Categorical| Product type                     |
| ChannelId          | Categorical| Transaction channel              |
| PricingStrategy    | Categorical| Pricing model                    |

**2.3 Data Split Statistics**

Total transactions: 95,662
Training set: 76,529 (80%)
Test set: 19,133 (20%)
High-risk rate: 11.5% (balanced enough)

## 3. Model Training

**3.1 Model Configurations**

XGBoost (Best Model)

params:
  n_estimators: 500
  max_depth: 7
  learning_rate: 0.1
  colsample_bytree: 0.8
  subsample: 1.0
  random_state: 42

Random Forest

params:
  n_estimators: 200
  max_depth: None
  min_samples_split: 10
  random_state: 42

Gradient Boosting

params:
  n_estimators: 200
  max_depth: 5
  learning_rate: 0.2
  random_state: 42
  
**3.2 Training Pipeline**

# 5-fold Stratified Cross-Validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV with scoring='roc_auc'

grid_search = GridSearchCV(
    estimator=model_class(),
    param_grid=config,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

**3.3 Performance Metrics**

| Model             | ROC-AUC | Precision | Recall | F1-Score |
|------------------|---------|-----------|--------|----------|
| XGBoost           | 0.873   | 0.637     | 0.197  | 0.301    |
| Random Forest     | 0.868   | 0.575     | 0.252  | 0.350    |
| Gradient Boosting | 0.865   | 0.642     | 0.144  | 0.236    |
| Logistic          | 0.758   | 0.000     | 0.000  | 0.000    |

Confusion Matrix (XGBoost):

| Actual \ Predicted | Low    | High  |
|------------------|--------|-------|
| Low               | 16,680 | 248   |
| High              | 1,770  | 435   |


## 4. Model Explainability (SHAP)

**4.1 SHAP Computation**

# SHAP explainer selection

if model_type in ['xgboost', 'random_forest', 'gradient_boosting','Logistic Regreation']:
    explainer = shap.TreeExplainer(model)
else:
    explainer = shap.LinearExplainer(model, X_sample)

# Values computed on 500 samples

shap_values = explainer.shap_values(X_sample.astype(np.float32))

**4.2 Feature Importance**

| Rank | Feature | Mean |SHAP| Interpretation |
|------|---------|-----------|----------------|
| 1 | TransactionMonth | 3.30 | Seasonal patterns strongest predictor |
| 2 | TransactionDay | 1.20 | Day-of-month matters |
| 3 | Amount | 0.40 | Transaction size indicator |
| 4 | TransactionHour | 0.28 | Time-of-day patterns |
| 5 | Value | 0.24 | Correlated with amount |

**4.3 SHAP Dependencies**

Positive SHAP → Increases risk

Negative SHAP → Decreases risk

Feature interactions captured

## 5. API Design

**5.1 Endpoints**

POST /predict

Request Schema:

{
  "CustomerId": "integer",
  "Amount": "float",
  "Value": "float",
  "TransactionStartTime": "string (ISO datetime)",
  "CountryCode": "string",
  "CurrencyCode": "string",
  "ProviderId": "string",
  "ProductCategory": "string",
  "ChannelId": "string",
  "PricingStrategy": "string"
}

Response Schema:

{
  "transaction_id": "uuid",
  "prediction": "integer (0/1)",
  "probability": "float",
  "risk_level": "string (Low/Medium/High)",
  "threshold_used": "float",
  "timestamp": "string (ISO datetime)"
}

**5.2 Performance**

Response time: <100ms

Concurrent requests: 100+

Throughput: 1000 req/sec

## 6. Deployment

**6.1 Local Deployment**

# API

uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Dashboard

streamlit run src/dashboard/app.py --server.port 8501

**6.2 Docker Deployment**
 dockerfile
 FROM python:3.11-slim
 WORKDIR /app
 install  requirements.txt .
 RUN pip install -r requirements.txt

CMD ["uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"]

**6.3 Environment Variables**

MLFLOW_TRACKING_URI=./mlruns
MODEL_PATH=./models/xgboost_model.pkl
LOG_LEVEL=INFO

## 7. Monitoring & Logging

**7.1 MLflow Tracking**

# Logged metrics

- model parameters
- ROC-AUC scores
- precision/recall
- feature importance
- model artifacts

**7.2 Application Logging**

# Log format

2026-02-16 21:44:37,640 - INFO - Transaction processed: ID=12345
2026-02-16 21:44:37,765 - INFO - Prediction: 0, prob=0.073
2026-02-16 21:44:37,766 - ERROR - Invalid input: missing field

##8. Testing Strategy

**8.1 Unit Tests**

def test_data_loading():
    assert X_train.shape[0] == 76529
    
def test_model_prediction():
    prob = model.predict_proba(X_test)[0,1]
    assert 0 <= prob <= 1

**8.2 Integration Tests**

 API endpoint testing

 Database connections

 File I/O operations

**8.3 Performance Tests**

 Load testing with 1000 concurrent requests

 Response time < 200ms at 95th percentile

## 9. Limitations & Future Work

**Current Limitations**

Proxy target based on RFM (not actual defaults)

Limited to 9 numeric features

XGBoost SHAP warning (version compatibility)

**Future Improvements**

Feature Engineering

Add customer lifetime value

Include macroeconomic indicators

Transaction velocity features

**Model Enhancements**

Ensemble of top models

Deep learning approaches

Online learning for real-time updates

**Deployment**

Kubernetes orchestration

A/B testing framework

Model drift detection

## 10. References

XGBoost Documentation

SHAP Values for Model Explainability

MLflow Tracking

FastAPI Production

Credit Risk Modeling Best Practices

## 11. Appendix

**A. Environment Setup**

  python -m venv my_env

  my_env\Scripts\activate

  pip install -r requirements.txt

**B.Data Dictionary**

| Column                | Type     | Description                  |
|-----------------------|----------|------------------------------|
| TransactionId         | string   | Unique transaction ID        |
| CustomerId            | int      | Customer identifier          |
| Amount                | float    | Transaction amount           |
| Value                 | float    | Transaction value            |
| TransactionStartTime  | datetime | Transaction timestamp        |
| ProductCategory       | string   | Product type                 |
| ChannelId             | string   | Transaction channel          |
| FraudResult           | int      | Target (0/1)                 |


**C.Model Files**

models/
├── xgboost_model.pkl          # 87.3% ROC-AUC
├── random_forest_model.pkl    # 86.8% ROC-AUC
├── gradient_boosting_model.pkl # 86.5% ROC-AUC
└── logistic_model.pkl         # 75.8% ROC-AUC

**D.Configuration File (configs/config.yaml)**