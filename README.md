# Credit Risk Prediction Model

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)  

[![MLflow Tracked](https://img.shields.io/badge/MLflow-tracked-brightgreen)](https://mlflow.org/)  

[![XGBoost 3.1.0](https://img.shields.io/badge/XGBoost-3.1.0-orange)](https://xgboost.readthedocs.io/)  

[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

📋 Project Overview

This project delivers a production-ready credit risk prediction system that uses machine learning to 

identify high-risk transactions in real-time. The solution combines:

1. High-accuracy predictive modeling using XGBoost with 87.3% ROC-AUC

2. Explainable AI via SHAP for interpretability

3. End-to-end ML lifecycle tracking using MLflow

4. REST API for seamless integration with business systems

5. Interactive dashboards for monitoring and business insights

**Business Impact:** Automates risk assessment, reduces false positives, and improves operational efficiency.

🎯 Key Features

| Feature                   | Benefit                                                                        |
| ------------------------- | ------------------------------------------------------------------------------ |
| **High Accuracy**         | XGBoost model with 87.3% ROC-AUC identifies high-risk transactions effectively |
| **Explainable AI**        | SHAP provides feature-level impact for decisions                               |
| **MLflow Tracking**       | Full experiment versioning, reproducibility, and logging                       |
| **REST API**              | FastAPI endpoint supports real-time predictions                                |
| **Interactive Dashboard** | Streamlit dashboard visualizes risk distributions and model performance        |
| **Production Ready**      | CI/CD pipelines, reproducible training, and deployment-ready structure         |

📁 Project Structure
```
credit-risk-model/
│
├── data/
│ ├── raw/ # Original transaction data
│ └── processed/ # Cleaned and feature-engineered datasets
│
├── models/ # Saved models
│ ├── xgboost_model.pkl # Best model (87.3% ROC-AUC)
│ ├── random_forest_model.pkl
│ ├── gradient_boosting_model.pkl
│ └── logistic_model.pkl
│
├── src/ # Source code
│ ├── data/ # Data processing
│ │ ├── preprocess.py
│ │ └── split.py
│ │
│ ├── models/ # Model training
│ │ └── train.py
│ ├── api/ # REST API
│ │ └── main.py
│ ├── dashboard/ # Streamlit dashboard
│ │ └── app.py
│ └── explainability/ # SHAP analysis
│ └── shap_analysis.py
│
├── reports/ # Outputs and visualizations
│ ├── shap_summary_xgboost.png
│ ├── model_comparison.csv
│ └── shap_values_xgboost.csv
│
├── docs/ # Documentation
│ ├── technical_report.md
│ └── presentation_outline.md
│
├── configs/
│ └── config.yaml # Model and pipeline configurations
│
├── requirements.txt
├── README.md
└── LICENSE
```

🚀 Quick Start

**Prerequisites**

python 3.11+

pip install -r requirements.txt

1. Data Processing

# Create proxy target and split data

python src/data/split.py

# Generates:
# - X_train.csv, X_test.csv
# - y_train.csv, y_test.csv

2. Train Models

# Train all models with MLflow tracking

python src/models/train.py

# Outputs:
# - Best model: xgboost_model.pkl (87.3% ROC-AUC)
# - Other models for comparison

3. Explainability (SHAP)

# Generate SHAP interpretability reports

python -m src.explainability.shap_analysis

# Outputs saved to /reports:
# - shap_summary_xgboost.png
# - shap_bar_xgboost.png
# - shap_values_xgboost.csv

4. Run API

# Start FastAPI server

python src/api/main.py

# API Docs: http://localhost:8000/docs

5. Launch Dashboard

# Streamlit web interface

streamlit run dashboard/app.py

# Dashboard available at: http://localhost:8501

📊 Model Performance

| Model             | ROC-AUC | Precision | Recall | F1-Score |
| ----------------- | ------- | --------- | ------ | -------- |
| 🏆 XGBoost        | 0.873   | 0.637     | 0.197  | 0.301    |
| 🌲 Random Forest  | 0.868   | 0.575     | 0.252  | 0.350    |
| 📊 Gradient Boost | 0.865   | 0.642     | 0.144  | 0.236    |
| 📉 Logistic       | 0.758   | 0.000     | 0.000  | 0.000    |

Confusion Matrix (XGBoost)

| Actual \ Predicted | Low    | High |
| ------------------ | ------ | ---- |
| Low                | 16,680 | 248  |
| High               | 1,770  | 435  |

🔧 API Usage

POST /predict
```
{
{
  "TransactionId": 1001,
  "BatchId": 2001,
  "AccountId": 3001,
  "SubscriptionId": 4001,
  "CustomerId": 5001,
  "CurrencyCode": "USD",
  "CountryCode": 840,
  "ProviderId": 6001,
  "ProductId": 7001,
  "ProductCategory": "airtime",
  "ChannelId": 1,
  "Amount": 150.0,
  "Value": 150.0,
  "TransactionStartTime": "2024-02-17 14:30:00",
  "PricingStrategy": 2,
  "FraudResult": 0
}
}

✅ Sample Response

{
{
  "transaction_id": "91386216-d77d-4198-bf1c-f7c4721eef4e",
  "prediction": 0,
  "probability": 0.0000043345212361600716,
  "risk_level": "Low",
  "model_used": "xgboost",
  "threshold_used": 0.5,
  "timestamp": "2026-02-17T16:36:28.656664",
  "features_used": [
    "CountryCode",
    "Amount",
    "Value",
    "PricingStrategy",
    "FraudResult",
    "TransactionHour",
    "TransactionDay",
    "TransactionMonth",
    "TransactionDayOfWeek"
  ]
}
}

```

📈 Feature Importance (SHAP)

| Feature          | Type     | Impact on Risk      |
| ---------------- | -------- | ------------------- |
| TransactionMonth | Temporal | Seasonal pattern    |
| TransactionDay   | Temporal | Day-of-month effect |
| Amount           | Numeric  | Transaction size    |
| TransactionHour  | Temporal | Time-of-day effect  |
| Value            | Numeric  | Transaction value   |

**Key Insights**

1. High-risk transactions peak in December

2. Larger amounts are more likely to be flagged

3. Late-night transactions are higher risk

🧪 Testing

# Run unit and integration tests

 pytest tests/

# Test API endpoints

 python tests/test_api.py

# Batch prediction test

 python src/predict.py --input x_test.csv
```
📝 Requirements

pandas==2.2.0
numpy==1.26.0
scikit-learn==1.6.1
xgboost==3.1.0
shap==0.49.1
mlflow==3.9.0
fastapi==0.115.0
uvicorn==0.29.0
streamlit==1.29.0
joblib==1.3.0
matplotlib==3.8.0
seaborn==0.13.0
pyyaml==6.0
```
🤝 Contributing

1. Fork the repository

2. Create a feature branch

3. Commit your changes

4. Push to branch

5. Open a Pull Request

📄 License

MIT License — see LICENSE file

👥 Author

Saron Zeleke

🙏 Acknowledgments

1. SHAP for model interpretability

2. MLflow for experiment tracking

3. FastAPI and Streamlit for production deployment