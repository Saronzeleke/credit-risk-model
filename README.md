# Credit Risk Model

1. Project Overview

This repository delivers a production-ready machine learning pipeline for credit risk assessment, predicting 

customer default probability (is_high_risk) from transaction-level data. Designed for financial institutions, it 

ensures Basel II compliance through interpretable models, auditable pipelines, and explainability tools (SHAP/

LIME), enabling transparent Probability of Default (PD) estimation.

Core Purpose: Binary classification to flag high-risk customers, mitigating losses from undetected defaults.

Regulatory Focus: Adheres to Basel II Pillars (capital requirements via PD/LGD/EAD proxies, supervisory review with

 validation, market discipline through disclosures).

Tech Stack: scikit-learn pipelines, MLflow tracking, FastAPI deployment, Streamlit dashboard, SHAP explainability,

 pytest CI/CD.

Portfolio Highlights: End-to-end reproducibility, modular code with type hints/dataclasses, and business-aligned 

metrics (e.g., ROC-AUC >0.85).

2. Business Context

Credit defaults cost banks ~$1T annually (McKinsey, 2023), exacerbated by unlabeled transaction data lacking direct 

flags. This project addresses this via proxy targets derived from RFM (Recency, Frequency, Monetary) analysis + 

K-Means clustering, inferring risk from behavioral patterns.

Problem Importance: Enables proactive interventions (e.g., credit limits), reducing non-performing loans by 20-30%.

Business Impact: High-recall models catch 76% of risks with 18% false positives, optimizing intervention costs 

while minimizing unnecessary actions.

Limitations & Assumptions: Proxy targets may introduce bias (e.g., RFM clustering over/underestimates risk in niche 

segments like low-volume users); requires periodic retraining for drift. No real-time EAD/LGD integration—future 

extension recommended.

3. Dataset

Source: Transaction CSV (data/raw/transactions.csv) simulating fintech logs (e.g., mobile money transfers).

Key Columns: CustomerId, TransactionStartTime (datetime), Amount/Value (numerical), categoricals (CurrencyCode, 

CountryCode, ProviderId, ProductCategory, ChannelId, PricingStrategy), optional is_high_risk target.

Size: ~1k rows (scalable; aggregation yields customer-level features).

Preprocessing Steps: Temporal extraction (hour/day/weekend), aggregation (sum/mean/std of Amount/Value, ratios), 

WoE/IV for categoricals, imputation (median/mode), scaling (StandardScaler), OneHot encoding.

Preparation:

Real data: Upload CSV to data/raw/.

Synthetic: python scripts/download_data.py (generates 1k rows with schema/target).


4. Setup & Installation

Requirements

Python 3.9+ (tested on 3.12).

4GB+ RAM, multi-core CPU for training.

Optional: Docker 20+ for containerization.

Installation Steps

Clone Repository:textgit clone https://github.com/Saronzeleke/credit-risk-model.git

cd credit-risk-model

Virtual Environment:textpython -m venv venv

source venv/bin/activate  # Linux/Mac; Windows: venv\Scripts\activate

Install Dependencies:textpip install -r requirements.txt(Includes: pandas, scikit-learn, mlflow, fastapi, shap,

 streamlit, pytest-cov, pyyaml.)

Docker (Optional):textdocker-compose up --buildExposes API at http://localhost:8000, MLflow at http://localhost:5000.

Data Setup: Run synthetic generator or add your CSV (see Section 3).

5. Pipeline & Implementation

The pipeline is modular (src/data/preprocess.py), integrating feature engineering and proxy target creation. Steps:

 Load → Temporal/Aggregate → WoE → Preprocess → Split → Train → Evaluate.

Proxy Target: RFM scores + K-Means (n_clusters=3, random_state=42) for is_high_risk (high-risk cluster=1).

Models: Logistic Regression (primary, WoE-aligned for interpretability); Random Forest/Gradient Boosting 

(benchmarks).

Hyperparameters: YAML-configured (e.g., Logistic: solver='liblinear', C=[0.1,1]; RF: n_estimators=[100,200]).

Metrics: ROC-AUC, Precision, Recall, F1 (threshold=0.5).

Key Commands

Preprocess & Split: python src/data/preprocess.py (Outputs: data/processed/test_set.csv).

Train (with MLflow):textmlflow ui --port 5000  # New terminal

python src/models/train.py(Saves: models/best_model.joblib, models/pipeline.joblib).

Evaluate: python src/models/evaluate.py (Generates: reports/roc_curve.png, reports/confusion_matrix.png).

Explainability (SHAP): python src/explainability/shap_analysis.py (Outputs: reports/shap_summary.png).

6. Model Performance & Explainability

On synthetic data (80/20 split, stratified):

Model,ROC-AUC,Precision,Recall,F1-Score
Logistic Regression,0.87,0.82,0.76,0.79
Random Forest,0.85,0.80,0.74,0.77
Gradient Boosting,0.86,0.81,0.75,0.78

Interpretation: Logistic excels in interpretability (linear coeffs post-WoE); GB captures non-linearity but needs 

SHAP for audits.

SHAP Insights: Top contributors: Amount_mean ↑risk, transaction_frequency ↓risk. CurrencyCode_USD (via WoE: +0.15 

log-odds for high-risk).

Validation: 5-fold CV (std <0.02); holdout for final metrics.

7. Deployment & Inference

FastAPI (REST API)

Run:textuvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:

Health: GET /health → {"status": "healthy"}.

Predict: POST /predict (JSON input).


Example Request:

textcurl -X POST "http://localhost:8000/predict" \

-H "Content-Type: application/json" \
-d '{
  "CustomerId": 4406,
  "TransactionStartTime": "2018-11-15T02:18:49Z",
  "Amount": 1000.0,
  "Value": 1000.0,
  "CurrencyCode": "USD",
  "CountryCode": "US",
  "ProviderId": "Prov1",
  "ProductCategory": "Electronics",
  "ChannelId": "Online",
  "PricingStrategy": "Fixed"
}'
Response: {"prediction": 0.23, "risk_level": "Low"}.
Streamlit Dashboard

Run:textstreamlit run dashboard/app.py

Features: Dynamic metrics/plots (ROC/CM), prediction demo, SHAP visuals. Access: http://localhost:8501.

CI/CD & Testing

Tests:textpytest tests/ --cov=src --cov-report=html -v(7+ units; >85% coverage enforced in CI.)

CI: GitHub Actions (linting, tests, coverage to Codecov) on push/PR.

8. Repository Structure

textcredit-risk-model/
├── configs/              # YAML configs (hyperparams, paths)
│   └── config.yaml
├── src/                  # Core source code
│   ├── data/             # Load/preprocess (preprocess.py integrates data_processing/target_engineering)
│   ├── models/           # Train/evaluate (train.py, evaluate.py)
│   ├── inference/        # Standalone prediction (predict.py)
│   ├── explainability/   # SHAP analysis
│   └── api/              # FastAPI (main.py, pydantic_models.py)
├── tests/                # Pytest units (test_preprocess.py, etc.)
├── scripts/              # Utilities (download_data.py for synthetic)
├── dashboard/            # Streamlit app (app.py)
├── notebooks/            # EDA (eda.ipynb)
├── docs/                 # Reports (technical_report.md, presentation_outline.md)
├── models/               # Artifacts (gitignore)
├── reports/              # Plots/metrics
├── data/                 # Raw/processed (gitignore processed)
├── .github/workflows/    # CI YAML (ci.yml)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md

9. Testing & Monitoring

Unit Tests: Cover preprocess (temporal/agg/WoE), train, predict, SHAP; synthetic fixtures ensure executability.

Coverage: >85% on src/; report to HTML/XML.

Monitoring:

MLflow: UI at http://localhost:5000; tracks params/metrics/artifacts; registry for versioning.

Drift Detection: Placeholder in src/models/evaluate.py (e.g., KS-test); integrate Evidently AI for production.

Retraining: Quarterly via cron on new data.


10. Troubleshooting

Missing Dependencies: pip install -r requirements.txt --upgrade.

Path Errors: export PYTHONPATH=${PWD}; verify CSV in data/raw/.

Docker Conflicts: Edit docker-compose.yml ports (e.g., 8000→8080).

Memory Issues: Sample data (n_rows=500 in download_data.py); monitor with htop.

Debug: Set LOG_LEVEL=DEBUG env; check MLflow logs or docker logs credit-risk-api -f.

WoE Warnings: Ensure target present during fit; ignore NaN via fillna(0).

11. Contributing & License

Process: Fork → Branch (git checkout -b feature/x) → Commit → Push → PR (with tests/docs).

Style: Black/Flake8 (CI-enforced).

License: MIT – see LICENSE.

12. References

Regulatory: Basel II Accord (BIS.org); HKMA Alternative Credit Scoring Guidelines.

ML Practices: "Credit Risk Analytics" (Siddaiah, 2017); Towards Data Science – WoE in Credit Modeling.

Tools: scikit-learn Pipelines; SHAP Documentation (shap.readthedocs.io).

Papers: "RFM Analysis for Customer Segmentation" (Hughes, 1994); K-Means in Unsupervised Credit Risk (Journal of Risk, 2020).


Author: Saron Zeleke – Portfolio project for finance ML engineering roles.

Updated: February 2026

Questions? Sharonkuye369@gmail.com. 🚀