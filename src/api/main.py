import os
import uuid
import joblib
import logging
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Globals
# ----------------------
model = None
preprocessor = None
model_info = None

# ----------------------
# Pydantic schemas
# ----------------------
class Transaction(BaseModel):
    AccountId: int
    BatchId: int
    CustomerId: int
    FraudResult: int
    ProductId: int
    ProviderId: int
    SubscriptionId: int
    Amount: float
    Value: float
    CurrencyCode: str
    CountryCode: int
    ProductCategory: str
    ChannelId: int
    PricingStrategy: int
    TransactionStartTime: str
    TransactionId: int

class PredictionResponse(BaseModel):
    transaction_id: str
    prediction: int
    probability: float
    risk_level: str
    threshold_used: float
    timestamp: str

# ----------------------
# Model Manager
# ----------------------
class ModelManager:
    @staticmethod
    def load_model(model_path: str = "C:/Users/admin/credit-risk-model/models/random_forest_model (1).pkl"):
        global model, model_info
        try:
            model = joblib.load(model_path)
            model_info = {
                'model_type': type(model).__name__,
                'threshold': 0.5
            }
            logger.info(f"Model loaded successfully: {model_info['model_type']}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @staticmethod
    def load_preprocessor(preprocessor_path: str = "C:/Users/admin/credit-risk-model/models/data_pipeline.pkl"):
        global preprocessor
        try:
            preprocessor = joblib.load(preprocessor_path)
            logger.info("Preprocessor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise

    @staticmethod
    def preprocess_transaction(transaction: Dict):
        """Convert transaction dict to model-ready NumPy array"""
        global preprocessor

        if preprocessor is None:
            raise HTTPException(status_code=500, detail="Preprocessor not loaded")

        df = pd.DataFrame([transaction])

        # Drop only non-feature identifiers (keep CustomerId, TransactionStartTime, etc.)
        NON_FEATURE_COLS = {'AccountId', 'BatchId', 'TransactionId', 'FraudResult'}
        df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns], errors='ignore')

        try:
            # Transform returns NumPy array; no need for get_feature_names_out()
            df_processed = preprocessor.transform(df)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

        return df_processed

    @staticmethod
    def predict_single(transaction: Dict) -> Dict[str, Any]:
        global model, model_info
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        try:
            df_processed = ModelManager.preprocess_transaction(transaction)
            threshold = model_info.get('threshold', 0.5)
            proba = model.predict_proba(df_processed)[0, 1]
            prediction = int(proba >= threshold)

            if proba < 0.3:
                risk_level = "Low"
            elif proba < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"

            return {
                "prediction": prediction,
                "probability": float(proba),
                "risk_level": risk_level,
                "threshold_used": threshold
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ----------------------
# Lifespan
# ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    ModelManager.load_model()
    ModelManager.load_preprocessor()
    logger.info("API startup complete")
    yield
    global model, preprocessor, model_info
    model = None
    preprocessor = None
    model_info = None
    logger.info("API shutdown complete")

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI(title="Credit Risk Prediction API", lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    logger.info("Received prediction request")
    transaction_dict = transaction.dict()
    result = ModelManager.predict_single(transaction_dict)
    response = PredictionResponse(
        transaction_id=str(uuid.uuid4()),
        prediction=result["prediction"],
        probability=result["probability"],
        risk_level=result["risk_level"],
        threshold_used=result["threshold_used"],
        timestamp=datetime.now().isoformat()
    )
    logger.info(f"Prediction completed: {response}")
    return response

# ----------------------
# Run uvicorn
# ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
