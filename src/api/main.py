import os
import sys
import uuid
import joblib
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants - CORRECTED PATHS
BASE_DIR = r"C:\Users\admin\credit-risk-model"
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Model paths - CORRECTED
MODEL_PATHS = {
    'xgboost': os.path.join(MODELS_DIR, "xgboost_model.pkl"),
    'random_forest': os.path.join(MODELS_DIR, "random_forest_model.pkl"),
    'gradient_boosting': os.path.join(MODELS_DIR, "gradient_boosting_model.pkl"),
    'logistic': os.path.join(MODELS_DIR, "logistic_model.pkl")
}

# Default model (best performing)
DEFAULT_MODEL = 'xgboost'

# Globals
model = None
preprocessor = None
model_info = {}
current_model_name = DEFAULT_MODEL

# Pydantic schemas

class Transaction(BaseModel):
    """Transaction input schema - matches your training data"""
    AccountId: int = Field(..., description="Account identifier")
    BatchId: int = Field(..., description="Batch identifier")
    CustomerId: int = Field(..., description="Customer identifier")
    ProductId: int = Field(..., description="Product identifier")
    ProviderId: int = Field(..., description="Provider identifier")
    SubscriptionId: int = Field(..., description="Subscription identifier")
    TransactionId: int = Field(..., description="Transaction identifier")
    
    # Numeric features
    Amount: float = Field(..., description="Transaction amount")
    Value: float = Field(..., description="Transaction value")
    CountryCode: int = Field(..., description="Country code")
    ChannelId: int = Field(..., description="Channel identifier")
    PricingStrategy: int = Field(..., description="Pricing strategy")
    
    # Temporal
    TransactionStartTime: str = Field(..., description="Transaction start time (YYYY-MM-DD HH:MM:SS)")
    
    # Categorical (will be encoded)
    CurrencyCode: str = Field(..., description="Currency code (e.g., USD, EUR)")
    ProductCategory: str = Field(..., description="Product category")
    
    # Optional (may not be used)
    FraudResult: Optional[int] = Field(0, description="Fraud result (optional)")


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    transaction_id: str
    prediction: int
    probability: float
    risk_level: str
    model_used: str
    threshold_used: float
    timestamp: str
    features_used: Optional[list] = None


class ModelInfoResponse(BaseModel):
    """Model information response"""
    current_model: str
    available_models: list
    model_metrics: Optional[dict] = None
    features_count: Optional[int] = None

# Model Manager
class ModelManager:
    """Manages model loading, switching, and prediction"""
    
    @staticmethod
    def load_model(model_name: str = DEFAULT_MODEL):
        """Load specified model and its metadata"""
        global model, model_info, current_model_name
        
        if model_name not in MODEL_PATHS:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_PATHS.keys())}")
        
        model_path = MODEL_PATHS[model_name]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load model data
            model_data = joblib.load(model_path)
            
            # Handle different save formats
            if isinstance(model_data, dict):
                model = model_data.get('model', model_data)
                model_info = {
                    'model_type': model_name,
                    'threshold': model_data.get('threshold', 0.5),
                    'metrics': model_data.get('metrics', {}),
                    'feature_names': model_data.get('feature_names', [])
                }
            else:
                model = model_data
                model_info = {
                    'model_type': model_name,
                    'threshold': 0.5,
                    'metrics': {},
                    'feature_names': []
                }
            
            current_model_name = model_name
            logger.info(f"✅ Model loaded: {model_name}")
            logger.info(f"   Threshold: {model_info['threshold']}")
            
            if model_info.get('metrics'):
                logger.info(f"   ROC-AUC: {model_info['metrics'].get('roc_auc', 'N/A'):.4f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise
    
    @staticmethod
    def load_preprocessor(preprocessor_path: str = None):
        """Load preprocessing pipeline"""
        global preprocessor
        
        if preprocessor_path is None:
            preprocessor_path = os.path.join(MODELS_DIR, "data_pipeline.pkl")
        
        if not os.path.exists(preprocessor_path):
            logger.warning(f"⚠️ Preprocessor not found at {preprocessor_path}")
            logger.warning("Continuing without preprocessor - ensure input matches model expectations")
            preprocessor = None
            return
        
        try:
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"✅ Preprocessor loaded: {preprocessor_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load preprocessor: {e}")
            preprocessor = None

    @staticmethod
    def preprocess_transaction(transaction: Dict) -> pd.DataFrame:
        """Convert transaction dict to model-ready DataFrame"""
        global preprocessor
           # Convert to DataFrame
        df = pd.DataFrame([transaction])
         # Add time features BEFORE dropping any columns
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek

         # Drop non-feature columns if preprocessor exists
        required_features = [
        'CountryCode', 'Amount', 'Value', 'PricingStrategy', 'FraudResult',
        'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionDayOfWeek'
          ]
        df = df[required_features]

        # Ensure numeric dtype
        df = df.astype(np.float32)
        return df
    
    @staticmethod
    def predict_single(transaction: Dict) -> Dict[str, Any]:
        """Generate prediction for a single transaction"""
        global model, model_info
        
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            # Preprocess
            df_processed = ModelManager.preprocess_transaction(transaction)
            
            # Convert to float32 for XGBoost
            df_processed = df_processed.astype(np.float32)
            
            # Predict
            threshold = model_info.get('threshold', 0.5)
            proba = model.predict_proba(df_processed)[0, 1]
            prediction = int(proba >= threshold)
            
            # Risk level
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
                "threshold_used": threshold,
                "model_used": current_model_name
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @staticmethod
    def list_available_models() -> list:
        """Return list of available models"""
        available = []
        for name, path in MODEL_PATHS.items():
            if os.path.exists(path):
                available.append(name)
        return available

# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Load default model on startup
    try:
        ModelManager.load_model(DEFAULT_MODEL)
        ModelManager.load_preprocessor()
        logger.info("🚀 API startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.warning("API starting without model - use /switch-model endpoint first")
    
    yield
    
    # Cleanup on shutdown
    global model, preprocessor, model_info
    model = None
    preprocessor = None
    model_info = {}
    logger.info("👋 API shutdown complete")


# FastAPI app

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict credit risk using multiple ML models",
    version="1.0.0",
    lifespan=lifespan
)

# API Endpoints

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Credit Risk Prediction API",
        "version": "1.0.0"
    }


@app.get("/models", response_model=ModelInfoResponse, tags=["Models"])
async def get_models():
    """Get information about available models and current model"""
    return ModelInfoResponse(
        current_model=current_model_name,
        available_models=ModelManager.list_available_models(),
        model_metrics=model_info.get('metrics', {}),
        features_count=len(model_info.get('feature_names', [])) if model_info.get('feature_names') else None
    )


@app.post("/switch-model/{model_name}", tags=["Models"])
async def switch_model(model_name: str):
    """Switch to a different model"""
    try:
        ModelManager.load_model(model_name)
        return {
            "message": f"Switched to {model_name} model",
            "current_model": model_name,
            "metrics": model_info.get('metrics', {})
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: Transaction):
    """Predict credit risk for a single transaction"""
    logger.info("📥 Received prediction request")
    
    transaction_dict = transaction.dict()
    result = ModelManager.predict_single(transaction_dict)
    
    response = PredictionResponse(
        transaction_id=str(uuid.uuid4()),
        prediction=result["prediction"],
        probability=result["probability"],
        risk_level=result["risk_level"],
        model_used=result["model_used"],
        threshold_used=result["threshold_used"],
        timestamp=datetime.now().isoformat(),
        features_used=model_info.get('feature_names', [])[:10]  # First 10 features
    )
    
    logger.info(f"📤 Prediction: {response.risk_level} risk (prob: {response.probability:.3f})")
    return response


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(transactions: list[Transaction]):
    """Predict credit risk for multiple transactions"""
    results = []
    
    for i, transaction in enumerate(transactions):
        try:
            result = ModelManager.predict_single(transaction.dict())
            results.append({
                "index": i,
                **result,
                "transaction_id": str(uuid.uuid4())
            })
        except Exception as e:
            results.append({
                "index": i,
                "error": str(e),
                "probability": 0.0,
                "prediction": 0
            })
    
    return {
        "total": len(results),
        "successful": sum(1 for r in results if "error" not in r),
        "results": results
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "current_model": current_model_name,
        "preprocessor_loaded": preprocessor is not None,
        "available_models": ModelManager.list_available_models()
    }


# Run uvicorn
if __name__ == "__main__":
    import uvicorn
    import numpy as np
    
    print("\n" + "="*60)
    print("🚀 CREDIT RISK PREDICTION API")
    print("="*60)
    print(f"📁 Models directory: {MODELS_DIR}")
    print(f"📊 Available models: {list(MODEL_PATHS.keys())[:4]}")
    print(f"⚙️  Default model: {DEFAULT_MODEL}")
    print("="*60 + "\n")
    
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )