"""
FastAPI application for credit risk prediction.
"""
import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib

# FastAPI imports
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Import Pydantic models
from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
    RiskPrediction,
    HealthCheck,
    ErrorResponse,
    TransactionFeatures
)

# Import data processing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_processing import create_feature_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
MODEL = None
MODEL_VERSION = None
PIPELINE = None
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting up Credit Risk API...")
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down Credit Risk API...")


# Create FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk based on transaction data",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # In production, specify exact hosts
)


def load_model():
    """Load the trained model from MLflow registry."""
    global MODEL, MODEL_VERSION, PIPELINE
    
    try:
        # MLflow configuration
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Initialize MLflow client
        client = MlflowClient()
        
        # Get the latest production model
        model_name = "credit_risk_model"
        
        try:
            # Try to get production model
            model_versions = client.get_latest_versions(model_name, stages=["Production"])
            if model_versions:
                model_version = model_versions[0]
            else:
                # Fallback to latest version
                model_versions = client.get_latest_versions(model_name)
                model_version = model_versions[0]
            
            model_uri = f"models:/{model_name}/{model_version.version}"
            
            # Load model
            MODEL = mlflow.sklearn.load_model(model_uri)
            MODEL_VERSION = model_version.version
            
            # Load feature pipeline (assuming it's saved as an artifact)
            pipeline_path = "models/feature_pipeline.pkl"
            if os.path.exists(pipeline_path):
                PIPELINE = joblib.load(pipeline_path)
            else:
                # Create default pipeline
                PIPELINE = create_feature_pipeline()
            
            logger.info(f"Successfully loaded model version {MODEL_VERSION}")
            logger.info(f"Model type: {type(MODEL).__name__}")
            
        except Exception as e:
            logger.warning(f"Could not load from MLflow registry: {str(e)}")
            
            # Fallback to local model file
            local_model_path = "models/best_model.pkl"
            if os.path.exists(local_model_path):
                MODEL = joblib.load(local_model_path)
                MODEL_VERSION = "local"
                
                # Try to load pipeline
                pipeline_path = "models/feature_pipeline.pkl"
                if os.path.exists(pipeline_path):
                    PIPELINE = joblib.load(pipeline_path)
                else:
                    PIPELINE = create_feature_pipeline()
                
                logger.info(f"Loaded local model from {local_model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {local_model_path}")
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        MODEL = None
        PIPELINE = None


def preprocess_transactions(transactions: List[TransactionFeatures]) -> pd.DataFrame:
    """Preprocess transaction data for model prediction."""
    # Convert to DataFrame
    data = pd.DataFrame([t.dict() for t in transactions])
    
    # Apply feature pipeline if available
    if PIPELINE is not None:
        try:
            processed_data = PIPELINE.transform(data)
        except Exception as e:
            logger.error(f"Error in pipeline transformation: {str(e)}")
            # Fallback to basic processing
            processed_data = data
    else:
        processed_data = data
    
    return processed_data


def predict_risk(features: pd.DataFrame) -> List[Dict[str, Any]]:
    """Make risk predictions using the loaded model."""
    if MODEL is None:
        raise ValueError("Model not loaded")
    
    predictions = []
    
    try:
        # Make predictions
        if hasattr(MODEL, 'predict_proba'):
            probabilities = MODEL.predict_proba(features)
            risk_scores = probabilities[:, 1]  # Probability of being high risk
        else:
            risk_scores = MODEL.predict(features)
        
        # Get customer IDs
        if 'CustomerId' in features.columns:
            customer_ids = features['CustomerId'].tolist()
        else:
            customer_ids = [f"customer_{i}" for i in range(len(features))]
        
        # Create predictions
        for i, (customer_id, score) in enumerate(zip(customer_ids, risk_scores)):
            # Determine risk class
            if score >= 0.7:
                risk_class = "HIGH"
            elif score >= 0.3:
                risk_class = "MEDIUM"
            else:
                risk_class = "LOW"
            
            # Calculate confidence (distance from decision boundary)
            confidence = abs(score - 0.5) * 2
            
            predictions.append({
                "customer_id": str(customer_id),
                "risk_score": float(score),
                "risk_class": risk_class,
                "confidence": float(confidence)
            })
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise
    
    return predictions


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - START_TIME
    
    return HealthCheck(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_version=MODEL_VERSION,
        uptime_seconds=round(uptime, 2)
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks = None):
    """
    Predict credit risk for transactions.
    
    Args:
        request: Prediction request with transaction data
        background_tasks: FastAPI background tasks
        
    Returns:
        Risk predictions for each customer
    """
    start_time = time.time()
    
    try:
        # Validate model is loaded
        if MODEL is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please check the health endpoint."
            )
        
        # Preprocess transactions
        logger.info(f"Processing {len(request.transactions)} transactions...")
        features = preprocess_transactions(request.transactions)
        
        # Make predictions
        predictions_data = predict_risk(features)
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Create response
        response = PredictionResponse(
            predictions=[RiskPrediction(**pred) for pred in predictions_data],
            model_version=MODEL_VERSION or "unknown",
            inference_time_ms=round(inference_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
        # Log prediction (optional background task)
        if background_tasks:
            background_tasks.add_task(
                log_prediction,
                request=request.dict(),
                response=response.dict()
            )
        
        logger.info(f"Prediction completed in {inference_time:.2f} ms")
        return response
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/predict/batch", response_model=PredictionResponse, tags=["Prediction"])
async def predict_batch(requests: List[PredictionRequest]):
    """
    Batch prediction endpoint for multiple requests.
    """
    # Combine all transactions
    all_transactions = []
    for req in requests:
        all_transactions.extend(req.transactions)
    
    # Create single request
    batch_request = PredictionRequest(transactions=all_transactions)
    
    # Call predict endpoint
    return await predict(batch_request)


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not loaded"
        )
    
    info = {
        "model_type": type(MODEL).__name__,
        "model_version": MODEL_VERSION,
        "features_required": getattr(MODEL, 'n_features_in_', 'unknown'),
        "training_date": "unknown",  # Could be extracted from model metadata
        "model_parameters": getattr(MODEL, 'get_params', lambda: {})()
    }
    
    return info


async def log_prediction(request: Dict, response: Dict):
    """Background task to log predictions."""
    # In production, log to database or monitoring system
    logger.info(f"Logged prediction: {len(response['predictions'])} predictions made")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    # Load model before starting server
    load_model()
    
    # Start server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )