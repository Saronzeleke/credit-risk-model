# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow
import joblib
import os
from typing import List, Optional
import uvicorn

from src.api.pydantic_models import PredictionRequest, PredictionResponse
from src.data_processing import create_data_pipeline

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk probability",
    version="1.0.0"
)

# Global variables for model and pipeline
model = None
pipeline = None

def load_model():
    """Load model from MLflow registry"""
    global model, pipeline
    
    try:
        # Load model from MLflow (adjust path as needed)
        model_path = "models/credit_risk_model"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            # Fallback: load from MLflow registry
            mlflow.set_tracking_uri("file:///mlruns")
            model_uri = "models:/credit_risk_model/Production"
            model = mlflow.sklearn.load_model(model_uri)
        
        # Load pipeline
        pipeline_path = "models/data_pipeline.pkl"
        if os.path.exists(pipeline_path):
            pipeline = joblib.load(pipeline_path)
        else:
            pipeline = create_data_pipeline()
        
        print("Model and pipeline loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Prediction API",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict credit risk probability"""
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Process data using pipeline
        if pipeline:
            processed_data = pipeline.transform(input_data)
        else:
            raise HTTPException(status_code=500, detail="Pipeline not loaded")
        
        # Make prediction
        if model:
            probability = model.predict_proba(processed_data)[0][1]
            prediction = 1 if probability >= 0.5 else 0
        else:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Create response
        response = PredictionResponse(
            customer_id=request.CustomerId,
            prediction=prediction,
            probability=float(probability),
            risk_level="HIGH" if prediction == 1 else "LOW"
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)