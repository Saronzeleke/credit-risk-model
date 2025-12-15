"""
Pydantic models for API request/response validation.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class TransactionFeatures(BaseModel):
    """Features for a single transaction."""
    CustomerId: str = Field(..., description="Unique customer identifier")
    TransactionDate: str = Field(..., description="Date of transaction (YYYY-MM-DD)")
    TransactionAmount: float = Field(..., description="Transaction amount", gt=0)
    TransactionType: Optional[str] = Field("PURCHASE", description="Type of transaction")
    MerchantCategory: Optional[str] = Field("RETAIL", description="Merchant category")
    
    @validator('TransactionDate')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('TransactionDate must be in YYYY-MM-DD format')


class PredictionRequest(BaseModel):
    """Request model for batch prediction."""
    transactions: List[TransactionFeatures] = Field(..., description="List of transactions")
    customer_id: Optional[str] = Field(None, description="Customer ID for single prediction")
    
    @validator('transactions')
    def validate_transactions(cls, v):
        if len(v) == 0:
            raise ValueError('At least one transaction is required')
        return v


class RiskPrediction(BaseModel):
    """Individual risk prediction."""
    customer_id: str = Field(..., description="Customer ID")
    risk_score: float = Field(..., description="Risk score (0-1)", ge=0, le=1)
    risk_class: str = Field(..., description="Risk classification")
    confidence: float = Field(..., description="Model confidence", ge=0, le=1)


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[RiskPrediction] = Field(..., description="List of risk predictions")
    model_version: str = Field(..., description="Model version used for prediction")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")