# src/api/pydantic_models.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str = Field(..., min_length=3, max_length=3)
    CountryCode: str = Field(..., min_length=3, max_length=3)
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str
    PricingStrategy: int = Field(..., ge=1, le=10)
    FraudResult: Optional[int] = 0
    
    class Config:
        schema_extra = {
            "example": {
                "TransactionId": "TransactionId_76871",
                "BatchId": "BatchId_36123",
                "AccountId": "AccountId_3957",
                "SubscriptionId": "SubscriptionId_887",
                "CustomerId": "CustomerId_4406",
                "CurrencyCode": "UGX",
                "CountryCode": "256",
                "ProviderId": "ProviderId_6",
                "ProductId": "ProductId_10",
                "ProductCategory": "airtime",
                "ChannelId": "ChannelId_3",
                "Amount": 1000.0,
                "Value": 1000.0,
                "TransactionStartTime": "2018-11-15T02:18:49Z",
                "PricingStrategy": 2,
                "FraudResult": 0
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    customer_id: str
    prediction: int
    probability: float
    risk_level: str
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CustomerId_4406",
                "prediction": 1,
                "probability": 0.85,
                "risk_level": "HIGH"
            }
        }