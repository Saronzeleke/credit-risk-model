import pandas as pd
import joblib
from typing import List, Dict, Any, Tuple
from pathlib import Path
from src.data.preprocess import create_data_pipeline


def load_model_and_pipeline(
    model_path: str = "models/best_model.joblib",
    pipeline_path: str = "models/pipeline.joblib"
) -> Tuple[Any, Any]:
    """
    Load trained model and fitted pipeline.
    
    Args:
        model_path: Path to model file
        pipeline_path: Path to pipeline file
    
    Returns:
        model: Loaded model
        pipeline: Loaded pipeline
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not Path(pipeline_path).exists():
        raise FileNotFoundError(f"Pipeline not found at {pipeline_path}")
    
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline


def predict(
    model: Any,
    pipeline: Any,
    input_data: List[Dict[str, Any]]
) -> List[float]:
    """
    Generate predictions on transaction data.
    
    Args:
        model: Trained model
        pipeline: Fitted preprocessing pipeline
        input_data: List of transaction dictionaries
    
    Returns:
        List of prediction probabilities
    """
    df = pd.DataFrame(input_data)
    
    # Ensure all required columns exist
    required_cols = [
        "CustomerId", "TransactionStartTime", "Amount", "Value",
        "CurrencyCode", "CountryCode", "ProviderId", "ProductCategory",
        "ChannelId", "PricingStrategy"
    ]
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Transform using pipeline
    processed = pipeline.transform(df)
    
    # Predict probabilities
    probs = model.predict_proba(processed)[:, 1]
    
    return probs.tolist()


if __name__ == "__main__":
    # Example usage
    try:
        model, pipeline = load_model_and_pipeline()
        
        # Sample transaction
        sample_data = [{
            "CustomerId": 12345,
            "TransactionStartTime": "2024-01-15 14:30:00",
            "Amount": 250.00,
            "Value": 225.00,
            "CurrencyCode": "USD",
            "CountryCode": "US",
            "ProviderId": "Prov1",
            "ProductCategory": "Electronics",
            "ChannelId": "Online",
            "PricingStrategy": "Fixed"
        }]
        
        probabilities = predict(model, pipeline, sample_data)
        print(f"Risk probability: {probabilities[0]:.2%}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")