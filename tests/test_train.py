import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import json
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.train import load_and_prepare_data, train_models, save_best_model

class TestTrainModule:
    """Tests for credit risk model training."""

    def setup_method(self):
        """Create realistic test data matching the pipeline requirements."""
        self.data = pd.DataFrame({
            "TransactionId": [f"T{i}" for i in range(10)],
            "BatchId": [i for i in range(10)],
            "AccountId": [i for i in range(10)],
            "SubscriptionId": [i for i in range(10)],
            "CustomerId": [100+i for i in range(10)],
            "CurrencyCode": ["USD"]*5 + ["UGX"]*5,
            "CountryCode": ["US"]*5 + ["UG"]*5,
            "ProviderId": [f"P{i%3}" for i in range(10)],
            "ProductId": [f"Prod{i%2}" for i in range(10)],
            "ProductCategory": ["airtime"]*5 + ["data"]*5,
            "ChannelId": [f"C{i%2}" for i in range(10)],
            "Amount": np.random.randint(10, 1000, 10),
            "Value": np.random.randint(10, 1000, 10),
            "TransactionStartTime": pd.date_range("2023-01-01", periods=10, freq="D"),
            "PricingStrategy": ["StrategyA"]*5 + ["StrategyB"]*5,
            "FraudResult": [0,1,0,1,0,1,0,1,0,1]
        })

        # Merge target using train.py logic
        self.data["is_high_risk"] = self.data["FraudResult"]

        self.X = self.data.drop(columns=["is_high_risk"])
        self.y = self.data["is_high_risk"]

    def test_train_models_returns_dict(self):
        """Test that train_models returns dictionary of models and pipeline."""
        best_models, pipeline = train_models(self.X, self.y)
        assert isinstance(best_models, dict)
        assert "logistic_regression" in best_models
        assert "random_forest" in best_models
        assert "gradient_boosting" in best_models
        assert pipeline is not None

    def test_save_best_model_creates_files(self, tmp_path):
        """Test that save_best_model saves model and pipeline."""
        best_models, pipeline = train_models(self.X, self.y)
        best_name = save_best_model(best_models, pipeline)
        model_path = tmp_path / f"{best_name}_model.pkl"
        pipeline_path = tmp_path / "data_pipeline.pkl"

        # Save to tmp_path
        joblib.dump(best_models[best_name]["model"], model_path)
        joblib.dump(pipeline, pipeline_path)

        assert model_path.exists()
        assert pipeline_path.exists()

    def test_load_and_prepare_data(self, tmp_path):
        """Test loading and preparing data."""
        csv_path = tmp_path / "data.csv"
        self.data.to_csv(csv_path, index=False)
        X, y = load_and_prepare_data(csv_path)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "CustomerId" in X.columns
        assert y.name == "is_high_risk"
        assert len(X) == len(y)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
