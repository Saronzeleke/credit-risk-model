import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import (
    TemporalFeatureExtractor,
    AggregateFeatureEngineer,
    WoeTransformer,
    create_data_pipeline,
    process_data,
    preprocess_and_split
)
from src.data.load_data import load_raw_data


@pytest.fixture
def sample_transactions():
    """Create sample transaction data matching YOUR schema."""
    return pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 3],
        'TransactionStartTime': [
            '2024-01-01 10:00:00',
            '2024-01-01 14:30:00',
            '2024-01-02 09:15:00',
            '2024-01-03 16:45:00',
            '2024-01-04 11:00:00'
        ],
        'Amount': [100.0, 250.0, 75.0, 300.0, 150.0],
        'Value': [90.0, 225.0, 70.0, 270.0, 140.0],
        'CurrencyCode': ['USD', 'USD', 'EUR', 'EUR', 'GBP'],
        'CountryCode': ['US', 'US', 'DE', 'DE', 'UK'],
        'ProviderId': ['Prov1', 'Prov1', 'Prov2', 'Prov2', 'Prov3'],
        'ProductCategory': ['Electronics', 'Electronics', 'Clothing', 'Clothing', 'Food'],
        'ChannelId': ['Online', 'Online', 'Store', 'Store', 'Mobile'],
        'PricingStrategy': ['Fixed', 'Dynamic', 'Fixed', 'Dynamic', 'Fixed'],
        'is_high_risk': [0, 1, 0, 1, 0]
    })


def test_temporal_extractor(sample_transactions):
    """Test temporal feature extraction."""
    extractor = TemporalFeatureExtractor()
    transformed = extractor.transform(sample_transactions)
    
    # Check new columns
    expected_cols = [
        'transaction_hour', 'transaction_day', 'transaction_month',
        'transaction_year', 'transaction_dayofweek', 'is_weekend'
    ]
    for col in expected_cols:
        assert col in transformed.columns
    
    # Check original date column removed
    assert 'TransactionStartTime' not in transformed.columns
    
    # Check values are reasonable
    assert transformed['transaction_hour'].between(0, 23).all()
    assert transformed['transaction_day'].between(1, 31).all()
    assert transformed['is_weekend'].isin([0, 1]).all()


def test_aggregate_engineer(sample_transactions):
    """Test aggregate feature engineering."""
    engineer = AggregateFeatureEngineer()
    transformed = engineer.transform(sample_transactions)
    
    # Check new aggregate columns
    expected_agg_cols = [
        'Amount_sum', 'Amount_mean', 'Amount_std', 'tx_count',
        'Value_sum', 'Value_mean', 'Value_std',
        'amount_value_ratio', 'transaction_frequency'
    ]
    for col in expected_agg_cols:
        assert col in transformed.columns
    
    # Check rows preserved (merge kept all rows)
    assert len(transformed) == len(sample_transactions)
    
    # Check CustomerId 1 has correct aggregates
    cust1_data = transformed[transformed['CustomerId'] == 1]
    assert cust1_data['Amount_sum'].iloc[0] == 350.0  # 100 + 250
    assert cust1_data['tx_count'].iloc[0] == 2
    assert cust1_data['amount_value_ratio'].iloc[0] > 0


def test_woe_transformer(sample_transactions):
    """Test Weight of Evidence transformer."""
    categorical_features = ['CurrencyCode', 'CountryCode']
    transformer = WoeTransformer(categorical_features)
    
    X = sample_transactions[categorical_features]
    y = sample_transactions['is_high_risk']
    
    # Fit with y
    transformer.fit(X, y)
    
    # Transform
    transformed = transformer.transform(X)
    
    # Check mapping happened
    for col in categorical_features:
        assert col in transformed.columns
        assert transformed[col].dtype in [np.float64, float, int]
        assert not transformed[col].isna().any()
    
    # Test inference mode (fit without y)
    transformer_no_y = WoeTransformer(categorical_features)
    transformer_no_y.fit(X)  # Should return self without fitting
    transformed_no_y = transformer_no_y.transform(X)
    assert transformed_no_y.equals(X)  # Should be unchanged


def test_full_pipeline(sample_transactions):
    """Test the complete preprocessing pipeline."""
    # Save temp file
    temp_path = "temp_test_data.csv"
    sample_transactions.to_csv(temp_path, index=False)
    
    # Create and fit pipeline
    pipeline = create_data_pipeline()
    X = sample_transactions.drop('is_high_risk', axis=1)
    y = sample_transactions['is_high_risk']
    
    transformed = pipeline.fit_transform(X, y)
    
    # Check output
    assert transformed.shape[0] == len(sample_transactions)
    assert transformed.shape[1] > 0  # Should have many features
    assert not np.isnan(transformed).any().any()
    
    # Clean up
    import os
    os.remove(temp_path)


def test_process_data(sample_transactions):
    """Test process_data function."""
    temp_path = "temp_test_data.csv"
    sample_transactions.to_csv(temp_path, index=False)
    
    # Test with target
    X_trans, feats, pipeline = process_data(
        temp_path,
        target_column='is_high_risk',
        fit=True
    )
    
    assert len(feats) > 0
    assert X_trans.shape[0] == len(sample_transactions)
    assert isinstance(pipeline, Pipeline)
    
    # Clean up
    import os
    os.remove(temp_path)


def test_preprocess_and_split(sample_transactions):
    """Test full preprocessing with split."""
    temp_path = "temp_test_data.csv"
    sample_transactions.to_csv(temp_path, index=False)
    
    X_train, y_train, X_test, y_test, feats, pipeline = preprocess_and_split(
        raw_data_path=temp_path,
        target_column='is_high_risk'
    )
    
    # Check splits
    assert len(X_train) + len(X_test) == len(sample_transactions)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert len(feats) > 0
    
    # Check stratification
    train_risk_rate = y_train.mean()
    test_risk_rate = y_test.mean()
    assert abs(train_risk_rate - test_risk_rate) < 0.3  # Rough check
    
    # Clean up
    import os
    os.remove(temp_path)