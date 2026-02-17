import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import TemporalFeatureExtractor, AggregateFeatureEngineer, create_data_pipeline
from src.target_engineering import RFMCalculator, RiskLabelGenerator

def test_temporal_feature_extractor():
    """Test temporal feature extraction"""
    # Create test data
    test_data = pd.DataFrame({
        'TransactionStartTime': [
            '2018-11-15T02:18:49Z',
            '2018-11-15T14:30:00Z',
            '2018-12-01T09:15:30Z'
        ],
        'CustomerId': ['C1', 'C2', 'C1'],
        'Amount': [1000, 500, 200]
    })
    
    # Initialize and transform
    extractor = TemporalFeatureExtractor()
    result = extractor.transform(test_data.copy())
    
    # Assertions
    assert 'transaction_hour' in result.columns
    assert 'transaction_day' in result.columns
    assert 'transaction_month' in result.columns
    assert 'transaction_year' in result.columns
    assert result['transaction_hour'].iloc[0] == 2
    assert result['transaction_year'].iloc[0] == 2018
    
    print("✓ TemporalFeatureExtractor test passed!")

def test_aggregate_feature_engineer():
    """Test aggregate feature engineering"""
    # Create test data
    test_data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C2'],
        'Amount': [100, 200, 50, 75, 125],
        'Value': [100, 200, 50, 75, 125]
    })
    
    # Initialize and transform
    engineer = AggregateFeatureEngineer()
    result = engineer.transform(test_data.copy())
    
    # Check for expected column names from your implementation
    expected_columns = ['Amount_sum', 'Amount_mean', 'Amount_std', 'tx_count', 
                       'Value_sum', 'Value_mean', 'Value_std', 'amount_value_ratio', 
                       'transaction_frequency']
    
    for col in expected_columns:
        assert col in result.columns, f"Expected column '{col}' not found in {result.columns.tolist()}"
    
    # Check calculations
    c1_data = result[result['CustomerId'] == 'C1'].iloc[0]
    assert c1_data['Amount_sum'] == 300
    assert c1_data['Amount_mean'] == 150
    assert c1_data['tx_count'] == 2
    
    print("✓ AggregateFeatureEngineer test passed!")

def test_rfm_target_engineer():
    """Test RFM target engineering"""
    # Create test data
    test_data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C1', 'C2', 'C2', 'C3', 'C3'],
        'TransactionStartTime': [
            '2018-11-01T10:00:00Z',
            '2018-11-10T14:00:00Z',
            '2018-11-15T09:00:00Z',
            '2018-11-05T11:00:00Z',
            '2018-11-20T16:00:00Z',
            '2018-10-01T08:00:00Z',
            '2018-10-10T12:00:00Z'
        ],
        'Amount': [100, 200, 150, 300, 400, 50, 75],
        'Value': [100, 200, 150, 300, 400, 50, 75]
    })
    
    # Initialize RFMCalculator (no random_state parameter)
    engineer = RFMCalculator(snapshot_date='2018-11-25')
    rfm_df = engineer.calculate_rfm(test_data)
    
    # Assertions for RFM DataFrame
    assert 'recency' in rfm_df.columns
    assert 'frequency' in rfm_df.columns
    assert 'monetary' in rfm_df.columns
    assert len(rfm_df) == 3  # 3 unique customers
    
    # Test the complete workflow with RiskLabelGenerator
    generator = RiskLabelGenerator(random_state=42)
    rfm_clustered = generator.cluster_customers(rfm_df)
    target_df = generator.create_risk_labels(rfm_clustered)
    
    # Check the final target DataFrame
    assert 'CustomerId' in target_df.columns
    assert 'is_high_risk' in target_df.columns
    assert len(target_df) == 3
    assert target_df['is_high_risk'].dtype in [np.int64, np.int32, int]
    
    print("✓ RFMTargetEngineer test passed!")

def test_data_pipeline():
    """Test complete data pipeline creation"""
    pipeline = create_data_pipeline()
    
    # Assertions
    assert isinstance(pipeline, type(create_data_pipeline()))
    
    # Get step names
    step_names = list(pipeline.named_steps.keys())
    print(f"Pipeline steps: {step_names}")
    
    # Check for expected steps
    assert 'temporal' in step_names
    assert 'aggregate' in step_names
    assert 'woe' in step_names
    
    print("✓ Data pipeline creation test passed!")

if __name__ == "__main__":
    test_temporal_feature_extractor()
    test_aggregate_feature_engineer()
    test_rfm_target_engineer()
    test_data_pipeline()
    print("\nAll tests passed!")