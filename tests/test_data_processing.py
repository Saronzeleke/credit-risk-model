"""
Unit tests for data processing functions.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import (
    TemporalFeatureExtractor,
    AggregateFeatureEngineer,
    MissingValueHandler,
    create_feature_pipeline
)
from target_engineering import RFMCalculator, RiskLabelGenerator


class TestTemporalFeatureExtractor:
    """Test TemporalFeatureExtractor class."""
    
    def setup_method(self):
        """Setup test data."""
        self.test_data = pd.DataFrame({
            'TransactionDate': pd.date_range('2024-01-01', periods=10, freq='D'),
            'CustomerId': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            'TransactionAmount': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
        })
        
        self.extractor = TemporalFeatureExtractor(datetime_col='TransactionDate')
    
    def test_fit_transform(self):
        """Test fit and transform methods."""
        # Fit the extractor
        self.extractor.fit(self.test_data)
        
        # Transform data
        transformed = self.extractor.transform(self.test_data)
        
        # Check that original column is removed
        assert 'TransactionDate' not in transformed.columns
        
        # Check that new columns are created
        expected_columns = [
            'TransactionDate_hour',
            'TransactionDate_day',
            'TransactionDate_month',
            'TransactionDate_year',
            'TransactionDate_dayofweek',
            'TransactionDate_is_weekend'
        ]
        
        for col in expected_columns:
            assert col in transformed.columns
        
        # Check data types
        assert transformed['TransactionDate_hour'].dtype in [np.int32, np.int64]
        assert transformed['TransactionDate_is_weekend'].dtype in [np.int32, np.int64]
    
    def test_missing_datetime_column(self):
        """Test behavior when datetime column is missing."""
        data_without_date = self.test_data.drop(columns=['TransactionDate'])
        transformed = self.extractor.transform(data_without_date)
        
        # Should return the same data unchanged
        assert transformed.equals(data_without_date)


class TestAggregateFeatureEngineer:
    """Test AggregateFeatureEngineer class."""
    
    def setup_method(self):
        """Setup test data."""
        self.test_data = pd.DataFrame({
            'CustomerId': [1, 1, 1, 2, 2, 3],
            'TransactionAmount': [100, 200, 300, 150, 250, 500]
        })
        
        self.engineer = AggregateFeatureEngineer(
            customer_id_col='CustomerId',
            amount_col='TransactionAmount'
        )
    
    def test_fit_transform(self):
        """Test fit and transform methods."""
        # Fit the engineer
        self.engineer.fit(self.test_data)
        
        # Transform data
        transformed = self.engineer.transform(self.test_data)
        
        # Check that aggregate features are added
        expected_columns = [
            'CustomerId_total',
            'CustomerId_avg',
            'CustomerId_std',
            'CustomerId_min',
            'CustomerId_max',
            'CustomerId_median',
            'CustomerId_count'
        ]
        
        for col in expected_columns:
            assert col in transformed.columns
        
        # Check calculated values for CustomerId 1
        customer_1_data = transformed[transformed['CustomerId'] == 1]
        
        assert customer_1_data['CustomerId_total'].iloc[0] == 600  # 100+200+300
        assert customer_1_data['CustomerId_avg'].iloc[0] == 200   # (100+200+300)/3
        assert customer_1_data['CustomerId_count'].iloc[0] == 3
    
    def test_missing_columns(self):
        """Test behavior when required columns are missing."""
        data_missing = self.test_data.drop(columns=['TransactionAmount'])
        engineer = AggregateFeatureEngineer()
        
        # Should handle missing columns gracefully
        engineer.fit(data_missing)
        transformed = engineer.transform(data_missing)
        
        # Should return original data unchanged
        assert transformed.equals(data_missing)


class TestMissingValueHandler:
    """Test MissingValueHandler class."""
    
    def setup_method(self):
        """Setup test data with missing values."""
        self.test_data = pd.DataFrame({
            'numeric_1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'numeric_2': [np.nan, 2.0, 3.0, np.nan, 5.0],
            'categorical_1': ['A', 'B', np.nan, 'A', 'C'],
            'categorical_2': ['X', np.nan, 'Y', 'Z', np.nan]
        })
        
        self.handler = MissingValueHandler(
            numeric_strategy='mean',
            categorical_strategy='most_frequent'
        )
    
    def test_fit_transform(self):
        """Test fit and transform methods with simple imputation."""
        # Fit the handler
        self.handler.fit(self.test_data)
        
        # Transform data
        transformed = self.handler.transform(self.test_data)
        
        # Check that there are no missing values
        assert transformed.isna().sum().sum() == 0
        
        # Check that numeric missing values are filled with mean
        numeric_mean = self.test_data['numeric_1'].mean()
        assert transformed['numeric_1'].iloc[2] == pytest.approx(numeric_mean, 0.01)
    
    def test_knn_imputation(self):
        """Test KNN imputation."""
        handler_knn = MissingValueHandler(knn_impute=True, n_neighbors=2)
        
        # Fit and transform
        handler_knn.fit(self.test_data)
        transformed = handler_knn.transform(self.test_data)
        
        # Check that there are no missing values
        assert transformed.isna().sum().sum() == 0


class TestRFMCalculator:
    """Test RFMCalculator class."""
    
    def setup_method(self):
        """Setup test transaction data."""
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        self.test_data = pd.DataFrame({
            'CustomerId': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4] * 2,
            'TransactionDate': dates,
            'TransactionAmount': [100, 200, 150, 300, 250, 50, 75, 100, 125, 400] * 2
        })
        
        self.calculator = RFMCalculator(
            customer_id_col='CustomerId',
            transaction_date_col='TransactionDate',
            amount_col='TransactionAmount',
            snapshot_date='2024-01-25'
        )
    
    def test_calculate_rfm(self):
        """Test RFM calculation."""
        rfm_df = self.calculator.calculate_rfm(self.test_data)
        
        # Check columns
        expected_columns = ['CustomerId', 'recency', 'frequency', 'monetary']
        assert all(col in rfm_df.columns for col in expected_columns)
        
        # Check shape (should have 4 unique customers)
        assert rfm_df.shape[0] == 4
        
        # Check RFM values for a specific customer
        customer_1_data = rfm_df[rfm_df['CustomerId'] == 1]
        
        # Customer 1 has 6 transactions, last on 2024-01-20
        # Recency = 5 days (from 2024-01-25 to 2024-01-20)
        assert customer_1_data['recency'].iloc[0] == 5
        assert customer_1_data['frequency'].iloc[0] == 6  # 3 transactions * 2 (due to duplication)
        assert customer_1_data['monetary'].iloc[0] == (100 + 200 + 150) * 2
    
    def test_calculate_rfm_scores(self):
        """Test RFM scoring."""
        rfm_df = self.calculator.calculate_rfm(self.test_data)
        rfm_scores = self.calculator.calculate_rfm_scores(rfm_df)
        
        # Check score columns
        score_columns = ['recency_score', 'frequency_score', 'monetary_score', 'rfm_score']
        assert all(col in rfm_scores.columns for col in score_columns)
        
        # Check score ranges (1-5)
        for col in ['recency_score', 'frequency_score', 'monetary_score']:
            assert rfm_scores[col].min() >= 1
            assert rfm_scores[col].max() <= 5


class TestRiskLabelGenerator:
    """Test RiskLabelGenerator class."""
    
    def setup_method(self):
        """Setup test RFM data."""
        self.rfm_data = pd.DataFrame({
            'CustomerId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'recency': [100, 50, 25, 10, 5, 200, 150, 75, 30, 15],
            'frequency': [1, 2, 5, 10, 15, 1, 2, 3, 6, 8],
            'monetary': [100, 500, 1000, 5000, 10000, 50, 200, 800, 2000, 4000]
        })
        
        self.generator = RiskLabelGenerator(
            n_clusters=3,
            random_state=42
        )
    
    def test_cluster_customers(self):
        """Test customer clustering."""
        rfm_clustered, silhouette_score = self.generator.cluster_customers(self.rfm_data)
        
        # Check that cluster column is added
        assert 'cluster' in rfm_clustered.columns
        
        # Check cluster values
        clusters = rfm_clustered['cluster'].unique()
        assert len(clusters) == 3  # Should have 3 clusters
        
        # Silhouette score should be between -1 and 1
        assert -1 <= silhouette_score <= 1
    
    def test_identify_high_risk_cluster(self):
        """Test identification of high-risk cluster."""
        rfm_clustered, _ = self.generator.cluster_customers(self.rfm_data)
        high_risk_cluster = self.generator.identify_high_risk_cluster(rfm_clustered)
        
        # High-risk cluster should be one of the cluster IDs
        assert high_risk_cluster in rfm_clustered['cluster'].unique()
    
    def test_create_risk_labels(self):
        """Test creation of risk labels."""
        rfm_clustered, _ = self.generator.cluster_customers(self.rfm_data)
        rfm_labeled = self.generator.create_risk_labels(rfm_clustered)
        
        # Check that risk label column is added
        assert 'is_high_risk' in rfm_labeled.columns
        
        # Check that risk labels are binary (0 or 1)
        assert set(rfm_labeled['is_high_risk'].unique()).issubset({0, 1})
        
        # Check that exactly one cluster is marked as high risk
        high_risk_cluster = rfm_labeled[rfm_labeled['is_high_risk'] == 1]['cluster'].unique()
        assert len(high_risk_cluster) == 1


def test_create_feature_pipeline():
    """Test creation of feature pipeline."""
    pipeline = create_feature_pipeline()
    
    # Check that pipeline is created
    assert pipeline is not None
    
    # Check that pipeline has expected steps
    expected_steps = ['preprocessing', 'preprocessor', 'variance_threshold']
    for step in expected_steps:
        assert step in pipeline.named_steps
    
    # Test with configuration
    config = {
        'datetime_col': 'TransactionDate',
        'customer_id_col': 'CustomerId',
        'amount_col': 'TransactionAmount',
        'categorical_cols': ['Category'],
        'use_woe': False
    }
    
    pipeline_with_config = create_feature_pipeline(config)
    assert pipeline_with_config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])