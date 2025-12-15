"""
Data processing pipeline for credit risk modeling.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

# Scikit-learn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.feature_selection import VarianceThreshold

# Feature engineering
from xverse.transformer import WOE

import warnings
warnings.filterwarnings('ignore')


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from datetime columns."""
    
    def __init__(self, datetime_col: str = 'TransactionDate'):
        self.datetime_col = datetime_col
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        # Get feature names after transformation
        if self.datetime_col in X.columns:
            self.feature_names_ = [
                f'{self.datetime_col}_hour',
                f'{self.datetime_col}_day',
                f'{self.datetime_col}_month',
                f'{self.datetime_col}_year',
                f'{self.datetime_col}_dayofweek',
                f'{self.datetime_col}_is_weekend'
            ]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        
        if self.datetime_col in X.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(X_transformed[self.datetime_col]):
                X_transformed[self.datetime_col] = pd.to_datetime(
                    X_transformed[self.datetime_col], errors='coerce'
                )
            
            # Extract temporal features
            X_transformed[f'{self.datetime_col}_hour'] = X_transformed[self.datetime_col].dt.hour
            X_transformed[f'{self.datetime_col}_day'] = X_transformed[self.datetime_col].dt.day
            X_transformed[f'{self.datetime_col}_month'] = X_transformed[self.datetime_col].dt.month
            X_transformed[f'{self.datetime_col}_year'] = X_transformed[self.datetime_col].dt.year
            X_transformed[f'{self.datetime_col}_dayofweek'] = X_transformed[self.datetime_col].dt.dayofweek
            X_transformed[f'{self.datetime_col}_is_weekend'] = X_transformed[f'{self.datetime_col}_dayofweek'].isin([5, 6]).astype(int)
            
            # Drop original datetime column
            X_transformed = X_transformed.drop(columns=[self.datetime_col])
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_


class AggregateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create aggregate features per customer."""
    
    def __init__(self, customer_id_col: str = 'CustomerId', 
                 amount_col: str = 'TransactionAmount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.agg_features_ = None
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        if self.customer_id_col in X.columns and self.amount_col in X.columns:
            # Calculate aggregate statistics per customer
            agg_dict = {
                f'total_{self.amount_col}': 'sum',
                f'avg_{self.amount_col}': 'mean',
                f'count_{self.customer_id_col}': 'size',
                f'std_{self.amount_col}': 'std',
                f'min_{self.amount_col}': 'min',
                f'max_{self.amount_col}': 'max',
                f'median_{self.amount_col}': 'median'
            }
            
            self.agg_features_ = X.groupby(self.customer_id_col).agg({
                self.amount_col: ['sum', 'mean', 'std', 'min', 'max', 'median'],
                self.customer_id_col: 'size'
            })
            
            # Flatten column names
            self.agg_features_.columns = [
                f'{self.customer_id_col}_{stat}' 
                for stat in ['total', 'avg', 'std', 'min', 'max', 'median', 'count']
            ]
            
            self.feature_names_ = list(self.agg_features_.columns)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.agg_features_ is not None:
            X_transformed = X.copy()
            # Merge aggregate features back to original data
            X_transformed = X_transformed.merge(
                self.agg_features_, 
                left_on=self.customer_id_col,
                right_index=True,
                how='left'
            )
            return X_transformed
        return X
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handle missing values with various strategies."""
    
    def __init__(self, 
                 numeric_strategy: str = 'median',
                 categorical_strategy: str = 'most_frequent',
                 knn_impute: bool = False,
                 n_neighbors: int = 5):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.knn_impute = knn_impute
        self.n_neighbors = n_neighbors
        self.numeric_imputer_ = None
        self.categorical_imputer_ = None
        self.knn_imputer_ = None
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        # Identify column types
        self.numeric_cols_ = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not self.knn_impute:
            # Simple imputation
            if self.numeric_cols_:
                self.numeric_imputer_ = SimpleImputer(strategy=self.numeric_strategy)
                self.numeric_imputer_.fit(X[self.numeric_cols_])
            
            if self.categorical_cols_:
                self.categorical_imputer_ = SimpleImputer(strategy=self.categorical_strategy)
                self.categorical_imputer_.fit(X[self.categorical_cols_])
        else:
            # KNN imputation for all columns
            self.knn_imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
            self.knn_imputer_.fit(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        
        if not self.knn_impute:
            # Apply simple imputation
            if self.numeric_cols_ and self.numeric_imputer_ is not None:
                X_transformed[self.numeric_cols_] = self.numeric_imputer_.transform(
                    X_transformed[self.numeric_cols_]
                )
            
            if self.categorical_cols_ and self.categorical_imputer_ is not None:
                X_transformed[self.categorical_cols_] = self.categorical_imputer_.transform(
                    X_transformed[self.categorical_cols_]
                )
        else:
            # Apply KNN imputation
            if self.knn_imputer_ is not None:
                X_transformed[:] = self.knn_imputer_.transform(X_transformed)
        
        return X_transformed


def create_feature_pipeline(config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Create a comprehensive feature engineering pipeline.
    
    Args:
        config: Configuration dictionary for pipeline parameters
        
    Returns:
        sklearn.Pipeline: Feature engineering pipeline
    """
    if config is None:
        config = {
            'datetime_col': 'TransactionDate',
            'customer_id_col': 'CustomerId',
            'amount_col': 'TransactionAmount',
            'categorical_cols': ['TransactionType', 'MerchantCategory'],
            'numeric_strategy': 'median',
            'categorical_strategy': 'most_frequent',
            'use_woe': True,
            'scale_method': 'standard',  # 'standard', 'minmax', or None
        }
    
    # Define preprocessing steps
    preprocessing_steps = [
        ('temporal_features', TemporalFeatureExtractor(
            datetime_col=config['datetime_col']
        )),
        ('aggregate_features', AggregateFeatureEngineer(
            customer_id_col=config['customer_id_col'],
            amount_col=config['amount_col']
        )),
        ('handle_missing', MissingValueHandler(
            numeric_strategy=config['numeric_strategy'],
            categorical_strategy=config['categorical_strategy']
        ))
    ]
    
    # Create preprocessing pipeline
    preprocessing_pipeline = Pipeline(preprocessing_steps)
    
    # Create column transformers for different feature types
    numeric_features = []
    categorical_features = config.get('categorical_cols', [])
    
    # Note: Actual columns will be determined during fit
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config['numeric_strategy'])),
        ('scaler', StandardScaler() if config['scale_method'] == 'standard' 
                  else MinMaxScaler() if config['scale_method'] == 'minmax'
                  else 'passthrough')
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config['categorical_strategy'])),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Create final pipeline
    pipeline_steps = [
        ('preprocessing', preprocessing_pipeline),
        ('preprocessor', preprocessor),
        ('variance_threshold', VarianceThreshold(threshold=0.0))
    ]
    
    # Add WoE transformation if specified
    if config.get('use_woe', False):
        pipeline_steps.append(('woe', WOE()))
    
    # Build the complete pipeline
    feature_pipeline = Pipeline(pipeline_steps)
    
    return feature_pipeline


def process_data(raw_data_path: str, 
                 processed_data_path: str,
                 config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Pipeline]:
    """
    Process raw data and save processed data.
    
    Args:
        raw_data_path: Path to raw data file
        processed_data_path: Path to save processed data
        config: Configuration for feature engineering
        
    Returns:
        Tuple of processed DataFrame and fitted pipeline
    """
    # Load raw data
    raw_data = pd.read_csv(raw_data_path)
    
    # Create and fit pipeline
    feature_pipeline = create_feature_pipeline(config)
    
    # Fit and transform data
    # Note: For WoE, we need target variable. Without it, we'll skip WoE in initial processing
    processed_data = feature_pipeline.fit_transform(raw_data)
    
    # Get feature names
    try:
        feature_names = feature_pipeline.get_feature_names_out()
    except:
        # Fallback if get_feature_names_out is not available
        feature_names = [f'feature_{i}' for i in range(processed_data.shape[1])]
    
    # Convert back to DataFrame
    if isinstance(processed_data, np.ndarray):
        processed_data = pd.DataFrame(processed_data, columns=feature_names)
    
    # Save processed data
    processed_data.to_csv(processed_data_path, index=False)
    
    return processed_data, feature_pipeline


if __name__ == "__main__":
    # Example usage
    config = {
        'datetime_col': 'TransactionDate',
        'customer_id_col': 'CustomerId',
        'amount_col': 'TransactionAmount',
        'categorical_cols': ['TransactionType', 'MerchantCategory'],
        'numeric_strategy': 'median',
        'categorical_strategy': 'most_frequent',
        'use_woe': False,  # Will be applied later with target variable
        'scale_method': 'standard',
    }
    
    # Process data
    processed_data, pipeline = process_data(
        raw_data_path='data/raw/transactions.csv',
        processed_data_path='data/processed/features.csv',
        config=config
    )
    
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Features: {processed_data.columns.tolist()}")