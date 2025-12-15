# src/data_processing.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from TransactionStartTime"""
    def __init__(self, date_column='TransactionStartTime'):
        self.date_column = date_column
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column])
        X['transaction_hour'] = X[self.date_column].dt.hour
        X['transaction_day'] = X[self.date_column].dt.day
        X['transaction_month'] = X[self.date_column].dt.month
        X['transaction_year'] = X[self.date_column].dt.year
        X['transaction_dayofweek'] = X[self.date_column].dt.dayofweek
        X['is_weekend'] = X['transaction_dayofweek'].isin([5, 6]).astype(int)
        return X.drop(columns=[self.date_column])

class AggregateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create aggregate features per customer"""
    def __init__(self, groupby_column='CustomerId'):
        self.groupby_column = groupby_column
        self.agg_features = None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Aggregate features per customer
        agg_features = X.groupby(self.groupby_column).agg({
            'Amount': ['sum', 'mean', 'std', 'count'],
            'Value': ['sum', 'mean', 'std']
        })
        
        # Flatten column names
        agg_features.columns = [
            'total_amount', 'avg_amount', 'std_amount', 'transaction_count',
            'total_value', 'avg_value', 'std_value'
        ]
        
        # Fill NaN values for std
        agg_features = agg_features.fillna(0)
        
        # Calculate additional features
        agg_features['amount_value_ratio'] = agg_features['total_amount'] / (agg_features['total_value'] + 1e-6)
        agg_features['transaction_frequency'] = agg_features['transaction_count'] / 30  # Assuming 30-day period
        
        self.agg_features = agg_features
        
        # Merge aggregated features back to original data
        X = X.merge(agg_features, on=self.groupby_column, how='left')
        return X

class WoeTransformer(BaseEstimator, TransformerMixin):
    """Weight of Evidence transformation for categorical features"""
    def __init__(self, categorical_features=None, target_column='is_high_risk'):
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.woe_dict = {}
        
    def fit(self, X, y):
        if self.categorical_features is None:
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.categorical_features:
            if col in X.columns and col != self.target_column:
                woe_df = self.calculate_woe(X[col], y)
                self.woe_dict[col] = woe_df
        return self
    
    def calculate_woe(self, feature, target):
        woe_df = pd.DataFrame({
            'feature': feature,
            'target': target
        })
        
        # Calculate distribution
        total_good = (target == 0).sum()
        total_bad = (target == 1).sum()
        
        woe_stats = woe_df.groupby('feature')['target'].agg(['count', 'sum'])
        woe_stats.columns = ['total', 'bad']
        woe_stats['good'] = woe_stats['total'] - woe_stats['bad']
        
        # Avoid division by zero
        woe_stats['bad_rate'] = woe_stats['bad'] / (total_bad + 1e-6)
        woe_stats['good_rate'] = woe_stats['good'] / (total_good + 1e-6)
        woe_stats['woe'] = np.log(woe_stats['good_rate'] / (woe_stats['bad_rate'] + 1e-6) + 1e-6)
        
        return woe_stats[['woe']]
    
    def transform(self, X):
        X = X.copy()
        for col, woe_df in self.woe_dict.items():
            if col in X.columns:
                X[col] = X[col].map(woe_df['woe']).fillna(0)
        return X

def create_data_pipeline(target_column='is_high_risk'):
    """Create the complete data processing pipeline"""
    
    # Numerical features
    numerical_features = [
        'Amount', 'Value', 'total_amount', 'avg_amount', 'std_amount',
        'transaction_count', 'total_value', 'avg_value', 'std_value',
        'amount_value_ratio', 'transaction_frequency'
    ]
    
    # Categorical features
    categorical_features = [
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
        'ChannelId', 'PricingStrategy'
    ]
    
    # Temporal features (already extracted)
    temporal_features = [
        'transaction_hour', 'transaction_day', 'transaction_month',
        'transaction_year', 'transaction_dayofweek', 'is_weekend'
    ]
    
    # Create transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('temp', 'passthrough', temporal_features)
        ])
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('temporal_extractor', TemporalFeatureExtractor()),
        ('aggregate_engineer', AggregateFeatureEngineer()),
        ('woe_transformer', WoeTransformer()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline

def process_data(raw_data_path, pipeline=None, fit=True, target=None):
    """Process data using the pipeline"""
    # Load data
    df = pd.read_csv(raw_data_path)
    
    # Create pipeline if not provided
    if pipeline is None:
        pipeline = create_data_pipeline()
    
    # Fit and transform
    if fit and target is not None:
        processed_data = pipeline.fit_transform(df, target)
    elif fit:
        processed_data = pipeline.fit_transform(df)
    else:
        processed_data = pipeline.transform(df)
    
    # Get feature names
    if hasattr(pipeline, 'get_feature_names_out'):
        feature_names = pipeline.get_feature_names_out()
    else:
        feature_names = [f'feature_{i}' for i in range(processed_data.shape[1])]
    
    return processed_data, feature_names, pipeline