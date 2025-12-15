# src/target_engineering.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class RFMTargetEngineer:
    """Create proxy target variable using RFM and clustering"""
    
    def __init__(self, snapshot_date=None, n_clusters=3, random_state=42):
        self.snapshot_date = snapshot_date
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def calculate_rfm(self, df, customer_col='CustomerId', date_col='TransactionStartTime', 
                     amount_col='Amount', value_col='Value'):
        """Calculate RFM metrics for each customer"""
        
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Set snapshot date if not provided
        if self.snapshot_date is None:
            self.snapshot_date = df[date_col].max()
        else:
            self.snapshot_date = pd.to_datetime(self.snapshot_date)
        
        # Calculate RFM metrics
        rfm_df = df.groupby(customer_col).agg({
            date_col: lambda x: (self.snapshot_date - x.max()).days,  # Recency
            amount_col: 'count',  # Frequency
            value_col: 'sum'  # Monetary
        }).reset_index()
        
        # Rename columns
        rfm_df.columns = [customer_col, 'recency', 'frequency', 'monetary']
        
        # Handle negative monetary values (assume absolute value for risk assessment)
        rfm_df['monetary'] = rfm_df['monetary'].abs()
        
        return rfm_df
    
    def create_proxy_target(self, rfm_df):
        """Create proxy target variable using K-Means clustering"""
        
        # Select RFM features for clustering
        rfm_features = rfm_df[['recency', 'frequency', 'monetary']].copy()
        
        # Scale features
        rfm_scaled = self.scaler.fit_transform(rfm_features)
        
        # Apply K-Means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(rfm_scaled)
        
        # Add cluster labels
        rfm_df['cluster'] = clusters
        
        # Analyze clusters to identify high-risk group
        cluster_stats = rfm_df.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
        
        # High-risk: High recency (recent activity), low frequency, low monetary
        # Calculate risk score
        cluster_stats['risk_score'] = (
            cluster_stats['recency'].rank(ascending=True) +  # Higher recency = higher risk
            cluster_stats['frequency'].rank(ascending=False) +  # Lower frequency = higher risk
            cluster_stats['monetary'].rank(ascending=False)  # Lower monetary = higher risk
        )
        
        # Identify high-risk cluster (highest risk score)
        high_risk_cluster = cluster_stats['risk_score'].idxmax()
        
        # Create binary target variable
        rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
        
        print(f"Cluster Statistics:\n{cluster_stats}")
        print(f"\nHigh-risk cluster: {high_risk_cluster}")
        print(f"Risk distribution: {rfm_df['is_high_risk'].value_counts(normalize=True)}")
        
        return rfm_df[['CustomerId', 'is_high_risk']]
    
    def engineer_target_variable(self, df, customer_col='CustomerId'):
        """Complete target engineering pipeline"""
        
        # Calculate RFM
        rfm_df = self.calculate_rfm(df)
        
        # Create proxy target
        target_df = self.create_proxy_target(rfm_df)
        
        # Merge target back to original data
        df_with_target = df.merge(target_df, on=customer_col, how='left')
        
        return df_with_target