"""
Proxy target variable engineering using RFM analysis and clustering.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')


class RFMCalculator:
    """Calculate Recency, Frequency, Monetary metrics for customers."""
    
    def __init__(self, 
                 customer_id_col: str = 'CustomerId',
                 transaction_date_col: str = 'TransactionDate',
                 amount_col: str = 'TransactionAmount',
                 snapshot_date: Optional[str] = None):
        """
        Initialize RFM calculator.
        
        Args:
            customer_id_col: Name of customer ID column
            transaction_date_col: Name of transaction date column
            amount_col: Name of transaction amount column
            snapshot_date: Reference date for recency calculation (YYYY-MM-DD)
                          If None, uses max date in data
        """
        self.customer_id_col = customer_id_col
        self.transaction_date_col = transaction_date_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date
        self.rfm_data_ = None
        
    def calculate_rfm(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics from transaction data.
        
        Args:
            data: Transaction data with customer IDs, dates, and amounts
            
        Returns:
            DataFrame with RFM metrics per customer
        """
        # Convert date column to datetime
        data[self.transaction_date_col] = pd.to_datetime(
            data[self.transaction_date_col], errors='coerce'
        )
        
        # Determine snapshot date
        if self.snapshot_date is None:
            snapshot_date = data[self.transaction_date_col].max()
        else:
            snapshot_date = pd.to_datetime(self.snapshot_date)
        
        # Calculate RFM metrics
        rfm_df = data.groupby(self.customer_id_col).agg({
            self.transaction_date_col: lambda x: (snapshot_date - x.max()).days,  # Recency
            self.customer_id_col: 'count',  # Frequency
            self.amount_col: 'sum'  # Monetary
        }).rename(columns={
            self.transaction_date_col: 'recency',
            self.customer_id_col: 'frequency',
            self.amount_col: 'monetary'
        }).reset_index()
        
        # Store for later use
        self.rfm_data_ = rfm_df.copy()
        
        return rfm_df
    
    def calculate_rfm_scores(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM scores (1-5 scale) for each metric.
        
        Args:
            rfm_df: DataFrame with raw RFM metrics
            
        Returns:
            DataFrame with RFM scores
        """
        rfm_scores = rfm_df.copy()
        
        # Create scoring for each metric (higher is better)
        # Recency: Lower recency (more recent) gets higher score
        rfm_scores['recency_score'] = pd.qcut(
            rfm_scores['recency'], 
            q=5, 
            labels=[5, 4, 3, 2, 1]
        ).astype(int)
        
        # Frequency: Higher frequency gets higher score
        rfm_scores['frequency_score'] = pd.qcut(
            rfm_scores['frequency'], 
            q=5, 
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Monetary: Higher monetary gets higher score
        rfm_scores['monetary_score'] = pd.qcut(
            rfm_scores['monetary'], 
            q=5, 
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Calculate combined RFM score
        rfm_scores['rfm_score'] = (
            rfm_scores['recency_score'].astype(str) + 
            rfm_scores['frequency_score'].astype(str) + 
            rfm_scores['monetary_score'].astype(str)
        ).astype(int)
        
        return rfm_scores


class RiskLabelGenerator:
    """Generate proxy risk labels using RFM clustering."""
    
    def __init__(self, 
                 n_clusters: int = 3,
                 random_state: int = 42,
                 risk_cluster_threshold: float = 0.7):
        """
        Initialize risk label generator.
        
        Args:
            n_clusters: Number of clusters for K-Means
            random_state: Random seed for reproducibility
            risk_cluster_threshold: Threshold for identifying high-risk cluster
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.risk_cluster_threshold = risk_cluster_threshold
        self.scaler_ = StandardScaler()
        self.kmeans_ = None
        self.cluster_labels_ = None
        self.risk_cluster_id_ = None
        
    def prepare_features(self, rfm_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare RFM features for clustering.
        
        Args:
            rfm_df: DataFrame with RFM metrics
            
        Returns:
            Scaled feature array
        """
        # Select RFM metrics
        features = rfm_df[['recency', 'frequency', 'monetary']].copy()
        
        # Handle zeros and negative values (log transformation)
        features['frequency'] = np.log1p(features['frequency'])
        features['monetary'] = np.log1p(features['monetary'] - features['monetary'].min() + 1)
        
        # Scale features
        scaled_features = self.scaler_.fit_transform(features)
        
        return scaled_features
    
    def cluster_customers(self, rfm_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Cluster customers based on RFM metrics.
        
        Args:
            rfm_df: DataFrame with RFM metrics
            
        Returns:
            Tuple of (DataFrame with clusters, silhouette score)
        """
        # Prepare features
        features = self.prepare_features(rfm_df)
        
        # Apply K-Means clustering
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        cluster_labels = self.kmeans_.fit_predict(features)
        self.cluster_labels_ = cluster_labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(features, cluster_labels)
        
        # Add cluster labels to RFM data
        rfm_clustered = rfm_df.copy()
        rfm_clustered['cluster'] = cluster_labels
        
        return rfm_clustered, silhouette_avg
    
    def identify_high_risk_cluster(self, rfm_clustered: pd.DataFrame) -> int:
        """
        Identify which cluster represents high-risk customers.
        
        Args:
            rfm_clustered: DataFrame with cluster labels
            
        Returns:
            Cluster ID for high-risk group
        """
        # Calculate average RFM metrics per cluster
        cluster_stats = rfm_clustered.groupby('cluster').agg({
            'recency': 'mean',  # Higher is worse (less recent)
            'frequency': 'mean',  # Lower is worse
            'monetary': 'mean'  # Lower is worse
        }).reset_index()
        
        # Create risk score (higher = more risky)
        # Normalize each metric
        for col in ['recency', 'frequency', 'monetary']:
            cluster_stats[f'{col}_norm'] = (
                cluster_stats[col] - cluster_stats[col].min()
            ) / (cluster_stats[col].max() - cluster_stats[col].min() + 1e-10)
        
        # For recency, higher is worse
        # For frequency and monetary, lower is worse (so invert)
        cluster_stats['risk_score'] = (
            cluster_stats['recency_norm'] +  # Higher recency = more risky
            (1 - cluster_stats['frequency_norm']) +  # Lower frequency = more risky
            (1 - cluster_stats['monetary_norm'])  # Lower monetary = more risky
        )
        
        # Identify high-risk cluster (highest risk score)
        high_risk_cluster = cluster_stats.loc[
            cluster_stats['risk_score'].idxmax(), 'cluster'
        ]
        
        self.risk_cluster_id_ = int(high_risk_cluster)
        
        return self.risk_cluster_id_
    
    def create_risk_labels(self, rfm_clustered: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary risk labels based on clustering.
        
        Args:
            rfm_clustered: DataFrame with cluster labels
            
        Returns:
            DataFrame with risk labels
        """
        if self.risk_cluster_id_ is None:
            self.identify_high_risk_cluster(rfm_clustered)
        
        # Create binary risk label
        rfm_labeled = rfm_clustered.copy()
        rfm_labeled['is_high_risk'] = (
            rfm_labeled['cluster'] == self.risk_cluster_id_
        ).astype(int)
        
        return rfm_labeled


def create_proxy_target(transaction_data: pd.DataFrame,
                        config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Create proxy target variable using RFM analysis and clustering.
    
    Args:
        transaction_data: Raw transaction data
        config: Configuration dictionary
        
    Returns:
        DataFrame with customer IDs and risk labels
    """
    if config is None:
        config = {
            'customer_id_col': 'CustomerId',
            'transaction_date_col': 'TransactionDate',
            'amount_col': 'TransactionAmount',
            'snapshot_date': None,
            'n_clusters': 3,
            'random_state': 42
        }
    
    # Step 1: Calculate RFM metrics
    rfm_calculator = RFMCalculator(
        customer_id_col=config['customer_id_col'],
        transaction_date_col=config['transaction_date_col'],
        amount_col=config['amount_col'],
        snapshot_date=config['snapshot_date']
    )
    
    rfm_df = rfm_calculator.calculate_rfm(transaction_data)
    
    # Step 2: Cluster customers
    label_generator = RiskLabelGenerator(
        n_clusters=config['n_clusters'],
        random_state=config['random_state']
    )
    
    rfm_clustered, silhouette_score = label_generator.cluster_customers(rfm_df)
    
    # Step 3: Create risk labels
    rfm_labeled = label_generator.create_risk_labels(rfm_clustered)
    
    print(f"RFM clustering completed with silhouette score: {silhouette_score:.3f}")
    print(f"High-risk cluster identified: {label_generator.risk_cluster_id_}")
    print(f"Risk distribution: {rfm_labeled['is_high_risk'].value_counts().to_dict()}")
    
    return rfm_labeled[['CustomerId', 'is_high_risk']]


def integrate_target_variable(features_df: pd.DataFrame,
                             target_df: pd.DataFrame,
                             customer_id_col: str = 'CustomerId') -> pd.DataFrame:
    """
    Integrate target variable into feature dataset.
    
    Args:
        features_df: DataFrame with engineered features
        target_df: DataFrame with target variable
        customer_id_col: Name of customer ID column
        
    Returns:
        Combined DataFrame with features and target
    """
    # Merge target variable
    if customer_id_col in features_df.columns:
        combined_df = features_df.merge(
            target_df,
            on=customer_id_col,
            how='left'
        )
    else:
        # If CustomerId is not in features, add it from index
        combined_df = features_df.copy()
        combined_df[customer_id_col] = combined_df.index
        combined_df = combined_df.merge(
            target_df,
            on=customer_id_col,
            how='left'
        )
    
    # Check for missing targets
    missing_targets = combined_df['is_high_risk'].isna().sum()
    if missing_targets > 0:
        print(f"Warning: {missing_targets} samples missing target variable")
        # Option 1: Drop rows with missing target
        combined_df = combined_df.dropna(subset=['is_high_risk'])
        # Option 2: Impute (not recommended for target variable)
    
    return combined_df


if __name__ == "__main__":
    # Example usage
    # Load transaction data
    transactions = pd.read_csv('data/raw/transactions.csv')
    
    # Create proxy target
    config = {
        'customer_id_col': 'CustomerId',
        'transaction_date_col': 'TransactionDate',
        'amount_col': 'TransactionAmount',
        'snapshot_date': '2024-12-31',
        'n_clusters': 3,
        'random_state': 42
    }
    
    target_df = create_proxy_target(transactions, config)
    
    # Save target variable
    target_df.to_csv('data/processed/target_labels.csv', index=False)
    
    # Load features and integrate target
    features_df = pd.read_csv('data/processed/features.csv')
    final_df = integrate_target_variable(features_df, target_df)
    
    # Save final dataset
    final_df.to_csv('data/processed/final_dataset.csv', index=False)
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Target distribution:\n{final_df['is_high_risk'].value_counts(normalize=True)}")