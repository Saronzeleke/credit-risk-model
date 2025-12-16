"""
Proxy target variable engineering using RFM analysis and clustering.
Fully compatible with your dataset:
Columns used:
- CustomerId
- TransactionStartTime
- Amount
Target created: 'is_high_risk'
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# -----------------------------
# RFM Calculator
# -----------------------------
class RFMCalculator:
    """Calculate Recency, Frequency, Monetary metrics for customers."""

    def __init__(
        self,
        customer_id_col: str = "CustomerId",
        transaction_date_col: str = "TransactionStartTime",
        amount_col: str = "Amount",
        snapshot_date: Optional[str] = None,
    ):
        self.customer_id_col = customer_id_col
        self.transaction_date_col = transaction_date_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date
        self.rfm_data_ = None

    def calculate_rfm(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.transaction_date_col] = pd.to_datetime(
            data[self.transaction_date_col], errors="coerce"
        )

        snapshot_date = (
            pd.to_datetime(self.snapshot_date)
            if self.snapshot_date
            else data[self.transaction_date_col].max()
        )

        rfm_df = data.groupby(self.customer_id_col).agg(
            recency=(self.transaction_date_col, lambda x: (snapshot_date - x.max()).days),
            frequency=(self.customer_id_col, "count"),
            monetary=(self.amount_col, "sum"),
        ).reset_index()

        self.rfm_data_ = rfm_df.copy()
        return rfm_df


# -----------------------------
# Risk Label Generator
# -----------------------------
class RiskLabelGenerator:
    """Generate proxy risk labels using RFM clustering."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler_ = StandardScaler()
        self.kmeans_ = None
        self.risk_cluster_id_ = None

    def prepare_features(self, rfm_df: pd.DataFrame) -> np.ndarray:
        features = rfm_df[["recency", "frequency", "monetary"]].copy()
        # Log transform frequency and monetary
        features["frequency"] = np.log1p(features["frequency"])
        features["monetary"] = np.log1p(features["monetary"] - features["monetary"].min() + 1)
        scaled_features = self.scaler_.fit_transform(features)
        return scaled_features

    def cluster_customers(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        features = self.prepare_features(rfm_df)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        cluster_labels = self.kmeans_.fit_predict(features)
        rfm_clustered = rfm_df.copy()
        rfm_clustered["cluster"] = cluster_labels
        silhouette_avg = silhouette_score(features, cluster_labels)
        self.identify_high_risk_cluster(rfm_clustered)
        print(f"RFM clustering completed. Silhouette score: {silhouette_avg:.3f}")
        print(f"High-risk cluster identified: {self.risk_cluster_id_}")
        return rfm_clustered

    def identify_high_risk_cluster(self, rfm_clustered: pd.DataFrame):
        # Higher recency = more risky, lower frequency/monetary = more risky
        cluster_stats = rfm_clustered.groupby("cluster").agg(
            recency_mean=("recency", "mean"),
            frequency_mean=("frequency", "mean"),
            monetary_mean=("monetary", "mean"),
        ).reset_index()

        # Normalize
        for col in ["recency_mean", "frequency_mean", "monetary_mean"]:
            cluster_stats[f"{col}_norm"] = (
                cluster_stats[col] - cluster_stats[col].min()
            ) / (cluster_stats[col].max() - cluster_stats[col].min() + 1e-10)

        cluster_stats["risk_score"] = (
            cluster_stats["recency_mean_norm"]
            + (1 - cluster_stats["frequency_mean_norm"])
            + (1 - cluster_stats["monetary_mean_norm"])
        )

        high_risk_cluster = cluster_stats.loc[
            cluster_stats["risk_score"].idxmax(), "cluster"
        ]
        self.risk_cluster_id_ = int(high_risk_cluster)

    def create_risk_labels(self, rfm_clustered: pd.DataFrame) -> pd.DataFrame:
        rfm_labeled = rfm_clustered.copy()
        rfm_labeled["is_high_risk"] = (
            rfm_labeled["cluster"] == self.risk_cluster_id_
        ).astype(int)
        print(f"Target distribution: {rfm_labeled['is_high_risk'].value_counts().to_dict()}")
        return rfm_labeled[["CustomerId", "is_high_risk"]]


# -----------------------------
# Public API
# -----------------------------
def create_proxy_target(transaction_data: pd.DataFrame, snapshot_date: Optional[str] = None):
    """Create proxy target variable."""
    rfm_calculator = RFMCalculator(snapshot_date=snapshot_date)
    rfm_df = rfm_calculator.calculate_rfm(transaction_data)

    generator = RiskLabelGenerator()
    rfm_clustered = generator.cluster_customers(rfm_df)
    target_df = generator.create_risk_labels(rfm_clustered)
    return target_df


def integrate_target_variable(features_df: pd.DataFrame, target_df: pd.DataFrame):
    """Merge target variable into features dataset."""
    if "CustomerId" not in features_df.columns:
        features_df = features_df.copy()
        features_df["CustomerId"] = features_df.index
    combined_df = features_df.merge(target_df, on="CustomerId", how="left")
    missing_targets = combined_df["is_high_risk"].isna().sum()
    if missing_targets > 0:
        print(f"Warning: {missing_targets} samples missing target. Dropping them.")
        combined_df = combined_df.dropna(subset=["is_high_risk"])
    return combined_df


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    transactions = pd.read_csv("data/raw/data.csv")
    features_df = pd.read_csv("data/processed/features.csv")

    target_df = create_proxy_target(transactions, snapshot_date=None)
    final_df = integrate_target_variable(features_df, target_df)
    final_df.to_csv("data/processed/final_dataset.csv", index=False)

    print(f"Final dataset shape: {final_df.shape}")
    print(f"Target distribution:\n{final_df['is_high_risk'].value_counts(normalize=True)}")