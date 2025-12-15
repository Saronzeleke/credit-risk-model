import pandas as pd
from src.target_engineering import create_proxy_target, integrate_target_variable

# Load data
transactions = pd.read_csv('data/raw/data.csv')
features_df = pd.read_csv('data/processed/features.csv')

# If CustomerId missing in features, add it from index
if 'CustomerId' not in features_df.columns:
    features_df['CustomerId'] = features_df.index

# Engineer target
target_df = create_proxy_target(transactions, snapshot_date=None)

# Sanitize CustomerId: remove any non-numeric characters, then convert to int
features_df['CustomerId'] = features_df['CustomerId'].astype(str).str.extract('(\d+)')[0].astype(int)
target_df['CustomerId'] = target_df['CustomerId'].astype(str).str.extract('(\d+)')[0].astype(int)

# Merge and drop any missing targets
final_df = integrate_target_variable(features_df, target_df)

# Save final dataset
final_df.to_csv('data/processed/final_dataset.csv', index=False)

# Diagnostics
print(f"Final dataset shape: {final_df.shape}")
print(f"Target distribution:\n{final_df['is_high_risk'].value_counts(normalize=True)}")
