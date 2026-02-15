import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_synthetic_transactions(n_rows=1000, output_path="data/raw/transactions.csv"):
    """Generate synthetic data matching your column schema for testing."""
    np.random.seed(42)
    dates = [datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(n_rows)]
    customers = np.random.randint(1, 100, n_rows)
    
    df = pd.DataFrame({
        'CustomerId': customers,
        'TransactionStartTime': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
        'Amount': np.random.uniform(10, 1000, n_rows),
        'Value': np.random.uniform(5, 500, n_rows),
        'CurrencyCode': np.random.choice(['USD', 'EUR', 'GBP'], n_rows),
        'CountryCode': np.random.choice(['US', 'UK', 'DE'], n_rows),
        'ProviderId': np.random.choice(['Prov1', 'Prov2'], n_rows),
        'ProductCategory': np.random.choice(['Electronics', 'Clothing'], n_rows),
        'ChannelId': np.random.choice(['Online', 'Store'], n_rows),
        'PricingStrategy': np.random.choice(['Fixed', 'Dynamic'], n_rows),
        'is_high_risk': np.random.choice([0, 1], n_rows, p=[0.7, 0.3])  # Optional target
    })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic data generated: {n_rows} rows at {output_path}")

if __name__ == "__main__":
    generate_synthetic_transactions()