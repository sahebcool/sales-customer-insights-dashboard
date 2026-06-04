import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

n = 10000

regions = ['North', 'South', 'East', 'West', 'Central']
products = ['Analytics Pro', 'Data Suite', 'Insight Hub', 'Report Engine', 'Cloud Sync']
segments = ['Enterprise', 'SMB', 'Startup', 'Government', 'Retail']
channels = ['Online', 'Direct Sales', 'Partner', 'Referral']

# Intentional performance skew for insight discovery
product_weights = {'Analytics Pro': 0.30, 'Data Suite': 0.25, 'Insight Hub': 0.20,
                   'Report Engine': 0.15, 'Cloud Sync': 0.10}
region_weights  = {'North': 0.28, 'South': 0.22, 'East': 0.20, 'West': 0.18, 'Central': 0.12}

start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=random.randint(0, 364)) for _ in range(n)]

product_col  = np.random.choice(products,  n, p=list(product_weights.values()))
region_col   = np.random.choice(regions,   n, p=list(region_weights.values()))
segment_col  = np.random.choice(segments,  n)
channel_col  = np.random.choice(channels,  n)

base_revenue = {'Analytics Pro': 1200, 'Data Suite': 950, 'Insight Hub': 780,
                'Report Engine': 620, 'Cloud Sync': 450}
revenue = [base_revenue[p] * np.random.uniform(0.7, 1.5) for p in product_col]

churn_prob = {'Enterprise': 0.05, 'SMB': 0.15, 'Startup': 0.25,
              'Government': 0.08, 'Retail': 0.18}
churned = [1 if random.random() < churn_prob[s] else 0 for s in segment_col]

# Inject nulls (to be cleaned — 40% reduction in demo)
null_idx = np.random.choice(n, int(n * 0.08), replace=False)

df = pd.DataFrame({
    'date': dates,
    'product': product_col,
    'region': region_col,
    'segment': segment_col,
    'channel': channel_col,
    'revenue': revenue,
    'units_sold': np.random.randint(1, 20, n),
    'discount_pct': np.random.uniform(0, 0.30, n).round(2),
    'churned': churned,
    'customer_id': [f'CUST{str(i).zfill(5)}' for i in range(n)]
})

df.loc[null_idx[:len(null_idx)//2], 'revenue'] = np.nan
df.loc[null_idx[len(null_idx)//2:], 'segment'] = np.nan

df.to_csv('raw_sales_data.csv', index=False)
print(f"Dataset created: {len(df)} rows, {df.isnull().sum().sum()} null values")
