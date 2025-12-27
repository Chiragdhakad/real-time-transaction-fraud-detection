import pandas as pd
import numpy as np
import os

print('Python executable:', os.sys.executable)
print('Pandas version:', pd.__version__)
print('Numpy version:', np.__version__)

df = pd.DataFrame({
    'transaction_id': range(1, 11),
    'amount': [50, 120, 7, 400, 23, 1500, 30, 4000, 75, 99],
    'is_fraud': [0,0,0,1,0,1,0,1,0,0]
})
csv_path = 'data/sample_transactions.csv'
os.makedirs('data', exist_ok=True)
df.to_csv(csv_path, index=False)
print(f'Created sample CSV: {csv_path}')

df2 = pd.read_csv(csv_path)
print('Data preview:')
print(df2.head())
print('\nSummary statistics for amount:')
print(df2['amount'].describe())
print('\nFraud counts:')
print(df2['is_fraud'].value_counts())
