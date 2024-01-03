import pandas as pd

df = pd.read_csv('data_lstm.csv')
print(df['Date'].nunique())
