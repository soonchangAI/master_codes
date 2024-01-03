'''
Noise removal:
3) Length < 10 or unique activities per instance < 6

'''
import pandas as pd 
import numpy as np 

data = pd.read_csv('data.csv',index_col = 0)

# Remove noise
dates = data['Date'].unique()
print(len(dates))
dates_torm = []
for date in dates:
    tempDf = data[data['Date'] == date].copy()
    tempDf.index = np.arange(len(tempDf))
    length = len(tempDf)

    activities = tempDf['Activity'].unique()
    if length < 10 or length>30:
       dates_torm.append(date)
    #if length >= 10 and len(activities) >= 6:
    #    dates_torm.append(date)
print(len(dates_torm))
new_data = data[~data['Date'].isin(dates_torm)].copy()
new_data.index = np.arange(len(new_data))
print(new_data['Date'].nunique())
new_data.to_csv('data_lstm.csv')

