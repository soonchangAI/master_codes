'''
convert from 'cleanedData.csv' into standard format
and saved in 'data.csv'
'''
import pandas as pd 
import numpy as np 
df = pd.read_csv('cleanedData.csv',index_col=0)
df.index = np.arange(len(df))

dates = list(df['Date'])
start_hour = []
start_min = []
end_hour = []
end_min = []
activity = list(df['Activity'])
for i in range(len(df)):
    start_hour.append(int(df.iloc[i]['start time'].split(':')[0]))
    start_min.append(int(df.iloc[i]['start time'].split(':')[1]))
    end_hour.append(int(df.iloc[i]['end time'].split(':')[0]))
    end_min.append(int(df.iloc[i]['end time'].split(':')[1])) 

new_df = pd.DataFrame(
    {
        'Date':dates,
        'start hour':start_hour,
        'start min':start_min,
        'end hour':end_hour,
        'end min':end_min,
        'Activity':activity
    }
)
new_df.to_csv('data.csv')