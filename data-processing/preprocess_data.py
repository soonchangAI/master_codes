'''
Load dataset from original public data in 'data.txt'
Remove noises:
1)Activity sandwich
2)Repeating activities
3)Rare activity - 'Respirate'
Put everything into format
Then saved as 'cleanedData.csv'
'''
import pandas as pd 
import numpy as np 

# load dataset, drop unwanted columns and remove 'Respirate' activity
names=['Date','start time','Sensor','sensor state',
        'Activity','State']
df = pd.read_csv('data.txt',delim_whitespace=True,header=None,names=names)
df.drop(['Sensor','sensor state'],axis=1,inplace=True)
df = df[pd.notnull(df['Activity'])]
df = df[df['Activity']!='Respirate']
df.index = np.arange(0,len(df))

# Some final element of a sequence (for a date) ends up at later day
# This function is to fix this problem
def restructure(activity):
    data = []
    for i in range(len(df)):
        if df['Activity'][i] == activity:
            # How data is saved: Index,date,state
            data.append([i,df['Date'][i],df['State'][i]])
            
    # Check if there's any error in state
    for i in range(0,len(data),2):
        if data[i][2] != 'begin' or data[i+1][2]!='end':
            print('error')
    '''
    If there is any dates not the same begin consecutive
    indices, fix that dates for odd indices
    '''
    for i in range(0,len(data),2):
        if data[i][1] != data[i+1][1] and int(data[i+1][0]) == int(data[i][0])+1:
            data[i+1][1] = data[i][1]
    # Modify the dates
    for i in range(len(data)):
        ind = data[i][0]
        date = data[i][1]
        df['Date'][ind] = date
# Do it on every type of activities      
for activity in df['Activity'].unique():
    restructure(activity)

# Check if there's any day that have odd length sequences
count = 0
dates = []
for date in df['Date'].unique():
    if len(df[df['Date']==date])%2 != 0:
        count+=1
        dates.append(date)
print(count)
print(dates)

# Remove dates with overlapping activities
dates_torm = []
for i in range(0,len(df),2):
    if df['Activity'][i] != df['Activity'][i+1]:
        assert(df['Date'][i] == df['Date'][i+1])
        dates_torm.append(df['Date'][i])
        dates_torm.append(df['Date'][i+1])
dates_torm = list(set(dates_torm))
print('Number of dates (original) :',df['Date'].nunique())
print('Number of dates to be removed :',len(dates_torm))

df = df[~df['Date'].isin(dates_torm)]
df.index = np.arange(len(df))
print(df['Date'].nunique())

# Add end time to dataframe and drop Odd rows
end_time = []
odd_index = []

for i in range(0,len(df),2):
    odd_index.append(i+1)
    end_time.append(df['start time'].iloc[i+1])
    
df = df.drop(index=odd_index)
df = df.drop(columns='State')
df.index = np.arange(len(df))
df['end time'] = end_time

# Get indices of rows with consecutively repeating activities in a day
tf = df.copy()

def find_repeated(tf):
    idxs = []
    init = 0
    temp = 0
    for i in range(len(tf)):
        flag = False
        index = [i]
        if i == init:
            for j in range(i+1,len(tf)):
                if tf['Date'][i] == tf['Date'][j] and tf['Activity'][i] == tf['Activity'][j] and flag == False:
                    index.append(j)
                elif (tf['Activity'][i] != tf['Activity'][j] or tf['Date'][i] != tf['Date'][j]) and flag == False:
                    flag = True
                    temp = j
            init = temp
            if len(index) > 1:
                idxs.append(index)
    return idxs

idxs = find_repeated(tf)
# Drop consecutive repeating activities in a day
for indices in idxs:
    tf['end time'][indices[0]] = tf['end time'][indices[-1]]
    tf = tf.drop(index = indices[1:])

# Number of to be deleted rows
count = 0
for i in range(len(idxs)):
    for j in range(1,len(idxs[i])):
        count += 1
print(count)
tf.to_csv('cleanedData.csv')