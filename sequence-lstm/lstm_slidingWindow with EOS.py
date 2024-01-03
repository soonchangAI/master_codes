import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import CuDNNLSTM, Dense, Embedding
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta
## Constant variables
win_size = 9
embedding_dim = 10
##
df = pd.read_csv('data_lstm.csv',index_col=0)
print('Data size :',df['Date'].nunique())

# Dictionary for activity label to integer conversion
activities = list(df['Activity'].unique())
act_to_ix = {activity:index for index, activity in enumerate(sorted(activities))}
act_to_ix['EOS'] =  10
print(act_to_ix)

# Encodes activity label into integer for a dataframe
def activity_encoder(dataframe):
    data = []
    for i in range(len(dataframe)):
        data.append(act_to_ix[dataframe.iloc[i]['Activity']])
    dataframe['Activity'] = data
    return dataframe


# Segmentation for a dataframe with window size, win_size
# return data with size (m,win_size)
def process_data(dataframe,win_size):
    dataframe = activity_encoder(dataframe)
    data = []
    target = []

    for date in dataframe['Date'].unique():
        tempDf = dataframe[dataframe['Date']==date].copy()
        tempDf.index = np.arange(len(tempDf))
        for i in range(len(tempDf)-win_size+1):
            
            if i == len(tempDf)-win_size:
                data.append(list(tempDf['Activity'][i:i+win_size]))
                target.append(10)
            else:
                data.append(list(tempDf['Activity'][i:i+win_size]))
                target.append(tempDf['Activity'][i+win_size])

    m = len(target)
    data = np.array(data)
    target_onehot = np.zeros((m,len(act_to_ix)))
    for i in range(m):
        target_onehot[i,target[i]] = 1

    return data, target_onehot


# Data partitioning
dates = df['Date'].unique()
train_dates,test_dates = train_test_split(dates,test_size=0.2,random_state=404)


# Convert train dataframe into training input and target pairs
xtrain, ytrain = process_data(train_set,win_size)
print(xtrain.shape)
print(ytrain.shape)
# Building deep learning model 

'''
Embedding: turns positive integers (indexes) into fixed dense vectors
input_dim : size of vocabulary
output_dim : Dimension of dense embedding
input length : Length of input sequences
'''

def build_model(architecture,name,win_size,embedding_dim):
    tf.set_random_seed(100)
    np.random.seed(404)
    model = Sequential()
    model.add(Embedding(
        input_dim = 10,
        output_dim = embedding_dim, 
        input_length = win_size   
    ))

    if len(architecture) == 1:
        model.add(CuDNNLSTM(architecture[0]))

    elif len(architecture) > 1:
        for i in range(len(architecture)):
            if i >= 0 and i < len(architecture)-1:
                model.add(CuDNNLSTM(architecture[i],return_sequences=True))
            
            elif i == len(architecture)-1:
                model.add(CuDNNLSTM(architecture[i]))
    model.add(Dense(11,activation='softmax'))
    model.summary()
    tensorboard = TensorBoard(log_dir='logs_with_EOS/'+name)
    optimizer = Adadelta()
    model.compile(optimizer=optimizer,loss = 'categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(xtrain,ytrain,epochs=200,callbacks=[tensorboard])   
    model.save(name + '.h5')
    
architectures = [[32],[64],[32,32],[64,64]]
names = [ 'lstm'+str(i+1)+'_win'+str(win_size) for i in range(len(architectures))]
for i in range(len(architectures)):
    build_model(architectures[i],names[i],win_size,embedding_dim)



