import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, TimeDistributed
from sklearn.model_selection import KFold

dir_data = '/data/'
df = pd.read_csv('data_lstm.csv')

# Dictionary for activity label to integer conversion
activities = list(df['Activity'].unique())
act_to_ix = {activity:index for index, activity in enumerate(sorted(activities))}
act_to_ix['EOS'] =  10
print(act_to_ix)

days = df['Date'].nunique()
print('Number of days in the dataset :', days)
data = []
for date in df['Date'].unique():
    tempDf = df[df['Date']==date].copy()
    tempDf = list(tempDf['Activity'])
    tempDf.append('EOS')
    data_per_day = []
    for i in range(len(tempDf)):
        data_per_day.append(act_to_ix[tempDf[i]])
    data.append(data_per_day)


train_data, test_data = train_test_split(data,test_size=0.2,random_state=404)

kf = KFold(n_splits=4, random_state=404)
kf.get_n_splits(train_data)
X_train = []
X_valid = []
for train_index, test_index in kf.split(train_data):
    tempTrain = []
    tempValid = []
    for index in train_index:
        tempTrain.append(train_data[index])
    for index in test_index:
        tempValid.append(train_data[i])
    X_train.append(tempTrain)
    X_valid.append(tempValid)
    print(len(tempTrain), len(tempValid))

def prepare_data(train_data):
    x_train = []
    y_train = []
    for i in range(len(train_data)):
        length = len(train_data[i])
        x = np.zeros((1,length,11))
        y = np.zeros((1,length,11))
        for m in range(length-1):
            x[:,m+1,train_data[i][m]] = 1
            y[:,m,train_data[i][m]] = 1
        y[:,length-1,10] = 1

        x_train.append(x)
        y_train.append(y)
    return x_train,y_train

def training_generator(x_train,y_train):
    while True:
        for i in range(len(x_train)):
            x = x_train[i]
            y = y_train[i]
            yield x,y
counter = 0
for i in range(len(X_train)):
    xtrain = X_train[i]
    print(xtrain)
    x_train,y_train = prepare_data(xtrain)
    model = Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(None,11)))
    model.add(TimeDistributed(Dense(11,activation='softmax')))

    optimizer = Adam(decay=1e-8)
    model.compile(loss='categorical_crossentropy',
                    optimizer = optimizer,
                    metrics = ['accuracy'])
    model.fit_generator(training_generator(x_train,y_train),
                        steps_per_epoch = 100,
                        epochs=500, verbose=1)
    counter += 1
    model.save('model'+str(counter)+'.h5')
