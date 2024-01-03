import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, TimeDistributed
from keras.models import load_model
from math import log10

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
#valid_data, test_data = train_test_split(test_data,test_size=0.5,random_state=404)
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



#### Validation Normal ####
def process_data(dataset):
    x = []
    y = []
    for i in range(len(dataset)):
        length = len(dataset[i])
        xtemp = np.zeros((1,length,11))
        ytemp = np.zeros((1,length,11))
        for m in range(length-1):
            xtemp[:,m+1,dataset[i][m]] = 1
            ytemp[:,m,dataset[i][m]] = 1
        ytemp[:,length-1,10] = 1

        x.append(xtemp)
        y.append(ytemp)
    
    return x,y

def process_normal_validation(model,valid_data):
    x_valid,y_valid = process_data(valid_data)
    scores = []
    real_scores = []
    for x in range(len(x_valid)):
        p = model.predict(x_valid[x])[0]
        product = 0

        for i in range(len(valid_data[x])):
            if i == 0:
                index = valid_data[x][i]
                product = p[i,index]
            else:
                index = valid_data[x][i]
                product = product*p[i,index]
        if product == 0:
            real_scores.append(0)
            
        else:
            scores.append(log10(product))
            real_scores.append(log10(product))
    return scores,real_scores



#### Validation Abnormal ####
def process_abnormal_data(dataset):
    np.random.seed(404)
    x = []
    y = []
    data = []
    for i in range(len(dataset)):
        length = len(dataset[i])
        endseq = dataset[i][-5:-1]
        np.random.shuffle(endseq)
        dataset[i][-5:-1] = endseq
        data.append(dataset[i])
        xtemp = np.zeros((1,length,11))
        ytemp = np.zeros((1,length,11))
        for m in range(length-1):
            xtemp[:,m+1,dataset[i][m]] = 1
            ytemp[:,m,dataset[i][m]] = 1
        ytemp[:,length-1,10] = 1

        x.append(xtemp)
        y.append(ytemp)
    
    return x,y,data

def process_abnormal_validation(model,valid_data):
    x_valid,y_valid, data = process_abnormal_data(valid_data)
    abnormal_scores = []
    real_abnormal_scores = []
    for x in range(len(x_valid)):
        p = model.predict(x_valid[x])[0]
        product = 0

        for i in range(len(data[x])):
            if i == 0:
                index = data[x][i]
                product = p[i,index]
            else:
                index = data[x][i]
                product = product*p[i,index]
        if product == 0:
            real_abnormal_scores.append(-999)
        else:
            abnormal_scores.append(log10(product))
            real_abnormal_scores.append(log10(product))
    return abnormal_scores, real_abnormal_scores

## Metric function ##
def calculate_metrics(normal_scores,abnormal_scores,threshold):
    # postive refers to abnormal and negative refers normal
    TP = 0
    TN = 0
    
    for score in normal_scores:
        if score <= threshold:
            TN += 1
        # others are normal (neg) misclassified as abnormal (pos) by the model
    
    for score in abnormal_scores:
        if score > threshold:
            TP += 1
        # others are abnormal (pos) misclassified as normal (neg) by the model
    
    perclass_size = len(normal_scores)
    FP = perclass_size - TN
    FN = perclass_size - TP
    # if none of the data instances were detected as positives, there's
    # no precision
    if (TP + FP) == 0:
        precision = np.nan
    else:
        precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    # If precision not available, f1 scores not available
    if precision == np.nan:
        f1_score = np.nan
    elif precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2*precision*recall/(precision + recall)
    accuracy = (TP + TN)/(perclass_size*2)
    
    return precision, recall, f1_score, accuracy   

## Metric end ##
def model_selection(scores,abnormal_scores,real_scores, real_abnormal_scores):
    scores = scores + abnormal_scores
    threshold_min = np.min(scores)
    threshold_max = np.max(scores)
    threshold_choices = np.linspace(threshold_min,threshold_max,100)

    # Computing performance metrics for each threshold choice
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    for choice in threshold_choices:
        p,r,f1,a = calculate_metrics(real_scores,real_abnormal_scores,choice)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        accuracies.append(a)

    # Find the best threshold choice
    idx = np.nanargmax(f1_scores)
    #print('Highest F1 score :',f1_scores[idx])
    best_choice = threshold_choices[idx]
    
    return precisions[idx],recalls[idx],f1_scores[idx],accuracies[idx],best_choice
p = []
r = []
f1 = []

for i in range(len(X_train)):
    
    model = load_model('model'+str(i+1)+'.h5')
    xtrain = X_train[i]
    xvalid = X_valid[i]
    scores,real_scores = process_normal_validation(model,xvalid)
    abnormal_scores, real_abnormal_scores = process_abnormal_validation(model,xvalid)
    pv,rv,f1v,av,threshold = model_selection(scores,abnormal_scores,real_scores,real_abnormal_scores)

    print('Validation :',pv,rv,f1v,av)
    scores,real_scores = process_normal_validation(model,test_data)
    abnormal_scores, real_abnormal_scores = process_abnormal_validation(model,test_data)
    p,r,f1,a = calculate_metrics(real_scores,real_abnormal_scores,threshold)
    print('Test :', p,r,f1,a )
    



