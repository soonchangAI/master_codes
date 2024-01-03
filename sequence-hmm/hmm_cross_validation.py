from __future__ import division
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from hmmlearn import hmm
from math import log10

# load dataset
df = pd.read_csv('data_hmm.csv',index_col=0)
print('Data size :',df['Date'].nunique())

# Dictionary for activity label to integer conversion
activities = list(df['Activity'].unique())
act_to_ix = {activity:index for index, activity in enumerate(sorted(activities))}

# Encodes activity label into integer for a dataframe
def activity_encoder(dataframe):
    data = []
    for i in range(len(dataframe)):
        data.append(act_to_ix[dataframe.iloc[i]['Activity']])
    dataframe['Activity'] = data
    return dataframe

def process_data(dataframe):
    dataframe = dataframe.copy()
    dataframe = activity_encoder(dataframe)
    
    data = []
    length = []
    for date in dataframe['Date'].unique():
        tempDf = dataframe[dataframe['Date']==date].copy()
        tempDf.index = np.arange(len(tempDf))
        length.append(len(tempDf))
        for i in range(len(tempDf)):
            data.append([tempDf['Activity'][i]])
    
    return data,length

def scoring(x,length,model):
    curr = 0
    scores = []
    for i in range(len(length)):
        end = length[i] + curr
        scores.append(model.score(x[curr:end]))
        curr += length[i]
    return scores

def abnormal_scoring(x,length,model):
    np.random.seed(404)
    curr = 0
    scores = []
    for i in range(len(length)):
        end = length[i] + curr
        seq = x[curr:end]
        end = seq[-5:-1]
        np.random.shuffle(end)
        seq[-5:] = end
        scores.append(model.score(seq))
        curr += length[i]
    return scores

'''
Function to calculate performance metrics such as precision, recall, F1 score
and accuracy using the scores of the data instances and threshold
'''    
def calculate_metrics(normal_scores,abnormal_scores,threshold):
    # postive refers to abnormal and negative refers normal
    TP = 0
    TN = 0
    
    for score in normal_scores:
        if score >= threshold:
            TN += 1
        # others are normal (neg) misclassified as abnormal (pos) by the model
    
    for score in abnormal_scores:
        if score < threshold:
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
    if precision == 0 and recall == 0:
        f1_score = np.nan
    else:
        f1_score = 2*precision*recall/(precision + recall)
    accuracy = (TP + TN)/(perclass_size*2)
    
    return precision, recall, f1_score, accuracy

# Data partitioning
dates = df['Date'].unique()
train_dates,test_dates = train_test_split(dates,test_size=0.2,random_state=404)
kf = KFold(n_splits=4,random_state=404)
kf.get_n_splits(train_dates)

trainDates = []
validDates = []
for train_index, valid_index in kf.split(train_dates):
    tempTrainDates = []
    tempValidDates = []
    for index in train_index:
        tempTrainDates.append(dates[index])

    for index in valid_index:
        tempValidDates.append(dates[index])

    trainDates.append(tempTrainDates)
    validDates.append(tempValidDates)
    print(len(tempTrainDates))
    print(len(tempValidDates))

def prepare_data(dataframe,dates):
    
    data = dataframe[dataframe['Date'].isin(dates)].copy()
    data.index = np.arange(len(data))
    x, length = process_data(data)

    return x, length


# Training for different training set and k fold
num_states = [10,20,30,40,50,60,70,80,90,100]

models = {}


for i in range(4):
    x_train,length_train = prepare_data(df,trainDates[i])
    for n in num_states:
        k = i+1
        key = str(n)+str(k)
        model = hmm.MultinomialHMM(n_components = n, random_state = 1)
        model.fit(x_train,length_train)
        models[key] = model
        
x_test, length_test = prepare_data(df,test_dates)
def evaluate_model(key,i):       
    model = models[key]

    detail = []
    x_val, length_val = prepare_data(df,validDates[i])

    normal_scores = scoring(x_val,length_val,model)
    abnormal_scores = abnormal_scoring(x_val,length_val,model)

    scores = normal_scores + abnormal_scores
    #print(scores)
    threshold_min = np.min(scores)
    threshold_max = np.max(scores)
    threshold_choices = np.linspace(threshold_min,threshold_max,100)
    
    # Computing performance 
    # Computing performance metrics for each threshold choice
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    for choice in threshold_choices:
        p,r,f1,a = calculate_metrics(normal_scores,abnormal_scores,choice)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        accuracies.append(a)
    
    # Find the best threshold choice
    idx = np.nanargmax(f1_scores)
    #print('Highest F1 score :',f1_scores[idx])
    best_choice = threshold_choices[idx]
    '''
    detail.append(best_choice)
    detail.append([precisions[idx],recalls[idx],f1_scores[idx],accuracies[idx]])
    '''
    normal_test_scores = scoring(x_test,length_test,model)
    abnormal_test_scores = abnormal_scoring(x_test,length_test,model)
    p_test,r_test, f1_test, a_test = calculate_metrics(normal_test_scores,abnormal_test_scores,best_choice)
    return p_test,r_test, f1_test, a_test
       
        
for i in range(4):
    ps = []
    rs = []
    f1s = []
    acs = []

    for n in num_states:
        k = i+1
        key = str(n)+str(k)
        p,r,f1,a = evaluate_model(key,i)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        acs.append(a)
    idx = np.argmax(acs)
    print('Test :',ps[idx],rs[idx],f1s[idx],acs[idx])
                                
                                     

        


























