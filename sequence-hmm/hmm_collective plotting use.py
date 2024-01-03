from __future__ import division
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from hmmlearn import hmm

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
train_dates,test_dates = train_test_split(dates,test_size=0.4,random_state=104)
test_dates, val_dates = train_test_split(test_dates,test_size=0.5,random_state=104)

print('Training size :',len(train_dates))
print('Validation size :',len(val_dates))
print('Test size :',len(test_dates))           

# Dataframe for normal train, validation and test set
train_set = df[df['Date'].isin(train_dates)].copy()
train_set.index = np.arange(len(train_set))

test_set = df[df['Date'].isin(test_dates)].copy()
test_set.index = np.arange(len(test_set))

validation_set = df[df['Date'].isin(val_dates)].copy()
validation_set.index = np.arange(len(validation_set))

x_train, length_train = process_data(train_set)
x_val, length_val = process_data(validation_set)
x_test, length_test = process_data(test_set)

models = {}
num_states = [10,20,30,40,50,60,70,80,90,100]

for n in num_states:
    model = hmm.MultinomialHMM(n_components=n,random_state=1)
    model.fit(x_train,length_train)
    models[n] = model

performances = {}
def evaluate_model(model,n):
    
    detail = []
    normal_scores = scoring(x_val,length_val,model)
    abnormal_scores = abnormal_scoring(x_val,length_val,model)
    '''
    import matplotlib.pyplot as plt
    x = np.arange(len(abnormal_scores))
    fig = plt.figure()
    plt.style.use('ggplot')
    plt.scatter(x,normal_scores,color='blue',label='normal')
    plt.scatter(x,abnormal_scores,color='red',label='abnormal')
    plt.title('N ='+str(n))
    plt.ylabel('Scores')
    plt.xlabel('Data points')
    plt.legend()
    plt.show()
    '''
    scores = normal_scores + abnormal_scores
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
    
    detail.append(best_choice)
    detail.append([precisions[idx],recalls[idx],f1_scores[idx],accuracies[idx]])
    
    normal_test_scores = scoring(x_test,length_test,model)
    abnormal_test_scores = abnormal_scoring(x_test,length_test,model)
    p_test,r_test, f1_test, a_test = calculate_metrics(normal_test_scores,abnormal_test_scores,best_choice)
    detail.append([p_test,r_test, f1_test, a_test])
    performances[n] = detail
    print(detail)
    
for n in num_states:
    evaluate_model(models[n],n)

def save_dict_to_file(dic):
    f = open('dict.txt','w')
    f.write(str(dic))
    f.close()

def load_dict_from_file():
    f = open('dict.txt','r')
    data=f.read()
    f.close()
    return eval(data)

save_dict_to_file(performances)
print(load_dict_from_file())




