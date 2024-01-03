import pandas as pd 
import numpy as np 
from tensorflow.keras.models import load_model
import os
from sklearn.model_selection import train_test_split

win_size = 5
# load dataset
df = pd.read_csv('data_lstm.csv',index_col=0)
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

# Segmentation for a dataframe with window size, win_size
# return data with size (m,win_size)
def process_data(dataframe,win_size):
    dataframe = dataframe.copy()
    dataframe = activity_encoder(dataframe)
    data = []
    target = []
    length = []

    for date in dataframe['Date'].unique():
        tempDf = dataframe[dataframe['Date']==date].copy()
        tempDf.index = np.arange(len(tempDf))
        counter = 0
        for i in range(len(tempDf)-win_size+1):
                
            if i == len(tempDf)-win_size:
                data.append(list(tempDf['Activity'][i:i+win_size]))
                target.append(10)
                counter += 1
            else:
                data.append(list(tempDf['Activity'][i:i+win_size]))
                target.append(tempDf['Activity'][i+win_size])
                counter += 1
                
        length.append(counter)
            
    m = len(target)
    data = np.array(data)
    target_onehot = np.zeros((m,11))
    for i in range(m):
        target_onehot[i,target[i]] = 1

    return data, target_onehot, length

# To return scores given a set of inputs and targets, respective lengths of sequence
# and a trained LSTM model
def scoring(x,y,lengths,model):
    scores = []
    curr = 0
    for i in range(len(lengths)):
        end = curr + lengths[i]
        inps = x[curr:end]
        targets = y[curr:end]
        curr += lengths[i]
        preds = model.predict(inps)
        diff_vecs = targets - preds
        score = np.sum(np.absolute(diff_vecs))/lengths[i]
        scores.append(score)
    return scores

# Randomize normal data instances to create abnormal data
# To return scores given a set of inputs and targets, 
# respective lengths of sequence
# and a trained LSTM model
def abnormal_scoring(x,y,lengths,model):
    np.random.seed(0)
    every_diff_vecs = []
    scores = []
    curr = 0
    for i in range(len(lengths)):
        end = curr + lengths[i]
        inps = x[curr:end]
        targets = y[curr:end]
        endseq = inps[-5:-1]
        np.random.shuffle(endseq)
        inps[-5:-1] = endseq
        curr += lengths[i]
        preds = model.predict(inps)
        diff_vecs = targets - preds
        every_diff_vecs.append(np.absolute(diff_vecs))
        score = np.sum(np.absolute(diff_vecs))/lengths[i]
        scores.append(score)
    return scores, every_diff_vecs
  
'''
Function to calculate performance metrics such as precision, recall, F1 score
and accuracy using the scores of the data instances and threshold
'''    
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
    else:
        f1_score = 2*precision*recall/(precision + recall)
    accuracy = (TP + TN)/(perclass_size*2)
    
    return precision, recall, f1_score, accuracy   
   
# Data partitioning
dates = df['Date'].unique()
train_dates,test_dates = train_test_split(dates,test_size=0.4,random_state=404)
test_dates, val_dates = train_test_split(test_dates,test_size=0.5,random_state=404)

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


def save_dict_to_file(dic):
    f = open('dict.txt','w')
    f.write(str(dic))
    f.close()

def load_dict_from_file():
    f = open('dict.txt','r')
    data=f.read()
    f.close()
    return eval(data)

model_name = 'lstm1_win5.h5'
model = load_model(model_name)
dictionary = load_dict_from_file()
threshold = dictionary['lstm1_win5.h5'][0]

x_test,y_test,len_test = process_data(test_set,win_size)
normal_test_scores = scoring(x_test,y_test,len_test,model)
abnormal_test_scores, every_diff_vecs = abnormal_scoring(x_test,y_test,len_test,model)
x_test,y_test,len_test = process_data(test_set,win_size)

print(every_diff_vecs)

miss_counter = 0
for i in range(len(every_diff_vecs)):
    length = len(every_diff_vecs[i])
    avgs = []
    for j in range(length):
        mean = np.mean(every_diff_vecs[i][j])
        avgs.append(mean)

    avgs = np.array(avgs)
    index = np.argmax(avgs)
    print(str(length) + ','+str(index+1))
    if length - 5  <= index + 1  <=length :
        pass
    else:
        miss_counter += 1

accuracy = (38 - miss_counter)/38*100
print(accuracy)
          








      
