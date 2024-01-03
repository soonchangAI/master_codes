import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

win_size = 9
# load dataset
df = pd.read_csv('data.csv',index_col=0)

# delete data with 10 < length < 30
date_rm = []
for date in df['Date'].unique():
    if len(df[df['Date'] == date]) < 10 or len(df[df['Date'] == date]) > 30:
        date_rm.append(date)
df = df[~df['Date'].isin(date_rm)].copy()
df.index = np.arange(len(df))
print('Number of dates :',df['Date'].nunique())

# save cleaned data into a .csv file
df.to_csv('cleaned_dataset.csv')

# To encode activity label into integer for a dataframe
def activity_encoder(dataframe):
    act2digit = {
        'Enter_Home': 10, 'Work': 6, 
        'Housekeeping': 7, 'Sleeping': 2, 
        'Eating': 4,'Leave_Home': 9, 
        'Wash_Dishes': 5, 'Meal_Preparation': 3,
        'Bed_to_Toilet': 1, 'Relax': 8
    }
    activities = []
    for i in range(len(dataframe)):
        activities.append(act2digit[dataframe.iloc[i]['Activity']])
    dataframe['Activity'] = activities
    return dataframe
# Encode activity label into unique integers
df = activity_encoder(df)

# Slice dataframe for a certain day into segments 
def segmentation(win_size,dataframe):
    dataframe.index = np.arange(len(dataframe))
    segments = []
    for i in range(0,len(dataframe)-win_size+1):
        segments.append(list(dataframe['Activity'].iloc[i:i+win_size]))
    return segments

# Shuffling to inject abnormalities into dataset and slice into segments
def abnormal_segmentation(win_size,dataframe):
    dataframe.index = np.arange(len(dataframe))
    activities = list(dataframe['Activity'])
    subsequence = activities[-6:-1]  
    np.random.seed(0)
    np.random.shuffle(subsequence)
    activities[-6:-1] = subsequence
    segments = []
    for i in range(0,len(activities)-win_size+1):
        segments.append(activities[i:i+win_size])
    return segments

# Build a database by using the training set
def database_generator(win_size,dataframe):
    seqs = []
    for date in dataframe['Date'].unique():
        partial_df = dataframe[dataframe['Date']==date].copy()
        seqs.append(segmentation(win_size,partial_df))
    seqs_unroll = []
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            seqs_unroll.append(seqs[i][j])
    database = [list(x) for x in set(tuple(x) for x in seqs_unroll)]
    return database 

# Calculate the score for a data instance
def scoring(segments,database):
    counter = 0
    for segment in segments:
        if (segment in database) == False:
            counter+=1
    score = counter/len(segments)*100
    return score


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

# Computing scores of data instances of validation set
def compute_scores(validation_dates,database):
    normal_val_scores = []
    abnormal_val_scores = []

    validation_set = df[df['Date'].isin(validation_dates)].copy()
    for date in validation_dates:
        tempDf = validation_set[validation_set['Date'] == date].copy()
        segments = segmentation(3,tempDf)
        abnormal_segments = abnormal_segmentation(win_size,tempDf)
        normal_val_scores.append(scoring(segments,database))
        abnormal_val_scores.append(scoring(abnormal_segments,database))
    return normal_val_scores, abnormal_val_scores    
#print(normal_val_scores)
#print(abnormal_val_scores)
    
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
    elif precision + recall == 0:
        f1_score = np.nan
    else:
        f1_score = 2*precision*recall/(precision + recall)
    accuracy = (TP + TN)/(perclass_size*2)
    
    return precision, recall, f1_score, accuracy
def model_selection(normal_val_scores,abnormal_val_scores):
    # Threshold sampling
    scores = list(normal_val_scores) + list(abnormal_val_scores)
    threshold_min = np.min(scores)
    threshold_max = np.max(scores)
    threshold_choices = np.linspace(threshold_min,threshold_max,100)

    # Computing performance metrics for each threshold choice
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []

    for choice in threshold_choices:
        p,r,f1,a = calculate_metrics(normal_val_scores,abnormal_val_scores,choice)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        accuracies.append(a)
    #print('Precision :',precisions)
    #print('Recalls :',recalls)
    #print('F1 scores :',f1_scores)

    # Find the best threshold choice
    idx = np.nanargmax(f1_scores)
    best_choice = threshold_choices[idx]
    '''
    print('Highest F1 score :',f1_scores[idx])
    
    print('Best threshold choice :',best_choice)

    print('Performance on validation set :')
    print('Precision :',precisions[idx])
    print('Recall :',recalls[idx])
    print('f1 score :',f1_scores[idx])
    print('Accuracy :',accuracies[idx])
'''
    # Model evaluation, evaluate the selected threshold choice on test set
    normal_test_scores = []
    abnormal_test_scores = []

    test_set = df[df['Date'].isin(test_dates)].copy()
    for date in test_dates:
        tempDf = test_set[test_set['Date'] == date].copy()
        segments = segmentation(win_size,tempDf)
        abnormal_segments = abnormal_segmentation(win_size,tempDf)
        normal_test_scores.append(scoring(segments,database))
        abnormal_test_scores.append(scoring(abnormal_segments,database))
    
    p_test,r_test,f1_test,a_test = calculate_metrics(normal_test_scores,abnormal_test_scores,best_choice)
    '''
    print('Model evaluation on test set :')
    print('Precision :',p_test)
    print('Recall :',r_test)
    print('f1 score :',f1_test)
    print('Accuracy :',a_test)
    '''
    return p_test,r_test,f1_test,a_test


win_sizes = [3,5,7,9]
precisions = []
recalls = []
f1scores = []
accuracies = []

for i in range(len(win_sizes)):    
    ps = []
    rs = []
    f1s = []
    acs = []
    win_size = win_sizes[i]
    for i in range(len(trainDates)):
        train_dates = trainDates[i]
        # Create database using training set
        train_set = df[df['Date'].isin(train_dates)].copy()
        train_set.index = np.arange(len(train_set))
        database = database_generator(win_size,train_set)
        # Calculate scores for respective validation set
        validation_dates = validDates[i]
        normal_val_scores, abnormal_val_scores = compute_scores(validation_dates,database)
        # Perform model selection 
        p,r,f1,a = model_selection(normal_val_scores,abnormal_val_scores)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        acs.append(a)
    precisions.append(ps)
    recalls.append(rs)
    f1scores.append(f1s)
    accuracies.append(acs)


fold = [1,2,3,4]
import matplotlib.pyplot as plt 
plt.plot(fold,accuracies[0],c='blue',label='3',marker='x')
plt.plot(fold,accuracies[1],c='red',label='5',marker='x')
plt.plot(fold,accuracies[2],c='green',label='7',marker='x')
plt.plot(fold,accuracies[3],c='black',label='9',marker='x')
plt.xticks(fold)
plt.xlabel('k')
plt.ylabel('Test accuracy')
plt.legend()
plt.show()
