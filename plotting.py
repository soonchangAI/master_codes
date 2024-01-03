import matplotlib.pyplot as plt

hmm = [46.05,50,44.74,59.21]
lstm_whole = [64.47, 59.21,57.89,57.89]
lstm = [73.68,76.32,76.32,81.58]
database = [78.95,82.89,92.11,84.21]

fold = [1,2,3,4]

plt.plot(fold,hmm,label='HMM',marker='x')
plt.plot(fold,lstm_whole,label='LSTM Language Model',marker='x')
plt.plot(fold,lstm,label='LSTM Method',marker='x')
plt.plot(fold,database,label='Database Method',marker='x')
plt.legend()
plt.xticks(fold)
plt.ylabel('Test Accuracy')
plt.xlabel('k')
plt.show()
