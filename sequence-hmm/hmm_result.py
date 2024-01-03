def load_dict_from_file():
    f = open('dict.txt','r')
    data=f.read()
    f.close()
    return eval(data)
import numpy as np
performances = load_dict_from_file()
print(performances)
num_states = [10,20,30,40,50,60,70,80,90,100]
#num_states = [2,4,6,8,10]

f1 = []
a = []
for n in num_states:
    data = performances[n]
    f1.append(data[1][2])
    a.append(data[2][3])

average_accuracy = np.mean(a)
print('Accuracy (avg) :',average_accuracy)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.plot(num_states,f1,label='F1 score',marker='x')
plt.plot(num_states,a,label='Test accuracy',marker='x')
plt.xlabel('Number of states')
plt.legend()
plt.show()
