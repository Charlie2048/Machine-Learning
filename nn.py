import os

import numpy as np

import function
import matplotlib.pyplot as plt

#get the current path of file
file_path=os.path.abspath(__file__)
father_path=os.path.dirname(file_path)
file_path=father_path
print(file_path)
test_file_path=file_path+'\\test.txt'
train_file_path=file_path+'\\train.txt'
test_target_file_path=file_path+'\\test_target.txt'
train_target_file_path=file_path+'\\train_target.txt'

x_train=np.loadtxt(train_file_path)
y_train=np.loadtxt(train_target_file_path)

x_test=np.loadtxt(test_file_path)
y_test=np.loadtxt(test_target_file_path)

net = function.nn([4, 20, 10, 3])
net.train(x_train, y_train)
net.accuracy(x_test,y_test)
plt.show