import numpy as np

import function
import os

#from sklearn.datasets import load_iris


#iris_set=load_iris(0)#load data
#data=iris_set.data  #get the data of feature  
#target=iris_set.target  #0:Iris-setosa, 1:Iris-versicolor,2:Iris-virginica'
#feature_names=iris_set.feature_names #get the feature_name
#data=np.column_stack((data,target))#combine the data and target
file_path=os.path.abspath(__file__)#get the current path of file
father_path=os.path.dirname(file_path)
file_path=father_path
print(file_path)
test_file_path=file_path+'\\test.txt'
train_file_path=file_path+'\\train.txt'
test_dataset=np.loadtxt(test_file_path)
train_dataset=np.loadtxt(train_file_path)
featureset = [0, 1, 2, 3]     # correlate to ['sepal length','sepal width','petal length','petal width']
tree = function.TreeGenerate(train_dataset, featureset)# compute accuracy
acc = function.ComputeAccuracy(test_dataset, tree)
print('Accuracy:', acc)

