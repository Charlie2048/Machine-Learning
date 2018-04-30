# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:13:58 2018

@author: Waddles
"""
from numpy import mat
from numpy import ones
import numpy
from matplotlib import pyplot
y= numpy.load('label_1_train.npy')
x= numpy.load('features_1_train.npy')
y_test= numpy.load('label_1_test.npy')
x_test= numpy.load('features_1_test.npy')
#对x的数据进行转化
x_train=mat(x)
y_train=mat(y)
y_train=y_train.T
identity=mat(ones((len(y),1)))
x_train=numpy.hstack((x_train,identity))
x_tt=x_train.T#保存数据
x_train=numpy.multiply(x_train,y_train)
x_train=x_train.T
w=mat(numpy.random.rand(3,1));#初始化权重
j=10
inter=j
l=0.01
while(j>0):
    for i in range(len(y)):
        if w.T*x_train[:,i]<0:
            w=w+l*x_train[:,i]
    j-=1
x_train=numpy.array(x_train)
p=pyplot.scatter(x[:,0], x[:,1], marker='o', c=y)
xs=numpy.arange(-2,2,0.2)
ys=-(w[0,0]*xs+w[2,0])/w[1,0]
m=pyplot.plot(xs,ys,'g--')
#pyplot.legend('sda')
pyplot.xlabel('x_2')
pyplot.ylabel('x_1')
pyplot.title('the experimental result base on data_1')
pyplot.text(-1,0,'Separation Curve')
pyplot.text(-1,-1,'Negtive sample')
pyplot.text(1,1,'Positive sample')


##计算训练集上的准确率
y_tt=w.T*x_tt
y_tt=y_tt[0]
acc_count=0
for i in range(len(y)):
    if ((y_tt[0,i]<0)and(y[i]<0))or((y_tt[0,i]>=0)and(y[i]>=0)):
        acc_count=acc_count+1
accuracy=acc_count/len(y)
print ('the train accuracy is ',accuracy*100,'%')
pyplot.text(2,-2.5,accuracy)
##计算测试集上的准确率
x_test=mat(x_test)
identity=mat(ones((len(y_test),1)))
x_test=numpy.hstack((x_test,identity))
x_test=x_test.T

y_tp=w.T*x_test
y_tp=y_tp[0]
acc_count=0
for i in range(len(y_test)):
    if ((y_tp[0,i]<0)and(y_test[i]<0))or((y_tp[0,i]>=0)and(y_test[i]>=0)):
        acc_count=acc_count+1
accuracy=acc_count/len(y_test)
print ('the test accuracy is ',accuracy*100,'%')
pyplot.text(2,-2.75,accuracy)
pyplot.text(-3,-2.75,'iteration=200')
pyplot.text(-3,-2.5,'L=0.01')
pyplot.show()