# generate regression dataset
import numpy
from sklearn.datasets import make_regression
# generate regression dataset
X,y = make_regression(n_samples=250, n_features=2, noise=0.1)
# 分成-1 和1 两类
for i in range(len(y)):
    if y[i]>0:
        y[i]=1
    else:
        y[i]=-1
numpy.save('label_1_train',y[0:200])
numpy.save('features_1_train',X[0:200,:])
numpy.save('label_1_test',y[200:250])
numpy.save('features_1_test',X[200:250,:])