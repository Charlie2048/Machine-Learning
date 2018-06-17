import numpy as np

import matplotlib.pyplot as plt

def sigmoid(z):
    '''
    compute the activation function
    '''
    sig=1/(1+np.exp(-z))
    return sig

class nn(object):
    def __init__(self,struct_num):
        '''
        caculate the neuralnetwork's layer num.
        pre-set the parameters
        '''
        self.layer_num=len(struct_num)-1
        self.weight_list=[]
        self.bias_list=[]
        self.bias_gradient_list=[]
        self.weight_gradient_list=[]
        self.cost_list=[]
        for i in range(self.layer_num):
            weight=np.random.normal(size=(struct_num[i],struct_num[i+1]))
            bias=np.zeros((1,struct_num[i+1]))
            self.weight_list.append(weight)
            self.bias_list.append(bias)
    
    def costgradient(self, x, y):
        """
        Compute the cost and gradient using input and label.
        :param x: input
        :param y: label
        :return: cost and activation list of gradient of cost function in each layer
        """
        activation_list = []
        activation = x
        for i in range(self.layer_num):
            activation_list.append(activation)
            z = np.dot(activation, self.weight_list[i]) + self.bias_list[i]
            activation = sigmoid(z)
        cost = np.sum((y - activation) ** 2) / 2 / len(y)


        delta_list = []
        delta = (activation - y) * activation * (1 - activation)
        for i in range(self.layer_num):
            delta_list.append(delta)
            delta = np.dot(delta, self.weight_list[-1 - i].T) * activation_list[-1 - i] * (1 - activation_list[-1 - i])
        delta_list.reverse()


        bias_gradient_list = []
        for i in range(self.layer_num):
            bias_gradient = np.sum(delta_list[i], axis=0, keepdims=True) / len(x)
            bias_gradient_list.append(bias_gradient)

        weight_gradient_list = []
        for i in range(self.layer_num):
            weight_gradient = np.dot(activation_list[i].T, delta_list[i]) / len(x)
            weight_gradient_list.append(weight_gradient)
        return cost, weight_gradient_list, bias_gradient_list   
        

    def train(self, x, y, step_size=0.1, interation=10000):
        '''
        train the network
        '''
        for p in range(interation):
            cost, weight_gradient_list, bias_gradient_list=self.costgradient(x,y)
            self.cost_list.append(cost)
            for i in range(self.layer_num):
                self.weight_list[i]-=step_size*weight_gradient_list[i]
                self.bias_list[i]-=step_size*bias_gradient_list[i]
        plt.plot(range(interation), self.cost_list)
        plt.xlabel('iteration number')
        plt.ylabel('cost')

    def predict(self, x):
        """
        Do rediction by net.
        :param x: input,can be one sample or multiple sample
        :return: activation,the prediction output of network on data x
        """
        if len(x.shape) == 1:#the shape may be uncomplete
            x.reshape(-1, len(x))
        for i in range(self.layer_num):
            z = np.dot(x, self.weight_list[i]) + self.bias_list[i]
            x = sigmoid(z)
        y_pre=x
        return y_pre

    def accuracy(self, x_test, y_test):
        """
        Compute the accuracy of net
        :param x_test: inputs of test dataset
        :param y_test: labels of test dataset
        :return: the accuracy in test dataset
        """
        pred = self.predict(x_test)
        temp = 0
        row, column = pred.shape
        for i in range(row):
            for j in range(column):
                if pred[i, j] >= 0.5:
                   pred[i, j] = 1
                else:
                    pred[i, j] = 0
            if (pred[i] == y_test[i]).all():
                temp += 1
        acc = temp / row
        print(acc)