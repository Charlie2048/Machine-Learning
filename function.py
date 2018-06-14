import copy

import numpy as np


class TreeNode(object):
    def __init__(self, classtype=None, leftnode=None, rightnode=None, feature=None, threshold=None):
        self.__classtype = classtype
        self.__leftnode = leftnode
        self.__rightnode = rightnode
        self.__feature = feature
        self.__threshold = threshold

    def set_classtype(self, classtype):
        self.__classtype = classtype

    def add_leftnode(self, leftnode):
        self.__leftnode = leftnode

    def add_rightnode(self, rightnode):
        self.__rightnode = rightnode

    def set_feature(self, feature):
        self.__feature = feature

    def set_threshold(self, threshold):
        self.__threshold = threshold

    def get_classtype(self):
        return self.__classtype

    def get_leftnode(self):
        return self.__leftnode

    def get_rightnode(self):
        return self.__rightnode

    def get_feature(self):
        return self.__feature

    def get_threshold(self):
        return self.__threshold

    def classification(self,data):
        """
        :param self: must be root node
        :param data: a 1D list like [5.1,3.5,1.4,0.2,'Iris-setosa']
        :return: the prediction class of the data
        """
        if self.__feature == None:
            classtype = self.__classtype
        else:
            if data[self.__feature] < self.__threshold:
                classtype = self.__leftnode.classification(data)
            else:
                classtype = self.__rightnode.classification(data)
        return classtype





def TreeGenerate(dataset, featureset):
    """
    Generate a tree by this function.
    :param dataset: a 2D list,each row of which is like [5.1,3.5,1.4,0.2,'Iris-setosa']
    :param featureset: a list of feature
    :return: a node
    """
    # generate a node
    node = TreeNode()

    # find most class
    iris_setosa = 0
    iris_versicolor = 0
    iris_virginica = 0
    for i in range(len(dataset)):
        if 0 == dataset[i][4]:#target 0:iris_setosa
            iris_setosa += 1
        if 1 == dataset[i][4]:#target 1:iris_versicolor
            iris_versicolor += 1
        if 2 == dataset[i][4]:#target 2:iris_virginica
            iris_virginica += 1
    most_num = iris_setosa
    most_class = 0
    if iris_versicolor > most_num:
        most_num = iris_versicolor
        most_class = 1
    if iris_virginica > most_num:
        most_num = iris_virginica
        most_class = 2

    # if all the samples belong to the same same class,set classtype and return
    # you can reduce the parameter alpha to weaken the effect of over-fitting
    alpha = 0.4
    if alpha*(len(dataset)) <= most_num:
        node.set_classtype(most_class)
        return node

    # if feature set is empty,set classtype and return
    if len(featureset) == 0:
        node.set_classtype(most_class)
        return node

    # find the best feature
    best_feature, threshold = BestFeature(dataset, featureset)
    node.set_feature(best_feature)
    node.set_threshold(threshold)

    # bulid branchs
    dataset1 = []
    dataset2 = []
    for i in range(len(dataset)):
        if dataset[i][best_feature] < threshold:
            dataset1.append(dataset[i])
        else:
            dataset2.append(dataset[i])
    # If features are discrete,replace replace featureset with sub_featureset
    # sub_feature = copy.deepcopy(featureset)
    # sub_featureset.remove(best_feature )
    node_son1 = TreeGenerate(dataset1, featureset)
    node_son2 = TreeGenerate(dataset2, featureset)
    node.add_leftnode(node_son1)
    node.add_rightnode(node_son2)
    return node

def BestFeature(dataset, featureset):
    """
    find the best feature.
    :param dataset: a 2D list,each row of which is like [5.1,3.5,1.4,0.2,0],0 represents 'Iris-setosa'
    :return:best feature and its threshold
    """
    largest_gain = 0
    best_feature = None
    best_threshold = 0
    for feature in featureset:
        gain_fea, threshold = ComputeGain(dataset, feature)
        if gain_fea > largest_gain:
            largest_gain = copy.deepcopy(gain_fea)
            best_feature = copy.deepcopy(feature)
            best_threshold = copy.deepcopy(threshold)
    return best_feature, best_threshold




def ComputeGain(dataset, feature):
    """
    Compute the gain of some feature.
    :param dataset: a 2D list,each row of which is like [5.1,3.5,1.4,0.2,'Iris-setosa']
    :param feature: an feature
    :return: the best gain and its corresponding threshold
    """
    fea_value = [row[feature] for row in dataset]#the first colunm
    fea_value = list(set(fea_value))
    fea_value.sort()
    thresholds = []
    for i in range(len(fea_value) - 1):
        thresholds.append((fea_value[i] + fea_value[i+1]) / 2)
    max_gain = 0
    max_threshold = 0
    for threshold in thresholds:
        dataset_sum = []
        dataset1 = []
        dataset2 = []
        for i in range(len(dataset)):
            dataset_sum.append([dataset[i][feature], dataset[i][4]])
            if dataset[i][feature] < threshold:
                dataset1.append([dataset[i][feature], dataset[i][4]])
            else:
                dataset2.append([dataset[i][feature], dataset[i][4]])
        gain = ComputeEntropy(dataset_sum) - (len(dataset1)/len(dataset))*ComputeEntropy(dataset1) - (len(dataset2)/len(dataset))*ComputeEntropy(dataset2)
        if gain > max_gain:
            max_gain = copy.deepcopy(gain)#？？
            max_threshold = copy.deepcopy(threshold)
    return max_gain, max_threshold

def ComputeEntropy(dataset):
    """
    Compute the entropy.
    :param dataset: a 2D list,each row of which is like [some feature value, classtype]
    :return: the entropy computed
    """
    p1_num = 0
    p2_num = 0
    p3_num = 0
    for i in range(len(dataset)):
        if dataset[i][1] == 0:
            p1_num += 1
        if dataset[i][1] == 1:
            p2_num += 1
        if dataset[i][1] == 2:
            p3_num += 1
    p1 = p1_num / (p1_num + p2_num + p3_num)
    p2 = p2_num / (p1_num + p2_num + p3_num)
    p3 = p3_num / (p1_num + p2_num + p3_num)
    ent = 0
    if p1 != 0:
        ent = ent - p1*np.log2(p1)
    if p2 != 0:
        ent = ent - p2 * np.log2(p2)
    if p3 != 0:
        ent = ent - p3 * np.log2(p3)
    return ent

def ComputeAccuracy(test_dataset, root_node):
    """
    Compute accuracy of classification of the Decision Tree .
    :param test_dataset: test dataset or validation dataset
    :param root_node: the root node of the Decision Tree
    :return: accuracy from [0,1]
    """
    TP = 0
    for data in test_dataset:
        classtype = root_node.classification(data)
        if classtype == data[4]:
            TP += 1
    acc = TP / len(test_dataset)
    return acc