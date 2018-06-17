import numpy as np
import struct
import os

from datetime import datetime

def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0] # shape[0] stands for the num of row
    #init_shape = newInput.shape[0]
    #newInput = newInput.reshape(1, init_shape)
    #np.tile(A,B)：重复A B次，相当于重复[A]*B
    #print np.tile(newInput, (numSamples, 1)).shape
    diff = np.tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise
    squaredDiff = diff ** 2 # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis = 1) # sum is performed by row
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)

    classCount = {} # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    ## step 5: the max voted class will return
    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex

def loadImageSet(filename):
	print("load image set",filename)
	binfile= open(filename, 'rb')
	buffers = binfile.read()
 
	head = struct.unpack_from('>IIII' , buffers ,0)
	print("head,",head)
 
	offset = struct.calcsize('>IIII')
	imgNum = head[1]
	width = head[2]
	height = head[3]
	#[60000]*28*28
	bits = imgNum * width * height
	bitsString = '>' + str(bits) + 'B' #like '>47040000B'
 
	imgs = struct.unpack_from(bitsString,buffers,offset)
 
	binfile.close()
	imgs = np.reshape(imgs,[imgNum,width*height])
	print ("load imgs finished")
	return imgs
 
def loadLabelSet(filename):
 
	print ("load label set",filename)
	binfile = open(filename, 'rb')
	buffers = binfile.read()
 
	head = struct.unpack_from('>II' , buffers ,0)
	print ("head,",head)
	imgNum=head[1]
 
	offset = struct.calcsize('>II')
	numString = '>'+str(imgNum)+"B"
	labels = struct.unpack_from(numString , buffers , offset)
	binfile.close()
	labels = np.reshape(labels,[imgNum])
 
	print ('load label finished')
	return labels



file_path=os.path.abspath(__file__)
father_path=os.path.dirname(file_path)
file_path=father_path
print(file_path)

#转化训练集
x_train = np.minimum(loadImageSet(file_path+"\\train-images.idx3-ubyte"),1)
y_train = np.minimum(loadLabelSet(file_path+"\\train-labels.idx1-ubyte"),1)

#转化测试集
x_test = np.minimum(loadImageSet(file_path+"\\t10k-images.idx3-ubyte"),1)
y_test=np.minimum(loadLabelSet(file_path+"\\t10k-labels.idx1-ubyte"),1)


a = datetime.now()
print("classification begin!")
numTestSamples = x_test.shape[0]
matchCount = 0
test_num = int(numTestSamples/10)
for i in range(test_num):
    predict = kNNClassify(x_test[i], x_train, y_train, 1)
    if predict == y_test[i]:
        matchCount += 1
    if (i % 100 == 0)and(i!=0):
        print ("完成%d张图片"%(i))
        b = datetime.now()
        print( "一共运行了%d秒"%((b-a).seconds))
b = datetime.now()
print ("完成%d张图片"%(test_num))
print( "一共运行了%d秒"%((b-a).seconds))
accuracy = float(matchCount) / test_num
print("准确率为%.1f%%"%(100*accuracy))
print("classification end!")
