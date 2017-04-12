'''
kNN: K近邻算法

Input:      inX: 要判断的数据
            dataSet: 训练数据集
            labels: 数据分类
            k: 输出结果的个数
            
Output:     输出欧氏距离最小的k个数
这个KNN算法的主程序

'''
from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    """
    训练模型，inx 
    """
    dataSetSize = dataSet.shape[0]  #长度
    diffMat = tile(inX, (dataSetSize,1)) - dataSet   #将预测减去测试样本
    sqDiffMat = diffMat**2   #平方差
    sqDistances = sqDiffMat.sum(axis=1) #多个平方之和
    distances = sqDistances**0.5        #开平方
    sortedDistIndicies = distances.argsort()   #数组值从小到大的索引值  
    classCount={}          
    for i in range(k):  
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1    #将相应分类的个数加1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  #对结果进行排序
    return sortedClassCount[0][0]  #输出分类结果

def createDataSet():
    """
    创建数据集
    """
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) #文件的数据
    labels = ['A','A','B','B']   #类别
    return group, labels

def file2matrix(filename):
    """
    将数据读取出来，将数据变成矩阵
    """  
    fr = open(filename)    #打开文件
    numberOfLines = len(fr.readlines())         #读取行数据，获取所有数据的行数
    returnMat = zeros((numberOfLines,3))        #创建矩阵，数据的行数，多少列
    classLabelVector = []                       #创建类别的矩阵
    fr = open(filename)
    index = 0
    for line in fr.readlines():   #遍历所有数据
        line = line.strip()   #清除数据的前后的空格
        listFromLine = line.split('\t')   #将数据按照tab键进行分割
        returnMat[index,:] = listFromLine[0:3]  #进行数据复制
        classLabelVector.append(int(listFromLine[-1]))   #添加类别
        index += 1
    return returnMat,classLabelVector    #返回类别、数据的矩阵
    
def autoNorm(dataSet):
    """
    进行归一化处理
    """ 
    minVals = dataSet.min(0)  #获取数据的最小值
    maxVals = dataSet.max(0) #获取数据的最大值
    ranges = maxVals - minVals   #整个数据集中差值范围
    normDataSet = zeros(shape(dataSet))   #读取数的长度（行长度，列长度），创建矩阵
    m = dataSet.shape[0]  #获取数据的训练样本
    normDataSet = dataSet - tile(minVals, (m,1))  #将minVals变成m行，1列的矩阵，使用当前值减去最小值
    normDataSet = normDataSet/tile(ranges, (m,1))   #将获取的值除以差值，获取一个范围[0-1]
    return normDataSet, ranges, minVals  #返回归一化之后的矩阵，差值范围，最小值
   
def datingClassTest():
    """
    约会类别测试
    """
    hoRatio = 0.50      #hold out 10%  确定测试的样本数
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #加载数据，数据矩阵、类别
    normMat, ranges, minVals = autoNorm(datingDataMat)  #进行归一化
    m = normMat.shape[0]    #数据行的长度
    numTestVecs = int(m*hoRatio)   #确定测试的样本数
    errorCount = 0.0     #错误率
    for i in range(numTestVecs):   #对文件进行遍历
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)   #分类结果
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs)) #判断错误率
    print errorCount

    
def img2vector(filename):

    """
    这个方法主要是把图片变成向量

    """
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    """
    手写数字图案判断
    """
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #获取目录中的内容，加载数据
    m = len(trainingFileList)  #求出长度
    trainingMat = zeros((m,1024))  #创建矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]  
        fileStr = fileNameStr.split('.')[0]     #将数据按照'.'进行拆分
        classNumStr = int(fileStr.split('_')[0])  #将数据按照'_'进行拆分
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)  #将图片变成矩阵，训练数据
    testFileList = listdir('testDigits')        #获取目录的文件内容
    errorCount = 0.0
    mTest = len(testFileList) #文件个数
    for i in range(mTest):
        fileNameStr = testFileList[i]   
        fileStr = fileNameStr.split('.')[0]     #将数据按照'.'进行拆分
        classNumStr = int(fileStr.split('_')[0])  #将数据按照'_'进行拆分
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)   #将测试数据变成矩阵，
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)   #进行分类，和约会使用的分类方法是一致的。
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
